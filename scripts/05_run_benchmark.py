"""
Phase 4: 500-sample LOO benchmark (full pipeline).

For each of 500 randomly sampled MP materials (seed=42):
  1. Exclude target from template pool (leave-one-out).
  2. Retrieve top-1 template via three strategies:
       Strategy A — unconstrained: cosine-similarity nearest neighbour, no filter
       Strategy B — sg_only:       filter pool to top-3 predicted SGs
       Strategy C — sg_ps:         SG+PS compatibility filter (top-5 SG × top-5 PS,
                                   crystallographic compatibility check, same as AWA paper)
  3. Perform ion substitution with SubstitutionEngine.
  4. Relax with MatterSim: BFGS + UnitCellFilter, fmax=0.05 eV/Å, 500 steps
     (matches AWA paper methodology exactly).
  5. Apply |ΔV/V| < 15% volume-change filter.
  6. Compare relaxed structure to ground-truth via:
       - SpacegroupAnalyzer  → sg_match  (relaxed SG == true SG)
       - StructureMatcher    → sm_match + rmsd_angstrom

Checkpointing: results appended to CSV after each sample; run is resumable.

Usage:
    conda activate csp
    python scripts/05_run_benchmark.py

Input:   data/MP/metadata_with_descriptors.csv + cifs/
         csp_workflow_mp/models/xgb_sg.pkl
         csp_workflow_mp/models/xgb_ps.pkl
Output:  results/benchmark_raw.csv
         results/benchmark_results.csv
         results/benchmark_report.md
"""

from __future__ import annotations

import logging
import os
import pickle
import signal
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from csp_workflow_mp.substitution_engine import SubstitutionEngine
from csp_workflow_mp.symmetry_filter import filter_compatible_pairs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])
MERGED_CSV   = PROJECT_ROOT / "data" / "MP" / "metadata_with_descriptors.csv"
CIF_DIR      = PROJECT_ROOT / "data" / "MP" / "cifs"
MODEL_DIR    = PROJECT_ROOT / "csp_workflow_mp" / "csp_workflow_mp" / "models"
RESULTS_DIR  = PROJECT_ROOT / "csp_workflow_mp" / "results"

COEF_COLS = [f"coef_{i:02d}" for i in range(1, 19)]
PROP_COLS  = [f"prop_{i:02d}" for i in range(1, 19)]
DESC_COLS  = COEF_COLS + PROP_COLS

N_SAMPLES       = 500
RANDOM_STATE    = 42
SG_FILTER_P     = 3       # top-P SGs for sg_only strategy
SG_PS_TOP_K     = 5       # top-K SGs and PSs for sg_ps compatibility filter

# Relaxation settings — identical to AWA paper (Section 5.6)
RELAX_FMAX      = 0.05    # eV/Å
RELAX_STEPS     = 500
RELAX_TIMEOUT   = 300     # seconds per structure
VOL_CHANGE_MAX  = 0.15    # |ΔV/V| < 15%
DEVICE          = "mps"

STRATEGIES = ["unconstrained", "sg_only", "sg_ps"]
RAW_CSV    = RESULTS_DIR / "benchmark_raw.csv"


# ── Timeout helper ────────────────────────────────────────────────────────────
class _Timeout(Exception):
    pass

def _alarm_handler(signum, frame):
    raise _Timeout()


def relax_structure(structure, calc, adaptor):
    """
    Relax with MatterSim using BFGS + UnitCellFilter (cell + atomic relaxation).
    Matches AWA paper methodology (Section 5.6).
    Returns (relaxed_structure, converged, elapsed, vol_change_frac).
    Returns (None, False, elapsed, nan) for disordered structures or on timeout/error.
    """
    try:
        from ase.filters import UnitCellFilter
    except ImportError:
        from ase.constraints import UnitCellFilter
    from ase.optimize import BFGS

    if not structure.is_ordered:
        return None, False, 0.0, float("nan")

    atoms = adaptor.get_atoms(structure)
    vol0  = atoms.get_volume()
    atoms.calc = calc

    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(RELAX_TIMEOUT)
    t0 = time.time()
    try:
        ucf = UnitCellFilter(atoms)
        opt = BFGS(ucf, logfile=None)
        opt.run(fmax=RELAX_FMAX, steps=RELAX_STEPS)
        converged = opt.converged()
    except _Timeout:
        signal.alarm(0)
        return None, False, RELAX_TIMEOUT, float("nan")
    except Exception as exc:
        signal.alarm(0)
        logger.debug("Relaxation error: %s", exc)
        return None, False, time.time() - t0, float("nan")
    finally:
        signal.alarm(0)

    elapsed  = time.time() - t0
    vol1     = atoms.get_volume()
    dv       = abs(vol1 - vol0) / max(vol0, 1e-6)
    relaxed  = adaptor.get_structure(atoms)
    return relaxed, converged, elapsed, dv


def sm_compare(s1: Structure, s2: Structure, matcher: StructureMatcher):
    """Return (match, rmsd_angstrom). rmsd=nan on failure."""
    try:
        match = matcher.fit(s1, s2)
        if match:
            rms, _ = matcher.get_rms_dist(s1, s2)
            avg_a  = (s1.lattice.a + s2.lattice.a) / 2
            rmsd   = float(rms) * avg_a
        else:
            rmsd = float("nan")
        return match, rmsd
    except Exception:
        return False, float("nan")


def get_relaxed_sg(structure) -> int | None:
    """Return space group number of structure via SpacegroupAnalyzer."""
    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        sga = SpacegroupAnalyzer(structure, symprec=0.1, angle_tolerance=5)
        return sga.get_space_group_number()
    except Exception:
        return None


def build_sg_ps_mask(sg_proba_i, ps_proba_i, y_sg, y_ps_enc, sg_enc, ps_enc):
    """
    SG+PS compatibility filter (AWA paper method):
    Cartesian product of top-K SG × top-K PS → keep crystallographically
    compatible pairs → pool mask.
    """
    top_k_sg_enc = np.argpartition(sg_proba_i, -SG_PS_TOP_K)[-SG_PS_TOP_K:]
    top_k_ps_enc = np.argpartition(ps_proba_i, -SG_PS_TOP_K)[-SG_PS_TOP_K:]

    sg_preds = [(int(sg_enc.inverse_transform([e])[0]), float(sg_proba_i[e]))
                for e in top_k_sg_enc]
    ps_preds = [(str(ps_enc.inverse_transform([e])[0]).strip(), float(ps_proba_i[e]))
                for e in top_k_ps_enc]

    compatible = filter_compatible_pairs(sg_preds, ps_preds, top_n=25)
    if not compatible:
        return None   # fall back to unconstrained

    compat_sgs = {int(sg) for sg, ps, _ in compatible}
    compat_ps  = {str(ps).strip() for sg, ps, _ in compatible}

    # Pool members matching ANY compatible (SG, PS) pair
    sg_raw_arr = np.array([int(sg_enc.inverse_transform([e])[0]) for e in y_sg])
    ps_raw_arr = np.array([str(ps_enc.inverse_transform([e])[0]).strip() for e in y_ps_enc])

    mask = np.zeros(len(y_sg), dtype=bool)
    for sg, ps, _ in compatible:
        mask |= (sg_raw_arr == sg) & (ps_raw_arr == ps)

    return mask if mask.any() else None


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    df = pd.read_csv(MERGED_CSV).dropna(subset=DESC_COLS).reset_index(drop=True)
    logger.info("Loaded %d rows", len(df))

    sg_col = "space_group" if "space_group" in df.columns else "space_group_number"
    ps_col = "pearson_symbol_prefix" if "pearson_symbol_prefix" in df.columns else "pearson_prefix"

    X        = df[DESC_COLS].to_numpy(dtype=np.float32)
    y_sg_raw = df[sg_col].values
    y_ps_raw = df[ps_col].values.astype(str)
    mat_ids  = df["material_id"].values
    nelems   = df["nelements"].values if "nelements" in df.columns else np.ones(len(df), int)

    # ── Load models ───────────────────────────────────────────────────────────
    with open(MODEL_DIR / "xgb_sg.pkl", "rb") as f:
        sg_pkg = pickle.load(f)
    sg_model, sg_enc = sg_pkg["model"], sg_pkg["encoder"]

    with open(MODEL_DIR / "xgb_ps.pkl", "rb") as f:
        ps_pkg = pickle.load(f)
    ps_model, ps_enc = ps_pkg["model"], ps_pkg["encoder"]

    logger.info("SG classes: %d  |  PS classes: %d", len(sg_enc.classes_), len(ps_enc.classes_))

    # ── Filter to known classes ───────────────────────────────────────────────
    known_sg = np.isin(y_sg_raw, sg_enc.classes_)
    known_ps = np.isin(y_ps_raw, ps_enc.classes_)
    known    = known_sg & known_ps
    df       = df[known].reset_index(drop=True)
    X        = X[known]; y_sg_raw = y_sg_raw[known]; y_ps_raw = y_ps_raw[known]
    mat_ids  = mat_ids[known]; nelems = nelems[known]
    y_sg     = sg_enc.transform(y_sg_raw).astype(np.int32)
    y_ps     = ps_enc.transform(y_ps_raw).astype(np.int32)
    logger.info("After class filter: %d rows", len(df))

    # ── Sample 500 test indices ───────────────────────────────────────────────
    rng      = np.random.default_rng(RANDOM_STATE)
    test_idx = rng.choice(len(df), size=N_SAMPLES, replace=False)

    # ── Normalised descriptor matrix for cosine similarity ───────────────────
    norms  = np.linalg.norm(X, axis=1, keepdims=True)
    norms  = np.where(norms < 1e-12, 1.0, norms)
    X_norm = (X / norms).astype(np.float32)
    logger.info("Computing similarity matrix (%d × %d) ...", N_SAMPLES, len(df))
    S = X_norm[test_idx] @ X_norm.T
    for i, idx in enumerate(test_idx):
        S[i, idx] = -np.inf
    logger.info("Similarity matrix ready.")

    # ── Predict SG and PS probabilities ──────────────────────────────────────
    logger.info("Predicting SG/PS probabilities for test samples ...")
    sg_proba = sg_model.predict_proba(X[test_idx])
    ps_proba = ps_model.predict_proba(X[test_idx])

    # ── Load heavy objects once ───────────────────────────────────────────────
    logger.info("Loading MatterSim (device=%s) ...", DEVICE)
    import warnings; warnings.filterwarnings("ignore")
    from mattersim.forcefield import MatterSimCalculator
    calc    = MatterSimCalculator(device=DEVICE)
    adaptor = AseAtomsAdaptor()
    engine  = SubstitutionEngine()
    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5, attempt_supercell=True)
    logger.info("Ready.")

    # ── Checkpoint ────────────────────────────────────────────────────────────
    done_keys: set = set()
    if RAW_CSV.exists():
        prev = pd.read_csv(RAW_CSV)
        done_keys = set(zip(prev["sample_idx"].astype(int), prev["strategy"]))
        logger.info("Resuming: %d rows already done.", len(prev))

    raw_rows = []

    # ── Main benchmark loop ───────────────────────────────────────────────────
    for loop_i, i in enumerate(range(N_SAMPLES)):
        tidx    = test_idx[i]
        mid     = mat_ids[tidx]
        formula = df.at[tidx, "formula"]
        true_sg = int(y_sg_raw[tidx])
        n_el    = int(nelems[tidx])
        gt_cif  = CIF_DIR / f"{mid}.cif"

        try:
            gt_struct = Structure.from_file(str(gt_cif))
            vol_gt    = gt_struct.volume
        except Exception as exc:
            logger.warning("[%d/%d] GT CIF error %s: %s", loop_i+1, N_SAMPLES, mid, exc)
            continue

        for strategy in STRATEGIES:
            if (i, strategy) in done_keys:
                continue

            # ── Template retrieval ────────────────────────────────────────────
            S_i = S[i].copy()

            if strategy == "sg_only":
                top_p_enc    = np.argpartition(sg_proba[i], -SG_FILTER_P)[-SG_FILTER_P:]
                sg_mask      = np.isin(y_sg, top_p_enc)
                S_i[~sg_mask] = -np.inf
                if not np.any(np.isfinite(S_i)):
                    S_i = S[i].copy()

            elif strategy == "sg_ps":
                ps_mask = build_sg_ps_mask(
                    sg_proba[i], ps_proba[i], y_sg, y_ps, sg_enc, ps_enc
                )
                if ps_mask is not None:
                    S_i[~ps_mask] = -np.inf
                # else fallback to unconstrained (S_i unchanged)

            tmpl_idx = int(np.argmax(S_i))
            tmpl_mid = mat_ids[tmpl_idx]
            tmpl_cif = CIF_DIR / f"{tmpl_mid}.cif"

            row = dict(
                sample_idx=i, material_id=mid, formula=formula,
                n_elements=n_el, true_sg=true_sg, strategy=strategy,
                template_id=tmpl_mid,
                sub_success=False, sub_method="none",
                relax_converged=False, vol_change=float("nan"),
                vol_filtered=False,
                sg_match=False, sm_match=False, rmsd_angstrom=float("nan"),
                relax_sec=float("nan"),
            )

            # ── Substitution ──────────────────────────────────────────────────
            try:
                tmpl_struct = Structure.from_file(str(tmpl_cif))
                sub_results = engine.find_substitutions(formula, tmpl_struct)
                sub_res     = next((r for r in sub_results if r.success), None)
                if sub_res is None:
                    raw_rows.append(row); continue
                pred_struct        = engine.apply_substitution(tmpl_struct, sub_res)
                row["sub_success"] = True
                row["sub_method"]  = sub_res.method or "unknown"
            except Exception as exc:
                logger.debug("[%d] sub error (%s/%s): %s", i, mid, strategy, exc)
                raw_rows.append(row); continue

            # ── Relaxation (BFGS + UnitCellFilter, matches AWA paper) ─────────
            relaxed, converged, elapsed, dv = relax_structure(pred_struct, calc, adaptor)
            row["relax_converged"] = converged
            row["relax_sec"]       = round(elapsed, 1)
            row["vol_change"]      = round(float(dv), 4) if not np.isnan(dv) else float("nan")

            # ── Volume-change filter (|ΔV/V| < 15%, same as AWA paper) ────────
            if np.isnan(dv) or dv > VOL_CHANGE_MAX:
                row["vol_filtered"] = True
                raw_rows.append(row); continue

            if relaxed is None:
                raw_rows.append(row); continue

            # ── Structural comparison (all metrics on relaxed structure) ───────
            compare_struct = relaxed
            if compare_struct.is_ordered:
                pred_sg = get_relaxed_sg(compare_struct)
                if pred_sg is not None:
                    row["sg_match"] = (pred_sg == true_sg)
                sm_match, rmsd     = sm_compare(gt_struct, compare_struct, matcher)
                row["sm_match"]      = sm_match
                row["rmsd_angstrom"] = rmsd

            raw_rows.append(row)

        # ── Checkpoint ────────────────────────────────────────────────────────
        if raw_rows:
            chunk        = pd.DataFrame(raw_rows)
            write_header = not RAW_CSV.exists()
            chunk.to_csv(RAW_CSV, mode="a", header=write_header, index=False)
            raw_rows = []

        if (loop_i + 1) % 10 == 0 or loop_i == N_SAMPLES - 1:
            logger.info("[%d/%d] done", loop_i + 1, N_SAMPLES)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    raw = pd.read_csv(RAW_CSV)
    logger.info("Raw results: %d rows", len(raw))

    def agg(grp):
        return pd.Series({
            "n":               len(grp),
            "sub_success":     grp["sub_success"].mean(),
            "relax_conv":      grp["relax_converged"].mean(),
            "vol_filtered":    grp["vol_filtered"].mean(),
            "sg_match":        grp["sg_match"].mean(),
            "sm_match":        grp["sm_match"].mean(),
            "rmsd_median":     grp["rmsd_angstrom"].median(),
        })

    agg_overall = raw.groupby("strategy").apply(agg).reset_index()
    agg_nelem   = raw.groupby(["strategy", "n_elements"]).apply(agg).reset_index()
    agg_method  = raw.groupby(["strategy", "sub_method"]).apply(agg).reset_index()

    agg_overall.to_csv(RESULTS_DIR / "benchmark_results.csv",      index=False)
    agg_nelem.to_csv(  RESULTS_DIR / "benchmark_by_nelements.csv", index=False)
    agg_method.to_csv( RESULTS_DIR / "benchmark_by_method.csv",    index=False)

    # ── Report ────────────────────────────────────────────────────────────────
    lines = [
        "# LOO Benchmark Report (Phase 4 — Full Pipeline)\n",
        f"N={N_SAMPLES}, seed={RANDOM_STATE} | Relaxation: BFGS + UnitCellFilter, "
        f"fmax={RELAX_FMAX} eV/Å, {RELAX_STEPS} steps, |ΔV/V|<{int(VOL_CHANGE_MAX*100)}% filter\n",
        "## Overall results",
        "| Strategy | n | Sub success | Relax conv | Vol filtered | SG match | SM match | RMSD (Å) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in agg_overall.iterrows():
        lines.append(
            f"| {r['strategy']} | {int(r['n'])} "
            f"| {r['sub_success']:.3f} | {r['relax_conv']:.3f} "
            f"| {r['vol_filtered']:.3f} | {r['sg_match']:.3f} "
            f"| {r['sm_match']:.3f} | {r['rmsd_median']:.3f} |"
        )
    lines += ["", "## By n_elements (all strategies)",
              "| Strategy | n_elements | n | Sub success | SG match | SM match | RMSD (Å) |",
              "|---|---|---|---|---|---|---|"]
    for _, r in agg_nelem.iterrows():
        lines.append(
            f"| {r['strategy']} | {int(r['n_elements'])} | {int(r['n'])} "
            f"| {r['sub_success']:.3f} | {r['sg_match']:.3f} "
            f"| {r['sm_match']:.3f} | {r['rmsd_median']:.3f} |"
        )
    lines += ["", "## By substitution method (sg_ps strategy)",
              "| Method | n | Relax conv | SG match | SM match | RMSD (Å) |",
              "|---|---|---|---|---|---|"]
    for _, r in agg_method[agg_method["strategy"] == "sg_ps"].iterrows():
        lines.append(
            f"| {r['sub_method']} | {int(r['n'])} "
            f"| {r['relax_conv']:.3f} | {r['sg_match']:.3f} "
            f"| {r['sm_match']:.3f} | {r['rmsd_median']:.3f} |"
        )

    report = "\n".join(lines) + "\n"
    (RESULTS_DIR / "benchmark_report.md").write_text(report)
    logger.info("\n%s", report)
    logger.info("Phase 4 complete. See results/benchmark_report.md")


if __name__ == "__main__":
    main()
