"""
Phase 4: 500-sample LOO benchmark — Windows edition (fair to AWA pipeline).

Differences from original 05_run_benchmark.py:
  * K unified to 10 (matches AWA SG+PS K=10).
  * top-N retry: instead of single argmax, try up to N (default 50) candidates
    per query in score order until one yields a feasible substitution
    (mirrors AWA's csp_workflow.template_retriever.search_multi_family +
    check_substitution=True behaviour).
  * Records template_rank per row (which retry index succeeded; -1 = all failed).
  * MatterSim device defaults to "cuda" (auto-fallback to cpu) for Windows + CUDA.
  * SIGALRM replaced with thread-based timeout (Windows has no SIGALRM).
  * RESULTS_DIR overridable via --output-dir; default is a NEW directory
    `results_windows_k10_retry50/`. Original `results/` is never touched.
  * CIF directory overridable via --cif-dir (default D:/csp_mp_data/cifs).

Usage:
    conda activate csp
    python scripts/05_run_benchmark_windows.py \\
        --output-dir csp_workflow_mp/results_windows_k10_retry50 \\
        --device cuda --sg-filter-p 10 --sg-ps-top-k 10 --n-retry 50

Input:   data/MP/metadata_with_descriptors.csv
         <--cif-dir>/{mp-id}.cif
         csp_workflow_mp/models/xgb_sg.pkl
         csp_workflow_mp/models/xgb_ps.pkl
Output:  <--output-dir>/benchmark_raw.csv (with template_rank column)
         <--output-dir>/benchmark_results.csv
         <--output-dir>/benchmark_report.md
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from csp_workflow_mp.substitution_engine import SubstitutionEngine
from csp_workflow_mp.symmetry_filter import filter_compatible_pairs

import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))
from structural_eval_helpers import sym_order_sm_match, soap_cosine_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])
MERGED_CSV   = PROJECT_ROOT / "data" / "MP" / "metadata_with_descriptors.csv"
DEFAULT_CIF_DIR     = Path(r"D:\csp_mp_data\cifs")
MODEL_DIR    = PROJECT_ROOT / "csp_workflow_mp" / "csp_workflow_mp" / "models"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "csp_workflow_mp" / "results_windows_k10_retry50"

COEF_COLS = [f"coef_{i:02d}" for i in range(1, 19)]
PROP_COLS  = [f"prop_{i:02d}" for i in range(1, 19)]
DESC_COLS  = COEF_COLS + PROP_COLS

N_SAMPLES       = 500
RANDOM_STATE    = 42

# Relaxation settings — identical to AWA paper (Section 5.6)
RELAX_FMAX      = 0.05    # eV/Å
RELAX_STEPS     = 500
RELAX_TIMEOUT   = 300     # seconds per structure
VOL_CHANGE_MAX  = 0.15    # |ΔV/V| < 15%

STRATEGIES = ["unconstrained", "sg_only", "sg_ps"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR,
                   help=f"Results dir (default: {DEFAULT_RESULTS_DIR.name})")
    p.add_argument("--cif-dir", type=Path, default=DEFAULT_CIF_DIR,
                   help=f"MP CIF directory (default: {DEFAULT_CIF_DIR})")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps", "auto"],
                   help="MatterSim device (default: cuda; 'auto' picks CUDA when available)")
    p.add_argument("--sg-filter-p", type=int, default=10,
                   help="top-P SG mask for sg_only (default: 10, matches AWA K=10)")
    p.add_argument("--sg-ps-top-k", type=int, default=10,
                   help="top-K SG and PS for sg_ps compatibility (default: 10)")
    p.add_argument("--n-retry", type=int, default=50,
                   help="Number of candidate templates to try per query (default: 50)")
    p.add_argument("--max-samples", type=int, default=N_SAMPLES,
                   help=f"Test samples (default: {N_SAMPLES})")
    p.add_argument("--strategies", default=",".join(STRATEGIES),
                   help=("Comma-separated list of strategies to run "
                         f"(default: {','.join(STRATEGIES)}; "
                         "useful for K-ablation where only sg_only needs to vary)"))
    return p.parse_args()


# ── Timeout helper (thread-based; Windows has no SIGALRM) ────────────────────
class _Timeout(Exception):
    pass

def _run_with_timeout(fn, *args, timeout=None, **kwargs):
    """Run fn(...) with a wall-clock timeout. Raises _Timeout if it overruns."""
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn, *args, **kwargs)
        try:
            return fut.result(timeout=timeout)
        except FuturesTimeout:
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

    def _do_relax():
        ucf = UnitCellFilter(atoms)
        opt = BFGS(ucf, logfile=None)
        opt.run(fmax=RELAX_FMAX, steps=RELAX_STEPS)
        return opt.converged()

    t0 = time.time()
    try:
        converged = _run_with_timeout(_do_relax, timeout=RELAX_TIMEOUT)
    except _Timeout:
        return None, False, RELAX_TIMEOUT, float("nan")
    except Exception as exc:
        logger.debug("Relaxation error: %s", exc)
        return None, False, time.time() - t0, float("nan")

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


def build_sg_ps_mask(sg_proba_i, ps_proba_i, y_sg, y_ps_enc, sg_enc, ps_enc, sg_ps_top_k):
    """
    SG+PS compatibility filter (AWA paper method):
    Cartesian product of top-K SG × top-K PS → keep crystallographically
    compatible pairs → pool mask.
    """
    top_k_sg_enc = np.argpartition(sg_proba_i, -sg_ps_top_k)[-sg_ps_top_k:]
    top_k_ps_enc = np.argpartition(ps_proba_i, -sg_ps_top_k)[-sg_ps_top_k:]

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
    args         = parse_args()
    RESULTS_DIR  = args.output_dir
    CIF_DIR      = args.cif_dir
    sg_filter_p  = args.sg_filter_p
    sg_ps_top_k  = args.sg_ps_top_k
    n_retry      = args.n_retry
    n_samples    = args.max_samples
    strategies_to_run = [s.strip() for s in args.strategies.split(",") if s.strip()]
    unknown = set(strategies_to_run) - set(STRATEGIES)
    if unknown:
        raise SystemExit(f"unknown strategies: {unknown}; valid: {STRATEGIES}")

    # Resolve device (auto → cuda if available, else cpu)
    device = args.device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_CSV = RESULTS_DIR / "benchmark_raw.csv"

    logger.info("Output dir : %s", RESULTS_DIR)
    logger.info("CIF dir    : %s", CIF_DIR)
    logger.info("Device     : %s", device)
    logger.info("sg_filter_p=%d, sg_ps_top_k=%d, n_retry=%d, samples=%d",
                sg_filter_p, sg_ps_top_k, n_retry, n_samples)

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

    # ── Sample N test indices (use canonical 500 then head(n_samples)) ───────
    rng        = np.random.default_rng(RANDOM_STATE)
    full_500   = rng.choice(len(df), size=N_SAMPLES, replace=False)
    test_idx   = full_500[:n_samples]

    # ── Normalised descriptor matrix for cosine similarity ───────────────────
    norms  = np.linalg.norm(X, axis=1, keepdims=True)
    norms  = np.where(norms < 1e-12, 1.0, norms)
    X_norm = (X / norms).astype(np.float32)
    logger.info("Computing similarity matrix (%d × %d) ...", n_samples, len(df))
    S = X_norm[test_idx] @ X_norm.T
    for i, idx in enumerate(test_idx):
        S[i, idx] = -np.inf
    logger.info("Similarity matrix ready.")

    # ── Predict SG and PS probabilities ──────────────────────────────────────
    logger.info("Predicting SG/PS probabilities for test samples ...")
    sg_proba = sg_model.predict_proba(X[test_idx])
    ps_proba = ps_model.predict_proba(X[test_idx])

    # ── Load heavy objects once ───────────────────────────────────────────────
    logger.info("Loading MatterSim (device=%s) ...", device)
    import warnings; warnings.filterwarnings("ignore")
    from mattersim.forcefield import MatterSimCalculator
    calc    = MatterSimCalculator(device=device)
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
    for loop_i, i in enumerate(range(n_samples)):
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
            logger.warning("[%d/%d] GT CIF error %s: %s", loop_i+1, n_samples, mid, exc)
            continue

        for strategy in strategies_to_run:
            if (i, strategy) in done_keys:
                continue

            # ── Template retrieval (apply mask, then top-N candidates) ────────
            S_i = S[i].copy()

            if strategy == "sg_only":
                top_p_enc    = np.argpartition(sg_proba[i], -sg_filter_p)[-sg_filter_p:]
                sg_mask      = np.isin(y_sg, top_p_enc)
                S_i[~sg_mask] = -np.inf
                if not np.any(np.isfinite(S_i)):
                    S_i = S[i].copy()

            elif strategy == "sg_ps":
                ps_mask = build_sg_ps_mask(
                    sg_proba[i], ps_proba[i], y_sg, y_ps, sg_enc, ps_enc, sg_ps_top_k
                )
                if ps_mask is not None:
                    S_i[~ps_mask] = -np.inf
                # else fallback to unconstrained (S_i unchanged)

            # Top-N candidates in descending score order (mirrors AWA's
            # search_multi_family + check_substitution=True retry policy)
            finite_count = int(np.isfinite(S_i).sum())
            top_n        = min(n_retry, finite_count)
            if top_n == 0:
                # Nothing to try — record failure with -1 rank
                row = dict(
                    sample_idx=i, material_id=mid, formula=formula,
                    n_elements=n_el, true_sg=true_sg, strategy=strategy,
                    template_id="", template_rank=-1,
                    sub_success=False, sub_method="none",
                    relax_converged=False, vol_change=float("nan"),
                    vol_filtered=False,
                    sg_match=False, sm_match=False, rmsd_angstrom=float("nan"),
                    sym_sm_match=False, sym_sm_rmsd=float("nan"),
                    soap_cosine=float("nan"),
                    relax_sec=float("nan"),
                )
                raw_rows.append(row); continue

            # argpartition gives unordered top-N; sort them descending by S
            cand_idx_unsorted = np.argpartition(S_i, -top_n)[-top_n:]
            cand_idx          = cand_idx_unsorted[np.argsort(-S_i[cand_idx_unsorted])]

            row = dict(
                sample_idx=i, material_id=mid, formula=formula,
                n_elements=n_el, true_sg=true_sg, strategy=strategy,
                template_id="", template_rank=-1,
                sub_success=False, sub_method="none",
                relax_converged=False, vol_change=float("nan"),
                vol_filtered=False,
                sg_match=False, sm_match=False, rmsd_angstrom=float("nan"),
                sym_sm_match=False, sym_sm_rmsd=float("nan"),
                soap_cosine=float("nan"),
                relax_sec=float("nan"),
            )

            # ── Substitution: try candidates in score order, take 1st success ─
            sub_res     = None
            tmpl_struct = None
            for rank, c_idx in enumerate(cand_idx):
                if not np.isfinite(S_i[c_idx]):
                    continue
                c_mid = mat_ids[c_idx]
                c_cif = CIF_DIR / f"{c_mid}.cif"
                try:
                    cs       = Structure.from_file(str(c_cif))
                    sr       = engine.find_substitutions(formula, cs)
                    sr_first = next((r for r in sr if r.success), None)
                    if sr_first is not None:
                        sub_res            = sr_first
                        tmpl_struct        = cs
                        row["template_id"] = c_mid
                        row["template_rank"] = rank
                        row["sub_success"] = True
                        row["sub_method"]  = sub_res.method or "unknown"
                        break
                except Exception as exc:
                    logger.debug("[%d] rank=%d sub error (%s/%s/tmpl=%s): %s",
                                 i, rank, mid, strategy, c_mid, exc)
                    continue

            if sub_res is None:
                # all top_n candidates failed
                raw_rows.append(row); continue

            try:
                pred_struct = engine.apply_substitution(tmpl_struct, sub_res)
            except Exception as exc:
                logger.debug("[%d] apply_substitution error (%s/%s): %s",
                             i, mid, strategy, exc)
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

            # ── New robust metrics: sym-order SM + SOAP cosine ─────────────────
            # Applies to ALL relaxed predictions (ordered or not). For ordered
            # predictions, sym-order SM degenerates to default SM.
            try:
                sym_sm, sym_rmsd = sym_order_sm_match(compare_struct, gt_struct)
                row["sym_sm_match"] = bool(sym_sm)
                row["sym_sm_rmsd"]  = sym_rmsd if sym_rmsd is not None else float("nan")
            except Exception:
                row["sym_sm_match"] = False
                row["sym_sm_rmsd"]  = float("nan")
            try:
                soap = soap_cosine_similarity(compare_struct, gt_struct)
                row["soap_cosine"] = soap if soap is not None else float("nan")
            except Exception:
                row["soap_cosine"] = float("nan")

            raw_rows.append(row)

        # ── Checkpoint ────────────────────────────────────────────────────────
        if raw_rows:
            chunk        = pd.DataFrame(raw_rows)
            write_header = not RAW_CSV.exists()
            chunk.to_csv(RAW_CSV, mode="a", header=write_header, index=False)
            raw_rows = []

        if (loop_i + 1) % 10 == 0 or loop_i == n_samples - 1:
            logger.info("[%d/%d] done", loop_i + 1, n_samples)

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
    (RESULTS_DIR / "benchmark_report.md").write_text(report, encoding="utf-8")
    logger.info("\n%s", report)
    logger.info("Phase 4 complete. See results/benchmark_report.md")


if __name__ == "__main__":
    main()
