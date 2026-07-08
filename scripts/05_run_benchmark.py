"""
LOEO benchmark on Materials Project.

For each target sampled from the MP pool the pipeline is:
  1. Exclude the target's own material_id from the template pool
     (leave-one-entry-out).
  2. Retrieve the top template by descriptor cosine similarity, subject
     to the strategy's SG mask (or no mask for the unconstrained case).
  3. Attempt a chemical-role substitution onto the retrieved template.
  4. Relax the substituted structure with MatterSim (BFGS + UnitCellFilter,
     fmax = 0.05 eV/Å, up to 500 steps; timeout 300 s).
  5. Apply the |ΔV/V| < 15% valid-subset filter.
  6. Compare the relaxed structure against ground truth:
       * SpacegroupAnalyzer → sg_match
       * StructureMatcher   → sm_match + rmsd_angstrom

The script writes a per-target raw CSV, three aggregated CSVs, and a
Markdown report.

Basic usage
-----------
    conda activate csp
    export MP_API_KEY="..."             # download / classifier training
    python scripts/05_run_benchmark.py                   # all strategies
    python scripts/05_run_benchmark.py --k 1             # SG-guided, K=1
    python scripts/05_run_benchmark.py --unconstrained   # unconstrained only

For every retrieval strategy the aggregated report exposes both the
all-500-target denominator and the valid-subset denominator, so the
paper numbers (31.2% and 57.5% on the valid subset) can be
recomputed from the raw CSV without any post-processing.

Inputs
------
    data/MP/metadata_with_descriptors.csv
    data/MP/cifs/{mp-id}.cif                   (or $CSP_MP_CIF_DIR)
    csp_workflow_mp/models/xgb_sg.pkl          (produced by 03_train_xgboost.py)

Outputs (under --output-dir, default: results/)
    benchmark_raw.csv
    benchmark_results.csv
    benchmark_by_nelements.csv
    benchmark_by_method.csv
    benchmark_report.md
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

# --- canonical repository paths (see csp_workflow_mp/_paths.py) ---
_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))
from csp_workflow_mp._paths import (      # noqa: E402
    METADATA_WITH_DESCRIPTORS_CSV,
    CIF_DIR as DEFAULT_CIF_DIR,
    MODEL_DIR as DEFAULT_MODEL_DIR,
    RESULTS_DIR as DEFAULT_RESULTS_DIR,
    ensure_data_dirs,
)
from csp_workflow_mp.substitution_engine import SubstitutionEngine   # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


COEF_COLS = [f"coef_{i:02d}" for i in range(1, 19)]
PROP_COLS = [f"prop_{i:02d}" for i in range(1, 19)]
DESC_COLS = COEF_COLS + PROP_COLS

# Relaxation settings — identical to the AWA paper (Section 5.6).
RELAX_FMAX     = 0.05        # eV/Å
RELAX_STEPS    = 500
RELAX_TIMEOUT  = 300         # seconds per structure
VOL_CHANGE_MAX = 0.15        # |ΔV/V| < 15%

ALL_STRATEGIES = ["unconstrained", "sg_only"]


# ─────────────────────── CLI ────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--k", type=int, default=None,
        help="Top-K SG mask for sg_only retrieval (paper: K=1 primary, K=3, K=10). "
             "When set this also implies --strategy sg_only.",
    )
    p.add_argument(
        "--strategy", default=None, choices=ALL_STRATEGIES,
        help="Run a single retrieval strategy. If omitted and --k is not set, "
             "runs 'unconstrained' + 'sg_only'. --unconstrained is a shortcut.",
    )
    p.add_argument(
        "--unconstrained", action="store_true",
        help="Shortcut for --strategy unconstrained.",
    )
    p.add_argument(
        "--n-samples", type=int, default=500,
        help="Number of LOEO test targets to sample from the MP pool.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for target sampling.",
    )
    p.add_argument(
        "--device", default="auto", choices=["auto", "cuda", "mps", "cpu"],
        help="MatterSim device. 'auto' picks cuda > mps > cpu based on availability.",
    )
    p.add_argument(
        "--data-csv", type=Path, default=METADATA_WITH_DESCRIPTORS_CSV,
        help="Metadata + descriptor CSV (default: data/MP/metadata_with_descriptors.csv).",
    )
    p.add_argument(
        "--cif-dir", type=Path, default=DEFAULT_CIF_DIR,
        help="Directory of MP CIF files (default: data/MP/cifs).",
    )
    p.add_argument(
        "--model-dir", type=Path, default=DEFAULT_MODEL_DIR,
        help="Directory containing xgb_sg.pkl (and xgb_ps.pkl if using sg_ps).",
    )
    p.add_argument(
        "--output-dir", type=Path, default=DEFAULT_RESULTS_DIR,
        help="Directory for raw CSV, aggregated CSVs, and report.",
    )
    p.add_argument(
        "--no-resume", action="store_true",
        help="Ignore any existing benchmark_raw.csv and start from scratch.",
    )
    args = p.parse_args()

    # ── strategy resolution ───────────────────────────────────────────────────
    if args.unconstrained:
        args.strategy = "unconstrained"

    if args.k is not None and args.strategy is None:
        args.strategy = "sg_only"

    args.strategies = (
        [args.strategy] if args.strategy is not None
        else ["unconstrained", "sg_only"]
    )
    if args.k is None:
        args.k = 3   # historical default for sg_only, kept for backwards compat

    return args


def resolve_device(choice: str) -> str:
    if choice != "auto":
        return choice
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


# ─────────────────────── cross-platform relax ───────────────────────────────

def relax_structure(structure, calc, adaptor, timeout: int = RELAX_TIMEOUT):
    """
    Relax with MatterSim (BFGS + UnitCellFilter). Uses a thread-based
    timeout so this works on Windows in addition to Linux/macOS.

    Returns (relaxed_structure, converged, elapsed, vol_change_frac).
    Returns (None, False, elapsed, nan) for disordered inputs, timeout,
    or any exception during relaxation.
    """
    try:
        from ase.filters import UnitCellFilter
    except ImportError:                                      # pragma: no cover
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
        with ThreadPoolExecutor(max_workers=1) as ex:
            converged = ex.submit(_do_relax).result(timeout=timeout)
    except FuturesTimeout:
        return None, False, timeout, float("nan")
    except Exception as exc:
        logger.debug("Relaxation error: %s", exc)
        return None, False, time.time() - t0, float("nan")

    elapsed = time.time() - t0
    vol1    = atoms.get_volume()
    dv      = abs(vol1 - vol0) / max(vol0, 1e-6)
    return adaptor.get_structure(atoms), converged, elapsed, dv


def sm_compare(s1, s2, matcher):
    try:
        match = matcher.fit(s1, s2)
        if match:
            rms, _ = matcher.get_rms_dist(s1, s2)
            avg_a  = (s1.lattice.a + s2.lattice.a) / 2
            return True, float(rms) * avg_a
        return False, float("nan")
    except Exception:
        return False, float("nan")


def get_relaxed_sg(structure) -> int | None:
    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        return SpacegroupAnalyzer(
            structure, symprec=0.1, angle_tolerance=5,
        ).get_space_group_number()
    except Exception:
        return None


# ─────────────────────── main ───────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    ensure_data_dirs()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = args.output_dir / "benchmark_raw.csv"
    device  = resolve_device(args.device)

    logger.info("Strategies: %s", args.strategies)
    logger.info("K = %d (sg_only mask width)", args.k)
    logger.info("n_samples = %d, seed = %d, device = %s", args.n_samples, args.seed, device)
    logger.info("Data:   %s", args.data_csv)
    logger.info("CIFs:   %s", args.cif_dir)
    logger.info("Model:  %s", args.model_dir)
    logger.info("Output: %s", args.output_dir)

    # ── Load data ────────────────────────────────────────────────────────────
    df = pd.read_csv(args.data_csv).dropna(subset=DESC_COLS).reset_index(drop=True)
    logger.info("Loaded %d rows", len(df))

    sg_col = "space_group" if "space_group" in df.columns else "space_group_number"

    X        = df[DESC_COLS].to_numpy(dtype=np.float32)
    y_sg_raw = df[sg_col].values
    mat_ids  = df["material_id"].values
    nelems   = df["nelements"].values if "nelements" in df.columns else np.ones(len(df), int)

    # ── Load classifier ─────────────────────────────────────────────────────
    with open(args.model_dir / "xgb_sg.pkl", "rb") as f:
        sg_pkg = pickle.load(f)
    sg_model, sg_enc = sg_pkg["model"], sg_pkg["encoder"]

    logger.info("SG classes: %d", len(sg_enc.classes_))

    # ── Filter to classes seen during training ──────────────────────────────
    known_mask = np.isin(y_sg_raw, sg_enc.classes_)
    df       = df[known_mask].reset_index(drop=True)
    X        = X[known_mask]
    y_sg_raw = y_sg_raw[known_mask]
    mat_ids  = mat_ids[known_mask]
    nelems   = nelems[known_mask]
    y_sg     = sg_enc.transform(y_sg_raw).astype(np.int32)

    # ── Sample targets ──────────────────────────────────────────────────────
    rng      = np.random.default_rng(args.seed)
    n_take   = min(args.n_samples, len(df))
    test_idx = rng.choice(len(df), size=n_take, replace=False)

    # ── Descriptor similarity matrix ────────────────────────────────────────
    norms  = np.linalg.norm(X, axis=1, keepdims=True)
    norms  = np.where(norms < 1e-12, 1.0, norms)
    X_norm = (X / norms).astype(np.float32)
    logger.info("Computing similarity matrix (%d × %d) ...", n_take, len(df))
    S = X_norm[test_idx] @ X_norm.T
    for i, idx in enumerate(test_idx):
        S[i, idx] = -np.inf

    # ── Classifier probabilities on targets ─────────────────────────────────
    sg_proba = sg_model.predict_proba(X[test_idx]) if "sg_only" in args.strategies else None

    # ── Load MatterSim, adaptor, substitution engine ────────────────────────
    logger.info("Loading MatterSim on device=%s ...", device)
    import warnings; warnings.filterwarnings("ignore")
    from mattersim.forcefield import MatterSimCalculator
    calc    = MatterSimCalculator(device=device)
    adaptor = AseAtomsAdaptor()
    engine  = SubstitutionEngine()
    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5, attempt_supercell=True)

    # ── Resume support ──────────────────────────────────────────────────────
    done_keys: set = set()
    if raw_csv.exists() and not args.no_resume:
        prev = pd.read_csv(raw_csv)
        done_keys = set(zip(prev["sample_idx"].astype(int), prev["strategy"]))
        logger.info("Resuming: %d rows already done.", len(prev))
    elif raw_csv.exists() and args.no_resume:
        raw_csv.unlink()
        logger.info("Started fresh (--no-resume).")

    raw_rows: list[dict] = []

    # ── Main loop ───────────────────────────────────────────────────────────
    for loop_i, i in enumerate(range(n_take)):
        tidx     = test_idx[i]
        mid      = mat_ids[tidx]
        formula  = df.at[tidx, "formula"]
        true_sg  = int(y_sg_raw[tidx])
        n_el     = int(nelems[tidx])
        gt_cif   = args.cif_dir / f"{mid}.cif"

        try:
            gt_struct = Structure.from_file(str(gt_cif))
        except Exception as exc:
            logger.warning("[%d/%d] GT CIF error %s: %s", loop_i + 1, n_take, mid, exc)
            continue

        for strategy in args.strategies:
            if (i, strategy) in done_keys:
                continue

            S_i = S[i].copy()
            if strategy == "sg_only":
                top_p_enc  = np.argpartition(sg_proba[i], -args.k)[-args.k:]
                sg_mask    = np.isin(y_sg, top_p_enc)
                S_i[~sg_mask] = -np.inf
                if not np.any(np.isfinite(S_i)):
                    S_i = S[i].copy()

            tmpl_idx = int(np.argmax(S_i))
            tmpl_mid = mat_ids[tmpl_idx]
            tmpl_cif = args.cif_dir / f"{tmpl_mid}.cif"

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

            relaxed, converged, elapsed, dv = relax_structure(pred_struct, calc, adaptor)
            row["relax_converged"] = converged
            row["relax_sec"]       = round(elapsed, 1)
            row["vol_change"]      = round(float(dv), 4) if not np.isnan(dv) else float("nan")

            if np.isnan(dv) or dv > VOL_CHANGE_MAX:
                row["vol_filtered"] = True
                raw_rows.append(row); continue

            if relaxed is None:
                raw_rows.append(row); continue

            if relaxed.is_ordered:
                pred_sg = get_relaxed_sg(relaxed)
                if pred_sg is not None:
                    row["sg_match"] = (pred_sg == true_sg)
                sm_match, rmsd     = sm_compare(gt_struct, relaxed, matcher)
                row["sm_match"]      = sm_match
                row["rmsd_angstrom"] = rmsd

            raw_rows.append(row)

        if raw_rows:
            chunk        = pd.DataFrame(raw_rows)
            write_header = not raw_csv.exists()
            chunk.to_csv(raw_csv, mode="a", header=write_header, index=False)
            raw_rows = []

        if (loop_i + 1) % 10 == 0 or loop_i == n_take - 1:
            logger.info("[%d/%d] done", loop_i + 1, n_take)

    # ── Aggregate + report ──────────────────────────────────────────────────
    if not raw_csv.exists():
        logger.warning("No rows written; nothing to aggregate.")
        return
    raw = pd.read_csv(raw_csv)
    logger.info("Raw results: %d rows", len(raw))

    raw["is_valid"] = (
        raw["sub_success"]
        & raw["relax_converged"]
        & (raw["vol_change"] < VOL_CHANGE_MAX)
    )

    def agg(grp: pd.DataFrame) -> pd.Series:
        """Aggregate one strategy (or one strategy × n_el / sub_method) group.

        All rates are computed on the valid subset (relaxation converged AND
        |ΔV/V| < 15%), matching the definition used throughout the paper:
        SG match / SM match / RMSD numbers appearing in the manuscript are
        exactly these fractions.
        """
        n_total = len(grp)
        n_sub   = int(grp["sub_success"].sum())
        n_valid = int(grp["is_valid"].sum())
        valid   = grp[grp["is_valid"]]
        return pd.Series({
            "n_total":          n_total,
            "n_sub_success":    n_sub,
            "n_valid_subset":   n_valid,
            "sub_success_rate": n_sub / max(n_total, 1),
            "sg_match":         float(valid["sg_match"].mean())         if n_valid else float("nan"),
            "sm_match":         float(valid["sm_match"].mean())         if n_valid else float("nan"),
            "rmsd_median":      float(valid["rmsd_angstrom"].median())  if n_valid else float("nan"),
        })

    agg_overall = raw.groupby("strategy").apply(agg).reset_index()
    agg_nelem   = raw.groupby(["strategy", "n_elements"]).apply(agg).reset_index()
    agg_method  = raw.groupby(["strategy", "sub_method"]).apply(agg).reset_index()

    agg_overall.to_csv(args.output_dir / "benchmark_results.csv",      index=False)
    agg_nelem.to_csv(  args.output_dir / "benchmark_by_nelements.csv", index=False)
    agg_method.to_csv( args.output_dir / "benchmark_by_method.csv",    index=False)

    lines = [
        "# LOEO Benchmark Report",
        "",
        f"n_samples = {args.n_samples}  |  seed = {args.seed}  |  device = {device}",
        f"strategies = {', '.join(args.strategies)}  |  K (sg_only mask) = {args.k}",
        f"Relaxation: BFGS + UnitCellFilter, fmax = {RELAX_FMAX} eV/Å, "
        f"{RELAX_STEPS} steps, |ΔV/V| < {int(VOL_CHANGE_MAX*100)}% valid filter",
        "",
        "SG match, SM match, and RMSD are computed on the valid subset "
        "(substitution succeeded, relaxation converged, and |ΔV/V| < 15%).",
        "",
        "## Overall results",
        "",
        "| Strategy | n_total | sub_success | n_valid | SG match | SM match | RMSD median (Å) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in agg_overall.iterrows():
        lines.append(
            f"| {r['strategy']} | {int(r['n_total'])} "
            f"| {int(r['n_sub_success'])} ({r['sub_success_rate']*100:.1f}%) "
            f"| {int(r['n_valid_subset'])} "
            f"| {r['sg_match']*100:.1f}% "
            f"| {r['sm_match']*100:.1f}% "
            f"| {r['rmsd_median']:.3f} |"
        )
    lines += ["", "## By number of constituent elements",
              "",
              "| Strategy | n_el | n_total | n_valid | SG match |",
              "|---|---:|---:|---:|---:|"]
    for _, r in agg_nelem.iterrows():
        lines.append(
            f"| {r['strategy']} | {int(r['n_elements'])} "
            f"| {int(r['n_total'])} | {int(r['n_valid_subset'])} "
            f"| {r['sg_match']*100:.1f}% |"
        )

    report = "\n".join(lines) + "\n"
    (args.output_dir / "benchmark_report.md").write_text(report, encoding="utf-8")
    logger.info("\n%s", report)
    logger.info("Wrote %s", args.output_dir / "benchmark_report.md")


if __name__ == "__main__":
    main()
