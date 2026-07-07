"""
Phase 5: Cross-database validation (AWA).

Applies the MP-trained SG classifier to AtomWork-Adv. (AWA) metadata to
test generalisability across databases. AWA descriptors
(coef_01..18, prop_01..18) are already stored in the metadata CSV.

**Not directly runnable from a fresh clone.**

    This script reads ``data/AWA/AWA_metadata_for_benchmark.csv``, which is
    derived from the AtomWork-Adv. database maintained by NIMS. The AWA
    database is not publicly redistributable and is therefore NOT included
    in this repository. The script is kept here for transparency: it is
    the exact code that produced the aggregated cross-database numbers
    reported in the Supplementary Information of the paper.

    Readers with NIMS AtomWork-Adv. access may prepare their own AWA
    metadata CSV in the same schema and re-run this script.
    Readers without access can trust that this file is what generated the
    SI cross-DB metrics, but cannot rerun it themselves.

Usage:
    conda activate csp
    python scripts/06_cross_db_validation.py

Input:   data/AWA/AWA_metadata_for_benchmark.csv   (NOT in repo)
         csp_workflow_mp/models/xgb_sg.pkl         (produced by 03_train_xgboost.py)
Output:  results/cross_db_results.csv
         results/cross_db_report.md
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])
AWA_CSV      = PROJECT_ROOT / "data" / "AWA" / "AWA_metadata_for_benchmark.csv"
MODEL_DIR    = PROJECT_ROOT / "csp_workflow_mp" / "csp_workflow_mp" / "models"
RESULTS_DIR  = PROJECT_ROOT / "csp_workflow_mp" / "results"

COEF_COLS = [f"coef_{i:02d}" for i in range(1, 19)]
PROP_COLS  = [f"prop_{i:02d}" for i in range(1, 19)]
DESC_COLS  = COEF_COLS + PROP_COLS

TOP_K_LIST = [1, 3, 5, 10]


def topk_accuracy(y_true: np.ndarray, proba: np.ndarray, k: int) -> float:
    top_k = np.argsort(proba, axis=1)[:, -k:]
    return float(np.mean([y_true[i] in top_k[i] for i in range(len(y_true))]))


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load AWA metadata ────────────────────────────────────────────────────
    df = pd.read_csv(AWA_CSV)
    logger.info("Loaded %d AWA rows", len(df))

    # Drop rows with missing descriptors
    before = len(df)
    df = df.dropna(subset=DESC_COLS).reset_index(drop=True)
    if len(df) < before:
        logger.warning("Dropped %d rows with missing descriptors", before - len(df))

    # ── Load MP-trained SG model ──────────────────────────────────────────────
    with open(MODEL_DIR / "xgb_sg.pkl", "rb") as f:
        sg_pkg = pickle.load(f)
    sg_model, sg_enc = sg_pkg["model"], sg_pkg["encoder"]
    logger.info("MP SG model: %d classes", len(sg_enc.classes_))

    # ── Filter to SGs seen during MP training ────────────────────────────────
    sg_col = "space_group_number" if "space_group_number" in df.columns else "space_group"
    y_sg_raw = df[sg_col].values

    known_mask = np.isin(y_sg_raw, sg_enc.classes_)
    n_unknown = (~known_mask).sum()
    if n_unknown:
        logger.warning(
            "%d AWA samples have SG not seen in MP training (%.1f%%) — excluded",
            n_unknown, 100 * n_unknown / len(df),
        )
    df_known = df[known_mask].reset_index(drop=True)
    logger.info("AWA samples with known SG: %d", len(df_known))

    X = df_known[DESC_COLS].to_numpy(dtype=float)
    y_sg = sg_enc.transform(df_known[sg_col].values)

    # ── Predict ───────────────────────────────────────────────────────────────
    logger.info("Predicting on %d AWA samples ...", len(X))
    sg_proba = sg_model.predict_proba(X)

    # ── Compute top-K accuracy ────────────────────────────────────────────────
    rows = []
    for k in TOP_K_LIST:
        acc = topk_accuracy(y_sg, sg_proba, k)
        rows.append({"top_k": k, "awa_accuracy": acc})
        logger.info("Top-%2d SG accuracy on AWA: %.3f", k, acc)

    result_df = pd.DataFrame(rows)
    result_df.to_csv(RESULTS_DIR / "cross_db_results.csv", index=False)

    # ── Load MP CV results for comparison ─────────────────────────────────────
    mp_cv_path = RESULTS_DIR / "cv_results.csv"
    mp_row = {}
    if mp_cv_path.exists():
        cv_df = pd.read_csv(mp_cv_path)
        sg_cv = cv_df[cv_df["task"] == "SG"].set_index("metric")["mean"]
        mp_row = {
            1:  sg_cv.get("top1",  float("nan")),
            3:  sg_cv.get("top3",  float("nan")),
            5:  sg_cv.get("top5",  float("nan")),
            10: sg_cv.get("top10", float("nan")),
        }

    # ── Markdown report ────────────────────────────────────────────────────────
    lines = [
        "# Cross-Database Validation Report (Phase 5)\n",
        f"Model: MP-trained XGBoost SG classifier ({len(sg_enc.classes_)} classes, n_estimators=200)\n",
        f"Test DB: AWA — {len(df_known):,} samples with known SG "
        f"({n_unknown:,} excluded — SG unseen in MP training)\n",
        "| Top-K | MP CV (in-distribution) | AWA (cross-DB) | Gap (pp) |",
        "|---|---|---|---|",
    ]
    for _, r in result_df.iterrows():
        k = int(r["top_k"])
        awa = r["awa_accuracy"]
        mp  = mp_row.get(k, float("nan"))
        gap = (awa - mp) * 100 if not (np.isnan(awa) or np.isnan(mp)) else float("nan")
        gap_str = f"{gap:+.2f}" if not np.isnan(gap) else "N/A"
        mp_str  = f"{mp:.3f}"   if not np.isnan(mp)  else "N/A"
        lines.append(f"| {k} | {mp_str} | {awa:.3f} | {gap_str} |")

    report = "\n".join(lines) + "\n"
    (RESULTS_DIR / "cross_db_report.md").write_text(report)
    logger.info("\n%s", report)
    logger.info("Phase 5 complete. See results/cross_db_report.md")


if __name__ == "__main__":
    main()
