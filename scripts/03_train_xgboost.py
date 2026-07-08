"""
Train the XGBoost space-group classifier used by the retrieval stage.

Usage:
    conda activate csp
    python scripts/03_train_xgboost.py

Input:   data/MP/metadata_with_descriptors.csv
Output:  csp_workflow_mp/models/xgb_sg.pkl
         results/cv_results.csv
         results/cv_summary.md
"""

from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# --- canonical repository paths (see csp_workflow_mp/_paths.py) ---
_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))
from csp_workflow_mp._paths import (          # noqa: E402
    METADATA_WITH_DESCRIPTORS_CSV,
    MODEL_DIR,
    RESULTS_DIR,
    ensure_data_dirs,
)

MERGED_CSV = METADATA_WITH_DESCRIPTORS_CSV

COEF_COLS = [f"coef_{i:02d}" for i in range(1, 19)]
PROP_COLS = [f"prop_{i:02d}" for i in range(1, 19)]
DESC_COLS = COEF_COLS + PROP_COLS

XGB_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    eval_metric="mlogloss",
    verbosity=0,
)

CV_FOLDS = 5


def topk_scorer(k: int):
    """Return a sklearn scorer that computes top-K accuracy."""
    def _topk(estimator, X, y):
        proba = estimator.predict_proba(X)
        top_k_preds = np.argsort(proba, axis=1)[:, -k:]
        return np.mean([y[i] in top_k_preds[i] for i in range(len(y))])
    return _topk


def train_and_eval(X: np.ndarray, y_raw, label: str) -> tuple:
    """Encode labels, run 5-fold CV, train final model on all data.
    Returns (model, encoder, cv_rows)."""
    enc = LabelEncoder()
    y = enc.fit_transform(y_raw)
    n_classes = len(enc.classes_)
    logger.info("%s: %d samples, %d classes", label, len(y), n_classes)

    model = XGBClassifier(num_class=n_classes, **XGB_PARAMS)
    skf   = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

    # Rare-class filter so every fold has ≥ 1 sample per class
    counts  = np.bincount(y)
    cv_mask = counts[y] >= CV_FOLDS
    X_cv, y_cv = X[cv_mask], y[cv_mask]
    n_cv_classes = len(np.unique(y_cv))
    dropped = n_classes - n_cv_classes
    if dropped:
        logger.warning(
            "CV: dropping %d rare classes (< %d samples); using %d/%d samples",
            dropped, CV_FOLDS, cv_mask.sum(), len(y),
        )
    cv_enc  = LabelEncoder().fit(y_cv)
    y_cv_re = cv_enc.transform(y_cv)
    cv_model = XGBClassifier(num_class=n_cv_classes, **XGB_PARAMS)

    scoring = {
        "top1":  topk_scorer(1),
        "top3":  topk_scorer(3),
        "top5":  topk_scorer(5),
        "top10": topk_scorer(10),
    }

    logger.info("Running %d-fold CV for %s ...", CV_FOLDS, label)
    cv = cross_validate(cv_model, X_cv, y_cv_re, cv=skf, scoring=scoring,
                        n_jobs=1, verbose=0)

    rows = []
    for metric, scores in cv.items():
        if metric.startswith("test_"):
            name = metric[5:]
            rows.append({
                "task":   label,
                "metric": name,
                "mean":   scores.mean(),
                "std":    scores.std(),
            })
            logger.info("  %s: %.3f ± %.3f", name, scores.mean(), scores.std())

    logger.info("Training final %s model on full data ...", label)
    model.fit(X, y)
    return model, enc, rows


def main() -> None:
    ensure_data_dirs()

    df = pd.read_csv(MERGED_CSV)
    logger.info("Loaded %d rows", len(df))

    before = len(df)
    df = df.dropna(subset=DESC_COLS)
    if len(df) < before:
        logger.warning("Dropped %d rows with missing descriptors", before - len(df))

    X = df[DESC_COLS].to_numpy(dtype=float)

    # ── Space-group classifier ───────────────────────────────────────────────
    sg_col = "space_group" if "space_group" in df.columns else "space_group_number"
    sg_model, sg_enc, sg_rows = train_and_eval(X, df[sg_col].values, "SG")

    with open(MODEL_DIR / "xgb_sg.pkl", "wb") as f:
        pickle.dump({"model": sg_model, "encoder": sg_enc}, f)
    logger.info("Saved: %s", MODEL_DIR / "xgb_sg.pkl")

    # ── Save CV results ──────────────────────────────────────────────────────
    cv_df = pd.DataFrame(sg_rows)
    cv_df.to_csv(RESULTS_DIR / "cv_results.csv", index=False)

    lines = [
        "# XGBoost CV Results (5-fold stratified, MP training data)\n",
        "| Task | Top-1 | Top-3 | Top-5 | Top-10 |",
        "|---|---|---|---|---|",
    ]
    row = cv_df[cv_df["task"] == "SG"].set_index("metric")["mean"]
    lines.append(
        f"| SG | {row.get('top1', float('nan')):.3f} "
        f"| {row.get('top3', float('nan')):.3f} "
        f"| {row.get('top5', float('nan')):.3f} "
        f"| {row.get('top10', float('nan')):.3f} |"
    )
    (RESULTS_DIR / "cv_summary.md").write_text("\n".join(lines) + "\n")
    logger.info("Wrote %s", RESULTS_DIR / "cv_summary.md")


if __name__ == "__main__":
    main()
