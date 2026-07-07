"""
Compute 36-dim periodic descriptors for all MP materials.

Usage:
    conda activate csp
    python scripts/02_compute_descriptors.py

Input:   data/MP/metadata.csv
Output:  data/MP/descriptors.npy          shape (N, 36)
         data/MP/descriptor_index.csv      material_id, row_index
         data/MP/metadata_with_descriptors.csv   (merged, ready for TemplatePool)
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# --- canonical repository paths (see csp_workflow_mp/_paths.py) ---
import sys as _sys
_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in _sys.path:
    _sys.path.insert(0, str(_HERE.parent))
from csp_workflow_mp._paths import (
    REPO_ROOT as PROJECT_ROOT,
    DATA_ROOT,
    CIF_DIR,
    METADATA_CSV,
    METADATA_WITH_DESCRIPTORS_CSV,
    DESCRIPTORS_NPY,
    MODEL_DIR,
    RESULTS_DIR,
    LOG_DIR,
    ensure_data_dirs,
)
METADATA_CSV = PROJECT_ROOT / "data" / "MP" / "metadata.csv"
OUT_NPY      = PROJECT_ROOT / "data" / "MP" / "descriptors.npy"
OUT_INDEX    = PROJECT_ROOT / "data" / "MP" / "descriptor_index.csv"
OUT_MERGED   = PROJECT_ROOT / "data" / "MP" / "metadata_with_descriptors.csv"

N_WORKERS = max(1, mp.cpu_count() - 1)


def _compute_one(formula: str) -> np.ndarray | None:
    """Worker function: compute descriptor for a single formula."""
    from csp_workflow_mp.descriptor import compute_periodic_descriptors
    try:
        return compute_periodic_descriptors(formula)
    except Exception:
        return None


def main() -> None:
    if not METADATA_CSV.exists():
        raise FileNotFoundError(f"Run 01_download_mp_data.py first: {METADATA_CSV}")

    df = pd.read_csv(METADATA_CSV)
    logger.info("Loaded %d rows from metadata.csv", len(df))

    formulas = df["formula"].tolist()
    n = len(formulas)

    logger.info("Computing descriptors for %d formulas using %d workers ...", n, N_WORKERS)

    # macOS spawn-safe: must be under __main__ guard (handled by this script's structure)
    with mp.Pool(processes=N_WORKERS) as pool:
        results = pool.map(_compute_one, formulas, chunksize=500)

    # Build array; rows where computation failed get zero vector
    desc_list = []
    failed_idx = []
    for i, r in enumerate(results):
        if r is not None:
            desc_list.append(r)
        else:
            desc_list.append(np.zeros(36, dtype=float))
            failed_idx.append(i)

    descriptors = np.vstack(desc_list).astype(np.float32)
    logger.info("Descriptor matrix shape: %s", descriptors.shape)

    if failed_idx:
        logger.warning("%d formulas failed descriptor computation (zero vector used): %s ...",
                       len(failed_idx), df.iloc[failed_idx[:5]]["formula"].tolist())

    # Sanity checks
    n_nan = np.isnan(descriptors).sum()
    n_zero_rows = (descriptors.sum(axis=1) == 0).sum()
    logger.info("NaN values: %d | All-zero rows: %d", n_nan, n_zero_rows)

    # Save .npy
    np.save(OUT_NPY, descriptors)
    logger.info("Saved: %s", OUT_NPY)

    # Save index CSV
    index_df = pd.DataFrame({"material_id": df["material_id"], "row_index": np.arange(n)})
    index_df.to_csv(OUT_INDEX, index=False)
    logger.info("Saved: %s", OUT_INDEX)

    # Save merged metadata (for TemplatePool)
    coef_cols = [f"coef_{i:02d}" for i in range(1, 19)]
    prop_cols  = [f"prop_{i:02d}" for i in range(1, 19)]
    desc_df = pd.DataFrame(descriptors, columns=coef_cols + prop_cols)
    desc_df.insert(0, "material_id", df["material_id"].values)

    merged = df.merge(desc_df, on="material_id", how="left")
    merged.to_csv(OUT_MERGED, index=False)
    logger.info("Saved merged metadata: %s  (%d rows, %d cols)",
                OUT_MERGED, len(merged), len(merged.columns))

    # Quick cosine-distance sanity check on a few identical formulas
    from scipy.spatial.distance import cosine as cosine_dist
    dupes = df[df.duplicated("formula", keep=False)]
    if len(dupes) >= 2:
        sample = dupes.groupby("formula").first().head(3)
        for formula, row in sample.iterrows():
            idx_a = df[df["formula"] == formula].index[0]
            idx_b = df[df["formula"] == formula].index[1]
            d = cosine_dist(descriptors[idx_a], descriptors[idx_b])
            logger.info("Cosine dist [same formula %s]: %.2e (expect ~0)", formula, d)

    logger.info("Phase 2 complete.")


if __name__ == "__main__":
    main()
