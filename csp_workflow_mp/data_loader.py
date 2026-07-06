"""
Materials Project data loading utilities.

Handles reading and writing of MP metadata and CIF files for use with
the csp_workflow_mp pipeline. The download logic itself lives in
scripts/01_download_mp_data.py; this module provides helpers for
reading the already-downloaded data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Set

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Elements with Z > 95 that MatterSim does not support
MATTERSIM_UNSUPPORTED = frozenset([
    "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
    "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
    "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
])


def load_mp_metadata(metadata_path: str | Path) -> pd.DataFrame:
    """
    Load MP metadata CSV produced by scripts/01_download_mp_data.py.

    Expected columns: material_id, formula, space_group_number,
    pearson_symbol_prefix, e_above_hull, nelements.
    """
    df = pd.read_csv(metadata_path)

    required = {"material_id", "formula", "space_group_number", "pearson_symbol_prefix"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"MP metadata missing columns: {missing}")

    # Normalize to the column names TemplatePool expects
    df = df.rename(columns={
        "space_group_number": "space_group",
        "pearson_symbol_prefix": "pearson_prefix",
    })

    logger.info("Loaded %d MP entries from %s", len(df), metadata_path)
    return df


def attach_cif_paths(df: pd.DataFrame, cif_dir: str | Path) -> pd.DataFrame:
    """
    Add a ``cif_path`` column pointing to ``{cif_dir}/{material_id}.cif``.

    Only rows whose CIF file actually exists get a non-null path; missing
    files get NaN so TemplatePool can skip them gracefully.
    """
    cif_dir = Path(cif_dir)
    paths = []
    missing = 0
    for mid in df["material_id"]:
        p = cif_dir / f"{mid}.cif"
        if p.exists():
            paths.append(str(p))
        else:
            paths.append(None)
            missing += 1

    df = df.copy()
    df["cif_path"] = paths

    if missing:
        logger.warning("%d / %d CIF files not found under %s", missing, len(df), cif_dir)

    return df


def filter_mattersim_compatible(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove entries containing elements not supported by MatterSim (Z > 95).
    """
    from pymatgen.core import Composition

    def has_unsupported(formula: str) -> bool:
        try:
            elems = {str(e) for e in Composition(formula).elements}
            return bool(elems & MATTERSIM_UNSUPPORTED)
        except Exception:
            return True

    mask = ~df["formula"].apply(has_unsupported)
    n_removed = (~mask).sum()
    if n_removed:
        logger.info("Removed %d entries with MatterSim-unsupported elements", n_removed)
    return df[mask].reset_index(drop=True)


def load_descriptors(
    descriptors_path: str | Path,
    index_path: str | Path,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Load precomputed PD descriptor array and its material_id index.

    Parameters
    ----------
    descriptors_path : path to descriptors.npy, shape (N, 36)
    index_path : path to descriptor_index.csv with columns material_id, row_index

    Returns
    -------
    (descriptors, index_df)
    """
    descriptors = np.load(descriptors_path)
    index_df = pd.read_csv(index_path)
    logger.info("Loaded descriptors: shape %s", descriptors.shape)
    return descriptors, index_df


def merge_descriptors_into_metadata(
    df: pd.DataFrame,
    descriptors: np.ndarray,
    index_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge precomputed descriptor columns into the metadata DataFrame.

    The 36 dimensions are split into 18 coef_ and 18 prop_ columns matching
    the TemplatePool.DEFAULT_COEF_COLS / DEFAULT_PROP_COLS naming convention.
    """
    coef_cols = [f"coef_{i:02d}" for i in range(1, 19)]
    prop_cols = [f"prop_{i:02d}" for i in range(1, 19)]
    all_desc_cols = coef_cols + prop_cols

    desc_df = pd.DataFrame(
        descriptors[index_df["row_index"].values],
        columns=all_desc_cols,
    )
    desc_df["material_id"] = index_df["material_id"].values

    merged = df.merge(desc_df, on="material_id", how="left")
    n_missing = merged[all_desc_cols[0]].isna().sum()
    if n_missing:
        logger.warning("%d entries have no precomputed descriptor", n_missing)

    return merged
