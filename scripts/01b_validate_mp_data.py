"""
Validate downloaded MP data and produce a quality report.

Usage:
    conda activate csp
    python scripts/01b_validate_mp_data.py

Output:
    data/MP/data_quality_report.md
"""

from __future__ import annotations

import logging
import os
from collections import Counter
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
REPORT_PATH  = DATA_ROOT / "data_quality_report.md"

ALL_PS_PREFIXES = {
    "aP", "mP", "mS", "oP", "oS", "oI", "oF",
    "tP", "tI", "hP", "hR", "cP", "cI", "cF",
}


def validate_cifs(df: pd.DataFrame) -> dict:
    """Check each CIF is readable by pymatgen."""
    from pymatgen.core import Structure

    n_ok = 0
    n_fail = 0
    fail_ids = []

    for mid in df["material_id"]:
        cif_path = CIF_DIR / f"{mid}.cif"
        if not cif_path.exists():
            n_fail += 1
            fail_ids.append(mid)
            continue
        try:
            Structure.from_file(str(cif_path))
            n_ok += 1
        except Exception as e:
            n_fail += 1
            fail_ids.append(mid)
            logger.debug("CIF parse error %s: %s", mid, e)

    return {"n_ok": n_ok, "n_fail": n_fail, "fail_ids": fail_ids[:20]}


def check_sg_ps_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Compare our sg→ps mapping against the downloaded ps prefix."""
    from csp_workflow_mp.symmetry_filter import sg_to_pearson_prefix

    our_prefix = df["space_group_number"].apply(sg_to_pearson_prefix)
    mismatch = df[our_prefix != df["pearson_symbol_prefix"]].copy()
    mismatch["our_prefix"] = our_prefix[mismatch.index]
    return mismatch


def build_report(df: pd.DataFrame, cif_check: dict, sg_ps_mismatches: pd.DataFrame) -> str:
    lines = [
        "# MP Data Quality Report",
        f"\nGenerated from: `{METADATA_CSV}`\n",
        "## Summary",
        f"- Total entries: **{len(df):,}**",
        f"- CIF readable: **{cif_check['n_ok']:,}** / {len(df):,}",
        f"- CIF failures: **{cif_check['n_fail']}**",
        f"- SG↔PS mismatches (vs our mapping): **{len(sg_ps_mismatches)}**",
        "",
        "## Space Group Distribution (top 20)",
        "| SG | Count | % |",
        "|---|---|---|",
    ]
    sg_counts = df["space_group_number"].value_counts().head(20)
    for sg, cnt in sg_counts.items():
        lines.append(f"| {sg} | {cnt:,} | {100*cnt/len(df):.1f}% |")

    lines += [
        "",
        "## Pearson Symbol Prefix Distribution",
        "| PS prefix | Count | % |",
        "|---|---|---|",
    ]
    ps_counts = df["pearson_symbol_prefix"].value_counts()
    for ps in sorted(ALL_PS_PREFIXES):
        cnt = ps_counts.get(ps, 0)
        lines.append(f"| {ps} | {cnt:,} | {100*cnt/len(df):.1f}% |")

    missing_ps = ALL_PS_PREFIXES - set(ps_counts.index)
    if missing_ps:
        lines.append(f"\n⚠️ Missing PS prefixes: {sorted(missing_ps)}")

    lines += [
        "",
        "## Element Count Distribution",
        "| # elements | Count | % |",
        "|---|---|---|",
    ]
    for nelems, cnt in sorted(df["nelements"].value_counts().items()):
        lines.append(f"| {nelems} | {cnt:,} | {100*cnt/len(df):.1f}% |")

    lines += [
        "",
        "## e_above_hull Distribution",
        f"- min:  {df['e_above_hull'].min():.4f} eV/atom",
        f"- mean: {df['e_above_hull'].mean():.4f} eV/atom",
        f"- max:  {df['e_above_hull'].max():.4f} eV/atom",
        f"- ≤ 0.01 eV/atom (near-stable): {(df['e_above_hull'] <= 0.01).sum():,}",
        f"- ≤ 0.05 eV/atom:               {(df['e_above_hull'] <= 0.05).sum():,}",
    ]

    if cif_check["fail_ids"]:
        lines += [
            "",
            "## CIF Failures (first 20)",
            ", ".join(cif_check["fail_ids"]),
        ]

    if len(sg_ps_mismatches) > 0:
        lines += [
            "",
            f"## SG↔PS Mismatches (first 10 of {len(sg_ps_mismatches)})",
            "| material_id | formula | SG | MP prefix | our prefix |",
            "|---|---|---|---|---|",
        ]
        for _, row in sg_ps_mismatches.head(10).iterrows():
            lines.append(
                f"| {row['material_id']} | {row['formula']} | {row['space_group_number']} "
                f"| {row['pearson_symbol_prefix']} | {row['our_prefix']} |"
            )

    return "\n".join(lines) + "\n"


def main() -> None:
    if not METADATA_CSV.exists():
        raise FileNotFoundError(f"Metadata not found: {METADATA_CSV}\nRun 01_download_mp_data.py first.")

    df = pd.read_csv(METADATA_CSV)
    logger.info("Loaded %d rows from metadata.csv", len(df))

    logger.info("Validating CIFs (this may take a few minutes) ...")
    cif_check = validate_cifs(df)
    logger.info("CIF check: %d OK, %d failed", cif_check["n_ok"], cif_check["n_fail"])

    logger.info("Checking SG↔PS consistency ...")
    mismatches = check_sg_ps_consistency(df)
    logger.info("SG↔PS mismatches: %d", len(mismatches))

    report = build_report(df, cif_check, mismatches)
    REPORT_PATH.write_text(report, encoding="utf-8")
    logger.info("Report written to %s", REPORT_PATH)
    print(report)


if __name__ == "__main__":
    main()
