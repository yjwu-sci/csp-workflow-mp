"""
Download Materials Project metadata and CIF files.

Strategy: ONE bulk query fetching metadata + structure in a single response.
The MP API's parallel pagination makes this ~15-30 min for ~100K materials,
vs. ~30 hours for individual structure fetches.

Usage:
    conda activate csp
    python scripts/01_download_mp_data.py

Output:
    data/MP/metadata.csv          — one row per material
    data/MP/cifs/{mp-id}.cif      — stored on local SSD via symlink

Resume: already-downloaded mp-ids are skipped automatically.
"""

from __future__ import annotations

import csv
import logging
import os
import time
from pathlib import Path

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

ensure_data_dirs()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "download.log"),
    ],
)
logger = logging.getLogger(__name__)

METADATA_COLS = [
    "material_id", "formula", "space_group_number",
    "pearson_symbol_prefix", "e_above_hull", "nelements",
]

E_HULL_MAX   = 0.1
MIN_ELEMENTS = 2
EXCLUDED_ELEMENTS = {
    "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
    "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
    "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
}


def get_downloaded_ids() -> set[str]:
    return {p.stem for p in CIF_DIR.glob("*.cif")}


def has_excluded_element(formula: str) -> bool:
    from pymatgen.core import Composition
    try:
        return bool({str(e) for e in Composition(formula).elements} & EXCLUDED_ELEMENTS)
    except Exception:
        return True


def compute_pearson_prefix(sg_number: int) -> str:
    from csp_workflow_mp.symmetry_filter import sg_to_pearson_prefix
    return sg_to_pearson_prefix(sg_number)


def download_all() -> None:
    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        raise RuntimeError("MP_API_KEY not set.")

    from mp_api.client import MPRester

    CIF_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_CSV.parent.mkdir(parents=True, exist_ok=True)

    already_done = get_downloaded_ids()
    logger.info("Already downloaded: %d CIFs", len(already_done))

    # ── Rebuild metadata set from existing CSV to avoid duplicate rows ────────
    existing_in_csv: set[str] = set()
    if METADATA_CSV.exists() and METADATA_CSV.stat().st_size > 0:
        with open(METADATA_CSV) as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_in_csv.add(row["material_id"])
    logger.info("Already in metadata CSV: %d rows", len(existing_in_csv))

    write_header = not METADATA_CSV.exists() or METADATA_CSV.stat().st_size == 0

    n_written = n_skipped = n_excluded = n_cif_fail = 0
    t_start = time.time()

    with MPRester(api_key) as mpr, open(METADATA_CSV, "a", newline="") as csv_fh:
        writer = csv.DictWriter(csv_fh, fieldnames=METADATA_COLS)
        if write_header:
            writer.writeheader()

        # ── Single bulk query: metadata + structure together ──────────────────
        # The MP API's parallel chunked download makes this the fastest strategy
        # (~15-30 min for 100K materials). 'structure' field is always included
        # when material_ids is not specified (API behaviour in mp-api 0.44).
        logger.info("Starting bulk query (e_above_hull <= %.2f) — this fetches "
                    "metadata + structure in one pass ...", E_HULL_MAX)

        docs = mpr.materials.summary.search(
            energy_above_hull=(0, E_HULL_MAX),
            fields=[
                "material_id",
                "formula_pretty",
                "symmetry",
                "energy_above_hull",
                "nelements",
                "structure",
            ],
        )

        n_total = len(docs)
        elapsed_query = time.time() - t_start
        logger.info("Bulk query complete: %d docs in %.0fs", n_total, elapsed_query)

        # ── Write CIFs + metadata ─────────────────────────────────────────────
        for doc in docs:
            mid = doc.material_id

            # Post-filter
            if doc.nelements < MIN_ELEMENTS:
                continue
            if has_excluded_element(doc.formula_pretty):
                n_excluded += 1
                continue

            # Skip if CIF already on disk
            if mid in already_done:
                n_skipped += 1
                # Still write metadata row if missing from CSV
                if mid not in existing_in_csv:
                    _write_row(writer, csv_fh, doc, mid, existing_in_csv)
                continue

            # Write CIF from the already-fetched structure object
            structure = doc.structure
            if structure is None:
                n_cif_fail += 1
                logger.warning("No structure for %s — skipping", mid)
                continue

            try:
                cif_path = CIF_DIR / f"{mid}.cif"
                structure.to(filename=str(cif_path), fmt="cif")
            except Exception as e:
                n_cif_fail += 1
                logger.warning("CIF write failed for %s: %s", mid, e)
                continue

            already_done.add(mid)
            n_written += 1
            _write_row(writer, csv_fh, doc, mid, existing_in_csv)

            if n_written % 1000 == 0:
                elapsed = time.time() - t_start
                rate = n_written / elapsed
                eta_h = (n_total - n_written - n_skipped) / max(rate, 1) / 3600
                logger.info(
                    "Progress: %d written | %d skipped | %d excluded | "
                    "%.0f CIFs/s | ETA %.1fh",
                    n_written, n_skipped, n_excluded, rate, eta_h,
                )

    elapsed_total = time.time() - t_start
    logger.info(
        "Done in %.0fs. Written: %d | Skipped: %d | Excluded: %d | CIF-fail: %d",
        elapsed_total, n_written, n_skipped, n_excluded, n_cif_fail,
    )


def _write_row(writer, fh, doc, mid: str, existing_in_csv: set) -> None:
    """Append one metadata row and track it."""
    sg  = doc.symmetry.number
    writer.writerow({
        "material_id":           mid,
        "formula":               doc.formula_pretty,
        "space_group_number":    sg,
        "pearson_symbol_prefix": compute_pearson_prefix(sg),
        "e_above_hull":          doc.energy_above_hull,
        "nelements":             doc.nelements,
    })
    fh.flush()
    existing_in_csv.add(mid)


if __name__ == "__main__":
    download_all()
