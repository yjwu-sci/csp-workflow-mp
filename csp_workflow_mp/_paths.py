"""
Canonical paths for the csp-workflow-mp repository.

Every script and every package module resolves data, model, and result
locations through this module, so the repository is portable across
machines and operating systems and no environment variables are strictly
required for the default layout.

Environment overrides
---------------------
Users who keep their MP data on a separate drive (common because the
CIF set is small but numerous) can override the default location:

    export CSP_MP_DATA_ROOT=/mnt/ssd/csp_mp_data      # bash / zsh
    $env:CSP_MP_DATA_ROOT="D:/csp_mp_data"            # PowerShell

When set, ``DATA_ROOT`` and everything derived from it (``CIF_DIR``,
``METADATA_CSV`` ...) point at the user-chosen location. All other
optional environment variables follow the same pattern.
"""
from __future__ import annotations

import os
from pathlib import Path


# ---- absolute anchors --------------------------------------------------------

# csp_workflow_mp/csp_workflow_mp/_paths.py --> csp_workflow_mp/csp_workflow_mp/
PACKAGE_ROOT: Path = Path(__file__).resolve().parent

# csp_workflow_mp/  (the repository root after `git clone`)
REPO_ROOT: Path = PACKAGE_ROOT.parent


# ---- overridable defaults ----------------------------------------------------

def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser().resolve() if value else default


# Where MP metadata and CIFs live.
DATA_ROOT: Path                  = _env_path("CSP_MP_DATA_ROOT", REPO_ROOT / "data" / "MP")
CIF_DIR: Path                    = _env_path("CSP_MP_CIF_DIR",   DATA_ROOT / "cifs")
METADATA_CSV: Path               = DATA_ROOT / "metadata.csv"
METADATA_WITH_DESCRIPTORS_CSV: Path = DATA_ROOT / "metadata_with_descriptors.csv"
DESCRIPTORS_NPY: Path            = DATA_ROOT / "descriptors.npy"

# Trained model weights (produced by scripts/03_train_xgboost.py, loaded by
# csp_workflow_mp.classifier). The MODEL_DIR is bound to the package itself
# so training and inference never disagree.
MODEL_DIR: Path                  = PACKAGE_ROOT / "models"
XGB_SG_PKL: Path                 = MODEL_DIR / "xgb_sg.pkl"

# Benchmark and other pipeline outputs.
RESULTS_DIR: Path                = _env_path("CSP_RESULTS_DIR",  REPO_ROOT / "results")

# Log files (script side channel; distinct from benchmark outputs).
LOG_DIR: Path                    = _env_path("CSP_LOG_DIR",      REPO_ROOT / "logs")


def ensure_data_dirs() -> None:
    """Create data/model/log/results directories if they don't already exist."""
    for p in (DATA_ROOT, CIF_DIR, MODEL_DIR, RESULTS_DIR, LOG_DIR):
        p.mkdir(parents=True, exist_ok=True)
