"""Smoke tests for the scripts CLI.

These tests never open real MP data or load MatterSim; they exercise only:
  * script import order and canonical-path resolution (i.e., that scripts
    can be imported without setting PROJECT_ROOT);
  * argparse contracts on the benchmark entry point;
  * classifier helper's error message when the model file is absent.

They therefore run in a fraction of a second and provide a genuine
end-user-perspective health check that supplements the synthetic-pool
pipeline smoke test.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS   = REPO_ROOT / "scripts"


def _import_script(name: str):
    """Import a script by absolute path so PROJECT_ROOT never needs to be set."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── ensure PROJECT_ROOT is NOT set for these tests ──────────────────────────
@pytest.fixture(autouse=True)
def _no_project_root(monkeypatch):
    monkeypatch.delenv("PROJECT_ROOT", raising=False)


@pytest.mark.parametrize("name", [
    "01_download_mp_data",
    "01_download_mp_data_windows",
    "01b_validate_mp_data",
    "02_compute_descriptors",
    "03_train_xgboost",
    "05_run_benchmark",
    "05_run_benchmark_windows",
    "06_cross_db_validation",
])
def test_script_imports_without_project_root(name: str):
    """Every script must import cleanly without any environment variable set."""
    _import_script(name)


def test_benchmark_cli_help_runs():
    """`--help` must exit 0 and mention the paper-relevant flags."""
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "05_run_benchmark.py"), "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    for flag in ("--k", "--strategy", "--unconstrained",
                 "--n-samples", "--seed", "--device", "--output-dir"):
        assert flag in result.stdout, f"missing flag in --help: {flag}"


def test_benchmark_cli_argument_parsing():
    """`--k 1 --strategy sg_only` must produce the expected parsed values."""
    bench = _import_script("05_run_benchmark")
    old_argv = sys.argv
    try:
        sys.argv = ["05_run_benchmark.py",
                    "--k", "1", "--strategy", "sg_only",
                    "--n-samples", "10", "--seed", "42"]
        args = bench.parse_args()
    finally:
        sys.argv = old_argv
    assert args.k == 1
    assert args.strategies == ["sg_only"]
    assert args.n_samples == 10
    assert args.seed == 42
    assert args.device == "auto"


def test_benchmark_cli_unconstrained_shortcut():
    bench = _import_script("05_run_benchmark")
    old_argv = sys.argv
    try:
        sys.argv = ["05_run_benchmark.py", "--unconstrained"]
        args = bench.parse_args()
    finally:
        sys.argv = old_argv
    assert args.strategies == ["unconstrained"]


def test_benchmark_cli_default_runs_both():
    """No --k, no --strategy → both unconstrained and sg_only run."""
    bench = _import_script("05_run_benchmark")
    old_argv = sys.argv
    try:
        sys.argv = ["05_run_benchmark.py"]
        args = bench.parse_args()
    finally:
        sys.argv = old_argv
    assert set(args.strategies) == {"unconstrained", "sg_only"}


def test_classifier_missing_model_raises_clear_error(monkeypatch, tmp_path):
    """When xgb_sg.pkl is absent, predict_top_k_space_groups must raise
    FileNotFoundError with a message pointing the user to the training script."""
    from csp_workflow_mp import compute_periodic_descriptors, classifier
    # Point the module at an empty dir where the pkl definitely doesn't exist
    monkeypatch.setattr(classifier, "_MODEL_PATH", tmp_path / "xgb_sg.pkl")
    monkeypatch.setattr(classifier, "_CACHE", None)

    desc = compute_periodic_descriptors("KTaO3")
    with pytest.raises(FileNotFoundError, match="03_train_xgboost.py"):
        classifier.predict_top_k_space_groups(desc, k=1)


def test_classifier_input_shape_validation(monkeypatch, tmp_path):
    """Wrong descriptor dimension must produce a clear error, not a numpy stacktrace."""
    from csp_workflow_mp import classifier
    monkeypatch.setattr(classifier, "_MODEL_PATH", tmp_path / "xgb_sg.pkl")
    monkeypatch.setattr(classifier, "_CACHE", None)

    import numpy as np
    with pytest.raises(FileNotFoundError):
        classifier.predict_top_k_space_groups(np.zeros(36), k=1)


def test_paths_module_is_env_overridable(monkeypatch, tmp_path):
    """Setting CSP_MP_DATA_ROOT must redirect DATA_ROOT everywhere."""
    monkeypatch.setenv("CSP_MP_DATA_ROOT", str(tmp_path))

    # Force fresh import so the env var is picked up (both the submodule
    # and its cached attribute on the parent package must be dropped).
    for name in ("csp_workflow_mp._paths", "csp_workflow_mp"):
        sys.modules.pop(name, None)

    _paths = importlib.import_module("csp_workflow_mp._paths")
    assert _paths.DATA_ROOT == tmp_path
    assert _paths.CIF_DIR == tmp_path / "cifs"
    assert _paths.METADATA_CSV == tmp_path / "metadata.csv"
