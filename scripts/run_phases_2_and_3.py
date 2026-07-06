"""
Auto-chain Phase 2 (descriptors) → Phase 3a (XGBoost) → Phase 3b (PS ablation).
Run this after 01_download_mp_data.py completes.

Usage:
    conda activate csp
    python scripts/run_phases_2_and_3.py
"""
import subprocess, sys, os
from pathlib import Path

PYTHON = sys.executable
SCRIPTS = Path(__file__).parent

steps = [
    ("Phase 2  — compute descriptors",   SCRIPTS / "02_compute_descriptors.py"),
    ("Phase 3a — train XGBoost",          SCRIPTS / "03_train_xgboost.py"),
    ("Phase 3b — PS ablation",            SCRIPTS / "04_ps_ablation.py"),
]

for label, script in steps:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}\n")
    result = subprocess.run([PYTHON, str(script)], env=os.environ.copy())
    if result.returncode != 0:
        print(f"\n[FAILED] {label} — stopping pipeline.")
        sys.exit(result.returncode)
    print(f"\n[OK] {label}")

print("\n" + "="*60)
print("  All phases complete.")
print("  Review results/ps_ablation_report.md before proceeding.")
print("="*60)
