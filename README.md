# csp-workflow-mp

Formula-to-Structure Generation via Space-Group-Guided Template Retrieval, trained on Materials Project data.

> **Paper:** Wu Y.-J. & Xu Y. (manuscript in preparation).

## Overview

This package turns a chemical formula into relaxation-ready candidate crystal structures by combining:

1. A 36-dimensional periodic descriptor computed directly from the formula (18 IUPAC groups × [atom fraction, weighted-average atomic number]).
2. An XGBoost space-group classifier trained on Materials Project.
3. Template retrieval restricted to the predicted space groups, ranked by descriptor cosine similarity.
4. Stoichiometry-aware element substitution onto the retrieved templates.
5. Relaxation with the MatterSim machine-learning interatomic potential.

Two retrieval modes are benchmarked in the paper:
- **Space-group-guided** — templates restricted to the classifier's top-K predicted space groups
- **Unconstrained** — templates ranked purely by periodic-descriptor cosine similarity

On the leave-one-entry-out benchmark, space-group-guided retrieval raises the space-group match rate on the valid relaxation subset from **31.2 % (unconstrained) to 57.5 % (K = 1)** on Materials Project — an absolute improvement of 26.3 percentage points — at a cost of only ~3 pp in substitution success rate.

## Installation

```bash
conda create -n csp python=3.10 -y
conda activate csp
pip install -e ".[relaxation]"
```

The pipeline requires a Materials Project API key. It is only used at data-download time (`scripts/01_download_mp_data.py`); the benchmark and prediction paths never touch the network.

```bash
export MP_API_KEY="your_key_here"   # bash / zsh
$env:MP_API_KEY="your_key_here"     # PowerShell
```

> **Do not commit your API key.** The commands above set it as an environment
> variable in the current shell session only. If you need it to persist, put it
> in your shell profile (`~/.bashrc`, `~/.zshrc`, or a PowerShell profile),
> never in a source file that could end up in git. This repository ships a
> `.gitignore` and a `gitleaks` pre-commit hook that together block most
> accidental leaks; the safest habit is still to never paste a real key into
> any file inside the repository.

### Optional path overrides

The pipeline uses the repository's `data/MP/` sub-tree by default. If the CIF
set is inconvenient to keep inside the repository (for example, because it
lives on an external SSD), point the pipeline at another location by setting:

```bash
export CSP_MP_DATA_ROOT=/path/to/csp_mp_data     # bash / zsh
$env:CSP_MP_DATA_ROOT="D:/csp_mp_data"           # PowerShell
```

All scripts and the package inference API pick this up automatically. No
other environment variable is required.

## Quick start

Give the package a chemical formula and get back a ranked list of
symmetry-compatible structure templates:

```python
from csp_workflow_mp import (
    compute_periodic_descriptors,
    predict_top_k_space_groups,
    TemplatePool,
)

# 1. Encode the target formula as a 36-dim periodic descriptor.
formula = "KTaO3"
desc = compute_periodic_descriptors(formula)

# 2. Predict its most likely space group with the trained classifier.
top_sg  = predict_top_k_space_groups(desc, k=1)[0]
top_3   = predict_top_k_space_groups(desc, k=3)
print(f"top-1 SG = {top_sg}   top-3 SG = {top_3}")

# 3. Load the template pool (produced by scripts/01_download_mp_data.py
#    and scripts/02_compute_descriptors.py).
pool = TemplatePool(
    "data/MP/metadata_with_descriptors.csv",
    cif_root="data/MP/cifs",
)

# 4. Retrieve the top-20 templates whose space group matches the
#    classifier's top-1 prediction, ranked by descriptor cosine distance.
hits = pool.search(space_group=top_sg, descriptor_vector=desc, top_n=20)
print(hits[["material_id", "formula", "pd_distance"]].head())
```

The next stage is element substitution and MatterSim relaxation, shown in `notebooks/03_predict_new_composition.ipynb`.

> **On disordered targets and relaxation.** `MatterSim` expects a single
> chemical species at every atomic site, so a candidate structure that carries
> partial occupancies must first be ordered. The pipeline applies a
> dominant-species approximation (replace each partially occupied site by its
> most abundant element) before relaxation. This is discussed in the paper
> Methods (§4.5) and its implications are laid out in the SI.

## Examples

| Example | Runtime | Requires | What you get |
|---|---|---|---|
| **`notebooks/03_predict_new_composition.ipynb`** | 10–15 min | full install + trained model + downloaded MP data + MatterSim | Realistic single-composition prediction using KTaO₃: formula → SG prediction → template retrieval on MP → substitution → MatterSim relaxation → CIF output. |
| **[Reproducing the paper benchmark](#reproducing-the-paper-benchmark)** | ~2 h + benchmark | full install + MP API key + ~300 MB free disk for CIFs | Reproduces the 500-target LOEO benchmark under each retrieval strategy. Regenerates every number in the main-text tables from scratch. |
| **`notebooks/02_visualise_predictions.ipynb`** | ~30 s | `results/benchmark_raw.csv` from the reproduction step above | Diagnostic figures for benchmark output: per-stage success rates, RMSD distribution, per-complexity breakdown. |

## Reproducing the paper benchmark

The leave-one-entry-out (LOEO) benchmark in the paper (500 MP targets, seed 42) can be reproduced end-to-end from this repository. The complete pipeline is:

```bash
# 1. Download MP structural data (~300 MB of CIFs) — needs MP_API_KEY.
python scripts/01_download_mp_data.py

# 2. Compute the 36-dimensional periodic descriptor for every entry.
python scripts/02_compute_descriptors.py

# 3. Train the XGBoost SG classifier (~40 min on a modern multi-core CPU).
python scripts/03_train_xgboost.py

# 4. Run the benchmark under each retrieval strategy.
python scripts/05_run_benchmark.py --unconstrained    # unconstrained baseline
python scripts/05_run_benchmark.py --k 1              # SG-guided K = 1 (paper primary)
python scripts/05_run_benchmark.py --k 3              # SG-guided K = 3
python scripts/05_run_benchmark.py --k 10             # SG-guided K = 10
```

`05_run_benchmark.py --help` lists all flags (device selection, custom output directory, resume support, sample count, seed).

Trained model files (`csp_workflow_mp/models/xgb_sg.pkl`, `xgb_ps.pkl`) are not committed to this repository because of their file size. They are regenerated by step 3 above.

### Reading the aggregated report

Each benchmark run writes a `benchmark_report.md` alongside the raw CSV. The overall table exposes **both** the all-500-target denominator (end-to-end SG-correct yield) and the valid-subset denominator (matches the paper's headline numbers):

| Denominator | What it measures | Reported in the paper as |
|---|---|---|
| `sg_match_all` — n / 500 | End-to-end yield including substitution / relaxation / volume filtering failures | Discussion / SI end-to-end column |
| `sg_match_valid` — n / |valid subset| | SG match on the valid relaxation subset (converged + \|ΔV/V\| < 15 %) | Main-text Table 2 (31.2 % and 57.5 %) |

### A note on classifier holdout

The classifier is trained on the same MP pool that the 500 benchmark targets are sampled from (only the target's own MP-ID is excluded at retrieval time). The paper's Supplementary Information reports a full out-of-fold (OOF) sensitivity analysis showing that the K = 10 result is essentially insensitive to this leakage and the K = 1 result has a conservative-bound SG match rate of at least 46.3 % — so the main claim (SG-guided retrieval > unconstrained) is robust to the entry-level holdout. See SI §S5.

## Data

Training data: Materials Project, `e_above_hull < 0.1 eV/atom`, `Z ≤ 95` (CC-BY 4.0). CIF files are downloaded via `scripts/01_download_mp_data.py`.

## Citation

```bibtex
@unpublished{Wu2026SGGuided,
  title  = {Space-Group-Guided Template Retrieval for Composition-to-Structure Prediction},
  author = {Wu, Yen-Ju and Xu, Yibin},
  year   = {2026},
  note   = {manuscript in preparation}
}
```

## License

MIT
