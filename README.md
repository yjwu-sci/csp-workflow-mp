# csp-workflow-mp

Formula-to-Structure Generation via Space-Group-Guided Template Retrieval, trained on Materials Project data.

> **Paper:** Wu & Xu, *npj Computational Materials* (in preparation)
> **DOI:** *(Zenodo DOI to be added at submission)*

## Overview

This package turns a chemical formula into relaxation-ready candidate crystal structures by combining:

1. Periodic-descriptor-based space-group prediction (XGBoost)
2. Template retrieval from the Materials Project (~100K structures)
3. Stoichiometry-aware element substitution
4. Relaxation with the MatterSim ML potential

Two retrieval modes are benchmarked in the paper:
- **Space-group-guided** — templates restricted to the classifier's top-K predicted space groups
- **Unconstrained** — templates ranked purely by periodic-descriptor cosine similarity

The main result is that space-group-guided retrieval (K = 1, the classifier's single most confident space-group prediction) raises the space-group match rate on the valid relaxation subset from 31.6 % (unconstrained) to 57.5 % on Materials Project, at a cost of only ~3 pp in substitution success rate. The advantage reproduces on the experimental AtomWork-Adv. database (+28.9 pp).

## Installation

```bash
conda create -n csp python=3.10 -y
conda activate csp
pip install -e ".[relaxation]"
```

Requires a Materials Project API key:
```bash
export MP_API_KEY="your_key_here"
```

## Quick start

```python
from csp_workflow_mp import compute_periodic_descriptors, TemplatePool

# 1. Compute the periodic descriptor for a target formula
desc = compute_periodic_descriptors("NaCoO2")

# 2. Load the template pool (requires scripts/01_download_mp_data.py
#    and scripts/02_compute_descriptors.py to be run first)
pool = TemplatePool(
    "data/MP/metadata_with_descriptors.csv",
    cif_root="data/MP/cifs",
)

# 3. Retrieve top-20 templates within a chosen space group,
#    ranked by descriptor similarity (the K=1 case is space_group =
#    the classifier's top-1 prediction; see scripts/03_train_xgboost.py)
hits = pool.search(space_group=166, descriptor_vector=desc, top_n=20)
print(hits[["material_id", "formula", "pd_distance"]].head())
```

A complete walk-through is in `notebooks/01_workflow_demo.ipynb`; visualisation utilities are in `notebooks/02_visualise_predictions.ipynb`.

## Reproducing the paper benchmark

The leave-one-out (LOO) benchmark in the paper (500 MP targets, random_state = 42) can be reproduced by:

```bash
python scripts/01_download_mp_data.py        # download MP CIFs (~5 GB)
python scripts/02_compute_descriptors.py     # build descriptor matrix
python scripts/03_train_xgboost.py           # train SG classifier
python scripts/05_run_benchmark.py --k 1     # SG-guided K=1 (primary)
python scripts/05_run_benchmark.py --k 10    # SG-guided K=10
python scripts/05_run_benchmark.py --unconstrained
```

Benchmark CSVs reported in the paper, plus the trained model weights, are archived on Zenodo (DOI TBD). The cross-database verification against AtomWork-Adv. is in `scripts/06_cross_db_validation.py`.

## Data

Training data: Materials Project, `e_above_hull < 0.1 eV/atom`, `Z ≤ 95` (CC-BY 4.0).
CIF files are downloaded via `scripts/01_download_mp_data.py`.

Cross-database validation uses AtomWork-Adv. (AWA) metadata. The AWA database is not openly downloadable; AWA CIFs and the AWA-trained model are therefore not redistributed here. The aggregated AWA cross-DB metrics reported in the paper Supplementary Information are reproducible from MP data via `scripts/06_cross_db_validation.py`.

## Citation

```bibtex
@article{Wu2026SGGuided,
  title   = {Space-Group-Guided Template Retrieval for Composition-to-Structure Prediction},
  author  = {Wu, Yen-Ju and Xu, Yibin},
  journal = {npj Computational Materials},
  year    = {2026},
  note    = {in preparation}
}
```

## License

MIT
