# csp-workflow-mp

Space-group-guided template retrieval for composition-to-structure prediction, benchmarked on Materials Project.

Companion code for **Wu Y.-J. & Xu Y.**, _Space-Group-Guided Template Retrieval for Composition-to-Structure Prediction_ (manuscript in preparation).

## What it does

Given a chemical formula, this workflow:
1. Encodes the formula as a 36-dim periodic descriptor (18 IUPAC groups × [atom fraction, weighted-average Z]).
2. Predicts the target space group with an XGBoost classifier trained on Materials Project.
3. Retrieves structural templates whose space group matches the classifier's top-K prediction, ranked by descriptor cosine similarity.
4. Substitutes target elements onto the top-ranked feasible template with a chemical-role-aware assignment.
5. Relaxes the substituted structure with MatterSim.

On the leave-one-entry-out benchmark (500 targets, MP), space-group-guided K=1 retrieval raises the SG match rate on the valid subset from **40.1% (unconstrained) to 63.0%** — a 22.9 pp absolute gain at a ~4 pp cost in substitution success.

## Install

```bash
conda create -n csp python=3.10 -y && conda activate csp
pip install -e ".[relaxation,dev]"
```

Any Python 3.10 env works (`venv` is fine too). The `[relaxation]` extra installs PyTorch and MatterSim (~1.5 GB, auto-picks the CUDA/MPS/CPU wheel). The `[dev]` extra adds pytest and Jupyter.

A Materials Project API key is required to download the pool (`scripts/01_download_mp_data.py`); the benchmark and prediction paths do not touch the network:

```bash
export MP_API_KEY="your_key"   # bash/zsh
$env:MP_API_KEY="your_key"     # PowerShell
```

## Quick start

Predict a structure from a formula in one call. Full runnable example in `notebooks/01_predict_composition_example.ipynb`:

```python
from csp_workflow_mp import predict_from_formula

# Case 1: standard — let the XGBoost classifier pick the space group
result = predict_from_formula('KTaO3', top_k_sg=1, do_relax=True)
print(result.summary())

# Case 2: user already knows the space group (XRD / Rietveld / literature)
result = predict_from_formula('KTaO3', known_sg=221, do_relax=True)

# Case 3: target with fractional stoichiometry (partial-occupancy handling)
result = predict_from_formula('BaFe0.5Mn0.5O3', known_sg=221, do_relax=True)
# → result.status == 'PARTIAL_OCCUPANCY'; substituted CIF is saved to disk,
#   MatterSim relaxation is skipped (cannot handle disordered structures).
# See notebooks/03_partial_occupancy_handling.ipynb for details.
```

The returned `PredictionResult` reports the retrieved template, saved CIF paths, warnings, and a `status` field:

| `status` | meaning |
|---|---|
| `SUCCESS` | substitution and relaxation both succeeded; \|ΔV/V\| < 15% |
| `SUBSTITUTED_ONLY` | substitution succeeded, relaxation skipped (`do_relax=False`) |
| `PARTIAL_OCCUPANCY` | substituted structure is disordered; CIF saved, relaxation skipped |
| `RELAX_FAILED` | relaxation did not converge or \|ΔV/V\| exceeded 15% |
| `SUBSTITUTION_FAILED` | no feasible mapping found in the top-N templates |
| `NO_CANDIDATE` | template pool had no entries matching the requested SG |

For a step-by-step walk-through of the pipeline (descriptor → classifier → template retrieval → substitution → relaxation) with per-step diagnostics, see `notebooks/01_predict_composition_example.ipynb`.

**Out of scope**: highly complex hypothetical compositions (many elements with fractional stoichiometry, e.g., glass-electrolyte-like) are template-poor by construction and typically return `SUBSTITUTION_FAILED`.

## Reproducing the paper benchmark

```bash
python scripts/01_download_mp_data.py            # ~300 MB CIFs (needs MP_API_KEY)
python scripts/02_compute_descriptors.py         # descriptors on 103k entries
python scripts/03_train_xgboost.py               # XGBoost SG classifier (~40 min)
python scripts/05_run_benchmark.py --k 1         # SG-guided K=1  (paper primary)
python scripts/05_run_benchmark.py --k 3         # SG-guided K=3
python scripts/05_run_benchmark.py --k 10        # SG-guided K=10
python scripts/05_run_benchmark.py --unconstrained
```

Each benchmark run writes `results/benchmark_raw.csv` (one row per target × strategy) and aggregated CSVs. Aggregate SG match / SM match / RMSD are computed on the valid subset (substitution succeeded ∧ relaxation converged ∧ |ΔV/V| < 15%), matching the paper's Table 2 definition. Use `05_run_benchmark.py --help` for CLI flags (`--n-retry` default 50, `--n-samples`, `--seed`, `--device`, `--output-dir`).

`notebooks/02_visualise_predictions.ipynb` reads `results/benchmark_raw.csv` and reproduces the diagnostic plots.

## Data & path overrides

Training data: Materials Project, `e_above_hull ≤ 0.1 eV/atom`, `Z ≤ 95` (CC-BY 4.0).

By default the pipeline reads `data/MP/`. If the CIFs live elsewhere, set `CSP_MP_DATA_ROOT=/path/to/csp_mp_data` before running.

## Handling of partial occupancy

Substitution candidates that carry partial site occupancies are excluded from the MatterSim relaxation step and counted as substitution failures on the valid subset. The workflow does not apply a dominant-species approximation; a real fix for partial-occupancy targets would require SQS-style enumeration and is discussed in the paper Discussion as future work.

## Citation

```bibtex
@unpublished{Wu2026SGGuided,
  title  = {Space-Group-Guided Template Retrieval for Composition-to-Structure Prediction},
  author = {Wu, Yen-Ju and Xu, Yibin},
  year   = {2026},
  note   = {manuscript in preparation}
}
```

External tools this workflow depends on (with licences and citation entries) are listed in [`NOTICES.md`](NOTICES.md).

## License

MIT
