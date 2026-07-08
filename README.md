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

Any Python 3.10 environment works — the two most common setups are shown below. Pick whichever you prefer.

**Option A — conda (recommended if you already use conda):**

```bash
conda create -n csp python=3.10 -y
conda activate csp
pip install -e ".[relaxation,dev]"     # keep the quotes — required by zsh
```

**Option B — the built-in `venv` module (no extra tooling required):**

```bash
python -m venv .venv
source .venv/bin/activate          # macOS / Linux (bash, zsh)
.venv\Scripts\Activate.ps1         # Windows PowerShell
pip install -e ".[relaxation,dev]"
```

The `.[relaxation]` extra installs PyTorch and MatterSim, which are needed for the relaxation step. On the first import of MatterSim, the appropriate wheel for your platform (CUDA / Apple Silicon MPS / CPU) is picked automatically. The `.[dev]` extra adds `pytest` and Jupyter so the test suite and notebooks run out of the box.

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

The following snippet walks a chemical formula through the entire pipeline
— predict space group → retrieve templates → substitute → relax with
MatterSim → write out the predicted CIF — using KTaO₃ (a cubic
perovskite in space group 221) as the demonstration target.

Prerequisites: complete steps 1 – 3 of the [reproduction pipeline](#reproducing-the-paper-benchmark) so that the descriptor table
(`data/MP/metadata_with_descriptors.csv`), CIF files (`data/MP/cifs/`), and
trained classifier (`csp_workflow_mp/models/xgb_sg.pkl`) are all in place.

```python
from pathlib import Path
from ase.optimize import BFGS
from ase.filters import UnitCellFilter
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from mattersim.forcefield import MatterSimCalculator

from csp_workflow_mp import (
    compute_periodic_descriptors,
    predict_top_k_space_groups,
    TemplatePool,
    SubstitutionEngine,
)

# 1. Encode the target formula as a 36-dim periodic descriptor.
formula = "KTaO3"
desc = compute_periodic_descriptors(formula)

# 2. Predict the top space group with the trained classifier.
top_sg = predict_top_k_space_groups(desc, k=1)[0]
print(f"top-1 predicted SG: {top_sg}")

# 3. Load the template pool and retrieve the descriptor-nearest MP
#    entries whose space group matches the classifier's top-1 prediction.
pool = TemplatePool(
    "data/MP/metadata_with_descriptors.csv",
    cif_root="data/MP/cifs",
)
hits = pool.search(space_group=top_sg, descriptor_vector=desc, top_n=20)
top_template = hits.iloc[0]
print(f"top template : {top_template['material_id']}  {top_template['formula']}")

# 4. Substitute the target elements onto the top template using
#    the chemical-role–aware substitution engine.
template_struct = Structure.from_file(f"data/MP/cifs/{top_template['material_id']}.cif")
engine   = SubstitutionEngine()
results  = engine.find_substitutions(formula, template_struct)
feasible = next(r for r in results if r.success)
predicted = engine.apply_substitution(template_struct, feasible)

# 5. Relax the predicted structure with MatterSim (BFGS + UnitCellFilter).
#    Auto-select the best available device so this cell works identically
#    on NVIDIA CUDA machines, Apple Silicon, and CPU-only boxes.
import torch
device = ("cuda" if torch.cuda.is_available()
          else "mps"  if torch.backends.mps.is_available()
          else "cpu")

adaptor  = AseAtomsAdaptor()
atoms    = adaptor.get_atoms(predicted)
atoms.calc = MatterSimCalculator(device=device)
opt      = BFGS(UnitCellFilter(atoms), logfile=None)
opt.run(fmax=0.05, steps=500)
relaxed  = adaptor.get_structure(atoms)

# 6. Write the predicted structure to a CIF file.
out_cif = Path("KTaO3_predicted.cif")
relaxed.to(filename=str(out_cif))
print(f"wrote {out_cif}")
```

`notebooks/01_predict_new_composition.ipynb` runs the same recipe end to end and adds a comparison against the MP reference (mp-3614).

> **On disordered targets and relaxation.** MatterSim expects a single
> chemical species at every atomic site, so a candidate structure that carries
> partial occupancies must first be ordered. The pipeline applies a
> dominant-species approximation (replace each partially occupied site by its
> most abundant element) before relaxation. This is discussed in the paper
> Methods (§4.5).

## Examples

| Example | Runtime | Requires | What you get |
|---|---|---|---|
| **`notebooks/01_predict_new_composition.ipynb`** | 10–15 min | full install + trained model + downloaded MP data + MatterSim | Realistic single-composition prediction using KTaO₃: formula → SG prediction → template retrieval on MP → substitution → MatterSim relaxation → CIF output. |
| **[Reproducing the paper benchmark](#reproducing-the-paper-benchmark)** | ~2 h + benchmark | full install + MP API key + ~300 MB free disk for CIFs | Reproduces the 500-target LOEO benchmark under each retrieval strategy. |
| **`notebooks/02_visualise_predictions.ipynb`** | ~30 s | `results/benchmark_raw.csv` from the reproduction step above | Diagnostic figures for benchmark output: per-stage success rates, RMSD distribution, per-complexity SG-match breakdown. |

## Reproducing the paper benchmark

The leave-one-entry-out (LOEO) benchmark in the paper (500 MP targets, seed 42) can be reproduced end-to-end from this repository:

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

`05_run_benchmark.py --help` lists all flags (device selection, custom output directory, resume support, sample count, seed). The aggregated report at `results/benchmark_report.md` reports SG match, SM match, and RMSD on the valid subset (the same definition the paper uses).

The trained model file `csp_workflow_mp/models/xgb_sg.pkl` is not committed to this repository because of its file size. It is regenerated by step 3 above.

## Data

Training data: Materials Project, `e_above_hull < 0.1 eV/atom`, `Z ≤ 95` (CC-BY 4.0). CIF files are downloaded via `scripts/01_download_mp_data.py`.

## Acknowledgements

This work builds on a number of open-source projects. Please cite them when using this repository. See [NOTICES.md](NOTICES.md) for full license text and citation entries.

- [XGBoost](https://xgboost.readthedocs.io) — Apache-2.0 — Chen & Guestrin, KDD 2016.
- [pymatgen](https://pymatgen.org) — MIT — Ong et al., *Comput. Mater. Sci.* 68, 314-319 (2013).
- [spglib](https://spglib.readthedocs.io) — BSD-3-Clause — Togo, Shinohara & Tanaka, *STAM: Methods* 4, 2384822 (2024).
- [Atomic Simulation Environment (ASE)](https://wiki.fysik.dtu.dk/ase/) — LGPL-2.1 — Larsen et al., *J. Phys.: Condens. Matter* 29, 273002 (2017).
- [DScribe](https://singroup.github.io/dscribe/) — Apache-2.0 — Himanen et al., *Comput. Phys. Commun.* 247, 106949 (2020).
- [MatterSim](https://github.com/microsoft/mattersim) — MIT — Yang et al., arXiv:2405.04967 (2024).
- [Materials Project](https://next-gen.materialsproject.org/) — CC-BY 4.0 — Jain et al., *APL Materials* 1, 011002 (2013).

## Citation

```bibtex
@unpublished{Wu2026SGGuided,
  title  = {Space-Group-Guided Template Retrieval for Composition-to-Structure Prediction},
  author = {Wu, Yen-Ju and Xu, Yibin},
  year   = {2026},
  note   = {manuscript in preparation}
}
```

## Troubleshooting

Common first-run issues, in decreasing order of frequency.

**`RuntimeError: MP_API_KEY not set` on `01_download_mp_data.py`.**
Set the environment variable in the current shell as shown in the
[Installation](#installation) section. If you are certain the variable
is set, verify with `echo $MP_API_KEY` (bash / zsh) or `$env:MP_API_KEY`
(PowerShell). Keys registered under a different email than the one
signed into <https://next-gen.materialsproject.org/> will 401.

**`torch` fails to import on Apple Silicon.**
This is almost always an outdated PyTorch wheel. Upgrade explicitly:

```bash
pip install --upgrade torch
python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
```

The second line should print `True` for `mps` availability on Apple Silicon.

**MatterSim downloads its model weights on first use.**
Expect ~1 GB of network traffic the first time `MatterSimCalculator` is
instantiated. The weights are cached under `~/.cache/mattersim/`. If you
are behind a proxy, set `HTTPS_PROXY` before running the notebook.

**`FileNotFoundError: xgb_sg.pkl` when calling `predict_top_k_space_groups`.**
The classifier is not distributed with the repository (it is 200 MB).
Regenerate it by completing steps 1–3 of the
[reproduction pipeline](#reproducing-the-paper-benchmark). The classifier
file lives at `csp_workflow_mp/models/xgb_sg.pkl` after step 3.

**Long-path errors on Windows.**
Windows imposes a 260-character path limit unless
[long paths are enabled](https://learn.microsoft.com/windows/win32/fileio/maximum-file-path-limitation).
The pipeline itself uses short paths, but the MatterSim cache under a
deep user profile directory (`C:\Users\<name with spaces>\.cache\...`)
can hit this on rare configurations. Enabling long paths is the cleanest
fix.

**`pip install -e ".[relaxation]"` fails with a compiler error.**
On Linux, install a C compiler first (`apt install build-essential` or
equivalent). On Windows, install the Visual C++ Build Tools. Most macOS
systems already ship a working toolchain via the Xcode command-line tools.

## License

MIT
