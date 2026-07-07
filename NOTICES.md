# Third-party notices

`csp-workflow-mp` depends on a number of open-source Python packages and on
public Materials Project data. This file lists their licenses and the
citations they ask users to give when they publish results produced with
their software.

None of these projects require modification of `csp-workflow-mp`; we
depend on them as installed packages via `pip install -e .[relaxation]`
and never redistribute their source. This file exists to make each project's
attribution obligation explicit for anyone who reuses `csp-workflow-mp`
in publications or downstream software.

## Runtime dependencies

### XGBoost (Apache-2.0)

- Repository: <https://github.com/dmlc/xgboost>
- License: Apache License 2.0 (`LICENSE` file at repository root).
- Used for: multi-class classification of space groups from the periodic
  descriptor (`scripts/03_train_xgboost.py`,
  `csp_workflow_mp/classifier.py`).
- Please cite: T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting
  System", *Proc. 22nd ACM SIGKDD*, 785–794 (2016).
  <https://doi.org/10.1145/2939672.2939785>

### pymatgen (MIT)

- Repository: <https://github.com/materialsproject/pymatgen>
- License: MIT.
- Used for: crystal structure I/O, SpacegroupAnalyzer, StructureMatcher,
  composition handling throughout the pipeline.
- Please cite: S. P. Ong et al., "Python Materials Genomics (pymatgen):
  A robust, open-source Python library for materials analysis",
  *Comput. Mater. Sci.* 68, 314–319 (2013).
  <https://doi.org/10.1016/j.commatsci.2012.10.028>

### spglib (BSD-3-Clause)

- Repository: <https://github.com/spglib/spglib>
- License: BSD 3-Clause.
- Used for: crystallographic symmetry analysis inside pymatgen and inside
  the substitution engine's site grouping.
- Please cite: A. Togo, K. Shinohara and I. Tanaka, "Spglib: a software
  library for crystal symmetry search", *Sci. Technol. Adv. Mater.:
  Methods* 4, 2384822 (2024).
  <https://doi.org/10.1080/27660400.2024.2384822>

### Atomic Simulation Environment (LGPL-2.1)

- Repository: <https://gitlab.com/ase/ase>
- License: LGPL 2.1. We do not modify ASE; we only import and call its
  optimiser and cell filter, which under LGPL does not impose any
  copyleft obligation on the calling code.
- Used for: BFGS geometry optimisation and `UnitCellFilter` during
  MatterSim relaxation.
- Please cite: A. H. Larsen et al., "The atomic simulation environment
  — a Python library for working with atoms", *J. Phys.: Condens.
  Matter* 29, 273002 (2017). <https://doi.org/10.1088/1361-648X/aa680e>

### DScribe (Apache-2.0)

- Repository: <https://github.com/SINGROUP/dscribe>
- License: Apache License 2.0.
- Used for: SOAP local-environment descriptor computation in the
  benchmark helper `scripts/structural_eval_helpers.py`.
- Please cite: L. Himanen et al., "DScribe: Library of descriptors for
  machine learning in materials science", *Comput. Phys. Commun.* 247,
  106949 (2020). <https://doi.org/10.1016/j.cpc.2019.106949>

### MatterSim (MIT)

- Repository: <https://github.com/microsoft/mattersim>
- License: MIT (`LICENSE` file at repository root).
- Used for: neural-network interatomic potential used to relax the
  substituted candidate structures.
- Model weights license: the pretrained MatterSim v1.0.0 weights are
  distributed under the MatterSim repository terms; users should consult
  Microsoft's licence text before commercial use.
- Please cite: H. Yang et al., "MatterSim: A Deep Learning Atomistic
  Model Across Elements, Temperatures and Pressures",
  arXiv:2405.04967 (2024). <https://arxiv.org/abs/2405.04967>

### mp-api (BSD, modified)

- Repository: <https://github.com/materialsproject/api>
- License: Modified BSD (distributed by Materials Project).
- Used for: bulk download of MP metadata and structural CIFs in
  `scripts/01_download_mp_data.py`.
- Please cite the Materials Project paper (see below) when publishing
  results derived from MP data obtained through this API.

### emmet-core (Modified BSD)

- Repository: <https://github.com/materialsproject/emmet>
- License: Modified BSD.
- Used indirectly as a schema dependency of mp-api.

### NumPy / SciPy / pandas / scikit-learn (BSD)

- Standard scientific-Python stack; all under BSD-family licences.
- Please cite the projects when they are central to your published
  analysis (NumPy: Harris et al., *Nature* 585, 357 (2020); SciPy:
  Virtanen et al., *Nat. Methods* 17, 261 (2020); pandas: McKinney,
  *SciPy* proceedings (2010); scikit-learn: Pedregosa et al., *JMLR*
  12, 2825–2830 (2011)).

### Matplotlib / seaborn (BSD-like)

- Matplotlib: PSF-based licence; please cite Hunter,
  *Comput. Sci. Eng.* 9, 90–95 (2007).
- seaborn: BSD-3-Clause; please cite Waskom, *JOSS* 6, 3021 (2021).

## Data

### Materials Project (CC-BY 4.0)

- URL: <https://next-gen.materialsproject.org/>
- License: CC-BY 4.0 for structural and property data.
- Used for: all training and benchmark data in `data/MP/`.
- Please cite: A. Jain et al., "The Materials Project: A materials
  genome approach to accelerating materials innovation",
  *APL Materials* 1, 011002 (2013).
  <https://doi.org/10.1063/1.4812323>

### AtomWork-Adv. (NIMS access-controlled)

- URL: <https://atomwork-adv.nims.go.jp/>
- The Supplementary Information of the paper reports cross-database
  results computed against AtomWork-Adv. metadata. This dataset is
  **not redistributed** with this repository. Access is granted to
  authorised users by NIMS. `scripts/06_cross_db_validation.py` is
  kept for transparency but cannot be run against the public data
  distribution alone.

---

## What this repository redistributes

Only the source code in this repository (see `LICENSE`, MIT). No
third-party source or model weights are bundled. Users obtain external
dependencies from PyPI / GitHub through the standard `pip install`
process, and they obtain MatterSim's pretrained weights and the
Materials Project's CIFs from the respective upstream projects.
