"""End-to-end smoke test: descriptor → tiny TemplatePool → substitution → relax-skip.

This deliberately avoids any dependency on the full MP/AWA databases or on
MatterSim — it builds a 3-template synthetic pool inline so it runs in < 1 s
on any developer machine.
"""

import numpy as np
import pandas as pd
import pytest
from pymatgen.core import Structure, Lattice

import csp_workflow_mp
from csp_workflow_mp import (
    SubstitutionEngine,
    TemplatePool,
    compute_periodic_descriptors,
    sg_to_pearson_prefix,
)


def test_package_version():
    assert csp_workflow_mp.__version__ == "0.1.0"


def test_public_api_symbols_importable():
    # __all__ must round-trip via star-import-style attribute access.
    for name in csp_workflow_mp.__all__:
        assert hasattr(csp_workflow_mp, name), f"missing public symbol: {name}"


def _build_tiny_pool(tmp_path):
    """Construct a 3-row metadata table with matching CIFs on disk."""
    rows = []
    cif_dir = tmp_path / "cifs"
    cif_dir.mkdir()

    specs = [
        ("syn-001", "NaCl",  225, "cF", "Na", "Cl"),
        ("syn-002", "KCl",   225, "cF", "K",  "Cl"),
        ("syn-003", "MgO",   225, "cF", "Mg", "O"),
    ]
    for mid, formula, sg, ps, cation, anion in specs:
        struct = Structure(Lattice.cubic(4.2), [cation, anion],
                           [[0, 0, 0], [0.5, 0.5, 0.5]])
        cif_path = cif_dir / f"{mid}.cif"
        struct.to(filename=str(cif_path))
        desc = compute_periodic_descriptors(formula)
        row = {
            "material_id": mid,
            "formula": formula,
            "space_group": sg,
            "pearson_prefix": ps,
            "cif_path": cif_path.name,
        }
        row.update({f"coef_{i:02d}": desc[i - 1]      for i in range(1, 19)})
        row.update({f"prop_{i:02d}": desc[17 + i]     for i in range(1, 19)})
        rows.append(row)
    return pd.DataFrame(rows), cif_dir


def test_end_to_end_synthetic_pipeline(tmp_path):
    metadata, cif_dir = _build_tiny_pool(tmp_path)

    # 1) Pool loads, descriptor matrix shape sane.
    pool = TemplatePool(metadata, cif_root=cif_dir)
    assert len(pool) == 3

    # 2) Symmetry filter agrees with the SG=225 templates.
    assert sg_to_pearson_prefix(225) == "cF"

    # 3) Substitute KBr (not in pool) onto the closest template.
    target_desc = compute_periodic_descriptors("KBr")
    hits = pool.search(
        space_group=225,
        pearson_prefix="cF",
        descriptor_vector=target_desc,
        top_n=3,
    )
    assert len(hits) >= 1
    # 4) Try substitution on the top hit.
    top = hits.iloc[0]
    template_struct = Structure.from_file(str(cif_dir / f"{top['material_id']}.cif"))
    engine = SubstitutionEngine()
    results = engine.find_substitutions("KBr", template_struct)
    assert any(r.success for r in results)
    pred = engine.apply_substitution(template_struct,
                                     next(r for r in results if r.success))
    assert {s.species_string for s in pred} == {"K", "Br"}
