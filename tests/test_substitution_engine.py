"""Smoke tests for SubstitutionEngine + symmetry filter."""

import numpy as np
import pytest
from pymatgen.core import Structure, Lattice

from csp_workflow_mp import (
    SubstitutionEngine,
    sg_to_pearson_prefix,
    is_valid_combination,
    allowed_pearson_set,
)
from csp_workflow_mp.substitution_engine import parse_formula


# ---------------------------------------------------------------- helpers


def _rocksalt(cation: str, anion: str, a: float = 4.2) -> Structure:
    """Build a 2-atom rocksalt-like cell inline (NaCl-type, Fm-3m, SG=225)."""
    lattice = Lattice.cubic(a)
    return Structure(lattice, [cation, anion], [[0, 0, 0], [0.5, 0.5, 0.5]])


# ---------------------------------------------------------------- parse_formula


def test_parse_formula_integer():
    elems = parse_formula("Li3PO4")
    assert elems == {"Li": 3.0, "P": 1.0, "O": 4.0}


def test_parse_formula_fractional():
    elems = parse_formula("Li6.5La3Zr1.5Ta0.5O12")
    assert elems["Li"] == pytest.approx(6.5)
    assert elems["Ta"] == pytest.approx(0.5)


# ---------------------------------------------------------------- symmetry filter


@pytest.mark.parametrize("sg,expected", [
    (1,   "aP"),    # triclinic
    (14,  "mP"),    # monoclinic P
    (62,  "oP"),    # orthorhombic P
    (139, "tI"),    # tetragonal I
    (166, "hR"),    # trigonal R
    (194, "hP"),    # hexagonal P
    (225, "cF"),    # cubic F (NaCl)
    (229, "cI"),    # cubic I
])
def test_sg_to_pearson_canonical(sg, expected):
    assert sg_to_pearson_prefix(sg) == expected


def test_is_valid_combination():
    assert is_valid_combination(225, "cF") is True
    assert is_valid_combination(225, "cP") is False


def test_allowed_pearson_set_cubic():
    # The deprecated multi-prefix function should accept all three cubic centerings for SG=225.
    s = allowed_pearson_set(225)
    assert {"cP", "cI", "cF"}.issubset(s)


# ---------------------------------------------------------------- SubstitutionEngine


def test_engine_instantiates_with_defaults():
    eng = SubstitutionEngine()
    assert eng.max_solutions == 10
    assert eng.use_relaxed_matching is True


def test_check_compatibility_identical_swap():
    """NaCl → KCl on a NaCl-type template: Na→K should map cleanly."""
    template = _rocksalt("Na", "Cl")
    eng = SubstitutionEngine()
    ok, _z, _issues = eng.check_compatibility("KCl", template)
    assert ok is True


def test_check_compatibility_incompatible_stoichiometry():
    """A binary 1:1 template cannot host a 3:1 target without role merging."""
    template = _rocksalt("Na", "Cl")
    eng = SubstitutionEngine(use_relaxed_matching=False)
    ok, _z, _issues = eng.check_compatibility("Li3P", template)
    assert ok is False


def test_find_substitutions_returns_success_for_simple_swap():
    template = _rocksalt("Na", "Cl")
    eng = SubstitutionEngine()
    results = eng.find_substitutions("KCl", template)
    assert any(r.success for r in results), "expected at least one feasible mapping"
    succ = next(r for r in results if r.success)
    assert "Na" in succ.substitution_dict
    assert succ.substitution_dict["Na"] == "K"


def test_apply_substitution_preserves_atom_count():
    template = _rocksalt("Na", "Cl")
    eng = SubstitutionEngine()
    results = eng.find_substitutions("KCl", template)
    succ = next(r for r in results if r.success)
    pred = eng.apply_substitution(template, succ)
    assert len(pred) == len(template)
    assert {site.species_string for site in pred} == {"K", "Cl"}
