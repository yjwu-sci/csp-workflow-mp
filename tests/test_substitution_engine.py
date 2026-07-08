"""Smoke tests for SubstitutionEngine + formula parsing."""

import pytest
from pymatgen.core import Structure, Lattice

from csp_workflow_mp import SubstitutionEngine
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
