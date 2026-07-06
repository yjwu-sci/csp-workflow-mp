"""Smoke tests for the 36-dim periodic descriptor."""

import numpy as np
import pytest

from csp_workflow_mp.descriptor import (
    compute_periodic_descriptors,
    compute_descriptors_batch,
)


def test_descriptor_shape():
    v = compute_periodic_descriptors("Li3PO4")
    assert v.shape == (36,)
    assert v.dtype == np.float64


def test_coefficients_sum_to_one():
    v = compute_periodic_descriptors("Li3PO4")
    coefs = v[:18]
    assert coefs.sum() == pytest.approx(1.0, abs=1e-9)


def test_known_groups_for_LiPO4():
    v = compute_periodic_descriptors("Li3PO4")
    coefs = v[:18]
    total_atoms = 3 + 1 + 4
    assert coefs[0]  == pytest.approx(3 / total_atoms)   # group 1 (Li)
    assert coefs[14] == pytest.approx(1 / total_atoms)   # group 15 (P)
    assert coefs[15] == pytest.approx(4 / total_atoms)   # group 16 (O)


def test_fractional_stoichiometry():
    v = compute_periodic_descriptors("Li6.5La3Zr1.5Ta0.5O12")
    assert v.shape == (36,)
    assert v[:18].sum() == pytest.approx(1.0, abs=1e-9)


def test_batch_returns_2d():
    formulas = ["NaCl", "MgO", "Al2O3"]
    M = compute_descriptors_batch(formulas)
    assert M.shape == (3, 36)
    for i, f in enumerate(formulas):
        np.testing.assert_allclose(M[i], compute_periodic_descriptors(f))


def test_unknown_element_does_not_crash():
    # ``Xx`` is a fake element symbol; descriptor should silently skip it.
    v = compute_periodic_descriptors("Xx1Li2")
    assert v.shape == (36,)
    # Li (group 1) should still register; total normalisation uses parsed atom count.
    assert v[0] > 0
