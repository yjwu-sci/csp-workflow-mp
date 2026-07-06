"""
Periodic descriptor computation from chemical formula.

Implements the 36-dimensional periodic descriptor scheme from:
  Wu et al., Sci. Technol. Adv. Mater.: Methods 5, 2513218 (2025).

For each of the 18 periodic-table groups, two features are computed:
  - Group coefficient: fraction of total atoms belonging to that group
  - Group property: weighted-average atomic number of elements in that group

This yields a compact, chemically interpretable feature vector suitable
for XGBoost classification of space group and Pearson symbol.
"""

from __future__ import annotations

from typing import Dict, Optional
import numpy as np

from csp_workflow_mp.substitution_engine import parse_formula

# Periodic table group assignments (IUPAC group 1–18)
# Element → group number
ELEMENT_GROUP: Dict[str, int] = {
    # Group 1
    "H": 1, "Li": 1, "Na": 1, "K": 1, "Rb": 1, "Cs": 1, "Fr": 1,
    # Group 2
    "Be": 2, "Mg": 2, "Ca": 2, "Sr": 2, "Ba": 2, "Ra": 2,
    # Group 3
    "Sc": 3, "Y": 3,
    "La": 3, "Ce": 3, "Pr": 3, "Nd": 3, "Pm": 3, "Sm": 3, "Eu": 3,
    "Gd": 3, "Tb": 3, "Dy": 3, "Ho": 3, "Er": 3, "Tm": 3, "Yb": 3, "Lu": 3,
    "Ac": 3, "Th": 3, "Pa": 3, "U": 3, "Np": 3, "Pu": 3, "Am": 3, "Cm": 3,
    # Group 4
    "Ti": 4, "Zr": 4, "Hf": 4,
    # Group 5
    "V": 5, "Nb": 5, "Ta": 5,
    # Group 6
    "Cr": 6, "Mo": 6, "W": 6,
    # Group 7
    "Mn": 7, "Tc": 7, "Re": 7,
    # Group 8
    "Fe": 8, "Ru": 8, "Os": 8,
    # Group 9
    "Co": 9, "Rh": 9, "Ir": 9,
    # Group 10
    "Ni": 10, "Pd": 10, "Pt": 10,
    # Group 11
    "Cu": 11, "Ag": 11, "Au": 11,
    # Group 12
    "Zn": 12, "Cd": 12, "Hg": 12,
    # Group 13
    "B": 13, "Al": 13, "Ga": 13, "In": 13, "Tl": 13,
    # Group 14
    "C": 14, "Si": 14, "Ge": 14, "Sn": 14, "Pb": 14,
    # Group 15
    "N": 15, "P": 15, "As": 15, "Sb": 15, "Bi": 15,
    # Group 16
    "O": 16, "S": 16, "Se": 16, "Te": 16, "Po": 16,
    # Group 17
    "F": 17, "Cl": 17, "Br": 17, "I": 17, "At": 17,
    # Group 18
    "He": 18, "Ne": 18, "Ar": 18, "Kr": 18, "Xe": 18, "Rn": 18,
}

# Atomic numbers for property computation
ATOMIC_NUMBER: Dict[str, int] = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22,
    "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29,
    "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
    "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43,
    "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
    "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57,
    "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64,
    "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71,
    "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78,
    "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85,
    "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90, "Pa": 91, "U": 92,
    "Np": 93, "Pu": 94, "Am": 95, "Cm": 96,
}


def compute_periodic_descriptors(
    formula: str,
    n_groups: int = 18,
) -> np.ndarray:
    """
    Compute the 36-dimensional periodic descriptor vector for a chemical formula.

    Parameters
    ----------
    formula : str
        Chemical formula, e.g. ``"Li3PO4"`` or ``"Li6.5La3Zr1.5Ta0.5O12"``.
    n_groups : int
        Number of periodic table groups (default 18).

    Returns
    -------
    np.ndarray of shape (36,)
        [coef_01, ..., coef_18, prop_01, ..., prop_18]
    """
    elements = parse_formula(formula)
    total_atoms = sum(elements.values())

    if total_atoms == 0:
        return np.zeros(2 * n_groups)

    coefficients = np.zeros(n_groups)
    properties = np.zeros(n_groups)

    # Accumulate per-group
    group_counts: Dict[int, float] = {}    # group → total atom count
    group_weighted_z: Dict[int, float] = {}  # group → sum(count * Z)

    for elem, count in elements.items():
        g = ELEMENT_GROUP.get(elem)
        z = ATOMIC_NUMBER.get(elem, 0)
        if g is None:
            continue
        group_counts[g] = group_counts.get(g, 0.0) + count
        group_weighted_z[g] = group_weighted_z.get(g, 0.0) + count * z

    for g in range(1, n_groups + 1):
        gc = group_counts.get(g, 0.0)
        coefficients[g - 1] = gc / total_atoms
        if gc > 0:
            properties[g - 1] = group_weighted_z.get(g, 0.0) / gc
        else:
            properties[g - 1] = 0.0

    return np.concatenate([coefficients, properties])


def compute_descriptors_batch(
    formulas: list,
    n_groups: int = 18,
) -> np.ndarray:
    """
    Compute periodic descriptors for a list of formulas.

    Returns
    -------
    np.ndarray of shape (n_formulas, 36)
    """
    return np.array([compute_periodic_descriptors(f, n_groups) for f in formulas])


DESCRIPTOR_COLUMNS = (
    [f"coef_{i:02d}" for i in range(1, 19)]
    + [f"prop_{i:02d}" for i in range(1, 19)]
)
