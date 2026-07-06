"""
Crystallographic compatibility filter for (space group, Pearson symbol prefix) pairs.

v2.0 — Corrected SG-primary approach
=====================================
Each space group maps to exactly ONE Pearson-symbol prefix (determined by
crystal system + centering type from International Tables Vol. A).
The previous Cartesian-product approach (top-5 SG × top-5 PS = 25 pairs)
was crystallographically incorrect and has been replaced.

New logic:
    1. Take top-k SG predictions
    2. Map each SG to its unique PS prefix via deterministic lookup
    3. Optionally rerank using PS model probability as a consistency signal
    4. Retrieve templates for each (SG, PS) family

The old ``filter_compatible_pairs()`` is retained for backward compatibility
and ablation studies but is marked as deprecated.
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, List, Optional, Set, Tuple

# ====================================================================
# Deterministic SG → Pearson-symbol prefix mapping  (230 entries)
# Source: International Tables for Crystallography, Volume A
# Rule: crystal system (from SG range) + centering type (from HM symbol)
# ====================================================================

# -- Monoclinic C-centered (all others in 3-15 are P) --
_MONOCLINIC_C: Set[int] = {5, 8, 9, 12, 15}

# -- Orthorhombic centering subsets --
_ORTHO_S: Set[int] = {20, 21, 35, 36, 37, 38, 39, 40, 41, 63, 64, 65, 66, 67, 68}
_ORTHO_F: Set[int] = {22, 42, 43, 69, 70}
_ORTHO_I: Set[int] = {23, 24, 44, 45, 46, 71, 72, 73, 74}

# -- Tetragonal I-centered (all others in 75-142 are P) --
_TETRA_I: Set[int] = {
    79, 80, 82, 87, 88, 97, 98,
    107, 108, 109, 110, 119, 120, 121, 122,
    139, 140, 141, 142,
}

# -- Trigonal R-centered (all others in 143-167 are hP) --
_TRIGONAL_R: Set[int] = {146, 148, 155, 160, 161, 166, 167}

# -- Cubic centering subsets --
_CUBIC_F: Set[int] = {196, 202, 203, 209, 210, 216, 219, 225, 226, 227, 228}
_CUBIC_I: Set[int] = {197, 199, 204, 206, 211, 214, 217, 220, 229, 230}


def sg_to_pearson_prefix(space_group: int) -> str:
    """
    Return the unique Pearson-symbol prefix for a given space group number.

    This is a deterministic mapping based on crystallographic conventions:
    each space group belongs to exactly one crystal system and has exactly
    one centering type, which together determine the Pearson prefix.

    Parameters
    ----------
    space_group : int
        Space group number (1–230).

    Returns
    -------
    str
        Two-character Pearson prefix (e.g., ``"cF"``, ``"hR"``, ``"oP"``).

    Raises
    ------
    ValueError
        If space_group is outside the valid range 1–230.

    Examples
    --------
    >>> sg_to_pearson_prefix(225)  # Fm-3m
    'cF'
    >>> sg_to_pearson_prefix(166)  # R-3m
    'hR'
    >>> sg_to_pearson_prefix(62)   # Pnma
    'oP'
    """
    sg = int(space_group)
    if sg < 1 or sg > 230:
        raise ValueError(f"Space group must be 1–230, got {sg}")

    # Triclinic
    if sg <= 2:
        return "aP"

    # Monoclinic
    if sg <= 15:
        return "mS" if sg in _MONOCLINIC_C else "mP"

    # Orthorhombic
    if sg <= 74:
        if sg in _ORTHO_S:
            return "oS"
        if sg in _ORTHO_F:
            return "oF"
        if sg in _ORTHO_I:
            return "oI"
        return "oP"

    # Tetragonal
    if sg <= 142:
        return "tI" if sg in _TETRA_I else "tP"

    # Trigonal
    if sg <= 167:
        return "hR" if sg in _TRIGONAL_R else "hP"

    # Hexagonal
    if sg <= 194:
        return "hP"

    # Cubic
    if sg <= 230:
        if sg in _CUBIC_F:
            return "cF"
        if sg in _CUBIC_I:
            return "cI"
        return "cP"

    raise ValueError(f"Space group must be 1–230, got {sg}")


# Build the complete mapping dict for fast lookup and export
SG_TO_PS_PREFIX: Dict[int, str] = {sg: sg_to_pearson_prefix(sg) for sg in range(1, 231)}


# ====================================================================
# SG-Primary filter with PS reranking  (NEW — corrected approach)
# ====================================================================

def filter_sg_primary(
    sg_predictions: List[Tuple[int, float]],
    ps_predictions: Optional[List[Tuple[str, float]]] = None,
    top_n_sg: int = 10,
    lambda_weight: float = 1.0,
) -> List[Tuple[int, str, float]]:
    """
    Build (SG, PS) candidate families using the SG-primary approach.

    Each top-k SG prediction is mapped to its unique crystallographic PS
    prefix. If PS predictions are provided, they are used as a reranking
    signal (consistency check) rather than an independent pairing axis.

    Parameters
    ----------
    sg_predictions : list of (space_group, probability)
        Ranked space-group predictions from the classifier.
        Only the first ``top_n_sg`` entries are used.
    ps_predictions : list of (pearson_prefix, probability), optional
        Ranked Pearson-prefix predictions. If provided, used for reranking
        via ``score = P_SG(g) * P_PS(f(g))^lambda``.
        If None, ranking is purely by SG probability.
    top_n_sg : int
        Number of top SG predictions to consider (default: 10).
    lambda_weight : float
        Weight for PS probability in reranking (default: 1.0).
        - lambda=0: ignore PS model (pure SG ranking)
        - lambda=1: equal weight to both models
        - lambda=0.5: reduced PS influence

    Returns
    -------
    list of (space_group, pearson_prefix, score)
        Candidate families sorted by descending score.

    Examples
    --------
    >>> sg_preds = [(225, 0.45), (123, 0.22), (221, 0.12)]
    >>> ps_preds = [("cF", 0.37), ("tP", 0.10), ("cP", 0.22)]
    >>> filter_sg_primary(sg_preds, ps_preds, top_n_sg=3, lambda_weight=1.0)
    [(225, 'cF', 0.1665), (221, 'cP', 0.0264), (123, 'tP', 0.022)]
    """
    # Build PS probability lookup
    ps_prob_map: Dict[str, float] = {}
    if ps_predictions is not None:
        for ps, prob in ps_predictions:
            ps_prob_map[str(ps).strip()] = float(prob)

    pairs = []
    seen_sg = set()

    for sg, sg_prob in sg_predictions[:top_n_sg]:
        sg = int(sg)
        if sg in seen_sg:
            continue
        seen_sg.add(sg)

        try:
            ps = sg_to_pearson_prefix(sg)
        except ValueError:
            continue

        sg_prob = float(sg_prob)

        if ps_predictions is not None and lambda_weight > 0:
            ps_prob = ps_prob_map.get(ps, 1e-6)  # small floor if PS not in top-k
            score = sg_prob * (ps_prob ** lambda_weight)
        else:
            score = sg_prob

        pairs.append((sg, ps, score))

    # Sort by score descending
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs


# ====================================================================
# Legacy functions (backward compatibility + ablation)
# ====================================================================

# Space groups that use rhombohedral centering (hR) within the trigonal range
_HR_SG: Set[int] = _TRIGONAL_R

# Trigonal space groups with hexagonal P setting (hP)
_HP_TRIGONAL_SG: Set[int] = {
    143, 144, 145, 147, 149, 150, 151, 152, 153, 154,
    156, 157, 158, 159, 162, 163, 164, 165,
}


def allowed_pearson_set(space_group: int) -> Set[str]:
    """
    Return the set of Pearson-symbol prefixes compatible with a given space group.

    .. deprecated::
        This function returns MULTIPLE possible prefixes per SG, which is
        crystallographically incorrect (each SG has exactly one prefix).
        It is retained for backward compatibility and ablation studies.
        Use ``sg_to_pearson_prefix()`` instead.

    Parameters
    ----------
    space_group : int
        Space group number (1–230).

    Returns
    -------
    set of str
        Compatible Pearson prefixes.
    """
    sg = int(space_group)

    if 1 <= sg <= 2:
        return {"aP"}
    if 3 <= sg <= 15:
        return {"mP", "mS"}
    if 16 <= sg <= 74:
        return {"oP", "oS", "oI", "oF"}
    if 75 <= sg <= 142:
        return {"tP", "tI"}
    if sg in _HR_SG:
        return {"hR"}
    if sg in _HP_TRIGONAL_SG:
        return {"hP"}
    if 168 <= sg <= 194:
        return {"hP"}
    if 195 <= sg <= 230:
        return {"cP", "cI", "cF"}

    return set()


def is_valid_combination(space_group: int, pearson_prefix: str) -> bool:
    """
    Check whether a (space_group, pearson_prefix) pair is crystallographically valid.

    Parameters
    ----------
    space_group : int
        Space group number (1–230).
    pearson_prefix : str
        Two-character Pearson prefix, e.g. ``"oP"``, ``"hR"``, ``"cF"``.

    Returns
    -------
    bool
    """
    return str(pearson_prefix).strip() == sg_to_pearson_prefix(int(space_group))


def filter_compatible_pairs(
    sg_predictions: List[Tuple[int, float]],
    ps_predictions: List[Tuple[str, float]],
    top_n: int = 25,
) -> List[Tuple[int, str, float]]:
    """
    Combine top-k SG and PS predictions via Cartesian product, keeping
    only crystallographically compatible pairs.

    .. deprecated::
        This uses the old (incorrect) Cartesian-product approach.
        Retained for backward compatibility and ablation comparison.
        Use ``filter_sg_primary()`` for the corrected approach.

    Parameters
    ----------
    sg_predictions : list of (space_group, probability)
    ps_predictions : list of (pearson_prefix, probability)
    top_n : int
        Maximum number of compatible pairs to return.

    Returns
    -------
    list of (space_group, pearson_prefix, score)
        Compatible pairs sorted by descending score = sg_prob × ps_prob.
    """
    pairs = []
    for sg, sg_prob in sg_predictions:
        allowed = allowed_pearson_set(sg)
        for ps, ps_prob in ps_predictions:
            if ps in allowed:
                pairs.append((sg, ps, sg_prob * ps_prob))

    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_n]


# Human-readable names for documentation
CRYSTAL_FAMILY_MAP = {
    "aP": "triclinic",
    "mP": "monoclinic-P",
    "mS": "monoclinic-C/A/I",
    "oP": "orthorhombic-P",
    "oS": "orthorhombic-C/A/B",
    "oI": "orthorhombic-I",
    "oF": "orthorhombic-F",
    "tP": "tetragonal-P",
    "tI": "tetragonal-I",
    "hP": "hexagonal-P / trigonal-P",
    "hR": "trigonal-R (rhombohedral)",
    "cP": "cubic-P",
    "cI": "cubic-I",
    "cF": "cubic-F",
}
