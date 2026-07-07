"""
Space-group classifier interface.

Provides a lazy-loaded wrapper around the trained XGBoost SG classifier so
that a user can go from a periodic descriptor to the top-K predicted space
groups in one call, without having to load pickle files or handle the
LabelEncoder manually.

The classifier model file ``models/xgb_sg.pkl`` is not distributed with the
repository because of its size. Run ``scripts/03_train_xgboost.py`` (roughly
40 min on a modern multi-core CPU) to generate it before calling
:func:`predict_top_k_space_groups` for the first time.

Example
-------
>>> from csp_workflow_mp import compute_periodic_descriptors, \
...                             predict_top_k_space_groups
>>> desc = compute_periodic_descriptors("SrTiO3")
>>> predict_top_k_space_groups(desc, k=1)      # doctest: +SKIP
[221]
>>> predict_top_k_space_groups(desc, k=3)      # doctest: +SKIP
[221, 99, 123]
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Union

import numpy as np


_MODEL_PATH = Path(__file__).parent / "models" / "xgb_sg.pkl"
_CACHE: tuple | None = None


def _load_model() -> tuple:
    """Lazy-load the pickled (model, encoder) pair, cached in module scope."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    if not _MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained SG classifier not found at {_MODEL_PATH}. "
            "Run scripts/03_train_xgboost.py first to generate it; this "
            "takes roughly 40 min on a modern multi-core CPU."
        )
    with open(_MODEL_PATH, "rb") as fh:
        pkg = pickle.load(fh)
    _CACHE = (pkg["model"], pkg["encoder"])
    return _CACHE


def predict_top_k_space_groups(
    descriptor: np.ndarray,
    k: int = 1,
) -> Union[List[int], List[List[int]]]:
    """
    Predict the top-K space groups for one or more periodic descriptors.

    Parameters
    ----------
    descriptor : np.ndarray
        Either a 1-D array of shape ``(36,)`` for a single composition, or a
        2-D array of shape ``(N, 36)`` for a batch.
    k : int, default 1
        Number of top predictions to return. ``k = 1`` corresponds to the
        primary retrieval mode reported in the paper.

    Returns
    -------
    list[int] or list[list[int]]
        Space-group numbers in descending order of predicted probability.
        Returns a flat list when the input is 1-D, a list of lists when
        the input is 2-D.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    model, encoder = _load_model()
    arr = np.atleast_2d(np.asarray(descriptor, dtype=np.float32))
    if arr.shape[-1] != 36:
        raise ValueError(
            f"Descriptor must have 36 features, got shape {arr.shape}. "
            "Use compute_periodic_descriptors(formula) to generate one."
        )

    proba = model.predict_proba(arr)
    if k > proba.shape[1]:
        raise ValueError(
            f"k = {k} exceeds the number of SG classes seen during "
            f"training ({proba.shape[1]})."
        )
    top_k_idx = np.argsort(proba, axis=1)[:, -k:][:, ::-1]
    predictions = [
        [int(encoder.inverse_transform([j])[0]) for j in row]
        for row in top_k_idx
    ]

    if np.ndim(descriptor) == 1:
        return predictions[0]
    return predictions
