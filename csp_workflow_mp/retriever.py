"""
Template retriever for space-group-guided crystal structure generation.

Searches a user-provided template pool (metadata CSV + CIF directory) for
structures whose space group matches a target, ranks them by periodic-
descriptor cosine similarity, and optionally checks substitution
feasibility.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine as cosine_dist

from csp_workflow_mp.substitution_engine import SubstitutionEngine, parse_formula   # noqa: F401

logger = logging.getLogger(__name__)


class TemplatePool:
    """
    A searchable pool of template crystal structures.

    Parameters
    ----------
    metadata : pd.DataFrame or str/Path
        Table of template metadata. Required columns:
        - ``formula``: reduced chemical formula
        - ``space_group``: space group number (int)
        - ``cif_path``: path to the CIF file (absolute or relative to ``cif_root``)

        Optional columns:
        - ``material_id``: unique identifier
        - ``coef_01`` ... ``coef_18``, ``prop_01`` ... ``prop_18``: periodic descriptors

    cif_root : str or Path, optional
        Root directory for CIF files if ``cif_path`` contains relative paths.
    descriptor_cols : list of str, optional
        Column names for periodic descriptor features. Default: ``coef_01..18 + prop_01..18``.
    exclude_ids : set, optional
        Material IDs to exclude from retrieval (e.g., for leave-one-out validation).
    """

    DEFAULT_COEF_COLS = [f"coef_{i:02d}" for i in range(1, 19)]
    DEFAULT_PROP_COLS = [f"prop_{i:02d}" for i in range(1, 19)]

    def __init__(
        self,
        metadata: Union[pd.DataFrame, str, Path],
        cif_root: Optional[Union[str, Path]] = None,
        descriptor_cols: Optional[List[str]] = None,
        exclude_ids: Optional[set] = None,
    ):
        if isinstance(metadata, (str, Path)):
            path = Path(metadata)
            if path.suffix == ".csv":
                self._df = pd.read_csv(path)
            elif path.suffix in (".xlsx", ".xls"):
                self._df = pd.read_excel(path)
            else:
                raise ValueError(f"Unsupported metadata format: {path.suffix}")
        else:
            self._df = metadata.copy()

        self._cif_root = Path(cif_root) if cif_root else None

        col_map = self._detect_columns()
        self._formula_col = col_map["formula"]
        self._sg_col = col_map["space_group"]
        self._cif_col = col_map.get("cif_path")
        self._id_col = col_map.get("material_id")

        if descriptor_cols is not None:
            self._desc_cols = descriptor_cols
        else:
            self._desc_cols = [
                c for c in self.DEFAULT_COEF_COLS + self.DEFAULT_PROP_COLS
                if c in self._df.columns
            ]

        self._desc_matrix = None
        if self._desc_cols:
            self._desc_matrix = self._df[self._desc_cols].fillna(0).to_numpy(dtype=float)

        if exclude_ids and self._id_col:
            mask = ~self._df[self._id_col].isin(exclude_ids)
            self._df = self._df[mask].reset_index(drop=True)
            if self._desc_matrix is not None:
                self._desc_matrix = self._desc_matrix[mask.values]

        logger.info(
            "TemplatePool: %d templates, %d descriptor features",
            len(self._df), len(self._desc_cols),
        )

    def _detect_columns(self) -> Dict[str, str]:
        """Auto-detect column names with flexible naming conventions."""
        cols = set(self._df.columns)
        mapping: Dict[str, str] = {}

        for candidate in ["formula", "Formula", "refined_formula",
                          "standard_formula", "composition"]:
            if candidate in cols:
                mapping["formula"] = candidate
                break
        if "formula" not in mapping:
            raise ValueError(
                "No formula column found. Expected one of: formula, Formula, "
                "refined_formula, etc."
            )

        for candidate in ["space_group", "space_group_number", "spg",
                          "sg", "Space_group"]:
            if candidate in cols:
                mapping["space_group"] = candidate
                break
        if "space_group" not in mapping:
            raise ValueError("No space group column found.")

        for candidate in ["cif_path", "cif_file", "CIF", "structure_file"]:
            if candidate in cols:
                mapping["cif_path"] = candidate
                break

        for candidate in ["material_id", "Material ID", "AWA-id", "mp_id",
                          "id", "substance_id"]:
            if candidate in cols:
                mapping["material_id"] = candidate
                break

        return mapping

    def search(
        self,
        space_group: Optional[int] = None,
        descriptor_vector: Optional[np.ndarray] = None,
        target_formula: Optional[str] = None,
        top_n: int = 20,
        check_substitution: bool = False,
        substitution_engine: Optional[SubstitutionEngine] = None,
    ) -> pd.DataFrame:
        """
        Return the top-N templates whose space group matches the input,
        ranked by descriptor cosine similarity.

        Parameters
        ----------
        space_group : int, optional
            If set, restrict candidates to entries with this space group.
            If None, all templates are considered (unconstrained retrieval).
        descriptor_vector : np.ndarray, optional
            36-dim periodic descriptor of the target composition. Required
            for descriptor-similarity ranking.
        target_formula : str, optional
            Only used when ``check_substitution=True`` to test whether
            each retrieved template admits a chemically valid substitution.
        top_n : int
            Number of highest-ranked templates to return.
        check_substitution : bool
            If True, run the substitution engine on each hit and add a
            ``substitution_feasible`` column.
        substitution_engine : SubstitutionEngine, optional
            Reused engine instance (avoids re-initialising for every call).

        Returns
        -------
        pd.DataFrame
            Selected metadata rows with a ``pd_distance`` column and,
            optionally, ``substitution_feasible``.
        """
        mask = pd.Series(True, index=self._df.index)

        if space_group is not None:
            mask &= self._df[self._sg_col].astype(int) == int(space_group)

        candidates = self._df[mask].copy()
        if len(candidates) == 0:
            return candidates

        if descriptor_vector is not None and self._desc_matrix is not None:
            desc_sub = self._desc_matrix[mask.values]
            v = np.array(descriptor_vector, dtype=float).reshape(1, -1)
            distances = np.array([
                cosine_dist(v.ravel(), row) if np.linalg.norm(row) > 1e-12 else 1.0
                for row in desc_sub
            ])
            candidates = candidates.copy()
            candidates["pd_distance"] = distances
            candidates = candidates.sort_values("pd_distance").head(top_n)
        else:
            candidates = candidates.head(top_n)

        if check_substitution and target_formula:
            if substitution_engine is None:
                substitution_engine = SubstitutionEngine()

            feasibility = []
            for _, row in candidates.iterrows():
                cif_path = self._resolve_cif(row)
                if cif_path and cif_path.exists():
                    try:
                        from pymatgen.core import Structure
                        template = Structure.from_file(str(cif_path))
                        ok, _z, _issues = substitution_engine.check_compatibility(
                            target_formula, template
                        )
                        feasibility.append(ok)
                    except Exception as e:
                        logger.debug("CIF load failed for %s: %s", cif_path, e)
                        feasibility.append(False)
                else:
                    feasibility.append(None)

            candidates["substitution_feasible"] = feasibility

        return candidates.reset_index(drop=True)

    def _resolve_cif(self, row) -> Optional[Path]:
        """Resolve the CIF path for a metadata row."""
        if self._cif_col is None or pd.isna(row.get(self._cif_col)):
            return None
        p = Path(str(row[self._cif_col]))
        if p.is_absolute() and p.exists():
            return p
        if self._cif_root:
            full = self._cif_root / p
            if full.exists():
                return full
        return p if p.exists() else None

    def __len__(self) -> int:
        return len(self._df)

    def __repr__(self) -> str:
        return f"TemplatePool({len(self._df)} templates, {len(self._desc_cols)} descriptors)"
