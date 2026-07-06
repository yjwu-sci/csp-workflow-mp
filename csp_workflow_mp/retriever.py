"""
Database-agnostic template retriever for crystal structure generation.

Searches any user-provided template pool (metadata CSV + CIF directory) for
structures matching predicted symmetry families, ranks them by periodic-
descriptor similarity, and checks substitution feasibility.

Supports AWA, Materials Project, ICSD, AFLOW, or any custom database —
the user provides a metadata table and a CIF directory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine as cosine_dist

from csp_workflow_mp.symmetry_filter import is_valid_combination, filter_compatible_pairs
from csp_workflow_mp.substitution_engine import SubstitutionEngine, parse_formula

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
        - ``pearson_prefix``: Pearson symbol prefix (e.g., ``"oP"``)
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
        self._ps_col = col_map["pearson_prefix"]
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
        mapping = {}

        for candidate in ["formula", "Formula", "refined_formula", "standard_formula", "composition"]:
            if candidate in cols:
                mapping["formula"] = candidate
                break
        if "formula" not in mapping:
            raise ValueError("No formula column found. Expected one of: formula, Formula, refined_formula, etc.")

        for candidate in ["space_group", "space_group_number", "spg", "sg", "Space_group"]:
            if candidate in cols:
                mapping["space_group"] = candidate
                break
        if "space_group" not in mapping:
            raise ValueError("No space group column found.")

        for candidate in ["pearson_prefix", "pearson_symbol_prefix", "ps_prefix", "Pearson_prefix"]:
            if candidate in cols:
                mapping["pearson_prefix"] = candidate
                break
        if "pearson_prefix" not in mapping:
            raise ValueError("No Pearson prefix column found.")

        for candidate in ["cif_path", "cif_file", "CIF", "structure_file"]:
            if candidate in cols:
                mapping["cif_path"] = candidate
                break

        for candidate in ["material_id", "Material ID", "AWA-id", "mp_id", "id", "substance_id"]:
            if candidate in cols:
                mapping["material_id"] = candidate
                break

        return mapping

    def search(
        self,
        space_group: Optional[int] = None,
        pearson_prefix: Optional[str] = None,
        descriptor_vector: Optional[np.ndarray] = None,
        target_formula: Optional[str] = None,
        top_n: int = 20,
        check_substitution: bool = False,
        substitution_engine: Optional[SubstitutionEngine] = None,
    ) -> pd.DataFrame:
        """
        Search for templates matching symmetry criteria, ranked by descriptor similarity.

        Returns
        -------
        pd.DataFrame
            Matching templates with columns: original metadata + ``pd_distance`` +
            optionally ``substitution_feasible``.
        """
        mask = pd.Series(True, index=self._df.index)

        if space_group is not None:
            mask &= self._df[self._sg_col].astype(int) == int(space_group)

        if pearson_prefix is not None:
            mask &= self._df[self._ps_col].astype(str).str.strip() == str(pearson_prefix).strip()

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
                        ok, z, issues = substitution_engine.check_compatibility(
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

    def search_multi_family(
        self,
        compatible_pairs: List[Tuple[int, str, float]],
        descriptor_vector: Optional[np.ndarray] = None,
        target_formula: Optional[str] = None,
        top_n_per_family: int = 5,
        check_substitution: bool = False,
        substitution_engine: Optional[SubstitutionEngine] = None,
    ) -> pd.DataFrame:
        """
        Search across multiple (spg, ps) families from compatibility filtering.

        Parameters
        ----------
        compatible_pairs : list of (space_group, pearson_prefix, score)
            From ``filter_compatible_pairs()``.

        Returns
        -------
        pd.DataFrame
            Combined results from all families, with ``family_score`` column.
        """
        all_results = []

        for sg, ps, family_score in compatible_pairs:
            candidates = self.search(
                space_group=sg,
                pearson_prefix=ps,
                descriptor_vector=descriptor_vector,
                target_formula=target_formula,
                top_n=top_n_per_family,
                check_substitution=check_substitution,
                substitution_engine=substitution_engine,
            )
            if len(candidates) > 0:
                candidates["predicted_sg"] = sg
                candidates["predicted_ps"] = ps
                candidates["family_score"] = family_score
                all_results.append(candidates)

        if not all_results:
            return pd.DataFrame()

        combined = pd.concat(all_results, ignore_index=True)

        if "pd_distance" in combined.columns:
            combined["combined_score"] = (
                combined["family_score"] * (1 - combined["pd_distance"].fillna(1.0))
            )
            combined = combined.sort_values("combined_score", ascending=False)
        else:
            combined = combined.sort_values("family_score", ascending=False)

        return combined.reset_index(drop=True)

    def _resolve_cif(self, row) -> Optional[Path]:
        """Resolve CIF path for a metadata row."""
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
