"""
Chemical role grouping for crystallographic site substitution.

Elements are grouped by their typical crystallographic behavior — i.e., which
elements tend to occupy similar Wyckoff sites in inorganic structures. This
grouping drives the substitution engine: elements within the same role group
are considered substitutable, while elements in different groups are not.

The default grouping is based on standard periodic-table categories refined
for inorganic crystallography. Users can override it with a custom mapping
loaded from CSV/JSON.
"""

from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Set

# ============================================================================
# Default role grouping — 9 roles based on standard periodic table categories
# ============================================================================
# Design rationale (from PeriodicTable_Category_legend.xlsx):
#   - Groups reflect standard chemical classification used in the PD scheme
#   - Alkali metals / alkaline earth metals: separate (different charge)
#   - Transition metals: one broad group (Sc-Zn, Y-Cd, Hf-Hg, etc.)
#   - Lanthanides: La-Lu (typically +3, large cation sites)
#   - Actinides: Ac-Lr
#   - Metalloids: B, Si, Ge, As, Sb, Te (network-formers / semi-metals)
#   - Post-transition metals: Al, Ga, In, Sn, Tl, Pb, Bi, Po, At
#   - Reactive non-metals: H, C, N, O, F, P, S, Cl, Se, Br, I
#     (anions + light non-metals — broad group for substitution flexibility)
#   - Noble gases: He, Ne, Ar, Kr, Xe, Rn

DEFAULT_ROLE_MAP: Dict[str, str] = {}

_DEFAULT_GROUPS: Dict[str, List[str]] = {
    "alkali_metals": ["Li", "Na", "K", "Rb", "Cs", "Fr"],
    "alkaline_earth_metals": ["Be", "Mg", "Ca", "Sr", "Ba", "Ra"],
    "transition_metals": [
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
        "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
        "Rf", "Db", "Sg", "Bh", "Hs",
    ],
    "lanthanides": [
        "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
        "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    ],
    "actinides": [
        "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm",
        "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
    ],
    "metalloids": ["B", "Si", "Ge", "As", "Sb", "Te"],
    "post_transition_metals": ["Al", "Ga", "In", "Sn", "Tl", "Pb", "Bi", "Po", "At"],
    "reactive_non_metals": ["H", "C", "N", "O", "F", "P", "S", "Cl", "Se", "Br", "I"],
    "noble_gases": ["He", "Ne", "Ar", "Kr", "Xe", "Rn"],
}

# Build the reverse map: element → role
for _role, _elements in _DEFAULT_GROUPS.items():
    for _elem in _elements:
        DEFAULT_ROLE_MAP[_elem] = _role


class RoleGrouping:
    """
    Maps chemical elements to substitution-compatible role groups.

    Parameters
    ----------
    custom_map : dict, optional
        A {element_symbol: role_name} dictionary. If provided, it completely
        overrides the default mapping for the listed elements.
    custom_file : str or Path, optional
        Path to a CSV or JSON file defining the mapping.
        - CSV: two columns, ``element`` and ``role`` (header required).
        - JSON: ``{"element": "role", ...}`` or ``{"role": ["elem", ...], ...}``.

    Examples
    --------
    >>> rg = RoleGrouping()
    >>> rg.get_role("Li")
    'alkali'
    >>> rg.get_role("Yb")
    'rare_earth'

    >>> rg = RoleGrouping(custom_map={"H": "alkali"})  # override H
    >>> rg.get_role("H")
    'alkali'
    """

    def __init__(
        self,
        custom_map: Optional[Dict[str, str]] = None,
        custom_file: Optional[str | Path] = None,
        merge_roles: Optional[Dict[str, List[str]]] = None,
    ):
        """
        Parameters
        ----------
        custom_map : dict, optional
            Direct {element: role} overrides.
        custom_file : str or Path, optional
            CSV or JSON file with role definitions.
        merge_roles : dict, optional
            Merge multiple roles into one. Example::

                {"anion": ["chalcogen", "halogen"]}

            This reassigns all elements in ``chalcogen`` and ``halogen``
            to the new role ``anion``. Useful when template sites mix
            chalcogens and halogens on the same Wyckoff positions.
        """
        # Start from defaults
        self._map: Dict[str, str] = dict(DEFAULT_ROLE_MAP)

        # Load from file if provided
        if custom_file is not None:
            file_map = self._load_file(Path(custom_file))
            self._map.update(file_map)

        # Direct overrides take highest priority
        if custom_map is not None:
            self._map.update(custom_map)

        # Apply role merging
        if merge_roles:
            for new_role, old_roles in merge_roles.items():
                for elem, role in list(self._map.items()):
                    if role in old_roles:
                        self._map[elem] = new_role

        # Build reverse index: role → set of elements
        self._groups: Dict[str, Set[str]] = {}
        for elem, role in self._map.items():
            self._groups.setdefault(role, set()).add(elem)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_role(self, element: str) -> Optional[str]:
        """Return the role name for an element, or None if unmapped."""
        return self._map.get(element)

    def get_elements(self, role: str) -> Set[str]:
        """Return all elements assigned to a given role."""
        return self._groups.get(role, set())

    def all_roles(self) -> List[str]:
        """Return sorted list of all role names."""
        return sorted(self._groups.keys())

    def are_compatible(self, elem_a: str, elem_b: str) -> bool:
        """Check if two elements belong to the same role group."""
        role_a = self._map.get(elem_a)
        role_b = self._map.get(elem_b)
        if role_a is None or role_b is None:
            return False
        return role_a == role_b

    def group_elements(self, elements: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """
        Group a {element: count} dict by role.

        Parameters
        ----------
        elements : dict
            Mapping of element symbol to stoichiometric count.

        Returns
        -------
        dict
            ``{role: {element: count, ...}, ...}``
            Elements without a role are placed under ``"_unmapped"``.
        """
        grouped: Dict[str, Dict[str, float]] = {}
        for elem, count in elements.items():
            role = self._map.get(elem, "_unmapped")
            grouped.setdefault(role, {})[elem] = count
        return grouped

    def to_dict(self) -> Dict[str, str]:
        """Return a copy of the full element→role mapping."""
        return dict(self._map)

    def save(self, path: str | Path, fmt: str = "csv") -> None:
        """Save the current mapping to CSV or JSON."""
        path = Path(path)
        if fmt == "csv":
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["element", "role"])
                for elem in sorted(self._map):
                    writer.writerow([elem, self._map[elem]])
        elif fmt == "json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._map, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {fmt!r}. Use 'csv' or 'json'.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_file(path: Path) -> Dict[str, str]:
        """Load a role mapping from CSV or JSON."""
        suffix = path.suffix.lower()

        if suffix == ".json":
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            # Support two JSON layouts:
            # (a) {"element": "role", ...}  — flat
            # (b) {"role": ["elem1", "elem2", ...], ...}  — grouped
            if isinstance(data, dict):
                first_val = next(iter(data.values()), None)
                if isinstance(first_val, list):
                    # layout (b): role → [elements]
                    result = {}
                    for role, elems in data.items():
                        for e in elems:
                            result[str(e)] = str(role)
                    return result
                else:
                    # layout (a): element → role
                    return {str(k): str(v) for k, v in data.items()}
            raise ValueError("JSON must be a dict of element→role or role→[elements].")

        if suffix == ".csv":
            result = {}
            with open(path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    elem = row.get("element", "").strip()
                    role = row.get("role", "").strip()
                    if elem and role:
                        result[elem] = role
            return result

        raise ValueError(f"Unsupported file extension: {suffix}. Use .csv or .json.")

    def __repr__(self) -> str:
        n_elem = len(self._map)
        n_role = len(self._groups)
        return f"RoleGrouping({n_elem} elements, {n_role} roles)"
