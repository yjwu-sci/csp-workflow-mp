"""
Automated stoichiometry-aware substitution engine for crystal structure generation.

Given a target chemical formula and a template crystal structure, this module
determines whether a valid element substitution exists and, if so, generates
the substituted structure(s).

Algorithm overview
------------------
1. Parse the template structure into **site groups** — sets of Wyckoff sites
   that share the same chemical role (e.g., all alkali-metal sites).
2. Parse the target formula into **element groups** by the same role mapping.
3. Check **capacity compatibility**: for each role, the total site multiplicity
   in the template must equal the target element count × Z (formula units).
4. Enumerate valid **element → site assignments** within each role group:
   - One-to-one: single target element fills all sites in the role group.
   - Multi-element: distribute multiple target elements across sites,
     preferring integer-occupancy assignments over fractional mixing.
5. Build the substituted structure and return it as a pymatgen Structure.

The engine is database-agnostic: it accepts any pymatgen Structure as template.
"""

from __future__ import annotations

import math
import itertools
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set

import numpy as np

from csp_workflow_mp.role_grouping import RoleGrouping, DEFAULT_ROLE_MAP

logger = logging.getLogger(__name__)


# ============================================================================
# Mergeable role pairs for relaxed matching
# ============================================================================
# These role pairs can share crystallographic sites based on similar
# ionic radii and coordination preferences.
# Order: (role_a, role_b) — symmetric, both directions allowed.

DEFAULT_MERGEABLE_ROLES = [
    # Large cations: commonly share A-site in perovskites, fluorite, etc.
    ("alkali_metals", "alkaline_earth_metals"),    # Li/Na/K ↔ Ca/Sr/Ba
    ("alkaline_earth_metals", "lanthanides"),       # Ca/Sr/Ba ↔ La/Nd/Gd
    # Small cations: commonly share B-site
    ("transition_metals", "post_transition_metals"),  # Fe/Co ↔ Al/Ga/In
]


# ============================================================================
# Data classes
# ============================================================================

@dataclass
class SiteInfo:
    """Metadata for one crystallographic site in the template.

    effective_capacity = multiplicity × total_occupancy, which equals the
    actual atom count contributed by this site to the unit cell.
    For full-occupancy sites this equals multiplicity.
    For partial-occupancy sites (e.g., argyrodite Li 48h with occ=0.5),
    effective_capacity = 48 × 0.5 = 24.
    """
    index: int                          # site index in the Structure
    element: str                        # majority element symbol
    species: Dict[str, float]           # {element: occupancy}
    multiplicity: int                   # Wyckoff multiplicity (physical)
    wyckoff: str                        # Wyckoff letter (if available)
    frac_coords: np.ndarray             # fractional coordinates
    role: Optional[str] = None          # assigned chemical role
    effective_capacity: float = 0.0     # multiplicity × Σ(occupancies)

    def __post_init__(self):
        total_occ = sum(self.species.values()) if self.species else 1.0
        self.effective_capacity = self.multiplicity * total_occ


@dataclass
class SiteGroup:
    """A group of sites that share the same chemical role in the template."""
    role: str
    sites: List[SiteInfo]
    total_capacity: float = 0.0         # sum of effective_capacity (may be fractional)

    def __post_init__(self):
        self.total_capacity = sum(s.effective_capacity for s in self.sites)


@dataclass
class SubstitutionResult:
    """Result of a substitution attempt."""
    success: bool
    target_formula: str
    template_formula: str
    z_factor: Optional[int] = None
    mapping: Optional[Dict[str, Dict[str, float]]] = None  # site_label → {elem: occ}
    substitution_dict: Optional[Dict[str, str]] = None      # simple elem→elem (one-to-one)
    site_assignments: Optional[List[Dict]] = None            # detailed per-site info
    score: float = 0.0                                       # quality score
    method: str = ""                                         # "one_to_one" | "multi_element" | "mixed_occupancy"
    issues: List[str] = field(default_factory=list)


# ============================================================================
# Formula parser
# ============================================================================

def parse_formula(formula: str) -> Dict[str, float]:
    """
    Parse a chemical formula string into {element: count}.

    Delegates to pymatgen.core.Composition to correctly handle
    parenthesised groups, decimal stoichiometry, and hydrate-style
    formulas. Examples::

        "Li3PO4"                → {"Li": 3.0, "P": 1.0, "O": 4.0}
        "MgP2(H8O5)2"           → {"Mg": 1.0, "P": 2.0, "H": 16.0, "O": 10.0}
        "Li6.5La3Zr1.5Ta0.5O12" → {"Li": 6.5, "La": 3.0, ...}

    Historical note: an earlier regex-based implementation silently
    dropped parenthesised outer multipliers, which caused
    misparsed target compositions to trigger the substitution engine's
    fallback path and produce misleading "fake successes" in the
    benchmark. See commit history for details.

    Parameters
    ----------
    formula : str

    Returns
    -------
    dict of {str: float}
    """
    from pymatgen.core import Composition
    return {str(elem): float(count) for elem, count in Composition(formula).as_dict().items()}


# ============================================================================
# Template analyzer
# ============================================================================

def analyze_template(
    structure,
    role_grouping: RoleGrouping,
) -> Tuple[List[SiteInfo], Dict[str, SiteGroup]]:
    """
    Analyze a pymatgen Structure to extract site information and group by role.

    Parameters
    ----------
    structure : pymatgen.core.Structure
        Template crystal structure.
    role_grouping : RoleGrouping
        Element-to-role mapping.

    Returns
    -------
    sites : list of SiteInfo
    site_groups : dict of {role: SiteGroup}
    """
    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        sga = SpacegroupAnalyzer(structure, symprec=0.1, angle_tolerance=5)
        sym_dataset = sga.get_symmetry_dataset()
        wyckoff_letters = sym_dataset.get("wyckoffs", [])
        equivalent_atoms = sym_dataset.get("equivalent_atoms", [])
    except Exception:
        wyckoff_letters = ["?" for _ in structure]
        equivalent_atoms = list(range(len(structure)))

    # Identify unique sites by equivalent_atoms
    unique_site_map: Dict[int, List[int]] = {}
    for i, eq in enumerate(equivalent_atoms):
        unique_site_map.setdefault(eq, []).append(i)

    sites: List[SiteInfo] = []
    for representative, equivalent_indices in unique_site_map.items():
        site = structure[representative]
        multiplicity = len(equivalent_indices)

        # Extract species composition
        species_dict = {}
        for sp, occ in site.species.items():
            species_dict[sp.symbol] = float(occ)

        # Majority element
        majority_elem = max(species_dict, key=species_dict.get)
        role = role_grouping.get_role(majority_elem)

        wyckoff = wyckoff_letters[representative] if representative < len(wyckoff_letters) else "?"

        si = SiteInfo(
            index=representative,
            element=majority_elem,
            species=species_dict,
            multiplicity=multiplicity,
            wyckoff=wyckoff,
            frac_coords=np.array(site.frac_coords),
            role=role,
        )
        sites.append(si)

    # Group by role
    groups: Dict[str, SiteGroup] = {}
    for si in sites:
        role = si.role or "_unmapped"
        if role not in groups:
            groups[role] = SiteGroup(role=role, sites=[])
        groups[role].sites.append(si)

    # Recompute capacities (use effective_capacity to account for partial occupancy)
    for g in groups.values():
        g.total_capacity = sum(s.effective_capacity for s in g.sites)

    return sites, groups


# ============================================================================
# Feasibility checker
# ============================================================================

def check_feasibility(
    target_elements: Dict[str, float],
    site_groups: Dict[str, SiteGroup],
    role_grouping: RoleGrouping,
) -> Tuple[bool, Optional[int], List[str]]:
    """
    Check if a target formula can be mapped onto template site groups.

    Returns
    -------
    feasible : bool
    z_factor : int or None
        Number of formula units per unit cell, if feasible.
    issues : list of str
        Explanation of any problems.
    """
    issues = []

    # Group target elements by role
    target_grouped = role_grouping.group_elements(target_elements)

    # Check for unmapped elements
    if "_unmapped" in target_grouped:
        unmapped = list(target_grouped["_unmapped"].keys())
        issues.append(f"Unmapped elements in target: {unmapped}")
        return False, None, issues

    # Check role coverage
    target_roles = set(target_grouped.keys())
    template_roles = set(site_groups.keys()) - {"_unmapped"}

    missing_roles = target_roles - template_roles
    if missing_roles:
        issues.append(f"Target requires roles not in template: {missing_roles}")
        return False, None, issues

    extra_roles = template_roles - target_roles
    if extra_roles:
        issues.append(f"Template has roles with no target elements: {extra_roles}")
        return False, None, issues

    # Compute Z factor for each role
    z_values = []
    for role in target_roles:
        target_count = sum(target_grouped[role].values())
        template_capacity = site_groups[role].total_capacity

        if target_count <= 0:
            issues.append(f"Role '{role}': target count is zero")
            return False, None, issues

        z = template_capacity / target_count
        z_values.append((role, z, target_count, template_capacity))

    # All Z values must be equal and a positive integer (or close to it)
    z_ref = z_values[0][1]
    for role, z, tc, cap in z_values:
        if abs(z - z_ref) > 1e-6:
            issues.append(
                f"Z mismatch: role '{role}' gives Z={z:.4f} "
                f"(capacity={cap}, target_count={tc}), "
                f"but reference Z={z_ref:.4f}"
            )
            return False, None, issues

    # Check Z is a positive integer
    z_int = round(z_ref)
    if abs(z_ref - z_int) > 0.01 or z_int < 1:
        issues.append(f"Z={z_ref:.4f} is not a positive integer")
        return False, None, issues

    return True, z_int, issues


def check_feasibility_relaxed(
    target_elements: Dict[str, float],
    site_groups: Dict[str, SiteGroup],
    role_grouping: RoleGrouping,
    mergeable_roles: Optional[List[Tuple[str, str]]] = None,
    z_tolerance: float = 0.20,
) -> Tuple[bool, Optional[float], List[str], Optional[Dict[str, str]]]:
    """
    Relaxed feasibility check with role merging and Z-factor tolerance.

    Tries in order:
    1. Strict match (integer Z, exact roles)
    2. Role merging (combine compatible role pairs)
    3. Z-factor tolerance (allow ±tolerance from nearest integer)
    4. Role merging + Z-factor tolerance

    Parameters
    ----------
    target_elements : dict
    site_groups : dict
    role_grouping : RoleGrouping
    mergeable_roles : list of (role_a, role_b), optional
    z_tolerance : float
        Maximum relative deviation from nearest integer Z (default 0.20 = 20%).

    Returns
    -------
    feasible : bool
    z_factor : float or None
        May be non-integer if tolerance was used.
    issues : list of str
    role_merge_map : dict or None
        If merging was used, maps original roles to merged role names.
        e.g., {"alkali_metals": "cation_A", "alkaline_earth_metals": "cation_A"}
    """
    if mergeable_roles is None:
        mergeable_roles = DEFAULT_MERGEABLE_ROLES

    # Step 1: Try strict (current logic)
    ok, z, issues = check_feasibility(target_elements, site_groups, role_grouping)
    if ok:
        return True, z, [], None

    # Step 2: Try with role merging only (still require integer Z)
    for merge_map in _enumerate_role_merges(target_elements, site_groups, role_grouping, mergeable_roles):
        merged_target, merged_groups = _apply_role_merge(
            target_elements, site_groups, role_grouping, merge_map
        )
        ok, z, iss = _check_feasibility_core(merged_target, merged_groups, strict_z=True)
        if ok:
            return True, z, [f"Used role merging: {merge_map}"], merge_map

    # Step 3: Try Z-factor tolerance only (no merging)
    target_grouped = role_grouping.group_elements(target_elements)
    ok, z, iss = _check_feasibility_core(target_grouped, site_groups, strict_z=False, z_tol=z_tolerance)
    if ok:
        return True, z, [f"Used Z-factor tolerance (Z={z:.3f})"], None

    # Step 4: Try both role merging + Z tolerance
    for merge_map in _enumerate_role_merges(target_elements, site_groups, role_grouping, mergeable_roles):
        merged_target, merged_groups = _apply_role_merge(
            target_elements, site_groups, role_grouping, merge_map
        )
        ok, z, iss = _check_feasibility_core(merged_target, merged_groups, strict_z=False, z_tol=z_tolerance)
        if ok:
            return True, z, [f"Used role merging + Z tolerance: {merge_map}, Z={z:.3f}"], merge_map

    return False, None, issues + ["Relaxed matching also failed"], None


def _check_feasibility_core(
    target_grouped: Dict[str, Dict[str, float]],
    site_groups: Dict[str, SiteGroup],
    strict_z: bool = True,
    z_tol: float = 0.20,
) -> Tuple[bool, Optional[float], List[str]]:
    """
    Core feasibility check on already-grouped target and site groups.
    """
    issues = []

    if "_unmapped" in target_grouped:
        return False, None, ["Unmapped elements"]

    target_roles = set(target_grouped.keys())
    template_roles = set(site_groups.keys()) - {"_unmapped"}

    missing = target_roles - template_roles
    extra = template_roles - target_roles
    if missing or extra:
        return False, None, [f"Role mismatch: missing={missing}, extra={extra}"]

    # Compute Z for each role
    z_values = []
    for role in target_roles:
        tc = sum(target_grouped[role].values())
        cap = site_groups[role].total_capacity
        if tc <= 0:
            return False, None, [f"Role '{role}': zero target count"]
        z = cap / tc
        z_values.append((role, z, tc, cap))

    if not z_values:
        return False, None, ["No roles"]

    # Check Z consistency across roles
    z_ref = z_values[0][1]
    for role, z, tc, cap in z_values:
        if abs(z - z_ref) / max(z_ref, 1e-9) > z_tol:
            return False, None, [f"Z mismatch: {role} Z={z:.3f} vs ref={z_ref:.3f}"]

    # Check Z is close to an integer
    z_avg = sum(z for _, z, _, _ in z_values) / len(z_values)
    z_nearest_int = round(z_avg)

    if z_nearest_int < 1:
        return False, None, [f"Z={z_avg:.3f} too small"]

    if strict_z:
        if abs(z_avg - z_nearest_int) > 0.01:
            return False, None, [f"Z={z_avg:.4f} not integer (strict)"]
        return True, z_nearest_int, []
    else:
        rel_err = abs(z_avg - z_nearest_int) / z_nearest_int
        if rel_err > z_tol:
            return False, None, [f"Z={z_avg:.3f} too far from integer (err={rel_err:.1%})"]
        return True, z_avg, []


def _enumerate_role_merges(
    target_elements: Dict[str, float],
    site_groups: Dict[str, SiteGroup],
    role_grouping: RoleGrouping,
    mergeable_roles: List[Tuple[str, str]],
) -> List[Dict[str, str]]:
    """
    Generate possible role merge maps that could resolve role mismatches.

    Only generates merges that are relevant (i.e., one role is in target but
    not in template, or vice versa, and a mergeable partner exists).
    """
    target_grouped = role_grouping.group_elements(target_elements)
    target_roles = set(target_grouped.keys()) - {"_unmapped"}
    template_roles = set(site_groups.keys()) - {"_unmapped"}

    missing = target_roles - template_roles  # target has, template doesn't
    extra = template_roles - target_roles    # template has, target doesn't

    if not missing and not extra:
        return []

    # Find merges that could resolve mismatches
    merge_candidates = []
    for role_a, role_b in mergeable_roles:
        # Can merging role_a and role_b help?
        # Case 1: role_a is missing, role_b is in template → merge target's role_a into role_b
        if role_a in missing and role_b in template_roles:
            merge_candidates.append({role_a: role_b})
        if role_b in missing and role_a in template_roles:
            merge_candidates.append({role_b: role_a})
        # Case 2: role_a is extra in template, role_b is in target → merge template's role_a into role_b
        if role_a in extra and role_b in target_roles:
            merge_candidates.append({role_a: role_b})
        if role_b in extra and role_a in target_roles:
            merge_candidates.append({role_b: role_a})

    # Also try merging both directions when both roles exist but Z doesn't match
    if not missing and not extra:
        for role_a, role_b in mergeable_roles:
            if role_a in target_roles and role_b in target_roles:
                merged_name = f"{role_a}+{role_b}"
                merge_candidates.append({role_a: merged_name, role_b: merged_name})

    # Deduplicate
    unique = []
    seen = set()
    for m in merge_candidates:
        key = tuple(sorted(m.items()))
        if key not in seen:
            seen.add(key)
            unique.append(m)

    return unique


def _apply_role_merge(
    target_elements: Dict[str, float],
    site_groups: Dict[str, SiteGroup],
    role_grouping: RoleGrouping,
    merge_map: Dict[str, str],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, SiteGroup]]:
    """
    Apply a role merge map to both target groups and site groups.

    merge_map: {old_role: new_role} — elements/sites with old_role get reassigned to new_role.
    """
    target_grouped = role_grouping.group_elements(target_elements)

    # Merge target groups
    merged_target: Dict[str, Dict[str, float]] = {}
    for role, elems in target_grouped.items():
        new_role = merge_map.get(role, role)
        if new_role not in merged_target:
            merged_target[new_role] = {}
        for elem, count in elems.items():
            merged_target[new_role][elem] = merged_target[new_role].get(elem, 0) + count

    # Merge site groups
    merged_groups: Dict[str, SiteGroup] = {}
    for role, group in site_groups.items():
        new_role = merge_map.get(role, role)
        if new_role not in merged_groups:
            merged_groups[new_role] = SiteGroup(role=new_role, sites=list(group.sites))
        else:
            merged_groups[new_role].sites.extend(group.sites)
            merged_groups[new_role].total_capacity = sum(
                s.effective_capacity for s in merged_groups[new_role].sites
            )

    return merged_target, merged_groups


# ============================================================================
# Assignment solvers
# ============================================================================

def _solve_one_to_one(
    target_elements: Dict[str, float],
    site_groups: Dict[str, SiteGroup],
    role_grouping: RoleGrouping,
    z_factor: int,
) -> Optional[SubstitutionResult]:
    """
    Solve the simple case: each role group has exactly one target element.
    Produces a direct element → element substitution dictionary.
    """
    target_grouped = role_grouping.group_elements(target_elements)
    substitution_dict = {}
    site_assignments = []

    for role, target_elems in target_grouped.items():
        if len(target_elems) != 1:
            return None  # Not a one-to-one case

        target_elem = list(target_elems.keys())[0]
        group = site_groups.get(role)
        if group is None:
            return None

        # Map all template elements in this role to the target element
        for site in group.sites:
            for template_elem in site.species:
                if template_elem != target_elem:
                    substitution_dict[template_elem] = target_elem

            site_assignments.append({
                "site_index": site.index,
                "wyckoff": site.wyckoff,
                "multiplicity": site.multiplicity,
                "original_element": site.element,
                "assigned_element": target_elem,
                "occupancy": 1.0,
                "role": role,
            })

    return SubstitutionResult(
        success=True,
        target_formula="",
        template_formula="",
        z_factor=z_factor,
        substitution_dict=substitution_dict,
        site_assignments=site_assignments,
        method="one_to_one",
        score=1.0,
    )


def _solve_multi_element(
    target_elements: Dict[str, float],
    site_groups: Dict[str, SiteGroup],
    role_grouping: RoleGrouping,
    z_factor: int,
    max_solutions: int = 10,
) -> List[SubstitutionResult]:
    """
    Solve multi-element assignment: distribute multiple target elements
    across sites within a role group.

    Strategy:
    1. Try integer assignments first (each site gets one element only).
    2. If no exact integer solution, fall back to mixed occupancy.
    """
    target_grouped = role_grouping.group_elements(target_elements)
    all_role_solutions: Dict[str, List[List[Dict]]] = {}

    for role, target_elems in target_grouped.items():
        group = site_groups.get(role)
        if group is None:
            return []

        # Target atom counts in the unit cell
        uc_counts = {elem: count * z_factor for elem, count in target_elems.items()}
        total_target = sum(uc_counts.values())

        if abs(total_target - group.total_capacity) > 0.5:
            return []

        sites = group.sites
        site_mults = [s.multiplicity for s in sites]

        if len(target_elems) == 1:
            # Trivial: one element fills all sites
            elem = list(target_elems.keys())[0]
            assignment = [
                {"site": s, "element": elem, "occupancy": 1.0}
                for s in sites
            ]
            all_role_solutions[role] = [assignment]
            continue

        # Try integer assignment: each site gets exactly one element
        integer_solutions = _enumerate_integer_assignments(
            sites, uc_counts, max_solutions=max_solutions
        )

        if integer_solutions:
            all_role_solutions[role] = integer_solutions
        else:
            # Fall back to mixed occupancy (uniform distribution)
            fractions = {
                elem: count / total_target for elem, count in uc_counts.items()
            }
            assignment = [
                {"site": s, "element": fractions, "occupancy": "mixed"}
                for s in sites
            ]
            all_role_solutions[role] = [assignment]

    # Combine solutions across role groups (cartesian product, limited)
    role_names = list(all_role_solutions.keys())
    role_solution_lists = [all_role_solutions[r] for r in role_names]

    results = []
    for combo in itertools.islice(
        itertools.product(*role_solution_lists), max_solutions
    ):
        site_assignments = []
        mapping = {}

        for role_name, role_assignment in zip(role_names, combo):
            for item in role_assignment:
                site = item["site"]
                if item["occupancy"] == "mixed":
                    # Mixed occupancy
                    fracs = item["element"]
                    mapping[f"site_{site.index}"] = fracs
                    for elem, frac in fracs.items():
                        site_assignments.append({
                            "site_index": site.index,
                            "wyckoff": site.wyckoff,
                            "multiplicity": site.multiplicity,
                            "original_element": site.element,
                            "assigned_element": elem,
                            "occupancy": frac,
                            "role": role_name,
                        })
                else:
                    elem = item["element"]
                    mapping[f"site_{site.index}"] = {elem: 1.0}
                    site_assignments.append({
                        "site_index": site.index,
                        "wyckoff": site.wyckoff,
                        "multiplicity": site.multiplicity,
                        "original_element": site.element,
                        "assigned_element": elem,
                        "occupancy": 1.0,
                        "role": role_name,
                    })

        results.append(SubstitutionResult(
            success=True,
            target_formula="",
            template_formula="",
            z_factor=z_factor,
            mapping=mapping,
            site_assignments=site_assignments,
            method="multi_element" if any(
                item.get("occupancy") != "mixed"
                for role_assign in combo
                for item in role_assign
            ) else "mixed_occupancy",
        ))

    return results


def _enumerate_integer_assignments(
    sites: List[SiteInfo],
    uc_counts: Dict[str, float],
    max_solutions: int = 10,
) -> List[List[Dict]]:
    """
    Enumerate integer-occupancy assignments of elements to sites.

    Each site gets exactly one element. The total count for each element
    across assigned sites must match its unit-cell count.

    Uses depth-first search with pruning for efficiency.
    """
    elements = list(uc_counts.keys())

    # ── FIX: detect fractional counts that cannot be integer-assigned ──
    # If any element's unit-cell count is significantly non-integer
    # (e.g., Fe:1.8, Mn:0.2 from target Fe0.9Mn0.1 × Z=2),
    # integer assignment is impossible — return [] so the caller
    # falls back to the mixed-occupancy path, which preserves
    # partial occupancy correctly.
    for elem, count in uc_counts.items():
        if abs(count - round(count)) > 0.01:
            logger.debug(
                "Fractional unit-cell count %s=%.3f — "
                "integer assignment not possible, deferring to mixed occupancy.",
                elem, count,
            )
            return []

    needed = {elem: int(round(count)) for elem, count in uc_counts.items()}

    # Drop elements with zero count after rounding (safety net)
    needed = {e: n for e, n in needed.items() if n > 0}
    elements = [e for e in elements if e in needed]

    # Use effective_capacity (rounded to int) to account for partial occupancy.
    # For full-occupancy sites, effective_capacity == multiplicity.
    # For partial-occupancy (e.g., 48h × 0.5), effective_capacity == 24.
    site_mults = [int(round(s.effective_capacity)) for s in sites]
    total_needed = sum(needed.values())
    total_available = sum(site_mults)
    if total_needed != total_available:
        return []

    solutions = []

    def dfs(site_idx: int, remaining: Dict[str, int], current: List[Dict]):
        if site_idx == len(sites):
            if all(v == 0 for v in remaining.values()):
                solutions.append(list(current))
            return

        if len(solutions) >= max_solutions:
            return

        site = sites[site_idx]
        mult = site_mults[site_idx]

        for elem in elements:
            if remaining[elem] >= mult:
                remaining[elem] -= mult
                current.append({"site": site, "element": elem, "occupancy": 1.0})
                dfs(site_idx + 1, remaining, current)
                current.pop()
                remaining[elem] += mult

    # Limit search for large problems
    if len(sites) > 12 or len(elements) > 6:
        logger.warning(
            "Large assignment problem (%d sites, %d elements): "
            "using heuristic instead of full enumeration.",
            len(sites), len(elements),
        )
        return _greedy_integer_assignment(sites, needed)

    dfs(0, dict(needed), [])
    return solutions


def _greedy_integer_assignment(
    sites: List[SiteInfo],
    needed: Dict[str, int],
) -> List[List[Dict]]:
    """
    Greedy heuristic for integer assignment when exact enumeration is too expensive.

    Assigns elements to sites in order of decreasing multiplicity,
    filling the element with the largest remaining need first.
    """
    remaining = dict(needed)
    assignment = []

    # Sort sites by decreasing effective_capacity for better packing
    sorted_sites = sorted(
        sites,
        key=lambda s: int(round(s.effective_capacity)),
        reverse=True,
    )

    for site in sorted_sites:
        mult = int(round(site.effective_capacity))
        # Pick the element with the largest remaining need that can fill this site
        best_elem = None
        best_remaining = -1
        for elem, rem in remaining.items():
            if rem >= mult and rem > best_remaining:
                best_elem = elem
                best_remaining = rem

        if best_elem is None:
            # Try any element that has remaining need
            for elem, rem in remaining.items():
                if rem > 0:
                    best_elem = elem
                    break

        if best_elem is None:
            return []

        remaining[best_elem] -= mult
        assignment.append({"site": site, "element": best_elem, "occupancy": 1.0})

    if all(v == 0 for v in remaining.values()):
        return [assignment]
    return []


# ============================================================================
# Main engine class
# ============================================================================

class SubstitutionEngine:
    """
    Automated element substitution engine for crystal structure generation.

    Parameters
    ----------
    role_grouping : RoleGrouping, optional
        Custom role grouping. Uses default if not provided.
    max_solutions : int
        Maximum number of substitution solutions to enumerate per template.

    Examples
    --------
    >>> from pymatgen.core import Structure
    >>> engine = SubstitutionEngine()
    >>> results = engine.find_substitutions(
    ...     target_formula="B5O12Yb3",
    ...     template=Structure.from_file("B5O12Lu3.cif"),
    ... )
    >>> results[0].success
    True
    >>> results[0].substitution_dict
    {'Lu': 'Yb'}
    """

    def __init__(
        self,
        role_grouping: Optional[RoleGrouping] = None,
        max_solutions: int = 10,
        mergeable_roles: Optional[List[Tuple[str, str]]] = None,
        z_tolerance: float = 0.20,
        use_relaxed_matching: bool = True,
    ):
        self.role_grouping = role_grouping or RoleGrouping()
        self.max_solutions = max_solutions
        self.mergeable_roles = mergeable_roles if mergeable_roles is not None else DEFAULT_MERGEABLE_ROLES
        self.z_tolerance = z_tolerance
        self.use_relaxed_matching = use_relaxed_matching

    def check_compatibility(
        self,
        target_formula: str,
        template,
    ) -> Tuple[bool, Optional[int], List[str]]:
        """
        Quick check: can this target formula be mapped onto this template?
        Tries strict matching first, then relaxed if enabled.
        """
        target_elements = parse_formula(target_formula)
        _, site_groups = analyze_template(template, self.role_grouping)

        # Try strict first
        ok, z, issues = check_feasibility(target_elements, site_groups, self.role_grouping)
        if ok:
            return True, z, issues

        # Try relaxed
        if self.use_relaxed_matching:
            ok, z, issues, merge_map = check_feasibility_relaxed(
                target_elements, site_groups, self.role_grouping,
                self.mergeable_roles, self.z_tolerance,
            )
            if ok:
                return True, round(z) if isinstance(z, float) else z, issues

        return False, None, issues

    def find_substitutions(
        self,
        target_formula: str,
        template,
    ) -> List[SubstitutionResult]:
        """
        Find all valid element substitution mappings from template to target.

        Tries strict matching first, then relaxed (role merging + Z tolerance).
        """
        # Allow passing a file path
        if isinstance(template, (str, type(None))):
            from pymatgen.core import Structure as PmgStructure
            template = PmgStructure.from_file(str(template))

        target_elements = parse_formula(target_formula)
        template_formula = template.composition.reduced_formula

        sites, site_groups = analyze_template(template, self.role_grouping)

        # === Try strict matching first ===
        feasible, z_factor, issues = check_feasibility(
            target_elements, site_groups, self.role_grouping
        )

        if feasible:
            return self._solve_and_return(
                target_elements, site_groups, target_formula,
                template_formula, z_factor, method_prefix=""
            )

        # === Try relaxed matching ===
        if self.use_relaxed_matching:
            ok, z_relax, relax_issues, merge_map = check_feasibility_relaxed(
                target_elements, site_groups, self.role_grouping,
                self.mergeable_roles, self.z_tolerance,
            )

            if ok:
                # Apply role merge to get updated groups
                if merge_map:
                    merged_target, merged_groups = _apply_role_merge(
                        target_elements, site_groups, self.role_grouping, merge_map
                    )
                else:
                    merged_target = self.role_grouping.group_elements(target_elements)
                    merged_groups = site_groups

                # Use nearest integer Z for assignment
                z_int = round(z_relax) if isinstance(z_relax, float) else z_relax
                if z_int < 1:
                    z_int = 1

                return self._solve_and_return(
                    target_elements, merged_groups, target_formula,
                    template_formula, z_int,
                    method_prefix="relaxed_",
                    extra_issues=relax_issues,
                    role_merge_map=merge_map,
                )

        # === All matching failed ===
        return [SubstitutionResult(
            success=False,
            target_formula=target_formula,
            template_formula=template_formula,
            issues=issues,
        )]

    def _solve_and_return(
        self,
        target_elements: Dict[str, float],
        site_groups: Dict[str, SiteGroup],
        target_formula: str,
        template_formula: str,
        z_factor: int,
        method_prefix: str = "",
        extra_issues: Optional[List[str]] = None,
        role_merge_map: Optional[Dict[str, str]] = None,
    ) -> List[SubstitutionResult]:
        """Try one-to-one then multi-element assignment and return results."""

        # Step 1: Try one-to-one
        result_1to1 = _solve_one_to_one(
            target_elements, site_groups, self.role_grouping, z_factor
        )

        if result_1to1 is not None:
            result_1to1.target_formula = target_formula
            result_1to1.template_formula = template_formula
            result_1to1.method = method_prefix + "one_to_one"
            result_1to1.score = 0.9 if method_prefix else 1.0
            if extra_issues:
                result_1to1.issues = extra_issues
            return [result_1to1]

        # Step 2: Multi-element assignment
        results = _solve_multi_element(
            target_elements, site_groups, self.role_grouping, z_factor,
            max_solutions=self.max_solutions,
        )

        for r in results:
            r.target_formula = target_formula
            r.template_formula = template_formula
            if method_prefix:
                r.method = method_prefix + r.method
            if extra_issues:
                r.issues = (r.issues or []) + extra_issues

        if not results:
            # When all assignment solvers fail, we cannot produce a valid
            # substituted structure. Previously this returned success=True with
            # an empty mapping, which caused apply_substitution to return the
            # unchanged template (a silent "fake success"). Return
            # success=False so the caller's retry loop moves on to the next
            # candidate template.
            results = [SubstitutionResult(
                success=False,
                target_formula=target_formula,
                template_formula=template_formula,
                z_factor=z_factor,
                method=method_prefix + "mixed_occupancy",
                score=0.0,
                issues=(extra_issues or []) + ["All assignment solvers failed; no substitution produced."],
            )]

        # Score results
        for r in results:
            r.score = self._score_assignment(r, target_elements)
            # Penalize relaxed matches
            if method_prefix:
                r.score *= 0.8

        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def apply_substitution(
        self,
        template,
        result: SubstitutionResult,
    ):
        """
        Apply a substitution result to a template structure.

        Parameters
        ----------
        template : pymatgen.core.Structure
            The template structure to modify.
        result : SubstitutionResult
            A successful substitution result from ``find_substitutions``.

        Returns
        -------
        pymatgen.core.Structure
            The substituted structure.
        """
        from pymatgen.core import Structure as PmgStructure, Element

        if not result.success:
            raise ValueError("Cannot apply a failed substitution result.")

        structure = template.copy()

        if result.method == "one_to_one" and result.substitution_dict:
            # Simple element replacement
            for old_elem, new_elem in result.substitution_dict.items():
                structure.replace_species({old_elem: new_elem})
            return structure

        # Multi-element or mixed occupancy: need per-site replacement
        if result.site_assignments:
            # Build a map: original site representative index → list of assignments
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
            try:
                sga = SpacegroupAnalyzer(structure, symprec=0.1, angle_tolerance=5)
                sym_data = sga.get_symmetry_dataset()
                equiv_atoms = sym_data.get("equivalent_atoms", list(range(len(structure))))
            except Exception:
                equiv_atoms = list(range(len(structure)))

            # Group assignments by site index (representative)
            assign_map: Dict[int, Dict[str, float]] = {}
            for sa in result.site_assignments:
                idx = sa["site_index"]
                elem = sa["assigned_element"]
                occ = sa["occupancy"]
                if idx not in assign_map:
                    assign_map[idx] = {}
                assign_map[idx][elem] = assign_map[idx].get(elem, 0.0) + occ

            # Apply to all equivalent sites
            for i in range(len(structure)):
                rep = equiv_atoms[i]
                if rep in assign_map:
                    new_species = {
                        Element(e): o for e, o in assign_map[rep].items()
                    }
                    structure[i] = new_species, structure[i].frac_coords

        return structure

    def _score_assignment(
        self,
        result: SubstitutionResult,
        target_elements: Dict[str, float],
    ) -> float:
        """
        Score a substitution result. Higher = better.

        Scoring criteria:
        - Prefer one-to-one over multi-element over mixed occupancy
        - Prefer assignments with similar ionic radii between original and new elements
        """
        if result.method == "one_to_one":
            return 1.0
        elif result.method == "multi_element":
            return 0.8
        else:
            return 0.5

    def batch_screen(
        self,
        target_formula: str,
        templates: List[Tuple[str, object]],
    ) -> List[Tuple[str, SubstitutionResult]]:
        """
        Screen multiple templates for substitution compatibility with a target.

        Parameters
        ----------
        target_formula : str
            Target chemical formula.
        templates : list of (name, Structure)
            Named template structures to screen.

        Returns
        -------
        list of (name, SubstitutionResult)
            Only feasible results, sorted by score.
        """
        results = []
        for name, template in templates:
            subs = self.find_substitutions(target_formula, template)
            if subs and subs[0].success:
                results.append((name, subs[0]))

        results.sort(key=lambda x: x[1].score, reverse=True)
        return results

    def __repr__(self) -> str:
        return f"SubstitutionEngine(roles={self.role_grouping}, max_solutions={self.max_solutions})"
