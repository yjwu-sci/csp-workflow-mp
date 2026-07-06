"""Shared helpers for fair structure comparison.

Functions:
  _order_by_majority(struct)         : disordered → ordered via majority-species replacement
  sym_order_sm_match(pred, ref)      : sym-order SM (both sides ordered before SM)
  soap_cosine_similarity(pred, ref)  : SOAP descriptor cosine similarity (handles partial occupancy natively)

SOAP parameters fixed to match the AWA pipeline (r_cut=6, n_max=8, l_max=6, average='inner')
so cross-comparison with the existing AWA SOAP code is meaningful.
"""
from __future__ import annotations

import numpy as np
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor

# SOAP hyperparameters (fixed)
SOAP_RCUT = 6.0
SOAP_NMAX = 8
SOAP_LMAX = 6


def _order_by_majority(struct: Structure) -> Structure:
    """Replace each disordered site with its highest-occupancy species.
    Returns a primitive-sized ordered cell (no supercell expansion).
    """
    new_species = []
    for site in struct:
        if site.is_ordered:
            new_species.append(site.specie)
        else:
            major = max(site.species.items(), key=lambda kv: kv[1])[0]
            new_species.append(major)
    return Structure(struct.lattice, new_species, struct.frac_coords,
                     coords_are_cartesian=False)


def sym_order_sm_match(pred: Structure, ref: Structure,
                       ltol: float = 0.2, stol: float = 0.3,
                       angle_tol: float = 5.0) -> tuple[bool, float | None]:
    """Sym-order SM match: apply majority-species ordering to BOTH structures
    before running pymatgen StructureMatcher.

    Returns (is_match: bool, rmsd: float | None).
    rmsd is the StructureMatcher's RMSD if computable, else None.
    """
    try:
        pred_o = _order_by_majority(pred)
        ref_o  = _order_by_majority(ref)
    except Exception:
        return False, None

    matcher = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol,
                                attempt_supercell=True)
    try:
        is_match = matcher.fit(pred_o, ref_o)
        rms = None
        try:
            r = matcher.get_rms_dist(pred_o, ref_o)
            if r is not None:
                rms = float(r[0])
        except Exception:
            pass
        return bool(is_match), rms
    except Exception:
        return False, None


def soap_cosine_similarity(pred: Structure, ref: Structure,
                           r_cut: float = SOAP_RCUT,
                           n_max: int = SOAP_NMAX,
                           l_max: int = SOAP_LMAX) -> float | None:
    """SOAP descriptor cosine similarity between two structures.
    Average='inner' (average over atomic environments before normalization).

    Returns cosine similarity in [-1, 1] (typically [0, 1] for similar structures),
    or None if computation fails.

    Handles partial occupancy via pymatgen → ase round-trip (which silently
    keeps partial occupancy info if SOAP is configured for it; we use the
    majority-species ordering before ase conversion to avoid ase failures).
    """
    try:
        from dscribe.descriptors import SOAP
    except ImportError:
        return None

    try:
        # Order both for ASE compatibility (ASE Atoms doesn't accept partial occ)
        p_o = _order_by_majority(pred)
        r_o = _order_by_majority(ref)
        adaptor = AseAtomsAdaptor()
        atoms_p = adaptor.get_atoms(p_o)
        atoms_r = adaptor.get_atoms(r_o)

        all_species = sorted(set(atoms_p.get_chemical_symbols()) |
                             set(atoms_r.get_chemical_symbols()))
        if not all_species:
            return None

        soap = SOAP(species=all_species, r_cut=r_cut, n_max=n_max, l_max=l_max,
                    periodic=True, average="inner")

        d_p = soap.create(atoms_p).flatten()
        d_r = soap.create(atoms_r).flatten()

        norm = np.linalg.norm(d_p) * np.linalg.norm(d_r)
        if norm < 1e-12:
            return None
        return float(np.dot(d_p, d_r) / norm)
    except Exception:
        return None
