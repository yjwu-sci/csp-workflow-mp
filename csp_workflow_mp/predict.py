"""High-level prediction API.

Two user-facing entry points:

- :func:`predict_from_formula` — full pipeline. Optional ``known_sg``
  parameter skips the classifier and filters templates to a
  user-specified space group.
- :class:`PredictionResult` — dataclass bundling the substituted and
  (optionally) relaxed structures plus diagnostics.

Special handling of partial occupancy: if substitution succeeds but the
resulting structure carries partial site occupancies (``is_ordered``
False), the CIF is saved unrelaxed and ``status`` is set to
``PARTIAL_OCCUPANCY``. MatterSim cannot relax disordered structures
directly; the pipeline stops at that point for such cases. What to do
next with the disk-side CIF is outside the scope of this repository.

Example — full classifier flow::

    from csp_workflow_mp import predict_from_formula
    result = predict_from_formula("KTaO3", top_k_sg=1)
    print(result.summary())

Example — user-specified space group (skip classifier)::

    result = predict_from_formula("KTaO3", known_sg=221)  # Pm-3m
    print(result.relaxed_structure)

Example — partial-occupancy target (subst succeeds, relax skipped)::

    result = predict_from_formula("Fe0.9Mn0.1O", known_sg=225)
    if result.status == "PARTIAL_OCCUPANCY":
        print("Substituted CIF saved at:", result.substituted_cif_path)
        for w in result.warnings:
            print("  !", w)
"""
from __future__ import annotations

import logging
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FTimeout
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from csp_workflow_mp._paths import (
    METADATA_WITH_DESCRIPTORS_CSV,
    CIF_DIR as DEFAULT_CIF_DIR,
    MODEL_DIR as DEFAULT_MODEL_DIR,
    RESULTS_DIR as DEFAULT_RESULTS_DIR,
)
from csp_workflow_mp.descriptor import compute_periodic_descriptors
from csp_workflow_mp.retriever import TemplatePool
from csp_workflow_mp.substitution_engine import SubstitutionEngine

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Result dataclass
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PredictionResult:
    """Bundle of the substituted structure, the (optionally) relaxed
    structure, and diagnostic info from a single-formula prediction.

    ``status`` is one of:

    - ``SUCCESS`` — substitution succeeded, MatterSim relaxation converged,
      volume change within the |ΔV/V|<15 % window.
    - ``PARTIAL_OCCUPANCY`` — substitution succeeded but the resulting
      structure has fractional site occupancies. The substituted CIF is
      saved; ``relaxed_structure`` is None. See ``warnings`` for details.
    - ``RELAX_FAILED`` — substitution succeeded, structure was ordered,
      but MatterSim either did not converge or the volume change exceeded
      the filter. See ``warnings``.
    - ``SUBSTITUTION_FAILED`` — none of the top-N ranked templates
      produced a feasible substitution.
    - ``NO_CANDIDATE`` — the template pool had no entries in the requested
      space-group mask (rare).
    """
    target_formula: str
    known_sg_used: Optional[int]                       # None if classifier was used
    classifier_top_k_sgs: Optional[List[int]]          # None if known_sg was provided
    template_material_id: Optional[str] = None
    template_rank: Optional[int] = None
    template_formula: Optional[str] = None
    substitution_method: Optional[str] = None
    substituted_structure: Optional[Structure] = None
    is_ordered: bool = False
    substituted_cif_path: Optional[Path] = None
    relaxed_structure: Optional[Structure] = None
    relaxed_cif_path: Optional[Path] = None
    relaxation_converged: bool = False
    volume_change: float = float("nan")
    predicted_space_group: Optional[int] = None
    warnings: List[str] = field(default_factory=list)
    status: str = "UNKNOWN"

    def summary(self) -> str:
        """Human-readable multi-line summary suitable for printing."""
        lines = [
            f"target: {self.target_formula}",
            f"status: {self.status}",
        ]
        if self.known_sg_used is not None:
            lines.append(f"space group: {self.known_sg_used} (user-specified)")
        elif self.classifier_top_k_sgs:
            lines.append(f"space group(s): {self.classifier_top_k_sgs} (classifier top-K)")
        if self.template_material_id:
            lines.append(f"template: {self.template_material_id} ({self.template_formula}) at rank {self.template_rank}")
            lines.append(f"substitution method: {self.substitution_method}")
            lines.append(f"is_ordered: {self.is_ordered}")
        if self.substituted_cif_path:
            lines.append(f"substituted CIF: {self.substituted_cif_path}")
        if self.relaxed_cif_path:
            lines.append(f"relaxed CIF: {self.relaxed_cif_path}")
        if not np.isnan(self.volume_change):
            lines.append(f"|ΔV/V|: {self.volume_change:.3f}")
        if self.predicted_space_group is not None:
            lines.append(f"predicted SG: {self.predicted_space_group}")
        if self.warnings:
            lines.append("warnings:")
            for w in self.warnings:
                lines.append(f"  ! {w}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# Relaxation helper (mirrors 05_run_benchmark.relax_structure)
# ═══════════════════════════════════════════════════════════════════════

_RELAX_FMAX = 0.05
_RELAX_STEPS = 500
_RELAX_TIMEOUT = 300
_VOL_CHANGE_MAX = 0.15


def _relax_with_mattersim(structure: Structure, device: str = "auto"):
    """Return (relaxed_structure, converged, vol_change_frac).
    Returns (None, False, nan) for disordered inputs or on error/timeout.
    """
    if not structure.is_ordered:
        return None, False, float("nan")

    try:
        from mattersim.forcefield import MatterSimCalculator
        try:
            from ase.filters import UnitCellFilter
        except ImportError:
            from ase.constraints import UnitCellFilter
        from ase.optimize import BFGS
    except ImportError as exc:
        logger.warning("MatterSim / ASE not available: %s", exc)
        return None, False, float("nan")

    if device == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        except ImportError:
            device = "cpu"

    calc = MatterSimCalculator(device=device)
    adaptor = AseAtomsAdaptor()
    atoms = adaptor.get_atoms(structure)
    vol0 = atoms.get_volume()
    atoms.calc = calc

    def _do():
        ucf = UnitCellFilter(atoms)
        opt = BFGS(ucf, logfile=None)
        opt.run(fmax=_RELAX_FMAX, steps=_RELAX_STEPS)
        return opt.converged()

    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            converged = ex.submit(_do).result(timeout=_RELAX_TIMEOUT)
    except (_FTimeout, Exception) as exc:
        logger.debug("Relaxation error: %s", exc)
        return None, False, float("nan")

    vol1 = atoms.get_volume()
    dv = abs(vol1 - vol0) / max(vol0, 1e-6)
    return adaptor.get_structure(atoms), converged, float(dv)


def _write_cif(structure: Structure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    structure.to(filename=str(path), fmt="cif")
    return path


def _validate_inputs(formula, known_sg, top_k_sg, n_retry) -> None:
    """Raise ValueError with an actionable message for bad user inputs.

    Called at the top of ``predict_from_formula`` before any expensive
    work. Catches inputs that would otherwise silently produce
    SUBSTITUTION_FAILED (e.g. unrecognised element symbols that pymatgen
    silently converts to DummySpecies) or trigger opaque downstream
    errors (e.g. top_k_sg=0, n_retry=0, known_sg out of range).
    """
    from pymatgen.core import Composition, Element

    if not isinstance(formula, str) or not formula.strip():
        raise ValueError(f"formula must be a non-empty string; got {formula!r}")
    try:
        comp = Composition(formula)
    except Exception as e:
        raise ValueError(f"Invalid formula {formula!r}: {e}") from e
    bad_elements = [
        getattr(sp, "symbol", str(sp))
        for sp in comp.elements
        if not Element.is_valid_symbol(getattr(sp, "symbol", str(sp)))
    ]
    if bad_elements:
        raise ValueError(
            f"formula {formula!r} contains unrecognised element symbol(s): "
            f"{bad_elements}. Use standard 1- or 2-letter element abbreviations "
            "(e.g. 'Fe', 'O', 'Cl')."
        )
    if known_sg is not None and not (1 <= int(known_sg) <= 230):
        raise ValueError(f"known_sg must be an integer in [1, 230]; got {known_sg}")
    if top_k_sg < 1:
        raise ValueError(f"top_k_sg must be >= 1; got {top_k_sg}")
    if n_retry < 1:
        raise ValueError(f"n_retry must be >= 1; got {n_retry}")


# ═══════════════════════════════════════════════════════════════════════
# Main API
# ═══════════════════════════════════════════════════════════════════════

def predict_from_formula(
    formula: str,
    known_sg: Optional[int] = None,
    top_k_sg: int = 1,
    n_retry: int = 50,
    metadata_csv: Optional[Path] = None,
    cif_dir: Optional[Path] = None,
    model_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    do_relax: bool = True,
    device: str = "auto",
    verbose: bool = True,
) -> PredictionResult:
    """Predict a crystal structure from a chemical formula.

    Parameters
    ----------
    formula : str
        Target chemical formula, e.g. ``"KTaO3"``.
    known_sg : int, optional
        User-specified space-group number. If given, the classifier is
        skipped and templates are filtered directly to this SG.
    top_k_sg : int
        Number of top-predicted SGs to use (ignored if ``known_sg`` is
        provided). Default 1.
    n_retry : int
        Number of top-ranked templates to try in rank order until a
        feasible substitution is found. Default 50.
    metadata_csv, cif_dir, model_dir, output_dir : Path, optional
        Override the default data locations from ``_paths.py``.
        ``output_dir`` defaults to ``<RESULTS_DIR>/predictions/<formula>/``.
    do_relax : bool
        If True, run MatterSim relaxation on ordered substituted
        structures. Default True.
    device : str
        MatterSim device: 'auto', 'cuda', 'mps', or 'cpu'.
    verbose : bool
        If True, print progress messages and warnings.

    Returns
    -------
    PredictionResult

    See Also
    --------
    :class:`PredictionResult` — return type, includes ``.summary()``.
    """
    _validate_inputs(formula, known_sg, top_k_sg, n_retry)

    metadata_csv = Path(metadata_csv or METADATA_WITH_DESCRIPTORS_CSV)
    cif_dir = Path(cif_dir or DEFAULT_CIF_DIR)
    model_dir = Path(model_dir or DEFAULT_MODEL_DIR)
    output_dir = Path(output_dir or (DEFAULT_RESULTS_DIR / "predictions" / formula.replace("/", "_")))
    output_dir.mkdir(parents=True, exist_ok=True)

    result = PredictionResult(
        target_formula=formula,
        known_sg_used=known_sg,
        classifier_top_k_sgs=None,
    )

    # ── 1. Descriptor ────────────────────────────────────────────────
    desc = compute_periodic_descriptors(formula)
    if verbose:
        logger.info("[predict] descriptor shape %s", desc.shape)

    # ── 2. Determine SG mask ─────────────────────────────────────────
    if known_sg is not None:
        sg_mask = [int(known_sg)]
        if verbose:
            logger.info("[predict] using user-specified SG %d (classifier skipped)", known_sg)
    else:
        # classifier top-K
        with open(model_dir / "xgb_sg.pkl", "rb") as f:
            sg_pkg = pickle.load(f)
        sg_model, sg_enc = sg_pkg["model"], sg_pkg["encoder"]
        proba = sg_model.predict_proba(desc.reshape(1, -1))[0]
        top_enc = np.argpartition(proba, -top_k_sg)[-top_k_sg:]
        top_sgs = [int(x) for x in sg_enc.inverse_transform(top_enc)]
        sg_mask = top_sgs
        result.classifier_top_k_sgs = top_sgs
        if verbose:
            logger.info("[predict] classifier top-%d SGs: %s", top_k_sg, top_sgs)

    # ── 3. Retrieve top-N templates within SG mask ───────────────────
    pool = TemplatePool(str(metadata_csv), cif_root=str(cif_dir))
    all_hits = []
    for sg in sg_mask:
        hits = pool.search(space_group=sg, descriptor_vector=desc, top_n=n_retry)
        all_hits.append(hits)
    import pandas as pd
    combined = pd.concat(all_hits, ignore_index=True) if all_hits else pd.DataFrame()
    if len(combined) == 0:
        result.status = "NO_CANDIDATE"
        result.warnings.append(f"Template pool has no entries in the requested SG mask {sg_mask}.")
        if verbose:
            for w in result.warnings:
                print(f"  ! {w}")
        return result
    combined = combined.sort_values(
        combined.columns[-1] if "similarity" in combined.columns[-1].lower()
        else combined.columns[-1],
        ascending=False,
    ).head(n_retry).reset_index(drop=True)

    # ── 4. Rank-order substitution loop ──────────────────────────────
    engine = SubstitutionEngine()
    sub_res = None
    tmpl_struct = None
    for rank, row in combined.iterrows():
        c_mid = row.get("material_id") or row.get("mp_id") or row.iloc[0]
        c_formula = row.get("formula", "?")
        try:
            c_cif = cif_dir / f"{c_mid}.cif"
            cs = Structure.from_file(str(c_cif))
        except Exception as exc:
            if verbose:
                logger.debug("[predict] rank=%d CIF load error (%s): %s", rank, c_mid, exc)
            continue
        try:
            subs = engine.find_substitutions(formula, cs)
            sr = next((r for r in subs if r.success), None)
        except Exception as exc:
            if verbose:
                logger.debug("[predict] rank=%d substitution error (%s): %s", rank, c_mid, exc)
            continue
        if sr is not None:
            sub_res = sr
            tmpl_struct = cs
            result.template_material_id = c_mid
            result.template_rank = int(rank)
            result.template_formula = c_formula
            result.substitution_method = sr.method
            if verbose:
                logger.info("[predict] substitution SUCCESS at rank %d, template %s (%s), method=%s",
                            rank, c_mid, c_formula, sr.method)
            break

    if sub_res is None:
        result.status = "SUBSTITUTION_FAILED"
        result.warnings.append(f"None of the top {len(combined)} templates produced a feasible substitution.")
        if verbose:
            for w in result.warnings:
                print(f"  ! {w}")
        return result

    # ── 5. Apply substitution → produce structure ────────────────────
    try:
        pred_struct = engine.apply_substitution(tmpl_struct, sub_res)
    except Exception as exc:
        result.status = "SUBSTITUTION_FAILED"
        result.warnings.append(f"apply_substitution raised: {exc}")
        if verbose:
            print(f"  ! {result.warnings[-1]}")
        return result

    result.substituted_structure = pred_struct
    result.is_ordered = bool(pred_struct.is_ordered)

    # ── 6. Save substituted CIF ──────────────────────────────────────
    sub_cif = output_dir / f"{formula.replace('/', '_')}_substituted.cif"
    _write_cif(pred_struct, sub_cif)
    result.substituted_cif_path = sub_cif
    if verbose:
        print(f"  ✓ substituted CIF saved: {sub_cif}")

    # ── 7. Partial occupancy branch ──────────────────────────────────
    if not result.is_ordered:
        result.status = "PARTIAL_OCCUPANCY"
        result.warnings.extend([
            "Substituted structure contains partial site occupancies (is_ordered=False).",
            "MatterSim cannot relax disordered structures directly; the "
            "substituted CIF is on disk but has not been relaxed.",
        ])
        if verbose:
            print("  ! partial occupancy detected — MatterSim relaxation skipped")
            for w in result.warnings:
                print(f"    - {w}")
        return result

    # ── 8. Relax (if requested) ──────────────────────────────────────
    if not do_relax:
        result.status = "SUBSTITUTED_ONLY"
        if verbose:
            print("  (do_relax=False — skipping relaxation)")
        return result

    if verbose:
        print("  … running MatterSim relaxation (may take 5–60 s)")
    relaxed, converged, dv = _relax_with_mattersim(pred_struct, device=device)
    result.relaxation_converged = bool(converged)
    result.volume_change = dv
    if relaxed is None or not converged or np.isnan(dv) or dv > _VOL_CHANGE_MAX:
        result.status = "RELAX_FAILED"
        if relaxed is None or not converged:
            result.warnings.append("MatterSim relaxation did not converge within 500 steps / 300 s timeout.")
        if not np.isnan(dv) and dv > _VOL_CHANGE_MAX:
            result.warnings.append(f"Volume change |ΔV/V| = {dv:.3f} exceeds 0.15 filter.")
        if verbose:
            for w in result.warnings:
                print(f"  ! {w}")
        # Still return relaxed structure if it exists (user can inspect why)
        result.relaxed_structure = relaxed
        return result

    # ── 9. Compute predicted SG + save relaxed CIF ───────────────────
    result.relaxed_structure = relaxed
    try:
        sga = SpacegroupAnalyzer(relaxed, symprec=0.1, angle_tolerance=5)
        result.predicted_space_group = int(sga.get_space_group_number())
    except Exception:
        pass
    relaxed_cif = output_dir / f"{formula.replace('/', '_')}_relaxed.cif"
    _write_cif(relaxed, relaxed_cif)
    result.relaxed_cif_path = relaxed_cif
    result.status = "SUCCESS"
    if verbose:
        print(f"  ✓ relaxed CIF saved: {relaxed_cif}")
        print(f"  ✓ predicted SG: {result.predicted_space_group}")
    return result
