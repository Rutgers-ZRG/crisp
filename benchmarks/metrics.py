"""Success detection and per-run metrics for the benchmark harness.

Success (strict): the first relaxed structure with
    |H - H_ref| < spec.success_dH_meV   AND
    (spglib spacegroup matches the GS reference at any symprec in
     SYMPRECS (enantiomorphs equivalent)  OR  d_fp < spec.success_dfp).

The energy gate is mandatory: per-atom Hungarian FP distances compress
badly for multi-species systems (impostors 0.3-3 eV/at away can sit at
d_fp ~ 0.02, below any useful standalone threshold — measured on the
2026-06 probe runs), so d_fp/spacegroup only adjudicate *identity among
energy-degenerate* structures.

A secondary tier `success_energy` (energy gate alone) is also reported
— it counts "found the GS energy level", which may be a degenerate
distinct framework.

The reference is the *potential's own* ground state from calibration
(possibly an empirically discovered one — see recalibrate.py), so the
criterion is well-defined even when the potential disagrees with the
literature.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
from ase import Atoms

logger = logging.getLogger(__name__)

# 1e-1 included: relaxations at fmax=0.05 leave sub-meV distortions that
# break tight-tolerance symmetry detection (measured: anatase found by
# search reads SG 43/70 below 0.1, SG 141 at 0.1).
SYMPRECS = (1e-3, 1e-2, 5e-2, 1e-1)

# The 11 enantiomorphic space-group pairs
_ENANTIOMORPHS = {
    76: 78, 78: 76, 91: 95, 95: 91, 92: 96, 96: 92,
    144: 145, 145: 144, 151: 153, 153: 151, 152: 154, 154: 152,
    169: 170, 170: 169, 171: 172, 172: 171, 178: 179, 179: 178,
    180: 181, 181: 180, 212: 213, 213: 212,
}


def sg_number(atoms: Atoms, symprec: float = 1e-3) -> Optional[int]:
    """spglib spacegroup number, or None on failure."""
    import spglib
    try:
        cell = (atoms.cell.array, atoms.get_scaled_positions(),
                atoms.get_atomic_numbers())
        ds = spglib.get_symmetry_dataset(cell, symprec=symprec)
        if ds is None:
            return None
        return ds.number if hasattr(ds, 'number') else ds['number']
    except Exception:
        return None


def sg_matches(found: Optional[int], ref: int) -> bool:
    """True if spacegroup numbers match (enantiomorphs equivalent)."""
    if found is None:
        return False
    return found == ref or _ENANTIOMORPHS.get(ref) == found


def relax_reference(atoms: Atoms, calc, pressure_GPa: float,
                    fmax: float = 0.005, steps: int = 500) -> Atoms:
    """Relax a reference polymorph at pressure on the given calculator."""
    from ase.optimize import LBFGS
    try:
        from ase.filters import FrechetCellFilter as CellFilter
    except ImportError:
        from ase.constraints import ExpCellFilter as CellFilter

    atoms = atoms.copy()
    atoms.calc = calc
    p_eV_A3 = pressure_GPa / 160.21766208
    ecf = CellFilter(atoms, scalar_pressure=p_eV_A3)
    opt = LBFGS(ecf, logfile=None)
    opt.run(fmax=fmax, steps=steps)
    return atoms


def enthalpy_per_atom(atoms: Atoms, pressure_GPa: float) -> float:
    """H/atom = (E + p V)/N with the calculator already attached."""
    e = atoms.get_potential_energy()
    v = atoms.get_volume()
    p_eV_A3 = pressure_GPa / 160.21766208
    return (e + p_eV_A3 * v) / len(atoms)


def build_records(entries, ref_fp: Optional[np.ndarray],
                  ref_H: float, ref_sg: int, fp_calc) -> List[dict]:
    """Per-entry comparison records, sorted by relaxation order.

    entries : archive entries with .fp, .enthalpy, .atoms, .metadata
    ref_fp : per-atom fingerprints of the relaxed GS reference at
        matching atom count, or None (disables the d_fp criterion).
    """
    records = []
    ordered = sorted(entries,
                     key=lambda e: e.metadata.get('relax_index', 1 << 30))
    for e in ordered:
        d_fp = float('inf')
        if ref_fp is not None and e.fp.shape == ref_fp.shape:
            try:
                types = fp_calc.atoms_to_cell(e.atoms)[2]
                d_fp = float(fp_calc._fp_dist(e.fp, ref_fp, types))
            except Exception as exc:
                logger.debug("d_fp failed: %s", exc)
        dH_meV = abs(e.enthalpy - ref_H) * 1000.0
        # Spacegroup match only evaluated when the cheap dH gate passes
        matched = False
        if dH_meV < 20.0:
            for sp in SYMPRECS:
                if sg_matches(sg_number(e.atoms, sp), ref_sg):
                    matched = True
                    break
        records.append({
            'relax_index': e.metadata.get('relax_index'),
            'generation': e.metadata.get('generation'),
            'd_fp': d_fp,
            'dH_meV': dH_meV,
            'H': float(e.enthalpy),
            'sg_match': matched,
        })
    return records


def detect_success_from_records(records: List[dict], success_dfp: float,
                                success_dH_meV: float) -> dict:
    """First record meeting the success criteria; summary stats.

    Strict success: dH < success_dH_meV AND (sg_match OR d_fp < dfp).
    Energy success: dH < success_dH_meV alone (degenerate frameworks
    count).
    """
    out = {
        'success': False,
        'n_relaxed_at_success': None,
        'gen_at_success': None,
        'success_energy': False,
        'n_relaxed_at_energy_hit': None,
        'd_fp_best': None,
        'dH_best_meV': None,
    }
    if records:
        out['d_fp_best'] = min(r['d_fp'] for r in records)
        out['dH_best_meV'] = min(r['dH_meV'] for r in records)
    for r in records:
        energy_ok = r['dH_meV'] < success_dH_meV
        if energy_ok and not out['success_energy']:
            out['success_energy'] = True
            out['n_relaxed_at_energy_hit'] = r['relax_index']
        hit = energy_ok and (r['sg_match'] or r['d_fp'] < success_dfp)
        if hit:
            out['success'] = True
            out['n_relaxed_at_success'] = r['relax_index']
            out['gen_at_success'] = r['generation']
            break
    return out
