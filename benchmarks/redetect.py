"""Recalibrate ground truths from run archives and re-detect success.

Two jobs, run at collection time (canonical numbers come from here, not
from the runner's best-effort in-run detection):

1. RECALIBRATE: if any archive for (system, potential) contains a sane
   structure with H < H_ref - 1 meV, the registry refs did not span the
   potential's ground state (e.g. MatterSim SiO2: an Fdd2 framework
   0.85 meV below alpha-cristobalite). Adopt the discovered minimum as
   the empirical GS reference (gs_name='discovered_sgNNN').

2. REDETECT: rebuild per-run records from each archive against the
   (possibly updated) calibration with the current criterion and
   symprec sweep, and rewrite the result JSONs in place.

Usage:
  python -m benchmarks.redetect --out results_harness_v1 --potential mattersim
"""

import argparse
import glob
import json
import logging
import os
import re

import numpy as np

from .metrics import (build_records, detect_success_from_records,
                      sg_number, SYMPRECS)
from .systems import SYSTEMS, ref_at_composition

logger = logging.getLogger(__name__)


def _load_archive_entries(archive_dir, fp_calc):
    """Minimal archive loader: returns objects with .atoms, .fp,
    .enthalpy, .metadata (mirrors crisp.archive entries)."""
    from ase.io import read as ase_read

    class E:
        pass

    meta_file = os.path.join(archive_dir, 'archive_meta.json')
    if not os.path.isfile(meta_file):
        return []
    meta_list = json.load(open(meta_file))
    entries = []
    for i, m in enumerate(meta_list):
        f = os.path.join(archive_dir, f'struct_{i:04d}.extxyz')
        if not os.path.isfile(f):
            continue
        e = E()
        e.atoms = ase_read(f)
        e.enthalpy = m['enthalpy']
        e.metadata = m.get('metadata', {})
        try:
            e.fp = fp_calc.get_fingerprints(e.atoms)
        except Exception:
            e.fp = np.zeros((len(e.atoms), 1))
        entries.append(e)
    return entries


def _structure_sane(atoms, spec):
    d = atoms.get_all_distances(mic=True)
    np.fill_diagonal(d, np.inf)
    if d.min() < 0.8 * spec.min_dist_ang:
        return False
    vpa = atoms.get_volume() / len(atoms)
    lo, hi = spec.vol_per_atom_range
    return 0.5 * lo < vpa < 1.5 * hi


def recalibrate(out_dir, potential, margin_meV=0.5):
    """Update calibration with search-discovered lower minima."""
    from ase.io import write as ase_write
    from crisp.fingerprint import FingerprintCalculator

    cal_path = os.path.join(out_dir, f'calibration_{potential}.json')
    cal = json.load(open(cal_path))
    changed = False

    for system, spec in SYSTEMS.items():
        if system not in cal or cal[system].get('H_ref') is None:
            continue
        fp_calc = FingerprintCalculator(cutoff=spec.fp_cutoff,
                                        natx=spec.fp_natx)
        best_H = cal[system]['H_ref']
        best_atoms = None
        pattern = os.path.join(
            out_dir, f'{system}_{potential}_*_archive')
        for adir in glob.glob(pattern):
            meta_file = os.path.join(adir, 'archive_meta.json')
            if not os.path.isfile(meta_file):
                continue
            meta_list = json.load(open(meta_file))
            for i, m in enumerate(meta_list):
                if m['enthalpy'] < best_H - margin_meV / 1000.0:
                    from ase.io import read as ase_read
                    f = os.path.join(adir, f'struct_{i:04d}.extxyz')
                    if not os.path.isfile(f):
                        continue
                    atoms = ase_read(f)
                    if _structure_sane(atoms, spec):
                        best_H = m['enthalpy']
                        best_atoms = atoms

        if best_atoms is not None:
            sg = None
            for sp in reversed(SYMPRECS):
                sg = sg_number(best_atoms, sp)
                if sg and sg > 1:
                    break
            ref_file = os.path.join(
                out_dir, f'ref_{system}_{potential}_discovered.extxyz')
            ase_write(ref_file, best_atoms)
            old = cal[system]['H_ref']
            cal[system].update({
                'gs_name': f'discovered_sg{sg}',
                'H_ref': round(float(best_H), 6),
                'sg_ref': sg,
                'ref_file': ref_file,
                'gt_mismatch': True,
                'previous_H_ref': old,
            })
            changed = True
            logger.info('%s/%s: empirical GS adopted — H %.6f -> %.6f '
                        '(sg %s)', system, potential, old, best_H, sg)

    if changed:
        with open(cal_path, 'w') as f:
            json.dump(cal, f, indent=2)
    return cal


def redetect(out_dir, potential):
    """Rebuild records + result JSONs from archives against the
    current calibration and criterion."""
    from ase.io import read as ase_read
    from crisp.fingerprint import FingerprintCalculator

    cal_path = os.path.join(out_dir, f'calibration_{potential}.json')
    cal = json.load(open(cal_path))
    n_updated = 0

    for res_file in sorted(glob.glob(os.path.join(out_dir, '*.json'))):
        base = os.path.basename(res_file)
        if base.startswith('calibration_') or base.endswith(
                '_records.json'):
            continue
        try:
            result = json.load(open(res_file))
        except json.JSONDecodeError:
            continue
        if result.get('potential') != potential:
            continue
        system = result.get('system')
        if system not in SYSTEMS or system not in cal:
            continue
        spec = SYSTEMS[system]
        c = cal[system]
        tag = base[:-5]
        archive_dir = os.path.join(out_dir, f'{tag}_archive')
        if not os.path.isdir(archive_dir):
            continue

        fp_calc = FingerprintCalculator(cutoff=spec.fp_cutoff,
                                        natx=spec.fp_natx)
        entries = _load_archive_entries(archive_dir, fp_calc)
        if not entries:
            continue
        ref_fp = None
        try:
            ref_atoms = ase_read(c['ref_file'])
            ref_sup = ref_at_composition(spec, ref_atoms)
            if ref_sup is None and len(ref_atoms) == spec.n_atoms:
                ref_sup = ref_atoms
            if ref_sup is not None:
                ref_fp = fp_calc.get_fingerprints(ref_sup)
        except Exception as exc:
            logger.debug('%s: ref fp unavailable: %s', tag, exc)

        records = build_records(entries, ref_fp, c['H_ref'], c['sg_ref'],
                                fp_calc)
        det = detect_success_from_records(records, spec.success_dfp,
                                          spec.success_dH_meV)
        result.update(det)
        result['ref_name'] = c.get('gs_name')
        result['ref_H'] = c.get('H_ref')
        result['gt_mismatch'] = c.get('gt_mismatch', False)
        result['redetected'] = True
        with open(res_file, 'w') as f:
            json.dump(result, f, indent=2)
        with open(os.path.join(out_dir, f'{tag}_records.json'),
                  'w') as f:
            json.dump(records, f, indent=1)
        n_updated += 1
        logger.info('%s: success=%s n_at_success=%s (energy tier: %s @ %s)',
                    tag, det['success'], det['n_relaxed_at_success'],
                    det['success_energy'], det['n_relaxed_at_energy_hit'])
    logger.info('redetect: %d results updated', n_updated)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--potential', required=True)
    p.add_argument('--skip-recalibrate', action='store_true')
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')
    if not args.skip_recalibrate:
        recalibrate(args.out, args.potential)
    redetect(args.out, args.potential)


if __name__ == '__main__':
    main()
