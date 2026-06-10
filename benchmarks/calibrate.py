"""Per-potential ground-truth calibration.

For each system: relax every reference polymorph on the potential
(3 jittered replicas each, keep the best), rank by enthalpy. The lowest
becomes the search target. Writes calibration_<potential>.json:

  {system: {gs_name, H_ref, sg_ref, ref_file, gt_mismatch,
            polymorphs: {name: {H, sg_relaxed, sg_kept}}}}

Usage:
  python -m benchmarks.calibrate --potential mattersim --out results_harness_v1
"""

import argparse
import json
import logging
import os

import numpy as np

from .metrics import relax_reference, enthalpy_per_atom, sg_number
from .runner import make_calc_factory
from .systems import SYSTEMS

logger = logging.getLogger(__name__)


def calibrate_system(spec, calc_factory, out_dir, potential,
                     n_jitter: int = 3, jitter_std: float = 0.03):
    """Relax all refs of one system; return calibration record."""
    from ase.io import write as ase_write

    results = {}
    best_name, best_H, best_atoms = None, np.inf, None
    for name, builder in spec.refs.items():
        ref = builder()
        best_local_H, best_local_atoms = np.inf, None
        for j in range(n_jitter):
            atoms = ref.copy()
            if j > 0:
                rng = np.random.default_rng(j)
                atoms.positions += jitter_std * rng.standard_normal(
                    atoms.positions.shape)
            try:
                calc = calc_factory()
                relaxed = relax_reference(atoms, calc, spec.pressure_GPa)
                H = enthalpy_per_atom(relaxed, spec.pressure_GPa)
            except Exception as exc:
                logger.warning("%s/%s jitter %d failed: %s",
                               spec.name, name, j, exc)
                continue
            if H < best_local_H:
                best_local_H, best_local_atoms = H, relaxed
        if best_local_atoms is None:
            results[name] = {'H': None, 'sg_relaxed': None, 'sg_kept': False}
            continue
        sg = sg_number(best_local_atoms, 1e-2) or \
            sg_number(best_local_atoms, 5e-2)
        # Did relaxation keep the reference basin? (spacegroup retained)
        sg_built = sg_number(builder(), 1e-3)
        results[name] = {'H': round(best_local_H, 6), 'sg_relaxed': sg,
                         'sg_kept': sg == sg_built}
        if best_local_H < best_H:
            best_name, best_H, best_atoms = name, best_local_H, \
                best_local_atoms

    record = {'polymorphs': results, 'gs_name': best_name,
              'H_ref': round(best_H, 6) if best_name else None,
              'sg_ref': None, 'ref_file': None,
              'gt_mismatch': best_name != spec.expected_gs}
    if best_atoms is not None:
        record['sg_ref'] = sg_number(best_atoms, 1e-2) or \
            sg_number(best_atoms, 5e-2)
        ref_file = os.path.join(
            out_dir, f"ref_{spec.name}_{potential}.extxyz")
        # copy() drops the (shared) calculator — its stale per-atom
        # results would otherwise corrupt the extxyz write
        ase_write(ref_file, best_atoms.copy())
        record['ref_file'] = ref_file
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--potential', required=True,
                   choices=['mattersim', 'matpes', 'lj'])
    p.add_argument('--out', required=True)
    p.add_argument('--systems', nargs='*', default=None,
                   help='subset of systems (default: all)')
    p.add_argument('--device', default=None)
    p.add_argument('--model-path', default="")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    os.makedirs(args.out, exist_ok=True)
    calc_factory = make_calc_factory(args.potential, device=args.device,
                                     model_path=args.model_path)

    cal_path = os.path.join(args.out, f"calibration_{args.potential}.json")
    cal = json.load(open(cal_path)) if os.path.isfile(cal_path) else {}

    names = args.systems or sorted(SYSTEMS)
    for name in names:
        spec = SYSTEMS[name]
        logger.info("=== Calibrating %s on %s (%d refs) ===",
                    name, args.potential, len(spec.refs))
        record = calibrate_system(spec, calc_factory, args.out,
                                  args.potential)
        cal[name] = record
        logger.info("%s: GS=%s H_ref=%s sg=%s mismatch=%s",
                    name, record['gs_name'], record['H_ref'],
                    record['sg_ref'], record['gt_mismatch'])
        for pname, pr in record['polymorphs'].items():
            logger.info("    %-14s H=%-12s sg_relaxed=%-5s kept=%s",
                        pname, pr['H'], pr['sg_relaxed'], pr['sg_kept'])
        with open(cal_path, 'w') as f:
            json.dump(cal, f, indent=2)
    logger.info("Calibration written to %s", cal_path)


if __name__ == '__main__':
    main()
