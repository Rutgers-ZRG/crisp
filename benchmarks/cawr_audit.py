"""CAWR symmetry-recovery audit.

Does CAWR provably *increase* symmetry recovery over plain relaxation
at a matched step budget? Protocol:

  - Take structures whose ideal spacegroup is a stable minimum of the
    potential (verified in calibration): si16 diamond (227), anatase
    (141), cristobalite (92), spinel (14-atom primitive, 227).
  - Perturb: Gaussian position noise sigma + 2% random strain
    (spglib then reads P1 at tight symprec).
  - Three arms at matched total step budget:
      plain  : unbiased LBFGS relax (bias_steps + cleanup_steps).
      cawr   : cawr_refine with the FIXED gradient (2*(fp-mu)).
      legacy : cawr_refine with the historical 4*n_c*(fp-mu) gradient.
  - Measure: spglib SG across a symprec sweep; recovery = detected SG
    equals the ideal SG at symprec <= recovery_symprec. Secondary:
    per-element FP variance reduction.

Usage:
  python -m benchmarks.cawr_audit --out results_cawr_audit --seeds 6
"""

import argparse
import json
import logging
import os

import numpy as np

from .metrics import sg_number
from .systems import (si_diamond, tio2_anatase, cristobalite_alpha,
                      spinel_mgal2o4)

logger = logging.getLogger(__name__)

SYMPREC_SWEEP = (1e-4, 1e-3, 1e-2, 5e-2, 1e-1)

AUDIT_SYSTEMS = {
    'si16_diamond': (lambda: si_diamond().repeat((2, 1, 1)), 227),
    'anatase': (tio2_anatase, 141),
    'cristobalite': (cristobalite_alpha, 92),
    'spinel': (spinel_mgal2o4, 227),
}


def legacy_cawr_loss_grad(fp, labels):
    """Historical (incorrect) gradient: 4*n_c*(fp_i - mu_c)."""
    nat, fp_dim = fp.shape
    loss = 0.0
    dL_dfp = np.zeros((nat, fp_dim))
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        n_c = len(idx)
        if n_c < 2:
            continue
        mu_c = fp[idx].mean(axis=0)
        diff = fp[idx] - mu_c
        loss += float((diff ** 2).sum())
        for j, i in enumerate(idx):
            dL_dfp[i] = 4.0 * n_c * diff[j]
    return loss, dL_dfp


def perturb(atoms, sigma, strain_frac, rng):
    out = atoms.copy()
    out.positions = out.positions + sigma * rng.standard_normal(
        out.positions.shape)
    raw = strain_frac * rng.standard_normal((3, 3))
    eps = 0.5 * (raw + raw.T)
    out.set_cell(out.cell.array @ (np.eye(3) + eps), scale_atoms=True)
    return out


def sg_profile(atoms):
    return {str(sp): sg_number(atoms, sp) for sp in SYMPREC_SWEEP}


def recovered(profile, true_sg, max_symprec=1e-2):
    return any(profile[str(sp)] == true_sg
               for sp in SYMPREC_SWEEP if sp <= max_symprec)


def per_element_fp_var(atoms, fp_calc):
    """Mean FP variance within each element group."""
    fp = fp_calc.get_fingerprints(atoms)
    symbols = np.array(atoms.get_chemical_symbols())
    out = []
    for s in np.unique(symbols):
        sel = fp[symbols == s]
        if len(sel) > 1:
            out.append(float(((sel - sel.mean(0)) ** 2).mean()))
    return float(np.mean(out)) if out else 0.0


def plain_relax(atoms, calc, steps):
    from ase.optimize import LBFGS
    try:
        from ase.filters import FrechetCellFilter as CellFilter
    except ImportError:
        from ase.constraints import ExpCellFilter as CellFilter
    atoms = atoms.copy()
    atoms.calc = calc
    opt = LBFGS(CellFilter(atoms), logfile=None)
    try:
        opt.run(fmax=0.01, steps=steps)
    except Exception as exc:
        logger.debug("plain relax stopped: %s", exc)
    return atoms


def run_audit(out_dir, n_seeds=6, sigmas=(0.10, 0.20), bias_steps=40,
              cleanup_steps=10, device='cpu'):
    import crisp.cawr as cawr_mod
    from crisp.cawr import CAWRConfig, cawr_refine, cawr_snap, spglib_snap
    from crisp.fingerprint import FingerprintCalculator
    from .runner import make_calc_factory

    os.makedirs(out_dir, exist_ok=True)
    calc_factory = make_calc_factory('mattersim', device=device)
    calc = calc_factory()
    fp_calc = FingerprintCalculator(cutoff=5.0, natx=150)
    fixed_grad = cawr_mod.cawr_loss_grad

    cfg = CAWRConfig(max_steps=bias_steps, cleanup_steps=cleanup_steps,
                     optimizer='LBFGS', relax_cell=True)
    total_steps = bias_steps + cleanup_steps

    rows = []
    for sys_name, (builder, true_sg) in AUDIT_SYSTEMS.items():
        ideal = builder()
        for sigma in sigmas:
            for seed in range(n_seeds):
                rng = np.random.default_rng(1000 * seed + 7)
                pert = perturb(ideal, sigma, 0.02, rng)
                prof0 = sg_profile(pert)
                var0 = per_element_fp_var(pert, fp_calc)

                arms = {}
                # plain control
                arms['plain'] = plain_relax(pert, calc, total_steps)
                # fixed-gradient CAWR
                cawr_mod.cawr_loss_grad = fixed_grad
                arms['cawr'] = cawr_refine(pert.copy(), fp_calc, calc, cfg)
                # legacy-gradient CAWR
                cawr_mod.cawr_loss_grad = legacy_cawr_loss_grad
                arms['legacy'] = cawr_refine(pert.copy(), fp_calc, calc,
                                             cfg)
                cawr_mod.cawr_loss_grad = fixed_grad
                # J+ FP-space snap (no MLIP calls) + full plain budget
                snapped = cawr_snap(pert, fp_calc)
                arms['jsnap'] = plain_relax(snapped, calc, total_steps)
                # spglib symmetrization snap + full plain budget
                arms['spgsnap'] = plain_relax(spglib_snap(pert), calc,
                                              total_steps)

                row = {'system': sys_name, 'sigma': sigma, 'seed': seed,
                       'true_sg': true_sg, 'sg_before': prof0,
                       'fp_var_before': var0}
                for arm, atoms in arms.items():
                    prof = sg_profile(atoms)
                    row[f'sg_after_{arm}'] = prof
                    row[f'recovered_{arm}'] = recovered(prof, true_sg)
                    row[f'fp_var_{arm}'] = per_element_fp_var(atoms,
                                                              fp_calc)
                rows.append(row)
                logger.info(
                    "%s s=%.2f seed=%d | plain:%s cawr:%s legacy:%s",
                    sys_name, sigma, seed,
                    row['recovered_plain'], row['recovered_cawr'],
                    row['recovered_legacy'])

    with open(os.path.join(out_dir, 'cawr_audit.json'), 'w') as f:
        json.dump(rows, f, indent=1)

    # ---- Summary ----
    print("\n" + "=" * 72)
    print(f"CAWR SYMMETRY-RECOVERY AUDIT ({len(rows)} cases, "
          f"budget {total_steps} steps/arm, recovery @ symprec<=1e-2)")
    print("=" * 72)
    print(f"{'system':<16} {'sigma':<6} {'plain':<8} {'cawr':<8} "
          f"{'legacy':<8} {'jsnap':<8} {'spgsnap':<8}")
    for sys_name in AUDIT_SYSTEMS:
        for sigma in sigmas:
            sel = [r for r in rows
                   if r['system'] == sys_name and r['sigma'] == sigma]
            if not sel:
                continue
            n = len(sel)
            rec = {a: sum(r[f'recovered_{a}'] for r in sel)
                   for a in ('plain', 'cawr', 'legacy', 'jsnap', 'spgsnap')}
            vr = {a: np.mean([1.0 - r[f'fp_var_{a}'] /
                              max(r['fp_var_before'], 1e-30)
                              for r in sel])
                  for a in ('plain', 'cawr')}
            print(f"{sys_name:<16} {sigma:<6.2f} "
                  f"{rec['plain']}/{n:<6} {rec['cawr']}/{n:<6} "
                  f"{rec['legacy']}/{n:<6} "
                  f"{rec['jsnap']}/{n:<6} {rec['spgsnap']}/{n:<6} "
                  f"varred c/p: {100*vr['cawr']:.0f}%/{100*vr['plain']:.0f}%")
    totals = {a: sum(r[f'recovered_{a}'] for r in rows)
              for a in ('plain', 'cawr', 'legacy', 'jsnap', 'spgsnap')}
    print(f"\nTOTALS: plain {totals['plain']}/{len(rows)}, "
          f"cawr {totals['cawr']}/{len(rows)}, "
          f"legacy {totals['legacy']}/{len(rows)}, "
          f"jsnap {totals['jsnap']}/{len(rows)}, "
          f"spgsnap {totals['spgsnap']}/{len(rows)}")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default='results_cawr_audit')
    p.add_argument('--seeds', type=int, default=6)
    p.add_argument('--device', default='cpu')
    p.add_argument('--bias-steps', type=int, default=40)
    p.add_argument('--cleanup-steps', type=int, default=10)
    p.add_argument('--sigmas', nargs='+', type=float, default=[0.10, 0.20])
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    run_audit(args.out, n_seeds=args.seeds, device=args.device,
              bias_steps=args.bias_steps, cleanup_steps=args.cleanup_steps,
              sigmas=tuple(args.sigmas))


if __name__ == '__main__':
    main()
