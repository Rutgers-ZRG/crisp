"""Run one (system x potential x mode x seed) benchmark to a result JSON.

Modes (all share the CRISPSearch pipeline so plumbing is identical):
  random : pyXtal generation + plain relaxation. No GP skipping, no
           mutations, no finisher/CAWR. The honest baseline.
  fponly : FP-only local treatment (ZeroCalculator base): finisher +
           CAWR driven purely by fingerprint forces (fixed lambda=1),
           FP-Jacobian mutations, GP filter on. Replicates the
           benchmark_b28_fponly.py configuration.
  crisp  : full v0.4 — softmutants + FP-J mutants + FP-target finisher
           + CAWR pretreat + GP filter + PSO stagnation.

Usage:
  python -m benchmarks.runner --system sio2_18 --potential mattersim \
      --mode crisp --seed 42 --budget-relax 600 --out results_harness_v1
"""

import argparse
import hashlib
import json
import logging
import os
import random
import time
from typing import Optional

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from .systems import SYSTEMS, SystemSpec, ref_at_composition

logger = logging.getLogger(__name__)


class ZeroCalculator(Calculator):
    """Zero energy/forces/stress — base for FP-only treatment."""
    implemented_properties = ["energy", "forces", "stress"]

    def calculate(self, atoms=None, properties=None,
                  system_changes=tuple(all_changes)):
        super().calculate(atoms, properties, system_changes)
        nat = len(self.atoms)
        self.results["energy"] = 0.0
        self.results["forces"] = np.zeros((nat, 3))
        self.results["stress"] = np.zeros(6)


# Spec used by unit tests: tiny LJ system, cheap and potential-free.
MICRO_TEST_SPEC = SystemSpec(
    name='micro_lj', composition={'Si': 4}, pressure_GPa=0.0,
    difficulty='easy', refs={}, expected_gs='',
    vol_per_atom_range=(14.0, 30.0), min_dist_ang=1.8,
    max_generations=3, n_random=6, n_mutants=2,
    fp_cutoff=5.0, fp_natx=40)


def make_calc_factory(potential: str, device: Optional[str] = None,
                      model_path: str = ""):
    """Return a calculator factory for the given potential.

    The factory caches a single calculator instance — model loading
    (MatterSim/MACE) is expensive and ASE calculators are reusable.
    """
    cache = {}

    def factory():
        if 'calc' in cache:
            return cache['calc']
        if potential == 'mattersim':
            import torch
            from mattersim.forcefield import MatterSimCalculator
            dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')
            kwargs = {'device': dev}
            if model_path:
                kwargs['load_path'] = model_path
            cache['calc'] = MatterSimCalculator(**kwargs)
        elif potential == 'matpes':
            from mace.calculators import MACECalculator
            import torch
            dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')
            mp = model_path or os.environ.get('MATPES_MODEL', '')
            if not mp:
                raise ValueError("matpes potential needs --model-path or "
                                 "MATPES_MODEL env var")
            cache['calc'] = MACECalculator(model_paths=mp, device=dev)
        elif potential == 'lj':
            from ase.calculators.lj import LennardJones
            cache['calc'] = LennardJones(sigma=2.0, epsilon=1.0, rc=5.0)
        else:
            raise ValueError(f"Unknown potential: {potential!r}")
        return cache['calc']

    return factory


VARIANTS = ('', 'nocawr', 'nofinisher', 'jsnap', 'gpguided', 'gptune',
            'strongfin', 'fpjheavy', 'zerotreat')


def build_search(spec: SystemSpec, mode: str, calc_factory,
                 budget_relax: int, max_generations: int,
                 checkpoint_dir: str, variant: str = ''):
    """Construct CRISPSearch for the given mode (documented configs).

    `variant` applies a single Phase-4 A/B modification to crisp mode:
      nocawr     : CAWR pretreat off
      nofinisher : FP-target finisher off
      jsnap      : CAWR pretreat replaced by the J+ FP-space snap
      gpguided   : GP-guided refinement on (fixed lambda=3)
      gptune     : GP hyperparameter auto-tune on
      strongfin  : finisher bias 120 steps, lambda_max 40
      fpjheavy   : no softmutants, 2x FP-Jacobian mutants (the fponly
                   mutation mix with the crisp MLIP treatment)
      zerotreat  : finisher/CAWR on ZeroCalculator with fixed lambda=1
                   (the fponly treatment with the crisp mutation mix)
    """
    from crisp import CRISPSearch, FingerprintCalculator, CAWRConfig
    from crisp.finishers.fp_target import FinisherConfig

    fp_calc = FingerprintCalculator(cutoff=spec.fp_cutoff, natx=spec.fp_natx)

    common = dict(
        mlip_calc_factory=calc_factory,
        fp_calc=fp_calc,
        composition=spec.composition,
        pressure_GPa=spec.pressure_GPa,
        n_random=spec.n_random,
        max_generations=max_generations,
        gp_length_scale=1.0,
        dup_threshold=spec.dup_threshold,
        convergence_gens=10 ** 6,        # budget terminates, not convergence
        min_dist_ang=spec.min_dist_ang,
        vol_per_atom_range=spec.vol_per_atom_range,
        screening_mode="filter",
        checkpoint_dir=checkpoint_dir,
        local_relax_mode="plain",
        budget_relax=budget_relax,
    )

    if mode == 'random':
        return CRISPSearch(
            use_mutations=False,
            enable_fpj_mutations=False,
            enable_fp_finisher=False,
            enable_cawr_pretreat=False,
            max_skip_frac=0.0,           # GP filter disabled via guardrail
            **common)

    if mode == 'fponly':
        finisher_cfg = FinisherConfig(
            matching_backend="hungarian", pre_steps=0, bias_steps=30,
            cleanup_fmax=0.05, cleanup_max_steps=0,
            eta=0.3, lambda_min=1.0, lambda_max=1.0, anneal_to_zero=False,
            optimizer="FIRE", gate_enabled=True, run_on_mutants=True,
            d_gate_init=0.50, d_gate_final=0.20, anneal_gens=10,
            matching_interval=10, relax_cell=True, min_dist_ang=1.2,
            pressure_GPa=spec.pressure_GPa,
            stagnation_gens=5, diversity_topk=3, repulsion_weight=0.3)
        cawr_cfg = CAWRConfig(
            eta=0.3, lambda_min=1.0, max_steps=30, min_k=2, max_k=8,
            recluster_interval=10, cleanup_steps=0, cleanup_fmax=0.05,
            pressure_GPa=spec.pressure_GPa, relax_cell=True,
            anneal_to_zero=False, optimizer="FIRE", min_dist_ang=1.2,
            max_f_bias_rms=50.0)
        return CRISPSearch(
            treatment_calc_factory=lambda: ZeroCalculator(),
            use_mutations=False,
            enable_fpj_mutations=True,
            n_fpj_mutants=spec.n_mutants + 5,
            enable_fp_finisher=True,
            finisher_config=finisher_cfg,
            n_finisher_targets=8,
            enable_cawr_pretreat=True,
            cawr_config=cawr_cfg,
            gp_energy_margin=0.2, gp_confidence_frac=0.15,
            **common)

    if mode == 'crisp':
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant: {variant!r}")
        bias_steps = 120 if variant == 'strongfin' else 60
        lambda_max = 40.0 if variant == 'strongfin' else 20.0
        finisher_cfg = FinisherConfig(
            matching_backend="hungarian", pre_steps=30,
            bias_steps=bias_steps,
            cleanup_fmax=0.02, cleanup_max_steps=60,
            eta=0.3, lambda_min=0.0, lambda_max=lambda_max,
            anneal_to_zero=True,
            optimizer="FIRE", gate_enabled=True, run_on_mutants=True,
            d_gate_init=0.50, d_gate_final=0.20, anneal_gens=10,
            matching_interval=5, relax_cell=True, min_dist_ang=1.0,
            pressure_GPa=spec.pressure_GPa,
            stagnation_gens=5, diversity_topk=3, repulsion_weight=0.3)
        cawr_cfg = CAWRConfig(
            eta=0.3, lambda_min=0.0, max_steps=30, min_k=2, max_k=8,
            recluster_interval=10, cleanup_steps=10, cleanup_fmax=0.05,
            pressure_GPa=spec.pressure_GPa, relax_cell=True,
            anneal_to_zero=True, optimizer="FIRE", min_dist_ang=1.2,
            max_f_bias_rms=50.0)
        if variant == 'zerotreat':
            finisher_cfg.pre_steps = 0
            finisher_cfg.bias_steps = 30
            finisher_cfg.cleanup_max_steps = 0
            finisher_cfg.lambda_min = 1.0
            finisher_cfg.lambda_max = 1.0
            finisher_cfg.anneal_to_zero = False
            cawr_cfg.lambda_min = 1.0
            cawr_cfg.anneal_to_zero = False
            cawr_cfg.cleanup_steps = 0
        return CRISPSearch(
            treatment_calc_factory=(
                (lambda: ZeroCalculator()) if variant == 'zerotreat'
                else None),
            use_mutations=(variant != 'fpjheavy'),
            n_mutants=spec.n_mutants,
            enable_fpj_mutations=True,
            n_fpj_mutants=(2 * spec.n_mutants if variant == 'fpjheavy'
                           else spec.n_mutants),
            enable_fp_finisher=(variant != 'nofinisher'),
            finisher_config=finisher_cfg,
            n_finisher_targets=8,
            enable_cawr_pretreat=(variant != 'nocawr'),
            cawr_config=cawr_cfg,
            cawr_pretreat_mode='snap' if variant == 'jsnap' else 'refine',
            enable_gp_guided=(variant == 'gpguided'),
            gp_auto_tune=(variant == 'gptune'),
            gp_energy_margin=0.2, gp_confidence_frac=0.15,
            **common)

    raise ValueError(f"Unknown mode: {mode!r}")


def _json_default(o):
    """numpy scalars (float32 from GPU calcs) and arrays -> JSON."""
    if hasattr(o, "item") and not hasattr(o, "__len__"):
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


def _gen0_fp_hash(archive) -> str:
    """Determinism witness: hash of gen-0 structures in relax order."""
    h = hashlib.sha1()
    entries = [e for e in archive.entries
               if e.metadata.get('generation') == 0]
    entries.sort(key=lambda e: e.metadata.get('relax_index', 1 << 30))
    for e in entries:
        h.update(np.round(e.atoms.positions, 5).tobytes())
        h.update(np.round(e.atoms.cell.array, 5).tobytes())
    return h.hexdigest()


def _config_echo(search) -> dict:
    keys = ['n_random', 'n_mutants', 'use_mutations', 'enable_fpj_mutations',
            'n_fpj_mutants', 'max_skip_frac', 'screening_mode',
            'dup_threshold', 'min_dist_ang', 'pressure_GPa',
            'local_relax_mode', 'budget_relax', 'max_generations']
    echo = {k: getattr(search, k, None) for k in keys}
    echo['finisher'] = search._finisher is not None
    echo['cawr'] = search._cawr_config is not None
    echo['gp_guided'] = search.enable_gp_guided
    echo['sanity_H_floor'] = search.sanity_H_floor
    return echo


def run_benchmark(system: str, potential: str, mode: str, seed: int,
                  budget_relax: int, out_dir: str,
                  max_generations: Optional[int] = None,
                  device: Optional[str] = None, model_path: str = "",
                  calibration: Optional[str] = None,
                  spec_override: Optional[SystemSpec] = None,
                  variant: str = '') -> dict:
    """Run one benchmark; write and return the result record."""
    spec = spec_override or SYSTEMS[system]
    max_gens = max_generations or spec.max_generations
    os.makedirs(out_dir, exist_ok=True)
    mode_label = f"{mode}-{variant}" if variant else mode
    tag = f"{spec.name}_{potential}_{mode_label}_s{seed}"
    ckpt_dir = os.path.join(out_dir, f"{tag}_ckpt")

    # Seed everything (numpy drives pyXtal + CRISP internals)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass

    calc_factory = make_calc_factory(potential, device=device,
                                     model_path=model_path)
    search = build_search(spec, mode, calc_factory, budget_relax,
                          max_gens, ckpt_dir, variant=variant)

    # PES-poisoning floor from calibration (H_ref - 0.5 eV/at)
    cal_file_early = calibration or os.path.join(
        out_dir, f"calibration_{potential}.json")
    if os.path.isfile(cal_file_early):
        cal_all = json.load(open(cal_file_early))
        if spec.name in cal_all and cal_all[spec.name].get('H_ref') is not None:
            search.sanity_H_floor = cal_all[spec.name]['H_ref'] - 0.5

    resume_from = ckpt_dir if os.path.isdir(ckpt_dir) and \
        any(p.startswith('gen_') for p in os.listdir(ckpt_dir)) else None
    if resume_from:
        logger.info("Resuming %s from %s", tag, resume_from)

    t0 = time.time()
    archive = search.run(resume_from=resume_from)
    wall_s = time.time() - t0

    result = {
        'system': spec.name, 'potential': potential, 'mode': mode_label,
        'variant': variant,
        'seed': seed, 'budget_relax': budget_relax,
        'n_relaxed_total': search.n_relaxed,
        'n_unique': len(archive.entries),
        'wall_s': round(wall_s, 1),
        'best_H': min((e.enthalpy for e in archive.entries), default=None),
        'gen0_fp_hash': _gen0_fp_hash(archive),
        'config_echo': _config_echo(search),
        'success': False, 'n_relaxed_at_success': None,
        'gen_at_success': None, 'd_fp_best': None, 'dH_best_meV': None,
        'gt_mismatch': None, 'ref_name': None, 'ref_H': None,
    }

    # Success detection against calibration (if provided)
    cal_file = calibration or os.path.join(
        out_dir, f"calibration_{potential}.json")
    if os.path.isfile(cal_file) and spec.name in json.load(open(cal_file)):
        from .metrics import build_records, detect_success_from_records
        from ase.io import read as ase_read
        cal = json.load(open(cal_file))[spec.name]
        ref_atoms = ase_read(cal['ref_file'])
        ref_sup = ref_at_composition(spec, ref_atoms)
        ref_fp = None
        if ref_sup is not None:
            ref_fp = search.fp_calc.get_fingerprints(ref_sup)
        records = build_records(archive.entries, ref_fp,
                                cal['H_ref'], cal['sg_ref'],
                                search.fp_calc)
        det = detect_success_from_records(records, spec.success_dfp,
                                          spec.success_dH_meV)
        result.update(det)
        result['gt_mismatch'] = cal.get('gt_mismatch', False)
        result['ref_name'] = cal.get('gs_name')
        result['ref_H'] = cal.get('H_ref')
        with open(os.path.join(out_dir, f"{tag}_records.json"), 'w') as f:
            json.dump(records, f, indent=1, default=_json_default)

    # Persist archive + result
    try:
        archive.save(os.path.join(out_dir, f"{tag}_archive"))
    except Exception as exc:
        logger.warning("Archive save failed: %s", exc)
    with open(os.path.join(out_dir, f"{tag}.json"), 'w') as f:
        json.dump(result, f, indent=2, default=_json_default)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--system', required=True, choices=sorted(SYSTEMS))
    p.add_argument('--potential', required=True,
                   choices=['mattersim', 'matpes', 'lj'])
    p.add_argument('--mode', required=True,
                   choices=['random', 'fponly', 'crisp'])
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--budget-relax', type=int, default=600)
    p.add_argument('--out', required=True)
    p.add_argument('--max-gens', type=int, default=None)
    p.add_argument('--device', default=None)
    p.add_argument('--model-path', default="")
    p.add_argument('--calibration', default=None)
    p.add_argument('--variant', default='',
                   choices=list(VARIANTS))
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    result = run_benchmark(
        system=args.system, potential=args.potential, mode=args.mode,
        seed=args.seed, budget_relax=args.budget_relax, out_dir=args.out,
        max_generations=args.max_gens, device=args.device,
        model_path=args.model_path, calibration=args.calibration,
        variant=args.variant)
    print(json.dumps({k: v for k, v in result.items()
                      if k != 'config_echo'}, indent=2))


if __name__ == '__main__':
    main()
