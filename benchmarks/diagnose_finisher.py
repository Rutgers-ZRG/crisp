"""Finisher failure-rate diagnosis.

Runs a short CRISP search with the finisher enabled and records EVERY
finisher invocation (including candidates later dropped as duplicates):
bias steps executed, stop reason, d_target trajectory (init -> bias end
-> after cleanup). Reports the silent-failure rate.

Usage:
  python -m benchmarks.diagnose_finisher --system si16 --potential \
      mattersim --seed 42 --budget 120 --out results_diag
"""

import argparse
import json
import logging
import os
import random

import numpy as np

from .runner import make_calc_factory, build_search
from .systems import SYSTEMS

logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--system', required=True, choices=sorted(SYSTEMS))
    p.add_argument('--potential', default='mattersim',
                   choices=['mattersim', 'matpes', 'lj'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--budget', type=int, default=120)
    p.add_argument('--max-gens', type=int, default=8)
    p.add_argument('--out', default='results_diag')
    p.add_argument('--device', default=None)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    logging.getLogger("crisp.archive").setLevel(logging.WARNING)
    logging.getLogger("crisp.surrogate").setLevel(logging.WARNING)
    os.makedirs(args.out, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        import torch
        torch.manual_seed(args.seed)
    except ImportError:
        pass

    spec = SYSTEMS[args.system]
    calc_factory = make_calc_factory(args.potential, device=args.device)
    search = build_search(spec, 'crisp', calc_factory, args.budget,
                          args.max_gens, checkpoint_dir=None)

    # Record every finisher invocation via a wrapper
    events = []
    finisher = search._finisher
    orig_run = finisher.run

    def recording_run(atoms, base_calc, is_mutant=False,
                      stagnation_count=0):
        try:
            out = orig_run(atoms, base_calc, is_mutant=is_mutant,
                           stagnation_count=stagnation_count)
            info = out.info
            events.append({
                'ok': True,
                'bias_steps': info.get('finisher_bias_steps'),
                'stop_reason': info.get('finisher_stop_reason'),
                'd_init': info.get('finisher_d_init'),
                'd_bias_end': info.get('finisher_d_bias_end'),
                'd_final': info.get('finisher_d_final'),
                'mode': info.get('finisher_mode'),
            })
            return out
        except Exception as exc:
            events.append({'ok': False,
                           'stop_reason': f'run_raised:{type(exc).__name__}:'
                                          f'{str(exc)[:80]}'})
            raise

    finisher.run = recording_run

    archive = search.run()

    # ---- Report ----
    n = len(events)
    print("\n" + "=" * 64)
    print(f"FINISHER DIAGNOSIS: {args.system}/{args.potential} "
          f"seed={args.seed} | {n} invocations, "
          f"{search.n_relaxed} relaxations, {len(archive.entries)} unique")
    print("=" * 64)
    if n == 0:
        print("Finisher never invoked — gating too tight?")
        return

    from collections import Counter
    reasons = Counter(e.get('stop_reason') or 'none' for e in events)
    print("\nStop reasons:")
    for r, c in reasons.most_common():
        print(f"  {c:4d}  {r}")

    bias_steps = [e['bias_steps'] for e in events
                  if e.get('bias_steps') is not None]
    if bias_steps:
        bias_steps = np.array(bias_steps)
        print(f"\nBias steps: median={np.median(bias_steps):.0f} "
              f"mean={bias_steps.mean():.1f} "
              f"min={bias_steps.min()} max={bias_steps.max()}")
        early = (bias_steps < 10).sum()
        print(f"Early termination (<10 steps): {early}/{len(bias_steps)} "
              f"= {100.0 * early / len(bias_steps):.0f}%  "
              f"<- the 'silent failure rate'")

    # d_target trajectory + cleanup reversion
    moved, reverted = [], []
    for e in events:
        if e.get('d_init') is not None and e.get('d_bias_end') is not None:
            moved.append(e['d_init'] - e['d_bias_end'])
            if e.get('d_final') is not None:
                reverted.append(e['d_final'] - e['d_bias_end'])
    if moved:
        print(f"\nBias-phase progress (d_init - d_bias_end): "
              f"median={np.median(moved):.4f} (n={len(moved)})")
    if reverted:
        print(f"Cleanup reversion (d_final - d_bias_end): "
              f"median={np.median(reverted):.4f} "
              f"frac>0: {100.0 * (np.array(reverted) > 0).mean():.0f}%")

    out_file = os.path.join(
        args.out, f"diag_finisher_{args.system}_{args.potential}"
                  f"_s{args.seed}.json")
    with open(out_file, 'w') as f:
        json.dump({'system': args.system, 'potential': args.potential,
                   'seed': args.seed, 'n_invocations': n,
                   'events': events}, f, indent=1)
    print(f"\nEvents written to {out_file}")


if __name__ == '__main__':
    main()
