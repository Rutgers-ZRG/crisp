"""Aggregate benchmark result JSONs into a summary table.

Usage:
  python -m benchmarks.aggregate results_harness_v1/ [--markdown]
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np


def load_results(out_dir):
    results = []
    for path in sorted(glob.glob(os.path.join(out_dir, "*.json"))):
        base = os.path.basename(path)
        if base.startswith("calibration_") or base.endswith("_records.json"):
            continue
        try:
            with open(path) as f:
                r = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if all(k in r for k in ('system', 'potential', 'mode', 'seed')):
            results.append(r)
    return results


def summarize(results):
    groups = defaultdict(list)
    for r in results:
        groups[(r['system'], r['potential'], r['mode'])].append(r)

    rows = []
    for (system, potential, mode), runs in sorted(groups.items()):
        n = len(runs)
        succ = [r for r in runs if r.get('success')]
        n_at = [r['n_relaxed_at_success'] for r in succ
                if r.get('n_relaxed_at_success') is not None]
        gens = [r['gen_at_success'] for r in succ
                if r.get('gen_at_success') is not None]
        rows.append({
            'system': system, 'potential': potential, 'mode': mode,
            'seeds': n,
            'success_rate': f"{len(succ)}/{n}",
            'median_n_relax': int(np.median(n_at)) if n_at else None,
            'median_gen': int(np.median(gens)) if gens else None,
            'mean_total_relax': int(np.mean([r['n_relaxed_total']
                                             for r in runs])),
            'mean_wall_h': round(np.mean([r['wall_s'] for r in runs])
                                 / 3600, 2),
            'best_d_fp': min((r['d_fp_best'] for r in runs
                              if r.get('d_fp_best') is not None),
                             default=None),
            'seeds_list': sorted(r['seed'] for r in runs),
        })
    return rows


def print_table(rows, markdown=False):
    cols = ['system', 'potential', 'mode', 'seeds', 'success_rate',
            'median_n_relax', 'median_gen', 'mean_total_relax',
            'mean_wall_h', 'best_d_fp']
    if markdown:
        print('| ' + ' | '.join(cols) + ' |')
        print('|' + '---|' * len(cols))
        for r in rows:
            print('| ' + ' | '.join(str(r.get(c)) for c in cols) + ' |')
    else:
        widths = {c: max(len(c), max((len(str(r.get(c))) for r in rows),
                                     default=4)) for c in cols}
        print('  '.join(c.ljust(widths[c]) for c in cols))
        for r in rows:
            print('  '.join(str(r.get(c)).ljust(widths[c]) for c in cols))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('out_dir')
    p.add_argument('--markdown', action='store_true')
    args = p.parse_args()
    rows = summarize(load_results(args.out_dir))
    if not rows:
        print(f"No results in {args.out_dir}")
        return
    print_table(rows, markdown=args.markdown)


if __name__ == '__main__':
    main()
