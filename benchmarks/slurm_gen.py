"""Generate sbatch scripts for a benchmark run matrix on Amarel.

One job per (system x potential x mode x seed). Jobs run the whole
search on a single GPU node with local-mode relaxation (no nested
Slurm submission). Resumable: re-submitting a job resumes from its
checkpoint directory.

Usage (from the repo root on Amarel, or locally then rsync):
  python -m benchmarks.slurm_gen --systems sio2_18 mgsio3_20 \
      --potentials mattersim --modes random crisp fponly \
      --seeds 42 123 314 --budget 600 --out results_harness_v1 \
      --script-dir slurm_jobs
  cd slurm_jobs && for f in *.sbatch; do sbatch $f; done
"""

import argparse
import os

from .systems import SYSTEMS

TEMPLATE = """#!/bin/bash
#SBATCH --job-name={tag}
#SBATCH --partition={partition}
#SBATCH {gres_line}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --output={out_dir}/slurm_{tag}_%j.out
#SBATCH --requeue

export MKL_NUM_THREADS={cpus}
export OMP_NUM_THREADS={cpus}
cd {workdir}
{python} -u -m benchmarks.runner \\
    --system {system} --potential {potential} --mode {mode} \\
    --seed {seed} --budget-relax {budget} --max-gens {max_gens} \\
    --out {out_dir} {variant_arg}{extra}
"""

ENV_PYTHON = {
    'mattersim': '~/miniconda3/envs/msim/bin/python',
    'matpes': '~/miniconda3/envs/mace/bin/python',
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--systems', nargs='+', required=True,
                   choices=sorted(SYSTEMS))
    p.add_argument('--potentials', nargs='+', default=['mattersim'],
                   choices=['mattersim', 'matpes'])
    p.add_argument('--modes', nargs='+', required=True,
                   choices=['random', 'fponly', 'crisp'])
    p.add_argument('--seeds', nargs='+', type=int, required=True)
    p.add_argument('--budget', type=int, default=600)
    p.add_argument('--out', default='results_harness_v1')
    p.add_argument('--script-dir', default='slurm_jobs')
    p.add_argument('--workdir',
                   default='/scratch/lz432/crisp_sota/struct-predict')
    p.add_argument('--partition', default='gpu')
    p.add_argument('--gres', default='gpu:1')
    p.add_argument('--cpus', type=int, default=4)
    p.add_argument('--mem', default='24G')
    p.add_argument('--time', default='12:00:00')
    p.add_argument('--model-path', default='')
    p.add_argument('--variant', default='',
                   help='crisp-mode variant (nocawr/jsnap/...)')
    args = p.parse_args()

    os.makedirs(args.script_dir, exist_ok=True)
    n = 0
    for system in args.systems:
        spec = SYSTEMS[system]
        # Ensure the generation cap cannot end the run before the
        # relaxation budget (random mode only relaxes n_random/gen).
        per_gen = {'random': spec.n_random,
                   'fponly': spec.n_random + spec.n_mutants + 5,
                   'crisp': spec.n_random + 2 * spec.n_mutants}
        for potential in args.potentials:
            for mode in args.modes:
                max_gens = max(spec.max_generations,
                               -(-args.budget // per_gen[mode]) + 2)
                mode_label = (f"{mode}-{args.variant}"
                              if args.variant and mode == 'crisp' else mode)
                variant_arg = (f"--variant {args.variant} "
                               if args.variant and mode == 'crisp' else '')
                for seed in args.seeds:
                    tag = f"{system}_{potential}_{mode_label}_s{seed}"
                    extra = ''
                    if args.model_path and potential == 'matpes':
                        extra = f"--model-path {args.model_path}"
                    gres_line = (f"--gres={args.gres}" if args.gres
                                 else "--constraint=skylake")
                    script = TEMPLATE.format(
                        tag=tag, partition=args.partition,
                        gres_line=gres_line, cpus=args.cpus, mem=args.mem,
                        time=args.time, workdir=args.workdir,
                        python=ENV_PYTHON[potential], system=system,
                        potential=potential, mode=mode, seed=seed,
                        budget=args.budget, max_gens=max_gens,
                        out_dir=args.out, variant_arg=variant_arg,
                        extra=extra)
                    path = os.path.join(args.script_dir, f"{tag}.sbatch")
                    with open(path, 'w') as f:
                        f.write(script)
                    n += 1
    print(f"Wrote {n} sbatch scripts to {args.script_dir}/")


if __name__ == '__main__':
    main()
