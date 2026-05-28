# CRISP

CRISP is an experimental toolkit for differentiable global optimization in
crystal structure prediction. It searches crystal-structure space by combining
random structure generation, fingerprint-space surrogate modeling,
gradient-based fingerprint projections, targeted refinement, and local or
Slurm-backed structure relaxation.

The central idea is to make global crystal structure prediction more
sample-efficient than pure random search plus local relaxation. CRISP represents
structures through differentiable fingerprints, trains Gaussian-process
surrogates on relaxed candidates, and uses those models to steer new structures
toward promising low-enthalpy regions while preserving compatibility with
standard ASE calculators.

The code is research software. Expect to adapt calculators, model paths, and
cluster settings for your environment before running production searches.

## Core Ideas

- Differentiable fingerprint-space optimization: map atomic and cell degrees of
  freedom into structure fingerprints, then use fingerprint gradients to guide
  candidate refinement.
- Global structure prediction loop: generate diverse candidates, relax them,
  deduplicate by fingerprint distance, update an archive, and propose the next
  generation from surrogate-guided search.
- Surrogate-assisted screening: use Gaussian-process models and acquisition
  scores to prioritize expensive relaxations.
- Targeted finishers and bias terms: refine promising structures with
  fingerprint-targeted optimization and optional cluster-aware reform steps.
- Calculator-agnostic relaxation: connect to ASE-compatible MLIPs or DFT
  backends locally, or dispatch relaxations through Slurm.

## What Is Included

- `crisp/`: the Python package.
- `tests/`: lightweight public tests that avoid private models and HPC access.
- `examples/`: templates for configuring local/HPC runs without hardcoded
  private paths.
- `docs/`: release notes and dependency/setup guidance.

Generated benchmark outputs, model weights, VASP/POSCAR files, scheduler logs,
and private scratch scripts are intentionally excluded from the public branch.

## Installation

Create an environment with Python 3.10 or newer, then install the package in
editable mode:

```bash
python -m pip install -e ".[dev]"
```

CRISP's fingerprint calculations require `torch_fplib`, which may need to be
installed from its source repository or local package depending on how you
obtained it. See [docs/dependencies.md](docs/dependencies.md) for the full
dependency matrix.

## Quick Verification

The public smoke tests do not require `torch_fplib`, MLIP model weights, VASP,
or Slurm:

```bash
python -m unittest discover -s tests
```

Full search workflows additionally require:

- `torch_fplib` for local fingerprints and autograd projections.
- `pyxtal` for random crystal generation.
- An ASE-compatible calculator, such as MACE, MatterSim, NequIP, or VASP.
- Optional SSH/Slurm access for `crisp.hpc.HPCRelaxer`.

## Minimal API Sketch

```python
from crisp import CRISPSearch, FingerprintCalculator

fp_calc = FingerprintCalculator(cutoff=6.0, natx=100, orbital="s")

search = CRISPSearch(
    mlip_calc_factory=my_ase_calculator_factory,
    fp_calc=fp_calc,
    composition={"Si": 8},
    n_random=20,
    max_generations=3,
)

archive = search.run()
```

Replace `my_ase_calculator_factory` with a function that returns a fresh ASE
calculator for each relaxation. For Slurm usage, start from
[examples/hpc_search_template.py](examples/hpc_search_template.py).

## License

MIT. See [LICENSE](LICENSE).
