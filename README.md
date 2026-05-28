# CRISP

CRISP is an experimental crystal structure prediction toolkit built around
fingerprint-space surrogate models. It combines random crystal generation,
fingerprint-based deduplication, Gaussian-process screening, optional biased
refinement stages, and local or Slurm-backed structure relaxation.

The code is research software. Expect to adapt calculators, model paths, and
cluster settings for your environment before running production searches.

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
