# Dependencies

CRISP separates lightweight import/test dependencies from full production
search dependencies.

## Base Package

These are declared in `pyproject.toml`:

- `numpy`
- `scipy`
- `ase`
- `scikit-learn`
- `torch`

Install with:

```bash
python -m pip install -e .
```

## Development Tests

The public smoke tests are written with `unittest` and can run without extra
test tooling:

```bash
python -m unittest discover -s tests
```

For pytest-style workflows:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

## Search Workflows

Full CRISP searches require additional runtime components:

- `torch_fplib`: required by `FingerprintCalculator` for local fingerprints,
  autograd force projection, and stress projection.
- `pyxtal`: required by `CRISPSearch` random crystal generation.
- An ASE-compatible physical calculator. Examples include MACE, MatterSim,
  NequIP, LAMMPS calculators, or VASP-backed workflows.

`torch_fplib` may not be published on PyPI in every environment. Install it
from its upstream source or your local package build, then verify:

```bash
python -c "import torch_fplib; print('torch_fplib ok')"
```

## Optional HPC/DFT Workflow

`crisp.hpc.HPCRelaxer` expects caller-provided SSH, upload, and download
functions plus a remote Slurm environment with the requested calculator stack.
The package does not ship private cluster paths, model weights, POTCAR files,
or scheduler account configuration. Start from
`examples/hpc_search_template.py` and set paths with environment variables or
your own configuration layer.
