# Changelog

This changelog records the public CRISP development history. The GitHub
repository starts from a curated public-release baseline; earlier private
scratch history is summarized here as development milestones rather than
replayed as synthetic commits.

## Public Git History

### 2026-05-28 - Emphasize differentiable global optimization

- Revised the README to present CRISP as a differentiable global optimization
  toolkit for crystal structure prediction.
- Added a concise summary of the fingerprint-space optimization loop,
  surrogate-guided screening, targeted finishers, and ASE/Slurm relaxation
  backends.

### 2026-05-28 - Initial public CRISP release

- Published the sanitized `crisp` Python package, public tests, setup metadata,
  dependency notes, and an HPC search template.
- Excluded generated benchmark outputs, model weights, private run artifacts,
  scheduler logs, and local scratch scripts from the public branch.
- Verified the public tree with lightweight unit tests, package metadata checks,
  and source-only content review.

## Development Milestones Before Public Release

### v0.4.0 - Differentiable CSP Workflow

- Added fingerprint-targeted finishers for refining promising structures toward
  selected fingerprint-space targets.
- Added cluster-aware reform bias terms for structure cleanup and
  symmetrization-like refinement.
- Added stagnation-aware global search behavior with diverse parent pools,
  larger target distances, strain perturbations, and random restarts.
- Added multi-backend relaxation support for ASE-compatible machine-learning
  potentials and DFT workflows.

### v0.3.x - Fingerprint-Space Mutation and Screening

- Integrated FP-Jacobian mutations into the core search loop.
- Added fingerprint-space momentum from successful parent-child displacement
  directions.
- Added archive-driven target libraries that combine known phases and discovered
  structure centroids.
- Improved surrogate screening with acquisition-driven candidate selection.

### v0.2.x - Local Orchestration and HPC Relaxation

- Split local search orchestration from remote relaxation execution.
- Added checkpoint/resume support for archive and Gaussian-process state.
- Added Slurm-backed relaxation submission, polling, collection, and error
  handling through `crisp.hpc.HPCRelaxer`.
- Added standalone Gaussian-process ASE calculator support for surrogate-only
  workflows.

### v0.1.x - Core CRISP Components

- Implemented random structure generation, fingerprint calculation,
  fingerprint-distance deduplication, archive management, and surrogate-guided
  search primitives.
- Established the public package layout, minimal API surface, and smoke tests
  used for the release branch.
