"""Standalone GP-only ASE Calculator for surrogate flow.

Extracts the surrogate-only logic from CRISPCalculator into a lightweight
calculator with no MLIP dependency.  Used for local surrogate flow where
structures are explored in fingerprint space before being sent to HPC
for MLIP/DFT relaxation.
"""

import logging
from typing import List, Optional

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from .bias import BiasPotential
from .fingerprint import FingerprintCalculator
from .projector import ForceProjector

logger = logging.getLogger(__name__)


class GPCalculator(Calculator):
    """ASE Calculator: energy/forces/stress from GP+bias via libfp Jacobians.

    This is functionally identical to ``CRISPCalculator(mode='surrogate_only')``
    but has no MLIP dependency, making it suitable for local-only workflows
    where MLIP relaxation happens remotely on HPC.

    Parameters
    ----------
    bias : BiasPotential
        The composite bias potential (GP + anchors + repulsion).
    projector : ForceProjector
        Projects FP-space gradients to Cartesian forces and stress.
    fp_calc : FingerprintCalculator
        Fingerprint calculator.
    """

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, bias: BiasPotential, projector: ForceProjector,
                 fp_calc: FingerprintCalculator, **kwargs):
        super().__init__(**kwargs)
        self.bias = bias
        self.projector = projector
        self.fp_calc = fp_calc

    def calculate(self, atoms=None, properties: Optional[List[str]] = None,
                  system_changes: tuple = all_changes) -> None:
        if properties is None:
            properties = ["energy"]
        super().calculate(atoms, properties, system_changes)

        fp = self.fp_calc.get_fingerprints(self.atoms)
        fp_pooled = self.fp_calc.pool_with_std(fp)
        V, grad_V = self.bias.evaluate_with_grad(fp_pooled)

        forces, stress = self.projector.compute_forces_and_stress(
            self.atoms, grad_V
        )

        # V is per-atom; projector returns -dV/dx (per-atom gradient).
        # Energy is extensive: E = V * N, so F = -dE/dx = N * (-dV/dx).
        nat = len(self.atoms)
        self.results["energy"] = V * nat
        self.results["forces"] = nat * forces
        self.results["stress"] = nat * stress
