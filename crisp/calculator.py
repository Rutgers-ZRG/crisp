"""ASE Calculator interface for CRISP.

Combines MLIP forces with fingerprint-space bias forces, operating in
three modes: surrogate-only, hybrid (MLIP + bias), or MLIP-only.
"""

import logging
from typing import List, Optional

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from .bias import BiasPotential
from .fingerprint import FingerprintCalculator
from .projector import ForceProjector

logger = logging.getLogger(__name__)


class CRISPCalculator(Calculator):
    """ASE Calculator that combines MLIP + bias forces.

    F_total = F_MLIP + lambda * F_bias

    where lambda can decay adaptively based on GP uncertainty:
        lambda(f) = lambda_0 * exp(-sigma^2 / sigma_0^2)
    so that high certainty regions receive less bias (trust MLIP more).

    Parameters
    ----------
    mlip_calc : Calculator or None
        Underlying MLIP calculator. Required for 'hybrid' and 'mlip_only'
        modes; can be None for 'surrogate_only'.
    bias : BiasPotential
        The composite bias potential.
    projector : ForceProjector
        Projects FP-space gradients to Cartesian forces.
    fp_calc : FingerprintCalculator
        Fingerprint calculator.
    lambda_0 : float
        Base bias mixing weight.
    sigma_0 : float
        GP sigma scale for adaptive lambda decay.
    mode : str
        ``'surrogate_only'``, ``'hybrid'``, or ``'mlip_only'``.
    """

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, mlip_calc, bias: BiasPotential,
                 projector: ForceProjector,
                 fp_calc: FingerprintCalculator,
                 lambda_0: float = 1.0, sigma_0: float = 0.5,
                 mode: str = "hybrid", **kwargs):
        super().__init__(**kwargs)
        self.mlip_calc = mlip_calc
        self.bias = bias
        self.projector = projector
        self.fp_calc = fp_calc
        self.lambda_0 = lambda_0
        self.sigma_0 = sigma_0
        self.mode = mode

    def calculate(self, atoms=None, properties: Optional[List[str]] = None,
                  system_changes: tuple = all_changes) -> None:
        if properties is None:
            properties = ["energy"]
        super().calculate(atoms, properties, system_changes)

        if self.mode == "mlip_only":
            self._calc_mlip_only()
        elif self.mode == "surrogate_only":
            self._calc_surrogate_only()
        elif self.mode == "hybrid":
            self._calc_hybrid()
        else:
            raise ValueError(f"Unknown mode: {self.mode!r}")

    def _calc_mlip_only(self) -> None:
        """Pure MLIP evaluation."""
        self.atoms.calc = self.mlip_calc
        self.results["energy"] = self.atoms.get_potential_energy()
        self.results["forces"] = self.atoms.get_forces()
        if "stress" in self.implemented_properties:
            try:
                self.results["stress"] = self.atoms.get_stress()
            except Exception:
                self.results["stress"] = np.zeros(6)

    def _calc_surrogate_only(self) -> None:
        """Forces entirely from bias potential — no MLIP calls."""
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

    def _calc_hybrid(self) -> None:
        """MLIP + lambda * bias, with adaptive lambda."""
        # MLIP contribution
        self.atoms.calc = self.mlip_calc
        e_mlip = self.atoms.get_potential_energy()
        f_mlip = self.atoms.get_forces()
        try:
            s_mlip = self.atoms.get_stress()
        except Exception:
            s_mlip = np.zeros(6)

        # Bias contribution
        fp = self.fp_calc.get_fingerprints(self.atoms)
        fp_pooled = self.fp_calc.pool_with_std(fp)
        V, grad_V = self.bias.evaluate_with_grad(fp_pooled)
        f_bias, s_bias = self.projector.compute_forces_and_stress(
            self.atoms, grad_V
        )

        # Adaptive lambda: decay bias in well-known regions
        if self.bias.gp.X_train is not None:
            _, sigma = self.bias.gp.predict(fp_pooled)
            lam = self.lambda_0 * np.exp(-(sigma ** 2) / (self.sigma_0 ** 2))
        else:
            lam = self.lambda_0

        # V is per-atom; projector returns -dV/dx. Scale to extensive.
        nat = len(self.atoms)
        self.results["energy"] = e_mlip + lam * V * nat
        self.results["forces"] = f_mlip + lam * nat * f_bias
        self.results["stress"] = s_mlip + lam * nat * s_bias
