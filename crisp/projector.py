"""Project fingerprint-space gradients to Cartesian forces and stress.

For mean-pooled fingerprints f_m = (1/N) Σ_i fp_{i,m}:

    F_{j,k} = -(1/N) Σ_{i,m} dfp[i,j,k,m] · ∂V/∂f_m

For pool_with_std, the pooled descriptor is [mean, std] and the full
chain rule applies:

    F_{j,k} = -Σ_m [ g_mean[m] · (1/N) Σ_i dfp[i,j,k,m]
                    + g_std[m]  · Σ_i w[i,m] · dfp[i,j,k,m] ]

where w[i,m] = (fp_{i,m} - mean_m) / (N · std_m) is the per-atom
weight from ∂std_m/∂fp_{i,m}.
"""

import logging
from typing import Tuple

import numpy as np

from .fingerprint import FingerprintCalculator

logger = logging.getLogger(__name__)


class ForceProjector:
    """Map fingerprint-space gradients to Cartesian forces and stress.

    Parameters
    ----------
    fp_calc : FingerprintCalculator
        Fingerprint calculator (used to obtain Jacobians).
    """

    def __init__(self, fp_calc: FingerprintCalculator):
        self.fp_calc = fp_calc

    def _effective_weights(self, fp: np.ndarray,
                           grad_V_pooled: np.ndarray) -> np.ndarray:
        """Compute per-atom effective gradient weights for the projection.

        For mean-only (len(grad_V) == fp_dim):
            w[i,m] = (1/N) · g[m]          (uniform)

        For mean+std (len(grad_V) == 2*fp_dim):
            w[i,m] = (1/N) · g_mean[m]
                    + (fp[i,m] - mean[m]) / (N · std[m]) · g_std[m]

        Parameters
        ----------
        fp : np.ndarray, shape (nat, fp_dim)
            Per-atom fingerprints.
        grad_V_pooled : np.ndarray, shape (fp_dim,) or (2*fp_dim,)
            Gradient of V w.r.t. pooled descriptor.

        Returns
        -------
        w : np.ndarray, shape (nat, fp_dim)
            Effective gradient weight per atom per FP component.
        """
        nat, fp_dim = fp.shape

        if grad_V_pooled.shape[0] == fp_dim:
            # Mean-only: uniform weights
            return np.broadcast_to(grad_V_pooled / nat, (nat, fp_dim)).copy()

        # Mean + std pooling
        g_mean = grad_V_pooled[:fp_dim]
        g_std = grad_V_pooled[fp_dim:]

        mean_fp = np.mean(fp, axis=0)       # (fp_dim,)
        std_fp = np.std(fp, axis=0)         # (fp_dim,)
        # Guard against zero std (e.g. single-atom or identical FPs)
        std_safe = np.where(std_fp > 1e-20, std_fp, 1.0)

        # w[i,m] = g_mean[m]/N + g_std[m] · (fp[i,m] - mean[m]) / (N · std[m])
        w = g_mean[np.newaxis, :] / nat + \
            g_std[np.newaxis, :] * (fp - mean_fp[np.newaxis, :]) / (nat * std_safe[np.newaxis, :])

        return w

    def compute_forces(self, atoms, grad_V_pooled: np.ndarray) -> np.ndarray:
        """Compute Cartesian forces from a pooled-FP gradient.

        Supports both mean-only and mean+std pooled gradients via the
        full chain rule through the pooling operation.

        Parameters
        ----------
        atoms : ase.Atoms
            Current atomic configuration.
        grad_V_pooled : np.ndarray, shape (fp_dim,) or (2*fp_dim,)
            Gradient of V w.r.t. pooled fingerprint.

        Returns
        -------
        forces : np.ndarray, shape (nat, 3)
        """
        fp = self.fp_calc.get_fingerprints(atoms)
        w = self._effective_weights(fp, grad_V_pooled)
        # w[i,m] is the effective per-atom dL/dfp — pass to autograd VJP
        return self.fp_calc.project_forces(atoms, w)

    def compute_forces_and_stress(self, atoms,
                                  grad_V_pooled: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Cartesian forces and Voigt stress from a pooled-FP gradient.

        Supports both mean-only and mean+std pooled gradients.

        Parameters
        ----------
        atoms : ase.Atoms
            Current atomic configuration.
        grad_V_pooled : np.ndarray, shape (fp_dim,) or (2*fp_dim,)
            Gradient of V w.r.t. pooled fingerprint.

        Returns
        -------
        forces : np.ndarray, shape (nat, 3)
        stress : np.ndarray, shape (6,)
            Voigt-order stress [xx, yy, zz, yz, xz, xy].
        """
        fp = self.fp_calc.get_fingerprints(atoms)
        w = self._effective_weights(fp, grad_V_pooled)
        # w[i,m] is the effective per-atom dL/dfp — pass to autograd VJP
        return self.fp_calc.project_forces_and_stress(atoms, w)
