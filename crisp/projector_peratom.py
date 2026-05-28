"""Per-atom fingerprint-space gradient projection to Cartesian forces and stress.

Unlike the pooled projector (projector.py), this operates on per-atom FP
gradients directly — no mean-pooling. Used by FP-targeted finishers where
the loss is defined on individual atomic fingerprints (e.g., Hungarian or
Sinkhorn OT matching).

Chain rule:
    F_{j,k} = -Σ_{i,m} dL/dfp_{i,m} · dfp[i,j,k,m]

Stress (Voigt):
    σ_v = -(1/V) Σ_{i,m} dL/dfp_{i,m} · dfpe[i,v,m]

where dfp[i,j,k,m] = ∂fp_{i,m}/∂r_{j,k} and dfpe[i,v,m] = ∂fp_{i,m}/∂ε_v.
"""

import numpy as np


def project_peratom_forces(dfp: np.ndarray, dL_dfp: np.ndarray) -> np.ndarray:
    """Project per-atom FP gradients to Cartesian forces.

    Parameters
    ----------
    dfp : np.ndarray, shape (nat, nat, 3, fp_dim)
        Position Jacobian from libfp: dfp[i,j,k,m] = ∂fp_{i,m}/∂r_{j,k}.
    dL_dfp : np.ndarray, shape (nat, fp_dim)
        Per-atom loss gradient: ∂L/∂fp_{i,m}.

    Returns
    -------
    forces : np.ndarray, shape (nat, 3)
        Cartesian forces F = -J^T ∇_fp L.
    """
    return -np.einsum("im,ijkm->jk", dL_dfp, dfp, optimize=True)


def project_peratom_stress(dfpe: np.ndarray, dL_dfp: np.ndarray,
                           volume: float) -> np.ndarray:
    """Project per-atom FP gradients to Voigt stress.

    Parameters
    ----------
    dfpe : np.ndarray, shape (nat, 6, fp_dim)
        Strain derivative from libfp: dfpe[i,v,m] = ∂fp_{i,m}/∂ε_v.
    dL_dfp : np.ndarray, shape (nat, fp_dim)
        Per-atom loss gradient: ∂L/∂fp_{i,m}.
    volume : float
        Cell volume in Å³.

    Returns
    -------
    stress : np.ndarray, shape (6,)
        Voigt-order stress [xx, yy, zz, yz, xz, xy].
    """
    return -(1.0 / volume) * np.einsum("im,ivm->v", dL_dfp, dfpe, optimize=True)


def project_peratom_forces_and_stress(
    dfp: np.ndarray, dfpe: np.ndarray, dL_dfp: np.ndarray, volume: float
) -> tuple:
    """Project per-atom FP gradients to both forces and stress.

    Returns
    -------
    forces : np.ndarray, shape (nat, 3)
    stress : np.ndarray, shape (6,)
    """
    forces = project_peratom_forces(dfp, dL_dfp)
    stress = project_peratom_stress(dfpe, dL_dfp, volume)
    return forces, stress
