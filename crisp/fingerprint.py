"""torch_fplib wrapper — the only module that imports torch_fplib.

All other CRISP modules go through this interface for fingerprints,
autograd-based force/stress projection, and structure distances.

Backend: torch_fplib (PyTorch reimplementation of libfp with autograd).
Exact derivatives via autograd replace libfp's analytical dfp/dfpe.
"""

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    torch = None

try:
    import torch_fplib
except ImportError:
    torch_fplib = None

_HAS_TORCH_FPLIB = torch is not None and torch_fplib is not None


def _require_backend() -> None:
    """Raise ImportError with a helpful message if torch_fplib is unavailable."""
    if not _HAS_TORCH_FPLIB:
        raise ImportError(
            "torch_fplib is required but not installed. "
            "Install torch_fplib and ensure it is importable. "
            "CRISP requires torch_fplib for structural fingerprints."
        )


def _fp_dist_hungarian(fp1, fp2, types):
    """Hungarian-matched fingerprint distance (pure numpy + scipy)."""
    from scipy.optimize import linear_sum_assignment

    fp1 = np.asarray(fp1, dtype=np.float64)
    fp2 = np.asarray(fp2, dtype=np.float64)
    types = np.asarray(types)

    nat, lenfp = fp1.shape
    unique_types = sorted(set(types.tolist()))
    fpd = 0.0

    for ityp in unique_types:
        idx = np.where(types == ityp)[0]
        n = len(idx)
        if n == 0:
            continue
        cost = np.zeros((n, n))
        for a in range(n):
            for b in range(n):
                diff = fp1[idx[a]] - fp2[idx[b]]
                cost[a, b] = np.sqrt(np.dot(diff, diff) / lenfp)
        row, col = linear_sum_assignment(cost)
        fpd += cost[row, col].sum()

    return fpd / nat


def _fp_dist_with_assignment(fp1, fp2, types):
    """Hungarian-matched FP distance + atom assignment (pure numpy + scipy)."""
    from scipy.optimize import linear_sum_assignment

    fp1 = np.asarray(fp1, dtype=np.float64)
    fp2 = np.asarray(fp2, dtype=np.float64)
    types = np.asarray(types)

    nat, lenfp = fp1.shape
    unique_types = sorted(set(types.tolist()))
    fpd = 0.0
    assignment = np.arange(nat, dtype=np.int32)

    for ityp in unique_types:
        idx = np.where(types == ityp)[0]
        n = len(idx)
        if n == 0:
            continue
        cost = np.zeros((n, n))
        for a in range(n):
            for b in range(n):
                diff = fp1[idx[a]] - fp2[idx[b]]
                cost[a, b] = np.sqrt(np.dot(diff, diff) / lenfp)
        row, col = linear_sum_assignment(cost)
        fpd += cost[row, col].sum()
        for a, b in zip(row, col):
            assignment[idx[a]] = idx[b]

    return fpd / nat, assignment


# Voigt index mapping: Voigt → (row, col) of 3×3 tensor
_VOIGT_IDX = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]


class FingerprintCalculator:
    """Wrapper around torch_fplib for fingerprints, autograd projections, and distances.

    Parameters
    ----------
    cutoff : float
        Cutoff radius for local environment (Angstrom).
    natx : int
        Maximum neighbors per atom (controls fingerprint dimension).
    orbital : str
        Orbital type: 's' or 'sp'. 'sp' gives 4x more FP components.
    """

    def __init__(self, cutoff: float = 6.0, natx: int = 100,
                 orbital: str = 's'):
        self.cutoff = cutoff
        self.natx = natx
        self.orbital = orbital

    def _check_cell(self, atoms) -> None:
        """Raise ValueError if the cutoff sphere would overflow the natx buffer."""
        from ase.neighborlist import neighbor_list
        i_list = neighbor_list('i', atoms, self.cutoff)
        if len(i_list) == 0:
            return
        counts = np.bincount(i_list, minlength=len(atoms))
        max_neighbors = int(counts.max())
        if max_neighbors + 1 > self.natx:
            raise ValueError(
                f"Too many neighbors for cutoff={self.cutoff:.1f} Å: "
                f"max={max_neighbors+1} (natx={self.natx}), "
                f"cell lengths={[f'{l:.2f}' for l in atoms.cell.lengths()]}"
            )

    def atoms_to_cell(self, atoms) -> tuple:
        """Convert ASE Atoms to cell tuple ``(lat, rxyz, types, znucl)``.

        Returns
        -------
        tuple
            ``(lat, rxyz, types, znucl)`` where:
            - lat: (3, 3) lattice vectors
            - rxyz: (nat, 3) Cartesian positions
            - types: (nat,) 1-indexed type integers
            - znucl: list of atomic numbers in type order
        """
        lat = np.array(atoms.cell)
        rxyz = atoms.get_positions()
        numbers = atoms.get_atomic_numbers()
        znucl = sorted(set(numbers))
        types = np.array([znucl.index(z) + 1 for z in numbers])
        return (lat, rxyz, types, znucl)

    # ── FP computation ────────────────────────────────────────────────

    def get_fingerprints(self, atoms) -> np.ndarray:
        """Compute fingerprints for all atoms.

        Returns
        -------
        np.ndarray
            Shape ``(nat, fp_dim)``.
        """
        _require_backend()
        self._check_cell(atoms)
        cell = self.atoms_to_cell(atoms)
        fp = torch_fplib.get_lfp(cell, cutoff=self.cutoff, natx=self.natx,
                                  orbital=self.orbital)
        return fp.detach().cpu().numpy()

    # ── Autograd force/stress projection (efficient VJP) ──────────────

    def project_forces(self, atoms, dL_dfp: np.ndarray) -> np.ndarray:
        """Project per-atom FP gradient to Cartesian forces via autograd VJP.

        Computes ``F = -J^T · dL_dfp`` where ``J = dfp/dr``, using a single
        backward pass (no full Jacobian materialization).

        Parameters
        ----------
        atoms : ase.Atoms
        dL_dfp : np.ndarray, shape (nat, fp_dim)
            Per-atom loss gradient ``∂L/∂fp_{i,m}``.

        Returns
        -------
        forces : np.ndarray, shape (nat, 3)
        """
        _require_backend()
        self._check_cell(atoms)
        cell = self.atoms_to_cell(atoms)
        lat_np, rxyz_np, types, znucl = cell

        lat_t = torch.tensor(lat_np, dtype=torch.float64)
        rxyz_t = torch.tensor(rxyz_np, dtype=torch.float64, requires_grad=True)

        fp = torch_fplib.get_lfp(
            (lat_t, rxyz_t, types, znucl),
            cutoff=self.cutoff, natx=self.natx, orbital=self.orbital)

        dL_t = torch.tensor(np.asarray(dL_dfp), dtype=torch.float64)
        S = (dL_t * fp).sum()
        grad_rxyz, = torch.autograd.grad(S, rxyz_t)

        return -grad_rxyz.detach().numpy()

    def project_forces_and_stress(self, atoms, dL_dfp: np.ndarray
                                  ) -> Tuple[np.ndarray, np.ndarray]:
        """Project per-atom FP gradient to forces and Voigt stress via autograd.

        Uses strain parametrization: ``cell' = (I + ε) @ cell``, positions from
        scaled coordinates. One forward + one backward pass.

        Parameters
        ----------
        atoms : ase.Atoms
        dL_dfp : np.ndarray, shape (nat, fp_dim)

        Returns
        -------
        forces : np.ndarray, shape (nat, 3)
        stress : np.ndarray, shape (6,) — Voigt [xx, yy, zz, yz, xz, xy]
        """
        _require_backend()
        self._check_cell(atoms)
        cell = self.atoms_to_cell(atoms)
        lat_np, rxyz_np, types, znucl = cell
        vol = atoms.get_volume()

        lat_t = torch.tensor(lat_np, dtype=torch.float64)
        spos = torch.tensor(atoms.get_scaled_positions(), dtype=torch.float64)

        # Symmetric strain variable, applied on the ASE side: with
        # row-vector lattices the deformation x' = (I+eps) x reads
        # lat' = lat @ (I+eps). (The historical (I+strain) @ lat strained
        # the wrong side and carried a sign flip — bias stress was not in
        # the ASE convention, so cell relaxation under bias steered wrong.)
        strain = torch.zeros(3, 3, dtype=torch.float64, requires_grad=True)
        eps_sym = 0.5 * (strain + strain.T)
        lat_s = lat_t @ (torch.eye(3, dtype=torch.float64) + eps_sym)

        # Fractional-coordinate variable for atomic forces
        spos_var = spos.clone().detach().requires_grad_(True)
        rxyz_t = spos_var @ lat_s

        fp = torch_fplib.get_lfp(
            (lat_s, rxyz_t, types, znucl),
            cutoff=self.cutoff, natx=self.natx, orbital=self.orbital)

        dL_t = torch.tensor(np.asarray(dL_dfp), dtype=torch.float64)
        S = (dL_t * fp).sum()

        grad_spos, grad_strain = torch.autograd.grad(S, [spos_var, strain])

        # Forces: convert fractional gradient to Cartesian
        # rxyz = spos @ lat  →  dS/d(rxyz) = dS/d(spos) @ inv(lat^T)
        lat_inv_T = torch.linalg.inv(lat_t.T)
        forces = -(grad_spos @ lat_inv_T).detach().numpy()

        # ASE convention: σ_v = +(1/V) · ∂E/∂ε (tensile positive).
        # eps_sym already symmetrizes the off-diagonal derivative.
        gs = grad_strain.detach().numpy()
        stress_voigt = np.array([gs[a, b] for a, b in _VOIGT_IDX]) / vol

        return forces, stress_voigt

    # ── Backward-compatible Jacobian API ──────────────────────────────
    # These compute the full Jacobian via autograd. Slower than VJP-based
    # project_* methods, but needed for FP-Jacobian mutations in search.py.

    def get_fingerprints_and_jacobian(self, atoms) -> Tuple[np.ndarray, np.ndarray]:
        """Compute fingerprints and their position Jacobian via autograd.

        Returns
        -------
        fp : np.ndarray, shape ``(nat, fp_dim)``
        dfp : np.ndarray, shape ``(nat, nat, 3, fp_dim)``
            ``dfp[i][j][k][m] = d fp_{i,m} / d r_{j,k}``.
        """
        _require_backend()
        self._check_cell(atoms)
        cell = self.atoms_to_cell(atoms)
        lat_np, rxyz_np, types, znucl = cell
        nat = len(rxyz_np)

        lat_t = torch.tensor(lat_np, dtype=torch.float64)
        rxyz_t = torch.tensor(rxyz_np, dtype=torch.float64, requires_grad=True)

        fp = torch_fplib.get_lfp(
            (lat_t, rxyz_t, types, znucl),
            cutoff=self.cutoff, natx=self.natx, orbital=self.orbital)

        fp_np = fp.detach().numpy().copy()
        fp_dim = fp.shape[1]

        # Full Jacobian via torch.autograd.functional.jacobian
        def fp_func(rxyz_flat):
            rxyz_r = rxyz_flat.reshape(nat, 3)
            result = torch_fplib.get_lfp(
                (lat_t, rxyz_r, types, znucl),
                cutoff=self.cutoff, natx=self.natx, orbital=self.orbital)
            return result.reshape(-1)

        # Forward-mode batches over the INPUT dim (3*nat), not the
        # output dim (nat*fp_dim) — 14x less memory at 28 atoms
        # (17 GB -> 1.2 GB peak); results identical to reverse mode
        # to ~1e-16.
        jac = torch.autograd.functional.jacobian(
            fp_func, rxyz_t.reshape(-1), vectorize=True,
            strategy="forward-mode")
        # jac shape: (nat*fp_dim, nat*3)
        # Reshape to (nat, fp_dim, nat, 3) then transpose to (nat, nat, 3, fp_dim)
        dfp = jac.detach().numpy().reshape(nat, fp_dim, nat, 3).transpose(0, 2, 3, 1)

        return fp_np, dfp

    def get_fingerprints_jacobian_strain(self, atoms) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute fingerprints, position Jacobian, and strain derivatives via autograd.

        Returns
        -------
        fp : np.ndarray, shape ``(nat, fp_dim)``
        dfp : np.ndarray, shape ``(nat, nat, 3, fp_dim)``
        dfpe : np.ndarray, shape ``(nat, 6, fp_dim)`` — Voigt [xx, yy, zz, yz, xz, xy]
        """
        _require_backend()
        self._check_cell(atoms)
        cell = self.atoms_to_cell(atoms)
        lat_np, rxyz_np, types, znucl = cell
        nat = len(rxyz_np)

        lat_t = torch.tensor(lat_np, dtype=torch.float64)
        spos = torch.tensor(atoms.get_scaled_positions(), dtype=torch.float64)

        # Position Jacobian (same as get_fingerprints_and_jacobian)
        rxyz_t = torch.tensor(rxyz_np, dtype=torch.float64, requires_grad=True)
        fp = torch_fplib.get_lfp(
            (lat_t, rxyz_t, types, znucl),
            cutoff=self.cutoff, natx=self.natx, orbital=self.orbital)
        fp_np = fp.detach().numpy().copy()
        fp_dim = fp.shape[1]

        def fp_func_pos(rxyz_flat):
            rxyz_r = rxyz_flat.reshape(nat, 3)
            return torch_fplib.get_lfp(
                (lat_t, rxyz_r, types, znucl),
                cutoff=self.cutoff, natx=self.natx, orbital=self.orbital
            ).reshape(-1)

        jac_pos = torch.autograd.functional.jacobian(
            fp_func_pos, rxyz_t.reshape(-1), vectorize=True,
            strategy="forward-mode")
        dfp = jac_pos.detach().numpy().reshape(nat, fp_dim, nat, 3).transpose(0, 2, 3, 1)

        # Strain Jacobian: dfpe[i, v, m] = d fp[i,m] / d strain_voigt[v]
        # Symmetric strain on the ASE side (lat' = lat @ (I+eps)); Voigt
        # shears split across the two off-diagonal elements, matching the
        # corrected project_forces_and_stress convention.
        def fp_func_strain(strain_voigt):
            eps = torch.zeros(3, 3, dtype=torch.float64)
            for iv, (a, b) in enumerate(_VOIGT_IDX):
                if a == b:
                    eps[a, a] = strain_voigt[iv]
                else:
                    eps[a, b] = 0.5 * strain_voigt[iv]
                    eps[b, a] = eps[b, a] + 0.5 * strain_voigt[iv]
            lat_s = lat_t @ (torch.eye(3, dtype=torch.float64) + eps)
            rxyz_s = spos @ lat_s
            return torch_fplib.get_lfp(
                (lat_s, rxyz_s, types, znucl),
                cutoff=self.cutoff, natx=self.natx, orbital=self.orbital
            ).reshape(-1)

        jac_strain = torch.autograd.functional.jacobian(
            fp_func_strain, torch.zeros(6, dtype=torch.float64),
            vectorize=True, strategy="forward-mode")
        # jac_strain shape: (nat*fp_dim, 6) → reshape to (nat, fp_dim, 6) → transpose
        dfpe = jac_strain.detach().numpy().reshape(nat, fp_dim, 6).transpose(0, 2, 1)

        return fp_np, dfp, dfpe

    # ── Distance computation ──────────────────────────────────────────

    def _fp_dist(self, fp1: np.ndarray, fp2: np.ndarray,
                 types: np.ndarray) -> float:
        """Hungarian-matched distance between pre-computed fp arrays."""
        return _fp_dist_hungarian(fp1, fp2, types)

    def get_distance(self, atoms1, atoms2) -> float:
        """Fingerprint distance between two structures (Hungarian-matched).

        Both structures must have the same composition.
        """
        fp1 = self.get_fingerprints(atoms1)
        fp2 = self.get_fingerprints(atoms2)
        types = self.atoms_to_cell(atoms1)[2]
        return _fp_dist_hungarian(fp1, fp2, types)

    def get_distance_and_assignment(self, atoms1, atoms2) -> Tuple[float, np.ndarray]:
        """Fingerprint distance and atom assignment between two structures.

        Returns
        -------
        dist : float
            Hungarian-matched fingerprint distance.
        assignment : np.ndarray
            Atom index mapping from atoms1 to atoms2.
        """
        fp1 = self.get_fingerprints(atoms1)
        fp2 = self.get_fingerprints(atoms2)
        types = self.atoms_to_cell(atoms1)[2]
        return _fp_dist_with_assignment(fp1, fp2, types)

    # ── Pooling ───────────────────────────────────────────────────────

    def pool(self, fp: np.ndarray) -> np.ndarray:
        """Mean-pool atomic fingerprints to a structure-level descriptor."""
        return np.mean(fp, axis=0)

    def pool_with_std(self, fp: np.ndarray) -> np.ndarray:
        """Pool with mean + std for a richer descriptor."""
        return np.concatenate([np.mean(fp, axis=0), np.std(fp, axis=0)])
