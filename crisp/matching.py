"""Atom matching backends for FP-targeted finishers.

Provides Hungarian (default, proven) and Sinkhorn OT (experimental)
matching between per-atom fingerprints of a candidate and a target.

Both backends return:
- loss: scalar matching cost
- dL_dfp: (nat, fp_dim) gradient of loss w.r.t. candidate per-atom FPs

Matching is type-aware: atoms are matched only within the same species.
"""

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _logsumexp_rows(log_M: np.ndarray) -> np.ndarray:
    """Log-sum-exp along columns (reduce axis=1) for numerical stability."""
    max_vals = np.max(log_M, axis=1, keepdims=True)
    return (max_vals.squeeze() +
            np.log(np.sum(np.exp(log_M - max_vals), axis=1)))


def _logsumexp_cols(log_M: np.ndarray) -> np.ndarray:
    """Log-sum-exp along rows (reduce axis=0) for numerical stability."""
    max_vals = np.max(log_M, axis=0, keepdims=True)
    return (max_vals.squeeze() +
            np.log(np.sum(np.exp(log_M - max_vals), axis=0)))


def hungarian_match(fp: np.ndarray, fp_target: np.ndarray,
                    types: np.ndarray,
                    types_target: np.ndarray = None) -> Tuple[float, np.ndarray]:
    """Hungarian (exact) per-atom FP matching.

    Matches atoms within each species block using the Hungarian algorithm
    via scipy.optimize.linear_sum_assignment, then computes the MSE loss
    and its gradient.

    Parameters
    ----------
    fp : np.ndarray, shape (nat, fp_dim)
        Current structure per-atom fingerprints.
    fp_target : np.ndarray, shape (nat_target, fp_dim)
        Target structure per-atom fingerprints.
    types : np.ndarray, shape (nat,)
        1-indexed type array for the candidate.
    types_target : np.ndarray, shape (nat_target,) or None
        1-indexed type array for the target. If None, uses ``types``
        (assumes same ordering — legacy behavior).

    Returns
    -------
    loss : float
        Mean squared FP distance under optimal assignment.
    dL_dfp : np.ndarray, shape (nat, fp_dim)
        Gradient ∂L/∂fp_i = 2(fp_i - fp_target_{a(i)}) / N.
    """
    from scipy.optimize import linear_sum_assignment

    if types_target is None:
        types_target = types

    nat, fp_dim = fp.shape
    dL_dfp = np.zeros_like(fp)
    total_cost = 0.0

    unique_types = np.unique(types)
    for t in unique_types:
        idx = np.where(types == t)[0]
        idx_tgt = np.where(types_target == t)[0]
        n_t = len(idx)
        n_tgt = len(idx_tgt)
        if n_t == 0 or n_tgt == 0:
            continue
        if n_t != n_tgt:
            logger.warning("Species %d count mismatch: candidate %d vs target %d",
                           t, n_t, n_tgt)
            continue

        fp_t = fp[idx]               # (n_t, fp_dim)
        fp_tgt_t = fp_target[idx_tgt]  # (n_tgt, fp_dim)

        # Cost matrix: C[i,j] = ||fp_t[i] - fp_tgt_t[j]||^2
        diff = fp_t[:, None, :] - fp_tgt_t[None, :, :]  # (n_t, n_t, fp_dim)
        cost = np.sum(diff ** 2, axis=2)  # (n_t, n_t)

        row_ind, col_ind = linear_sum_assignment(cost)

        for ri, ci in zip(row_ind, col_ind):
            atom_idx = idx[ri]
            residual = fp[atom_idx] - fp_target[idx_tgt[ci]]
            total_cost += np.sum(residual ** 2)
            dL_dfp[atom_idx] = 2.0 * residual / nat

    loss = total_cost / nat
    return loss, dL_dfp


def sinkhorn_match(fp: np.ndarray, fp_target: np.ndarray,
                   types: np.ndarray,
                   types_target: np.ndarray = None,
                   tau: float = 0.05,
                   n_iters: int = 50) -> Tuple[float, np.ndarray]:
    """Sinkhorn OT (soft) per-atom FP matching — EXPERIMENTAL.

    Differentiable alternative to Hungarian. Uses entropic regularization
    to compute a soft transport plan, then derives gradients from it.

    Parameters
    ----------
    fp : np.ndarray, shape (nat, fp_dim)
        Current structure per-atom fingerprints.
    fp_target : np.ndarray, shape (nat_target, fp_dim)
        Target structure per-atom fingerprints.
    types : np.ndarray, shape (nat,)
        1-indexed type array for the candidate.
    types_target : np.ndarray, shape (nat_target,) or None
        1-indexed type array for the target. If None, uses ``types``.
    tau : float
        Entropic regularization parameter. Smaller → closer to Hungarian.
    n_iters : int
        Number of Sinkhorn iterations.

    Returns
    -------
    loss : float
        Sinkhorn OT cost.
    dL_dfp : np.ndarray, shape (nat, fp_dim)
        Gradient ∂L/∂fp_i = 2 Σ_j P_{ij} (fp_i - fp_target_j).
    """
    if types_target is None:
        types_target = types

    nat, fp_dim = fp.shape
    dL_dfp = np.zeros_like(fp)
    total_cost = 0.0

    unique_types = np.unique(types)
    for t in unique_types:
        idx = np.where(types == t)[0]
        idx_tgt = np.where(types_target == t)[0]
        n_t = len(idx)
        n_tgt = len(idx_tgt)
        if n_t == 0 or n_tgt == 0:
            continue
        if n_t != n_tgt:
            logger.warning("Species %d count mismatch: candidate %d vs target %d",
                           t, n_t, n_tgt)
            continue

        fp_t = fp[idx]
        fp_tgt_t = fp_target[idx_tgt]

        # Cost matrix
        diff = fp_t[:, None, :] - fp_tgt_t[None, :, :]  # (n_t, n_t, fp_dim)
        C = np.sum(diff ** 2, axis=2)  # (n_t, n_t)

        # Sinkhorn iterations: produce doubly-stochastic P with marginals = 1/n_t
        # Using log-domain for numerical stability
        log_K = -C / tau
        log_u = np.zeros(n_t)
        log_v = np.zeros(n_t)
        for _ in range(n_iters):
            log_u = -np.log(n_t) - _logsumexp_rows(log_K + log_v[None, :])
            log_v = -np.log(n_t) - _logsumexp_cols(log_K + log_u[:, None])

        # Transport plan P = diag(exp(u)) @ K @ diag(exp(v))
        log_P = log_u[:, None] + log_K + log_v[None, :]
        P = np.exp(log_P)
        # P rows and cols should each sum to 1/n_t

        # OT cost: Σ_{ij} P_{ij} C_{ij}, then scale to match Hungarian convention
        # Hungarian returns (1/N) Σ_i ||fp_i - fp_target_{a(i)}||^2
        # With perfect P → each row has one entry = 1/n_t, so cost = (1/n_t) Σ_i ...
        # Scale by n_t to get sum of matched distances, then divide by nat below
        cost_t = np.sum(P * C) * n_t
        total_cost += cost_t

        # Gradient: ∂L/∂fp_i = (2·n_t/nat) Σ_j P_{ij} (fp_i - fp_tgt_j)
        for i_local in range(n_t):
            atom_idx = idx[i_local]
            grad = 2.0 * n_t * np.sum(P[i_local, :, None] * diff[i_local], axis=0)
            dL_dfp[atom_idx] = grad / nat

    loss = total_cost / nat
    return loss, dL_dfp
