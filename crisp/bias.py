"""Composite bias potential in fingerprint space.

V(f) = mu(f) - kappa * sigma(f) + beta * U_anchor(f) + gamma * U_rep(f)

All terms and their gradients are computed w.r.t. the pooled fingerprint
vector f.  The GP provides mu, sigma, and their gradients; the anchor
and repulsion terms are analytical Gaussian potentials.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np

from .surrogate import ExactGP

logger = logging.getLogger(__name__)


class BiasPotential:
    """Composite bias potential for structure search.

    Parameters
    ----------
    gp : ExactGP
        Trained Gaussian process.
    kappa : float
        Exploration weight in LCB: lower kappa favors exploitation.
    beta : float
        Anchor attraction strength.
    gamma : float
        Repulsion from visited minima.
    rep_width : float
        Width of repulsive Gaussian bumps in FP space.
    rep_height : float
        Height of repulsive Gaussian bumps (eV/atom equivalent).
    anchor_width : float
        Width of anchor attraction wells in FP space.
    """

    def __init__(self, gp: ExactGP, kappa: float = 1.0, beta: float = 0.1,
                 gamma: float = 0.5, rep_width: float = 0.3,
                 rep_height: float = 1.0, anchor_width: float = 0.5):
        self.gp = gp
        self.kappa = kappa
        self.beta = beta
        self.gamma = gamma
        self.rep_width = rep_width
        self.rep_height = rep_height
        self.anchor_width = anchor_width
        self.anchors: List[np.ndarray] = []
        self.repulsion_centers: List[np.ndarray] = []

    def set_anchors(self, fps_pooled: List[np.ndarray]) -> None:
        """Set anchor fingerprints (diverse low-energy minima)."""
        self.anchors = [np.asarray(f, dtype=np.float64) for f in fps_pooled]

    def set_repulsion_centers(self, fps_pooled: List[np.ndarray]) -> None:
        """Set repulsion centers (all visited minima)."""
        self.repulsion_centers = [np.asarray(f, dtype=np.float64) for f in fps_pooled]

    def evaluate(self, f_pooled: np.ndarray) -> float:
        """Evaluate V(f) at a pooled fingerprint.

        Parameters
        ----------
        f_pooled : np.ndarray, shape (fp_dim,)

        Returns
        -------
        float
            Bias potential value.
        """
        V, _ = self.evaluate_with_grad(f_pooled)
        return V

    def evaluate_with_grad(self, f_pooled: np.ndarray) -> Tuple[float, np.ndarray]:
        """Evaluate V(f) and its gradient.

        Returns
        -------
        V : float
        grad_V : np.ndarray, shape (fp_dim,)
        """
        f = np.asarray(f_pooled, dtype=np.float64)
        d = len(f)

        V = 0.0
        grad_V = np.zeros(d)

        # GP terms: mu - kappa * sigma
        if self.gp.X_train is not None:
            mu, sigma, grad_mu, grad_sigma = self.gp.predict_with_grad(f)
            V += mu - self.kappa * sigma
            grad_V += grad_mu - self.kappa * grad_sigma

        # Anchor attraction
        if self.beta > 0 and self.anchors:
            U_anc, grad_anc = self._anchor_potential(f)
            V += self.beta * U_anc
            grad_V += self.beta * grad_anc

        # Repulsion from visited
        if self.gamma > 0 and self.repulsion_centers:
            U_rep, grad_rep = self._repulsion_potential(f)
            V += self.gamma * U_rep
            grad_V += self.gamma * grad_rep

        return V, grad_V

    def _anchor_potential(self, f: np.ndarray) -> Tuple[float, np.ndarray]:
        """Soft-min attraction toward anchor structures.

        U_anchor = -log( sum_a exp(-||f - f_a||^2 / (2 w^2)) )

        This is minimized when f is close to any anchor.

        Returns (U, grad_U).
        """
        w2 = self.anchor_width ** 2
        exps = []
        diffs = []
        for fa in self.anchors:
            diff = f - fa
            sq = np.sum(diff ** 2)
            exps.append(np.exp(-sq / (2.0 * w2)))
            diffs.append(diff)

        Z = sum(exps)
        if Z < 1e-100:
            return 0.0, np.zeros_like(f)

        U = -np.log(Z)

        # grad U = -1/Z * sum_a exp_a * (-diff_a / w^2) = (1/Z) * sum(exp_a * diff_a) / w^2
        grad_U = np.zeros_like(f)
        for exp_a, diff_a in zip(exps, diffs):
            grad_U += exp_a * diff_a
        grad_U /= Z * w2

        return U, grad_U

    def _repulsion_potential(self, f: np.ndarray) -> Tuple[float, np.ndarray]:
        """Gaussian repulsion bumps from visited minima.

        U_rep = sum_c h * exp(-||f - f_c||^2 / (2 w^2))

        Returns (U, grad_U).
        """
        w2 = self.rep_width ** 2
        h = self.rep_height
        U = 0.0
        grad_U = np.zeros_like(f)

        for fc in self.repulsion_centers:
            diff = f - fc
            sq = np.sum(diff ** 2)
            g = h * np.exp(-sq / (2.0 * w2))
            U += g
            grad_U += g * (-diff / w2)

        return U, grad_U
