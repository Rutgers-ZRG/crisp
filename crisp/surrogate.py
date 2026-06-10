"""Exact Gaussian Process on pooled fingerprints.

Predicts energy/enthalpy mu(f) and uncertainty sigma(f), with analytical
gradients for the bias potential.  Numpy-only (no sklearn, no GPy).
"""

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ExactGP:
    """Exact GP regression on pooled structural fingerprints.

    Parameters
    ----------
    kernel : str
        Kernel type. Currently only ``'rbf'`` is supported.
    length_scale : float
        RBF length scale.
    signal_var : float
        Signal variance (output scale) sigma_f^2.
    noise : float
        Observation noise / Tikhonov regularization sigma_n^2.
    normalize_y : bool
        If True, subtract mean and divide by std before training.
    """

    def __init__(self, kernel: str = "rbf", length_scale: float = 1.0,
                 signal_var: float = 1.0, noise: float = 1e-3,
                 normalize_y: bool = True, auto_tune: bool = False,
                 auto_tune_min_points: int = 10):
        if kernel != "rbf":
            raise ValueError(f"Unsupported kernel: {kernel!r}. Use 'rbf'.")
        self.kernel = kernel
        self.length_scale = length_scale
        self.signal_var = signal_var
        self.noise = noise
        self.normalize_y = normalize_y
        self.auto_tune = auto_tune
        self.auto_tune_min_points = auto_tune_min_points

        # Set after training
        self.X_train: Optional[np.ndarray] = None  # (N, d)
        self.y_train: Optional[np.ndarray] = None   # (N,)  (normalized)
        self.K_inv: Optional[np.ndarray] = None      # (N, N)
        self.alpha: Optional[np.ndarray] = None       # (N,) = K_inv @ y
        self._y_mean: float = 0.0
        self._y_std: float = 1.0

    # ------------------------------------------------------------------
    # Kernel
    # ------------------------------------------------------------------

    def _rbf(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Scalar RBF kernel: k(x1, x2) = sf2 * exp(-||x1-x2||^2 / (2 l^2))."""
        sq = np.sum((x1 - x2) ** 2)
        return self.signal_var * np.exp(-sq / (2.0 * self.length_scale ** 2))

    def _kernel_matrix(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Kernel matrix K[i, j] = k(X1[i], X2[j]). Shape (N1, N2)."""
        sq = np.sum((X1[:, np.newaxis, :] - X2[np.newaxis, :, :]) ** 2, axis=2)
        return self.signal_var * np.exp(-sq / (2.0 * self.length_scale ** 2))

    def _kernel_vec(self, x: np.ndarray, X: np.ndarray) -> np.ndarray:
        """k(x, X[i]) for all training points. Returns (N,)."""
        sq = np.sum((x - X) ** 2, axis=1)
        return self.signal_var * np.exp(-sq / (2.0 * self.length_scale ** 2))

    def _kernel_vec_grad(self, x: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Kernel vector and its gradient w.r.t. x.

        Returns
        -------
        k_vec : np.ndarray, shape (N,)
        dk_vec : np.ndarray, shape (N, d)
            dk_vec[i] = dk(x, X[i]) / dx
        """
        diff = x - X           # (N, d)
        sq = np.sum(diff ** 2, axis=1)  # (N,)
        k_vec = self.signal_var * np.exp(-sq / (2.0 * self.length_scale ** 2))
        # dk/dx = -k * (x - x_i) / l^2
        dk_vec = -k_vec[:, np.newaxis] * diff / (self.length_scale ** 2)  # (N, d)
        return k_vec, dk_vec

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the GP.

        Parameters
        ----------
        X : np.ndarray, shape (N, d)
            Pooled fingerprints.
        y : np.ndarray, shape (N,)
            Target values (energy or enthalpy per atom).
        """
        N = X.shape[0]
        if N == 0:
            logger.warning("ExactGP.train called with 0 data points")
            return

        self.X_train = np.array(X, dtype=np.float64)

        # Normalize targets
        if self.normalize_y and N > 1:
            self._y_mean = np.mean(y)
            self._y_std = np.std(y)
            if self._y_std < 1e-12:
                self._y_std = 1.0
            self.y_train = (y - self._y_mean) / self._y_std
        else:
            self._y_mean = 0.0
            self._y_std = 1.0
            self.y_train = np.array(y, dtype=np.float64)

        # Optional hyperparameter selection by log marginal likelihood
        # over a data-scaled grid (median pairwise distance heuristic).
        if self.auto_tune and N >= self.auto_tune_min_points:
            self._tune_hyperparameters()

        # Kernel matrix + regularization
        K = self._kernel_matrix(self.X_train, self.X_train)
        K += self.noise * np.eye(N)

        # Invert via Cholesky for numerical stability
        try:
            L = np.linalg.cholesky(K)
            self.alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y_train))
            L_inv = np.linalg.solve(L, np.eye(N))
            self.K_inv = L_inv.T @ L_inv
        except np.linalg.LinAlgError:
            logger.warning("Cholesky failed, falling back to direct inverse")
            self.K_inv = np.linalg.inv(K)
            self.alpha = self.K_inv @ self.y_train

        logger.info("GP trained on %d points (l=%.3f, sf2=%.3f, sn2=%.3e)",
                     N, self.length_scale, self.signal_var, self.noise)

    def _log_marginal_likelihood(self, length_scale: float,
                                 noise: float) -> float:
        """Log marginal likelihood of the (normalized) training targets."""
        N = self.X_train.shape[0]
        sq = np.sum((self.X_train[:, None, :] -
                     self.X_train[None, :, :]) ** 2, axis=2)
        K = self.signal_var * np.exp(-sq / (2.0 * length_scale ** 2))
        K += noise * np.eye(N)
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return -np.inf
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y_train))
        return float(-0.5 * self.y_train @ alpha
                     - np.sum(np.log(np.diag(L)))
                     - 0.5 * N * np.log(2.0 * np.pi))

    def _tune_hyperparameters(self) -> None:
        """Grid-select length_scale (and noise) by marginal likelihood.

        The grid is scaled to the data: multiples of the median pairwise
        distance, which adapts across systems with very different pooled-
        fingerprint scales (a fixed l=1.0 cannot).
        """
        sq = np.sum((self.X_train[:, None, :] -
                     self.X_train[None, :, :]) ** 2, axis=2)
        d = np.sqrt(sq[np.triu_indices_from(sq, k=1)])
        d = d[d > 1e-12]
        if d.size == 0:
            return
        med = float(np.median(d))
        ls_grid = med * np.array([0.125, 0.25, 0.5, 1.0, 2.0, 4.0])
        noise_grid = [1e-4, 1e-3, 1e-2]

        best = (-np.inf, self.length_scale, self.noise)
        for ls in ls_grid:
            for nz in noise_grid:
                lml = self._log_marginal_likelihood(ls, nz)
                if lml > best[0]:
                    best = (lml, float(ls), float(nz))
        if np.isfinite(best[0]):
            self.length_scale, self.noise = best[1], best[2]
            logger.info("GP auto-tune: l=%.4f noise=%.0e (lml=%.2f, "
                        "median d=%.4f)", self.length_scale, self.noise,
                        best[0], med)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, x: np.ndarray) -> Tuple[float, float]:
        """Predict mean and std at a single point.

        Parameters
        ----------
        x : np.ndarray, shape (d,)

        Returns
        -------
        mu : float
            Predicted mean (in original units).
        sigma : float
            Predicted standard deviation (in original units).
        """
        if self.X_train is None:
            raise RuntimeError("GP not trained yet. Call train() first.")

        k_star = self._kernel_vec(x, self.X_train)
        mu_norm = k_star @ self.alpha
        v = self.K_inv @ k_star
        var = self.signal_var - k_star @ v
        sigma_norm = np.sqrt(max(var, 0.0))

        mu = mu_norm * self._y_std + self._y_mean
        sigma = sigma_norm * self._y_std
        return mu, sigma

    def predict_with_grad(self, x: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """Predict mean, std, and their gradients w.r.t. x.

        Returns
        -------
        mu : float
        sigma : float
        grad_mu : np.ndarray, shape (d,)
            d mu / d x
        grad_sigma : np.ndarray, shape (d,)
            d sigma / d x
        """
        if self.X_train is None:
            raise RuntimeError("GP not trained yet. Call train() first.")

        k_star, dk_star = self._kernel_vec_grad(x, self.X_train)  # (N,), (N, d)

        # Mean and its gradient
        mu_norm = k_star @ self.alpha
        grad_mu_norm = dk_star.T @ self.alpha  # (d,)

        # Variance and its gradient
        v = self.K_inv @ k_star  # (N,)
        var = self.signal_var - k_star @ v
        sigma_norm = np.sqrt(max(var, 1e-20))

        # d(var)/dx = -2 * dk_star^T @ K_inv @ k_star
        #           = -2 * dk_star^T @ v
        grad_var = -2.0 * dk_star.T @ v  # (d,)
        grad_sigma_norm = grad_var / (2.0 * sigma_norm)

        # Transform back to original scale
        mu = mu_norm * self._y_std + self._y_mean
        sigma = sigma_norm * self._y_std
        grad_mu = grad_mu_norm * self._y_std
        grad_sigma = grad_sigma_norm * self._y_std

        return mu, sigma, grad_mu, grad_sigma
