"""GP hyperparameter auto-tuning (opt-in) — marginal-likelihood grid."""

import unittest

import numpy as np


def _make_data(n=120, d=2, true_l=0.2, noise=0.01, seed=5):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, size=(n, d))
    # Draw a smooth function with characteristic scale true_l
    sq = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    K = np.exp(-sq / (2 * true_l ** 2)) + 1e-10 * np.eye(n)
    y = np.linalg.cholesky(K) @ rng.standard_normal(n)
    y += noise * rng.standard_normal(n)
    return X, y


class TestGPAutotune(unittest.TestCase):

    def test_autotune_improves_holdout_rmse(self):
        from crisp.surrogate import ExactGP
        X, y = _make_data()
        Xtr, ytr = X[:90], y[:90]
        Xte, yte = X[90:], y[90:]

        # Fixed default lengthscale 1.0 — badly mismatched to true 0.3
        gp_fixed = ExactGP(length_scale=1.0)
        gp_fixed.train(Xtr, ytr)
        rmse_fixed = np.sqrt(np.mean(
            [(gp_fixed.predict(x)[0] - t) ** 2 for x, t in zip(Xte, yte)]))

        gp_auto = ExactGP(length_scale=1.0, auto_tune=True)
        gp_auto.train(Xtr, ytr)
        rmse_auto = np.sqrt(np.mean(
            [(gp_auto.predict(x)[0] - t) ** 2 for x, t in zip(Xte, yte)]))

        self.assertNotAlmostEqual(gp_auto.length_scale, 1.0)
        self.assertLess(rmse_auto, rmse_fixed * 0.9,
                        f"auto {rmse_auto:.4f} vs fixed {rmse_fixed:.4f}")

    def test_autotune_skipped_below_min_points(self):
        from crisp.surrogate import ExactGP
        X, y = _make_data(n=6)
        gp = ExactGP(length_scale=1.0, auto_tune=True)
        gp.train(X, y)
        self.assertEqual(gp.length_scale, 1.0)


if __name__ == '__main__':
    unittest.main()
