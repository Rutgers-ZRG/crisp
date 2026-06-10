"""CAWR correctness tests.

The CAWR loss is L = sum_c sum_{i in c} ||fp_i - mu_c||^2 with mu_c the
cluster mean. Because sum_{j in c} (fp_j - mu_c) = 0, the exact gradient
is dL/dfp_i = 2 (fp_i - mu_c) — the mu_c dependence cancels. Any
n_c-dependent prefactor distorts the bias direction across clusters.
"""

import unittest

import numpy as np


def _loss(fp, labels):
    L = 0.0
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        mu = fp[idx].mean(axis=0)
        L += float(((fp[idx] - mu) ** 2).sum())
    return L


class TestCAWRGradient(unittest.TestCase):

    def test_loss_grad_matches_finite_difference(self):
        from crisp.cawr import cawr_loss_grad

        rng = np.random.default_rng(0)
        fp = rng.standard_normal((10, 6))
        # Unequal cluster sizes — the case that exposes an n_c prefactor
        labels = np.array([0] * 3 + [1] * 7)

        loss, grad = cawr_loss_grad(fp, labels)
        self.assertAlmostEqual(loss, _loss(fp, labels), places=10)

        eps = 1e-6
        fd = np.zeros_like(fp)
        for i in range(fp.shape[0]):
            for k in range(fp.shape[1]):
                fp_p = fp.copy()
                fp_p[i, k] += eps
                fp_m = fp.copy()
                fp_m[i, k] -= eps
                fd[i, k] = (_loss(fp_p, labels) - _loss(fp_m, labels)) \
                    / (2 * eps)

        np.testing.assert_allclose(grad, fd, rtol=1e-5, atol=1e-7)

    def test_singleton_cluster_gets_zero_gradient(self):
        from crisp.cawr import cawr_loss_grad
        rng = np.random.default_rng(1)
        fp = rng.standard_normal((5, 4))
        labels = np.array([0, 1, 1, 1, 1])
        _, grad = cawr_loss_grad(fp, labels)
        np.testing.assert_allclose(grad[0], 0.0)


if __name__ == '__main__':
    unittest.main()
