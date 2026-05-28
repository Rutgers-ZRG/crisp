import importlib.util
import unittest

import numpy as np
from ase import Atoms

import crisp
from crisp import BiasPotential, ExactGP
from crisp.fingerprint import FingerprintCalculator
from crisp.hpc import HPCRelaxer, RelaxBackend, VASPConfig
from crisp.matching import hungarian_match, sinkhorn_match
from crisp.projector_peratom import (
    project_peratom_forces,
    project_peratom_forces_and_stress,
    project_peratom_stress,
)


class PublicCoreTests(unittest.TestCase):
    def test_package_imports_without_optional_fingerprint_backend(self):
        self.assertEqual(crisp.__version__, "0.4.0")

    def test_hungarian_match_finds_permuted_target(self):
        fp_target = np.eye(4, 6) * 0.5
        fp = fp_target[[2, 0, 3, 1]]
        types = np.ones(4, dtype=int)

        loss, grad = hungarian_match(fp, fp_target, types)

        self.assertLess(loss, 1e-12)
        self.assertTrue(np.allclose(grad, 0.0))

    def test_sinkhorn_gradient_has_hungarian_direction(self):
        rng = np.random.default_rng(42)
        fp = rng.normal(size=(4, 6)) * 0.5
        fp_target = rng.normal(size=(4, 6)) * 0.5
        types = np.ones(4, dtype=int)

        _, grad_h = hungarian_match(fp, fp_target, types)
        _, grad_s = sinkhorn_match(fp, fp_target, types, tau=0.01, n_iters=100)
        cos = np.sum(grad_h * grad_s) / (
            np.linalg.norm(grad_h) * np.linalg.norm(grad_s) + 1e-30
        )

        self.assertGreater(cos, 0.8)

    def test_peratom_projector_force_and_stress_signs(self):
        rng = np.random.default_rng(7)
        nat, fp_dim = 4, 8
        volume = 100.0
        dfp = rng.normal(size=(nat, nat, 3, fp_dim)) * 0.1
        dfpe = rng.normal(size=(nat, 6, fp_dim)) * 0.01
        dL_dfp = rng.normal(size=(nat, fp_dim))

        forces, stress = project_peratom_forces_and_stress(
            dfp, dfpe, dL_dfp, volume
        )

        self.assertTrue(np.allclose(forces, project_peratom_forces(dfp, dL_dfp)))
        self.assertTrue(np.allclose(stress, project_peratom_stress(dfpe, dL_dfp, volume)))

        eps = 1e-5
        for voigt_idx in range(6):
            delta_fp = dfpe[:, voigt_idx, :] * eps
            delta_loss = np.sum(dL_dfp * delta_fp)
            expected_delta_loss = -volume * stress[voigt_idx] * eps
            self.assertAlmostEqual(delta_loss, expected_delta_loss, places=12)

    def test_autograd_stress_projection_uses_ase_sign_convention(self):
        self.assertIsNotNone(importlib.util.find_spec("torch"))

        import crisp.fingerprint as fp_module

        old_has = fp_module._HAS_TORCH_FPLIB
        old_backend = fp_module.torch_fplib

        class FakeTorchFPLib:
            @staticmethod
            def get_lfp(cell, cutoff, natx, orbital):
                lat, rxyz, types, _znucl = cell
                value = lat[0, 0] + 0.0 * rxyz.sum()
                return value.reshape(1, 1).repeat(len(types), 1)

        try:
            fp_module._HAS_TORCH_FPLIB = True
            fp_module.torch_fplib = FakeTorchFPLib

            atoms = Atoms(
                "HH",
                positions=[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
                cell=np.eye(3),
                pbc=True,
            )
            fp_calc = FingerprintCalculator(cutoff=1.0, natx=8, orbital="s")
            forces, stress = fp_calc.project_forces_and_stress(atoms, np.ones((2, 1)))

            self.assertTrue(np.allclose(forces, 0.0))
            self.assertAlmostEqual(stress[0], -2.0)
            self.assertTrue(np.allclose(stress[1:], 0.0))
        finally:
            fp_module._HAS_TORCH_FPLIB = old_has
            fp_module.torch_fplib = old_backend

    def test_exact_gp_prediction_roundtrip(self):
        X = np.array([[0.0], [1.0], [2.0]], dtype=float)
        y = np.array([1.0, 0.0, 1.0], dtype=float)
        gp = ExactGP(length_scale=1.0, noise=1e-6)
        gp.train(X, y)

        mu, sigma, grad_mu, grad_sigma = gp.predict_with_grad(np.array([1.0]))

        self.assertLess(mu, 1e-3)
        self.assertGreaterEqual(sigma, 0.0)
        self.assertEqual(grad_mu.shape, (1,))
        self.assertEqual(grad_sigma.shape, (1,))

    def test_bias_potential_is_finite_after_gp_training(self):
        X = np.array([[0.0], [1.0], [2.0]], dtype=float)
        y = np.array([1.0, 0.0, 1.0], dtype=float)
        gp = ExactGP(length_scale=1.0, noise=1e-6)
        gp.train(X, y)
        bias = BiasPotential(gp)

        value, grad = bias.evaluate_with_grad(np.array([1.0]))

        self.assertTrue(np.isfinite(value))
        self.assertEqual(grad.shape, (1,))

    def test_public_hpc_defaults_are_generic(self):
        cfg = VASPConfig()
        self.assertNotIn("/home/", cfg.vasp_cmd)
        self.assertEqual(cfg.potcar_dir, "")

        def fake_ssh(_cmd, timeout=120):
            return ""

        relaxer = HPCRelaxer(
            ssh_exec=fake_ssh,
            upload=lambda _local, _remote: None,
            download=lambda _remote, _local: None,
            remote_base="/scratch/$USER/crisp-runs",
            backend=RelaxBackend.MACE_MP,
        )
        self.assertEqual(relaxer.conda_path, "conda")


if __name__ == "__main__":
    unittest.main()
