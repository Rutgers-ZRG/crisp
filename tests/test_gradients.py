"""Finite-difference audits of the FP-bias gradient chain.

These pin down the three links the finisher/CAWR/GP-guided mechanisms
depend on:
  1. hungarian_match loss gradient in fp space
  2. projected bias FORCES F = -J^T dL/dfp vs FD of L(r)
  3. projected bias STRESS vs FD of L(strain), in the ASE sign
     convention (sigma = +1/V dE/d_eps), cross-checked against a real
     ASE calculator.
"""

import unittest

import numpy as np

try:
    import torch_fplib  # noqa: F401
    HAS_FPLIB = True
except ImportError:
    HAS_FPLIB = False


def _si_cell():
    from ase.build import bulk
    atoms = bulk('Si', 'diamond', a=5.43)
    atoms = atoms.repeat((2, 1, 1))   # 4 atoms
    atoms.rattle(0.08, seed=11)
    return atoms


class TestHungarianGradient(unittest.TestCase):

    def test_loss_grad_matches_fd(self):
        from crisp.matching import hungarian_match
        rng = np.random.default_rng(2)
        fp = rng.standard_normal((6, 5))
        fp_target = rng.standard_normal((6, 5))
        types = np.array([1, 1, 1, 2, 2, 2])

        loss, grad = hungarian_match(fp, fp_target, types)

        eps = 1e-6
        fd = np.zeros_like(fp)
        for i in range(fp.shape[0]):
            for k in range(fp.shape[1]):
                fp_p = fp.copy(); fp_p[i, k] += eps
                fp_m = fp.copy(); fp_m[i, k] -= eps
                lp, _ = hungarian_match(fp_p, fp_target, types)
                lm, _ = hungarian_match(fp_m, fp_target, types)
                fd[i, k] = (lp - lm) / (2 * eps)
        np.testing.assert_allclose(grad, fd, rtol=1e-5, atol=1e-8)


@unittest.skipUnless(HAS_FPLIB, "torch_fplib required")
class TestProjectedBiasGradients(unittest.TestCase):

    def setUp(self):
        from crisp.fingerprint import FingerprintCalculator
        from crisp.matching import hungarian_match
        self.fp_calc = FingerprintCalculator(cutoff=4.5, natx=60)
        self.atoms = _si_cell()
        target = _si_cell()
        target.rattle(0.15, seed=23)
        self.fp_target = self.fp_calc.get_fingerprints(target)
        self.types = self.fp_calc.atoms_to_cell(self.atoms)[2]
        self.hungarian = hungarian_match

    def _loss_of(self, atoms):
        fp = self.fp_calc.get_fingerprints(atoms)
        loss, _ = self.hungarian(fp, self.fp_target, self.types)
        return loss

    def test_bias_forces_match_fd(self):
        fp = self.fp_calc.get_fingerprints(self.atoms)
        _, dL_dfp = self.hungarian(fp, self.fp_target, self.types)
        forces = self.fp_calc.project_forces(self.atoms, dL_dfp)

        eps = 1e-5
        for (i, k) in [(0, 0), (1, 2), (3, 1)]:
            ap = self.atoms.copy()
            ap.positions[i, k] += eps
            am = self.atoms.copy()
            am.positions[i, k] -= eps
            fd = (self._loss_of(ap) - self._loss_of(am)) / (2 * eps)
            # F = -dL/dr
            self.assertAlmostEqual(
                forces[i, k], -fd, delta=2e-4 * max(1.0, abs(fd)),
                msg=f"force[{i},{k}] = {forces[i,k]:.6e} vs -FD {-fd:.6e}")

    def test_bias_stress_matches_fd_in_ase_convention(self):
        fp = self.fp_calc.get_fingerprints(self.atoms)
        _, dL_dfp = self.hungarian(fp, self.fp_target, self.types)
        _, stress = self.fp_calc.project_forces_and_stress(
            self.atoms, dL_dfp)

        vol = self.atoms.get_volume()
        eps = 1e-5
        voigt = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
        fd_stress = np.zeros(6)
        for v, (a, b) in enumerate(voigt):
            strain = np.zeros((3, 3))
            if a == b:
                strain[a, a] = eps
            else:
                strain[a, b] = eps / 2
                strain[b, a] = eps / 2
            # ASE-side deformation for row-vector cells: cell' = cell @ (I+eps)
            ap = self.atoms.copy()
            ap.set_cell(ap.cell.array @ (np.eye(3) + strain),
                        scale_atoms=True)
            am = self.atoms.copy()
            am.set_cell(am.cell.array @ (np.eye(3) - strain),
                        scale_atoms=True)
            fd_stress[v] = (self._loss_of(ap) - self._loss_of(am)) \
                / (2 * eps) / vol
        # ASE convention: sigma = +1/V dE/d_eps
        np.testing.assert_allclose(stress, fd_stress, rtol=5e-3, atol=1e-7)

    def test_peratom_projection_consistent_with_vjp(self):
        """projector_peratom (full-Jacobian path) must agree with the
        corrected VJP path for both forces and stress."""
        from crisp.projector_peratom import \
            project_peratom_forces_and_stress
        fp = self.fp_calc.get_fingerprints(self.atoms)
        _, dL_dfp = self.hungarian(fp, self.fp_target, self.types)

        f_vjp, s_vjp = self.fp_calc.project_forces_and_stress(
            self.atoms, dL_dfp)
        _, dfp, dfpe = self.fp_calc.get_fingerprints_jacobian_strain(
            self.atoms)
        f_pa, s_pa = project_peratom_forces_and_stress(
            dfp, dfpe, dL_dfp, self.atoms.get_volume())

        np.testing.assert_allclose(f_pa, f_vjp, rtol=1e-8, atol=1e-12)
        np.testing.assert_allclose(s_pa, s_vjp, rtol=1e-8, atol=1e-12)

    def test_ase_convention_witness(self):
        """Pin the FD construction to ASE's stress convention with LJ."""
        from ase.calculators.lj import LennardJones
        atoms = _si_cell()
        atoms.calc = LennardJones(sigma=2.0, epsilon=1.0, rc=5.0)
        sigma = atoms.get_stress(voigt=True)
        vol = atoms.get_volume()
        eps = 1e-6

        def e_of(strain):
            a = atoms.copy()
            a.set_cell(a.cell.array @ (np.eye(3) + strain),
                       scale_atoms=True)
            a.calc = LennardJones(sigma=2.0, epsilon=1.0, rc=5.0)
            return a.get_potential_energy()

        # xx (Voigt 0) and xz shear (Voigt 4)
        for v, (a_, b_) in [(0, (0, 0)), (4, (0, 2))]:
            strain = np.zeros((3, 3))
            if a_ == b_:
                strain[a_, a_] = eps
            else:
                strain[a_, b_] = eps / 2
                strain[b_, a_] = eps / 2
            fd = (e_of(strain) - e_of(-strain)) / (2 * eps) / vol
            self.assertAlmostEqual(
                sigma[v], fd, delta=1e-3 * max(1.0, abs(fd)),
                msg=f"ASE stress Voigt[{v}]: {sigma[v]:.6e} vs FD {fd:.6e}")


if __name__ == '__main__':
    unittest.main()
