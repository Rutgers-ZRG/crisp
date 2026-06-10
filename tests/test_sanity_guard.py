"""Archive sanity guard — protect the archive/GP from PES poisoning
(e.g. MACE-MP-0 collapse structures with absurdly low energies)."""

import unittest

import numpy as np

try:
    import torch_fplib  # noqa: F401
    HAS_FPLIB = True
except ImportError:
    HAS_FPLIB = False


@unittest.skipUnless(HAS_FPLIB, "torch_fplib required")
class TestSanityGuard(unittest.TestCase):

    def _search(self, **kwargs):
        from crisp import CRISPSearch, FingerprintCalculator
        fpc = FingerprintCalculator(cutoff=4.0, natx=80)
        return CRISPSearch(mlip_calc_factory=lambda: None, fp_calc=fpc,
                           composition={'Si': 4}, min_dist_ang=1.8,
                           vol_per_atom_range=(14.0, 30.0), **kwargs)

    def _atoms(self, min_dist_ok=True):
        from ase.build import bulk
        a = bulk('Si', 'fcc', a=3.8).repeat((2, 2, 1))[:4]
        a = bulk('Si', 'fcc', a=3.8)
        a = a.repeat((2, 2, 1))
        if not min_dist_ok:
            a.positions[1] = a.positions[0] + np.array([0.3, 0.0, 0.0])
        return a

    def test_h_floor_rejects_poisoned_energy(self):
        s = self._search(sanity_H_floor=-6.0)
        self.assertFalse(s._sanity_ok(self._atoms(), h=-12.0))
        self.assertTrue(s._sanity_ok(self._atoms(), h=-5.0))

    def test_min_dist_rejects_collapsed_structure(self):
        s = self._search()
        self.assertFalse(s._sanity_ok(self._atoms(min_dist_ok=False),
                                      h=-5.0))
        self.assertTrue(s._sanity_ok(self._atoms(), h=-5.0))

    def test_no_floor_accepts_low_energy(self):
        s = self._search()
        self.assertTrue(s._sanity_ok(self._atoms(), h=-12.0))


if __name__ == '__main__':
    unittest.main()
