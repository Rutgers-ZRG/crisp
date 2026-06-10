"""FP-guided species-swap mutation tests.

Motivation (2026-06 spinel verdict): every search mode funnels into a
cation-coordination-inverted impostor (+55 meV) because no operator
changes the cation arrangement. The swap mutation exchanges unlike-
species sites, preferring atoms whose FP environments deviate most
from their own species' mean (the 'misfits').
"""

import unittest

import numpy as np

try:
    import torch_fplib  # noqa: F401
    HAS_FPLIB = True
except ImportError:
    HAS_FPLIB = False


@unittest.skipUnless(HAS_FPLIB, "torch_fplib required")
class TestSwapMutation(unittest.TestCase):

    def _spinel(self):
        from benchmarks.systems import spinel_mgal2o4
        return spinel_mgal2o4()

    def test_swap_preserves_composition_and_changes_arrangement(self):
        from crisp.search import swap_mutate
        from crisp.fingerprint import FingerprintCalculator
        fpc = FingerprintCalculator(cutoff=5.0, natx=100)
        atoms = self._spinel()
        rng = np.random.default_rng(3)

        out = swap_mutate(atoms, fpc, rng=rng, species_pair=('Mg', 'Al'))
        self.assertIsNotNone(out)
        # Composition preserved
        self.assertEqual(sorted(out.get_chemical_symbols()),
                         sorted(atoms.get_chemical_symbols()))
        # Species arrangement changed (one Mg<->Al pair exchanged)
        diff = [i for i, (a, b) in enumerate(zip(
            atoms.get_chemical_symbols(), out.get_chemical_symbols()))
            if a != b]
        self.assertEqual(len(diff), 2)
        # Sites only rattled, not displaced (max 0.05 A rattle)
        np.testing.assert_allclose(out.positions, atoms.positions,
                                   atol=0.3)

    def test_swap_handles_elemental_gracefully(self):
        from crisp.search import swap_mutate
        from crisp.fingerprint import FingerprintCalculator
        from ase.build import bulk
        fpc = FingerprintCalculator(cutoff=5.0, natx=100)
        atoms = bulk('Si', 'diamond', a=5.43)
        out = swap_mutate(atoms, fpc, rng=np.random.default_rng(0))
        self.assertIsNone(out)  # nothing to swap


if __name__ == '__main__':
    unittest.main()
