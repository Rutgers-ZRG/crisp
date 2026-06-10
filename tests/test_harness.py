"""Tests for the benchmark harness (registry, metrics, runner plumbing).

Registry/metrics tests need only ase+spglib+numpy. Runner integration
tests additionally need torch_fplib and are skipped when unavailable.
"""

import unittest

import numpy as np

try:
    import spglib
    HAS_SPGLIB = True
except ImportError:
    HAS_SPGLIB = False

try:
    import torch_fplib  # noqa: F401
    HAS_FPLIB = True
except ImportError:
    HAS_FPLIB = False


EXPECTED_SG = {
    'si16': {'diamond': 227, 'beta_tin': 141},
    'b28': {'gamma_b28': 58},
    'tio2_24': {'rutile': 136, 'anatase': 141, 'brookite': 61},
    'spinel': {'spinel': 227},
    'sio2_24': {'quartz': 152, 'cristobalite': 92},
    'mgsio3_20': {'perovskite': 62, 'ilmenite': 148},
}


@unittest.skipUnless(HAS_SPGLIB, "spglib required")
class TestSystemsRegistry(unittest.TestCase):

    def _sg_number(self, atoms, symprec=1e-3):
        cell = (atoms.cell.array, atoms.get_scaled_positions(),
                atoms.get_atomic_numbers())
        ds = spglib.get_symmetry_dataset(cell, symprec=symprec)
        return ds.number if hasattr(ds, 'number') else ds['number']

    def test_refs_build_and_match_spacegroup(self):
        from benchmarks.systems import SYSTEMS, composition_of
        for name, spec in SYSTEMS.items():
            self.assertIn(spec.expected_gs, spec.refs)
            for ref_name, builder in spec.refs.items():
                atoms = builder()
                # Right stoichiometry ratio
                comp = composition_of(atoms)
                self.assertEqual(set(comp), set(spec.composition),
                                 f"{name}/{ref_name}: species mismatch")
                ratios = {s: spec.composition[s] / comp[s] for s in comp}
                self.assertEqual(len(set(ratios.values())), 1,
                                 f"{name}/{ref_name}: non-uniform ratio")
                # Detected spacegroup
                sg = self._sg_number(atoms)
                self.assertEqual(
                    sg, EXPECTED_SG[name][ref_name],
                    f"{name}/{ref_name}: spglib SG {sg} != "
                    f"{EXPECTED_SG[name][ref_name]}")
                # No overlapping atoms (gamma-B28's validated reference
                # has a 1.19 A contact, so the floor is 1.05)
                d = atoms.get_all_distances(mic=True)
                np.fill_diagonal(d, np.inf)
                self.assertGreater(d.min(), 1.05,
                                   f"{name}/{ref_name}: atoms overlap")

    def test_gs_ref_supercell_matches_composition(self):
        """The expected-GS ref must tile to the search composition
        (needed for the d_fp success criterion)."""
        from benchmarks.systems import SYSTEMS, ref_at_composition, \
            composition_of
        for name, spec in SYSTEMS.items():
            ref = spec.refs[spec.expected_gs]()
            sup = ref_at_composition(spec, ref)
            self.assertIsNotNone(
                sup, f"{name}: GS ref cannot tile composition")
            self.assertEqual(composition_of(sup), spec.composition,
                             f"{name}: supercell composition mismatch")


class TestMetrics(unittest.TestCase):

    def test_detect_success_first_hit(self):
        from benchmarks.metrics import detect_success_from_records
        # Synthetic per-relaxation records (already in relax order)
        records = [
            {'relax_index': 1, 'generation': 0, 'd_fp': 0.5, 'dH_meV': 200.0, 'sg_match': False},
            {'relax_index': 2, 'generation': 0, 'd_fp': 0.2, 'dH_meV': 50.0, 'sg_match': False},
            {'relax_index': 3, 'generation': 1, 'd_fp': 0.04, 'dH_meV': 1.0, 'sg_match': True},
            {'relax_index': 4, 'generation': 1, 'd_fp': 0.01, 'dH_meV': 0.5, 'sg_match': True},
        ]
        out = detect_success_from_records(records, success_dfp=0.05,
                                          success_dH_meV=5.0)
        self.assertTrue(out['success'])
        self.assertEqual(out['n_relaxed_at_success'], 3)
        self.assertEqual(out['gen_at_success'], 1)

    def test_detect_success_dh_requires_sg(self):
        from benchmarks.metrics import detect_success_from_records
        # Low dH alone (degenerate impostor) must NOT count without SG match
        records = [
            {'relax_index': 1, 'generation': 0, 'd_fp': 0.3, 'dH_meV': 2.0, 'sg_match': False},
        ]
        out = detect_success_from_records(records, success_dfp=0.05,
                                          success_dH_meV=5.0)
        self.assertFalse(out['success'])
        self.assertIsNone(out['n_relaxed_at_success'])

    def test_detect_success_none(self):
        from benchmarks.metrics import detect_success_from_records
        out = detect_success_from_records([], success_dfp=0.05,
                                          success_dH_meV=5.0)
        self.assertFalse(out['success'])

    def test_enantiomorph_match(self):
        from benchmarks.metrics import sg_matches
        self.assertTrue(sg_matches(152, 152))
        self.assertTrue(sg_matches(154, 152))   # quartz enantiomorphs
        self.assertTrue(sg_matches(213, 212))
        self.assertFalse(sg_matches(152, 136))


@unittest.skipUnless(HAS_FPLIB and HAS_SPGLIB, "torch_fplib required")
class TestRunnerIntegration(unittest.TestCase):
    """End-to-end micro-run with a cheap toy calculator."""

    def _micro_run(self, mode, seed=7):
        import tempfile
        from benchmarks.runner import run_benchmark, MICRO_TEST_SPEC
        with tempfile.TemporaryDirectory() as td:
            result = run_benchmark(
                system='micro_lj', potential='lj', mode=mode, seed=seed,
                budget_relax=12, out_dir=td, max_generations=3,
                spec_override=MICRO_TEST_SPEC)
        return result

    def test_random_mode_runs_and_counts(self):
        result = self._micro_run('random')
        self.assertGreater(result['n_relaxed_total'], 0)
        self.assertLessEqual(result['n_relaxed_total'], 12 + MICRO_BATCH_MAX)
        for key in ('system', 'potential', 'mode', 'seed', 'success',
                    'n_relaxed_total', 'best_H', 'wall_s', 'config_echo'):
            self.assertIn(key, result)

    def test_seeding_reproducible(self):
        r1 = self._micro_run('random', seed=11)
        r2 = self._micro_run('random', seed=11)
        self.assertEqual(r1['gen0_fp_hash'], r2['gen0_fp_hash'])

    def test_seeding_differs(self):
        r1 = self._micro_run('random', seed=11)
        r2 = self._micro_run('random', seed=12)
        self.assertNotEqual(r1['gen0_fp_hash'], r2['gen0_fp_hash'])

    def test_crisp_mode_runs(self):
        result = self._micro_run('crisp')
        self.assertGreater(result['n_relaxed_total'], 0)
        self.assertTrue(result['config_echo']['finisher'])
        self.assertTrue(result['config_echo']['cawr'])

    def test_fponly_mode_runs(self):
        result = self._micro_run('fponly')
        self.assertGreater(result['n_relaxed_total'], 0)
        self.assertTrue(result['config_echo']['finisher'])


MICRO_BATCH_MAX = 8  # n_random + n_mutants of MICRO_TEST_SPEC


if __name__ == '__main__':
    unittest.main()
