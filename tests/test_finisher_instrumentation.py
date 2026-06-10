"""Instrumentation tests: finisher and CAWR must record how their bias
phases actually terminated (steps executed + stop reason) so silent
failures are measurable."""

import unittest

import numpy as np

try:
    import torch_fplib  # noqa: F401
    HAS_FPLIB = True
except ImportError:
    HAS_FPLIB = False


def _lj():
    from ase.calculators.lj import LennardJones
    return LennardJones(sigma=2.0, epsilon=1.0, rc=5.0)


def _si4():
    from ase.build import bulk
    atoms = bulk('Si', 'fcc', a=3.6).repeat((2, 2, 1))
    atoms.rattle(0.05, seed=3)
    return atoms


@unittest.skipUnless(HAS_FPLIB, "torch_fplib required")
class TestFinisherInstrumentation(unittest.TestCase):

    def test_finisher_records_bias_steps_and_stop_reason(self):
        from crisp.fingerprint import FingerprintCalculator
        from crisp.finishers.fp_target import (FPTargetFinisher,
                                               FinisherConfig)
        from crisp.targets import TargetLibrary

        fp_calc = FingerprintCalculator(cutoff=4.0, natx=80)
        target_lib = TargetLibrary(fp_calc, n_targets=4)
        ref = _si4()
        target_lib.add_known_phase(ref, label='ref')

        cfg = FinisherConfig(pre_steps=2, bias_steps=10,
                             cleanup_max_steps=2, optimizer='FIRE',
                             gate_enabled=False)
        fin = FPTargetFinisher(fp_calc, target_lib, cfg)
        atoms = _si4()
        atoms.rattle(0.1, seed=7)
        out = fin.run(atoms, _lj())

        self.assertIn('finisher_bias_steps', out.info)
        self.assertIsInstance(out.info['finisher_bias_steps'], int)
        self.assertIn('finisher_stop_reason', out.info)
        self.assertIsInstance(out.info['finisher_stop_reason'], str)

    def test_cawr_records_bias_steps_and_stop_reason(self):
        from crisp.fingerprint import FingerprintCalculator
        from crisp.cawr import CAWRConfig, cawr_refine

        fp_calc = FingerprintCalculator(cutoff=4.0, natx=80)
        cfg = CAWRConfig(max_steps=10, cleanup_steps=2, optimizer='FIRE')
        atoms = _si4()
        out = cawr_refine(atoms, fp_calc, _lj(), cfg)

        self.assertIn('cawr_bias_steps', out.info)
        self.assertIn('cawr_stop_reason', out.info)


if __name__ == '__main__':
    unittest.main()
