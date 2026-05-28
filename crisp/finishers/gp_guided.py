"""GP-guided relaxation — steer structures toward GP-predicted low-energy basins.

Unlike the FP-targeted finisher (which drives toward a *known* structure's FP),
this mode follows the GP surrogate's energy gradient in FP-space to *discover*
new low-energy basins.  The key insight: E(fp) is smooth, so the GP gradient
provides a reliable "global navigation" signal that can bypass local minima
on the physical PES.

    F_total = F_phys + λ · F_GP
    F_GP = -J^T · dV_GP/dfp

where V_GP = mu(fp) - κ·σ(fp) is the GP lower-confidence-bound,
J = dfp/dR is the fingerprint Jacobian (via autograd VJP), and λ is
adaptive (force-scale matching) with annealing to zero.

Three-phase refinement:
1. Pre-relax:  unbiased physical relaxation (settle into nearest basin)
2. GP-biased:  follow GP gradient with adaptive λ, annealed to zero
3. Cleanup:    unbiased relaxation to convergence on the physical PES
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize import LBFGS, FIRE

try:
    from ase.filters import FrechetCellFilter as CellFilter
except ImportError:
    try:
        from ase.filters import ExpCellFilter as CellFilter
    except ImportError:
        from ase.constraints import ExpCellFilter as CellFilter

from ..bias import BiasPotential
from ..fingerprint import FingerprintCalculator
from ..projector import ForceProjector

logger = logging.getLogger(__name__)


@dataclass
class GPGuidedConfig:
    """Configuration for GP-guided relaxation."""
    # Schedule
    pre_steps: int = 40
    bias_steps: int = 100
    cleanup_fmax: float = 0.005
    cleanup_max_steps: int = 80

    # Adaptive λ
    eta: float = 0.3
    lambda_min: float = 0.0
    lambda_max: float = 10.0
    anneal_to_zero: bool = True

    # Lambda mode: 'adaptive', 'fixed' (recommended), 'inverse'
    #   adaptive: λ = η * |f_phys| / |f_bias|  (proportional to physical force — BAD for basin-hopping)
    #   fixed:    λ = eta                        (constant — best tested, 69 meV mean improvement)
    #   inverse:  λ = η * |f_bias| / (|f_phys| + eps)  (strong at minima)
    lambda_mode: str = "fixed"

    # Trust region: limit step in fp-space per iteration
    # (GP prediction unreliable far from training data)
    max_f_bias_rms: float = 30.0

    # Safety
    min_dist_ang: float = 1.0

    # Gating
    sigma_gate: float = 0.0       # only apply if GP sigma > this (0 = always)
    min_gp_training: int = 20     # need at least this many GP training points

    # Physics
    pressure_GPa: float = 0.0
    relax_cell: bool = True

    # Optimizer
    optimizer: str = "LBFGS"

    # Recompute interval (GP gradient is smooth, doesn't change fast)
    recompute_interval: int = 5


@dataclass
class KickRelaxConfig:
    """Configuration for kick-and-relax GP-guided search.

    Algorithm:
      for each cycle:
        1. Relax to local minimum (f_phys → 0)
        2. Compute GP gradient at minimum
        3. Kick atoms along -J^T · dV_GP/dfp (scaled to kick_size)
        4. Relax again — may land in a different basin

    Key insight: GP gradient is only meaningful at a local minimum,
    where f_phys ≈ 0 and the GP can provide the dominant signal.
    The blended approach fails because adaptive λ → 0 at minima.
    """
    n_cycles: int = 5
    kick_size: float = 0.3     # RMS displacement per kick (Å)
    relax_fmax: float = 0.01
    relax_max_steps: int = 200
    cleanup_fmax: float = 0.005
    cleanup_max_steps: int = 100

    # Whether to also kick the cell
    relax_cell: bool = True
    kick_cell: bool = True
    cell_kick_scale: float = 0.02  # strain magnitude

    # Physics
    pressure_GPa: float = 0.0

    # Safety
    min_dist_ang: float = 1.0
    min_gp_training: int = 20

    # Optimizer
    optimizer: str = "LBFGS"


class _GPGuidedSafetyStop(Exception):
    """Raised when atoms get too close during GP-guided relaxation."""
    pass


class GPGuidedCalculator(Calculator):
    """ASE Calculator blending physical forces with GP-gradient bias.

    F_total = F_phys + λ · F_GP

    where F_GP = -J^T · dV_GP/dfp (projected GP gradient), and λ is
    adaptive (force-scale matching) with optional annealing.
    """

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, base_calc: Calculator,
                 bias: BiasPotential,
                 projector: ForceProjector,
                 fp_calc: FingerprintCalculator,
                 config: GPGuidedConfig,
                 **kwargs):
        super().__init__(**kwargs)
        self.base_calc = base_calc
        self.bias = bias
        self.projector = projector
        self.fp_calc = fp_calc
        self.config = config

        self._step = 0
        self._cached_f_bias = None
        self._cached_s_bias = None
        self._cached_e_bias = 0.0

    def calculate(self, atoms=None, properties=None,
                  system_changes=tuple(all_changes)):
        if properties is None:
            properties = ["energy", "forces", "stress"]
        super().calculate(atoms, properties, system_changes)

        # Physical forces
        self.atoms.calc = self.base_calc
        e_phys = self.atoms.get_potential_energy()
        f_phys = self.atoms.get_forces()
        try:
            s_phys = self.atoms.get_stress()
        except Exception:
            s_phys = np.zeros(6)

        # GP bias
        cfg = self.config
        try:
            recompute = (self._step % cfg.recompute_interval == 0
                         or self._cached_f_bias is None)

            if recompute:
                fp = self.fp_calc.get_fingerprints(self.atoms)
                fp_pooled = self.fp_calc.pool_with_std(fp)
                V, grad_V = self.bias.evaluate_with_grad(fp_pooled)

                if cfg.relax_cell:
                    f_bias, s_bias = self.projector.compute_forces_and_stress(
                        self.atoms, grad_V)
                else:
                    f_bias = self.projector.compute_forces(
                        self.atoms, grad_V)
                    s_bias = np.zeros(6)

                nat = len(self.atoms)
                self._cached_f_bias = nat * f_bias
                self._cached_s_bias = nat * s_bias
                self._cached_e_bias = V * nat

            f_bias = self._cached_f_bias
            s_bias = self._cached_s_bias

            # Safety clamp
            f_bias_rms = np.sqrt(np.mean(f_bias ** 2)) + 1e-30
            if f_bias_rms > cfg.max_f_bias_rms:
                scale = cfg.max_f_bias_rms / f_bias_rms
                f_bias = f_bias * scale
                s_bias = s_bias * scale

            # Compute λ based on mode
            f_phys_rms = np.sqrt(np.mean(f_phys ** 2)) + 1e-30
            f_bias_rms = np.sqrt(np.mean(f_bias ** 2)) + 1e-30

            if cfg.lambda_mode == "fixed":
                lam = cfg.eta
            elif cfg.lambda_mode == "inverse":
                # Strong at minima (f_phys small), weak far from equilibrium
                lam = cfg.eta * f_bias_rms / (f_phys_rms + 1e-2)
            else:  # "adaptive" (original)
                lam = cfg.eta * f_phys_rms / f_bias_rms
            lam = np.clip(lam, cfg.lambda_min, cfg.lambda_max)

            # Anneal to zero
            if cfg.anneal_to_zero and cfg.bias_steps > 0:
                progress = min(self._step / cfg.bias_steps, 1.0)
                lam *= (1.0 - progress)

            self.results["energy"] = e_phys
            self.results["forces"] = f_phys + lam * f_bias
            self.results["stress"] = s_phys + lam * s_bias

        except Exception as exc:
            logger.debug("GP-guided bias failed (step %d): %s",
                         self._step, exc)
            self.results["energy"] = e_phys
            self.results["forces"] = f_phys
            self.results["stress"] = s_phys

        self._step += 1


def gp_guided_relax(atoms: Atoms,
                    base_calc: Calculator,
                    bias: BiasPotential,
                    projector: ForceProjector,
                    fp_calc: FingerprintCalculator,
                    config: Optional[GPGuidedConfig] = None) -> Atoms:
    """Apply GP-guided relaxation to a structure.

    Three phases:
    1. Unbiased pre-relax (settle into nearest basin)
    2. GP-biased relaxation (follow GP gradient, annealed)
    3. Unbiased cleanup (converge on physical PES)

    Parameters
    ----------
    atoms : ase.Atoms
    base_calc : Calculator
        Physical MLIP calculator.
    bias : BiasPotential
        GP-based bias potential (must be trained).
    projector : ForceProjector
    fp_calc : FingerprintCalculator
    config : GPGuidedConfig or None

    Returns
    -------
    atoms : ase.Atoms
        Refined structure.
    """
    if config is None:
        config = GPGuidedConfig()

    # Check GP is trained
    if bias.gp.X_train is None or len(bias.gp.X_train) < config.min_gp_training:
        logger.debug("GP-guided: insufficient training data (%s), skipping",
                     0 if bias.gp.X_train is None else len(bias.gp.X_train))
        return atoms

    # Optional gating on GP uncertainty
    if config.sigma_gate > 0:
        try:
            fp = fp_calc.get_fingerprints(atoms)
            fp_pooled = fp_calc.pool_with_std(fp)
            _, sigma = bias.gp.predict(fp_pooled)
            if sigma < config.sigma_gate:
                logger.debug("GP-guided: sigma=%.4f < gate=%.4f, skipping",
                             sigma, config.sigma_gate)
                return atoms
        except Exception:
            pass

    atoms = atoms.copy()
    p_eV_A3 = config.pressure_GPa / 160.21766208
    OptCls = FIRE if config.optimizer == "FIRE" else LBFGS

    # Record initial state for logging
    try:
        fp_init = fp_calc.get_fingerprints(atoms)
        fp_pooled_init = fp_calc.pool_with_std(fp_init)
        mu_init, sigma_init = bias.gp.predict(fp_pooled_init)
    except Exception:
        mu_init, sigma_init = None, None

    # Phase 1: unbiased pre-relax
    if config.pre_steps > 0:
        atoms.calc = base_calc
        if config.relax_cell:
            ecf = CellFilter(atoms, scalar_pressure=p_eV_A3)
            opt = OptCls(ecf, logfile=None)
        else:
            opt = OptCls(atoms, logfile=None)
        try:
            opt.run(fmax=0.1, steps=config.pre_steps)
        except (RuntimeError, Exception):
            logger.debug("GP-guided: pre-relax stopped")

    # Phase 2: GP-biased relaxation
    gp_calc = GPGuidedCalculator(
        base_calc=base_calc,
        bias=bias,
        projector=projector,
        fp_calc=fp_calc,
        config=config,
    )
    atoms.calc = gp_calc

    if config.relax_cell:
        ecf = CellFilter(atoms, scalar_pressure=p_eV_A3)
        opt = OptCls(ecf, logfile=None)
    else:
        opt = OptCls(atoms, logfile=None)

    # Safety check
    class _SafetyCheck:
        def __init__(self, atoms_ref, min_dist):
            self.atoms = atoms_ref
            self.min_dist = min_dist
        def __call__(self):
            dists = self.atoms.get_all_distances(mic=True)
            np.fill_diagonal(dists, np.inf)
            if dists.min() < self.min_dist:
                raise _GPGuidedSafetyStop

    opt.attach(_SafetyCheck(atoms, config.min_dist_ang), interval=1)

    import warnings
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message="Casting complex")
            opt.run(fmax=0.01, steps=config.bias_steps)
    except (_GPGuidedSafetyStop, RuntimeError, Warning):
        logger.debug("GP-guided: bias phase stopped (safety/numerical)")
    except Exception as exc:
        logger.debug("GP-guided: bias phase exception: %s", exc)

    # Phase 3: unbiased cleanup
    if config.cleanup_max_steps > 0:
        atoms.calc = base_calc
        if config.relax_cell:
            ecf = CellFilter(atoms, scalar_pressure=p_eV_A3)
            opt = OptCls(ecf, logfile=None)
        else:
            opt = OptCls(atoms, logfile=None)
        try:
            opt.run(fmax=config.cleanup_fmax, steps=config.cleanup_max_steps)
        except (RuntimeError, Exception):
            logger.debug("GP-guided: cleanup stopped")

    # Log result
    try:
        fp_final = fp_calc.get_fingerprints(atoms)
        fp_pooled_final = fp_calc.pool_with_std(fp_final)
        mu_final, sigma_final = bias.gp.predict(fp_pooled_final)
        logger.info("GP-guided relax: mu %.4f → %.4f, sigma %.4f → %.4f "
                     "(%d bias steps)",
                     mu_init or 0, mu_final, sigma_init or 0, sigma_final,
                     gp_calc._step)
        atoms.info['gp_guided'] = True
        atoms.info['gp_mu_init'] = float(mu_init) if mu_init is not None else None
        atoms.info['gp_mu_final'] = float(mu_final)
        atoms.info['gp_sigma_init'] = float(sigma_init) if sigma_init is not None else None
        atoms.info['gp_sigma_final'] = float(sigma_final)
    except Exception:
        atoms.info['gp_guided'] = True

    return atoms


def gp_kick_relax(atoms: Atoms,
                  base_calc: Calculator,
                  bias: BiasPotential,
                  projector: ForceProjector,
                  fp_calc: FingerprintCalculator,
                  config: Optional[KickRelaxConfig] = None) -> Atoms:
    """Kick-and-relax GP-guided search.

    Alternates between:
      1. Full physical relaxation to local minimum
      2. GP-gradient kick to escape the basin

    The GP gradient is computed at the minimum (where f_phys ≈ 0),
    so it provides the dominant structural change signal.
    After the kick, physical relaxation finds the nearest basin —
    which may be a new, lower-energy one.

    Returns the lowest-enthalpy structure found across all cycles.
    """
    if config is None:
        config = KickRelaxConfig()

    if bias.gp.X_train is None or len(bias.gp.X_train) < config.min_gp_training:
        return atoms

    atoms = atoms.copy()
    p_eV_A3 = config.pressure_GPa / 160.21766208
    OptCls = FIRE if config.optimizer == "FIRE" else LBFGS

    def _relax(a, fmax, max_steps):
        a.calc = base_calc
        if config.relax_cell:
            ecf = CellFilter(a, scalar_pressure=p_eV_A3)
            opt = OptCls(ecf, logfile=None)
        else:
            opt = OptCls(a, logfile=None)
        try:
            opt.run(fmax=fmax, steps=max_steps)
        except Exception:
            pass
        return a

    def _enthalpy(a):
        a.calc = base_calc
        e = a.get_potential_energy()
        v = a.get_volume()
        return (e + p_eV_A3 * v) / len(a)

    def _safety_check(a):
        dists = a.get_all_distances(mic=True)
        np.fill_diagonal(dists, np.inf)
        return dists.min() > config.min_dist_ang

    # Track best structure
    best_atoms = None
    best_H = np.inf
    mu_trajectory = []

    for cycle in range(config.n_cycles):
        # Phase 1: relax to local minimum
        atoms = _relax(atoms, config.relax_fmax, config.relax_max_steps)
        H_before = _enthalpy(atoms)

        # Track
        try:
            fp = fp_calc.get_fingerprints(atoms)
            fp_pooled = fp_calc.pool_with_std(fp)
            mu, sigma = bias.gp.predict(fp_pooled)
            mu_trajectory.append(mu)
        except Exception:
            mu, sigma = None, None

        if H_before < best_H:
            best_H = H_before
            best_atoms = atoms.copy()

        # Phase 2: compute GP kick direction at the minimum
        try:
            fp = fp_calc.get_fingerprints(atoms)
            fp_pooled = fp_calc.pool_with_std(fp)
            V, grad_V = bias.evaluate_with_grad(fp_pooled)

            if config.relax_cell:
                f_gp, s_gp = projector.compute_forces_and_stress(
                    atoms, grad_V)
            else:
                f_gp = projector.compute_forces(atoms, grad_V)
                s_gp = None

            # Scale to kick_size: normalize force direction, apply as displacement
            # F_GP = -dV/dR points toward lower GP energy
            nat = len(atoms)
            f_kick = nat * f_gp  # undo per-atom normalization
            f_kick_rms = np.sqrt(np.mean(f_kick ** 2)) + 1e-30

            # Displacement = kick_size * direction
            displacement = config.kick_size * (f_kick / f_kick_rms)
            atoms.positions += displacement

            # Cell kick along GP stress direction
            if config.kick_cell and s_gp is not None and config.relax_cell:
                s_kick = nat * s_gp
                s_rms = np.sqrt(np.mean(s_kick ** 2)) + 1e-30
                # Convert Voigt stress to strain perturbation
                strain = np.eye(3)
                s_norm = config.cell_kick_scale * s_kick / s_rms
                strain[0, 0] += s_norm[0]
                strain[1, 1] += s_norm[1]
                strain[2, 2] += s_norm[2]
                strain[1, 2] += s_norm[3] * 0.5
                strain[2, 1] += s_norm[3] * 0.5
                strain[0, 2] += s_norm[4] * 0.5
                strain[2, 0] += s_norm[4] * 0.5
                strain[0, 1] += s_norm[5] * 0.5
                strain[1, 0] += s_norm[5] * 0.5
                atoms.set_cell(atoms.cell @ strain.T, scale_atoms=True)

            if not _safety_check(atoms):
                logger.debug("Kick-relax: cycle %d safety stop, reverting",
                             cycle)
                atoms = best_atoms.copy()
                continue

            logger.info("Kick-relax cycle %d: H=%.4f, mu=%.4f, "
                        "kick_rms=%.3f Å",
                        cycle, H_before, mu if mu else 0,
                        config.kick_size)

        except Exception as exc:
            logger.debug("Kick-relax: cycle %d kick failed: %s", cycle, exc)
            continue

    # Final cleanup
    atoms = _relax(atoms, config.cleanup_fmax, config.cleanup_max_steps)
    H_final = _enthalpy(atoms)
    if H_final < best_H:
        best_H = H_final
        best_atoms = atoms.copy()

    # Log
    try:
        fp_final = fp_calc.get_fingerprints(best_atoms)
        fp_pooled_final = fp_calc.pool_with_std(fp_final)
        mu_final, sigma_final = bias.gp.predict(fp_pooled_final)
        mu_trajectory.append(mu_final)
        logger.info("Kick-relax: %d cycles, mu trajectory: %s, "
                     "best H=%.4f",
                     config.n_cycles,
                     " → ".join(f"{m:.4f}" for m in mu_trajectory),
                     best_H)
        best_atoms.info['gp_kick_relax'] = True
        best_atoms.info['gp_mu_trajectory'] = [float(m) for m in mu_trajectory]
        best_atoms.info['gp_best_H'] = float(best_H)
        best_atoms.info['gp_n_cycles'] = config.n_cycles
    except Exception:
        best_atoms.info['gp_kick_relax'] = True

    return best_atoms
