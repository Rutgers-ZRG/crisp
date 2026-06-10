"""FP-targeted finisher — steer candidates toward structural prototypes.

Uses per-atom FP matching (Hungarian or Sinkhorn) to generate a bias
force that pushes the structure toward a target motif. The bias is
mixed with physical forces using adaptive λ (force-scale matching),
annealed to zero, and followed by unbiased cleanup relaxation.

This implements the "mutations + FP-targeted relaxation" mechanism
that succeeded in v0.3 FP-targeting experiments (Test 5).
"""

import logging
from dataclasses import dataclass, field
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

from ..fingerprint import FingerprintCalculator
from ..matching import hungarian_match, sinkhorn_match
from ..targets import Target, TargetLibrary

logger = logging.getLogger(__name__)

def _ascii(msg, limit=80):
    """ASCII-sanitized truncation for atoms.info strings — error texts
    can contain non-ASCII (e.g. Angstrom symbols) and extxyz writes
    crash on ASCII-locale compute nodes."""
    return str(msg)[:limit].encode("ascii", "replace").decode()



class _FinisherSafetyStop(Exception):
    """Raised by safety check to stop the bias optimization loop."""
    pass


@dataclass
class FinisherConfig:
    """Configuration for the FP-targeted finisher."""
    # Matching
    matching_backend: str = "hungarian"   # "hungarian" | "sinkhorn"
    sinkhorn_tau: float = 0.05
    sinkhorn_iters: int = 50
    matching_interval: int = 5            # recompute matching every N steps

    # Schedule
    pre_steps: int = 80                   # unbiased pre-relax steps
    bias_steps: int = 200                 # biased relaxation steps
    cleanup_fmax: float = 0.005           # unbiased cleanup convergence
    cleanup_max_steps: int = 100

    # Adaptive λ
    eta: float = 0.3                      # force-ratio scaling
    lambda_min: float = 0.0
    lambda_max: float = 20.0
    anneal_to_zero: bool = True           # linear anneal λ → 0 over bias_steps

    # Gating
    gate_enabled: bool = True
    run_on_mutants: bool = True           # always run on mutant candidates
    d_gate_init: float = 0.20             # initial FP distance gate
    d_gate_final: float = 0.10            # final FP distance gate
    anneal_gens: int = 10                 # generations over which gate tightens

    # Safety
    max_f_bias_rms: float = 50.0          # eV/Å — clamp if exceeded
    max_e_bias_per_atom: float = 5.0      # eV/atom — reject if exceeded
    min_dist_ang: float = 1.0             # minimum interatomic distance during bias

    # Pressure
    pressure_GPa: float = 0.0

    # Cell relaxation
    relax_cell: bool = True

    # Optimizer
    optimizer: str = "LBFGS"              # "LBFGS" | "FIRE"

    # PSO-inspired adaptive exploration (active only during stagnation)
    stagnation_gens: int = 5          # gens without best-H improvement → stagnation
    diversity_topk: int = 3           # pick random from top-k targets
    repulsion_weight: float = 0.3     # parent repulsion strength


class BiasCalculator(Calculator):
    """ASE Calculator that mixes physical forces with FP-targeted bias.

    E_total = E_phys + λ·E_bias
    F_total = F_phys + λ·F_bias
    σ_total = σ_phys + λ·σ_bias

    where λ is adaptive (force-scale matching) and anneals to zero.
    """

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, base_calc: Calculator,
                 fp_calc: FingerprintCalculator,
                 target: Target,
                 config: FinisherConfig,
                 parent_fp: np.ndarray = None,
                 parent_types: np.ndarray = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.base_calc = base_calc
        self.fp_calc = fp_calc
        self.target = target
        self.config = config
        self.parent_fp = parent_fp
        self.parent_types = parent_types

        self._step = 0
        self._cached_dL_dfp = None
        self._cached_loss = None

    def set_step(self, step: int) -> None:
        self._step = step

    def _advance_step(self) -> None:
        """Auto-increment step counter (called each calculate)."""
        self._step += 1

    def calculate(self, atoms=None, properties=None,
                  system_changes=tuple(all_changes)):
        if properties is None:
            properties = ["energy", "forces"]
        super().calculate(atoms, properties, system_changes)

        # Physical contribution
        self.atoms.calc = self.base_calc
        e_phys = self.atoms.get_potential_energy()
        f_phys = self.atoms.get_forces()
        try:
            s_phys = self.atoms.get_stress()
        except Exception:
            s_phys = np.zeros(6)

        # Bias contribution
        e_bias, f_bias, s_bias = self._compute_bias(self.atoms)

        # Adaptive λ with force-scale matching
        lam = self._compute_lambda(f_phys, f_bias)

        # Safety clamp on bias forces
        f_bias_rms = np.sqrt(np.mean(f_bias ** 2))
        if f_bias_rms > self.config.max_f_bias_rms:
            scale = self.config.max_f_bias_rms / (f_bias_rms + 1e-30)
            f_bias *= scale
            e_bias *= scale
            s_bias *= scale
            logger.debug("Bias forces clamped: rms %.1f → %.1f eV/Å",
                         f_bias_rms, self.config.max_f_bias_rms)

        self.results["energy"] = e_phys + lam * e_bias
        self.results["forces"] = f_phys + lam * f_bias
        self.results["stress"] = s_phys + lam * s_bias

        self._advance_step()

    def _compute_bias(self, atoms: Atoms):
        """Compute FP-targeting bias energy, forces, and stress.

        Both the matching (dL_dfp) and Jacobians (dfp, dfpe) are cached
        and only recomputed at matching_interval steps. Between intervals,
        the cached forces are reused — the bias direction is approximately
        constant over a few LBFGS steps, and avoiding dfp recomputation
        is critical for performance (dfp is O(nat²) and ~0.5s for 16 atoms).
        """
        nat = len(atoms)

        recompute = (self._step % self.config.matching_interval == 0
                     or self._cached_dL_dfp is None)

        if recompute:
            fp = self.fp_calc.get_fingerprints(atoms)
            types = self.fp_calc.atoms_to_cell(atoms)[2]

            if self.config.matching_backend == "sinkhorn":
                loss, dL_dfp = sinkhorn_match(
                    fp, self.target.fp, types,
                    types_target=self.target.types,
                    tau=self.config.sinkhorn_tau,
                    n_iters=self.config.sinkhorn_iters,
                )
            else:
                loss, dL_dfp = hungarian_match(
                    fp, self.target.fp, types,
                    types_target=self.target.types)

            # PSO-inspired parent repulsion (active during stagnation)
            if self.parent_fp is not None:
                try:
                    _, dL_parent = hungarian_match(
                        fp, self.parent_fp, types,
                        types_target=self.parent_types)
                    dL_dfp = dL_dfp - self.config.repulsion_weight * dL_parent
                except Exception:
                    pass  # parent matching failed, use social only

            # Project via autograd VJP (no Jacobian materialization)
            e_bias = loss * nat
            if self.config.relax_cell:
                f_bias, s_bias = self.fp_calc.project_forces_and_stress(
                    atoms, dL_dfp)
            else:
                f_bias = self.fp_calc.project_forces(atoms, dL_dfp)
                s_bias = np.zeros(6)

            self._cached_dL_dfp = dL_dfp
            self._cached_loss = loss
            self._cached_f_bias = f_bias
            self._cached_e_bias = e_bias
            self._cached_s_bias = s_bias
        else:
            # Reuse cached bias forces (approximately constant over few steps)
            f_bias = self._cached_f_bias
            e_bias = self._cached_e_bias
            s_bias = self._cached_s_bias

        return e_bias, f_bias, s_bias

    def _compute_lambda(self, f_phys: np.ndarray,
                        f_bias: np.ndarray) -> float:
        """Adaptive λ with force-scale matching and annealing."""
        cfg = self.config

        # Force-scale matching: λ = η · |F_phys|_rms / |F_bias|_rms
        f_phys_rms = np.sqrt(np.mean(f_phys ** 2)) + 1e-30
        f_bias_rms = np.sqrt(np.mean(f_bias ** 2)) + 1e-30
        lam = cfg.eta * f_phys_rms / f_bias_rms

        # Clip
        lam = np.clip(lam, cfg.lambda_min, cfg.lambda_max)

        # Anneal to zero over bias_steps
        if cfg.anneal_to_zero and cfg.bias_steps > 0:
            progress = min(self._step / cfg.bias_steps, 1.0)
            lam *= (1.0 - progress)

        return float(lam)


class FPTargetFinisher:
    """FP-targeted finisher: steer candidates toward structural prototypes.

    Runs a three-phase refinement:
    1. Pre-relax: unbiased physical relaxation (settle into nearest basin)
    2. Biased relax: FP-targeting bias with adaptive λ, annealed to zero
    3. Cleanup: unbiased relaxation to convergence

    Parameters
    ----------
    fp_calc : FingerprintCalculator
        Fingerprint calculator.
    target_lib : TargetLibrary
        Library of structural targets.
    config : FinisherConfig
        Finisher configuration.
    """

    def __init__(self, fp_calc: FingerprintCalculator,
                 target_lib: TargetLibrary,
                 config: Optional[FinisherConfig] = None):
        self.fp_calc = fp_calc
        self.target_lib = target_lib
        self.config = config or FinisherConfig()

    def should_run(self, atoms: Atoms, generation: int,
                   is_mutant: bool = False) -> bool:
        """Determine whether to apply the finisher to this candidate.

        Skips gen 0 (no archive → no targets). For gen ≥ 1, runs if:
        - candidate is a mutant (always), OR
        - min distance to any target < d_gate
        """
        if generation == 0:
            return False

        if not self.target_lib.targets:
            return False

        if not self.config.gate_enabled:
            return True

        if self.config.run_on_mutants and is_mutant:
            return True

        # Distance gating with annealed threshold
        try:
            fp = self.fp_calc.get_fingerprints(atoms)
        except (ValueError, RuntimeError):
            logger.debug("Finisher gating: FP computation failed, skipping")
            return False
        d_min = self.target_lib.min_target_distance(fp)

        d_gate = self._current_d_gate(generation)
        return d_min < d_gate

    def run(self, atoms: Atoms, base_calc: Calculator,
            is_mutant: bool = False,
            stagnation_count: int = 0) -> Atoms:
        """Execute the three-phase finisher.

        Parameters
        ----------
        atoms : ase.Atoms
            Input candidate (modified in place).
        base_calc : Calculator
            Physical calculator (MLIP) for forces.
        is_mutant : bool
            If True, preferentially target known phases.
        stagnation_count : int
            Generations since last best-H improvement. When >= stagnation_gens,
            PSO-inspired exploration is activated: diverse targets + parent repulsion.

        Returns
        -------
        ase.Atoms
            Refined structure.
        """
        cfg = self.config
        nat = len(atoms)
        stagnating = stagnation_count >= cfg.stagnation_gens

        # Find target — diverse selection during stagnation
        try:
            fp = self.fp_calc.get_fingerprints(atoms)
            types = self.fp_calc.atoms_to_cell(atoms)[2]
        except (ValueError, RuntimeError) as exc:
            raise RuntimeError(
                f"FP computation failed: {exc}") from exc

        if stagnating:
            target = self.target_lib.get_random_from_topk(
                fp, types, k=cfg.diversity_topk)
        else:
            target = self.target_lib.get_nearest_target(
                fp, types, prefer_known=is_mutant)
        if target is None:
            raise RuntimeError(
                "no matching target (no known phases and/or atom count mismatch)")

        initial_dist = self.target_lib.min_target_distance(fp)

        # Phase 1: unbiased pre-relax
        if cfg.pre_steps > 0:
            atoms = self._unbiased_relax(atoms, base_calc, cfg.pre_steps, fmax=0.1)

        # Phase 2: biased relaxation
        # Use a single optimizer + single ExpCellFilter to avoid memory leaks
        # from repeated construction (critical with PyTorch-backed calcs like MACE).
        # BiasCalculator auto-increments its step counter each calculate() call,
        # so annealing and matching_interval work automatically.

        # PSO-inspired: parent repulsion during stagnation
        parent_fp = None
        parent_types = None
        if stagnating and is_mutant:
            parent_fp = atoms.info.get('parent_fp', None)
            parent_types = atoms.info.get('parent_types', None)

        bias_calc = BiasCalculator(
            base_calc=base_calc,
            fp_calc=self.fp_calc,
            target=target,
            config=cfg,
            parent_fp=parent_fp,
            parent_types=parent_types,
        )
        atoms.calc = bias_calc

        p_eV_A3 = cfg.pressure_GPa / 160.21766208

        OptCls = FIRE if cfg.optimizer == "FIRE" else LBFGS
        if cfg.relax_cell:
            ecf = CellFilter(atoms, scalar_pressure=p_eV_A3)
            opt = OptCls(ecf, logfile=None)
        else:
            opt = OptCls(atoms, logfile=None)

        # Safety check via attach — runs every step, raises to stop optimizer
        class _SafetyCheck:
            def __init__(self, atoms_ref, min_dist):
                self.atoms = atoms_ref
                self.min_dist = min_dist
            def __call__(self):
                dists = self.atoms.get_all_distances(mic=True)
                np.fill_diagonal(dists, np.inf)
                if dists.min() < self.min_dist:
                    logger.debug("Finisher: atoms too close (%.2f Å), "
                                 "stopping bias phase", dists.min())
                    raise _FinisherSafetyStop

        opt.attach(_SafetyCheck(atoms, cfg.min_dist_ang), interval=1)

        import warnings
        stop_reason = "completed"
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", message="Casting complex")
                converged = opt.run(fmax=0.01, steps=cfg.bias_steps)
                if converged:
                    stop_reason = "converged"
        except _FinisherSafetyStop:
            stop_reason = "safety_min_dist"
            logger.debug("Finisher: bias phase stopped (safety)")
        except Warning as exc:
            stop_reason = "warning:" + _ascii(str(exc))
            logger.debug("Finisher: bias phase warning: %s", exc)
        except RuntimeError as exc:
            stop_reason = "runtime_error:" + _ascii(str(exc))
            logger.debug("Finisher: bias phase runtime error: %s", exc)
        except Exception as exc:
            stop_reason = ("exception:" + type(exc).__name__ + ":"
                           + _ascii(str(exc)))
            logger.debug("Finisher: bias phase exception: %s", exc)
        atoms.info['finisher_bias_steps'] = int(getattr(opt, 'nsteps', 0))
        atoms.info['finisher_stop_reason'] = stop_reason

        # Record target distance at the end of the bias phase (before
        # cleanup) so cleanup reversion is measurable
        try:
            fp_bias_end = self.fp_calc.get_fingerprints(atoms)
            atoms.info['finisher_d_bias_end'] = float(
                self.target_lib.min_target_distance(fp_bias_end))
        except (ValueError, RuntimeError):
            atoms.info['finisher_d_bias_end'] = None

        # Phase 3: unbiased cleanup
        atoms = self._unbiased_relax(atoms, base_calc,
                                     cfg.cleanup_max_steps,
                                     fmax=cfg.cleanup_fmax)

        # Log result + write provenance to atoms.info
        try:
            fp_final = self.fp_calc.get_fingerprints(atoms)
            final_dist = self.target_lib.min_target_distance(fp_final)
            mode = "PSO-explore" if stagnating else "exploit"
            repel = parent_fp is not None
            logger.info("FP-target finisher [%s]: d_target %.4f → %.4f "
                        "(target: %s, repel=%s)",
                        mode, initial_dist, final_dist, target.label, repel)
            atoms.info['finisher_applied'] = True
            atoms.info['finisher_target'] = target.label
            atoms.info['finisher_d_init'] = float(initial_dist)
            atoms.info['finisher_d_final'] = float(final_dist)
            atoms.info['finisher_mode'] = mode
            atoms.info['finisher_repel'] = repel
        except (ValueError, RuntimeError):
            logger.info("FP-target finisher: completed (final FP unavailable)")
            atoms.info['finisher_applied'] = True
            atoms.info['finisher_target'] = target.label
            atoms.info['finisher_d_init'] = float(initial_dist)
            atoms.info['finisher_mode'] = "PSO-explore" if stagnating else "exploit"
            atoms.info['finisher_repel'] = parent_fp is not None

        return atoms

    def _unbiased_relax(self, atoms: Atoms, calc: Calculator,
                        max_steps: int, fmax: float = 0.05) -> Atoms:
        """Run unbiased physical relaxation."""
        atoms.calc = calc
        p_eV_A3 = self.config.pressure_GPa / 160.21766208

        OptCls = FIRE if self.config.optimizer == "FIRE" else LBFGS
        if self.config.relax_cell:
            ecf = CellFilter(atoms, scalar_pressure=p_eV_A3)
            opt = OptCls(ecf, logfile=None)
        else:
            opt = OptCls(atoms, logfile=None)

        import warnings
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", message="Casting complex")
                opt.run(fmax=fmax, steps=max_steps)
        except (RuntimeError, Warning):
            logger.debug("Finisher: cleanup stopped (numerical instability)")
        return atoms

    def _current_d_gate(self, generation: int) -> float:
        """Compute annealed distance gate for the current generation."""
        cfg = self.config
        if cfg.anneal_gens <= 0:
            return cfg.d_gate_init

        # Linear anneal from d_gate_init to d_gate_final
        progress = min(generation / cfg.anneal_gens, 1.0)
        return cfg.d_gate_init + progress * (cfg.d_gate_final - cfg.d_gate_init)
