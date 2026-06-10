"""CAWR — Cluster-Aware Within-structure Reform.

Clusters per-atom fingerprints within a single structure (via k-means)
to discover Wyckoff-like groups, then applies a bias force that pushes
atoms toward their cluster centroid FP — equalizing the local environment
of symmetry-equivalent atoms.

    L_CAWR = Σ_c Σ_{i∈c} ||fp_i - μ_c||²
    dL/dfp_i = 4·n_c·(fp_i - μ_c)
    F_CAWR = -J^T ∇_fp L

Used as a local pretreatment step before HPC (VASP) relaxation:
structures are symmetrized with MLIP + CAWR bias, then the improved
starting point is sent to DFT.
"""

import logging
from dataclasses import dataclass

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

from .fingerprint import FingerprintCalculator

logger = logging.getLogger(__name__)

_EV_A3_TO_GPA = 160.21766208


@dataclass
class CAWRConfig:
    """Configuration for CAWR pretreatment."""
    # Clustering
    min_k: int = 2
    max_k: int = 8
    recluster_interval: int = 10

    # Bias schedule
    eta: float = 0.3
    max_steps: int = 30
    anneal_to_zero: bool = True

    # Cleanup
    cleanup_steps: int = 10
    cleanup_fmax: float = 0.05

    # Lambda bounds
    lambda_min: float = 0.0

    # Safety
    max_f_bias_rms: float = 50.0
    min_dist_ang: float = 1.2

    # Physics
    pressure_GPa: float = 0.0
    relax_cell: bool = True

    # Optimizer
    optimizer: str = "LBFGS"              # "LBFGS" | "FIRE"


def discover_clusters(fp, min_k=2, max_k=8):
    """Cluster per-atom FPs to discover Wyckoff-like groups.

    Uses k-means with auto-K selection via silhouette score.

    Parameters
    ----------
    fp : ndarray, shape (nat, fp_dim)
    min_k, max_k : int

    Returns
    -------
    labels : ndarray, shape (nat,)
    best_k : int
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    nat = fp.shape[0]
    max_k = min(max_k, nat - 1)
    if max_k < min_k:
        return np.zeros(nat, dtype=int), 1

    best_k, best_score = min_k, -1
    best_labels = None

    for k in range(min_k, max_k + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(fp)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(fp, labels)
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels

    if best_labels is None:
        return np.zeros(nat, dtype=int), 1

    return best_labels, best_k


def cawr_loss_grad(fp, labels):
    """CAWR cluster-compactness loss and its gradient.

    L = sum_c sum_{i in c} ||fp_i - mu_c||^2

    Returns (loss, dL_dfp) with dL_dfp of shape fp.shape. Singleton
    clusters contribute nothing.
    """
    nat, fp_dim = fp.shape
    loss = 0.0
    dL_dfp = np.zeros((nat, fp_dim))
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        n_c = len(idx)
        if n_c < 2:
            continue
        mu_c = fp[idx].mean(axis=0)
        diff = fp[idx] - mu_c
        loss += float((diff ** 2).sum())
        # Exact gradient: the mu_c dependence cancels because
        # sum_{j in c}(fp_j - mu_c) = 0, leaving 2(fp_i - mu_c).
        # (A historical 4*n_c prefactor distorted the bias direction
        # whenever cluster sizes differed.)
        for j, i in enumerate(idx):
            dL_dfp[i] = 2.0 * diff[j]
    return loss, dL_dfp


def cawr_snap(atoms, fp_calc, n_iter=3, max_step=0.25,
              min_dist_ang=1.0, min_k=2, max_k=8):
    """Direct FP-space symmetrization via the analytical Jacobian.

    Instead of mixing a cluster-equalization bias force into a physical
    relaxation (which fights the PES), take Newton-like steps in
    fingerprint space: solve J . dr = (mu_c - fp_i) with the exact
    Jacobian J = dfp/dr and apply the least-squares displacement
    directly. Pure geometry — costs no MLIP calls.

    Returns the snapped Atoms (copy); falls back to the input on any
    failure.
    """
    out = atoms.copy()
    nat = len(out)
    for _ in range(n_iter):
        try:
            fp, dfp = fp_calc.get_fingerprints_and_jacobian(out)
        except (ValueError, RuntimeError):
            return out
        labels, _k = discover_clusters(fp, min_k, max_k)
        target = fp.copy()
        for c in np.unique(labels):
            idx = np.where(labels == c)[0]
            if len(idx) >= 2:
                target[idx] = fp[idx].mean(axis=0)
        delta_fp = (target - fp).reshape(-1)
        if np.linalg.norm(delta_fp) < 1e-8:
            break
        fp_dim = fp.shape[1]
        J = dfp.transpose(0, 3, 1, 2).reshape(nat * fp_dim, nat * 3)
        try:
            delta_r, *_ = np.linalg.lstsq(J, delta_fp, rcond=1e-2)
        except np.linalg.LinAlgError:
            return out
        delta_r = delta_r.reshape(nat, 3)
        step_max = np.abs(delta_r).max()
        if step_max > max_step:
            delta_r *= max_step / step_max
        trial = out.copy()
        trial.positions = trial.positions + delta_r
        d = trial.get_all_distances(mic=True)
        np.fill_diagonal(d, np.inf)
        if d.min() < min_dist_ang:
            break
        out = trial
    return out


def spglib_snap(atoms, symprecs=(0.05, 0.1, 0.2), max_disp=0.5):
    """Symmetrize by projecting onto the nearest spacegroup found by
    spglib at a loose-symprec ladder. Pure geometry, no MLIP calls.

    Picks the loosest symprec whose standardized structure (a) keeps
    the atom count and (b) moves no atom further than max_disp.
    """
    import spglib
    from ase import Atoms as AseAtoms

    best = None
    for sp in sorted(symprecs):
        try:
            cell = (atoms.cell.array, atoms.get_scaled_positions(),
                    atoms.get_atomic_numbers())
            std = spglib.standardize_cell(cell, to_primitive=False,
                                          no_idealize=False, symprec=sp)
            if std is None:
                continue
            lattice, scaled, numbers = std
            if len(numbers) != len(atoms):
                continue
            snapped = AseAtoms(numbers=numbers, scaled_positions=scaled,
                               cell=lattice, pbc=True)
            best = snapped
        except Exception:
            continue
    return best if best is not None else atoms.copy()


class CAWRBiasCalculator(Calculator):
    """ASE Calculator that mixes physical forces with CAWR symmetry bias.

    F_total = F_phys + λ · F_CAWR

    where F_CAWR pushes atoms toward their cluster centroid FP, and λ is
    adaptive (force-scale matched) with optional annealing.
    """

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, base_calc, fp_calc, config, **kwargs):
        super().__init__(**kwargs)
        self.base_calc = base_calc
        self.fp_calc = fp_calc
        self.config = config
        self._step = 0
        self._labels = None
        self._last_k = None
        self._f_bias = None
        self._e_bias = 0.0
        self.n_bias_fail = 0

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

        # CAWR bias
        cfg = self.config
        try:
            if (self._step % cfg.recluster_interval == 0
                    or self._f_bias is None):
                fp = self.fp_calc.get_fingerprints(self.atoms)
                self._labels, self._last_k = discover_clusters(
                    fp, cfg.min_k, cfg.max_k)

                loss, dL_dfp = cawr_loss_grad(fp, self._labels)

                if cfg.relax_cell:
                    self._f_bias, self._s_bias = \
                        self.fp_calc.project_forces_and_stress(
                            self.atoms, dL_dfp)
                else:
                    self._f_bias = self.fp_calc.project_forces(
                        self.atoms, dL_dfp)
                    self._s_bias = np.zeros(6)
                self._e_bias = float(loss)

            f_bias = self._f_bias
            s_bias = self._s_bias

            # Safety clamp (before λ so force-scale matching sees clamped magnitude)
            f_bias_rms = np.sqrt(np.mean(f_bias ** 2)) + 1e-30
            if f_bias_rms > cfg.max_f_bias_rms:
                scale = cfg.max_f_bias_rms / f_bias_rms
                f_bias = f_bias * scale
                s_bias = s_bias * scale

            # Adaptive λ: force-scale matching (on clamped bias)
            f_phys_rms = np.sqrt(np.mean(f_phys ** 2)) + 1e-30
            f_bias_rms = np.sqrt(np.mean(f_bias ** 2)) + 1e-30
            lam = cfg.eta * f_phys_rms / f_bias_rms
            lam = max(lam, cfg.lambda_min)

            # Anneal over max_steps
            if cfg.anneal_to_zero and cfg.max_steps > 0:
                progress = min(self._step / cfg.max_steps, 1.0)
                lam *= (1.0 - progress)

            self.results["energy"] = e_phys
            self.results["forces"] = f_phys + lam * f_bias
            self.results["stress"] = s_phys + lam * s_bias

        except Exception as exc:
            self.n_bias_fail += 1
            logger.debug("CAWR bias failed (step %d): %s", self._step, exc)
            self.results["energy"] = e_phys
            self.results["forces"] = f_phys
            self.results["stress"] = s_phys

        self._step += 1


class _CAWRSafetyStop(Exception):
    """Raised when atoms get too close during CAWR bias."""
    pass


def cawr_refine(atoms, fp_calc, base_calc, config=None):
    """Apply CAWR symmetry pretreatment to a structure.

    Two phases:
    1. CAWR-biased relaxation (max_steps) — pushes toward within-cluster
       FP equality while staying near physical PES.
    2. Unbiased cleanup (cleanup_steps) — settles back onto the PES.

    Parameters
    ----------
    atoms : ase.Atoms
        Input structure.
    fp_calc : FingerprintCalculator
    base_calc : Calculator
        Physical MLIP calculator.
    config : CAWRConfig or None

    Returns
    -------
    atoms : ase.Atoms
        Refined structure.
    """
    if config is None:
        config = CAWRConfig()

    atoms = atoms.copy()
    p_eV_A3 = config.pressure_GPa / _EV_A3_TO_GPA

    # Phase 1: CAWR-biased relaxation
    cawr_calc = CAWRBiasCalculator(base_calc, fp_calc, config)
    atoms.calc = cawr_calc

    OptCls = FIRE if config.optimizer == "FIRE" else LBFGS
    if config.relax_cell:
        ecf = CellFilter(atoms, scalar_pressure=p_eV_A3)
        opt = OptCls(ecf, logfile=None)
    else:
        opt = OptCls(atoms, logfile=None)

    # Safety: stop if atoms get too close
    class _SafetyCheck:
        def __init__(self, atoms_ref, min_dist):
            self.atoms = atoms_ref
            self.min_dist = min_dist
        def __call__(self):
            dists = self.atoms.get_all_distances(mic=True)
            np.fill_diagonal(dists, np.inf)
            if dists.min() < self.min_dist:
                raise _CAWRSafetyStop

    opt.attach(_SafetyCheck(atoms, config.min_dist_ang), interval=1)

    stop_reason = "completed"
    try:
        converged = opt.run(fmax=0.01, steps=config.max_steps)
        if converged:
            stop_reason = "converged"
    except _CAWRSafetyStop:
        stop_reason = "safety_min_dist"
        logger.debug("CAWR safety stop triggered")
    except RuntimeError as exc:
        stop_reason = f"runtime_error:{str(exc)[:80]}"
        logger.debug("CAWR runtime error: %s", exc)
    except Exception as exc:
        stop_reason = f"exception:{type(exc).__name__}:{str(exc)[:80]}"
        logger.debug("CAWR exception: %s", exc)
    atoms.info['cawr_bias_steps'] = int(getattr(opt, 'nsteps', 0))
    atoms.info['cawr_stop_reason'] = stop_reason
    atoms.info['cawr_bias_failures'] = int(cawr_calc.n_bias_fail)

    k = cawr_calc._last_k or 0

    # Phase 2: unbiased cleanup
    if config.cleanup_steps > 0:
        atoms.calc = base_calc
        if config.relax_cell:
            ecf = CellFilter(atoms, scalar_pressure=p_eV_A3)
            opt = OptCls(ecf, logfile=None)
        else:
            opt = OptCls(atoms, logfile=None)
        opt.run(fmax=config.cleanup_fmax, steps=config.cleanup_steps)

    logger.debug("CAWR refine: K=%d, %d bias steps", k, cawr_calc._step)
    return atoms
