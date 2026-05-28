"""Top-level CRISP search loop.

Supports two operation modes:

- **Local mode** (legacy): all relaxation happens locally via ``mlip_calc_factory``.
- **HPC mode**: local orchestration (generation, screening, GP flow) with
  remote batch relaxation via ``HPCRelaxer``.

Each generation:
1. Generate random structures + mutants from archive (gen 1+)
2. GP confidence-based filter: skip only structures confidently predicted as bad
3. Optional surrogate flow for uncertain candidates (disabled by default in HPC mode)
4. Relaxation (local MLIP or HPC batch)
5. Add to archive, retrain GP
6. Checkpoint and convergence check
"""

import logging
from typing import Callable, Dict, List, Optional

import numpy as np
from ase import Atoms
from ase.optimize import LBFGS

try:
    from ase.filters import FrechetCellFilter as CellFilter
except ImportError:
    try:
        from ase.filters import ExpCellFilter as CellFilter
    except ImportError:
        from ase.constraints import ExpCellFilter as CellFilter

from .archive import StructureArchive
from .bias import BiasPotential
from .calculator import CRISPCalculator
from .fingerprint import FingerprintCalculator
from .cawr import CAWRConfig, cawr_refine
from .finishers.fp_target import FPTargetFinisher, FinisherConfig
from .finishers.gp_guided import GPGuidedConfig, gp_guided_relax
from .flow import LangevinFlow, Trajectory
from .gp_calculator import GPCalculator
from .projector import ForceProjector
from .surrogate import ExactGP
from .targets import TargetLibrary

logger = logging.getLogger(__name__)

# Conversion: 1 eV/A^3 = 160.21766208 GPa
_EV_A3_TO_GPA = 160.21766208


class CRISPSearch:
    """Crystal structure prediction via fingerprint-space GP surrogate.

    Parameters
    ----------
    mlip_calc_factory : callable or None
        Factory that returns a fresh ASE Calculator for the MLIP.
        Required for local mode; None for HPC-only mode.
    hpc_relaxer : HPCRelaxer or None
        HPC batch relaxer. When provided, all relaxation is dispatched
        to HPC instead of running locally.
    fp_calc : FingerprintCalculator
        Fingerprint calculator.
    composition : dict
        Target composition, e.g. ``{'Si': 16}`` or ``{'C': 8}``.
    pressure_GPa : float
        External pressure for enthalpy ranking.
    n_random : int
        Number of random structures per generation.
    n_mutants : int
        Number of mutated structures per generation (from gen 1+).
    n_select : int
        Number selected after screening (legacy V(f) ranking mode).
    n_flow_steps : int
        Flow integration steps per walker.
    max_generations : int
        Maximum search generations.
    gp_kernel : str
        GP kernel type.
    gp_length_scale : float
        GP RBF length scale.
    kappa : float
        LCB exploration weight.
    beta : float
        Anchor attraction weight.
    gamma : float
        Repulsion from visited minima.
    dup_threshold : float
        Fingerprint distance below which two structures are duplicates.
    convergence_gens : int
        Stop if no new minima found for this many consecutive generations.
    flow_temperature : float
        Langevin noise temperature.
    flow_dt : float
        Langevin time step.
    sigma_threshold : float
        GP sigma above which Mode 1 (surrogate flow) is used.
    min_dist_ang : float
        Minimum interatomic distance in random structure generation.
    vol_per_atom_range : tuple of float
        (min, max) volume per atom in A^3 for random cell generation.
    checkpoint_dir : str or None
        Directory for generation-level checkpoints.
    screening_mode : str
        ``'filter'`` (default): confidence-based GP filter — only skip structures
        the GP confidently predicts as bad. Much safer than ranking.
        ``'rank'``: legacy V(f) ranking + farthest-point selection.
    use_mutations : bool
        Whether to generate mutant structures from archive (default True).
    mutation_strain_std : float
        Strain std for softmutation (default 0.1).
    mutation_pos_std : float
        Position perturbation std for softmutation (default 0.3).
    gp_energy_margin : float
        Only skip if predicted H > H_best + this (eV/at). Default 0.2.
    gp_confidence_frac : float
        Only skip if σ < this fraction of σ_max. Default 0.15.
    enable_flow : bool
        Whether to apply surrogate flow in HPC mode (default False).
    enable_fp_finisher : bool
        Whether to enable the FP-targeted finisher (default False).
    finisher_config : FinisherConfig or None
        Configuration for the FP-targeted finisher.
    known_phases : list of ase.Atoms or None
        Known reference structures to use as finisher targets.
    n_finisher_targets : int
        Number of archive centroid targets for the finisher.
    enable_cawr_pretreat : bool
        Whether to enable CAWR pretreatment before HPC relaxation.
        Clusters per-atom FPs within each structure and applies a bias
        force to equalize atoms within each cluster (mimics Wyckoff
        positions). Runs locally with MLIP before sending to HPC.
    cawr_config : CAWRConfig or None
        Configuration for CAWR pretreatment.
    max_skip_frac : float
        Maximum fraction of candidates the GP filter can skip (coverage guardrail).
    min_relax_per_gen : int
        Minimum number of candidates to relax per generation (coverage guardrail).
    enable_fpj_mutations : bool
        Whether to generate FP-Jacobian mutants (J⁺-projected FP displacements).
    n_fpj_mutants : int
        Number of FP-Jacobian mutants per generation.
    fpj_target_d : float
        Target FP displacement magnitude for FP-J mutations.
    fpj_strain_std : float
        Random strain std applied alongside FP-J displacement.
    fpj_min_dist : float
        Minimum interatomic distance for FP-J mutant validity.
    fpj_momentum_weight : float
        Weight for FP-space momentum (0=pure random, 1=pure momentum).
        When a parent's lineage had a successful FP displacement direction,
        the next mutation blends: α·momentum + (1-α)·random.
    """

    def __init__(
        self,
        mlip_calc_factory: Optional[Callable] = None,
        hpc_relaxer=None,
        fp_calc: FingerprintCalculator = None,
        composition: Dict[str, int] = None,
        pressure_GPa: float = 0.0,
        n_random: int = 50,
        n_mutants: int = 10,
        n_select: int = 10,
        n_flow_steps: int = 50,
        max_generations: int = 50,
        gp_kernel: str = "rbf",
        gp_length_scale: float = 1.0,
        kappa: float = 1.0,
        beta: float = 0.1,
        gamma: float = 0.5,
        dup_threshold: float = 0.03,
        convergence_gens: int = 5,
        flow_temperature: float = 0.1,
        flow_dt: float = 0.01,
        sigma_threshold: float = 0.3,
        min_dist_ang: float = 1.5,
        vol_per_atom_range: tuple = (8.0, 30.0),
        checkpoint_dir: Optional[str] = None,
        screening_mode: str = "filter",
        use_mutations: bool = True,
        mutation_strain_std: float = 0.1,
        mutation_pos_std: float = 0.3,
        gp_energy_margin: float = 0.2,
        gp_confidence_frac: float = 0.15,
        enable_flow: bool = False,
        enable_fp_finisher: bool = False,
        finisher_config: Optional[FinisherConfig] = None,
        known_phases: Optional[List[Atoms]] = None,
        n_finisher_targets: int = 12,
        enable_cawr_pretreat: bool = False,
        cawr_config: Optional[CAWRConfig] = None,
        max_skip_frac: float = 0.60,
        min_relax_per_gen: int = 8,
        enable_fpj_mutations: bool = False,
        n_fpj_mutants: int = 5,
        fpj_target_d: float = 0.05,
        fpj_strain_std: float = 0.02,
        fpj_min_dist: float = 1.3,
        fpj_momentum_weight: float = 0.3,
        enable_gp_guided: bool = False,
        gp_guided_config: Optional[GPGuidedConfig] = None,
        gp_guided_top_n: int = 5,
    ):
        if mlip_calc_factory is None and hpc_relaxer is None:
            raise ValueError(
                "Must provide either mlip_calc_factory (local mode) "
                "or hpc_relaxer (HPC mode)"
            )
        if fp_calc is None:
            raise ValueError("fp_calc is required")
        if composition is None:
            raise ValueError("composition is required")

        self.mlip_calc_factory = mlip_calc_factory
        self.hpc_relaxer = hpc_relaxer
        self.fp_calc = fp_calc
        self.composition = composition
        self.pressure_GPa = pressure_GPa
        self.n_random = n_random
        self.n_mutants = n_mutants
        self.n_select = n_select
        self.n_flow_steps = n_flow_steps
        self.max_generations = max_generations
        self.gp_kernel = gp_kernel
        self.gp_length_scale = gp_length_scale
        self.kappa = kappa
        self.beta = beta
        self.gamma = gamma
        self.dup_threshold = dup_threshold
        self.convergence_gens = convergence_gens
        self.flow_temperature = flow_temperature
        self.flow_dt = flow_dt
        self.sigma_threshold = sigma_threshold
        self.min_dist_ang = min_dist_ang
        self.vol_per_atom_range = vol_per_atom_range
        self.checkpoint_dir = checkpoint_dir
        self.screening_mode = screening_mode
        self.use_mutations = use_mutations
        self.mutation_strain_std = mutation_strain_std
        self.mutation_pos_std = mutation_pos_std
        self.gp_energy_margin = gp_energy_margin
        self.gp_confidence_frac = gp_confidence_frac
        self.enable_flow = enable_flow
        self.enable_fp_finisher = enable_fp_finisher
        self.max_skip_frac = max_skip_frac
        self.min_relax_per_gen = min_relax_per_gen

        # FP-Jacobian mutation config
        self.enable_fpj_mutations = enable_fpj_mutations
        self.n_fpj_mutants = n_fpj_mutants
        self.fpj_target_d = fpj_target_d
        self.fpj_strain_std = fpj_strain_std
        self.fpj_min_dist = fpj_min_dist
        self.fpj_momentum_weight = fpj_momentum_weight

        # Detect mode
        self._use_hpc = hpc_relaxer is not None

        # FP-targeted finisher setup
        self._finisher = None
        if enable_fp_finisher:
            fcfg = finisher_config or FinisherConfig()
            fcfg.pressure_GPa = pressure_GPa
            target_lib = TargetLibrary(fp_calc, n_targets=n_finisher_targets)
            if known_phases:
                for i, phase in enumerate(known_phases):
                    target_lib.add_known_phase(phase, label=f"phase_{i}")
            self._finisher = FPTargetFinisher(fp_calc, target_lib, fcfg)
            self._target_lib = target_lib

        # CAWR pretreatment setup
        self._cawr_config = None
        if enable_cawr_pretreat:
            ccfg = cawr_config or CAWRConfig()
            ccfg.pressure_GPa = pressure_GPa
            self._cawr_config = ccfg

        # GP-guided refinement setup
        self.enable_gp_guided = enable_gp_guided
        self._gp_guided_config = None
        self._gp_guided_top_n = gp_guided_top_n
        if enable_gp_guided:
            gcfg = gp_guided_config or GPGuidedConfig()
            gcfg.pressure_GPa = pressure_GPa
            self._gp_guided_config = gcfg

        # Derived
        self._symbols = []
        for elem, count in sorted(composition.items()):
            self._symbols.extend([elem] * count)
        self._nat = len(self._symbols)

    def run(self, resume_from: Optional[str] = None) -> StructureArchive:
        """Run the full search. Returns the archive of found structures.

        Parameters
        ----------
        resume_from : str or None
            Path to checkpoint directory to resume from. If provided,
            loads the latest generation checkpoint and continues.
        """
        archive = StructureArchive(self.fp_calc, self.dup_threshold)
        gp = ExactGP(kernel=self.gp_kernel, length_scale=self.gp_length_scale)
        bias = BiasPotential(gp, kappa=self.kappa, beta=self.beta,
                             gamma=self.gamma)
        projector = ForceProjector(self.fp_calc)

        start_gen = 0
        no_new_count = 0
        best_h = float('inf')
        stagnation_count = 0

        # Resume from checkpoint
        if resume_from is not None:
            start_gen = archive.load_checkpoint(resume_from, gp=gp) + 1
            if gp.X_train is not None:
                anchors = archive.get_diverse(n=3, pool="best", pool_size=10)
                bias.set_anchors([a.fp_pooled for a in anchors])
                bias.set_repulsion_centers(
                    [e.fp_pooled for e in archive.entries]
                )
            logger.info("Resuming from gen %d (%d structures in archive)",
                        start_gen, len(archive.entries))

        for gen in range(start_gen, self.max_generations):
            print(f"\n=== Generation {gen} ===")
            new_this_gen = 0

            # Step 1: Generate random structures
            candidates = self._generate_random(self.n_random)
            print(f"  Generated {len(candidates)} random structures")

            if gen == 0:
                # Bootstrap: relax ALL random structures to seed GP
                new_this_gen = self._run_bootstrap(
                    candidates, archive, gp, bias
                )
                print(f"  Bootstrap complete: {new_this_gen} unique, "
                      f"Total: {len(archive.entries)}")
                if archive.entries:
                    best_h = min(e.enthalpy for e in archive.entries)
            else:
                # Generate mutants + screen/filter + relax
                new_this_gen = self._run_generation(
                    candidates, archive, gp, bias, projector, gen,
                    stagnation_count=stagnation_count,
                )
                current_best = min(e.enthalpy for e in archive.entries)
                # Track stagnation: gens since best-H improved by ≥1 meV
                if current_best < best_h - 0.001:
                    best_h = current_best
                    stagnation_count = 0
                else:
                    stagnation_count += 1
                stag_str = f" [STAGNANT {stagnation_count}]" if stagnation_count >= 5 else ""
                print(f"  New minima: {new_this_gen}, "
                      f"Total: {len(archive.entries)}, "
                      f"Best H: {current_best:.6f} eV/at{stag_str}")

            # Checkpoint
            if self.checkpoint_dir is not None:
                archive.save_checkpoint(self.checkpoint_dir, gp=gp,
                                        generation=gen)

            # Convergence check
            if gen > 0:
                if new_this_gen == 0:
                    no_new_count += 1
                    if no_new_count >= self.convergence_gens:
                        print(f"Converged: no new minima for "
                              f"{self.convergence_gens} generations.")
                        break
                else:
                    no_new_count = 0

        return archive

    # ------------------------------------------------------------------
    # Bootstrap (gen 0)
    # ------------------------------------------------------------------

    def _run_bootstrap(self, candidates: List[Atoms],
                       archive: StructureArchive, gp: ExactGP,
                       bias: BiasPotential) -> int:
        """Generation 0: relax all random structures to seed the GP."""
        print(f"  Bootstrap: relaxing all {len(candidates)} structures")

        if self._use_hpc:
            new_count = self._relax_batch_hpc(candidates, archive, generation=0)
        else:
            new_count = self._relax_batch_local(candidates, archive, generation=0)

        # Train GP on bootstrap data
        if len(archive.entries) >= 2:
            X = archive.get_all_pooled_fps()
            y = archive.get_all_enthalpies()
            gp.train(X, y)
            anchors = archive.get_diverse(n=3, pool="best", pool_size=10)
            bias.set_anchors([a.fp_pooled for a in anchors])
            bias.set_repulsion_centers(
                [e.fp_pooled for e in archive.entries]
            )

        return new_count

    # ------------------------------------------------------------------
    # Generation 1+ dispatch
    # ------------------------------------------------------------------

    def _run_generation(self, candidates: List[Atoms],
                        archive: StructureArchive, gp: ExactGP,
                        bias: BiasPotential, projector: ForceProjector,
                        gen: int, stagnation_count: int = 0) -> int:
        """Generation 1+: mutate, screen/filter, optionally finish, relax, update GP."""
        # Track which candidates are mutants (for finisher gating)
        n_random = len(candidates)
        is_mutant = [False] * n_random

        # Step 1b: Generate mutants from archive
        if self.use_mutations and archive.entries:
            mutants = self._generate_mutants(archive,
                                             stagnation_count=stagnation_count)
            candidates = candidates + mutants
            is_mutant.extend([True] * len(mutants))
            print(f"  + {len(mutants)} mutants = {len(candidates)} total candidates")

        # Step 1c: FP-Jacobian mutants with momentum
        if self.enable_fpj_mutations and archive.entries:
            fpj_mutants = self._generate_fpj_mutants(
                archive, stagnation_count=stagnation_count)
            candidates = candidates + fpj_mutants
            is_mutant.extend([True] * len(fpj_mutants))
            if fpj_mutants:
                print(f"  + {len(fpj_mutants)} FP-J mutants = "
                      f"{len(candidates)} total")

        # Step 2: Screen or filter (with coverage guardrails)
        if gp.X_train is not None:
            if self.screening_mode == "filter":
                candidates, is_mutant, n_skipped = self._gp_filter_with_origins(
                    candidates, is_mutant, gp
                )
                print(f"  GP filter: kept {len(candidates)}, skipped {n_skipped}")
            elif self.screening_mode == "rank":
                candidates, is_mutant = self._screen_and_select_with_origins(
                    candidates, is_mutant, bias, self.n_select)
                print(f"  Selected {len(candidates)} after V(f) ranking")
            else:
                raise ValueError(f"Unknown screening_mode: {self.screening_mode!r}")
        else:
            if self.screening_mode == "rank":
                n_sel = min(self.n_select, len(candidates))
                candidates = candidates[:n_sel]
                is_mutant = is_mutant[:n_sel]
                print(f"  No GP yet: using {len(candidates)} candidates")

        # Step 3: Surrogate flow for uncertain candidates (optional, legacy)
        if self.enable_flow and gp.X_train is not None:
            candidates = self._apply_surrogate_flow(
                candidates, gp, bias, projector
            )

        # Tag stagnation state on all candidates for provenance
        for atoms in candidates:
            atoms.info['stagnation_count'] = stagnation_count

        # Step 3b: FP-targeted finisher (v0.4)
        if self._finisher is not None and gen > 0:
            # Update target library from archive
            self._target_lib.update_from_archive(archive)
            candidates, is_mutant = self._apply_finisher(
                candidates, is_mutant, gen,
                stagnation_count=stagnation_count,
            )

        # Step 3c: CAWR pretreatment (symmetrize before HPC)
        if self._cawr_config is not None:
            candidates = self._apply_cawr_pretreatment(candidates)

        # Step 4: Relaxation
        if self._use_hpc:
            new_count = self._relax_batch_hpc(candidates, archive,
                                               generation=gen)
        else:
            new_count = self._relax_batch_local_with_bias(
                candidates, archive, gp, bias, projector, gen
            )

        # Step 5: Update GP
        if len(archive.entries) >= 2:
            X = archive.get_all_pooled_fps()
            y = archive.get_all_enthalpies()
            gp.train(X, y)
            if self.screening_mode == "rank":
                anchors = archive.get_diverse(n=3, pool="best", pool_size=10)
                bias.set_anchors([a.fp_pooled for a in anchors])
                bias.set_repulsion_centers(
                    [e.fp_pooled for e in archive.entries]
                )

        # Step 5b: GP-guided refinement (post-relaxation)
        if (self.enable_gp_guided and self.mlip_calc_factory is not None
                and gen > 0 and gp.X_train is not None):
            n_refined = self._apply_gp_guided(
                archive, bias, projector, gen)
            if n_refined > 0:
                new_count += n_refined
                # Retrain GP with refined structures
                X = archive.get_all_pooled_fps()
                y = archive.get_all_enthalpies()
                gp.train(X, y)

        return new_count

    # ------------------------------------------------------------------
    # Evolutionary operators
    # ------------------------------------------------------------------

    def _generate_mutants(self, archive: StructureArchive,
                          stagnation_count: int = 0) -> List[Atoms]:
        """Generate mutant structures via softmutation of archive entries.

        Uses tournament selection biased toward low-enthalpy structures.
        During stagnation (5+ gens without best-H improvement):
        1. Diverse parent pool (farthest-point from top-40, not just top-10)
        2. Increased mutation strength (2x strain, 2x position)
        3. Random restarts (~30% of slots are fresh random structures)
        """
        if not archive.entries or self.n_mutants <= 0:
            return []

        stagnating = stagnation_count >= 5
        sorted_entries = sorted(archive.entries, key=lambda e: e.enthalpy)

        # 1. Parent pool: diverse during stagnation, top-N otherwise
        if stagnating and len(sorted_entries) > 10:
            pool = self._diverse_parent_pool(sorted_entries, n=20)
        else:
            n_pool = min(len(sorted_entries), 10)
            pool = sorted_entries[:n_pool]

        # 3. Random restarts: 30% of slots during stagnation
        if stagnating:
            n_restart = max(1, self.n_mutants // 3)
            n_mutant_target = self.n_mutants - n_restart
        else:
            n_restart = 0
            n_mutant_target = self.n_mutants

        # 2. Mutation strength: 2x during stagnation
        strain_scale = 2.0 if stagnating else 1.0
        pos_scale = 2.0 if stagnating else 1.0

        mutants = []
        attempts = 0
        max_attempts = n_mutant_target * 20

        while len(mutants) < n_mutant_target and attempts < max_attempts:
            attempts += 1
            parent = pool[np.random.randint(len(pool))]

            strain_std = strain_scale * np.random.uniform(
                self.mutation_strain_std * 0.5,
                self.mutation_strain_std * 1.5
            )
            pos_std = pos_scale * np.random.uniform(
                self.mutation_pos_std * 0.3,
                self.mutation_pos_std * 1.5
            )

            mutant = self._softmutate(parent.atoms, strain_std, pos_std)
            if mutant is not None:
                # Provenance: origin + parent info
                mutant.info['origin'] = 'mutant'
                mutant.info['parent_enthalpy'] = parent.enthalpy
                rank = sorted_entries.index(parent)
                mutant.info['parent_rank'] = rank
                mutant.info['mutation_strain_std'] = strain_std
                mutant.info['mutation_pos_std'] = pos_std
                # Track parent FP for PSO-inspired repulsion
                try:
                    mutant.info['parent_fp'] = self.fp_calc.get_fingerprints(
                        parent.atoms)
                    mutant.info['parent_types'] = self.fp_calc.atoms_to_cell(
                        parent.atoms)[2]
                except Exception:
                    pass
                mutants.append(mutant)

        if len(mutants) < n_mutant_target:
            logger.warning("Only generated %d / %d mutants after %d attempts",
                          len(mutants), n_mutant_target, attempts)

        # Random restarts (fresh random structures to escape basin)
        if n_restart > 0:
            restarts = self._generate_random(n_restart)
            for r in restarts:
                r.info['origin'] = 'restart'  # override 'random' tag
            mutants.extend(restarts)
            print(f"  Stagnation mode: {len(mutants)-n_restart} mutants "
                  f"(2x strength, diverse pool) + {n_restart} random restarts")

        return mutants

    def _diverse_parent_pool(self, sorted_entries, n: int = 20) -> list:
        """Select diverse parents from top entries using farthest-point.

        Takes the top 2*n entries by enthalpy, then selects n diverse ones
        by greedy farthest-point on pooled FPs. Ensures the parent pool
        spans multiple structural basins instead of clustering in one.
        """
        pool_size = min(len(sorted_entries), max(n * 2, 40))
        candidates = sorted_entries[:pool_size]

        if len(candidates) <= n:
            return list(candidates)

        fps = np.array([e.fp_pooled for e in candidates])
        selected = [0]  # start with best enthalpy

        for _ in range(n - 1):
            min_dists = np.full(len(candidates), np.inf)
            for si in selected:
                d = np.linalg.norm(fps - fps[si], axis=1)
                min_dists = np.minimum(min_dists, d)
            min_dists[list(selected)] = -1.0
            selected.append(int(np.argmax(min_dists)))

        return [candidates[i] for i in selected]

    @staticmethod
    def _softmutate(atoms: Atoms, strain_std: float = 0.1,
                    pos_std: float = 0.3,
                    min_dist: float = 1.5) -> Optional[Atoms]:
        """Soft mutation: random strain + position perturbation.

        Returns None if the mutant has atoms too close together.
        """
        mutant = atoms.copy()

        # Random symmetric strain
        raw = strain_std * np.random.randn(3, 3)
        strain = np.eye(3) + 0.5 * (raw + raw.T)
        new_cell = mutant.cell @ strain
        mutant.set_cell(new_cell, scale_atoms=True)

        # Position perturbation
        mutant.positions += pos_std * np.random.randn(*mutant.positions.shape)
        mutant.wrap()

        # Minimum distance check
        dists = mutant.get_all_distances(mic=True)
        np.fill_diagonal(dists, np.inf)
        if dists.min() < min_dist:
            return None

        return mutant

    def _generate_fpj_mutants(self, archive: StructureArchive,
                              stagnation_count: int = 0) -> List[Atoms]:
        """Generate mutants via FP-Jacobian projection with optional momentum.

        For each mutant:
        1. Pick a parent from the low-enthalpy pool
        2. Compute FP Jacobian J = dfp/dr
        3. Build displacement: delta_fp = alpha*momentum + (1-alpha)*random
           where momentum is the FP-direction that improved H in the parent's lineage
        4. Project to real space: delta_r = J+ @ delta_fp
        5. Apply random strain + wrap

        During stagnation (5+ gens without improvement):
        - Diverse parent pool (farthest-point from top-40)
        - 3x target_d (larger FP displacement)
        - 30% random restarts

        Returns list of valid mutant Atoms.
        """
        if not archive.entries:
            return []

        stagnating = stagnation_count >= 5
        sorted_entries = sorted(archive.entries, key=lambda e: e.enthalpy)

        # Parent pool: diverse during stagnation
        if stagnating and len(sorted_entries) > 10:
            pool = self._diverse_parent_pool(sorted_entries, n=20)
        else:
            n_pool = min(len(sorted_entries), 10)
            pool = sorted_entries[:n_pool]

        # Random restarts: 30% of slots during stagnation
        if stagnating:
            n_restart = max(1, self.n_fpj_mutants // 3)
            n_fpj_target = self.n_fpj_mutants - n_restart
        else:
            n_restart = 0
            n_fpj_target = self.n_fpj_mutants

        # Mutation strength: 3x during stagnation
        target_d = self.fpj_target_d * (3.0 if stagnating else 1.0)
        strain_std = self.fpj_strain_std * (2.0 if stagnating else 1.0)

        alpha = self.fpj_momentum_weight
        mutants = []
        max_attempts = n_fpj_target * 4

        for _ in range(max_attempts):
            if len(mutants) >= n_fpj_target:
                break

            parent = pool[np.random.randint(len(pool))]

            try:
                fp, dfp = self.fp_calc.get_fingerprints_and_jacobian(
                    parent.atoms)
            except (ValueError, RuntimeError):
                continue

            nat, fp_dim = fp.shape
            # J: (nat*fp_dim, nat*3)
            J = dfp.transpose(0, 3, 1, 2).reshape(nat * fp_dim, nat * 3)
            try:
                U, S, Vt = np.linalg.svd(J, full_matrices=False)
            except np.linalg.LinAlgError:
                continue
            s_thresh = 0.01 * S[0] if S[0] > 0 else 1e-10
            S_inv = np.where(S > s_thresh, 1.0 / S, 0.0)
            J_pinv = (Vt.T * S_inv[np.newaxis, :]) @ U.T

            # Build FP displacement: blend momentum + random
            delta_fp_random = np.random.randn(nat * fp_dim)
            delta_fp_random /= np.linalg.norm(delta_fp_random)

            momentum_fp = parent.metadata.get('momentum_fp', None)
            if momentum_fp is not None and alpha > 0:
                mom = np.asarray(momentum_fp)
                mom_norm = np.linalg.norm(mom)
                if mom_norm > 1e-10:
                    mom_unit = mom / mom_norm
                    # Blend: alpha * momentum_direction + (1-alpha) * random
                    delta_fp = alpha * mom_unit + (1 - alpha) * delta_fp_random
                    delta_fp /= np.linalg.norm(delta_fp)
                else:
                    delta_fp = delta_fp_random
            else:
                delta_fp = delta_fp_random

            delta_fp *= target_d

            # Project to Cartesian
            delta_r = (J_pinv @ delta_fp).reshape(nat, 3)

            # Random strain
            raw = strain_std * np.random.randn(3, 3)
            strain = np.eye(3) + 0.5 * (raw + raw.T)

            mutant = parent.atoms.copy()
            new_cell = mutant.cell @ strain
            mutant.set_cell(new_cell, scale_atoms=True)
            mutant.positions += delta_r
            mutant.wrap()

            # Validity check
            dists = mutant.get_all_distances(mic=True)
            np.fill_diagonal(dists, np.inf)
            if dists.min() < self.fpj_min_dist:
                continue

            # Provenance
            mutant.info['origin'] = 'fpj_mutant'
            mutant.info['parent_enthalpy'] = parent.enthalpy
            rank = sorted_entries.index(parent)
            mutant.info['parent_rank'] = rank
            mutant.info['fpj_target_d'] = target_d
            mutant.info['fpj_strain_std'] = strain_std
            has_momentum = (momentum_fp is not None and alpha > 0)
            mutant.info['fpj_has_momentum'] = has_momentum
            # Store parent FP for momentum tracking after relaxation
            mutant.info['parent_fp_flat'] = fp.ravel().tolist()
            # Store parent FP for PSO repulsion in finisher
            try:
                mutant.info['parent_fp'] = self.fp_calc.get_fingerprints(
                    parent.atoms)
                mutant.info['parent_types'] = self.fp_calc.atoms_to_cell(
                    parent.atoms)[2]
            except Exception:
                pass

            mutants.append(mutant)

        # Random restarts during stagnation
        if n_restart > 0:
            restarts = self._generate_random(n_restart)
            for r in restarts:
                r.info['origin'] = 'restart'
            mutants.extend(restarts)

        if mutants:
            n_mom = sum(1 for m in mutants
                        if m.info.get('fpj_has_momentum', False))
            n_fpj_actual = len(mutants) - n_restart
            if stagnating:
                print(f"  FP-J mutations [STAGNANT {stagnation_count}]: "
                      f"{n_fpj_actual} FP-J (td={target_d:.2f}, "
                      f"{n_mom} momentum) + {n_restart} restarts")
            else:
                print(f"  FP-J mutations: {n_fpj_actual}/{n_fpj_target} "
                      f"({n_mom} with momentum)")

        return mutants

    def _update_momentum(self, atoms: Atoms, archive: StructureArchive):
        """After relaxation, compute momentum for successful lineages.

        If the child improved over its parent, store the FP displacement
        direction (child_fp - parent_fp) as momentum in the archive entry.
        This is used by future FP-J mutations from this structure.
        """
        parent_fp_flat = atoms.info.get('parent_fp_flat', None)
        if parent_fp_flat is None:
            return

        parent_h = atoms.info.get('parent_enthalpy', None)
        if parent_h is None:
            return

        # Find the archive entry for this structure (most recently added)
        if not archive.entries:
            return
        entry = archive.entries[-1]

        # Only store momentum if child improved over parent
        if entry.enthalpy >= parent_h:
            return

        try:
            child_fp = entry.fp.ravel()
            parent_fp = np.array(parent_fp_flat)
            if child_fp.shape != parent_fp.shape:
                return
            momentum = child_fp - parent_fp
            entry.metadata['momentum_fp'] = momentum.tolist()
            logger.debug("Momentum stored: |delta_fp|=%.4f, dH=%.4f",
                         np.linalg.norm(momentum),
                         entry.enthalpy - parent_h)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # GP confidence-based filter
    # ------------------------------------------------------------------

    def _gp_filter(self, candidates: List[Atoms],
                   gp: ExactGP) -> tuple:
        """Filter candidates using GP confidence criterion.

        Keep structure if:
        - GP uncertainty is high (σ > confidence_frac × σ_max) → explore
        - OR GP predicts low enthalpy (μ < h_best + energy_margin) → exploit

        Skip only if:
        - GP is confident (σ < threshold) AND predicts high enthalpy

        Returns (kept_candidates, n_skipped).
        """
        predictions = []
        for atoms in candidates:
            try:
                fp = self.fp_calc.get_fingerprints(atoms)
                fp_pooled = self.fp_calc.pool_with_std(fp)
                mu, sigma = gp.predict(fp_pooled)
                predictions.append((mu, sigma))
            except Exception:
                predictions.append((None, float('inf')))

        # Compute sigma range
        sigma_vals = [s for _, s in predictions if s != float('inf')]
        sigma_max = max(sigma_vals) if sigma_vals else 1.0
        sigma_threshold = self.gp_confidence_frac * sigma_max

        # Compute h_best from GP's training data (original scale)
        if gp._y_std > 0:
            h_best = gp._y_mean + gp._y_std * np.min(gp.y_train)
        else:
            h_best = np.min(gp.y_train)

        kept = []
        n_skipped = 0

        for i, atoms in enumerate(candidates):
            mu, sigma = predictions[i]

            if mu is None:
                # Failed to compute FP → keep to be safe
                kept.append(atoms)
                continue

            if sigma > sigma_threshold:
                # High uncertainty → always keep (explore)
                kept.append(atoms)
            elif mu < h_best + self.gp_energy_margin:
                # Predicted good → keep (exploit)
                kept.append(atoms)
            else:
                # Confident AND predicted bad → skip
                n_skipped += 1

        return kept, n_skipped

    def _gp_filter_with_origins(self, candidates: List[Atoms],
                                is_mutant: List[bool],
                                gp: ExactGP) -> tuple:
        """GP filter that preserves mutant origin tracking and enforces coverage caps.

        Returns (kept_candidates, kept_is_mutant, n_skipped).
        """
        n_total = len(candidates)

        predictions = []
        for atoms in candidates:
            try:
                fp = self.fp_calc.get_fingerprints(atoms)
                fp_pooled = self.fp_calc.pool_with_std(fp)
                mu, sigma = gp.predict(fp_pooled)
                predictions.append((mu, sigma))
            except Exception:
                predictions.append((None, float('inf')))

        sigma_vals = [s for _, s in predictions if s != float('inf')]
        sigma_max = max(sigma_vals) if sigma_vals else 1.0
        sigma_threshold = self.gp_confidence_frac * sigma_max

        if gp._y_std > 0:
            h_best = gp._y_mean + gp._y_std * np.min(gp.y_train)
        else:
            h_best = np.min(gp.y_train)

        # First pass: decide keep/skip for each candidate
        keep_flags = []
        for i in range(n_total):
            mu, sigma = predictions[i]
            if mu is None:
                keep_flags.append(True)
            elif sigma > sigma_threshold:
                keep_flags.append(True)
            elif mu < h_best + self.gp_energy_margin:
                keep_flags.append(True)
            else:
                keep_flags.append(False)

        # Coverage guardrails: enforce max_skip_frac and min_relax_per_gen
        n_kept = sum(keep_flags)
        max_skip = int(n_total * self.max_skip_frac)
        n_skipped_raw = n_total - n_kept

        if n_skipped_raw > max_skip or n_kept < self.min_relax_per_gen:
            # Rescue most uncertain skipped candidates
            skipped_indices = [i for i, k in enumerate(keep_flags) if not k]
            # Sort by sigma descending (rescue most uncertain first)
            skipped_indices.sort(
                key=lambda i: predictions[i][1] if predictions[i][1] != float('inf') else 0,
                reverse=True,
            )
            n_need = max(self.min_relax_per_gen - n_kept,
                         n_skipped_raw - max_skip, 0)
            for idx in skipped_indices[:n_need]:
                keep_flags[idx] = True

        kept = [c for c, k in zip(candidates, keep_flags) if k]
        kept_mutant = [m for m, k in zip(is_mutant, keep_flags) if k]
        n_skipped = n_total - len(kept)

        return kept, kept_mutant, n_skipped

    # ------------------------------------------------------------------
    # FP-targeted finisher (v0.4)
    # ------------------------------------------------------------------

    def _apply_finisher(self, candidates: List[Atoms],
                        is_mutant: List[bool],
                        gen: int,
                        stagnation_count: int = 0) -> tuple:
        """Apply FP-targeted finisher to eligible candidates.

        Returns (candidates, is_mutant) — candidates are modified in place
        where the finisher ran.
        """
        import gc

        if self._finisher is None:
            return candidates, is_mutant

        if self.mlip_calc_factory is None:
            logger.warning("FP finisher requires mlip_calc_factory but none provided "
                           "(HPC-only mode). Skipping finisher.")
            return candidates, is_mutant

        n_finished = 0
        for i, atoms in enumerate(candidates):
            mutant_flag = is_mutant[i] if i < len(is_mutant) else False
            if not self._finisher.should_run(atoms, gen, is_mutant=mutant_flag):
                atoms.info['finisher_applied'] = False
                continue
            calc = self.mlip_calc_factory()
            try:
                candidates[i] = self._finisher.run(
                    atoms, calc, is_mutant=mutant_flag,
                    stagnation_count=stagnation_count)
                n_finished += 1
            except Exception as exc:
                logger.warning("Finisher failed for candidate %d: %s", i, exc)
            finally:
                del calc
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass

        if n_finished > 0:
            stag = stagnation_count >= self._finisher.config.stagnation_gens
            mode = "PSO-explore" if stag else "exploit"
            print(f"  FP finisher [{mode}]: applied to "
                  f"{n_finished}/{len(candidates)} candidates")

        return candidates, is_mutant

    def _apply_cawr_pretreatment(self, candidates):
        """Apply CAWR within-structure symmetrization before HPC relaxation.

        Clusters per-atom FPs via k-means to discover Wyckoff-like groups,
        then applies a bias force equalizing atoms within each cluster.
        Runs locally with MLIP (fast, ~1-2s per structure).
        """
        import gc
        if self._cawr_config is None or self.mlip_calc_factory is None:
            return candidates
        n_refined = 0
        for i, atoms in enumerate(candidates):
            calc = self.mlip_calc_factory()
            try:
                candidates[i] = cawr_refine(
                    atoms, self.fp_calc, calc, self._cawr_config)
                candidates[i].info['cawr_applied'] = True
                n_refined += 1
            except Exception as exc:
                logger.warning("CAWR pretreat failed for candidate %d: %s",
                               i, exc)
            finally:
                del calc
                gc.collect()
        if n_refined > 0:
            print(f"  CAWR pretreat: refined {n_refined}/{len(candidates)}"
                  " candidates")
        return candidates

    # ------------------------------------------------------------------
    # GP-guided refinement (post-relaxation)
    # ------------------------------------------------------------------

    def _apply_gp_guided(self, archive: StructureArchive,
                         bias: BiasPotential,
                         projector: ForceProjector,
                         gen: int) -> int:
        """Apply GP-guided relaxation to top candidates from this generation.

        After HPC/local relaxation and GP retraining, take the top-N entries
        from this generation and run GP-guided refinement locally. If the
        refined structure improves, add it to the archive.

        Returns the number of new structures added.
        """
        import gc

        cfg = self._gp_guided_config
        if cfg is None:
            return 0

        # Get entries from this generation
        gen_entries = [e for e in archive.entries
                       if e.metadata.get('generation') == gen]
        if not gen_entries:
            return 0

        # Sort by enthalpy, take top N
        gen_entries.sort(key=lambda e: e.enthalpy)
        n_refine = min(self._gp_guided_top_n, len(gen_entries))
        top_entries = gen_entries[:n_refine]

        n_improved = 0
        p_eV_A3 = self.pressure_GPa / _EV_A3_TO_GPA

        for entry in top_entries:
            calc = self.mlip_calc_factory()
            try:
                refined = gp_guided_relax(
                    entry.atoms, calc, bias, projector, self.fp_calc, cfg)

                # Compute enthalpy of refined structure
                refined.calc = self.mlip_calc_factory()
                e = refined.get_potential_energy()
                v = refined.get_volume()
                nat = len(refined)
                h = (e + p_eV_A3 * v) / nat
                e_pa = e / nat

                # Only add if meaningfully better (>1 meV/atom)
                if h < entry.enthalpy - 0.001:
                    meta = {"generation": gen, "gp_guided": True,
                            "gp_guided_from_h": float(entry.enthalpy)}
                    added = archive.add(refined, e_pa, h,
                                        self.pressure_GPa, metadata=meta)
                    if added:
                        n_improved += 1
                        logger.info("GP-guided: H %.4f → %.4f (Δ=%.1f meV)",
                                    entry.enthalpy, h,
                                    1000 * (h - entry.enthalpy))
            except Exception as exc:
                logger.warning("GP-guided failed for entry H=%.4f: %s",
                               entry.enthalpy, exc)
            finally:
                del calc
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass

        if n_improved > 0:
            print(f"  GP-guided: {n_improved}/{n_refine} improved")
        else:
            print(f"  GP-guided: 0/{n_refine} improved")

        return n_improved

    # ------------------------------------------------------------------
    # Legacy V(f) ranking (preserved for backward compatibility)
    # ------------------------------------------------------------------

    def _screen_and_select(self, candidates: List[Atoms],
                           bias: BiasPotential,
                           n_select: int) -> List[Atoms]:
        """Score candidates with V(f), return diverse top subset.

        1. Compute V(f) for all candidates.
        2. Sort by V ascending.
        3. Greedy farthest-point selection from top 2*n_select to get n_select.
        """
        scores = []
        fps_pooled = []
        for atoms in candidates:
            try:
                fp = self.fp_calc.get_fingerprints(atoms)
                fp_pooled = self.fp_calc.pool_with_std(fp)
                V = bias.evaluate(fp_pooled)
                scores.append(V)
                fps_pooled.append(fp_pooled)
            except Exception:
                scores.append(np.inf)
                fps_pooled.append(None)

        # Sort by V
        order = np.argsort(scores)
        pool_size = min(2 * n_select, len(order))
        pool_idx = order[:pool_size]

        # Filter out failed candidates
        pool_idx = [i for i in pool_idx if fps_pooled[i] is not None]
        if len(pool_idx) <= n_select:
            return [candidates[i] for i in pool_idx]

        # Greedy farthest-point selection
        pool_fps = np.array([fps_pooled[i] for i in pool_idx])
        selected = [0]
        for _ in range(n_select - 1):
            min_dists = np.full(len(pool_idx), np.inf)
            for si in selected:
                d = np.linalg.norm(pool_fps - pool_fps[si], axis=1)
                min_dists = np.minimum(min_dists, d)
            min_dists[selected] = -1.0
            selected.append(int(np.argmax(min_dists)))

        return [candidates[pool_idx[s]] for s in selected]

    def _screen_and_select_with_origins(
        self, candidates: List[Atoms], is_mutant: List[bool],
        bias: BiasPotential, n_select: int
    ) -> tuple:
        """V(f) ranking that correctly remaps mutant-origin tracking.

        Returns (selected_candidates, selected_is_mutant).
        """
        scores = []
        fps_pooled = []
        for atoms in candidates:
            try:
                fp = self.fp_calc.get_fingerprints(atoms)
                fp_pooled = self.fp_calc.pool_with_std(fp)
                V = bias.evaluate(fp_pooled)
                scores.append(V)
                fps_pooled.append(fp_pooled)
            except Exception:
                scores.append(np.inf)
                fps_pooled.append(None)

        order = np.argsort(scores)
        pool_size = min(2 * n_select, len(order))
        pool_idx = order[:pool_size]
        pool_idx = [i for i in pool_idx if fps_pooled[i] is not None]

        if len(pool_idx) <= n_select:
            return ([candidates[i] for i in pool_idx],
                    [is_mutant[i] for i in pool_idx])

        pool_fps = np.array([fps_pooled[i] for i in pool_idx])
        selected = [0]
        for _ in range(n_select - 1):
            min_dists = np.full(len(pool_idx), np.inf)
            for si in selected:
                d = np.linalg.norm(pool_fps - pool_fps[si], axis=1)
                min_dists = np.minimum(min_dists, d)
            min_dists[selected] = -1.0
            selected.append(int(np.argmax(min_dists)))

        final_idx = [pool_idx[s] for s in selected]
        return ([candidates[i] for i in final_idx],
                [is_mutant[i] for i in final_idx])

    # ------------------------------------------------------------------
    # HPC batch relaxation
    # ------------------------------------------------------------------

    def _relax_batch_hpc(self, candidates: List[Atoms],
                         archive: StructureArchive,
                         generation: int) -> int:
        """Submit candidates to HPC, poll, collect, add to archive."""
        jobs = self.hpc_relaxer.submit_batch(candidates, generation)
        jobs = self.hpc_relaxer.poll_and_collect(jobs)

        new_count = 0
        for job in jobs:
            if job.status != "completed":
                continue
            if job.atoms_out is None or job.energy is None:
                continue
            try:
                # Propagate provenance from atoms.info to archive metadata
                meta = {"generation": generation,
                        "struct_idx": job.struct_idx}
                provenance_keys = [
                    'origin', 'spacegroup', 'parent_enthalpy', 'parent_rank',
                    'mutation_strain_std', 'mutation_pos_std',
                    'fpj_target_d', 'fpj_strain_std',
                    'stagnation_count',
                    'finisher_applied', 'finisher_target',
                    'finisher_d_init', 'finisher_d_final',
                    'finisher_mode', 'finisher_repel',
                    'cawr_applied',
                    'fpj_has_momentum',
                ]
                for key in provenance_keys:
                    val = job.atoms_in.info.get(key)
                    if val is not None:
                        meta[key] = val
                added = archive.add(
                    job.atoms_out, job.energy, job.enthalpy,
                    self.pressure_GPa,
                    metadata=meta,
                )
                if added:
                    new_count += 1
                    # Track FP-space momentum for successful lineages
                    if self.enable_fpj_mutations:
                        self._update_momentum(job.atoms_in, archive)
            except Exception as exc:
                logger.warning("Failed to add job %s to archive: %s",
                               job.job_id, exc)

        return new_count

    # ------------------------------------------------------------------
    # Local relaxation (legacy)
    # ------------------------------------------------------------------

    def _relax_batch_local(self, candidates: List[Atoms],
                           archive: StructureArchive,
                           generation: int) -> int:
        """Relax candidates locally with pure MLIP (bootstrap)."""
        new_count = 0
        for ci, atoms in enumerate(candidates):
            try:
                atoms, h, e, v = self._mlip_quench(atoms)
                added = archive.add(atoms, e, h, self.pressure_GPa,
                                    metadata={"generation": generation})
                if added:
                    new_count += 1
            except Exception as exc:
                logger.warning("Bootstrap candidate %d failed: %s", ci, exc)
        return new_count

    def _relax_batch_local_with_bias(self, candidates: List[Atoms],
                                     archive: StructureArchive,
                                     gp: ExactGP, bias: BiasPotential,
                                     projector: ForceProjector,
                                     gen: int) -> int:
        """Legacy local mode: process + quench each candidate serially."""
        new_count = 0
        for ci, atoms in enumerate(candidates):
            try:
                atoms = self._process_candidate(atoms, gp, bias,
                                                 projector, gen)
            except Exception as exc:
                logger.warning("Candidate %d failed: %s", ci, exc)
                continue

            try:
                atoms, h, e, v = self._mlip_quench(atoms)
            except Exception as exc:
                logger.warning("Quench failed for candidate %d: %s", ci, exc)
                continue

            added = archive.add(atoms, e, h, self.pressure_GPa,
                                metadata={"generation": gen})
            if added:
                new_count += 1

        return new_count

    # ------------------------------------------------------------------
    # Surrogate flow (local, GP-only)
    # ------------------------------------------------------------------

    def _apply_surrogate_flow(self, candidates: List[Atoms],
                              gp: ExactGP, bias: BiasPotential,
                              projector: ForceProjector) -> List[Atoms]:
        """Apply surrogate flow to uncertain candidates.

        In HPC mode, all candidates above sigma_threshold get surrogate flow
        before being sent for HPC relaxation.  In local mode, this is called
        by _process_candidate for the flow-then-quench path.
        """
        flowed = []
        for atoms in candidates:
            try:
                fp = self.fp_calc.get_fingerprints(atoms)
                fp_pooled = self.fp_calc.pool_with_std(fp)
                _, sigma = gp.predict(fp_pooled)

                if sigma > self.sigma_threshold:
                    atoms = self._surrogate_flow(atoms, bias, projector)
            except Exception as exc:
                logger.debug("Surrogate flow failed: %s", exc)
            flowed.append(atoms)
        return flowed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_candidate(self, atoms: Atoms, gp: ExactGP,
                           bias: BiasPotential,
                           projector: ForceProjector,
                           gen: int) -> Atoms:
        """Flow or relax a single candidate based on GP confidence."""
        if gp.X_train is not None:
            fp = self.fp_calc.get_fingerprints(atoms)
            fp_pooled = self.fp_calc.pool_with_std(fp)
            _, sigma = gp.predict(fp_pooled)

            if sigma > self.sigma_threshold:
                atoms = self._surrogate_flow(atoms, bias, projector)
            else:
                atoms = self._biased_relax(atoms, bias, projector)
        return atoms

    def _generate_random(self, n: int) -> List[Atoms]:
        """Generate *n* random structures using pyXtal with random space groups.

        Uses pyXtal to generate crystallographically valid random structures
        with randomly selected compatible space groups, producing structures
        that are much more physically reasonable than naive random placement.
        """
        from pyxtal import pyxtal

        structures = []
        max_attempts = n * 10

        species = list(self.composition.keys())
        numIons = list(self.composition.values())

        # Pre-cache compatible space groups on first call
        if not hasattr(self, "_compatible_sgs"):
            self._compatible_sgs = self._find_compatible_sgs(species, numIons)
            logger.info("Found %d compatible space groups for %s",
                        len(self._compatible_sgs),
                        dict(zip(species, numIons)))

        attempts = 0
        while len(structures) < n and attempts < max_attempts:
            attempts += 1
            sg = int(np.random.choice(self._compatible_sgs))
            try:
                xtal = pyxtal()
                xtal.from_random(
                    dim=3,
                    group=sg,
                    species=species,
                    numIons=numIons,
                    factor=np.random.uniform(0.8, 1.2),
                )
                atoms = xtal.to_ase()
                if len(atoms) != self._nat:
                    continue
                dists = atoms.get_all_distances(mic=True)
                np.fill_diagonal(dists, np.inf)
                if dists.min() > self.min_dist_ang:
                    atoms.info['origin'] = 'random'
                    atoms.info['spacegroup'] = sg
                    structures.append(atoms)
            except Exception:
                continue

        if len(structures) < n:
            logger.warning("Only generated %d / %d structures "
                           "after %d attempts", len(structures), n,
                           max_attempts)
        return structures

    @staticmethod
    def _find_compatible_sgs(species: list, numIons: list) -> List[int]:
        """Find space groups compatible with the given composition."""
        from pyxtal import pyxtal

        compatible = []
        for sg in range(1, 231):
            try:
                xtal = pyxtal()
                xtal.from_random(3, sg, species, numIons, factor=1.0)
                if len(xtal.to_ase()) == sum(numIons):
                    compatible.append(sg)
            except Exception:
                pass
        if not compatible:
            compatible = [1]  # P1 always works
        return compatible

    def _surrogate_flow(self, atoms: Atoms, bias: BiasPotential,
                        projector: ForceProjector) -> Atoms:
        """Mode 1: pure surrogate flow — no MLIP calls."""
        calc = GPCalculator(bias, projector, self.fp_calc)

        traj = Trajectory(
            LangevinFlow(dt=self.flow_dt, temperature=self.flow_temperature),
            n_steps=self.n_flow_steps,
        )

        def force_func(a: Atoms) -> np.ndarray:
            a.calc = calc
            return a.get_forces()

        traj.run(atoms, force_func)
        return atoms

    def _biased_relax(self, atoms: Atoms, bias: BiasPotential,
                      projector: ForceProjector) -> Atoms:
        """Mode 2: MLIP + bias pre-relaxation (short, ~40 steps).

        Only used in local mode.
        """
        calc = CRISPCalculator(
            self.mlip_calc_factory(), bias, projector, self.fp_calc,
            mode="hybrid", lambda_0=1.0, sigma_0=0.5,
        )
        atoms.calc = calc
        p_eV_A3 = self.pressure_GPa / _EV_A3_TO_GPA
        ecf = CellFilter(atoms, scalar_pressure=p_eV_A3)
        opt = LBFGS(ecf, logfile=None)
        opt.run(fmax=0.1, steps=40)
        return atoms

    def _mlip_quench(self, atoms: Atoms) -> tuple:
        """Pure MLIP relaxation to the nearest local minimum.

        Returns (atoms, enthalpy_per_atom, energy_per_atom, volume_per_atom).
        Raises ValueError if the relaxed energy is unphysical (> +5 eV/at).

        Only used in local mode.
        """
        atoms.calc = self.mlip_calc_factory()
        p_eV_A3 = self.pressure_GPa / _EV_A3_TO_GPA
        ecf = CellFilter(atoms, scalar_pressure=p_eV_A3)
        opt = LBFGS(ecf, logfile=None)
        opt.run(fmax=0.05, steps=200)

        nat = len(atoms)
        e = atoms.get_potential_energy() / nat
        v = atoms.get_volume()
        h = (atoms.get_potential_energy() + p_eV_A3 * v) / nat

        if e > 5.0:
            raise ValueError(f"Unphysical energy after quench: {e:.2f} eV/at")

        return atoms, h, e, v / nat
