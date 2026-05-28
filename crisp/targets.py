"""Target library for FP-targeted finishers.

Provides structural targets (per-atom fingerprints) that the FP-targeted
finisher steers candidates toward. Targets can come from:
1. Known phases (user-supplied reference structures)
2. Archive centroids (k-medoids clustering of relaxed structures)

The library is updated periodically as the archive grows.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from ase import Atoms

from .fingerprint import FingerprintCalculator

logger = logging.getLogger(__name__)


@dataclass
class Target:
    """A structural target in fingerprint space."""
    fp: np.ndarray          # (nat, fp_dim) per-atom fingerprints
    types: np.ndarray       # (nat,) 1-indexed type array
    label: str = ""
    enthalpy: Optional[float] = None
    source: str = ""        # "known_phase" or "archive_centroid"


class TargetLibrary:
    """Manages structural targets for FP-targeted finishers.

    Parameters
    ----------
    fp_calc : FingerprintCalculator
        Fingerprint calculator.
    n_targets : int
        Maximum number of archive centroid targets to maintain.
    """

    def __init__(self, fp_calc: FingerprintCalculator, n_targets: int = 12):
        self.fp_calc = fp_calc
        self.n_targets = n_targets
        self.targets: List[Target] = []
        self._known_phases: List[Target] = []

    def add_known_phase(self, atoms: Atoms, label: str = "") -> None:
        """Register a known phase as a permanent target.

        Known phases are always included and never removed during updates.
        """
        fp = self.fp_calc.get_fingerprints(atoms)
        types = self.fp_calc.atoms_to_cell(atoms)[2]
        target = Target(fp=fp, types=types, label=label, source="known_phase")
        self._known_phases.append(target)
        self.targets.append(target)
        logger.info("Added known phase target: %s", label)

    def update_from_archive(self, archive) -> None:
        """Update targets from archive using k-medoids on pooled FPs.

        Selects diverse, low-enthalpy structures as centroid targets.
        Preserves known phases and replaces archive-derived targets.

        Parameters
        ----------
        archive : StructureArchive
            The current structure archive.
        """
        if len(archive.entries) < 2:
            return

        # Keep known phases, replace archive centroids
        self.targets = list(self._known_phases)

        n_archive_targets = self.n_targets - len(self._known_phases)
        if n_archive_targets <= 0:
            return

        # Use greedy farthest-point from low-enthalpy pool as centroids
        # (same strategy as archive.get_diverse but returns per-atom FPs)
        sorted_entries = sorted(archive.entries, key=lambda e: e.enthalpy)
        pool_size = min(len(sorted_entries), max(n_archive_targets * 3, 20))
        pool = sorted_entries[:pool_size]

        if len(pool) <= n_archive_targets:
            centroids = pool
        else:
            centroids = self._farthest_point_select(pool, n_archive_targets)

        for entry in centroids:
            types = self.fp_calc.atoms_to_cell(entry.atoms)[2]
            target = Target(
                fp=entry.fp,
                types=types,
                label=f"centroid_H={entry.enthalpy:.4f}",
                enthalpy=entry.enthalpy,
                source="archive_centroid",
            )
            self.targets.append(target)

        logger.info("Target library updated: %d known + %d centroids = %d total",
                     len(self._known_phases), len(centroids), len(self.targets))

    def get_nearest_target(self, fp: np.ndarray,
                           types: np.ndarray,
                           prefer_known: bool = False) -> Optional[Target]:
        """Find the target closest to the given per-atom FPs.

        Uses pooled L2 distance for speed (not Hungarian).
        Known phases get a 2x distance bonus (distances halved) so they
        are selected more often than archive centroids.

        Parameters
        ----------
        fp : np.ndarray, shape (nat, fp_dim)
            Candidate per-atom fingerprints.
        types : np.ndarray, shape (nat,)
            1-indexed type array.
        prefer_known : bool
            If True, prefer known phases (nearest one). Falls back to
            archive centroids if no known phases are available.

        Returns
        -------
        Target or None
            Nearest target, or None if library is empty.
        """
        if not self.targets:
            return None

        fp_pooled = np.mean(fp, axis=0)
        best_dist = np.inf
        best_target = None

        # If prefer_known, first try known phases only
        if prefer_known and self._known_phases:
            for target in self._known_phases:
                if len(target.fp) != len(fp):
                    continue
                tgt_pooled = np.mean(target.fp, axis=0)
                d = np.linalg.norm(fp_pooled - tgt_pooled)
                if d < best_dist:
                    best_dist = d
                    best_target = target
            if best_target is not None:
                return best_target

        # Fall back to all targets (known phases get 2x distance bonus)
        for target in self.targets:
            if len(target.fp) != len(fp):
                continue
            tgt_pooled = np.mean(target.fp, axis=0)
            d = np.linalg.norm(fp_pooled - tgt_pooled)
            if target.source == "known_phase":
                d *= 0.5
            if d < best_dist:
                best_dist = d
                best_target = target

        return best_target

    def get_random_from_topk(self, fp: np.ndarray,
                             types: np.ndarray,
                             k: int = 3) -> Optional[Target]:
        """Return a random target from the k nearest (diverse exploration).

        Used during stagnation to break mode collapse — different candidates
        get different targets instead of all converging to the nearest one.
        """
        if not self.targets:
            return None

        fp_pooled = np.mean(fp, axis=0)
        distances = []
        valid_targets = []

        for target in self.targets:
            if len(target.fp) != len(fp):
                continue
            tgt_pooled = np.mean(target.fp, axis=0)
            d = np.linalg.norm(fp_pooled - tgt_pooled)
            distances.append(d)
            valid_targets.append(target)

        if not valid_targets:
            return None

        order = np.argsort(distances)
        topk = min(k, len(valid_targets))
        chosen_idx = order[np.random.randint(topk)]
        return valid_targets[chosen_idx]

    def min_target_distance(self, fp: np.ndarray) -> float:
        """Minimum pooled-FP L2 distance to any target.

        Used for finisher gating decisions.
        """
        if not self.targets:
            return np.inf

        fp_pooled = np.mean(fp, axis=0)
        dists = []
        for target in self.targets:
            if len(target.fp) != len(fp):
                continue
            tgt_pooled = np.mean(target.fp, axis=0)
            dists.append(np.linalg.norm(fp_pooled - tgt_pooled))

        return min(dists) if dists else np.inf

    @staticmethod
    def _farthest_point_select(pool, n: int) -> list:
        """Greedy farthest-point selection on pooled FPs."""
        fps = np.array([e.fp_pooled for e in pool])
        selected = [0]

        for _ in range(n - 1):
            min_dists = np.full(len(pool), np.inf)
            for si in selected:
                d = np.linalg.norm(fps - fps[si], axis=1)
                min_dists = np.minimum(min_dists, d)
            min_dists[selected] = -1.0
            selected.append(int(np.argmax(min_dists)))

        return [pool[i] for i in selected]
