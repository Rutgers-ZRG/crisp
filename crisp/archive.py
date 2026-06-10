"""Structure archive with fingerprint-based deduplication.

Stores relaxed structures together with their fingerprints, energies,
and enthalpies.  Supports duplicate detection, ranking, and diversity
selection for GP training and anchor/repulsion updates.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
from ase import Atoms
from ase.io import read as ase_read, write as ase_write

from .fingerprint import FingerprintCalculator

logger = logging.getLogger(__name__)


@dataclass
class ArchiveEntry:
    """A single structure stored in the archive."""
    atoms: Atoms
    fp: np.ndarray          # (nat, fp_dim) — per-atom fingerprints
    fp_pooled: np.ndarray   # (fp_dim,) — mean-pooled
    energy: float           # eV/atom
    enthalpy: float         # eV/atom (energy + PV)
    pressure: float         # GPa
    metadata: dict = field(default_factory=dict)
    generation: int = 0


class StructureArchive:
    """Store relaxed structures with fingerprints, energies, and deduplication.

    Parameters
    ----------
    fp_calc : FingerprintCalculator
        Fingerprint calculator instance.
    fp_threshold : float
        Minimum libfp Hungarian-matched distance to consider two structures
        distinct.  Uses full per-atom fingerprints (not pooled).
    energy_threshold : float
        Maximum energy difference (eV/atom) to consider two structures
        the same phase.  Both criteria must be met for dedup.
    """

    def __init__(self, fp_calc: FingerprintCalculator,
                 fp_threshold: float = 0.03,
                 energy_threshold: float = 0.01,
                 dup_threshold: float = None):
        self.fp_calc = fp_calc
        # Accept legacy dup_threshold for backward compat
        self.fp_threshold = dup_threshold if dup_threshold is not None else fp_threshold
        self.energy_threshold = energy_threshold
        self.entries: List[ArchiveEntry] = []

    def add(self, atoms: Atoms, energy_per_atom: float,
            enthalpy_per_atom: Optional[float] = None,
            pressure_GPa: float = 0.0,
            metadata: Optional[dict] = None) -> bool:
        """Add a structure if it is not a duplicate.

        Uses libfp Hungarian-matched distance on full per-atom fingerprints
        combined with energy difference for deduplication.

        Returns True if the structure was added, False if it was flagged
        as a duplicate.
        """
        fp = self.fp_calc.get_fingerprints(atoms)
        fp_pooled = self.fp_calc.pool_with_std(fp)
        types = self.fp_calc.atoms_to_cell(atoms)[2]

        for entry in self.entries:
            try:
                d_fp = self.fp_calc._fp_dist(fp, entry.fp, types)
            except Exception:
                d_fp = np.inf
            d_e = abs(energy_per_atom - entry.energy)
            if d_fp < self.fp_threshold and d_e < self.energy_threshold:
                logger.debug("Duplicate detected (d_fp=%.4f, dE=%.4f), "
                             "skipping", d_fp, d_e)
                return False

        if enthalpy_per_atom is None:
            enthalpy_per_atom = energy_per_atom

        entry = ArchiveEntry(
            atoms=atoms.copy(),
            fp=fp,
            fp_pooled=fp_pooled,
            energy=energy_per_atom,
            enthalpy=enthalpy_per_atom,
            pressure=pressure_GPa,
            metadata=metadata or {},
            generation=metadata.get("generation", 0) if metadata else 0,
        )
        self.entries.append(entry)
        logger.info("Archive: added structure #%d (E=%.4f, H=%.4f eV/at)",
                     len(self.entries), energy_per_atom, enthalpy_per_atom)
        return True

    def get_best(self, n: int = 5, key: str = "enthalpy") -> List[ArchiveEntry]:
        """Return the *n* lowest-energy (or -enthalpy) entries."""
        sorted_entries = sorted(self.entries,
                                key=lambda e: getattr(e, key))
        return sorted_entries[:n]

    def get_diverse(self, n: int = 3, pool: str = "best",
                    pool_size: int = 10) -> List[ArchiveEntry]:
        """Select *n* diverse entries via greedy farthest-point sampling.

        Parameters
        ----------
        pool : str
            ``'best'`` — select from top *pool_size* by enthalpy.
            ``'all'`` — select from all entries.
        pool_size : int
            How many top entries to consider when ``pool='best'``.
        """
        if pool == "best":
            candidates = self.get_best(pool_size, key="enthalpy")
        else:
            candidates = list(self.entries)

        if len(candidates) <= n:
            return candidates

        # Greedy farthest-point selection
        fps = np.array([e.fp_pooled for e in candidates])
        selected_idx = [0]  # start with lowest enthalpy
        for _ in range(n - 1):
            min_dists = np.full(len(candidates), np.inf)
            for si in selected_idx:
                dists = np.linalg.norm(fps - fps[si], axis=1)
                min_dists = np.minimum(min_dists, dists)
            min_dists[selected_idx] = -1.0  # exclude already selected
            selected_idx.append(int(np.argmax(min_dists)))

        return [candidates[i] for i in selected_idx]

    def get_all_pooled_fps(self) -> np.ndarray:
        """Return ``(N, fp_dim)`` array of pooled fingerprints for GP training."""
        if not self.entries:
            return np.empty((0, 0))
        return np.array([e.fp_pooled for e in self.entries])

    def get_all_energies(self) -> np.ndarray:
        """Return ``(N,)`` array of energies per atom."""
        return np.array([e.energy for e in self.entries])

    def get_all_enthalpies(self) -> np.ndarray:
        """Return ``(N,)`` array of enthalpies per atom."""
        return np.array([e.enthalpy for e in self.entries])

    def get_repulsion_centers(self) -> np.ndarray:
        """Return ``(N, fp_dim)`` of all pooled fps as repulsion centers."""
        return self.get_all_pooled_fps()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save archive to a directory: structures as extxyz, metadata as JSON."""
        dirpath = Path(path)
        dirpath.mkdir(parents=True, exist_ok=True)

        meta_list = []
        for i, entry in enumerate(self.entries):
            struct_file = dirpath / f"struct_{i:04d}.extxyz"
            ase_write(str(struct_file), entry.atoms, format="extxyz")
            meta_list.append({
                "energy": entry.energy,
                "enthalpy": entry.enthalpy,
                "pressure": entry.pressure,
                "metadata": entry.metadata,
                "generation": entry.generation,
                "fp_pooled": entry.fp_pooled.tolist(),
            })

        meta_file = dirpath / "archive_meta.json"
        with open(meta_file, "w") as f:
            json.dump(meta_list, f, indent=2)
        logger.info("Archive saved to %s (%d structures)", path, len(self.entries))

    def load(self, path: str) -> None:
        """Load archive from a directory previously created by ``save``."""
        dirpath = Path(path)
        meta_file = dirpath / "archive_meta.json"
        with open(meta_file) as f:
            meta_list = json.load(f)

        self.entries.clear()
        for i, meta in enumerate(meta_list):
            struct_file = dirpath / f"struct_{i:04d}.extxyz"
            atoms = ase_read(str(struct_file), format="extxyz")
            fp = self.fp_calc.get_fingerprints(atoms)
            fp_pooled = np.array(meta["fp_pooled"])
            entry = ArchiveEntry(
                atoms=atoms,
                fp=fp,
                fp_pooled=fp_pooled,
                energy=meta["energy"],
                enthalpy=meta["enthalpy"],
                pressure=meta["pressure"],
                metadata=meta.get("metadata", {}),
                generation=meta.get("generation", 0),
            )
            self.entries.append(entry)
        logger.info("Archive loaded from %s (%d structures)", path, len(self.entries))

    # ------------------------------------------------------------------
    # Checkpoint support (GP state serialization)
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str, gp=None, generation: int = 0,
                        extra_meta: Optional[dict] = None) -> None:
        """Save archive + GP state as a generation checkpoint.

        Creates ``path/gen_NNNN/`` with archive data and optional GP state.

        Parameters
        ----------
        path : str
            Base checkpoint directory.
        gp : ExactGP or None
            GP to serialize (X_train, K_inv, alpha, y_mean, y_std).
        generation : int
            Current generation number.
        """
        gen_dir = Path(path) / f"gen_{generation:04d}"
        self.save(str(gen_dir / "archive"))

        if gp is not None and gp.X_train is not None:
            gp_file = gen_dir / "gp_state.npz"
            np.savez(
                str(gp_file),
                X_train=gp.X_train,
                y_train=gp.y_train,
                K_inv=gp.K_inv,
                alpha=gp.alpha,
                y_mean=np.array([gp._y_mean]),
                y_std=np.array([gp._y_std]),
                length_scale=np.array([gp.length_scale]),
                signal_var=np.array([gp.signal_var]),
                noise=np.array([gp.noise]),
            )

        # Write generation metadata
        meta = {"generation": generation, "n_entries": len(self.entries)}
        if extra_meta:
            meta.update(extra_meta)
        with open(gen_dir / "checkpoint_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("Checkpoint saved: gen=%d, %d structures → %s",
                     generation, len(self.entries), gen_dir)

    def load_checkpoint(self, path: str, gp=None) -> int:
        """Load the latest generation checkpoint.

        Parameters
        ----------
        path : str
            Base checkpoint directory (contains ``gen_NNNN/`` subdirs).
        gp : ExactGP or None
            GP to restore state into.

        Returns
        -------
        int
            The generation number that was loaded.
        """
        base = Path(path)
        gen_dirs = sorted(base.glob("gen_*"))
        if not gen_dirs:
            raise FileNotFoundError(f"No checkpoints found in {path}")

        latest = gen_dirs[-1]
        self.load(str(latest / "archive"))

        if gp is not None:
            gp_file = latest / "gp_state.npz"
            if gp_file.exists():
                data = np.load(str(gp_file))
                gp.X_train = data["X_train"]
                gp.y_train = data["y_train"]
                gp.K_inv = data["K_inv"]
                gp.alpha = data["alpha"]
                gp._y_mean = float(data["y_mean"][0])
                gp._y_std = float(data["y_std"][0])
                gp.length_scale = float(data["length_scale"][0])
                gp.signal_var = float(data["signal_var"][0])
                gp.noise = float(data["noise"][0])
                logger.info("GP state restored (%d training points)",
                             gp.X_train.shape[0])

        with open(latest / "checkpoint_meta.json") as f:
            meta = json.load(f)
        self.last_checkpoint_meta = meta

        gen = meta["generation"]
        logger.info("Checkpoint loaded: gen=%d from %s", gen, latest)
        return gen
