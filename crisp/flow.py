"""Integrators and trajectory management for flow-based search.

Provides overdamped Langevin dynamics and FIRE for flowing structures
downhill in the bias potential landscape.
"""

import logging
from typing import Callable, List

import numpy as np
from ase import Atoms

logger = logging.getLogger(__name__)


class Integrator:
    """Base class for flow integrators."""

    def step(self, atoms: Atoms, forces: np.ndarray) -> None:
        """Advance the atomic positions by one step."""
        raise NotImplementedError


class LangevinFlow(Integrator):
    """Overdamped Langevin dynamics: dx = dt*F + sqrt(2*dt*T)*noise.

    Parameters
    ----------
    dt : float
        Time step (Angstrom / (eV/Angstrom) units, effectively).
    temperature : float
        Noise temperature (controls exploration; 0 = deterministic).
    """

    def __init__(self, dt: float = 0.01, temperature: float = 0.1):
        self.dt = dt
        self.temperature = temperature

    def step(self, atoms: Atoms, forces: np.ndarray) -> None:
        """Advance positions by one Langevin step."""
        displacement = self.dt * forces
        if self.temperature > 0:
            noise_scale = np.sqrt(2.0 * self.dt * self.temperature)
            displacement += noise_scale * np.random.randn(*forces.shape)
        atoms.positions += displacement


class FIREFlow(Integrator):
    """FIRE-like adaptive integrator for faster convergence.

    Adapts the effective step size by tracking the power (F dot v).
    When power is positive (moving downhill), speed up; when negative,
    slow down and reset velocity.

    Parameters
    ----------
    dt_start : float
        Initial time step.
    dt_max : float
        Maximum allowed time step.
    f_inc : float
        Factor to increase dt when power > 0.
    f_dec : float
        Factor to decrease dt on power < 0.
    alpha_start : float
        Initial mixing parameter for velocity + force.
    f_alpha : float
        Factor to decrease alpha when power > 0.
    n_min : int
        Number of consecutive positive-power steps before increasing dt.
    """

    def __init__(self, dt_start: float = 0.01, dt_max: float = 0.1,
                 f_inc: float = 1.1, f_dec: float = 0.5,
                 alpha_start: float = 0.1, f_alpha: float = 0.99,
                 n_min: int = 5):
        self.dt = dt_start
        self.dt_max = dt_max
        self.f_inc = f_inc
        self.f_dec = f_dec
        self.alpha_start = alpha_start
        self.f_alpha = f_alpha
        self.n_min = n_min

        # Internal state
        self._v = None
        self._alpha = alpha_start
        self._n_pos = 0

    def step(self, atoms: Atoms, forces: np.ndarray) -> None:
        """Advance positions by one FIRE step."""
        if self._v is None:
            self._v = np.zeros_like(forces)

        # Compute power
        power = np.sum(self._v * forces)

        if power > 0:
            self._n_pos += 1
            if self._n_pos > self.n_min:
                self.dt = min(self.dt * self.f_inc, self.dt_max)
                self._alpha *= self.f_alpha
            # FIRE velocity mixing: v = (1-alpha)*v + alpha*|v|*F_hat
            f_norm = np.linalg.norm(forces)
            v_norm = np.linalg.norm(self._v)
            if f_norm > 1e-20:
                self._v = ((1.0 - self._alpha) * self._v
                           + self._alpha * v_norm * forces / f_norm)
        else:
            # Reset
            self._v[:] = 0.0
            self.dt *= self.f_dec
            self._alpha = self.alpha_start
            self._n_pos = 0

        # Verlet-like update
        self._v += self.dt * forces
        atoms.positions += self.dt * self._v

    def reset(self) -> None:
        """Reset internal state (call between independent trajectories)."""
        self._v = None
        self._alpha = self.alpha_start
        self._n_pos = 0


class Trajectory:
    """Manage a flow trajectory: run N steps, collect snapshots.

    Parameters
    ----------
    integrator : Integrator
        Flow integrator instance.
    n_steps : int
        Number of integration steps.
    snapshot_interval : int
        Save a snapshot every this many steps.
    """

    def __init__(self, integrator: Integrator, n_steps: int = 50,
                 snapshot_interval: int = 10):
        self.integrator = integrator
        self.n_steps = n_steps
        self.snapshot_interval = snapshot_interval

    def run(self, atoms: Atoms,
            force_func: Callable[[Atoms], np.ndarray]) -> List[Atoms]:
        """Run the trajectory.

        Parameters
        ----------
        atoms : ase.Atoms
            Starting structure (modified in-place).
        force_func : callable
            ``force_func(atoms) -> forces`` array of shape ``(nat, 3)``.

        Returns
        -------
        list of ase.Atoms
            Snapshots (copies) at intervals and at the end.
        """
        snapshots = [atoms.copy()]
        for step in range(self.n_steps):
            forces = force_func(atoms)
            self.integrator.step(atoms, forces)
            if (step + 1) % self.snapshot_interval == 0:
                snapshots.append(atoms.copy())
        # Always include the final state
        if self.n_steps % self.snapshot_interval != 0:
            snapshots.append(atoms.copy())
        return snapshots
