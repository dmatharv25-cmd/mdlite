"""Particle system: box, FCC lattice initialization, Maxwell-Boltzmann velocities.

All quantities are in reduced Lennard-Jones units (sigma = epsilon = m = k_B = 1).
"""

from dataclasses import dataclass

import numpy as np

from mdlite.wrap_parallel import wrap_inplace


@dataclass
class System:
    positions: np.ndarray   # shape (N, 3)
    velocities: np.ndarray  # shape (N, 3)
    box: float              # cubic box side length
    forces: np.ndarray = None

    def __post_init__(self):
        if self.forces is None:
            self.forces = np.zeros_like(self.positions)

    @property
    def n_particles(self) -> int:
        return self.positions.shape[0]

    def kinetic_energy(self) -> float:
        return 0.5 * float(np.sum(self.velocities ** 2))

    def temperature(self) -> float:
        # 3N - 3 degrees of freedom (total momentum is fixed at zero)
        return 2.0 * self.kinetic_energy() / (3.0 * (self.n_particles - 1))

    def wrap(self) -> None:
        """Wrap positions back into the periodic box [0, L)."""
        wrap_inplace(self.positions, float(self.box))


def fcc_lattice(n_cells: int, density: float):
    """Place 4 * n_cells^3 particles on an FCC lattice at the given number density.

    Returns (positions, box_length).
    """
    a = (4.0 / density) ** (1.0 / 3.0)  # conventional cell edge
    basis = np.array([[0.0, 0.0, 0.0],
                      [0.5, 0.5, 0.0],
                      [0.5, 0.0, 0.5],
                      [0.0, 0.5, 0.5]])
    idx = np.arange(n_cells)
    cells = np.array(np.meshgrid(idx, idx, idx, indexing="ij")).reshape(3, -1).T
    positions = (cells[:, None, :] + basis[None, :, :]).reshape(-1, 3) * a
    return positions, n_cells * a


def maxwell_boltzmann_velocities(n: int, temperature: float, rng) -> np.ndarray:
    """Gaussian velocities with zero net momentum, rescaled to exactly the target T."""
    v = rng.normal(0.0, np.sqrt(temperature), size=(n, 3))
    v -= v.mean(axis=0)                           # remove center-of-mass drift
    current_T = np.sum(v ** 2) / (3.0 * (n - 1))
    v *= np.sqrt(temperature / current_T)         # hit target T exactly
    return v


def create_fcc_system(n_cells: int, density: float, temperature: float, seed=None) -> System:
    rng = np.random.default_rng(seed)
    positions, box = fcc_lattice(n_cells, density)
    velocities = maxwell_boltzmann_velocities(len(positions), temperature, rng)
    return System(positions=positions, velocities=velocities, box=box)
