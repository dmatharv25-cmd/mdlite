"""Lennard-Jones forces: brute-force O(N^2) reference implementation (pure NumPy).

Reduced units (sigma = epsilon = 1). Shifted potential so U(r_c) = 0.
"""

import numpy as np


def lj_shift(r_cut: float) -> float:
    """Value of the unshifted LJ potential at the cutoff."""
    inv6 = r_cut ** -6
    return 4.0 * (inv6 * inv6 - inv6)


def compute_forces_brute(positions: np.ndarray, box: float, r_cut: float = 2.5):
    """Compute LJ forces, potential energy and virial with periodic boundaries.

    Returns (forces, potential_energy, virial) where virial = sum over pairs of r_ij . f_ij.
    """
    n = positions.shape[0]
    d = positions[:, None, :] - positions[None, :, :]   # (N, N, 3) separation vectors
    d -= box * np.round(d / box)                         # minimum image convention
    r2 = np.sum(d * d, axis=-1)                          # (N, N) squared distances
    np.fill_diagonal(r2, np.inf)                         # exclude self-interaction

    mask = r2 < r_cut * r_cut
    inv_r2 = np.where(mask, 1.0 / r2, 0.0)
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 * inv_r6

    # F_ij = 24 (2 r^-12 - r^-6) / r^2 * d_ij
    f_over_r = 24.0 * (2.0 * inv_r12 - inv_r6) * inv_r2
    forces = np.sum(f_over_r[:, :, None] * d, axis=1)

    shift = lj_shift(r_cut)
    pair_u = np.where(mask, 4.0 * (inv_r12 - inv_r6) - shift, 0.0)
    potential = 0.5 * np.sum(pair_u)                     # each pair counted twice
    virial = 0.5 * np.sum(f_over_r * r2 * mask)          # sum_pairs r . f
    return forces, float(potential), float(virial)
