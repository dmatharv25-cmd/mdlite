"""Numba-compiled Lennard-Jones force kernel (serial, brute-force O(N^2) pair loop).

Uses Newton's third law: each pair (i, j > i) is computed once and applied to both atoms.
"""

import math

import numpy as np
from numba import njit


@njit(cache=True)
def _forces_kernel(pos, box, r_cut):
    n = pos.shape[0]
    forces = np.zeros((n, 3))
    rc2 = r_cut * r_cut
    inv6c = 1.0 / (rc2 * rc2 * rc2)
    shift = 4.0 * (inv6c * inv6c - inv6c)
    pot = 0.0
    vir = 0.0
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dz = pos[i, 2] - pos[j, 2]
            dx -= box * math.floor(dx / box + 0.5)   # minimum image
            dy -= box * math.floor(dy / box + 0.5)
            dz -= box * math.floor(dz / box + 0.5)
            r2 = dx * dx + dy * dy + dz * dz
            if r2 < rc2:
                inv2 = 1.0 / r2
                inv6 = inv2 * inv2 * inv2
                inv12 = inv6 * inv6
                f = 24.0 * (2.0 * inv12 - inv6) * inv2
                forces[i, 0] += f * dx
                forces[i, 1] += f * dy
                forces[i, 2] += f * dz
                forces[j, 0] -= f * dx
                forces[j, 1] -= f * dy
                forces[j, 2] -= f * dz
                pot += 4.0 * (inv12 - inv6) - shift
                vir += f * r2
    return forces, pot, vir


def compute_forces_numba(positions: np.ndarray, box: float, r_cut: float = 2.5):
    """Same interface and return values as compute_forces_brute."""
    pos = np.ascontiguousarray(positions, dtype=np.float64)
    forces, pot, vir = _forces_kernel(pos, float(box), float(r_cut))
    return forces, float(pot), float(vir)
