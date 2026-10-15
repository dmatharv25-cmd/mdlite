"""Verlet neighbor lists: pair list within r_cut + skin, rebuilt lazily.

The list is rebuilt when any atom has moved more than skin/2 since the last build,
which guarantees no pair can enter the cutoff sphere without being in the list.
"""

import math

import numpy as np
from mdlite.cells import build_pairs_cells, n_cells_per_dim
from mdlite.forces_parallel import forces_from_pairs_parallel
from mdlite.cells_parallel import build_pairs_cells_parallel
from mdlite.integrate_parallel import max_displacement2_parallel
from numba import njit


@njit(cache=True)
def _build_pairs(pos, box, r_list):
    n = pos.shape[0]
    rl2 = r_list * r_list
    count = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dz = pos[i, 2] - pos[j, 2]
            dx -= box * math.floor(dx / box + 0.5)
            dy -= box * math.floor(dy / box + 0.5)
            dz -= box * math.floor(dz / box + 0.5)
            if dx * dx + dy * dy + dz * dz < rl2:
                count += 1
    pair_i = np.empty(count, dtype=np.int32)
    pair_j = np.empty(count, dtype=np.int32)
    k = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dz = pos[i, 2] - pos[j, 2]
            dx -= box * math.floor(dx / box + 0.5)
            dy -= box * math.floor(dy / box + 0.5)
            dz -= box * math.floor(dz / box + 0.5)
            if dx * dx + dy * dy + dz * dz < rl2:
                pair_i[k] = i
                pair_j[k] = j
                k += 1
    return pair_i, pair_j


@njit(cache=True)
def _max_displacement2(pos, ref, box):
    m = 0.0
    for i in range(pos.shape[0]):
        dx = pos[i, 0] - ref[i, 0]
        dy = pos[i, 1] - ref[i, 1]
        dz = pos[i, 2] - ref[i, 2]
        dx -= box * math.floor(dx / box + 0.5)
        dy -= box * math.floor(dy / box + 0.5)
        dz -= box * math.floor(dz / box + 0.5)
        d2 = dx * dx + dy * dy + dz * dz
        if d2 > m:
            m = d2
    return m


@njit(cache=True)
def _forces_from_pairs(pos, box, r_cut, pair_i, pair_j):
    n = pos.shape[0]
    forces = np.zeros((n, 3))
    rc2 = r_cut * r_cut
    inv6c = 1.0 / (rc2 * rc2 * rc2)
    shift = 4.0 * (inv6c * inv6c - inv6c)
    pot = 0.0
    vir = 0.0
    for k in range(pair_i.shape[0]):
        i = pair_i[k]
        j = pair_j[k]
        dx = pos[i, 0] - pos[j, 0]
        dy = pos[i, 1] - pos[j, 1]
        dz = pos[i, 2] - pos[j, 2]
        dx -= box * math.floor(dx / box + 0.5)
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


class VerletListForces:
    """Callable force function with the same interface as compute_forces_brute.

    Holds the neighbor list as state, so create one instance per simulation.
    """

    def __init__(self, skin: float = 0.5, use_cells: bool = True, parallel: bool = False):
        self.skin = float(skin)
        self.use_cells = bool(use_cells)
        self.parallel = bool(parallel)
        self.pair_i = None
        self.pair_j = None
        self._ref = None
        self._box = None
        self._r_cut = None
        self.n_builds = 0

    def _needs_rebuild(self, pos, box, r_cut) -> bool:
        if self._ref is None or self._ref.shape != pos.shape:
            return True
        if box != self._box or r_cut != self._r_cut:
            return True
        half_skin = 0.5 * self.skin
        if self.parallel:
            disp2 = max_displacement2_parallel(pos, self._ref, box)
        else:
            disp2 = _max_displacement2(pos, self._ref, box)
        return disp2 > half_skin * half_skin

    def _build(self, pos, box, r_cut):
        r_list = r_cut + self.skin
        if self.use_cells and n_cells_per_dim(box, r_list) >= 3:
            if self.parallel:
                self.pair_i, self.pair_j = build_pairs_cells_parallel(pos, box, r_list)
            else:
                self.pair_i, self.pair_j = build_pairs_cells(pos, box, r_list)
        else:
            self.pair_i, self.pair_j = _build_pairs(pos, box, r_list)
        self._ref = pos.copy()
        self._box = box
        self._r_cut = r_cut
        self.n_builds += 1

    def __call__(self, positions, box, r_cut: float = 2.5):
        pos = np.ascontiguousarray(positions, dtype=np.float64)
        box = float(box)
        r_cut = float(r_cut)
        if box < 2.0 * (r_cut + self.skin):
            raise ValueError("box must be at least 2 * (r_cut + skin) for the minimum image convention")
        if self._needs_rebuild(pos, box, r_cut):
            self._build(pos, box, r_cut)
        if self.parallel:
            forces, pot, vir = forces_from_pairs_parallel(pos, box, r_cut, self.pair_i, self.pair_j)
        else:
            forces, pot, vir = _forces_from_pairs(pos, box, r_cut, self.pair_i, self.pair_j)
        return forces, float(pot), float(vir)
