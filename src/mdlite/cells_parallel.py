"""Parallel cell-list construction of the Verlet pair list.

Atoms are sorted into cells (counting sort, O(N)) and positions are copied into cell
order so the inner loops read contiguous memory. Each cell is compared with itself
(later atoms only) and with its 13 "forward" neighbor cells, so every pair is visited
exactly once. Two prange passes over cells count and then fill the pairs. Each atom
owns a disjoint slice of the output, so there are no races. Pairs are returned with
i < j. This needs at least 3 cells per dimension so the 26 neighbors are all distinct.
"""

import math

import numpy as np
from numba import njit, prange

from mdlite.cells import n_cells_per_dim


@njit(cache=True)
def _sort_into_cells(pos, box, nc):
    n = pos.shape[0]
    ncell = nc * nc * nc
    cell_size = box / nc
    cell_of_atom = np.empty(n, np.int64)
    start = np.zeros(ncell + 1, np.int64)
    for i in range(n):
        cx = int(math.floor(pos[i, 0] / cell_size)) % nc
        cy = int(math.floor(pos[i, 1] / cell_size)) % nc
        cz = int(math.floor(pos[i, 2] / cell_size)) % nc
        c = (cx * nc + cy) * nc + cz
        cell_of_atom[i] = c
        start[c + 1] += 1
    for c in range(ncell):
        start[c + 1] += start[c]
    fill = start[:-1].copy()
    order = np.empty(n, np.int64)
    for i in range(n):
        c = cell_of_atom[i]
        order[fill[c]] = i
        fill[c] += 1
    return order, start


@njit(parallel=True, cache=True)
def _pairs_kernel(pos, box, r_list, nc):
    n = pos.shape[0]
    ncell = nc * nc * nc
    rl2 = r_list * r_list
    order, start = _sort_into_cells(pos, box, nc)
    spos = np.empty((n, 3))
    for a in prange(n):
        o = order[a]
        spos[a, 0] = pos[o, 0]
        spos[a, 1] = pos[o, 1]
        spos[a, 2] = pos[o, 2]

    counts = np.zeros(n, np.int64)
    for c in prange(ncell):
        cx = c // (nc * nc)
        cy = (c // nc) % nc
        cz = c % nc
        nbr = np.empty(13, np.int64)
        m = 0
        for ox in range(-1, 2):
            for oy in range(-1, 2):
                for oz in range(-1, 2):
                    if ox > 0 or (ox == 0 and (oy > 0 or (oy == 0 and oz > 0))):
                        nbr[m] = (((cx + ox) % nc) * nc + (cy + oy) % nc) * nc + (cz + oz) % nc
                        m += 1
        cend = start[c + 1]
        for a in range(start[c], cend):
            xa = spos[a, 0]
            ya = spos[a, 1]
            za = spos[a, 2]
            cnt = 0
            for b in range(a + 1, cend):
                dx = xa - spos[b, 0]
                dy = ya - spos[b, 1]
                dz = za - spos[b, 2]
                dx -= box * math.floor(dx / box + 0.5)
                dy -= box * math.floor(dy / box + 0.5)
                dz -= box * math.floor(dz / box + 0.5)
                if dx * dx + dy * dy + dz * dz < rl2:
                    cnt += 1
            for q in range(13):
                c2 = nbr[q]
                for b in range(start[c2], start[c2 + 1]):
                    dx = xa - spos[b, 0]
                    dy = ya - spos[b, 1]
                    dz = za - spos[b, 2]
                    dx -= box * math.floor(dx / box + 0.5)
                    dy -= box * math.floor(dy / box + 0.5)
                    dz -= box * math.floor(dz / box + 0.5)
                    if dx * dx + dy * dy + dz * dz < rl2:
                        cnt += 1
            counts[a] = cnt

    offsets = np.zeros(n + 1, np.int64)
    for a in range(n):
        offsets[a + 1] = offsets[a] + counts[a]
    total = offsets[n]
    pair_i = np.empty(total, np.int32)
    pair_j = np.empty(total, np.int32)

    for c in prange(ncell):
        cx = c // (nc * nc)
        cy = (c // nc) % nc
        cz = c % nc
        nbr = np.empty(13, np.int64)
        m = 0
        for ox in range(-1, 2):
            for oy in range(-1, 2):
                for oz in range(-1, 2):
                    if ox > 0 or (ox == 0 and (oy > 0 or (oy == 0 and oz > 0))):
                        nbr[m] = (((cx + ox) % nc) * nc + (cy + oy) % nc) * nc + (cz + oz) % nc
                        m += 1
        cend = start[c + 1]
        for a in range(start[c], cend):
            oa = order[a]
            xa = spos[a, 0]
            ya = spos[a, 1]
            za = spos[a, 2]
            k = offsets[a]
            for b in range(a + 1, cend):
                dx = xa - spos[b, 0]
                dy = ya - spos[b, 1]
                dz = za - spos[b, 2]
                dx -= box * math.floor(dx / box + 0.5)
                dy -= box * math.floor(dy / box + 0.5)
                dz -= box * math.floor(dz / box + 0.5)
                if dx * dx + dy * dy + dz * dz < rl2:
                    ob = order[b]
                    if oa < ob:
                        pair_i[k] = oa
                        pair_j[k] = ob
                    else:
                        pair_i[k] = ob
                        pair_j[k] = oa
                    k += 1
            for q in range(13):
                c2 = nbr[q]
                for b in range(start[c2], start[c2 + 1]):
                    dx = xa - spos[b, 0]
                    dy = ya - spos[b, 1]
                    dz = za - spos[b, 2]
                    dx -= box * math.floor(dx / box + 0.5)
                    dy -= box * math.floor(dy / box + 0.5)
                    dz -= box * math.floor(dz / box + 0.5)
                    if dx * dx + dy * dy + dz * dz < rl2:
                        ob = order[b]
                        if oa < ob:
                            pair_i[k] = oa
                            pair_j[k] = ob
                        else:
                            pair_i[k] = ob
                            pair_j[k] = oa
                        k += 1
    return pair_i, pair_j


def build_pairs_cells_parallel(pos, box, r_list):
    """Return (pair_i, pair_j) with i < j for all pairs closer than r_list."""
    nc = n_cells_per_dim(box, r_list)
    if nc < 3:
        raise ValueError("cell list needs at least 3 cells per dimension")
    pos = np.ascontiguousarray(pos, dtype=np.float64)
    return _pairs_kernel(pos, float(box), float(r_list), nc)
