"""Cell-list (linked-cell) construction of the Verlet pair list in O(N).

The box is divided into nc^3 cubic cells, each at least r_list wide. Only pairs in the
same or adjacent cells are examined. Requires nc >= 3 so no neighbor cell is visited twice.
"""

import math

import numpy as np
from numba import njit


def n_cells_per_dim(box: float, r_list: float) -> int:
    return int(math.floor(box / r_list))


@njit(cache=True)
def _cell_pairs(pos, box, r_list, nc, fill, pair_i, pair_j):
    n = pos.shape[0]
    rl2 = r_list * r_list
    cell_size = box / nc
    head = np.full(nc * nc * nc, -1, np.int32)
    nxt = np.full(n, -1, np.int32)
    for i in range(n):
        cx = int(math.floor(pos[i, 0] / cell_size)) % nc
        cy = int(math.floor(pos[i, 1] / cell_size)) % nc
        cz = int(math.floor(pos[i, 2] / cell_size)) % nc
        c = (cx * nc + cy) * nc + cz
        nxt[i] = head[c]
        head[c] = i
    count = 0
    for cx in range(nc):
        for cy in range(nc):
            for cz in range(nc):
                c = (cx * nc + cy) * nc + cz
                for ox in range(-1, 2):
                    for oy in range(-1, 2):
                        for oz in range(-1, 2):
                            c2 = (((cx + ox) % nc) * nc + (cy + oy) % nc) * nc + (cz + oz) % nc
                            i = head[c]
                            while i != -1:
                                j = head[c2]
                                while j != -1:
                                    if j > i:
                                        dx = pos[i, 0] - pos[j, 0]
                                        dy = pos[i, 1] - pos[j, 1]
                                        dz = pos[i, 2] - pos[j, 2]
                                        dx -= box * math.floor(dx / box + 0.5)
                                        dy -= box * math.floor(dy / box + 0.5)
                                        dz -= box * math.floor(dz / box + 0.5)
                                        if dx * dx + dy * dy + dz * dz < rl2:
                                            if fill:
                                                pair_i[count] = i
                                                pair_j[count] = j
                                            count += 1
                                    j = nxt[j]
                                i = nxt[i]
    return count


def build_pairs_cells(pos, box, r_list):
    """Return (pair_i, pair_j) with i < j for all pairs closer than r_list."""
    nc = n_cells_per_dim(box, r_list)
    if nc < 3:
        raise ValueError("cell list needs at least 3 cells per dimension")
    pos = np.ascontiguousarray(pos, dtype=np.float64)
    dummy = np.empty(0, dtype=np.int32)
    count = _cell_pairs(pos, float(box), float(r_list), nc, False, dummy, dummy)
    pair_i = np.empty(count, dtype=np.int32)
    pair_j = np.empty(count, dtype=np.int32)
    _cell_pairs(pos, float(box), float(r_list), nc, True, pair_i, pair_j)
    return pair_i, pair_j
