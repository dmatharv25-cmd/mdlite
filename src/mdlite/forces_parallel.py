"""Parallel force kernel over a Verlet pair list using per-chunk force buffers.

The pair list is split into nchunks contiguous slices. Each slice accumulates into its
own (n, 3) buffer, so threads never write to the same memory. A second parallel loop
over atoms then sums the buffers.
"""

import math

import numba
import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def _kernel(pos, box, r_cut, pair_i, pair_j, nchunks):
    n = pos.shape[0]
    m = pair_i.shape[0]
    rc2 = r_cut * r_cut
    inv6c = 1.0 / (rc2 * rc2 * rc2)
    shift = 4.0 * (inv6c * inv6c - inv6c)
    buf = np.zeros((nchunks, n, 3))
    pots = np.zeros(nchunks)
    virs = np.zeros(nchunks)
    for t in prange(nchunks):
        lo = (np.int64(t) * m) // nchunks
        hi = ((np.int64(t) + 1) * m) // nchunks
        pot = 0.0
        vir = 0.0
        for k in range(lo, hi):
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
                buf[t, i, 0] += f * dx
                buf[t, i, 1] += f * dy
                buf[t, i, 2] += f * dz
                buf[t, j, 0] -= f * dx
                buf[t, j, 1] -= f * dy
                buf[t, j, 2] -= f * dz
                pot += 4.0 * (inv12 - inv6) - shift
                vir += f * r2
        pots[t] = pot
        virs[t] = vir
    forces = np.zeros((n, 3))
    for i in prange(n):
        for t in range(nchunks):
            forces[i, 0] += buf[t, i, 0]
            forces[i, 1] += buf[t, i, 1]
            forces[i, 2] += buf[t, i, 2]
    return forces, pots.sum(), virs.sum()


def forces_from_pairs_parallel(pos, box, r_cut, pair_i, pair_j, nchunks=None):
    if nchunks is None:
        nchunks = numba.get_num_threads()
    return _kernel(pos, float(box), float(r_cut), pair_i, pair_j, int(nchunks))
