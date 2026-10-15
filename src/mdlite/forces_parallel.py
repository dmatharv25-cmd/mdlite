"""Parallel force kernel over a Verlet pair list using per-chunk force buffers.

The pair list is split into nchunks contiguous slices. Each slice accumulates into its
own (n, 3) buffer, so threads never write to the same memory. A second parallel loop
over atoms then sums the buffers.

The buffers are cached between calls (reallocated only when n or nchunks changes) and
each thread zeroes its own slice inside the parallel loop. The cache is shared module
state, so the kernel must not be called from several Python threads at the same time.

The minimum image uses comparisons instead of divide + floor. This requires positions
to be wrapped into the box (all coordinate differences within +-1.5 box), which the
integrator guarantees by calling wrap() every step.
"""

import numba
import numpy as np
from numba import njit, prange

_buf_cache = {"buf": None}


@njit(parallel=True, cache=True)
def _kernel(pos, box, r_cut, pair_i, pair_j, nchunks, buf):
    n = pos.shape[0]
    m = pair_i.shape[0]
    rc2 = r_cut * r_cut
    hb = 0.5 * box
    inv6c = 1.0 / (rc2 * rc2 * rc2)
    shift = 4.0 * (inv6c * inv6c - inv6c)
    pots = np.zeros(nchunks)
    virs = np.zeros(nchunks)
    for t in prange(nchunks):
        for i in range(n):
            buf[t, i, 0] = 0.0
            buf[t, i, 1] = 0.0
            buf[t, i, 2] = 0.0
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
            if dx > hb:
                dx -= box
            elif dx < -hb:
                dx += box
            if dy > hb:
                dy -= box
            elif dy < -hb:
                dy += box
            if dz > hb:
                dz -= box
            elif dz < -hb:
                dz += box
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
    forces = np.empty((n, 3))
    for i in prange(n):
        fx = 0.0
        fy = 0.0
        fz = 0.0
        for t in range(nchunks):
            fx += buf[t, i, 0]
            fy += buf[t, i, 1]
            fz += buf[t, i, 2]
        forces[i, 0] = fx
        forces[i, 1] = fy
        forces[i, 2] = fz
    return forces, pots.sum(), virs.sum()


def forces_from_pairs_parallel(pos, box, r_cut, pair_i, pair_j, nchunks=None):
    if nchunks is None:
        nchunks = numba.get_num_threads()
    nchunks = int(nchunks)
    n = pos.shape[0]
    buf = _buf_cache["buf"]
    if buf is None or buf.shape != (nchunks, n, 3):
        buf = np.empty((nchunks, n, 3))
        _buf_cache["buf"] = buf
    return _kernel(pos, float(box), float(r_cut), pair_i, pair_j, nchunks, buf)
