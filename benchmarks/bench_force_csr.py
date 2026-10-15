import time

import numba
import numpy as np
from numba import njit, prange
import math

from mdlite.forces_parallel import forces_from_pairs_parallel
from mdlite.integrators import run_nve
from mdlite.neighbor import VerletListForces
from mdlite.system import create_fcc_system


def build_csr(pair_i, pair_j, n):
    src = np.concatenate([pair_i, pair_j]).astype(np.int32)
    dst = np.concatenate([pair_j, pair_i]).astype(np.int32)
    order = np.argsort(src, kind="stable")
    dst = dst[order]
    counts = np.bincount(src, minlength=n)
    start = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=start[1:])
    return start, dst


@njit(parallel=True, cache=True)
def forces_csr(pos, box, r_cut, start, nbrs):
    n = pos.shape[0]
    rc2 = r_cut * r_cut
    inv6c = 1.0 / (rc2 * rc2 * rc2)
    shift = 4.0 * (inv6c * inv6c - inv6c)
    forces = np.zeros((n, 3))
    pots = np.zeros(n)
    virs = np.zeros(n)
    for i in prange(n):
        fx = 0.0
        fy = 0.0
        fz = 0.0
        pot = 0.0
        vir = 0.0
        xi = pos[i, 0]
        yi = pos[i, 1]
        zi = pos[i, 2]
        for k in range(start[i], start[i + 1]):
            j = nbrs[k]
            dx = xi - pos[j, 0]
            dy = yi - pos[j, 1]
            dz = zi - pos[j, 2]
            dx -= box * math.floor(dx / box + 0.5)
            dy -= box * math.floor(dy / box + 0.5)
            dz -= box * math.floor(dz / box + 0.5)
            r2 = dx * dx + dy * dy + dz * dz
            if r2 < rc2:
                inv2 = 1.0 / r2
                inv6 = inv2 * inv2 * inv2
                inv12 = inv6 * inv6
                f = 24.0 * (2.0 * inv12 - inv6) * inv2
                fx += f * dx
                fy += f * dy
                fz += f * dz
                pot += 0.5 * (4.0 * (inv12 - inv6) - shift)
                vir += 0.5 * f * r2
        forces[i, 0] = fx
        forces[i, 1] = fy
        forces[i, 2] = fz
        pots[i] = pot
        virs[i] = vir
    return forces, pots.sum(), virs.sum()


s = create_fcc_system(22, 0.8, 1.0, seed=1)
fn = VerletListForces(0.5, use_cells=True, parallel=True)
run_nve(s, 200, 0.005, force_fn=fn)
pos = np.ascontiguousarray(s.positions)
box = float(s.box)
pi, pj = fn.pair_i, fn.pair_j
n = pos.shape[0]

t0 = time.perf_counter()
start, nbrs = build_csr(pi, pj, n)
print(f"N={n} half pairs={len(pi)} full entries={len(nbrs)}  CSR build (numpy): {1000*(time.perf_counter()-t0):.1f} ms")


def med(f, reps=30):
    f()
    f()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t0)
    return 1000.0 * float(np.median(ts)), 1000.0 * min(ts)


old = med(lambda: forces_from_pairs_parallel(pos, box, 2.5, pi, pj))
new = med(lambda: forces_csr(pos, box, 2.5, start, nbrs))
print(f"current (buffers) : median {old[0]:6.2f} ms  min {old[1]:6.2f}")
print(f"CSR full list     : median {new[0]:6.2f} ms  min {new[1]:6.2f}")
print(f"speedup: {old[0] / new[0]:.2f}x")

fo, po, vo = forces_from_pairs_parallel(pos, box, 2.5, pi, pj)
fnw, pn, vn = forces_csr(pos, box, 2.5, start, nbrs)
print(f"max |dF| = {np.abs(fo - fnw).max():.2e}  dPot = {abs(po - pn):.2e}  dVir = {abs(vo - vn):.2e}")
