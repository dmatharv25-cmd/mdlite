import time
import math

import numba
import numpy as np
from numba import njit, prange

from mdlite.forces_parallel import forces_from_pairs_parallel
from mdlite.integrators import run_nve
from mdlite.neighbor import VerletListForces
from mdlite.system import create_fcc_system


@njit(parallel=True, cache=True)
def kernel_persist(pos, box, r_cut, pair_i, pair_j, nchunks, buf):
    n = pos.shape[0]
    m = pair_i.shape[0]
    rc2 = r_cut * r_cut
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


s = create_fcc_system(22, 0.8, 1.0, seed=1)
fn = VerletListForces(0.5, use_cells=True, parallel=True)
run_nve(s, 200, 0.005, force_fn=fn)
pos = np.ascontiguousarray(s.positions)
box = float(s.box)
pi, pj = fn.pair_i, fn.pair_j
n = pos.shape[0]
nt = numba.get_num_threads()


def med(f, reps=40):
    f()
    f()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t0)
    return 1000.0 * float(np.median(ts)), 1000.0 * min(ts)


empty = np.empty(0, dtype=np.int32)
print(f"N={n} pairs={len(pi)} threads={nt}")

# interleave old and new so drift hits both equally
bufs = {nc: np.zeros((nc, n, 3)) for nc in (nt, 2 * nt)}
rows = {"current (alloc each call)": [], "persistent, nchunks=12": [], "persistent, nchunks=24": []}
for rnd in range(3):
    rows["current (alloc each call)"].append(med(lambda: forces_from_pairs_parallel(pos, box, 2.5, pi, pj)))
    rows["persistent, nchunks=12"].append(med(lambda: kernel_persist(pos, box, 2.5, pi, pj, nt, bufs[nt])))
    rows["persistent, nchunks=24"].append(med(lambda: kernel_persist(pos, box, 2.5, pi, pj, 2 * nt, bufs[2 * nt])))
print(f"\n{'variant':<28} {'median of 3 rounds':>20} {'best min':>10}")
for name, r in rows.items():
    print(f"{name:<28} {np.median([x[0] for x in r]):>17.2f} ms {min(x[1] for x in r):>7.2f} ms")

e_old = med(lambda: forces_from_pairs_parallel(pos, box, 2.5, empty, empty))
e_new = med(lambda: kernel_persist(pos, box, 2.5, empty, empty, nt, bufs[nt]))
print(f"\nempty list: current {e_old[0]:.2f} ms, persistent {e_new[0]:.2f} ms")

fo, po, vo = forces_from_pairs_parallel(pos, box, 2.5, pi, pj)
fnw, pn, vn = kernel_persist(pos, box, 2.5, pi, pj, nt, bufs[nt])
print(f"max |dF| = {np.abs(fo - fnw).max():.2e}  dPot = {abs(po - pn):.2e}  dVir = {abs(vo - vn):.2e}")
