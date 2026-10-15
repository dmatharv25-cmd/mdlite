import math
import time

import numba
import numpy as np
from numba import njit, prange

from mdlite.forces_parallel import forces_from_pairs_parallel as base_fn
from mdlite.integrators import run_nve
from mdlite.neighbor import VerletListForces
from mdlite.system import create_fcc_system


@njit(parallel=True)
def k_cond(pos, box, r_cut, pair_i, pair_j, nchunks, buf):
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


@njit(parallel=True)
def k_masked(pos, box, r_cut, pair_i, pair_j, nchunks, buf):
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
            msk = 1.0 if r2 < rc2 else 0.0
            inv2 = 1.0 / r2
            inv6 = inv2 * inv2 * inv2
            inv12 = inv6 * inv6
            f = msk * 24.0 * (2.0 * inv12 - inv6) * inv2
            buf[t, i, 0] += f * dx
            buf[t, i, 1] += f * dy
            buf[t, i, 2] += f * dz
            buf[t, j, 0] -= f * dx
            buf[t, j, 1] -= f * dy
            buf[t, j, 2] -= f * dz
            pot += msk * (4.0 * (inv12 - inv6) - shift)
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
n = len(pos)
bufs = {c: np.empty((c, n, 3)) for c in (6, 12, 24)}
pi0, pj0 = pi[:0].copy(), pj[:0].copy()


def run(kern, nch, a=pi, b=pj):
    return kern(pos, box, 2.5, a, b, nch, bufs[nch])


variants = [
    ("base (current) 12 chunks", lambda: base_fn(pos, box, 2.5, pi, pj)),
    ("cond-minimage 12", lambda: run(k_cond, 12)),
    ("cond-minimage 6", lambda: run(k_cond, 6)),
    ("cond-minimage 24", lambda: run(k_cond, 24)),
    ("masked (no cutoff branch) 12", lambda: run(k_masked, 12)),
    ("overhead only (0 pairs) 12", lambda: run(k_cond, 12, pi0, pj0)),
]

ref_f, ref_pot, ref_vir = base_fn(pos, box, 2.5, pi, pj)
print(f"N={n} pairs={len(pi)} threads={numba.get_num_threads()}")
for name, f in variants:
    out = f()
    if "overhead" in name:
        continue
    df = float(np.abs(out[0] - ref_f).max())
    print(f"check {name:<30} max|dF|={df:.2e}  dPot={abs(out[1] - ref_pot):.2e}  dVir={abs(out[2] - ref_vir):.2e}")


def timeit(f, reps=20):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t0)
    return 1000.0 * float(np.median(ts))


res = {name: [] for name, _ in variants}
for _ in range(3):
    for name, f in variants:
        res[name].append(timeit(f))
print()
print(f"{'variant':<32} {'median ms':>10} {'best round':>11}")
for name, r in res.items():
    print(f"{name:<32} {float(np.median(r)):>10.2f} {min(r):>11.2f}")
