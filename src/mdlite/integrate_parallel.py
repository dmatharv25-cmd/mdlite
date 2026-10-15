"""Fused parallel kernels for the velocity Verlet step.

kick_drift_wrap does half kick, drift and wrap in one pass over the atoms; half_kick is
the second half kick. The arithmetic matches the NumPy expressions in run_nve
(v += (0.5*dt)*f, x += dt*v, wrap), so results are identical. max_displacement2_parallel
is the parallel version of neighbor._max_displacement2.
"""

import math

import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def kick_drift_wrap(pos, vel, forces, dt, box):
    n = pos.shape[0]
    h = 0.5 * dt
    for i in prange(n):
        for k in range(3):
            v = vel[i, k] + h * forces[i, k]
            vel[i, k] = v
            x = pos[i, k] + dt * v
            if x < 0.0 or x >= box:
                x = x - box * math.floor(x / box)
            pos[i, k] = x


@njit(parallel=True, cache=True)
def half_kick(vel, forces, dt):
    n = vel.shape[0]
    h = 0.5 * dt
    for i in prange(n):
        for k in range(3):
            vel[i, k] += h * forces[i, k]


@njit(parallel=True, cache=True)
def max_displacement2_parallel(pos, ref, box):
    n = pos.shape[0]
    nch = 64
    part = np.zeros(nch)
    for t in prange(nch):
        lo = (np.int64(t) * n) // nch
        hi = ((np.int64(t) + 1) * n) // nch
        m = 0.0
        for i in range(lo, hi):
            dx = pos[i, 0] - ref[i, 0]
            dy = pos[i, 1] - ref[i, 1]
            dz = pos[i, 2] - ref[i, 2]
            dx -= box * math.floor(dx / box + 0.5)
            dy -= box * math.floor(dy / box + 0.5)
            dz -= box * math.floor(dz / box + 0.5)
            d2 = dx * dx + dy * dy + dz * dz
            if d2 > m:
                m = d2
        part[t] = m
    r = 0.0
    for t in range(nch):
        if part[t] > r:
            r = part[t]
    return r
