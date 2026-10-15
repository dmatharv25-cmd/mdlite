"""Parallel in-place wrapping of positions into the periodic box [0, box).

Most atoms are already inside the box, so only out-of-range values are modified.
"""

import math

from numba import njit, prange


@njit(parallel=True, cache=True)
def wrap_inplace(pos, box):
    n = pos.shape[0]
    for i in prange(n):
        for k in range(3):
            x = pos[i, k]
            if x < 0.0 or x >= box:
                pos[i, k] = x - box * math.floor(x / box)
