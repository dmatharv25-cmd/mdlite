import time

import numba
import numpy as np

from mdlite.forces_parallel import forces_from_pairs_parallel
from mdlite.integrators import run_nve
from mdlite.neighbor import VerletListForces, _forces_from_pairs
from mdlite.system import create_fcc_system

s = create_fcc_system(22, 0.8, 1.0, seed=1)
fn = VerletListForces(0.5, use_cells=True, parallel=True)
run_nve(s, 200, 0.005, force_fn=fn)
pos = np.ascontiguousarray(s.positions)
box = float(s.box)
pi, pj = fn.pair_i, fn.pair_j
n = pos.shape[0]
print(f"N={n} pairs={len(pi)} numba threads={numba.get_num_threads()}")

def median_ms(f, reps=40):
    f()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t0)
    return 1000.0 * float(np.median(ts)), 1000.0 * min(ts)

par = median_ms(lambda: forces_from_pairs_parallel(pos, box, 2.5, pi, pj))
ser = median_ms(lambda: _forces_from_pairs(pos, box, 2.5, pi, pj))
zer = median_ms(lambda: np.zeros((n, 3)))
print(f"parallel force : median {par[0]:7.2f} ms  min {par[1]:7.2f}")
print(f"serial force   : median {ser[0]:7.2f} ms  min {ser[1]:7.2f}")
print(f"np.zeros(n,3)  : median {zer[0]:7.3f} ms")
print(f"speedup parallel vs serial: {ser[0] / par[0]:.2f}x")

fp, pp, vp = forces_from_pairs_parallel(pos, box, 2.5, pi, pj)
fs, ps, vs = _forces_from_pairs(pos, box, 2.5, pi, pj)
print(f"max |dF| = {np.abs(fp - fs).max():.2e}  dPot = {abs(pp - ps):.2e}  dVir = {abs(vp - vs):.2e}")
