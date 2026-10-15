import time

import numba
import numpy as np

from mdlite.forces_parallel import forces_from_pairs_parallel
from mdlite.integrators import run_nve
from mdlite.neighbor import VerletListForces
from mdlite.system import create_fcc_system

s = create_fcc_system(22, 0.8, 1.0, seed=1)
fn = VerletListForces(0.5, use_cells=True, parallel=True)
run_nve(s, 20 0.005, force_fn=fn)
pos = np.ascontiguousarray(s.positions)
box = float(s.box)
pi, pj = fn.pair_i, fn.pair_j
n = pos.shape[0]
max_threads = numba.get_num_threads()


def med(f, reps=30):
    f()
    f()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t0)
    return 1000.0 * float(np.median(ts)), 1000.0 * min(ts)


empty = np.empty(0, dtype=np.int32)
ov = med(lambda: forces_from_pairs_parallel(pos, box, 2.5, empty, empty))
print(f"N={n} pairs={len(pi)} threads={max_threads}")
print(f"empty pair list (buffer zero + reduction only): median {ov[0]:.2f} ms  min {ov[1]:.2f}")

print("\nnchunks sweep (all threads):")
print(f"{'nchunks':>8} {'median ms':>10} {'min ms':>8} {'buffer MB':>10}")
for nc in (1, 2, 4, 6, 12, 24, 48):
    m = med(lambda: forces_from_pairs_parallel(pos, box, 2.5, pi, pj, nchunks=nc))
    print(f"{nc:>8} {m[0]:>10.2f} {m[1]:>8.2f} {nc * n * 3 * 8 / 1e6:>10.1f}")

print("\nthread sweep (nchunks = threads):")
print(f"{'threads':>8} {'median ms':>10} {'min ms':>8} {'speedup':>8}")
base = None
for th in (1, 2, 4, 6, 8, 12):
    if th > max_threads:
        continue
    numba.set_num_threads(th)
    m = med(lambda: forces_from_pairs_parallel(pos, box, 2.5, pi, pj))
    if base is None:
        base = m[0]
    print(f"{th:>8} {m[0]:>10.2f} {m[1]:>8.2f} {base / m[0]:>8.2f}")
numba.set_num_threads(max_threads)
