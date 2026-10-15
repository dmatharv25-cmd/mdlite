import time

import numpy as np

from old_cells_parallel import build_pairs_cells_parallel as old_build
from mdlite.cells_parallel import build_pairs_cells_parallel as new_build
from mdlite.integrators import run_nve
from mdlite.neighbor import VerletListForces
from mdlite.system import create_fcc_system

s = create_fcc_system(22, 0.8, 1.0, seed=1)
fn = VerletListForces(0.5, use_cells=True, parallel=True)
run_nve(s, 200, 0.005, force_fn=fn)
pos = np.ascontiguousarray(s.positions)
box = float(s.box)
r_list = 3.0
n = len(pos)


def med(f, reps=10):
    f()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t0)
    return 1000.0 * float(np.median(ts)), 1000.0 * min(ts)


rows = {"old": [], "new": []}
for _ in range(3):
    rows["old"].append(med(lambda: old_build(pos, box, r_list)))
    rows["new"].append(med(lambda: new_build(pos, box, r_list)))

print(f"N={n} r_list={r_list}")
for name, r in rows.items():
    print(f"{name}: median of 3 rounds {np.median([x[0] for x in r]):7.2f} ms   best min {min(x[1] for x in r):7.2f} ms")
print(f"speedup: {np.median([x[0] for x in rows['old']]) / np.median([x[0] for x in rows['new']]):.2f}x")

oi, oj = old_build(pos, box, r_list)
ni, nj = new_build(pos, box, r_list)
key = lambda i, j: np.sort(i.astype(np.int64) * n + j.astype(np.int64))
print(f"pairs old={len(oi)} new={len(ni)}  same set={np.array_equal(key(oi, oj), key(ni, nj))}  all i<j={bool((ni < nj).all())}")
