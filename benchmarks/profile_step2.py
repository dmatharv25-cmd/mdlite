import time

import numba
import numpy as np

from mdlite.cells_parallel import build_pairs_cells_parallel
from mdlite.forces_parallel import forces_from_pairs_parallel
from mdlite.integrators import run_nve
from mdlite.neighbor import VerletListForces, _max_displacement2
from mdlite.system import create_fcc_system

SKIN = 0.5
run_nve(create_fcc_system(5, 0.8, 1.0, seed=1), 5, 0.005, force_fn=VerletListForces(SKIN, parallel=True))

s = create_fcc_system(22, 0.8, 1.0, seed=1)
fn = VerletListForces(SKIN, use_cells=True, parallel=True)
run_nve(s, 200, 0.005, force_fn=fn)


def med(f, reps=30):
    f()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t0)
    return 1000.0 * float(np.median(ts))


totals, builds = [], []
for _ in range(5):
    fn.n_builds = 0
    t0 = time.perf_counter()
    run_nve(s, 100, 0.005, force_fn=fn)
    totals.append(1000.0 * (time.perf_counter() - t0) / 100)
    builds.append(fn.n_builds)
total = float(np.median(totals))
nb = float(np.mean(builds))

pos = np.ascontiguousarray(s.positions)
box = float(s.box)
pi, pj, ref = fn.pair_i, fn.pair_j, fn._ref
f_ms = med(lambda: forces_from_pairs_parallel(pos, box, 2.5, pi, pj))
b_ms = med(lambda: build_pairs_cells_parallel(pos, box, 2.5 + SKIN), reps=10)
w_ms = med(lambda: s.wrap())
d_ms = med(lambda: _max_displacement2(pos, ref, box))

print(f"N={len(pos)} pairs={len(pi)} threads={numba.get_num_threads()}")
print(f"full step (median of 5x100): {total:.2f} ms   range {min(totals):.2f}-{max(totals):.2f}   builds/100 = {nb:.1f}")
print()
print(f"{'piece':<28} {'ms/call':>9} {'ms/step':>9} {'share':>7}")
rows = [
    ("force kernel", f_ms, f_ms),
    ("list build (amortized)", b_ms, b_ms * nb / 100),
    ("wrap", w_ms, w_ms),
    ("displacement check", d_ms, d_ms),
]
acc = 0.0
for name, per_call, per_step in rows:
    acc += per_step
    print(f"{name:<28} {per_call:>9.2f} {per_step:>9.2f} {100 * per_step / total:>6.0f}%")
rest = total - acc
print(f"{'rest (integrator, loop)':<28} {'':>9} {rest:>9.2f} {100 * rest / total:>6.0f}%")
