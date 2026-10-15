import time

import numpy as np

from mdlite.integrators import run_nve
from mdlite.neighbor import VerletListForces
from mdlite.system import create_fcc_system
from mdlite.wrap_parallel import wrap_inplace

s = create_fcc_system(22, 0.8, 1.0, seed=1)
box = float(s.box)
base = s.positions + np.random.default_rng(0).normal(0, 0.3, s.positions.shape)
outside = int(np.count_nonzero((base < 0) | (base >= box)))
wrap_inplace(base.copy(), box)  # compile


def median_ms(fn, reps=50):
    ts = []
    for _ in range(reps):
        a = base.copy()
        t0 = time.perf_counter()
        fn(a)
        ts.append(time.perf_counter() - t0)
    return 1000.0 * float(np.median(ts))


def np_wrap(a):
    a %= box


print(f"wrap at N={len(base)} ({outside} of {base.size} coordinates outside the box)")
print(f"  numpy %=        : {median_ms(np_wrap):7.3f} ms")
print(f"  numba parallel  : {median_ms(lambda a: wrap_inplace(a, box)):7.3f} ms")
a = base.copy(); a %= box
b = base.copy(); wrap_inplace(b, box)
d = a - b
d -= box * np.round(d / box)
print(f"  max periodic difference: {np.abs(d).max():.2e}")

run_nve(create_fcc_system(5, 0.8, 1.0, seed=1), 5, 0.005, force_fn=VerletListForces(0.3, parallel=True))

print(f"\n{'skin':>5} {'median ms/step':>15} {'min':>8} {'max':>8} {'builds/100':>11}")
for skin in (0.3, 0.4, 0.5):
    s = create_fcc_system(22, 0.8, 1.0, seed=1)
    fn = VerletListForces(skin, use_cells=True, parallel=True)
    run_nve(s, 200, 0.005, force_fn=fn)  # equilibrate
    times, builds = [], []
    for _ in range(5):
        fn.n_builds = 0
        t0 = time.perf_counter()
        run_nve(s, 100, 0.005, force_fn=fn)
        times.append(1000.0 * (time.perf_counter() - t0) / 100)
        builds.append(fn.n_builds)
    print(f"{skin:>5.1f} {np.median(times):>15.2f} {min(times):>8.2f} {max(times):>8.2f} {np.mean(builds):>11.1f}")
