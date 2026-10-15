import time
import numba
import numpy as np

from mdlite.cells import build_pairs_cells
from mdlite.forces_parallel import forces_from_pairs_parallel
from mdlite.integrators import run_nve
from mdlite.neighbor import VerletListForces, _forces_from_pairs, _max_displacement2
from mdlite.system import create_fcc_system

s = create_fcc_system(22, 0.8, 1.0, seed=1)
pos = np.ascontiguousarray(s.positions)
box = s.box
pi, pj = build_pairs_cells(pos, box, 2.8)
print(f"N={len(pos)}  pairs in list={len(pi)}")


def ms(fn, reps=20):
    fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return 1000.0 * (time.perf_counter() - t0) / reps


t_build = ms(lambda: build_pairs_cells(pos, box, 2.8), 5)
t_disp = ms(lambda: _max_displacement2(pos, pos, box))
t_serial = ms(lambda: _forces_from_pairs(pos, box, 2.5, pi, pj))
print(f"list build (cells)     : {t_build:8.3f} ms  (happens ~every 10 steps)")
print(f"displacement check     : {t_disp:8.3f} ms  (every step)")
print(f"serial force kernel    : {t_serial:8.3f} ms")

nmax = numba.get_num_threads()
for th in (1, 4, nmax):
    numba.set_num_threads(th)
    for nch in (th, 2 * th):
        t = ms(lambda: forces_from_pairs_parallel(pos, box, 2.5, pi, pj, nch))
        print(f"parallel kernel threads={th:2d} chunks={nch:2d}: {t:8.3f} ms")
numba.set_num_threads(nmax)

s2 = create_fcc_system(22, 0.8, 1.0, seed=1)
fn = VerletListForces(0.3, use_cells=True, parallel=False)
run_nve(s2, 5, 0.005, force_fn=fn)
t0 = time.perf_counter()
run_nve(s2, 50, 0.005, force_fn=fn)
t_step = 1000.0 * (time.perf_counter() - t0) / 50
accounted = t_serial + t_disp + t_build / 10
print(f"\nfull serial step       : {t_step:8.3f} ms")
print(f"force+disp+build/10    : {accounted:8.3f} ms")
print(f"everything else (integrator, logging, Python): {t_step - accounted:8.3f} ms")
