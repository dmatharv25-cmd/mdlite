import time

from mdlite.integrators import run_nve
from mdlite.neighbor import VerletListForces
from mdlite.system import create_fcc_system

run_nve(create_fcc_system(5, 0.8, 1.0, seed=1), 5, 0.005, force_fn=VerletListForces(0.3, parallel=True))

print(f"{'skin':>5} {'ms/step':>9} {'builds/100':>11} {'pairs in list':>14}")
for skin in (0.2, 0.3, 0.4, 0.5, 0.6):
    s = create_fcc_system(22, 0.8, 1.0, seed=1)
    fn = VerletListForces(skin, use_cells=True, parallel=True)
    run_nve(s, 5, 0.005, force_fn=fn)
    fn.n_builds = 0
    t0 = time.perf_counter()
    run_nve(s, 100, 0.005, force_fn=fn)
    t = 1000.0 * (time.perf_counter() - t0) / 100
    print(f"{skin:>5.1f} {t:>9.2f} {fn.n_builds:>11} {len(fn.pair_i):>14}")
