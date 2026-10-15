import time

import numpy as np

from mdlite.integrators import run_nve
from mdlite.neighbor import VerletListForces
from mdlite.system import create_fcc_system

run_nve(create_fcc_system(5, 0.8, 1.0, seed=1), 5, 0.005, force_fn=VerletListForces(0.3, parallel=True))

skins = (0.3, 0.4, 0.5, 0.6)
runs = {}
for skin in skins:
    s = create_fcc_system(22, 0.8, 1.0, seed=1)
    fn = VerletListForces(skin, use_cells=True, parallel=True)
    run_nve(s, 200, 0.005, force_fn=fn)  # equilibrate
    runs[skin] = (s, fn)

times = {skin: [] for skin in skins}
builds = {skin: [] for skin in skins}
for rnd in range(6):
    for skin in skins:
        s, fn = runs[skin]
        fn.n_builds = 0
        t0 = time.perf_counter()
        run_nve(s, 100, 0.005, force_fn=fn)
        times[skin].append(1000.0 * (time.perf_counter() - t0) / 100)
        builds[skin].append(fn.n_builds)

print(f"{'skin':>5} {'median ms/step':>15} {'min':>8} {'max':>8} {'builds/100':>11} {'pairs':>10}")
for skin in skins:
    t = times[skin]
    print(f"{skin:>5.1f} {np.median(t):>15.2f} {min(t):>8.2f} {max(t):>8.2f} {np.mean(builds[skin]):>11.1f} {len(runs[skin][1].pair_i):>10}")
print()
for skin in skins:
    print(skin, " ".join(f"{x:.1f}" for x in times[skin]))
