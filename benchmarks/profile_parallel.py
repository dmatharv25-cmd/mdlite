import cProfile
import pstats

from mdlite.integrators import run_nve
from mdlite.neighbor import VerletListForces
from mdlite.system import create_fcc_system

s = create_fcc_system(22, 0.8, 1.0, seed=1)
fn = VerletListForces(0.3, use_cells=True, parallel=True)
run_nve(s, 5, 0.005, force_fn=fn)
fn.n_builds = 0
pr = cProfile.Profile()
pr.enable()
run_nve(s, 100, 0.005, force_fn=fn)
pr.disable()
print("list builds in 100 steps:", fn.n_builds)
pstats.Stats(pr).sort_stats("tottime").print_stats(8)
