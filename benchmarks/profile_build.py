import time
import numpy as np

from mdlite.cells import build_pairs_cells
from mdlite.cells_parallel import build_pairs_cells_parallel
from mdlite.system import create_fcc_system

s = create_fcc_system(22, 0.8, 1.0, seed=1)
pos = np.ascontiguousarray(s.positions)
for name, fn in (("serial", build_pairs_cells), ("parallel", build_pairs_cells_parallel)):
    fn(pos, s.box, 2.8)  # compile outside the timed region
    t0 = time.perf_counter()
    for _ in range(5):
        pi, pj = fn(pos, s.box, 2.8)
    print(f"{name:>8} build: {1000.0 * (time.perf_counter() - t0) / 5:8.1f} ms   pairs={len(pi)}")
