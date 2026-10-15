"""Benchmark: scaling of cost per MD step for the three force strategies."""

import csv
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mdlite.forces_numba import compute_forces_numba
from mdlite.integrators import run_nve
from mdlite.neighbor import VerletListForces
from mdlite.system import create_fcc_system

N_STEPS = 100
DT = 0.005
CELLS = [4, 6, 8, 10, 14, 18]
MAX_N_ALLPAIRS = 4000
MAX_N_LIST_BRUTE = 11000

# warm-up: compile everything outside the timed regions
for fn in (compute_forces_numba, VerletListForces(use_cells=False), VerletListForces(use_cells=True)):
    run_nve(create_fcc_system(5, 0.8, 1.0, seed=1), 5, DT, force_fn=fn)


def time_steps(n_cells, make_fn):
    s = create_fcc_system(n_cells, 0.8, 1.0, seed=1)
    fn = make_fn()
    t0 = time.perf_counter()
    run_nve(s, N_STEPS, DT, force_fn=fn)
    return 1000.0 * (time.perf_counter() - t0) / N_STEPS


results = {"allpairs": [], "list_brute_build": [], "list_cell_build": []}
print(f"{'N':>6} {'all-pairs':>11} {'list (N^2 build)':>17} {'list (cell build)':>18}   ms/step")
for nc in CELLS:
    n = 4 * nc ** 3
    row = [n]
    if n <= MAX_N_ALLPAIRS:
        t = time_steps(nc, lambda: compute_forces_numba)
        results["allpairs"].append((n, t)); row.append(f"{t:11.3f}")
    else:
        row.append(f"{'-':>11}")
    if n <= MAX_N_LIST_BRUTE:
        t = time_steps(nc, lambda: VerletListForces(0.3, use_cells=False))
        results["list_brute_build"].append((n, t)); row.append(f"{t:17.3f}")
    else:
        row.append(f"{'-':>17}")
    t = time_steps(nc, lambda: VerletListForces(0.3, use_cells=True))
    results["list_cell_build"].append((n, t)); row.append(f"{t:18.3f}")
    print(f"{row[0]:>6} {row[1]} {row[2]} {row[3]}")

labels = {"allpairs": "Numba all-pairs", "list_brute_build": "Verlet list, O(N^2) build",
          "list_cell_build": "Verlet list + cell list"}
with open("benchmarks/scaling.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method", "N", "ms_per_step"])
    for k, v in results.items():
        for n, t in v:
            w.writerow([k, n, t])

plt.figure(figsize=(6, 4.5))
for k, v in results.items():
    arr = np.array(v)
    slope = np.polyfit(np.log(arr[-3:, 0]), np.log(arr[-3:, 1]), 1)[0]
    print(f"Scaling exponent (largest 3 sizes) {labels[k]}: N^{slope:.2f}")
    plt.loglog(arr[:, 0], arr[:, 1], "o-", label=f"{labels[k]} (~N^{slope:.2f})")
plt.xlabel("N particles"); plt.ylabel("ms per MD step"); plt.legend()
plt.title("MD step cost vs system size")
plt.tight_layout()
plt.savefig("benchmarks/scaling.png", dpi=150)
print("Saved benchmarks/scaling.csv and .png")
