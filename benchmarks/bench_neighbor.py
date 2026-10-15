"""Benchmark: cost per MD step, Numba brute force vs Verlet neighbor list."""

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

N_STEPS = 200
DT = 0.005

# warm-up so compilation time is not measured
run_nve(create_fcc_system(4, 0.8, 1.0, seed=1), 5, DT, force_fn=compute_forces_numba)
run_nve(create_fcc_system(4, 0.8, 1.0, seed=1), 5, DT, force_fn=VerletListForces())

rows = []
print(f"{'N':>6} {'brute ms/step':>14} {'list ms/step':>14} {'speedup':>9} {'builds':>8}")
for n_cells in [4, 6, 8, 10]:
    s1 = create_fcc_system(n_cells, 0.8, 1.0, seed=1)
    t0 = time.perf_counter()
    run_nve(s1, N_STEPS, DT, force_fn=compute_forces_numba)
    t_brute = 1000.0 * (time.perf_counter() - t0) / N_STEPS

    s2 = create_fcc_system(n_cells, 0.8, 1.0, seed=1)
    vl = VerletListForces(skin=0.3)
    t0 = time.perf_counter()
    run_nve(s2, N_STEPS, DT, force_fn=vl)
    t_list = 1000.0 * (time.perf_counter() - t0) / N_STEPS

    n = s1.n_particles
    print(f"{n:>6} {t_brute:>14.3f} {t_list:>14.3f} {t_brute / t_list:>8.1f}x {vl.n_builds:>8}")
    rows.append((n, t_brute, t_list, vl.n_builds))

with open("benchmarks/neighbor_list.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["N", "brute_ms_per_step", "list_ms_per_step", "n_builds"])
    w.writerows(rows)

arr = np.array(rows)
plt.figure(figsize=(5.5, 4))
plt.loglog(arr[:, 0], arr[:, 1], "s-", label="Numba brute force")
plt.loglog(arr[:, 0], arr[:, 2], "^-", label="Numba + Verlet list")
plt.xlabel("N particles"); plt.ylabel("ms per MD step"); plt.legend()
plt.title("Neighbor list speedup")
plt.tight_layout()
plt.savefig("benchmarks/neighbor_list.png", dpi=150)
print("Saved benchmarks/neighbor_list.csv and .png")
