"""Baseline benchmark: pure-NumPy brute-force force calculation vs particle number."""

import csv
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mdlite.forces import compute_forces_brute
from mdlite.system import create_fcc_system

rows = []
print(f"{'N':>6} {'ms per force call':>20}")
for n_cells in [2, 3, 4, 5, 6, 7]:
    s = create_fcc_system(n_cells, density=0.8, temperature=1.0, seed=1)
    compute_forces_brute(s.positions, s.box)           # warm-up
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        compute_forces_brute(s.positions, s.box)
        times.append(time.perf_counter() - t0)
    best_ms = 1000.0 * min(times)
    rows.append((s.n_particles, best_ms))
    print(f"{s.n_particles:>6} {best_ms:>20.3f}")

with open("benchmarks/baseline_numpy.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["N", "ms_per_force_call"])
    w.writerows(rows)

n = np.array([r[0] for r in rows], dtype=float)
t = np.array([r[1] for r in rows])
plt.figure(figsize=(5.5, 4))
plt.loglog(n, t, "o-", label="NumPy brute force")
plt.loglog(n, t[0] * (n / n[0]) ** 2, "--", color="gray", label="O(N^2) reference")
plt.xlabel("N particles"); plt.ylabel("ms per force call"); plt.legend()
plt.title("Baseline scaling")
plt.tight_layout()
plt.savefig("benchmarks/baseline_numpy.png", dpi=150)
print("Saved benchmarks/baseline_numpy.csv and .png")
