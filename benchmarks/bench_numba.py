"""Benchmark: Numba kernel vs the saved NumPy baseline."""

import csv
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mdlite.forces_numba import compute_forces_numba
from mdlite.system import create_fcc_system

baseline = {}
with open("benchmarks/baseline_numpy.csv") as f:
    for row in csv.DictReader(f):
        baseline[int(row["N"])] = float(row["ms_per_force_call"])

warm = create_fcc_system(2, density=0.8, temperature=1.0, seed=1)
compute_forces_numba(warm.positions, warm.box)     # trigger/compile once

rows = []
print(f"{'N':>6} {'numba ms':>12} {'numpy ms':>12} {'speedup':>10}")
for n_cells in [2, 3, 4, 5, 6, 7, 8, 10, 12]:
    s = create_fcc_system(n_cells, density=0.8, temperature=1.0, seed=1)
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        compute_forces_numba(s.positions, s.box)
        times.append(time.perf_counter() - t0)
    ms = 1000.0 * min(times)
    base = baseline.get(s.n_particles)
    speed = f"{base / ms:>9.1f}x" if base else "        -"
    print(f"{s.n_particles:>6} {ms:>12.4f} {base if base else float('nan'):>12.3f} {speed}")
    rows.append((s.n_particles, ms))

with open("benchmarks/numba_serial.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["N", "ms_per_force_call"])
    w.writerows(rows)

n = np.array([r[0] for r in rows], dtype=float)
t = np.array([r[1] for r in rows])
bn = np.array(sorted(baseline), dtype=float)
bt = np.array([baseline[int(k)] for k in bn])
plt.figure(figsize=(5.5, 4))
plt.loglog(bn, bt, "o-", label="NumPy brute force")
plt.loglog(n, t, "s-", label="Numba serial (N^2 loop)")
plt.loglog(n, t[0] * (n / n[0]) ** 2, "--", color="gray", label="O(N^2) reference")
plt.xlabel("N particles"); plt.ylabel("ms per force call"); plt.legend()
plt.title("Force kernel scaling")
plt.tight_layout()
plt.savefig("benchmarks/numba_serial.png", dpi=150)
print("Saved benchmarks/numba_serial.csv and .png")
