"""Benchmark: serial vs parallel force loop (both with cell-list neighbor build)."""

import csv
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numba
import numpy as np

from mdlite.integrators import run_nve
from mdlite.neighbor import VerletListForces
from mdlite.system import create_fcc_system

N_STEPS = 100
DT = 0.005
CELLS = [6, 10, 14, 18, 22]
MAX_THREADS = numba.get_num_threads()

for par in (False, True):
    run_nve(create_fcc_system(5, 0.8, 1.0, seed=1), 5, DT, force_fn=VerletListForces(0.3, parallel=par))


def time_steps(n_cells, parallel):
    s = create_fcc_system(n_cells, 0.8, 1.0, seed=1)
    fn = VerletListForces(0.3, use_cells=True, parallel=parallel)
    t0 = time.perf_counter()
    run_nve(s, N_STEPS, DT, force_fn=fn)
    return 1000.0 * (time.perf_counter() - t0) / N_STEPS


rows = []
print(f"{'N':>6} {'serial ms':>10} {'parallel ms':>12} {'speedup':>8}   ({MAX_THREADS} threads)")
for nc in CELLS:
    n = 4 * nc ** 3
    ts = time_steps(nc, False)
    tp = time_steps(nc, True)
    rows.append((n, ts, tp))
    print(f"{n:>6} {ts:>10.3f} {tp:>12.3f} {ts / tp:>7.1f}x")

n_big = 4 * CELLS[-1] ** 3
thread_rows = []
print(f"\nThread scaling at N={n_big}:")
print(f"{'threads':>8} {'ms/step':>9} {'speedup vs 1':>13}")
base = None
for th in [t for t in (1, 2, 4, 8, 12) if t <= MAX_THREADS]:
    numba.set_num_threads(th)
    time_steps(CELLS[-1], True)  # settle
    t = time_steps(CELLS[-1], True)
    base = base or t
    thread_rows.append((th, t))
    print(f"{th:>8} {t:>9.3f} {base / t:>12.2f}x")
numba.set_num_threads(MAX_THREADS)

with open("benchmarks/parallel.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["N", "serial_ms", "parallel_ms"])
    w.writerows(rows)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
arr = np.array(rows)
ax[0].loglog(arr[:, 0], arr[:, 1], "o-", label="serial")
ax[0].loglog(arr[:, 0], arr[:, 2], "s-", label=f"parallel ({MAX_THREADS} threads)")
ax[0].set_xlabel("N particles"); ax[0].set_ylabel("ms per MD step"); ax[0].legend()
th = np.array(thread_rows)
ax[1].plot(th[:, 0], th[0, 1] / th[:, 1], "o-", label="measured")
ax[1].plot(th[:, 0], th[:, 0], "k--", label="ideal")
ax[1].set_xlabel("threads"); ax[1].set_ylabel(f"speedup (N={n_big})"); ax[1].legend()
plt.tight_layout()
plt.savefig("benchmarks/parallel.png", dpi=150)
print("Saved benchmarks/parallel.csv and .png")
