"""NVE run: energy conservation of velocity Verlet for a Lennard-Jones liquid."""

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mdlite.integrators import run_nve
from mdlite.system import create_fcc_system

system = create_fcc_system(4, density=0.8, temperature=1.0, seed=42)
print(f"N = {system.n_particles}, box = {system.box:.4f}")

t0 = time.perf_counter()
log = run_nve(system, n_steps=2000, dt=0.005, record_every=10)
print(f"Run time: {time.perf_counter() - t0:.1f} s")

e = log["total"]
drift = np.abs(e - e[0]) / abs(e[0])
print(f"E0 = {e[0]:.4f}, E_final = {e[-1]:.4f}")
print(f"Max relative energy drift: {drift.max():.2e}")
print(f"Mean temperature after melting: {log['temperature'][50:].mean():.3f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(log["time"], log["kinetic"], label="Kinetic")
ax[0].plot(log["time"], log["potential"], label="Potential")
ax[0].plot(log["time"], log["total"], label="Total", color="k")
ax[0].set_xlabel("time"); ax[0].set_ylabel("energy"); ax[0].legend()
ax[0].set_title("Energy components (NVE)")
ax[1].semilogy(log["time"], drift + 1e-16)
ax[1].set_xlabel("time"); ax[1].set_ylabel("|E - E0| / |E0|")
ax[1].set_title("Relative energy drift")
plt.tight_layout()
plt.savefig("examples/nve_energy.png", dpi=150)
print("Saved examples/nve_energy.png")
