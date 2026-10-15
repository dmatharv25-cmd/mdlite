"""Validate mdlite against NIST reference MD data for the Lennard-Jones fluid.

Reference: NIST LJ_PURE, NVE MD, N=500, dt=0.005, rc=3 (truncated, not shifted)
plus standard long-range corrections, T* ~ 0.85.
https://mmlapps.nist.gov/srs/LJ_PURE/md.htm
"""

import math
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from mdlite.forces import compute_forces_brute, lj_shift
from mdlite.integrators import run_nve
from mdlite.neighbor import VerletListForces
from mdlite.system import create_fcc_system

RC, DT, T0 = 3.0, 0.005, 0.85
T_MELT = 2.5
# (rho, U_ref per particle, p_ref) from the NIST table
REFERENCE = [(0.776, -5.517, 0.030), (0.84, -5.909, 0.910), (0.9, -6.234, 2.544)]


def pair_energy_offset():
    """Energy per pair the code adds relative to the plain 12-6 potential."""
    r = 1.3
    pos = np.array([[1.0, 1.0, 1.0], [1.0 + r, 1.0, 1.0]])
    _, pot, _ = compute_forces_brute(pos, 10.0, RC)
    return pot - 4.0 * (r ** -12 - r ** -6)


def lrc(rho):
    """Standard long-range corrections: energy per particle and pressure."""
    u = (8.0 / 3.0) * math.pi * rho * ((1.0 / 3.0) * RC ** -9 - RC ** -3)
    p = (16.0 / 3.0) * math.pi * rho ** 2 * ((2.0 / 3.0) * RC ** -9 - RC ** -3)
    return u, p


def rdf_accumulate(pos, box, hist, nbins):
    d = pos[:, None, :] - pos[None, :, :]
    d -= box * np.round(d / box)
    r = np.sqrt((d ** 2).sum(-1))[np.triu_indices(len(pos), 1)]
    hist += np.histogram(r, bins=nbins, range=(0.0, box / 2))[0]


def mean_sem(x, nblocks=10):
    m = np.array([c.mean() for c in np.array_split(np.asarray(x), nblocks)])
    return m.mean(), m.std(ddof=1) / math.sqrt(nblocks)


def run_scaled(system, force_fn, target, time_units):
    """NVE chunks of one time unit, rescaling velocities to `target` after each."""
    for _ in range(time_units):
        run_nve(system, 200, DT, r_cut=RC, force_fn=force_fn, record_every=200)
        system.velocities *= math.sqrt(target / system.temperature())


def simulate(rho, offset):
    system = create_fcc_system(5, density=rho, temperature=T_MELT, seed=7)
    n, box = system.n_particles, float(system.box)
    volume = box ** 3
    force_fn = VerletListForces(0.3, use_cells=True, parallel=True)

    run_scaled(system, force_fn, T_MELT, 40)     # hot: destroy the FCC lattice
    target = T0
    for _ in range(4):                           # cool, then steer the NVE mean T to T0
        run_scaled(system, force_fn, target, 30)
        log = run_nve(system, 2000, DT, r_cut=RC, force_fn=force_fn, record_every=200)
        target += T0 - float(np.mean(log["temperature"]))

    nbins = 100
    hist = np.zeros(nbins)
    nrdf = 0
    temps, us, ps = [], [], []
    for k in range(1000):  # 100 time units, sample every 0.1
        run_nve(system, 20, DT, r_cut=RC, force_fn=force_fn, record_every=20)
        _, pot, vir = force_fn(system.positions, box, RC)
        pos = system.positions % box
        pos[pos >= box] = 0.0
        tree = cKDTree(pos, boxsize=box)
        npairs = (tree.count_neighbors(tree, RC) - n) // 2
        us.append((pot - offset * npairs) / n)       # undo the shift: plain truncated 12-6
        ps.append((2.0 * system.kinetic_energy() + vir) / (3.0 * volume))
        temps.append(system.temperature())
        if k % 10 == 0:
            rdf_accumulate(pos, box, hist, nbins)
            nrdf += 1

    edges = np.linspace(0.0, box / 2, nbins + 1)
    shell = (4.0 / 3.0) * math.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    g = hist / (nrdf * n * (n / volume) * shell / 2.0)
    r = 0.5 * (edges[1:] + edges[:-1])
    u, u_err = mean_sem(us)
    p, p_err = mean_sem(ps)
    return float(np.mean(temps)), u, u_err, p, p_err, r, g


def main():
    offset = pair_energy_offset()
    print(f"energy offset per pair vs plain 12-6: {offset:+.6f}   (-lj_shift(rc) = {-lj_shift(RC):+.6f})")
    print("N = 500, rc = 3.0, dt = 0.005, NVE production 100 time units\n")
    print(f"{'rho':>6} {'T':>7} {'U mdlite':>10} {'+-':>6} {'U NIST':>8} {'dU':>7} "
          f"{'p mdlite':>9} {'+-':>6} {'p NIST':>7} {'dp':>7}")
    curves = {}
    for rho, u_ref, p_ref in REFERENCE:
        t_start = time.perf_counter()
        temp, u, u_err, p, p_err, r, g = simulate(rho, offset)
        u_lrc, p_lrc = lrc(rho)
        u += u_lrc
        p += p_lrc
        print(f"{rho:>6.3f} {temp:>7.3f} {u:>10.3f} {u_err:>6.3f} {u_ref:>8.3f} {u - u_ref:>+7.3f} "
              f"{p:>9.3f} {p_err:>6.3f} {p_ref:>7.3f} {p - p_ref:>+7.3f}   ({time.perf_counter() - t_start:.0f} s)")
        curves[rho] = (r, g)

    print()
    fig, ax = plt.subplots(figsize=(6, 4))
    for rho, (r, g) in curves.items():
        i = int(np.argmax(g))
        print(f"rho = {rho}: first peak at r = {r[i]:.3f}, g = {g[i]:.2f}; mean g over last 10 bins = {g[-10:].mean():.3f}")
        ax.plot(r, g, label=f"rho = {rho}")
    ax.axhline(1.0, color="k", lw=0.5)
    ax.set_xlabel("r"); ax.set_ylabel("g(r)"); ax.set_title("Radial distribution function (T = 0.85)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("validation/rdf.png", dpi=150)
    print("Saved validation/rdf.png")


if __name__ == "__main__":
    main()