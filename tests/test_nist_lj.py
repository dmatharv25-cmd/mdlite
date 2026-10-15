"""Physics validation of mdlite against NIST LJ_PURE Monte Carlo data.

Reference: https://mmlapps.nist.gov/srs/LJ_PURE/mc.htm
NVT Monte Carlo, N = 500, T* = 0.85, truncation at 3 sigma (not shifted)
plus standard long-range corrections.

The NIST MC table is used for pressure, not the MD page: at rho = 0.9 the MD page
lists p = 2.544, which disagrees with NIST's own MC value of 2.231 (+-0.027).
Slow: run with `pytest -m slow -s`.
"""

import math

import numpy as np
import pytest
from scipy.spatial import cKDTree

from mdlite.forces import compute_forces_brute
from mdlite.integrators import run_nve
from mdlite.neighbor import VerletListForces
from mdlite.system import create_fcc_system

RC, DT, T_TARGET, T_MELT = 3.0, 0.005, 0.85, 2.5
# d/dT at fixed density: secant slopes between the NIST MC isotherms at T* = 0.85 and 0.90
DP_DT = {0.82: 5.41, 0.86: 6.24, 0.90: 7.07}
DU_DT = {0.82: 0.98, 0.86: 1.10, 0.90: 1.24}

# rho: (U per particle, p) from the NIST MC table at T* = 0.85
MC_REF = {
    0.82: (-5.7947, 0.55355),
    0.86: (-6.0305, 1.2660),
    0.90: (-6.2391, 2.2314),
}


def _pair_energy_offset():
    r = 1.3
    pos = np.array([[1.0, 1.0, 1.0], [1.0 + r, 1.0, 1.0]])
    _, pot, _ = compute_forces_brute(pos, 10.0, RC)
    return pot - 4.0 * (r ** -12 - r ** -6)


def _lrc(rho):
    u = (8.0 / 3.0) * math.pi * rho * ((1.0 / 3.0) * RC ** -9 - RC ** -3)
    p = (16.0 / 3.0) * math.pi * rho ** 2 * ((2.0 / 3.0) * RC ** -9 - RC ** -3)
    return u, p


def _run_scaled(system, force_fn, target, time_units):
    for _ in range(time_units):
        run_nve(system, 200, DT, r_cut=RC, force_fn=force_fn, record_every=200)
        system.velocities *= math.sqrt(target / system.temperature())


def _measure(rho):
    offset = _pair_energy_offset()
    u_lrc, p_lrc = _lrc(rho)
    system = create_fcc_system(5, density=rho, temperature=T_MELT, seed=1)
    n, box = system.n_particles, float(system.box)
    volume = box ** 3
    force_fn = VerletListForces(0.3, use_cells=True, parallel=True)

    _run_scaled(system, force_fn, T_MELT, 30)    # melt the lattice
    _run_scaled(system, force_fn, T_TARGET, 40)  # cool to the target
    run_nve(system, 4000, DT, r_cut=RC, force_fn=force_fn, record_every=4000)  # free NVE settle

    temps, us, ps = [], [], []
    for _ in range(600):  # 60 time units, sample every 0.1
        run_nve(system, 20, DT, r_cut=RC, force_fn=force_fn, record_every=20)
        _, pot, vir = force_fn(system.positions, box, RC)
        pos = system.positions % box
        pos[pos >= box] = 0.0
        tree = cKDTree(pos, boxsize=box)
        npairs = (tree.count_neighbors(tree, RC) - n) // 2
        us.append((pot - offset * npairs) / n)
        ps.append((2.0 * system.kinetic_energy() + vir) / (3.0 * volume))
        temps.append(system.temperature())
    return float(np.mean(temps)), float(np.mean(us)) + u_lrc, float(np.mean(ps)) + p_lrc


@pytest.mark.slow
@pytest.mark.parametrize("rho", sorted(MC_REF))
def test_matches_nist_monte_carlo(rho):
    u_ref, p_ref = MC_REF[rho]
    temp, u, p = _measure(rho)
    u_adj = u + DU_DT[rho] * (T_TARGET - temp)
    p_adj = p + DP_DT[rho] * (T_TARGET - temp)
    print(f"\nrho={rho}: T={temp:.3f}  U={u_adj:.4f} (ref {u_ref:.4f}, d={u_adj - u_ref:+.4f})"
          f"  p={p_adj:.4f} (ref {p_ref:.4f}, d={p_adj - p_ref:+.4f})")
    assert abs(temp - T_TARGET) < 0.04
    assert abs(u_adj - u_ref) < 0.03
    assert abs(p_adj - p_ref) < 0.10