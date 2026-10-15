import numpy as np

from mdlite.forces_numba import compute_forces_numba
from mdlite.integrate_parallel import half_kick, kick_drift_wrap, max_displacement2_parallel
from mdlite.integrators import run_nve
from mdlite.neighbor import _max_displacement2
from mdlite.system import create_fcc_system


def _numpy_run(system, n_steps, dt, force_fn, r_cut=2.5):
    forces, pot, _ = force_fn(system.positions, system.box, r_cut)
    system.forces = forces
    for _ in range(n_steps):
        system.velocities += 0.5 * dt * system.forces
        system.positions += dt * system.velocities
        system.wrap()
        system.forces, pot, _ = force_fn(system.positions, system.box, r_cut)
        system.velocities += 0.5 * dt * system.forces


def test_fused_kernels_match_numpy_including_wrap():
    s = create_fcc_system(5, density=0.8, temperature=1.0, seed=1)
    rng = np.random.default_rng(5)
    pos = s.positions.copy()
    vel = s.velocities * 40.0
    f = rng.normal(0, 3.0, pos.shape)
    box = float(s.box)
    dt = 0.05
    ref_v = vel + 0.5 * dt * f
    ref_p = pos + dt * ref_v
    s2 = create_fcc_system(5, density=0.8, temperature=1.0, seed=1)
    s2.positions = ref_p
    s2.wrap()
    ref_p = s2.positions
    unwrapped = pos + dt * ref_v
    wrapped_any = bool(((unwrapped < 0) | (unwrapped >= box)).any())
    kick_drift_wrap(pos, vel, f, dt, box)
    assert wrapped_any
    assert np.allclose(vel, ref_v, atol=1e-13, rtol=0)
    assert np.allclose(pos, ref_p, atol=1e-12, rtol=0)
    assert (pos >= 0).all() and (pos < box + 1e-9).all()
    ref_v2 = vel + 0.5 * dt * f
    half_kick(vel, f, dt)
    assert np.allclose(vel, ref_v2, atol=1e-13, rtol=0)


def test_parallel_max_displacement_matches_serial():
    s = create_fcc_system(6, density=0.8, temperature=1.0, seed=2)
    box = float(s.box)
    ref = s.positions.copy()
    moved = ref + np.random.default_rng(3).normal(0, 0.2, ref.shape)
    moved[0, 0] += box
    moved[1, 1] -= box
    a = _max_displacement2(moved, ref, box)
    b = max_displacement2_parallel(moved, ref, box)
    assert abs(a - b) <= 1e-12 * max(1.0, a)


def test_run_nve_fused_matches_numpy_loop():
    a = create_fcc_system(4, density=0.8, temperature=1.0, seed=7)
    b = create_fcc_system(4, density=0.8, temperature=1.0, seed=7)
    _numpy_run(a, 150, 0.005, compute_forces_numba)
    run_nve(b, n_steps=150, dt=0.005, force_fn=compute_forces_numba)
    assert np.allclose(a.positions, b.positions, atol=1e-10, rtol=0)
    assert np.allclose(a.velocities, b.velocities, atol=1e-10, rtol=0)
