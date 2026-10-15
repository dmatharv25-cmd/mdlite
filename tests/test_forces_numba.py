import numpy as np

from mdlite.forces import compute_forces_brute
from mdlite.forces_numba import compute_forces_numba
from mdlite.integrators import run_nve
from mdlite.system import create_fcc_system


def _perturbed_system(seed):
    s = create_fcc_system(4, density=0.8, temperature=1.0, seed=seed)
    s.positions += np.random.default_rng(seed).normal(0, 0.08, s.positions.shape)
    s.wrap()
    return s


def test_numba_matches_numpy_reference():
    s = _perturbed_system(7)
    f_ref, u_ref, w_ref = compute_forces_brute(s.positions, s.box)
    f_nb, u_nb, w_nb = compute_forces_numba(s.positions, s.box)
    assert np.allclose(f_nb, f_ref, atol=1e-9)
    assert np.isclose(u_nb, u_ref)
    assert np.isclose(w_nb, w_ref)


def test_numba_total_force_is_zero():
    s = _perturbed_system(8)
    f, _, _ = compute_forces_numba(s.positions, s.box)
    assert np.allclose(f.sum(axis=0), 0.0, atol=1e-9)


def test_nve_with_numba_kernel_conserves_energy():
    s = create_fcc_system(3, density=0.8, temperature=1.0, seed=5)
    log = run_nve(s, n_steps=200, dt=0.005, force_fn=compute_forces_numba)
    e = log["total"]
    assert np.max(np.abs(e - e[0])) / abs(e[0]) < 1e-3
