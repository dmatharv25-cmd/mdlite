import numpy as np
import pytest

from mdlite.forces_numba import compute_forces_numba
from mdlite.integrators import run_nve
from mdlite.neighbor import VerletListForces
from mdlite.system import create_fcc_system


def test_verlet_list_matches_brute_force():
    s = create_fcc_system(4, density=0.8, temperature=1.0, seed=7)
    s.positions += np.random.default_rng(7).normal(0, 0.08, s.positions.shape)
    s.wrap()
    f_ref, u_ref, w_ref = compute_forces_numba(s.positions, s.box)
    f, u, w = VerletListForces(skin=0.3)(s.positions, s.box)
    assert np.allclose(f, f_ref, atol=1e-9)
    assert np.isclose(u, u_ref) and np.isclose(w, w_ref)


def test_trajectory_matches_brute_force_after_200_steps():
    a = create_fcc_system(4, density=0.8, temperature=2.0, seed=11)
    b = create_fcc_system(4, density=0.8, temperature=2.0, seed=11)
    run_nve(a, n_steps=200, dt=0.005, force_fn=compute_forces_numba)
    run_nve(b, n_steps=200, dt=0.005, force_fn=VerletListForces(skin=0.3))
    d = a.positions - b.positions
    d -= a.box * np.round(d / a.box)
    assert np.abs(d).max() < 1e-6


def test_energy_conserved_with_neighbor_list():
    s = create_fcc_system(4, density=0.8, temperature=1.5, seed=3)
    log = run_nve(s, n_steps=400, dt=0.005, force_fn=VerletListForces(skin=0.3))
    e = log["total"]
    assert np.max(np.abs(e - e[0])) / abs(e[0]) < 1e-3


def test_list_is_rebuilt_lazily():
    s = create_fcc_system(4, density=0.8, temperature=2.0, seed=4)
    vl = VerletListForces(skin=0.3)
    run_nve(s, n_steps=300, dt=0.005, force_fn=vl)
    assert 1 < vl.n_builds < 150   # rebuilt sometimes, but far fewer than every step


def test_small_box_is_rejected():
    s = create_fcc_system(2, density=0.8, temperature=1.0, seed=1)  # box ~3.4
    with pytest.raises(ValueError):
        VerletListForces()(s.positions, s.box)
