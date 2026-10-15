import numpy as np

from mdlite.cells import build_pairs_cells, n_cells_per_dim
from mdlite.forces_numba import compute_forces_numba
from mdlite.integrators import run_nve
from mdlite.neighbor import VerletListForces, _build_pairs
from mdlite.system import create_fcc_system


def _perturbed(n_cells, seed):
    s = create_fcc_system(n_cells, density=0.8, temperature=1.0, seed=seed)
    s.positions += np.random.default_rng(seed).normal(0, 0.08, s.positions.shape)
    s.wrap()
    return s


def _as_set(pi, pj):
    return set(zip(pi.tolist(), pj.tolist()))


def test_cell_path_is_actually_active_for_5_cells():
    s = _perturbed(5, 1)
    assert n_cells_per_dim(s.box, 2.8) >= 3


def test_cell_pairs_equal_brute_pairs():
    s = _perturbed(5, 2)
    pi, pj = build_pairs_cells(s.positions, s.box, 2.8)
    bi, bj = _build_pairs(s.positions, s.box, 2.8)
    assert _as_set(pi, pj) == _as_set(bi, bj)
    assert len(_as_set(pi, pj)) == len(pi)   # no duplicates


def test_forces_with_cell_list_match_brute_force():
    s = _perturbed(5, 3)
    f_ref, u_ref, w_ref = compute_forces_numba(s.positions, s.box)
    f, u, w = VerletListForces(skin=0.3, use_cells=True)(s.positions, s.box)
    assert np.allclose(f, f_ref, atol=1e-9)
    assert np.isclose(u, u_ref) and np.isclose(w, w_ref)


def test_trajectory_with_cell_list_matches_brute_force():
    a = create_fcc_system(5, density=0.8, temperature=2.0, seed=11)
    b = create_fcc_system(5, density=0.8, temperature=2.0, seed=11)
    run_nve(a, n_steps=200, dt=0.005, force_fn=compute_forces_numba)
    run_nve(b, n_steps=200, dt=0.005, force_fn=VerletListForces(skin=0.3, use_cells=True))
    d = a.positions - b.positions
    d -= a.box * np.round(d / a.box)
    assert np.abs(d).max() < 1e-6


def test_small_box_falls_back_to_brute_build():
    s = _perturbed(4, 4)   # box ~6.8 -> only 2 cells per side
    f_ref, _, _ = compute_forces_numba(s.positions, s.box)
    f, _, _ = VerletListForces(skin=0.3, use_cells=True)(s.positions, s.box)
    assert np.allclose(f, f_ref, atol=1e-9)
