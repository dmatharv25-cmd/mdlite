import numpy as np

from mdlite.cells import build_pairs_cells
from mdlite.forces_numba import compute_forces_numba
from mdlite.forces_parallel import forces_from_pairs_parallel
from mdlite.integrators import run_nve
from mdlite.neighbor import VerletListForces, _forces_from_pairs
from mdlite.system import create_fcc_system


def _perturbed(n_cells, seed):
    s = create_fcc_system(n_cells, density=0.8, temperature=1.0, seed=seed)
    s.positions += np.random.default_rng(seed).normal(0, 0.08, s.positions.shape)
    s.wrap()
    return s


def test_parallel_kernel_matches_serial_kernel_for_any_chunk_count():
    s = _perturbed(5, 21)
    pi, pj = build_pairs_cells(s.positions, s.box, 2.8)
    f_ref, u_ref, w_ref = _forces_from_pairs(s.positions, s.box, 2.5, pi, pj)
    for nchunks in (1, 3, 12):
        f, u, w = forces_from_pairs_parallel(s.positions, s.box, 2.5, pi, pj, nchunks)
        assert np.allclose(f, f_ref, atol=1e-9)
        assert np.isclose(u, u_ref) and np.isclose(w, w_ref)


def test_parallel_wrapper_matches_brute_force():
    s = _perturbed(5, 22)
    f_ref, u_ref, w_ref = compute_forces_numba(s.positions, s.box)
    f, u, w = VerletListForces(skin=0.3, parallel=True)(s.positions, s.box)
    assert np.allclose(f, f_ref, atol=1e-9)
    assert np.isclose(u, u_ref) and np.isclose(w, w_ref)


def test_parallel_trajectory_matches_brute_force():
    a = create_fcc_system(5, density=0.8, temperature=2.0, seed=11)
    b = create_fcc_system(5, density=0.8, temperature=2.0, seed=11)
    run_nve(a, n_steps=200, dt=0.005, force_fn=compute_forces_numba)
    run_nve(b, n_steps=200, dt=0.005, force_fn=VerletListForces(skin=0.3, parallel=True))
    d = a.positions - b.positions
    d -= a.box * np.round(d / a.box)
    assert np.abs(d).max() < 1e-6


def test_parallel_energy_conserved():
    s = create_fcc_system(5, density=0.8, temperature=1.5, seed=3)
    log = run_nve(s, n_steps=400, dt=0.005, force_fn=VerletListForces(skin=0.3, parallel=True))
    e = log["total"]
    assert np.max(np.abs(e - e[0])) / abs(e[0]) < 1e-3


def test_parallel_kernel_handles_empty_pair_list():
    pos = np.random.default_rng(0).random((4, 3)) * 10.0
    empty = np.empty(0, dtype=np.int32)
    f, u, w = forces_from_pairs_parallel(pos, 20.0, 2.5, empty, empty, 4)
    assert np.all(f == 0.0) and u == 0.0 and w == 0.0
