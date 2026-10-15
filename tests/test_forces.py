import numpy as np

from mdlite.forces import compute_forces_brute, lj_shift
from mdlite.system import create_fcc_system


def test_two_particles_at_potential_minimum():
    # LJ minimum is at r = 2^(1/6) where force = 0 and U = -1 (before shift)
    r_min = 2.0 ** (1.0 / 6.0)
    pos = np.array([[1.0, 1.0, 1.0], [1.0 + r_min, 1.0, 1.0]])
    f, u, _ = compute_forces_brute(pos, box=10.0, r_cut=2.5)
    assert np.allclose(f, 0.0, atol=1e-12)
    assert np.isclose(u, -1.0 - lj_shift(2.5))


def test_two_particles_repulsive_at_close_range():
    pos = np.array([[1.0, 1.0, 1.0], [1.9, 1.0, 1.0]])  # r = 0.9 < r_min
    f, _, _ = compute_forces_brute(pos, box=10.0)
    assert f[0, 0] < 0.0 and f[1, 0] > 0.0  # pushed apart


def test_newtons_third_law_total_force_zero():
    s = create_fcc_system(3, density=0.8, temperature=1.0, seed=3)
    s.positions += np.random.default_rng(0).normal(0, 0.05, s.positions.shape)  # perturb lattice
    f, _, _ = compute_forces_brute(s.positions, s.box)
    assert np.allclose(f.sum(axis=0), 0.0, atol=1e-9)


def test_periodic_images_equivalent():
    # Two particles near opposite faces are actually close neighbors through the boundary
    box = 10.0
    pos = np.array([[0.2, 5.0, 5.0], [9.5, 5.0, 5.0]])  # separation 0.7 via periodic wrap
    f, u, _ = compute_forces_brute(pos, box)
    assert u > 0.0                      # strongly repulsive
    assert f[0, 0] > 0.0 and f[1, 0] < 0.0  # pushed apart through the boundary


def test_perfect_fcc_lattice_has_zero_net_force_on_each_atom():
    s = create_fcc_system(3, density=0.8, temperature=1.0, seed=3)
    f, _, _ = compute_forces_brute(s.positions, s.box)
    assert np.allclose(f, 0.0, atol=1e-9)  # symmetry
