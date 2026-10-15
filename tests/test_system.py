import numpy as np

from mdlite.system import create_fcc_system


def test_particle_count():
    s = create_fcc_system(3, density=0.8, temperature=1.0, seed=1)
    assert s.n_particles == 4 * 3 ** 3  # 108


def test_density():
    s = create_fcc_system(3, density=0.8, temperature=1.0, seed=1)
    assert np.isclose(s.n_particles / s.box ** 3, 0.8)


def test_zero_total_momentum():
    s = create_fcc_system(4, density=0.8, temperature=1.5, seed=2)
    assert np.allclose(s.velocities.sum(axis=0), 0.0, atol=1e-10)


def test_temperature_exact():
    s = create_fcc_system(4, density=0.8, temperature=1.5, seed=2)
    assert np.isclose(s.temperature(), 1.5)


def test_positions_inside_box():
    s = create_fcc_system(3, density=0.8, temperature=1.0, seed=1)
    assert s.positions.min() >= 0.0 and s.positions.max() < s.box


def test_nearest_neighbor_distance_is_a_over_sqrt2():
    s = create_fcc_system(3, density=0.8, temperature=1.0, seed=1)
    d = s.positions[:, None, :] - s.positions[None, :, :]
    d -= s.box * np.round(d / s.box)  # minimum image convention
    r = np.sqrt((d ** 2).sum(axis=-1))
    np.fill_diagonal(r, np.inf)
    a = (4.0 / 0.8) ** (1.0 / 3.0)
    assert np.isclose(r.min(), a / np.sqrt(2.0))
