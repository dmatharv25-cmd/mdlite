import numpy as np

from mdlite.system import create_fcc_system
from mdlite.wrap_parallel import wrap_inplace


def _periodic_diff(a, b, box):
    d = a - b
    return d - box * np.round(d / box)


def test_wrap_matches_numpy_remainder():
    rng = np.random.default_rng(0)
    box = 7.3
    pos = rng.uniform(-2.0 * box, 3.0 * box, size=(5000, 3))
    expected = pos % box
    got = pos.copy()
    wrap_inplace(got, box)
    assert np.abs(_periodic_diff(got, expected, box)).max() < 1e-12
    assert got.min() >= 0.0 and got.max() <= box


def test_wrap_leaves_in_range_values_untouched():
    rng = np.random.default_rng(1)
    box = 5.0
    pos = rng.uniform(0.0, box, size=(1000, 3))
    got = pos.copy()
    wrap_inplace(got, box)
    assert np.array_equal(got, pos)


def test_system_wrap_puts_all_atoms_in_box():
    s = create_fcc_system(4, density=0.8, temperature=1.0, seed=5)
    s.positions += np.random.default_rng(5).normal(0, 3.0, s.positions.shape)
    s.wrap()
    assert s.positions.min() >= 0.0 and s.positions.max() <= s.box


def test_wrap_works_in_place_on_fortran_ordered_array():
    box = 4.0
    pos = np.asfortranarray(np.random.default_rng(2).uniform(-10.0, 10.0, size=(200, 3)))
    expected = pos.copy() % box
    wrap_inplace(pos, box)
    assert np.abs(_periodic_diff(pos, expected, box)).max() < 1e-12
