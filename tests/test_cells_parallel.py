import numpy as np

from mdlite.cells import build_pairs_cells
from mdlite.cells_parallel import build_pairs_cells_parallel
from mdlite.neighbor import _build_pairs
from mdlite.system import create_fcc_system


def _perturbed(n_cells, seed):
    s = create_fcc_system(n_cells, density=0.8, temperature=1.0, seed=seed)
    s.positions += np.random.default_rng(seed).normal(0, 0.08, s.positions.shape)
    s.wrap()
    return s


def _as_set(pi, pj):
    return set(zip(pi.tolist(), pj.tolist()))


def test_parallel_cell_pairs_equal_brute_pairs():
    s = _perturbed(5, 31)
    pi, pj = build_pairs_cells_parallel(s.positions, s.box, 2.8)
    bi, bj = _build_pairs(s.positions, s.box, 2.8)
    assert _as_set(pi, pj) == _as_set(bi, bj)


def test_parallel_cell_pairs_equal_serial_cell_pairs_with_four_cells_per_side():
    s = _perturbed(8, 32)   # box ~13.7 -> 4 cells per side
    pi, pj = build_pairs_cells_parallel(s.positions, s.box, 2.8)
    si, sj = build_pairs_cells(s.positions, s.box, 2.8)
    assert _as_set(pi, pj) == _as_set(si, sj)


def test_parallel_cell_pairs_are_unique_and_ordered():
    s = _perturbed(6, 33)
    pi, pj = build_pairs_cells_parallel(s.positions, s.box, 2.8)
    assert np.all(pi < pj)
    assert len(_as_set(pi, pj)) == len(pi)
