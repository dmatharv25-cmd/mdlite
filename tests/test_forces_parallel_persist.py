import numpy as np

from mdlite.forces_parallel import forces_from_pairs_parallel
from mdlite.neighbor import VerletListForces, _forces_from_pairs
from mdlite.system import create_fcc_system


def _list_and_system(cells, seed):
    s = create_fcc_system(cells, density=0.8, temperature=1.0, seed=seed)
    s.positions += np.random.default_rng(seed).normal(0, 0.05, s.positions.shape)
    s.wrap()
    fn = VerletListForces(0.3, use_cells=False, parallel=False)
    fn(s.positions, s.box)
    return np.ascontiguousarray(s.positions), float(s.box), fn.pair_i, fn.pair_j


def _check(cells, seed, nchunks):
    pos, box, pi, pj = _list_and_system(cells, seed)
    fs, ps, vs = _forces_from_pairs(pos, box, 2.5, pi, pj)
    fp, pp, vp = forces_from_pairs_parallel(pos, box, 2.5, pi, pj, nchunks=nchunks)
    assert np.abs(fs - fp).max() < 1e-10
    assert abs(ps - pp) < 1e-7 * max(1.0, abs(ps))
    assert abs(vs - vp) < 1e-7 * max(1.0, abs(vs))


def test_parallel_matches_serial_for_several_chunk_counts():
    for nc in (None, 1, 3, 7):
        _check(4, 11, nc)


def test_buffer_is_reallocated_when_system_size_changes():
    for cells in (4, 5, 4):
        _check(cells, 3, None)


def test_repeated_calls_on_reused_buffer_are_identical():
    pos, box, pi, pj = _list_and_system(4, 7)
    f1, p1, v1 = forces_from_pairs_parallel(pos, box, 2.5, pi, pj)
    f2, p2, v2 = forces_from_pairs_parallel(pos, box, 2.5, pi, pj)
    assert np.array_equal(f1, f2) and p1 == p2 and v1 == v2
