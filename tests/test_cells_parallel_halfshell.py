import numpy as np

from mdlite.cells_parallel import build_pairs_cells_parallel
from mdlite.neighbor import _build_pairs
from mdlite.system import create_fcc_system


def _keys(pi, pj, n):
    return np.sort(pi.astype(np.int64) * n + pj.astype(np.int64))


def test_half_shell_pairs_match_brute_force():
    for cells, seed, r_list in ((6, 1, 2.8), (7, 2, 2.8), (6, 3, 3.4), (8, 4, 2.0)):
        s = create_fcc_system(cells, density=0.8, temperature=1.0, seed=seed)
        s.positions += np.random.default_rng(seed).normal(0, 0.1, s.positions.shape)
        s.wrap()
        pos = np.ascontiguousarray(s.positions)
        box = float(s.box)
        n = len(pos)
        ei, ej = _build_pairs(pos, box, r_list)
        gi, gj = build_pairs_cells_parallel(pos, box, r_list)
        assert (gi < gj).all()
        assert len(gi) == len(ei)
        assert np.array_equal(_keys(gi, gj, n), _keys(ei, ej, n))
