import numpy as np

from mdlite.forces import compute_forces_brute
from mdlite.forces_parallel import forces_from_pairs_parallel
from mdlite.neighbor import _build_pairs
from mdlite.system import create_fcc_system


def test_conditional_minimage_matches_brute_force():
    for seed, nch in ((1, None), (2, 6), (3, 24)):
        s = create_fcc_system(5, density=0.8, temperature=1.0, seed=seed)
        s.positions += np.random.default_rng(seed).normal(0, 0.1, s.positions.shape)
        s.wrap()
        pos = np.ascontiguousarray(s.positions)
        box = float(s.box)
        pi, pj = _build_pairs(pos, box, 2.8)
        f, pot, vir = forces_from_pairs_parallel(pos, box, 2.5, pi, pj, nch)
        f0, pot0, vir0 = compute_forces_brute(pos, box, 2.5)
        assert np.allclose(f, f0, atol=1e-9)
        assert abs(pot - pot0) < 1e-7 * max(1.0, abs(pot0))
        assert abs(vir - vir0) < 1e-7 * max(1.0, abs(vir0))
