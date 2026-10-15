import numpy as np

from mdlite.integrators import run_nve
from mdlite.system import create_fcc_system


def test_nve_energy_conserved_and_momentum_zero():
    s = create_fcc_system(3, density=0.8, temperature=1.0, seed=5)
    log = run_nve(s, n_steps=200, dt=0.005, record_every=10)
    e = log["total"]
    assert np.max(np.abs(e - e[0])) / abs(e[0]) < 1e-3
    assert np.allclose(s.velocities.sum(axis=0), 0.0, atol=1e-8)
