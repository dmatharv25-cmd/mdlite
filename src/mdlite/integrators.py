"""Time integration: velocity Verlet in the NVE (microcanonical) ensemble."""

import numpy as np

from mdlite.forces import compute_forces_brute
from mdlite.integrate_parallel import half_kick, kick_drift_wrap


def _can_fuse(system) -> bool:
    for a in (system.positions, system.velocities):
        if a.dtype != np.float64 or not a.flags["C_CONTIGUOUS"] or not a.flags["WRITEABLE"]:
            return False
    return True


def run_nve(system, n_steps: int, dt: float, r_cut: float = 2.5,
            force_fn=compute_forces_brute, record_every: int = 10) -> dict:
    """Advance the system with velocity Verlet, recording energies.

    Returns a dict of arrays: time, kinetic, potential, total, temperature.
    """
    forces, pot, _ = force_fn(system.positions, system.box, r_cut)
    system.forces = forces

    t_log, ke_log, pe_log, temp_log = [], [], [], []

    def record(step, pot_energy):
        t_log.append(step * dt)
        ke_log.append(system.kinetic_energy())
        pe_log.append(pot_energy)
        temp_log.append(system.temperature())

    fused = _can_fuse(system)
    record(0, pot)
    for step in range(1, n_steps + 1):
        if fused:
            f = np.ascontiguousarray(system.forces, dtype=np.float64)
            kick_drift_wrap(system.positions, system.velocities, f, dt, float(system.box))
        else:
            system.velocities += 0.5 * dt * system.forces      # half kick
            system.positions += dt * system.velocities          # drift
            system.wrap()
        system.forces, pot, _ = force_fn(system.positions, system.box, r_cut)
        if fused:
            f = np.ascontiguousarray(system.forces, dtype=np.float64)
            half_kick(system.velocities, f, dt)
        else:
            system.velocities += 0.5 * dt * system.forces      # half kick
        if step % record_every == 0:
            record(step, pot)

    ke = np.array(ke_log)
    pe = np.array(pe_log)
    return {
        "time": np.array(t_log),
        "kinetic": ke,
        "potential": pe,
        "total": ke + pe,
        "temperature": np.array(temp_log),
    }
