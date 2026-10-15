# mdlite

![CI](https://github.com/dmatharv25-cmd/mdlite/actions/workflows/ci.yml/badge.svg)

A small Lennard-Jones molecular dynamics code in Python, built step by step from a plain NumPy baseline up to a parallel Numba implementation. Each optimization is benchmarked and tested against the previous one.

## Features

- FCC lattice setup with Maxwell-Boltzmann velocities and periodic boundaries
- Velocity Verlet integrator (NVE)
- Lennard-Jones forces: NumPy all-pairs baseline, Numba serial kernel, parallel kernel with persistent per-chunk force buffers
- Verlet neighbor lists with a skin distance and lazy rebuild
- O(N) cell-list neighbor list construction, including a parallel half-shell build
- Fused parallel integrator kernels and parallel displacement check
- Test suite (pytest) covering each kernel against the reference implementation
- Physics validation against NIST Lennard-Jones Monte Carlo reference data (see Validation)

## Install

```bash
git clone https://github.com/dmatharv25-cmd/mdlite.git
cd mdlite
pip install -e .
pip install pytest matplotlib
python -m pytest -q
```

## Quick start

```python
from mdlite.integrators import run_nve
from mdlite.system import create_fcc_system

system = create_fcc_system(4, density=0.8, temperature=1.0, seed=42)
log = run_nve(system, n_steps=2000, dt=0.005, record_every=10)

print(log["total"][0], log["total"][-1])  # total energy at start and end
```

The full script is `examples/nve_energy.py`. It runs a Lennard-Jones liquid and plots energy components and relative energy drift.

![NVE energy](examples/nve_energy.png)

## Running the parallel version

The fast path uses the cell-list neighbor build and the parallel force kernel. Pass a `VerletListForces` object with `parallel=True` to `run_nve`:

```python
import numba
from mdlite.integrators import run_nve
from mdlite.neighbor import VerletListForces
from mdlite.system import create_fcc_system

system = create_fcc_system(10, density=0.8, temperature=1.0, seed=1)
force_fn = VerletListForces(0.3, use_cells=True, parallel=True)  # 0.3 = neighbor-list skin distance (sigma)

# optional: limit the number of threads (default is all cores)
# numba.set_num_threads(4)

run_nve(system, n_steps=100, dt=0.005, force_fn=force_fn)
print("ok, N =", system.n_particles)
```

The first call is slow because Numba compiles the kernels (results are cached afterwards). To compare against the serial version, set `parallel=False`. `benchmarks/bench_parallel.py` does exactly that and also measures thread scaling.

## Benchmarks

All timings were measured on a single development machine, so absolute numbers will differ on yours. The raw data is in `benchmarks/`.

### Force evaluation: NumPy vs Numba (ms per force call)

| N | NumPy | Numba serial | Speedup |
|------:|--------:|-------------:|--------:|
| 108 | 2.02 | 0.138 | 14.7x |
| 256 | 13.82 | 0.478 | 28.9x |
| 500 | 59.32 | 1.614 | 36.8x |
| 864 | 178.90 | 4.449 | 40.2x |
| 1372 | 377.28 | 10.81 | 34.9x |

### Neighbor lists: all-pairs vs cell list (ms per step)

| N | All-pairs | Cell-list neighbor list | Speedup |
|------:|----------:|------------------------:|--------:|
| 864 | 5.06 | 2.80 | 1.8x |
| 2048 | 25.64 | 6.42 | 4.0x |
| 4000 | 91.30 | 10.04 | 9.1x |

The cell-list version keeps scaling roughly linearly: going from N = 4000 to N = 23328 (5.8x more particles) takes 7.0x longer.

### Parallel vs serial (ms per step)

| N | Serial | Parallel | Speedup |
|------:|-------:|---------:|--------:|
| 864 | 2.91 | 1.07 | 2.7x |
| 4000 | 10.32 | 4.43 | 2.3x |
| 10976 | 31.68 | 9.15 | 3.5x |
| 23328 | 72.41 | 18.29 | 4.0x |
| 42592 | 122.47 | 29.96 | 4.1x |

Parallel efficiency improves with system size because the fixed per-step overhead matters less.

![Parallel scaling](benchmarks/parallel.png)
![Neighbor list scaling](benchmarks/scaling.png)

## Validation

Besides the kernel-vs-reference unit tests, mdlite is checked against the NIST Lennard-Jones benchmark pages (https://mmlapps.nist.gov/srs/LJ_PURE/). The setup follows NIST: N = 500, T* = 0.85, cutoff 3 sigma (truncated, not shifted), standard long-range corrections, velocity Verlet with dt = 0.005. The code's shifted potential is converted to the plain truncated one when computing the energy.

Reference values are from NIST's NVT Monte Carlo table (`mc.htm`). Energy is per particle, in reduced units:

| rho | T (run) | U mdlite | U NIST MC | dU | p mdlite | p NIST MC (+-) | dp |
|----:|--------:|---------:|----------:|-------:|---------:|---------------:|-------:|
| 0.82 | 0.843 | -5.790 | -5.795 | +0.005 | 0.564 | 0.554 (0.004) | +0.011 |
| 0.86 | 0.850 | -6.029 | -6.031 | +0.002 | 1.272 | 1.266 (0.014) | +0.006 |
| 0.90 | 0.873 | -6.228 | -6.239 | +0.012 | 2.282 | 2.231 (0.027) | +0.051 |

Notes and limits:

- Each state is one run (single seed) with 60 time units of production after melting and equilibration. The NVE temperature drifts from the 0.85 target by up to 0.02, so U and p are corrected to T = 0.85 with fixed slopes taken from the NIST MC isotherms at T = 0.85 and 0.90. The test asserts |dU| < 0.03 and |dp| < 0.10 (and |T - 0.85| < 0.04), which is looser than the observed agreement.
- At rho = 0.9 the pressure is about two MC standard deviations above the NIST value. I have not resolved whether that is statistical noise or a small systematic effect.
- NIST's MD page (`md.htm`) lists p = 2.544 at rho = 0.9, T = 0.851, which disagrees with NIST's own MC value (2.231 +- 0.027) by about 0.3. mdlite agrees with the MC value. In cross-checks, the neighbor-list virial matched a brute-force and an independent NumPy virial, and the diffusion coefficient matched NIST's MD value. I do not know the source of the difference between the two NIST tables, so the MC table is used for pressure.
- Only N = 500, T* = 0.85 and rho = 0.82 to 0.90 are covered.

Run it (about 90 seconds) with:

```bash
python -m pytest -m slow -s
```

`validation/nist_lj.py` is the exploratory script that also computes g(r) (`validation/rdf.png`). The test in `tests/test_nist_lj.py` is the authoritative check.

## Project layout

```text
src/mdlite/     package: forces, neighbor lists, cell lists, integrators, parallel kernels, system setup
tests/          pytest suite; test_nist_lj.py is the slow physics validation
benchmarks/     timing scripts, CSV data and plots
examples/       nve_energy.py and its plot
validation/     NIST comparison script and g(r) plot
```
