# Installation

OpenSG-2.0 is a JAX code with one conda-packaged dependency block (the FEniCSx
`basix` basis/quadrature stack) and one optional accelerator (the Intel-MKL
Pardiso direct solver). The repository ships an `environment.yml` that installs
all of it — including the repository itself in editable mode — in one command.

## Requirements

- Linux x86-64 (the environment below is the one the examples and timings in
  these docs were produced on)
- [conda](https://docs.conda.io) (Miniconda or Miniforge)
- No GPU is required. CUDA is optional and strictly for acceleration.

## Install (recommended route)

```bash
git clone https://github.com/bagla0/OpenSG-2.0.git
```

```bash
cd OpenSG-2.0
```

```bash
conda env create -f environment.yml
```

```bash
conda activate opensg_2_0
```

The environment file does three things, in order: it installs the conda-forge
scientific stack with the FEniCSx components pinned (`fenics-basix 0.8.0` and
its companions — pip cannot install these), it pip-installs the JAX stack at
the verified versions, and it finishes with `pip install -e .` so
`import opensg_solid` and `import opensg_shell` work from anywhere with no
`sys.path` boilerplate — every example script relies on that.

The editable install also puts three console scripts on your `PATH`:

| command | what it is |
|---|---|
| `opensg <sg.yaml> [H\|D]` | the unified entry point — dispatches on the yaml's `msg:` key |
| `opensg_solid <sg.yaml> [H\|D]` | the general 1-D / 2-D / 3-D SG engine |
| `opensg_shell <sg.yaml> [H\|D]` | the msg-shell contour / surface engine |

Each also runs as `python -m opensg_solid` / `python -m opensg_shell`. Pass
`--help` (or no argument) to any of them to print the header contract it reads.

The load-bearing pins, and the versions the environment resolves to as
verified on a clean install:

| package | version | why it is pinned |
|---|---|---|
| `python` | 3.13 | the verified interpreter |
| `jax` / `jaxlib` | 0.10.2 | the solver stack is timed and gated at this release |
| `fenics-basix` (with `fenics-dolfinx`, `fenics-ffcx` 0.8.0, `fenics-ufl` 2024.1.0) | 0.8.0 | basis tables and quadrature rules; the 0.9 API is not compatible |
| `numpy` / `scipy` | 2.3 / 1.16 (resolved) | free — resolved by conda |
| `jaxopt`, `flax`, `numba`, `meshio`, `libigl`, `psutil` | resolved | `pyproject.toml` dependencies |
| `pypardiso` | 0.4.7 (resolved) | the default `solver="direct"` factorization (CPU/MKL) |
| `mpi4py`, `petsc4py`, `mpich` | resolved | conda-forge FEniCSx companions |
| `matplotlib`, `pyvista`, `pyyaml` | resolved | mesh figures and YAML input |
| `windio` | 2.1.1 (resolved) | reading windIO v2 blade files (blade pipeline only) |

## GPU install

Edit the pip section of `environment.yml` before creating the environment:
replace the two pinned lines `jax==0.10.2`, `jaxlib==0.10.2` with

```text
- jax[cuda13]
- cupy-cuda13x
```

(`cuda12` variants are also supported — the pair must match your CUDA
toolkit). Two consequences to know about: `pypardiso` is a CPU/MKL solver, so
on GPU pass `solver="cg"` to the homogenization calls instead of the default
`"direct"`; and the optional `pyamgx` multigrid preconditioner (see the
repository README) needs NVIDIA AMGX built separately.

## Verify the install

The general FE core test suite:

```bash
pytest tests
```

The RM plate gate — the 15-test suite that every homogenization/recovery
change is gated on:

```bash
python -m pytest src/opensg_solid/rm_plate_1D/tests -q
```

And a two-second end-to-end smoke test that exercises YAML input, a real
solve and the timed `.out` output, through the CLI:

```bash
cd examples/OpenSG-solid/3_get_plate_props_from_2D_SG
```

```bash
opensg RHC_SW_2UC_45.yaml
```

A successful run echoes the resolved engine, SG dimension and macro model,
prints the plate law, and writes `RHC_SW_2UC_45.out` — the law in the
SwiftComp `.K` layout, closing with a ` Time taken:` line.

The 1-D layup route is the same check from the other side; it also generates
its SG mesh, so it is run through its driver:

```bash
cd examples/OpenSG-solid/1_get_plate_props_from_1DSG
```

```bash
python 1d_sg.py
```

which writes `1dsg.yaml`, `1dsg.png` and `layup_db_plate_homo.out`.

## What the package configures at import time

These are set up by the package `__init__` files; nothing to do, but worth
knowing when reading timings or debugging an unusual machine:

- **float64 everywhere.** `opensg_solid` and `opensg_shell` switch JAX to
  `jax_enable_x64` in their package `__init__`. This is load-bearing: at
  float32 the homogenization gates drift at single-precision epsilon. It is
  also why the unified `opensg` dispatcher is deliberately import-light — it
  pulls in neither JAX nor either engine until a file has been resolved, so
  the engine's own `__init__` is still the first thing that runs.
- **Persistent JAX compilation cache** at `~/.cache/opensg_jax` (override with
  `JAX_COMPILATION_CACHE_DIR`). The first run of a given problem shape pays
  each XLA compile once; warm runs load the compiled binaries, which is why a
  cold first timing is larger than the ` Time taken:` footers quoted in these
  docs.
- **`FE_JAX_CORE`** (optional) prepends an external `fe_jax` checkout to
  `sys.path`, overriding the copy bundled at `src/fe_jax`. Leave it unset for
  a normal install.
- **`PYPARDISO_MKL_RT`** is pointed at the conda MKL runtime automatically,
  avoiding a slow filesystem scan on the first Pardiso solve.

## Companion tool for blade work

The windIO blade pipeline (see {doc}`tutorials/windio_blade_pipeline`) uses
**OpenSG_io** — a separate repository — to turn a windIO blade description
into per-station shell SG YAMLs. Install it alongside OpenSG-2.0 following
that page.
