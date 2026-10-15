# Installation

OpenSG is a JAX code with one conda-packaged dependency block (the FEniCSx
`basix` basis and quadrature library, PETSc and MPI) and one accelerator
(the Intel-MKL Pardiso direct solver, `pypardiso`). The repository ships an
`environment.yml` that installs all of it, including the repository itself
in editable mode, in one command.

## Requirements

- Linux x86-64 (the environment below is the one the examples and timings in
  these docs were produced on)
- [conda](https://docs.conda.io) (Miniconda or Miniforge)
- No GPU is required. CUDA is optional and strictly for acceleration.

## Install (recommended route)

```bash
git clone https://github.com/wenbinyugroup/OpenSG.git
```

```bash
cd OpenSG
```

```bash
conda env create -f environment.yml
```

```bash
conda activate opensg
```

To install a released version instead of the branch head, clone a tag:
`git clone --branch v2.1.0 https://github.com/wenbinyugroup/OpenSG.git`.

The environment file does three things, in order: it installs the conda-forge
scientific stack with the FEniCSx `basix` pinned (`fenics-basix 0.8.0`; pip
cannot install it on every platform), it pip-installs the JAX stack, and it
finishes with `pip install -e .` so `import opensg_solid` and
`import opensg_shell` work from anywhere with no `sys.path` boilerplate; every
example script relies on that.

The editable install also puts three console scripts on your `PATH`:

| command | what it is |
|---|---|
| `opensg <sg.yaml> [H\|D]` | the unified entry point: dispatches on the yaml's `msg:` key |
| `opensg_solid <sg.yaml> [H\|D]` | the general 1-D / 2-D / 3-D SG engine |
| `opensg_shell <sg.yaml> [H\|D]` | the msg-shell contour / surface engine |

Each also runs as `python -m opensg_solid` / `python -m opensg_shell`. Pass
`--help` (or no argument) to any of them to print the header contract it reads.

What `environment.yml` declares, and why:

| package | source | why |
|---|---|---|
| `python` 3.13 | conda-forge | the verified interpreter |
| `fenics-basix` 0.8.0 | conda-forge | basis tables and quadrature rules of the FE core; the 0.9 API is not compatible |
| `petsc` 3.23.7 (CUDA-12 build), `petsc4py` 3.23.7, `mpich`, `mpi4py` | conda-forge | the optional GAMG solver (`--solver iter 4`); the version pin keeps the resolver on Python 3.13 |
| `numpy`, `scipy`, `meshio`, `pyyaml`, `matplotlib`, `pyvista`, `numba`, `psutil` | conda-forge | resolved by conda |
| `jax[cuda12]`, `cupy-cuda12x` | pip | the JAX stack; the CUDA build runs on CPU-only machines too (JAX falls back to the CPU) |
| `jaxopt`, `flax`, `libigl`, `pyamg` | pip | `pyproject.toml` dependencies |
| `pypardiso` | pip | the default `solver="direct"` factorization (CPU, MKL) |
| `windio` | pip | reading windIO v2 blade files (blade pipeline only) |

## CPU-only and GPU variants

`environment.yml` installs the CUDA-12 JAX build by default so that one
environment serves every machine; on a machine without a GPU the only cost is
the size of the NVIDIA wheels. For a lean CPU-only environment replace the two
pip lines `jax[cuda12]` and `cupy-cuda12x` with plain `jax`, and the petsc
pin with plain `petsc4py`, before creating the environment.

For CUDA 13 replace them with `jax[cuda13]` and `cupy-cuda13x` (the pair must
match your CUDA toolkit). Two consequences to know about: `pypardiso` is a
CPU/MKL solver, so on a GPU pass `--solver iter` on the command line
(`solver="cg"` in the Python API) instead of the default direct factorization;
and the optional `pyamgx` multigrid preconditioner needs NVIDIA AMGX built
separately.

## Plain pip

On a machine with MPI and a C compiler the package also installs with pip:

```bash
pip install -e .
```

This installs the `pyproject.toml` dependencies (`jax` CPU build, `fenics-basix`,
`pypardiso`, ...). The extras are `.[cuda12]`, `.[cuda13]`, `.[gamg]`
(petsc4py, which pip builds from source), `.[dev]` (pytest) and `.[docs]`.

## Verify the install

The test suite:

```bash
pytest tests
```

The RM plate gate, the suite that every homogenization or recovery change is
gated on:

```bash
python -m pytest tests/rm_plate_1D -q
```

And a two-second end-to-end smoke test that exercises yaml input, a real
solve and the timed `.out` output, through the CLI:

```bash
cd examples/OpenSG-solid/3_get_plate_props_from_2D_SG
```

```bash
opensg RHC_SW_2UC_45.yaml
```

A successful run echoes the resolved engine, SG dimension and macro model,
prints the plate law, and writes `RHC_SW_2UC_45.out`, the law in the
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
  also why the unified `opensg` dispatcher is deliberately import-light: it
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
**OpenSG_io**, a separate repository, to turn a windIO blade description
into per-station shell SG yamls. Install it alongside OpenSG following
that page.
