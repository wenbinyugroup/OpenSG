# Download and installation

OpenSG is a Python code built on JAX. Besides the JAX stack it needs one conda-packaged dependency block, the FEniCSx
`basix` basis and quadrature library together with PETSc and MPI, and one accelerator, the Intel MKL (Math Kernel
Library) Pardiso direct solver through `pypardiso`. The repository ships an `environment.yml` that installs all of it,
including OpenSG itself in editable mode, in one command.

## Prerequisites

- Linux x86-64. The environment below is the one the examples and timings in this documentation were produced on.
- [conda](https://docs.conda.io), through Miniconda or Miniforge.
- git, to download the repository.

:::{note}
No GPU is required. CUDA is optional and serves acceleration only: every analysis runs on the CPU.
:::

## Install with conda (recommended)

Download the repository:

```bash
git clone https://github.com/wenbinyugroup/OpenSG.git
```

Change to its root folder:

```bash
cd OpenSG
```

Create the environment from the file the repository ships:

```bash
conda env create -f environment.yml
```

Activate it:

```bash
conda activate opensg
```

% source: docs/installation.md (release tag)
:::{tip}
To install a released version instead of the head of the branch, clone a tag:
`git clone --branch v2.1.0 https://github.com/wenbinyugroup/OpenSG.git`.
:::

The environment file does three things, in order:

1. It installs the conda-forge scientific stack, with the FEniCSx `basix` pinned to `fenics-basix 0.8.0` (pip cannot
   install `basix` on every platform).
2. It pip-installs the JAX stack.
3. It finishes with `pip install -e .`, so `import opensg_solid` and `import opensg_shell` work from any folder with no
   `sys.path` lines; every example script relies on that.

The editable install also puts three console scripts on your `PATH` ({numref}`tab-install-commands`).

% source: docs/installation.md; pyproject.toml [project.scripts]
```{table} Commands installed with OpenSG
:name: tab-install-commands

| Command | What it is |
|---|---|
| `opensg <sg.yaml> [H\|D]` | the unified entry point: it dispatches on the `msg:` key of the yaml |
| `opensg_solid <sg.yaml> [H\|D]` | the general 1-D / 2-D / 3-D structure gene (SG) engine |
| `opensg_shell <sg.yaml> [H\|D]` | the msg-shell contour / surface engine |
```

Each engine also runs as `python -m opensg_solid` or `python -m opensg_shell`. Pass `--help`, or no argument, to any
of the three commands to print its usage; `opensg_solid --help` and `opensg_shell --help` also print the header keys
the engine reads. {doc}`running` explains every option.

{numref}`tab-install-environment` lists what `environment.yml` declares and why.

% source: environment.yml; docs/installation.md
```{table} Packages declared in environment.yml
:name: tab-install-environment

| Package | Source | Why |
|---|---|---|
| `python` 3.13 | conda-forge | the verified interpreter |
| `fenics-basix` 0.8.0 | conda-forge | basis tables and quadrature rules of the finite-element (FE) core; the 0.9 API is not compatible |
| `petsc` 3.23.7 (CUDA-12 build), `petsc4py` 3.23.7, `mpich`, `mpi4py` | conda-forge | the optional PETSc GAMG (geometric algebraic multigrid) solver (`--solver iter 4`); the version pin keeps the resolver on Python 3.13 |
| `numpy`, `scipy`, `meshio`, `pyyaml`, `matplotlib`, `pyvista`, `numba`, `psutil` | conda-forge | resolved by conda |
| `jax[cuda12]`, `cupy-cuda12x` | pip | the JAX stack; the CUDA build runs on CPU-only machines too (JAX falls back to the CPU) |
| `jaxopt`, `flax`, `libigl`, `pyamg` | pip | `pyproject.toml` dependencies |
| `pypardiso` | pip | the default `solver="direct"` factorization (CPU, MKL) |
| `windio` | pip | reading windIO v2 blade files (blade pipeline only) |
```

:::{warning}
Keep the two pins. The `fenics-basix` 0.9 API is not compatible with the FE core, and without the `petsc` /
`petsc4py` 3.23.7 pin conda resolves a `petsc4py` that has no Python 3.13 build and takes the whole environment down to
Python 3.10.
:::

:::{note}
The FEniCSx DOLFINx stack (`fenics-dolfinx`, `-ufl`, `-ffcx`) is not imported anywhere in OpenSG and is deliberately
left out of the environment: `dolfinx` 0.8.0 pins `petsc` below 3.22, which blocks the CUDA `petsc` above.
:::

## CPU-only and GPU variants

% source: environment.yml (pip-section comment: ~2.5 GB wheels, jax==0.10.2/jaxlib==0.10.2; petsc comment); docs/installation.md
`environment.yml` installs the CUDA-12 JAX build by default so that one environment serves every machine; on a machine
without a GPU the only cost is the size of the NVIDIA wheels (about 2.5 GB, as the comment in `environment.yml` notes).

**Lean CPU-only environment.** Before creating the environment, replace the two pip lines `jax[cuda12]` and
`cupy-cuda12x` with plain `jax` (the comment in `environment.yml` gives the pinned pair `jax==0.10.2` and
`jaxlib==0.10.2`), and replace the `petsc` pin with plain `petsc4py`.

**CUDA 13.** Replace the two pip lines with `jax[cuda13]` and `cupy-cuda13x`; the pair must match your CUDA toolkit.
Two consequences follow:

- `pypardiso` is a CPU/MKL solver, so on a GPU pass `--solver iter` on the command line (`solver="cg"` in the Python
  API) instead of the default direct factorization.
- The optional `pyamgx` multigrid preconditioner needs NVIDIA AMGX, built separately.

A CUDA build of `petsc` also runs on CPU-only machines; it only cannot offer the `aijcusparse` matrix type, and OpenSG
probes for that type and falls back to `aij`.

## Install with pip

On a machine with MPI and a C compiler the package also installs with pip, from the repository root:

```bash
pip install -e .
```

% source: pyproject.toml [project] dependencies, requires-python; docs/installation.md
This installs the `pyproject.toml` dependencies (the `jax` CPU build, `fenics-basix`, `pypardiso`, and the rest) and
needs Python 3.10 or newer. Optional extras add to it ({numref}`tab-install-extras`).

% source: docs/installation.md; pyproject.toml [project.optional-dependencies], requires-python
```{table} Optional extras of the pip install
:name: tab-install-extras

| Extra | Adds |
|---|---|
| `.[cuda12]` | `jax[cuda12]`, `cupy-cuda12x` |
| `.[cuda13]` | `jax[cuda13]`, `cupy-cuda13x` |
| `.[gamg]` | `petsc4py`, which pip builds from source |
| `.[dev]` | `pytest` |
| `.[docs]` | Sphinx and the MyST parser, for building this documentation |
```

The documentation itself builds from a clean checkout without the JAX or `basix` runtime:
`pip install -r docs/requirements.txt`.

## Verify the installation

Run the test suite from the repository root:

```bash
pytest tests
```

The Reissner-Mindlin (RM) plate gate is the suite that every homogenization or recovery change of the code is checked
against:

```bash
python -m pytest tests/rm_plate_1D -q
```

% source: docs/installation.md
A two-second end-to-end smoke test exercises the yaml input, a real solve and the timed `.out` output through the
command line. Change to the 2-D honeycomb plate example:

```bash
cd examples/OpenSG-solid/3_get_plate_props_from_2D_SG
```

and run it:

```bash
opensg RHC_SW_2UC_45.yaml
```

A successful run echoes the resolved engine, SG dimension and macro model, prints the plate law, and writes
`RHC_SW_2UC_45.out`, the law in the SwiftComp `.K` layout, closing with a ` Time taken:` line.

The 1-D layup route is the same check from the other side. It also generates its SG mesh, so it runs through its
driver script:

```bash
cd examples/OpenSG-solid/1_get_plate_props_from_1DSG
```

```bash
python 1d_sg.py
```

The script writes `1dsg.yaml`, `1dsg.png` and `layup_db_plate_homo.out`.

## What the package configures at import time

The package `__init__` files set the following up. Nothing needs to be done, but they are worth knowing when reading
timings or debugging an unusual machine.

**Float64 everywhere.**
: `opensg_solid` and `opensg_shell` switch JAX to `jax_enable_x64` in their package `__init__`. This matters: at
  float32 the homogenization results drift at single-precision epsilon. It is also why the unified `opensg` dispatcher
  is import-light: it loads neither JAX nor either engine until the input file has been resolved, so the engine's own
  `__init__` is still the first thing that runs.

**Persistent JAX compilation cache.**
: Compiled kernels are cached at `~/.cache/opensg_jax` (override with `JAX_COMPILATION_CACHE_DIR`). The first run of a
  given problem shape pays each XLA (Accelerated Linear Algebra) compile once; later runs load the compiled binaries. A cold first timing is
  therefore larger than the ` Time taken:` lines quoted in this documentation.

**`FE_JAX_CORE` (optional).**
: Prepends an external `fe_jax` checkout to `sys.path`, overriding the copy bundled at `src/fe_jax`. Leave it unset
  for a normal install.

**`PYPARDISO_MKL_RT`.**
: Pointed at the conda MKL runtime automatically, which avoids a slow file-system scan on the first Pardiso solve.

## Companion tool for blade work

The windIO blade pipeline ({doc}`../examples/blade_windio_pipeline`) uses **OpenSG_io**, a separate repository, to
turn a windIO blade description into per-station shell SG yamls. Install it alongside OpenSG as described on that
page.

Once OpenSG is installed, continue with {doc}`running`.
