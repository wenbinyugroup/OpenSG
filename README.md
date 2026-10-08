# FEA in JAX

`fea-in-jax` is a Finite Element Analysis (FEA) library written in JAX. It leverages JAX's composable function transformations—JIT compilation, automatic differentiation, and vectorization—to provide a high-performance solver capable of running on GPUs and TPUs.

## OpenSG: multiscale structural mechanics

![OpenSG: Structure Genes homogenized into solid, plate/shell and beam models](docs/images/opensg_overview.webp)

OpenSG is the multiscale structural-mechanics capability of `fea-in-jax`, built on the Mechanics of Structure Genome (MSG). It uses a Structure Gene (SG) to obtain the constitutive relations of solid, plate/shell and beam models, and it performs dehomogenization to recover the local stresses from the structural response.

**Structure Gene (SG):** heterogeneity and anisotropy come from a Structure Gene meshed with solid elements. An SG can be built from solid, shell and beam elements.

| SG meshed with | examples | homogenized model |
|---|---|---|
| solid elements | 1D SG, 2D SG, 3D SG | 3D solid, 2D plate/shell, 1D beam |
| shell elements | airfoil, tapered shell, TPMS shell | 1D beam, 2D plate/shell |
| beam elements | BCC, octahedron, FCC lattices | 3D solid, 2D plate/shell |

### Installing OpenSG

OpenSG needs one conda-packaged dependency block (the FEniCSx `basix` basis and quadrature stack) on top of the JAX stack. The repository ships an `environment.yml` that installs everything, including the repository itself in editable mode:

```bash
git clone -b akshat/opensg https://github.com/KeithBallard/fea-in-jax.git
cd fea-in-jax
conda env create -f environment.yml
conda activate opensg
```

The editable install puts three commands on your `PATH`; each takes an SG yaml and `H` (homogenize) or `D` (dehomogenize):

| command | what it is |
|---|---|
| `opensg <sg.yaml> [H\|D]` | the unified entry point, dispatching on the yaml's `msg:` key |
| `opensg_solid <sg.yaml> [H\|D]` | the general 1D / 2D / 3D SG engine (solid elements) |
| `opensg_shell <sg.yaml> [H\|D]` | the shell-element SG engine (contours and surfaces) |

Worked examples are under `examples/OpenSG-solid` and `examples/OpenSG_shell`. Version pins, the GPU variant and the solver options are in [docs/installation.md](docs/installation.md).

## Features

*   **GPU Acceleration**: Native support for hardware acceleration via JAX.
*   **Differentiability**: Differentiate through the physics simulation for gradient-based optimization and machine learning integration.
*   **Batched Computation**: Designed to efficiently handle large batches of elements and quadrature points.

## Project Structure

*   `src/fe_jax`: Core library source code, including element definitions, quadrature rules, and solver implementations.
*   `tests`: extensive test suite that also serves as a catalogue of usage examples.
*   `docs`: Documentation and theoretical background.

## Getting Started

### Prerequisites

*   Python 3.10+
*   [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (optional, strictly for GPU acceleration)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd fea-in-jax
    ```

2.  **Set up a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: `jax` installation instructions vary depending on your hardware (CPU, GPU, TPU). Please refer to the [JAX installation guide](https://github.com/google/jax#installation) if the default pip install does not match your system configuration.*

    For development, install the package in editable mode with the test dependency:
    ```bash
    pip install -e ".[dev]"
    ```

4.  **(Optional) Install `pyamgx`:**
    To enable GPU-accelerated algebraic multigrid preconditioners:
    1.  Install NVIDIA's [AMGX](https://github.com/NVIDIA/AMGX?tab=readme-ov-file#quickstart).
    2.  Install [pyamgx](https://pyamgx.readthedocs.io/en/latest/install.html).

## Running Tests

To verify the installation and run the test suite:

```bash
pytest tests
```

## Usage

The `tests` directory contains numerous examples demonstrating how to define meshes, apply boundary conditions, and solve boundary value problems.

*   **Basic Linear Elasticity**: See `tests/test_simple_fea_solve.py` for a straightforward example.
*   **Complex Scenarios**: See `tests/test_fea_solve.py`.

## Theory and Implementation

For detailed information on the nonlinear solver derivation, handling of Dirichlet boundary conditions, and internal variable definitions, please refer to [docs/theory.md](docs/theory.md).

## Resources

*   **JAX Interoperability**: [External Callbacks](https://apxml.com/courses/advanced-jax/chapter-5-jax-interoperability-custom-operations/using-jax-pure-callback)
*   **Scientific Computing in JAX**:
    *   [Wrapping Scipy KD trees](https://robertdyro.com/articles/jax_advanced/)
    *   [HPC Lecture Notes](https://tbetcke.github.io/hpc_lecture_notes/intro.html)
*   **Performance Optimization**:
    *   [JAX GPU Performance Tips](https://jax.readthedocs.io/en/latest/gpu_performance_tips.html)
    *   [JAX AOT Compilation](https://jax.readthedocs.io/en/latest/aot.html)
    *   [Multi-Process/Distributed Support](https://jax.readthedocs.io/en/latest/gpu_performance_tips.html#multi-process)

### Profiling Performance

*   [JAX Profiling Docs](https://jax.readthedocs.io/en/latest/profiling.html)
*   [NVIDIA JAX Toolbox](https://github.com/NVIDIA/JAX-Toolbox/blob/main/docs/profiling.md)
*   [NSys-JAX Wrapper](https://github.com/NVIDIA/JAX-Toolbox/blob/main/docs/nsys-jax.md)
*   [JAX Device Memory Profiling](https://jax.readthedocs.io/en/latest/device_memory_profiling.html)
*   [jax-smi (GPU Memory Tracking)](https://github.com/ayaka14732/jax-smi)

To profile time and memory for JIT-compiled sections:
1.  Collect trace: `jax.profiler.start_trace("<directory>/prof")`
2.  Visualize: using TensorBoard or `xprof`.

## Public Release Information

Distribution Statement A. Approved for public release: distribution is unlimited. Case #: AFRL-2025-4644
