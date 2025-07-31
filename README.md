# OpenSG: Open Source Structural Analysis for Wind Turbine Blades

OpenSG is a Python package for structural analysis of wind turbine blades using Mixed-Space-Galerkin (MSG) formulations and finite element methods.

## Features

- **Blade Mesh Management**: Handle complex wind turbine blade geometries
- **Segment Analysis**: Analyze individual blade segments
- **ABD Matrix Computation**: Compute stiffness matrices for composite laminates
- **Beam Models**: Support for both Euler-Bernoulli and Timoshenko beam theories
- **Boundary Conditions**: Advanced boundary condition handling
- **Material Properties**: Comprehensive material database support

## Installation

### Prerequisites

- Python 3.8 or higher
- MPI implementation (e.g., OpenMPI, MPICH)
- PETSc and SLEPc libraries

### From source

```bash
git clone https://github.com/opensg/opensg.git
cd opensg
conda env create --file environment.yml
```

## Quick Start

```python
import opensg
import numpy as np

# Load mesh data
mesh_data = opensg.load_yaml("blade_mesh.yaml")

# Create blade mesh object
blade_mesh = opensg.BladeMesh(mesh_data)

# Generate segment mesh
segment_mesh = blade_mesh.generate_segment_mesh(segment_index=1, filename="section.msh")

# Compute ABD matrices
ABD = segment_mesh.compute_ABD()

# Compute stiffness matrices
timo_stiffness, eb_stiffness, l_stiffness, r_stiffness = segment_mesh.compute_stiffness(ABD)
```

## Package Structure

```
opensg/
├── core/           # Core computation modules
│   ├── __init__.py
│   ├── shell.py    # Shell element computations and ABD matrices
│   └── solid.py    # Solid element computations
├── mesh/           # Mesh handling
│   ├── __init__.py
│   ├── blade.py    # BladeMesh class for wind turbine blade meshes
│   ├── segment.py  # SegmentMesh class for blade segments
│   ├── solidblade.py # SolidBladeMesh class
│   └── solidsegment.py # SolidSegmentMesh class
├── io/             # Input/output operations
│   ├── __init__.py
│   └── io.py       # File I/O functions (YAML, mesh files)
├── utils/          # Utility functions
│   ├── __init__.py
│   ├── shared.py   # Shared utilities (nullspace, solvers, constraints)
│   └── shell.py    # Shell-specific utilities
├── tests/          # Test suite
│   ├── __init__.py
│   ├── data/       # Test data files
│   ├── test_imports.py
│   └── test_workflow.py
├── opensg_solid/   # Solid analysis module
│   ├── __init__.py
│   ├── compute_utils.py
│   ├── io.py
│   ├── mesh.py
│   ├── solve.py
│   └── stress_recov.py
├── examples/       # Example scripts and data
│   ├── data/       # Example mesh and configuration files
│   ├── compute_eb_blade.py
│   └── ...
├── docs/           # Documentation
│   ├── _build/
│   ├── _templates/
│   ├── api-doc.rst
│   ├── conf.py
│   ├── index.rst
│   ├── user-guide/
│   └── ...
├── akshat_examples/ # Additional examples and legacy code
│   ├── Beam Model/
│   ├── Plate Model/
│   ├── Shell Model/
│   └── ...
├── __init__.py     # Main package interface
├── setup.py        # Package setup configuration
├── pyproject.toml  # Modern Python packaging configuration
├── requirements.txt # Python dependencies
└── environment.yml # Conda environment specification
```

## Documentation

For detailed documentation, please visit [docs.opensg.org](https://docs.opensg.org).

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

TODO

## Support

For support and questions, please:
- Check the [documentation](https://docs.opensg.org)
- Open an [issue](https://github.com/opensg/opensg/issues)
- Join our [discussion forum](https://github.com/opensg/opensg/discussions)
