# OpenSG: Open Source Structural Analysis for Wind Turbine Blades

OpenSG is an open source platform for multiscale structural analysis using FEniCS backend which works on the Mechanics of Structure Genome (MSG) 
theory. OpenSG has capabilties to obtain the structural homogenized properties in terms of either of Cauchy continuum (3D) model, plate/shell model or beam model. The obtained homogenized structural properties are accurate, fast and precise for complex structural profile like wind blades. 

OpenSG evaluates the homogenization using the mathematical building block of structure, called Structure Gene (SG), which can be of one, two or three dimension. It has the capability to generate local stress and displacement field (dehomogenization) of the structure based on user defined input load. OpenSG gives the research community to leverage the advantages of MSG theory for generating complex models based on the specific applications. 

## Features

TODO

## Installation

```bash
git clone https://github.com/wenbinyugroup/OpenSG.git
cd OpenSG
conda env create --file environment.yml
conda activate opensg_env
```

## Quick Start

```python
from pathlib import Path
import opensg
import numpy as np
from opensg.mesh.segment import ShellSegmentMesh

# 1) Load blade mesh data and generate segment files
blade_mesh_file = Path("examples", "data", "bar_urc_shell_mesh.yaml")

# Generate all segment mesh files
opensg.io.generate_segment_shell_mesh_files(
    blade_mesh_file, 
    segment_folder="segments/"
)

# 2) Compute stiffness matrices for each blade segment
segment_stiffness_matrices = []
for i in range(3):  # Process first 3 segments
    segment_file = Path("segments", f"{blade_mesh_file.stem}_segment_{i+1}.yaml")
    
    # Create segment mesh object
    segment_mesh = ShellSegmentMesh(segment_file)
    
    # Compute ABD matrices for the segment
    ABD = segment_mesh.compute_ABD()

    # Compute stiffness matrices using both beam theories
    timo_segment_stiffness, eb_segment_stiffness, l_timo_stiffness, r_timo_stiffness = segment_mesh.compute_stiffness(ABD)

    # Store results
    segment_stiffness_matrices.append(timo_segment_stiffness)

# 3) Combine and save results
combined_stiffness_matrices = np.concat(segment_stiffness_matrices)
np.savetxt('stiffness_matrices.txt', combined_stiffness_matrices, fmt='%d')
```

## Package Structure

```
opensg/
├── core/           # Core computation modules
│   ├── __init__.py
│   ├── shell.py    # Shell element computations and ABD matrices
│   ├── solid.py    # Solid element computations
│   └── stress_recov.py # Stress recovery utilities
├── mesh/           # Mesh handling
│   ├── __init__.py
│   ├── blade.py    # Deprecated classes for mesh management.
│   └── segment.py  # ShellSegmentMesh and SolidSegmentMesh classes for mesh management
├── io/             # Input/output operations
│   ├── __init__.py
│   ├── io.py       # File I/O functions (YAML, mesh files)
│   └── util.py     # I/O utility functions
├── utils/          # Utility functions
│   ├── __init__.py
│   ├── eigenvalue_solver.py # Eigenvalue solver utilities
│   ├── shared.py   # Shared utilities (nullspace, solvers, constraints)
│   ├── shell.py    # Shell-specific utilities
│   └── solid.py    # Solid-specific utilities
├── tests/          # Test suite
```

## Documentation

For detailed documentation, please visit [wenbinyugroup.github.io/OpenSG/](https://wenbinyugroup.github.io/OpenSG/).

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the BSD License - see the [LICENSE](LICENSE) file for details.

## Citation

TBD

## Support

For support and questions, please:
- Check the [documentation](https://wenbinyugroup.github.io/OpenSG/)
- Open an [issue](https://github.com/wenbinyugroup/OpenSG/issues)
- Join our [discussion forum](https://github.com/wenbinyugroup/OpenSG/discussions)
