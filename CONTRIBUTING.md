# Contributing to OpenSG

Maintainer: Akshat Bagla (@bagla0)

OpenSG 2.x, the JAX line, is developed on `main`. OpenSG 1.0, the FEniCSx line, is frozen on the
branch `opensg/fenicsx/v1` (tag `v1.0.0`) and takes no changes.

## Set up

```bash
git clone https://github.com/wenbinyugroup/OpenSG.git
cd OpenSG
conda env create -f environment.yml
conda activate opensg
pytest tests
```

`environment.yml` installs the package in editable mode, so edits under `src/` are live.

## Where things are

- `src/opensg/`: the `opensg` command line, which dispatches a yaml to its engine, and the Blade entry point
- `src/opensg_shell/`: SGs of shell elements (cross-section contours and surfaces): beam and solid models, dehomogenization, the blade workflow
- `src/opensg_solid/`: SGs of solid elements in 1-D, 2-D and 3-D: plate, beam and solid models, dehomogenization, format conversions
- `src/fe_jax/`: the finite-element core (basis and quadrature, assembly maps, solvers)
- `tests/`: the pytest suite; `docs/`: Sphinx sources, tutorials and rules; `examples/`: runnable examples with their inputs

## Workflow

1. Open an issue first for anything larger than a typo.
2. Branch from `main`.
3. Add or update a test for every change of behaviour. `pytest tests` must stay green.
4. Keep generated run output (`.out`, `.SM`, `.EM`, `.U`, `.vtk`, figures) and large meshes out of git.
5. Open a pull request against `main`, one topic per pull request, linking the issue.

## Reporting a result that disagrees with SwiftComp, VABS or a 3-D finite-element solution

Attach the SG yaml, the command, the two results and the version of the reference tool.
A verification discrepancy is treated as a bug.

## License

Contributions are licensed under the Apache License 2.0 (see `LICENSE`).
