# CLAUDE.md — OpenSG-2.0, branch `rm_plate`

Guidance for Claude Code in this repository. This branch was created from
`keith/main` (`KeithBallard/fea-in-jax`, unrelated history to the old
`Akshat_msg_solid` branch) and ADDS the OpenSG-RM stack on top of Keith's
core without restructuring it.

## Layout

- `src/fe_jax/`, `src/jetsci/`, `benchmarks/`, `docs/`, `experiments/`,
  `tests/{unit,integration,regression}` — Keith's general JAX FE core.
  Do not restructure; upstream is `keith` (fetch to sync).
- `src/rm_plate/` — the 1-D laminate OpenSG-RM stack (from OpenSG-TW),
  KEPT SEPARATE per the user: `msg_rm_plate.py` (+ README, USER_GUIDE),
  `msg_materials.py`, `msg_transverse_shear.py`, `segment_plate.py`,
  `rm_homo.py`, `rm_dehom.py`, `cli_rm_plate.py`;
  `helper/` — format conversions shared by BOTH stacks: `sc_to_yaml.py`
  (SwiftComp .sc → yaml + gmsh .msh; run as
  `python -m rm_plate.helper.sc_to_yaml file.sc`), `abq2ff.py` (Abaqus
  .dat → .ff), `abaqus_inp.py`/`abaqus_3dfea.py`/`make_abaqus_dyn.py`
  (yaml → .inp decks);
  `tests/` — THE PRE-UPDATE GATE. Run before ANY homo/dehom change:
  `python -m pytest src/rm_plate/tests -q` (15 tests).
- `src/opensg_solid/` — the GENERAL structure-gene engine: ONE shared
  code for beam/plate/3-D-elastic macro models (`n_model` 1/2/3) from a
  1-D/2-D/3-D SG, split file-per-concern (Keith-style):
  `sg_materials.py` (C builders/rotations + the dedupe/convention notes
  vs `rm_plate_1D.msg_materials`), `sg_mesh.py` (`load_sg_input`,
  `_cell_basis`, `plot_sg_mesh`), `sg_assembly.py` (SSDM element
  kernels verbatim from `Dehom_plate_plots_SSDM.py`, periodic assembly,
  block-Jacobi/Chebyshev preconditioners, the Beam_solid energy-form
  assembly + KKT solver), `sg_homo.py` (`plate_homo_2d`; SSDM pipeline
  for n_model 2/3 — default `solver="direct"` pardiso factorization,
  `solver="cg"` = the verbatim Chebyshev-CG; the Beam_solid
  Timoshenko/KKT beam path for n_model 1 -> Timo 6x6 with the EB 4x4 in
  `r["C_eff_EB"]`), `sg_dehom.py` (`plate_dehom_2d`, `export_gauss`,
  the beam recovery chain).  Every sg_* module carries an ALL-VARIABLES
  docstring block.  The OneDrive `Claude_code/Beam_solid.py` original
  lives outside the repo.  Every homo run also writes `<base>_mesh.png`.
  Timings: `BENCHMARKS.md` at the repo root (RHC ~8-13 s per model).
- The repo is installed editable (`pip install -e . --no-deps`), so
  `import rm_plate / opensg_solid` works everywhere — example scripts
  carry NO sys.path boilerplate.
- `examples/OpenSG-solid/` — numbered, run in place, SSDM-style
  informative User-Input blocks (no argparse in the runners; input is
  the YAML, the .sc conversion is the separate helper step):
  1 layup → plate props (1-D SG); 2 FF field → 3-D stress (.ff route);
  3/4 plate homo/dehom from the 2-D honeycomb RHC SG;
  5/6 solid (n_model 3) homo/dehom — OPEN: the bundled SW_2UC_45.sc has
  9 nonzero connectivity slots/element (convention unconfirmed);
  7/8 beam (n_model 1, Timoshenko/KKT) homo/dehom.
  `CS_OpeNSG_exampels/static`, `dynamic_ex5` — the validated Pagano
  static suite and the Nayak transient benchmark (S8R vs C3D20).

## Load-bearing gotchas

- `src/rm_plate/__init__.py` sets `jax_enable_x64` — REQUIRED; without
  it every gate drifts at float32 eps (V2L faces ~1e-3 instead of 1e-11).
- `opensg_solid` needs the LEGACY fe_jax layout (basix basis/quadrature,
  `mesh_to_periodic_sparse_assembly_map` in `setup.py`) — the
  `$FE_JAX_CORE` sys.path bootstrap (default `~/OpenSG_2.0`, the
  OneDrive-"latest" clone) lives in `opensg_solid/__init__.py`, which
  also enables the persistent JAX compilation cache
  (`~/.cache/opensg_jax`) — import the engine THROUGH the package.
  Keith's `src/fe_jax` is NOT yet API-compatible for those pieces.
  Extra dep installed into the conda env for it: `jaxopt`.
- `plate_homo_2d(n_model=1)` routes through the merged Timoshenko/KKT
  beam engine (l-chain energies + rigid-body Lagrange constraints +
  optional `elem_rotation`; pypardiso, scipy-SuperLU fallback) and
  needs n_sg >= 2. The shared SSDM kernels still carry the EB 4-mode
  beam masks (reachable via `full_homogenization_pipeline` directly).
- `.sc` type-2 material blocks are PRE-ROTATED ply C's; the
  `mat_override.yaml` route rebuilds from engineering constants + angles
  instead (the documented working SSDM run).
- Output conventions: `<base>_plate_ABD.out` (6×6 [N11 N22 N12 M11 M22
  M12]) / `<base>_beam_Timo.out` (6×6 [eps11 gam12 gam13 kap1 kap2
  kap3]) / `<base>_solid_C.out` (6×6),
  `<base>_dehom.txt/.vtk` (Gauss cloud, SwiftComp order xx yy zz yz xz
  xy), `<base>_mesh.png` (elements colored by material, no title).

## Working rules (the user's standing preferences)

- Run the `src/rm_plate/tests` gate before/after any core change.
- Figures: never a title; render the REAL computed mesh, never a sketch.
- Commit in batches with explicit paths; verify with `git show --stat`;
  never push without being asked.
- Conda env `opensg_2_0` on msg.ecn.purdue.edu; heavy compute there.
