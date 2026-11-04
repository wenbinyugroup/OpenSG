# Changelog

## 2.1.1 (unreleased)

### Solid segment route

- A 3-D solid beam segment with aperiodic ends gives its beam properties. A solid yaml with `n_model: 1` and
  `aperiodic: 1` on a 3-D mesh (mesh `z` = the beam axis) runs the new `opensg_solid/sg_segment.py`. It writes the
  Timoshenko 6x6 (or the 4x4 for `refined: 0`) and the mass matrix per unit length to `<base>.out`, and each end
  section's 6x6 to `<base>_ends.out`. The law is referred to the line `x = y = 0` of the mesh. Elements: hex8, tet4,
  mixed hex8 and tet4, and tet10. The solves use SuperLU.
- The algorithm is FEniCS OpenSG v1.2's tapered solid segment, through its JAX port in OpenSG-TW-claude
  (`opensg_jax/fe_jax/solid_taper.py`). The end warping is pinned by its area means, as in FEniCS v1.2 and the 2-D
  beam route (the port pins it to the nodal rigid modes). Tapered tubes match FEniCS v1.2 to 6e-14 at the same
  quadrature, and a prismatic segment matches OpenSG's 2-D beam route.
- A 2-D beam SG with `aperiodic: 1` is refused with a message that names the route. `opensg <segment.yaml> D` and the
  iterative solvers are refused too. See `docs/user_guide/solid_segment.md`.

### Fixed

- Solid beam dehomogenization (`n_model: 1`, `refined: 1`) under a shear force. The recovery
  is now the generalized-Timoshenko chain of the msg-shell route
  (`sg_dehom.compute_fluctuations_gpu`). The beam force seeds it directly, and the fixed
  operator `R1` gives `F' = R1 F`. The field is the zeroth-order (`V0`) part plus the
  first-order (`V1`) correction, so `F2` and `F3` recover (`int s12 = F2`, `int s13 = F3`)
  and the field is linear in the load. 2.1.0 seeded the chain with `C^-2 F` through a
  load-dependent operator, and recovered `1/GA^2` of the applied shear.
- Per-element frames on the beam D route are applied once (`sigma = C_ess Gamma`, SG axes).
  `--material` now rotates into each element's frame first, then by the ply angle. 2.1.0
  ignored the frames there.
- `--material` undoes the ply angle the stiffness was actually built with: the `angles=`
  argument of `plate_homo_2d` when given (2.1.0 read the material blocks only). A type-2
  block with no `angle` keeps its stored 6x6, and the command line names it in a note.

### Changed

- The per-element frame block is `orientation:` (PreVABS' word), the one name OpenSG writes.
  The shipped example yamls use it, and their laws are unchanged. `elementOrientations:` is
  still read, with a one-line note printed once per file in a run. A file or SG dict that
  carries both names is refused. OpenSG 2.1.0 and earlier ignore an `orientation:` block.
- `opensg <yaml> D` writes `<base>_dehom.SM/.EM` for `--material` (the default) and
  `<base>_dehom_global.SM/.EM` for `--global`. `export_gauss` without a frame writes the
  SG-axes names `<prefix>_global.SM/.EM`, where 2.1.0 wrote `<prefix>.SM/.EM`. A free-text
  `frame=` note is read by its first word, or by its words when they name one frame. A note
  that names neither frame, or both, is refused (pass `frame_kind=`). A file of the other
  frame's name next to the prefix is reported. The summary line lists the files written.
- The beam D route takes exactly one driver, `epsilon_bar` (the strain) or the new
  `beam_force=` (the force, the `1:` identifier of a `.ff`). The log prints the resultants the
  field integrates to (`sg_dehom.beam_expected_resultants`): the applied `[F2 F3]`, and
  `[F1 M1 M2 M3]` plus `Delta F_cl`, the classical excess of this recovery. VABS 4.1 has the
  same excess, and it depends on the section origin (see the recovery note of
  `docs/examples/solid_beam_2d.md`).
- A `refined: 0` beam runs the classical recovery (`V0`, the 4x4 law) and refuses `F2`, `F3`
  or a shear strain.
- The D route refuses frames that are not proper rotations: `det M <= 0`, or
  `max|M M^T - I|` above 1.7321e-6, the bound of rows written with 6 decimals. The
  homogenization law does not change.
- A refused input ends the command with its one-line reason (exit 1), not a traceback.
- API: `compute_fluctuations_gpu(Deff_srt, st, V0, V1, F0)` takes the force that seeds the
  chain (2.1.0: `(Deff_srt, st, V0, V1)`) and refuses an `st`, `F0` pair that is not one
  state. `recover_gauss_point_fields` lost its `elem_rotation` argument, and
  `get_Rsig_matrix` is gone. `sg_mesh.load_sg_input(dict)` returns a renamed copy and leaves
  the caller's dict as it was.

### Also in 2.1.1 (since 2.1.0)

- `msh_to_yaml` reads gmsh meshes as gmsh writes them, order-2 `tet10` included.
- The repository holds the OpenSG packages only. The tests moved out of `src`, and the
  documentation and packaging were rewritten.
- The documentation is rebuilt in the PreVABS structure (getting started, user guide,
  examples, theory, reference) on the pydata Sphinx theme.
- The OpenSG 2.x documentation is published at `/OpenSG_v2_0` by a Pages workflow.
- A CPU-only conda environment, vendored PETSc options, and a CI workflow that runs
  `opensg --help` and a smoke run of the 2-D SG plate example.
- CI runs the optional PETSc GAMG test apart from the suite, without failing the job.
- `CITATION.cff`, `CODEOWNERS` and `CONTRIBUTING.md`. The README names the authors.
- Test-side fixes: the 23 earlier failures of `tests/msg_k_reference` and four stale unit
  tests. The engine was not changed by them.
- The IEA 51-station sweep example reads its station folder from `OPENSG_IEA51_DIR`
  (default `data/iea_all_stations/shell51/1d_yaml`) instead of a hard-coded path.
