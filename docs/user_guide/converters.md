# Converting meshes and results

OpenSG runs one yaml per structure gene (SG). The converters write that yaml from the mesh files
of other tools (a SwiftComp `.sc`, a gmsh `.msh`, an Abaqus `.inp`) and, in the other direction,
write SwiftComp inputs from an OpenSG yaml and its `.ff` macro state, so the two codes can be run
on the same problem. Each converter is a sub-command of `opensg`. The yamls they emit carry
`msg: solid` in the header, so the engine choice stays in the file.

% source: src/opensg/cli.py
| sub-command | reads | writes |
|---|---|---|
| `opensg sc_to_yaml` | SwiftComp `.sc` | `<base>.yaml` and `<base>.msh` |
| `opensg msh_to_yaml` | gmsh 2.2 solid mesh | `<base>.yaml` |
| `opensg inp_to_yaml` | Abaqus `.inp` | `<base>.yaml` and `<base>.msh` |
| `opensg yaml_to_sc` | solid SG yaml | `<base>.sc` |
| `opensg ff_to_glb` | `.ff` macro state | `<base>.sc.glb` |
| `opensg_shell.helper.msh_to_yaml.convert` (Python) | gmsh surface mesh, format 2.2 or 4.1 | shell SG yaml |

Every sub-command expands a glob (PowerShell passes globs through literally), all but
`sc_to_yaml` also take several file names, and each returns exit code 0 when every file
converted, 1 when any file failed and 2 on a usage error.
`opensg <sub-command> --help` prints the flags.

## SwiftComp `.sc` to yaml: `sc_to_yaml`

% source: opensg sc_to_yaml --help (src/opensg/cli.py)
```text
usage: opensg sc_to_yaml [-h] --n_model {1,2,3} [--refine {0,1}] file.sc
```

% source: src/opensg/cli.py (sc_to_yaml); `opensg sc_to_yaml --help`
| flag | meaning | default |
|---|---|---|
| `file.sc` | the SwiftComp `.sc` to convert | required |
| `--n_model {1,2,3}` | the `n_model:` header key: 1 beam, 2 plate, 3 solid | required |
| `--refine {0,1}` (also `--refined`) | the `refined:` header key: 0 classical, 1 shear-refined | 0 |

The converter writes `<base>.yaml` (the canonical solid dialect of {doc}`materials_and_layups`)
and `<base>.msh`, a gmsh 2.2 mesh with the material id as physical and elementary tag. An
existing, up-to-date `<base>.yaml` is kept, because hand edits win: every other header key
(`omega:` and the rest) is added by hand after the conversion. Delete the yaml to re-emit it. An
explicit `--refine` that disagrees with the kept header re-emits it, and so does a `.sc` newer
than the yaml.

The reader expects this anatomy:

% source: docs/input_format.md (the SwiftComp .sc input); src/opensg_solid/io/sc_to_yaml.py
| block | content |
|---|---|
| model header | 1 to 3 short lines (submodel flags and similar) |
| size line | `dim  n_nodes  n_elems  n_mats` followed by further integers; the first line holding at least five integers |
| node records | `n_nodes` lines of `id  x [y [z]]` |
| element records | `n_elems` lines of `id  mat_id  conn`, the connectivity zero-padded to a fixed slot count; the padding zeros are not nodes |
| material blocks | per material a header `mat_id  mat_type  n`, an auxiliary line such as `0 0`, then the properties: type 0 is `E  nu`, type 1 is `E1 E2 E3 / G12 G13 G23 / v12 v13 v23`, type 2 the 21 upper-triangular constants of the $6\times6$ over six lines |
| trailing line | the SG measure $\omega$ (volume or thickness normalization), kept for the cross-check below |

The trailing value is not written as a header key, because OpenSG measures $\omega$ from the mesh,
and the two agree for a correct conversion: the `RHC_SW_2UC_45` run prints
`omega 35.248001` in its `.out` banner against the trailing `35.248` of the `.sc`, the check that
the mesh and the `.sc` header describe the same cell.
The one exception is a 3-D SG homogenized to a 3-D solid (`n_model: 3`): there the value is
written as `omega:` when it differs from the node bounding-box volume by more than $10^{-6}$
relative, and dropped otherwise, with a printed note when it disagrees with the measurement.

In `examples/OpenSG-solid/3_get_plate_props_from_2D_SG/RHC_SW_2UC_45.sc` the size line is
`2 4251 6640 3 0 0` (a 2-D SG with 4251 nodes, 6640 elements and 3 materials) and the file ends
with the lone line `35.248`. Its last element record shows the zero padding:

% source: examples/OpenSG-solid/3_get_plate_props_from_2D_SG/RHC_SW_2UC_45.sc
```text
6640 3 4251 2988 3079 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
```

The same conversion is available from Python, and `plate_homo_2d` performs it automatically when
it is handed a `.sc` path. The module has no `__main__` block, so import `convert` rather than
calling it with `python -m`:

% source: docs/input_format.md (the SwiftComp .sc input)
```python
from opensg_solid.io.sc_to_yaml import convert

sc = convert("RHC_SW_2UC_45.sc")
print(sc["dim"], len(sc["nodes"]), len(sc["cells"]))
```

## gmsh mesh to yaml: `msh_to_yaml`

% source: opensg msh_to_yaml --help
```text
usage: opensg msh_to_yaml [-h] [--out-base BASE] [--n_model {1,2,3}]
                          [--refined {0,1}] [--materials PATH]
                          [--h_refine [N]] [--p_refine] [--force]
                          msh [msh ...]
```

% source: src/opensg/cli.py (msh_to_yaml); `opensg msh_to_yaml --help`
| flag | meaning | default |
|---|---|---|
| `msh` | one or more `.msh` files | required |
| `--mat<K> NAME[:ANGLE]` | the material of gmsh physical tag $K$, by library name, with an optional ply angle in degrees about $y_3$ | `FILL_IN` placeholder |
| `--n_model {1,2,3}` | the `n_model:` header key; never guessed from the mesh | `FILL_IN` placeholder |
| `--refined {0,1}` | the `refined:` header key | the engine default |
| `--materials PATH` | the material library | `./materials.yaml`, then one next to the `.msh`, then the packaged `opensg_solid/io/materials.yaml` |
| `--h_refine [N]` | uniform conforming subdivision before any `--p_refine`, halving $h$ per level (tri3 and quad4 into 4, tet4 and hex8 into 8); the bare flag is one level, and the level is written into the output name (`mesh_h2`, `_h4`, `_h8`) | off |
| `--p_refine` | elevate a linear tet, tri or quad mesh to its conforming quadratic twin `<stem>_quad.msh` (tet10, tri6, quad9) and convert that; an already quadratic mesh is refused | off |
| `--out-base BASE` | basename of the emitted yaml (single input only) | the `.msh` stem |
| `--force` | re-emit even when `<base>.yaml` is newer than the `.msh` | off |

The mesh side (`nodes`, 1-based `elements`, `elementOrientations`, one `sets: element:` entry
per physical tag named from `$PhysicalNames`) is always complete. A `.msh` carries no material and
no orientation: every element gets the constant frame with $e_1$ out of plane, and the materials
come from the `--mat<K>` flags (see {doc}`materials_and_layups`). Tags without a flag, and
`n_model` without `--n_model`, stay `FILL_IN` placeholders, and `opensg` refuses to run the yaml
until they are filled. A material or header flag is an explicit request, so it always re-emits;
otherwise an up-to-date yaml is skipped unless `--force`.
The yaml header records the element grade as the informational key `mesh_order:`.

% source: docs/input_format.md (opensg msh_to_yaml)
```bash
opensg msh_to_yaml cell.msh --mat1 Al69 --mat2 epoxy --n_model 3
```

- **Format.** Legacy ASCII 2.2 only. gmsh's default msh 4.1 and binary files are refused with the
  fix `gmsh cell.msh -format msh22 -o cell_msh22.msh -save` (API: `Mesh.MshFileVersion 2.2`,
  `Mesh.Binary 0`).
- **Cells.** The cells of the highest dimension form the SG and must be one accepted type: tri3
  or tri6, quad4 or quad9, tet4 or tet10, or hex8 (gmsh types 2/9, 3/10, 4/11, 5). Two cell types
  of that dimension, or an unsupported one (prism, pyramid, quad8, hex20 or hex27, tet20), is an
  error. Every lower-dimensional cell is dropped and counted. 2-D cells that span $z$ are refused
  (a 3-D mesh saved with Physical Surfaces only, or a shell surface, which belongs to the shell
  helper below), and so are cells written twice.
- **Nodes.** Node ids are honoured as ids, in any row order and with gaps; a cell naming a missing
  id is an error. Unused nodes are compacted away, and coordinates are written at full precision.
- **Tags.** Sets follow the physical tag. Without physical tags (or with `Mesh.SaveAll 1`) the sets
  come from the elementary tags; `--mat0` alone still fills the whole mesh as one set.

The console line reports what was dropped, compacted or fallen back, for example:

% source: docs/input_format.md
```text
cell.msh -> cell.yaml   (tet10; 35782 nodes / 107346 dofs; dropped 12090 tri6 + 836 line3 + 92 point lower-dim cells; no physical tags: sets from elementary tags [1])
```

:::{tip}
A mesh can go straight to the analysis: `opensg core.msh --mat1 Al --n_model 2` converts the
mesh with the conversion flags, then runs the emitted yaml with the remaining flags. `--refined 0`
is added when the flag is absent, and with `--p_refine` the analysed file is `<stem>_quad.yaml`.
`--h_refine` is refused on this path; run `opensg msh_to_yaml` first.
:::

## Abaqus deck to yaml: `inp_to_yaml`

% source: opensg inp_to_yaml --help
```text
usage: opensg inp_to_yaml [-h] --n_model {1,2,3} [--refined] [--material NAME]
                          [--map ELSET=MAT [ELSET=MAT ...]] [--out-base BASE]
                          [--force]
                          inp [inp ...]
```

% source: src/opensg/cli.py (inp_to_yaml); `opensg inp_to_yaml --help`
| flag | meaning | default |
|---|---|---|
| `inp` | one or more `.inp` files | required |
| `--n_model {1,2,3}` | the `n_model:` header key; never guessed from the mesh | required |
| `--refined` | write `refined: 1` (plate ABDG, beam Timoshenko) | `refined: 0` |
| `--material NAME` | one deck material for every element | none |
| `--map ELSET=MAT [...]` | explicit element-set to material pairs, winning over the deck's section cards; `MAT@DEGREES` makes plies that share a deck material but differ in orientation separate SG materials with `angle:`, e.g. `--map P1=Aluminum L1=Laminate@45 L2=Laminate@-45` | none |
| `--out-base BASE` | basename of the emitted files (single input only) | the `.inp` stem |
| `--force` | re-emit even when the outputs are newer and already carry the requested header | off |

The deck carries its own materials, so the yaml comes out runnable. The converter reads `*Node`,
every `*Element` block, `*Elset` (including `generate`), `*Solid Section` and `*Shell Section`,
and `*Material` with `*Elastic` (isotropic or engineering constants) and `*Density`; steps, loads
and output requests are ignored. Without section cards, `--material` or `--map`, all deck
materials are written, every element defaults to material 1 with a printed note, and the binding
is your edit of `mat_id:` and `materials:` in the yaml.

% source: src/opensg_solid/io/inp_to_yaml.py
| element | Abaqus types |
|---|---|
| tri3 | S3, S3R, STRI3, CPS3, CPE3, CPS3T, CPE3T |
| quad4 | S4, S4R, S4RS, CPS4, CPS4R, CPE4, CPE4R, CPE4I |
| tet4 | C3D4 |
| hex8 | C3D8, C3D8R, C3D8I |
| tet10 | C3D10 (the last two midside nodes are swapped to the gmsh order) |

The SG occupies the deck columns that actually vary, written into the leading yaml columns: a shell
mesh drawn in the $x$-$y$ plane becomes a 2-D SG. A mixed mesh (for example S4R with S3) comes out
as ragged cells. From Python:

% source: src/opensg_solid/io/inp_to_yaml.py (module docstring)
```python
from opensg_solid.io.inp_to_yaml import convert
convert("HC_2UC_SW.inp", n_model=2, refined=1,
        elset_map={"P1": "Aluminum", "L1": "Laminate@45",
                   "L2": "Laminate@-45"})
```

## Yaml to SwiftComp `.sc`: `yaml_to_sc`

% source: opensg yaml_to_sc --help
```text
usage: opensg yaml_to_sc [-h] [--out-base BASE] [--orientation {bake,ignore}]
                         [--drop-density]
                         yaml [yaml ...]
```

% source: src/opensg/cli.py (yaml_to_sc); `opensg yaml_to_sc --help`
| flag | meaning | default |
|---|---|---|
| `yaml` | one or more SG yamls | required |
| `--out-base BASE` | basename of the emitted `.sc` (single input only) | the yaml stem |
| `--orientation {bake,ignore}` | per-element frames, which the `.sc` has no slot for: `bake` folds each distinct frame into its own pre-rotated type 2 material (exact), `ignore` drops them (correct only when they are all the identity) | needed only for a yaml with such frames |
| `--drop-density` | write the `0 0` auxiliary line for a material with nonzero density instead of refusing | off |

There are no model flags: the yaml header decides. `n_model:` must be 2 (plate) or 3 (solid); the
beam is refused. `refined:` sets the submodel line, and the SG dimension is measured from the node
columns that vary. Run the result as `SwiftComp <base>.sc 2D H` (`n_model: 2`) or
`SwiftComp <base>.sc 3D H` (`n_model: 3`). A material `angle:` goes into the SwiftComp layer
table, so the material blocks keep their engineering constants; frames that are all the identity
(the `msh_to_yaml` default) are dropped without a flag. Every call rewrites the `.sc`.

:::{note}
For a plate, a header `refined: 1` is written as the classical submodel line: the SwiftComp
plate deck that this writer emits is the classical one.
:::

## Macro state to SwiftComp `.glb`: `ff_to_glb`

% source: opensg ff_to_glb --help
```text
usage: opensg ff_to_glb [-h] [--yaml YAML] [--n_model {2,3}] [--refined {0,1}]
                        [--out PATH] [--id1 {auto,0,1}] [--sc PATH]
                        [--no-check-sc]
                        ff [ff ...]
```

% source: src/opensg/cli.py (ff_to_glb); `opensg ff_to_glb --help`
| flag | meaning | default |
|---|---|---|
| `ff` | one or more `.ff` macro state files ({doc}`recovery`) | required |
| `--yaml YAML` | the SG yaml whose header shapes the `.glb` (single input only) | `<ff stem>.yaml` beside the `.ff` |
| `--n_model {2,3}` | explicit macro model: 2 plate, 3 solid; the beam is refused | from the yaml |
| `--refined {0,1}` | explicit submodel: 0 classical (6 resultants), 1 shear-refined (8) | from the yaml |
| `--out PATH` | the `.glb` to write (single input only) | `<ff stem>.sc.glb` |
| `--id1 {auto,0,1}` | `1` writes the `0:` generalized strain, `0` the `1:`/`FF` generalized stresses, which SwiftComp inverts through its own law; `auto` is `1` when the `.ff` has a `0:` line | `auto` |
| `--sc PATH` | the `.sc` whose submodel line the `.glb` must agree with (single input only) | the `.glb` path minus `.glb` |
| `--no-check-sc` | downgrade that check to a warning | off |

The `.glb` is the SwiftComp elastic dehomogenization input:

% source: src/opensg_solid/io/ff_to_glb.py
```text
v1 v2 v3            macro displacements
C11 C12 C13         macro DIRECTION COSINES, Bi = Cij bj
C21 C22 C23
C31 C32 C33
id1                 0 = the next line is generalized STRESSES,
                    1 = generalized strains
eps_bar | sig_bar   Kirchhoff-Love plate: e11 e22 2e12 k11 k22 2k12
                                        | N11 N22 N12 M11 M22 M12
                    Reissner-Mindlin:      ... + g13 g23 | N13 N23
                    3-D Cauchy:           e11 e22 e33 2e23 2e13 2e12
                                        | s11 s22 s33 s23 s13 s12
<blank line>        (the manual asks both inputs to end blank)
```

The direction cosines follow the SwiftComp manual's definition, which is the transpose of the
rotation OpenSG's own recovery applies and carries the macro strain on its diagonal. The
resultant count comes from the yaml, or the flags; with neither, the classical plate (6) is
assumed and printed. A header `refined: 1` under `n_model: 2` is sized for the classical plate,
to match the deck `yaml_to_sc` writes; `--refined 1` forces 8. For a shear-refined plate the
transverse pair comes from the `.ff` key `Q:`, or is written as `0 0` with a note. Every call
rewrites the `.glb`; then run `SwiftComp <base>.sc 2D L` (or `3D L`).

:::{warning}
The count must match the submodel line of the `.sc`. A mismatch is silent in SwiftComp: it
over-reads the `.glb` and leaves every local-field output at zero bytes. `ff_to_glb` therefore
refuses to write a `.glb` that disagrees with the sibling `.sc`.
:::

## Shell surface meshes: `opensg_shell.helper.msh_to_yaml`

A surface mesh out of gmsh knows the geometry and nothing else. The shell helper writes the mesh
side of the shell dialect from an ASCII `.msh` (format 2.2 or 4.1) and refuses to invent the rest:

% source: docs/input_format.md (msh_to_yaml, shell)
```python
from opensg_shell.helper.msh_to_yaml import convert

convert("schwarz_p_D2_shell.msh")                      # FILL_IN template
convert("schwarz_p_D2_shell.msh", thickness=0.036457)  # runnable, one alu ply
```

% source: docs/input_format.md (msh_to_yaml, shell); src/opensg_shell/helper/msh_to_yaml.py
| call | result |
|---|---|
| `convert(path)` | the mesh blocks complete, `sections:` and `materials:` written as a marked `FILL_IN` template that `opensg_shell` refuses to run and whose missing fields it names |
| `convert(path, thickness=t)` | a runnable single-ply yaml; the default wall material is aluminium, $E = 69$ GPa, $\nu = 0.30$, $\rho = 2700$ kg/m³; pass `material={...}` for your own |
| `convert(path, layup=[[mat, t, angle], ...], materials=[...])` | the general multi-ply form |

Each facet gets $e_3$ = the facet normal, $e_2$ = its first edge and $e_1 = e_2 \times e_3$, and one
`layup_<k>` element set is written per gmsh physical tag. Only 3-node triangles (gmsh type 2) and
4-node quads (type 3) are accepted; the 0-D and 1-D entities that gmsh writes for physical points
and curves are skipped, and any other element type is an error. The `sections:` and `materials:`
blocks are written first, above the mesh, because they are the part you edit.

The blade sub-commands `gen_windio_cs`, `windio_st` and `pynumad` are listed in
{doc}`../getting_started/running` and walked through in {doc}`../examples/blade_windio_pipeline`
and {doc}`../examples/blade_iea_beam`.

## See also

- {doc}`materials_and_layups` for the material blocks the converters write.
- {doc}`mesh_and_elements` for the yaml dialects themselves.
- {doc}`output_files` for the files a run writes from the converted yaml.
