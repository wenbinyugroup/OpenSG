# Input Formats and Conventions

OpenSG-2.0 reads three kinds of structure-gene (SG) description, plus one laminate
shorthand and one legacy converter:

| dialect | package that reads it | example file |
|---|---|---|
| 1-D shell SG (cross-section contour) | `opensg_shell` | `examples/OpenSG_shell/1_get_beam_props_from_shell_cross_section/iea_s10_shell.yaml` |
| 3-D shell SG (surface in space) | `opensg_shell` | `examples/OpenSG_shell/4_get_solid_props_from_shell_3D_SG/schwarz_p_3Dshell.yaml` |
| solid SG (1-D / 2-D / 3-D mesh) | `opensg_solid` | `examples/OpenSG-solid/3_get_plate_props_from_2D_SG/RHC_SW_2UC_45.yaml` |
| SwiftComp `.sc` | converted to the solid SG YAML | `examples/OpenSG-solid/3_get_plate_props_from_2D_SG/RHC_SW_2UC_45.sc` |
| through-thickness layup database | `opensg_solid.rm_plate_1D` | `examples/OpenSG-solid/1_get_plate_props_from_1DSG/layup_db.yaml` |

The two shell dialects share one schema — the same six top-level keys, differing only in
element dimension — while the solid dialect is a separate, deliberately minimal schema whose
material blocks mirror SwiftComp's.

Every run, in every dialect, writes a timed `.out` in the SwiftComp `.K` layout: an ` OpenSG`
banner line naming the model, `The Effective <Name> Stiffness Matrix`, `The Effective <Name>
Compliance Matrix`, and a `Time taken` footer in seconds. `<Name>` is **always the macro law's
console title**, so the line in the file and the line the terminal printed are word for word
the same:

| macro law | `<Name>` |
|---|---|
| equivalent 3-D solid $6\times6$ | `Cauchy Continuum` |
| beam $6\times6$ (shear-refined) | `Timoshenko Beam` |
| beam $4\times4$ (classical) | `Euler-Bernoulli Beam` |
| plate ABD $6\times6$ (classical) | `Classical Plate` |
| plate ABDG $8\times8$ (shear-refined) | `Reissner-Mindlin` |

Only the 3-D solid law also prints `The Engineering Constants (Approximated as Orthotropic)`.
The writer is `opensg_solid.sg_homo.write_sc_K`; the contract is
`Rules/output_format_and_timing.md`. The per-section **wall** plate laws that a shell run also
emits as `<base>_ABDG.out` come from a different writer,
`opensg_shell.sg_homo.write_abdg_out`, and keep the title
`The Effective Reissner-Mindlin Plate Stiffness Matrix` — one block per section.

## The YAML header — the analysis request

Above the first mesh block every SG YAML may carry a short block of scalar keys. This is the
**analysis request the file makes about itself**, read by `sg_mesh.read_yaml_header` without
parsing the mesh. Every key is defaulted, so a headerless mesh runs (as a classical plate
homogenization on the solid engine, a classical beam on the shell engine).

| key | values | meaning |
|---|---|---|
| `msg` | `shell` \| `solid` | which **engine** owns the file. Omit it and the mesh dialect decides, so older files keep working; the unified `opensg` command dispatches on exactly this key. A `msg:` that contradicts the dialect is an error, not a silent override. |
| `n_model` | `1` \| `2` \| `3` | the macro model: 1 beam, 2 plate, 3 equivalent 3-D solid. `n_model: 2` is the `opensg_solid` route — `opensg_shell` has no plate macro model, because there the **wall** is the plate. |
| `refined` | `0` \| `1` | `0` classical (plate ABD $6\times6$ / beam Euler–Bernoulli $4\times4$), `1` shear-refined (plate ABDG $8\times8$ / beam Timoshenko $6\times6$). Ignored for the solid macro model. |
| `analysis` | `H` \| `D` | homogenization or dehomogenization, when you would rather carry the switch in the file than on the command line. The command-line argument wins. |
| `epsilon_bar` | 6 floats | the macro state a `D` run recovers from: `[e11 e22 2e12 k11 k22 2k12]` (plate/solid) or `[ext sh2 sh3 twist bend2 bend3]` (beam). A `<base>.ff` file next to the YAML supersedes it. |
| `omega` | float | **optional** user SG measure, overriding the one measured from the mesh. Set it only when the equivalent continuum occupies something other than the measured cell — e.g. the wall *material* area of a closed tube. |
| `aperiodic` | `1` | **optional**, and only for an SG that genuinely is not periodic: zero fluctuation on every bounding-box-face node instead of the periodic tie. Periodic is the default at every SG dimension — omit the key entirely for a periodic SG. |
| `junction` | `off` \| `flag` \| `exclude` | **optional, msg-shell `D` only** — the junction-aware recovery tier; see the shell dehomogenization tutorial. Default `flag`. |
| `junction_bl` | float | optional, shell `D` only — the junction boundary-layer radius in units of the thickest wall there. Default `1.0`. |
| `junction_ang` | float, deg | optional, shell `D` only — the tangent-grouping tolerance of the junction detector. Default `1.0`. |

**There is no `dim:` key and no `scale:` key.** The SG dimension is read from the mesh — the
element node count, with the ambiguous solid counts broken by the leading node coordinates
that actually vary — and the SG measure $\omega$ is measured from the mesh (or declared with
`omega:`). Neither is something you write into a file.

## The 1-D shell SG YAML

This is the cross-section of a thin-walled beam: the wall **midline** (or mold line) meshed
with 2-node line segments, each segment carrying a laminate. It is the input to
`opensg_shell.build_rm_bundle`, which returns the Timoshenko $6\times6$.

`iea_s10_shell.yaml` is station $r/R = 0.2$ of the IEA-22 blade: 307 nodes, 310 line
elements, 6 layups, 6 materials. Its top-level keys, in the order OpenSG's own writers emit
them, are `nodes`, `elements`, `sets`, `sections`, `elementOrientations`, `materials`,
`reference`.

### `nodes`

One row per node, three coordinates. In the shell dialect the row is written as a **bracketed,
space-separated** triple (a single YAML scalar inside a flow sequence), not a comma list:

```yaml
nodes:
- [4.70852238 0.09722431 0.00000000]
- [4.64563825 0.11238283 0.00000000]
```

The first two components are the cross-section coordinates $(y_2, y_3)$ and the **third is the
beam-axis coordinate**, which is identically zero for a plane cross-section
(`sg_mesh.load_ring` sets `ax = 2`, `cross = [0, 1]`). The space-separated form is required,
not merely conventional: the orientation plotter `fe_jax/orient_plot.py` parses each row with
`str(row[0]).split()`, so a comma-separated list of floats will not plot.

### `elements`

One row per element, the node ids of a 2-node line segment, **1-based**:

```yaml
elements:
- [1 2]
- [2 3]
```

The loader detects the base (`if cells.min() == 1: cells = cells - 1`), so a 0-based mesh is
also accepted, but every generator in the repository writes 1-based. Element $k$ in this list
is element label $k$ everywhere else in the file, and row $k$ of `elementOrientations`
belongs to it.

### `sets` and `sections`

`sets` groups element labels by name; `sections` gives each named group its laminate. The tie
between them is the `elementSet` field, which must equal a set `name`:

```yaml
sets:
  element:
  - name: layup_0
    labels:
    - 1
    - 2
    - 3
sections:
- type: shell
  elementSet: layup_0
  layup:
  - - gelcoat
    - 0.0005
    - 0.0
  - - glass_triax
    - 0.00300294
    - 0.0
  - - glass_uniax
    - 0.02609467
    - 0.0
  - - glass_triax
    - 0.00300294
    - 0.0
```

Both blocks are shown abridged: in `iea_s10_shell.yaml`, `layup_0` lists 25 labels (elements
1–12 and 209–221) and there are six sets and six sections in all.

Labels are 1-based element indices into `elements`. `_material_by_section` builds one ABD per
entry of `sections` and keys it by the section's **position in the list**, and the loader then
maps every element through `rsub[label - 1] = section_index`; an element that appears in no
set silently falls to section 0, so every element should appear in exactly one set. `type:
shell` is descriptive metadata — the shell readers key only off `elementSet` and `layup`.

Each `layup` row is `[material, thickness, angle_deg]`: a material name that must exist in
`materials`, a ply thickness in metres, and a ply angle in degrees. The list is ordered
**outer face first** and stacks inward along $e_3$; `compute_ABD_matrix` documents its input as
"layer thicknesses, bottom to top" with the reference surface at that bottom (outer) face, and
`emit_station_abd` records the same order as `plies OML->IML`. The sum of the row thicknesses
is the wall thickness used for the reference shift.

### `elementOrientations`

One row of nine direction cosines per element, `[e1(3) e2(3) e3(3)]`, in the same component
order as the node rows. Unlike `nodes` and `elements`, these rows are true YAML lists of
floats:

```yaml
elementOrientations:
- [0.0, 0.0, 1.0, -0.9721541292349604, 0.2343423756204075, 0.0, -0.2343423756204075,
  -0.9721541292349604, 0.0]
```

The meaning of the triad is covered under "Material orientation" at the end of this page.

### `materials`

A list of named materials, each with a `density` and an `elastic` block holding three-component
`E`, `G` and `nu`:

```yaml
materials:
- name: glass_triax
  density: 1940.0
  elastic:
    E:
    - 28211400000.0
    - 16238800000.0
    - 15835500000.0
    G:
    - 8248220000.0
    - 3491240000.0
    - 3491240000.0
    nu:
    - 0.497511
    - 0.18091
    - 0.27481
```

The triples are $[E_1, E_2, E_3]$, $[G_{12}, G_{13}, G_{23}]$ and $[\nu_{12}, \nu_{13},
\nu_{23}]$ — the argument order of `build_stiffness_6x6`, which inverts the orthotropic
compliance to a $6\times6$. Units follow the input; the examples are SI, so moduli are in Pa,
thicknesses in m, `density` in kg/m$^3$. `density` feeds the section mass per unit area that
`emit_station_abd` reports as `mass_per_area`; it is not needed for the stiffness.

### `reference`

```yaml
reference: center
```

This single scalar names the surface the wall ABD — and therefore the whole cross-section
model — is referenced to, and it is the single source of truth: `build_rm_bundle(shell_yaml)`
reads it when its `ref` argument is left `None`, and the same value then drives the ring
laminate reference, the plate-SG `z_ref`, the emitted ABD file, and the depth conversion used
in stress recovery. Absent, it defaults to `center`.

| value | meaning | fraction of thickness from the outer face |
|---|---|---|
| `center` | the contour is the laminate **mid-surface**; the ABD is parallel-axis shifted by $t/2$ | 0.5 |
| `oml` | the contour is the **outer mold line**; the laminate stacks inward from it, no shift | 0.0 |
| `oml_flip` | diagnostic: the reference is moved the full thickness the other way | 1.0 |
| `iml` | the contour is the **inner mold line**; the laminate stacks outward | 1.0 |

Mid-surface meshes (tubes, ellipses, generated cross-sections) use `center`; airfoil YAMLs
written on the OML use `oml`. Getting this wrong shifts $B$ and $D$ by a parallel-axis term of
the wall thickness and moves every bending and torsion entry of the beam $6\times6$, so it is
recorded in the file rather than passed at the call site.

### Running it

```bash
opensg iea_s10_shell.yaml
```

The header of `iea_s10_shell.yaml` carries `msg: shell`, `n_model: 1` and `refined: 1`, so the
run is the Reissner–Mindlin ring and the result is the Timoshenko $6\times6$, written to
`iea_s10_shell_Timo.out` alongside the per-section wall laws in `iea_s10_shell_ABDG.out`.

The example folder also ships a driver that does the same homogenization through the API and
adds the two figures the CLI does not emit:

```bash
python beam_homo_shell.py
```

The driver in `examples/OpenSG_shell/1_get_beam_props_from_shell_cross_section/` sets
`name = "iea_s10"`, reads `iea_s10_shell.yaml`, and reports four files: `iea_s10_shell_Timo.out`
(the Timoshenko $6\times6$ and its compliance), `iea_s10_shell_ABDG.out` (one $8\times8$ RM wall
law per section, rows `[eps11 eps22 2eps12 | K11 K22 K12+K21 | 2g13 2g23]`), `iea_s10_mesh.png`
(the contour coloured by layup) and `iea_s10_orient.png` (the orientation check). On the first
run it additionally caches the per-station wall laws as `abd/iea_s10_shell_abd.yaml`, at the
same reference the YAML declared, for reuse by dehomogenization and shell buckling.

## The 3-D shell SG YAML

Same six keys, but the mesh is a **surface embedded in 3-D**: `nodes` are genuine
three-dimensional points and `elements` are 3-noded triangles or 4-noded quads. There is no
`reference` key — `shell_sg3d` references every wall law to its mid-surface. This is the input
to `opensg_shell.sg_homo.shell_sg3d`, which returns the equivalent 3-D solid $6\times6$; its
header carries `msg: shell` and `n_model: 3`, so `opensg <yaml>` reaches it directly.

`schwarz_p_3Dshell.yaml` is a Schwarz-P TPMS cell: 13536 nodes and 26360 triangles, one
aluminium layup. Triangles are read and padded to the quad connectivity by repeating the last
node (`el = np.array([e + [e[-1]]*(4 - len(e)) for e in el], int) - 1`), so tri and quad meshes
go through one element path; ids are 1-based as in the 1-D dialect.

![Schwarz-P TPMS shell SG mesh](_static/schwarz_p_mesh.png)

`make_schwarz_yaml.py` in the same folder is the canonical way to write one. Its emission loop
shows every row format at once — the frame is built per triangle from the geometry
($e_3$ = unit facet normal, $e_2$ = the unit $p_1 \to p_2$ edge, $e_1 = e_2 \times e_3$):

```python
o = ["nodes:"]
o += ["- [%.12f %.12f %.12f]" % tuple(p) for p in nd]
o.append("elements:")
o += ["- [%d %d %d]" % tuple(e) for e in el]
o.append("elementOrientations:")
for k in range(ne):
    e1 = np.cross(e2[k], e3[k])
    o.append("- [%.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f]"
             % (*e1, *e2[k], *e3[k]))
o += ["sections:", "- type: shell", "  elementSet: layup_0", "  layup:",
      "  - - alu", "    - %.6e" % T, "    - 0.0",
      "materials:", "- name: alu", "  density: 2700.0", "  elastic:",
      "    E: [%.6e, %.6e, %.6e]" % (E, E, E),
      "    G: [%.6e, %.6e, %.6e]" % (G, G, G),
      "    nu: [%.6f, %.6f, %.6f]" % (nu, nu, nu),
      "sets:", "  element:", "  - name: layup_0", "    labels:"]
o += ["    - %d" % (e+1) for e in range(ne)]
open("schwarz_p_3Dshell.yaml", "w").write("\n".join(o) + "\n")
```

Note the asymmetry that the loop makes explicit and that trips up hand-written files: `nodes`
and `elements` rows are space-separated inside brackets, `elementOrientations` rows are
comma-separated, and the `sets` labels are 1-based. The generator prints the surface area and
the relative density `area * T` so the shell idealisation can be checked against the solid cell
it replaces.

```bash
python make_schwarz_yaml.py
```

```bash
opensg schwarz_p_3Dshell.yaml
```

The second command writes `schwarz_p_3Dshell_C3D.out`, whose banner records the mesh size, the
element the facets were assembled with, the junction-edge count and the boundary treatment —
for this cell,
`13536 nodes, 26360 elems [26360 tri3/MITC3], 0 junction edges, periodic in 3 dirs, per unit cell 1`.
It also
writes `schwarz_p_3Dshell_ABDG.out`, the **step-1 wall plate law** the surface carries: an
$8\times8$ Reissner–Mindlin ABDG stiffness and its compliance, one block per section, headed

```text
 OpenSG msg-shell wall plate laws, one block per section
 rows [eps11 eps22 2eps12 | K11 K22 K12+K21 | 2g13 2g23]

 section 0: layup_0  layup [['alu', 0.036457, 0.0]]
```

Every route that reduces a layup to a wall plate law now emits this record — the cross-section
rings (beam `refined: 1`, and the `n_model: 3` equivalent solid) as well as the 3-D shell SG —
so the two-step reduction can be inspected at its halfway point.
(`examples/OpenSG_shell/4_get_solid_props_from_shell_3D_SG/schwarz_solid_props.py` is the same
homogenization through the API, plus the boundary-mode comparison.)

### From a gmsh mesh: `msh_to_yaml`

A surface mesh out of gmsh knows the geometry and nothing else — no layup, no material, no
macro model. `opensg_shell.helper.msh_to_yaml` writes the whole **mesh** side of the dialect
from an ASCII `.msh` (format 2.2 or 4.1) and refuses to invent the rest:

```python
from opensg_shell.helper.msh_to_yaml import convert

convert("schwarz_p_D2_shell.msh")                      # FILL_IN template
convert("schwarz_p_D2_shell.msh", thickness=0.036457)  # runnable, one alu ply
```

| call | what you get |
|---|---|
| `convert(path)` | the mesh blocks complete, `sections:`/`materials:` written as a marked `FILL_IN` **template**. The file is deliberately not runnable: `opensg_shell` rejects it and `check_filled` names every field still to be filled. |
| `convert(path, thickness=t)` | a runnable single-ply yaml. The default wall material is aluminium, $E = 69$ GPa, $\nu = 0.30$, $\rho = 2700$ kg/m³; pass `material={...}` for your own. |
| `convert(path, layup=[[mat, t, angle], ...], materials=[...])` | the general multi-ply form. |

The per-facet frame is the one `make_schwarz_yaml.py` established — $e_3$ the facet normal,
$e_2$ the first edge, $e_1 = e_2 \times e_3$ — and one `layup_<k>` element set is written per
gmsh physical tag. Only 3-node triangles (gmsh type 2) and 4-node quads (type 3) are accepted;
0-/1-D entities gmsh writes for physical points and curves are skipped, anything else is an
error. The `sections:` and `materials:` blocks are emitted **first**, above the mesh, because
they are the part a human edits and a `sections:` key buried at line 66262 of a 4 MB file looks
missing.

## The solid SG YAML

The yaml is the **native user input** — you author it directly (a SwiftComp `.sc` is just one
optional way to generate one). The solid dialect describes a meshed *domain* rather than a
laminated *surface*, so it carries no layups and no orientations — each material block is
either a $6\times6$ stiffness or engineering constants with a ply angle.
`RHC_SW_2UC_45.yaml` is a two-unit-cell honeycomb plate SG with 4251 nodes, 6640 triangles
and 3 materials; the excerpt below keeps the analysis header, two rows per list and two of the
three material blocks:

```yaml
n_model: 2      # 1 = beam, 2 = plate, 3 = solid -- the macro model this SG homogenizes to
refined: 1      # 0 = classical (plate ABD / beam EB); 1 = shear-refined (plate ABDG / beam Timoshenko)
msg: solid      # the ENGINE this SG belongs to (opensg_solid); `opensg <yaml>` dispatches on it
nodes:
- [17.6240005, 23.3500004, 0.0]
- [-17.6240005, 23.3500004, 0.0]
cells:
- [0, 66, 345]
- [345, 3, 0]
mat_id: [1, 1, 1, 1, 1]
materials:
  1:
    type: 1
    engineering: [108000.0, 8000.0, 8000.0, 4000.0, 4000.0, 3000.0, 0.32, 0.32, 0.30]
    angle: 45.0
  3:
    type: 1
    engineering: [69000.0, 69000.0, 69000.0, 26540.0, 26540.0, 26540.0, 0.30, 0.30, 0.30]
    angle: 0.0
```

| key | meaning |
|---|---|
| header keys | `msg`, `n_model`, `refined`, `analysis`, `epsilon_bar`, `omega`, `aperiodic` — the analysis request, every key defaulted; a headerless mesh runs as a classical plate homogenization. See "The YAML header" above. There is no `dim:` and no `scale:`: the SG dimension is inferred from the mesh (the coordinates occupy the leading columns, `points = nodes[:, 0:n_sg]`, so a column that never varies is padding) and $\omega$ is measured from it. |
| `nodes` | comma-separated coordinate rows, always three components. |
| `cells` | element connectivity, **0-based** in this dialect (note the `0` in the first row). All elements in one SG must have the same node count — mixed element types raise. |
| `mat_id` | one material id per cell, **1-based** (the reader forms `mat_id - 1` internally). |
| `materials` | a mapping from that id to a material block. |

Each material block carries a `type` that selects how the $6\times6$ is built, matching the
SwiftComp material types:

| `type` | fields | interpretation |
|---|---|---|
| 0 | `E`, `nu` | isotropic; $G = E/2(1+\nu)$ |
| 1 | `engineering` | the nine constants $E_1\,E_2\,E_3\,G_{12}\,G_{13}\,G_{23}\,\nu_{12}\,\nu_{13}\,\nu_{23}$ |
| 2 | `C` | the $6\times6$ stiffness stored directly — typically an already-rotated ply, so no angle should be supplied with it |

Any block may additionally carry `angle:` (degrees — the in-plane ply rotation, applied when
the material is built from constants; how a $\pm45$ laminate lives in the file) and
`density:` (kg/m³ or the mesh's own mass unit — used by the beam route's $6\times6$ mass
matrix). `aux` is the auxiliary line carried through verbatim when a file was converted from
an `.sc`; the elastic path does not consume it and a hand-written yaml simply omits it.

![RHC honeycomb 2-D solid SG mesh](_static/rhc_2dsg_mesh.png)

The recommended pattern keeps the ply data in exactly one place — the yaml itself: `type: 1`
material blocks carrying the nine engineering constants plus their `angle:`, and the analysis
request in the yaml header (`msg: solid`, `n_model: 2`, `refined: 1`; every key defaulted, and
`analysis:` omitted because `H` is the default). The run is then one command:

```bash
opensg RHC_SW_2UC_45.yaml
```

(`opensg_solid RHC_SW_2UC_45.yaml` and `python -m opensg_solid RHC_SW_2UC_45.yaml` are the same
entry point — `opensg` simply reads `msg: solid` from the header and forwards to it.)
Or, through the API, one call:

```python
from opensg_solid.sg_homo import plate_homo_2d

r = plate_homo_2d("RHC_SW_2UC_45.yaml")   # header + materials from the file
print(r["law_title"], r["law"])
```

(The `material_param=`/`angles=` override arguments remain for scripting over pre-rotated
`type: 2` files, as examples 4 and 7 do.) The header's `n_model:` picks the macro model — 1
beam, 2 plate, 3 solid — and `refined:` upgrades plate ABD to ABDG or beam EB to Timoshenko.
This file ships with `refined: 1`, so the run
writes `RHC_SW_2UC_45.out` — `The Effective Reissner-Mindlin Stiffness Matrix`, the $8\times8$
on rows `[e11 e22 g12 k11 k22 k12 2g13 2g23]`, with the banner
` OpenSG msg-solid plate model, omega 35.248001, periodic` — plus `RHC_SW_2UC_45_mesh.png`.
Set `refined: 0` and the same file reports the classical $6\times6$ ABD on rows
`[N11 N22 N12 M11 M22 M12]`, titled `The Effective Classical Plate Stiffness Matrix`.
The printed $\omega = 35.248001$ is the in-plane span measured from the mesh — the yaml
carries no normalization key, because the homogenizer always measures its own SG. For a cell
converted from SwiftComp it agrees with the `.sc` trailing line (`35.248` here) to the
digits that line carries, which is the check that the mesh and the `.sc` header describe the
same cell.

## The SwiftComp `.sc` input and the converter

A `.sc` file can be used directly — `load_sg_input` dispatches on the extension and converts —
or converted once and kept as YAML. The anatomy the reader expects, from the
`opensg_solid.io.sc_to_yaml` module docstring:

| block | content |
|---|---|
| model header | 1 to 3 short lines (submodel flags and similar) |
| META line | `dim  n_nodes  n_elems  n_mats` followed by further integers; located as the first line holding at least five integers |
| node records | `n_nodes` lines of `id  x [y [z]]` |
| element records | `n_elems` lines of `id  mat_id  conn`, the connectivity zero-padded to a fixed slot count; the padding zeros are not nodes |
| material blocks | per material: a header `mat_id  mat_type  n`, an auxiliary line such as `0 0`, then the properties — type 0 is `E  nu`, type 1 is `E1 E2 E3 / G12 G13 G23 / v12 v13 v23`, type 2 is the 21 upper-triangular constants of the $6\times6$ over six lines |
| trailing line | the `scale` (volume or thickness normalization) — read and kept on the parsed dict for the cross-check above. It is not written into the yaml as a `scale:` key: $\omega$ is measured from the mesh. The converter re-emits it as the optional `omega:` header key in the one case the solver would consume it — a 3-D SG homogenized to a 3-D solid (`n_model: 3`), whose measure is exactly the node bounding-box volume — and then only when it differs from that volume by more than $10^{-6}$ relative; otherwise the value is dropped, with a printed note when it disagrees with the measurement |

In `RHC_SW_2UC_45.sc` the META line is `2 4251 6640 3 0 0` — a 2-D SG with 4251 nodes, 6640
elements and 3 materials — and the file ends with the lone line `35.248`. The file's last
triangle, element 6640 of material 3 on nodes 4251/2988/3079, shows the zero padding in full:

```
6640 3 4251 2988 3079 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
```

The documented way to run the conversion is the module's own `convert`:

```python
from opensg_solid.io.sc_to_yaml import convert

sc = convert("RHC_SW_2UC_45.sc")
print(sc["dim"], len(sc["nodes"]), len(sc["cells"]))
```

`convert` writes **two** files next to the input — `<base>.yaml`, the solid SG YAML documented
above, and `<base>.msh`, a gmsh v2.2 mesh carrying the material id as both physical and
elementary tag — and returns the parsed dict. The module defines no `__main__` block, so import
`convert` rather than invoking the module with `python -m`. Because `plate_homo_2d` performs
the same conversion automatically when handed a `.sc` path, converting by hand is only needed
when you want to inspect or edit the YAML before homogenizing.

## The through-thickness `layup_db.yaml`

For a plain laminate there is no mesh to write: the 1-D SG is one element stack through the
thickness, and `layup_db.yaml` is the whole user input. `1d_sg.py` reads it, generates the SG
mesh (`1dsg.yaml` plus `1dsg.png`) through `segment_plate.plate_sg_yaml`, homogenizes, and
writes `layup_db_plate_homo.out`.

![Through-thickness 1-D plate SG](_static/plate_1dsg.png)

| key | meaning |
|---|---|
| `model` | `0` prints the classical $6\times6$ ABD, `1` the shear-refined $8\times8$ ABDG. This choice also names the matrix in the `.out`: `Classical Plate` or `Reissner-Mindlin Plate`. |
| `fraction` | the reference surface as a fraction of the total thickness: `0` = bottom/OML face, `0.5` = mid-surface, `1` = top. This is the solid-side analogue of the shell YAML's `reference`. |
| `mesh.n_per_layer` | SG elements per physical ply. |
| `mesh.elem_order` | element polynomial order $p$; nodes per element are $p+1$, and the engine supports 1 to 4. `4` — the five-noded quartic — is the default in the example and is exact for the warping ladder. |
| `materials` | a mapping from short name to `E` (3), `G` (3), `nu` (3) and `rho`, with an optional `full_name`. Every number must be a float; the example's comment about `128.0e9` parsing as a string is a real PyYAML trap. |
| `layup` | the stacking sequence, **bottom ply first**, each entry `{material, thickness, angle}`. An optional `divisions` key is read only by the Abaqus 3-D deck generator and ignored by the homogenizer. |

The Nayak sandwich in the example is a nine-ply $(0/90/0/90/\text{core})_s$ stack of
graphite/epoxy faces on a HEREX C70.130 PVC foam core:

```yaml
mesh:
  n_per_layer: 1
  elem_order: 4
fraction: 0.5
layup:
  - {material: ge,    thickness: 0.0009525, angle:  0.0}
  - {material: ge,    thickness: 0.0009525, angle: 90.0}
  - {material: ge,    thickness: 0.0009525, angle:  0.0}
  - {material: ge,    thickness: 0.0009525, angle: 90.0}
  - {material: herex, thickness: 0.1447800, angle:  0.0, divisions: 8}
  - {material: ge,    thickness: 0.0009525, angle: 90.0}
  - {material: ge,    thickness: 0.0009525, angle:  0.0}
  - {material: ge,    thickness: 0.0009525, angle: 90.0}
  - {material: ge,    thickness: 0.0009525, angle:  0.0}
```

```bash
python 1d_sg.py
```

The `.out` banner echoes back every number that mattered, which is the fastest way to confirm
the file was read as intended:
` OpenSG msg-solid plate model from 1dsg.yaml: 9 plies, h = 0.152400 m, fraction = 0.5,
rho*h = 30.251400 kg/m^2, rows [e11 e22 g12 k11 k22 k12 2g13 2g23]`. The generated
`1dsg.yaml` records the same decisions in its header — `sg: {type: plate_1d, elem_order: 4,
n_per_layer: 1, reference_fraction: 0.5, thickness: 0.15239999999999998, n_ply: 9}` — and its
`elements` rows are the expected five-node quartics, `- [1, 2, 3, 4, 5]`, with coordinates
running
from $-h/2$ to $+h/2$ because `fraction: 0.5` put the origin at mid-thickness. `1d_sg.py`
accepts an alternative database as its one positional argument.

## Finite elements used by each route

The two packages discretize differently, and knowing which element you are getting explains
both the DOF count and the failure modes.

### Shell route

The shell SG never assembles a line element directly. The contour is **extruded one quad deep**
along the beam axis by `sg_homo._strip` (extrusion length $h$ = the mean contour element
length), giving quads `[a, b, m+b, m+a]`, and the top node row is DOF-mapped onto the bottom
row so the field is exactly span-invariant — the operator's own prismatic reduction. The
element is therefore a **4-node bilinear quad** integrated at $2\times2$ Gauss points, and the
1-D cross-section problem is recovered as its span-invariant limit.

Two element families exist:

| entry point | DOF per node | DOF per element | drilling |
|---|---|---|---|
| `sg_homo.ring_indep` (production) | 6: `[w1, w2, w3, om1, om2, om3]` | 24 | independent nodal $\omega_3$, tied back by a finite drilling residual (`NDOF6 = 6`) |
| `sg_assembly.ring_general` | 5: `[w1, w2, w3, om1, om2]` | 20 | eliminated algebraically through $\omega_3 = S/(2C_{33}) - (C_{3b}/C_{33})\,\omega_b$ (`NDOF = 5`) |

The independent-drilling element exists because the elimination divides by $C_{33} = n \cdot
b_3$, which vanishes on a flat wall. In the 6-DOF element $\omega_3$ is a genuine nodal DOF and
the in-plane symmetry that defined it is re-imposed in its finite, undivided form as the
drilling residual $DR = C_{33}\omega_3 + C_{3b}\omega_b - S/2$, enforced by an element-wise
Lagrange multiplier (`lam_space="elem"`) or a penalty — finite even at $C_{33} = 0$.
Transverse shear uses MITC assumed-strain tying (Dvorkin–Bathe): the production ring scheme is
`shear="mitc4_g23"`, which ties only $\gamma_{23}$, because under span invariance $\gamma_{13}$
is algebraic in the directors and carries no fluctuation gradient.

The 3-D shell SG uses the same quad element, with triangles entering as quads whose last node
is repeated.

### Solid route

`opensg_solid.sg_mesh._cell_basis` maps the SG dimension and the node count per element onto a
basix cell type and Lagrange degree — this table *is* the list of supported elements:

| SG dim | nodes/elem | basix cell | degree |
|---|---|---|---|
| 1 | 2 | `interval` | 1 |
| 1 | 3 | `interval` | 2 |
| 1 | 4 | `interval` | 3 |
| 1 | 5 | `interval` | 4 |
| 2 | 3 | `triangle` | 1 |
| 2 | 4 | `quadrilateral` | 1 |
| 3 | 4 | `tetrahedron` | 1 |
| 3 | 8 | `hexahedron` | 1 |
| 3 | 10 | `tetrahedron` | 2 |

Anything else raises `unsupported <dim>D element with <n> nodes`. Basis functions are
equispaced Lagrange with quadrature degree 6 for degree $> 1$ and 2 otherwise. The table is
consulted by `plate_homo_2d` alone, so it does not describe the `layup_db.yaml` route:
`1d_sg.py` builds its mesh with `rm_plate_1D.segment_plate`, whose per-element connectivity is
sequential — the five-noded quartic is written `- [1, 2, 3, 4, 5]` — and homogenizes with
`rm_plate_1D.msg_rm_plate.rm_plate_msg`, which carries its own Lagrange basis. The ends-first
gmsh/basix line ordering, the two end nodes followed by the interior nodes left to right,
belongs instead to `opensg_solid.sg_mesh.laminate_to_sg`, the laminate-to-engine route whose
output does go through `_cell_basis`.

## Sign and order conventions

Three orderings appear across the codebase, and every one of them is stated in a module
docstring rather than inferred.

**Solid (3-D) Voigt order** is the SwiftComp order $[11\;22\;33\;23\;13\;12]$ with
**engineering shears** — so the compliance carries $S_{44} = 1/G_{23}$, $S_{55} = 1/G_{13}$,
$S_{66} = 1/G_{12}$, and the shear rows of a recovered strain are $2\gamma$, not $\gamma$. This
is the order of `write_sc_K`'s output, of `build_stiffness_6x6`, and of every `type: 2` material
`C` block.

**Beam order** is

$$[\,\varepsilon_{11}\;\;\gamma_{12}\;\;\gamma_{13}\;\;\kappa_1\;\;\kappa_2\;\;\kappa_3\,],$$

whose diagonal is the VABS ordering $[\,EA\;\;GA_2\;\;GA_3\;\;GJ\;\;EI_2\;\;EI_3\,]$ — the
`LBL` list used by the shell drivers. Axis 1 is the beam axis; axes 2 and 3 are the
cross-section axes. The solver indexes global axes as $(1,2,3) = (\text{axial}, y_2, y_3)$
while an SG YAML writes each 3-vector as $(y_2, y_3, \text{axial})$, which is why
`sg_materials.elem_rotation_from_yaml` cycles `[a, b, c] -> [c, a, b]` before a YAML
orientation can be used as a direction-cosine matrix in the solid beam path. Skipping that
cycle points $e_1$ along $y_3$: the run still completes and looks plausible, but a square tube
comes out with $EI_2 \neq EI_3$.

**Plate order** is $[\,N_{11}\;N_{22}\;N_{12}\;M_{11}\;M_{22}\;M_{12}\,]$ for the classical
$6\times6$, i.e. the strain vector $[\varepsilon_{11}\;\varepsilon_{22}\;2\varepsilon_{12}\;
\kappa_{11}\;\kappa_{22}\;\kappa_{12}]$. The Reissner–Mindlin $8\times8$ appends the two
transverse shears, giving rows `[e11 e22 g12 k11 k22 k12 2g13 2g23]` and the block structure

$$\mathbf{ABDG} = \begin{bmatrix} A & B & 0 \\ B & D & 0 \\ 0 & 0 & G \end{bmatrix}.$$

For a plate SG the **last SG coordinate is the thickness direction** — the $y_3$ of the solid
Voigt order above. On a 2-D SG that is node coordinate index 1, since only the first
$n_{\rm sg}$ coordinates are read; for a through-thickness 1-D SG it is the single coordinate.
This is why
the RHC cell reports $\omega = 35.248$, the span of coordinate 0 (the in-plane periodic
direction), and not its larger extent in coordinate 1.

Changing the reference surface moves membrane strain to $m + z_0 \kappa$, so the ABD transforms
as $\mathbf{ABD}_{\text{new}} = T^{-\mathsf{T}}\,\mathbf{ABD}\,T^{-1}$ with $T = [[I, z_0 I],
[0, I]]$ (`shift_abd_reference`). Use it rather than reversing the layup: reversal flips $e_3$
and breaks agreement with the material orientation. The transverse-shear block $G$ is
reference-independent.

## Material orientation

Each row of `elementOrientations` is nine direction cosines forming three stacked unit vectors,

```
[ e1x e1y e1z | e2x e2y e2z | e3x e3y e3z ]
```

with the roles fixed across both dialects:

- **$e_1$ is the beam axis** — the prismatic direction, which for a plane cross-section meshed
  in $(y_2, y_3)$ is *out of plane*, `e1 = [0, 0, 1]`.
- **$e_2$ is the in-plane ply-flow direction**, the wall tangent along the contour.
- **$e_3 = e_1 \times e_2$ is the wall normal**, in plane for a cross-section SG, and the
  direction the layup stacks along from the reference surface.

The triad must be orthonormal and right-handed. `sg_mesh.frame_report` checks that numerically
— unit norms, $\max|e_i \cdot e_j| < 10^{-6}$, $e_1 \times e_2 \cdot e_3 > 0.99$, and the
cylinder-specific $e_1 \cdot \hat{x} > 0.999$ — and **returns** the pair `(ok, txt)`, a boolean
and a one-line summary ending in `OK` or `CHECK`; the callers do the printing. Its remaining
test, the mean of $e_3 \cdot (-\hat{r})$, is cylinder-specific as its own docstring says, and it
is reported in the summary text without entering the pass/fail gate — a frame whose $e_3$ has
flipped to point outward still returns `OK`, so read that number rather than trusting the
verdict for it.

The ply angle is **not** baked into $e_1$. It lives in the third column of each `layup` row
(shell dialect) or in the `angles` argument (solid dialect), and it rotates the material about
$e_3$, away from $e_1$: `rotation_6x6(theta)` leaves the Voigt-33 row untouched and mixes the
11/22/12 rows, and `_ply_Q_and_G` builds

$$\bar{Q}_{11} = Q_{11}\cos^4\theta + 2(Q_{12} + 2Q_{66})\sin^2\theta\cos^2\theta
  + Q_{22}\sin^4\theta ,$$

so `angle: 0.0` places the fibre along $e_1$ — the beam axis. A wind-blade unidirectional spar
cap therefore carries `angle: 0.0`, and a $\pm45$ ply carries `45.0` / `-45.0`. The governing
repo rule is `Rules/orientation_e1_out_of_plane.md`, which also records the failure mode: a
YAML whose $e_1$ already contains the fibre rotation (so $|e_{1z}| < 1$) is a *fibre* frame, and
combining it with a non-zero `angle` applies the rotation twice.

Because any mesh handling can renumber nodes and cells, an orientation PNG is a compulsory
deliverable of every run — but no homogenization entry point emits one for you. It is the
example driver that calls `opensg_shell.auto_emit`, as
`examples/OpenSG_shell/1_get_beam_props_from_shell_cross_section/beam_homo_shell.py` does, and
your own driver should do the same. The call is cached once per path per process and never
raises, so it is safe to place inside a compute path. The colour convention of
`fe_jax/orient_plot.py` is **$e_2$ blue, $e_3$ green**, drawn at element centroids over a light
mesh; $e_1$ is out of the page and so not drawn. The canonical example is the IEA-22
$r/R = 0.2$ station:

![Material orientation of the IEA-22 r/R = 0.2 shell cross-section, e2 in blue and e3 in green](_static/shell_xsec_orient.png)

Read it as a physical check, not decoration. On this station the green $e_3$ arrows point
*inward* everywhere on the skin, which is what an outward-first layup stacked from the
mid-surface requires, and the blue $e_2$ arrows are tangent to every wall — running
consistently around the closed skin loop and along each of the three shear webs. An $e_3$ that
flips between neighbouring skin elements, or a web whose $e_2$ opposes its neighbours, means
the layup is being stacked in two directions at once: visible here in seconds, and invisible in
the resulting $6\times6$.
