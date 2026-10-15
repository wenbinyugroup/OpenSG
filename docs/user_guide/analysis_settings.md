# Analysis settings

The header of an SG yaml is the analysis request the file makes about itself. It is a short block of scalar keys at the
top of the file, above the first mesh block, and it states which engine owns the file, which macro model the structure
gene (SG) is homogenized to (beam, plate or 3-D solid), whether the shear-refined law is wanted, and whether the run is a
homogenization (`H`) or a dehomogenization (`D`). The engines read the header without parsing the mesh, so a run reports
what it will do before the solve starts.

Every key has a default, so a mesh without a header still runs: a solid SG as a classical plate homogenization, a shell
SG as a classical beam homogenization with the node contour taken as the outer mold line (OML). The command line
carries no model codes. Besides the yaml path it takes the optional analysis token (`H` or `D`, which wins over the
`analysis:` key) and a few flags: `--center` (see [the laminate reference](#ug-laminate-reference) below), `--mesh`,
`--solver` and the output-frame flags of a dehomogenization, all described in {doc}`../getting_started/running`.

There is no `dim:` key and no `scale:` key. The SG dimension is read from the mesh (see {doc}`mesh_and_elements`), and
the SG measure $\omega$ is measured from the mesh unless `omega:` declares it.

## Header keys

% source: src/opensg_solid/sg_mesh.py (read_yaml_header, resolve_msg); src/opensg_shell/cli.py (main, --help);
% src/opensg_solid/cli.py (main, --help); src/opensg_solid/sg_homo.py (plate_homo_2d);
% src/opensg_shell/sg_dehom_junction.py (read_tier); docs/input_format.md ("The YAML header")
```{table} Header keys of an SG yaml. "Shell" and "solid" name the engine that reads a key; a key without such a note applies to both.
:name: tab-ug-header-keys

| key | values | meaning | default |
|---|---|---|---|
| `msg` | `shell` \| `solid` | The engine that owns the file. The unified `opensg` command dispatches on this key. A `msg:` that contradicts the mesh blocks is an error. | inferred from the mesh blocks |
| `n_model` | `1` \| `2` \| `3` | The macro model: 1 beam, 2 plate, 3 equivalent 3-D solid. `n_model: 2` is a solid-engine route: `opensg_shell` refuses it, because there the wall is the plate. | shell: `1`; solid: `2` |
| `refined` | `0` \| `1` | `0` classical (plate ABD (membrane, coupling and bending) $6\times6$, beam Euler-Bernoulli $4\times4$), `1` shear-refined (plate Reissner-Mindlin ABDG $8\times8$, beam Timoshenko $6\times6$). Ignored for `n_model: 3`. | `0` |
| `analysis` | `H` \| `D` | Homogenization or dehomogenization, when the switch is carried in the file. The command-line token wins. | `H` |
| `epsilon_bar` | 6 floats | The macro state a `D` run recovers from: `[e11 e22 2e12 k11 k22 2k12]` (plate, solid) or `[ext sh2 sh3 twist bend2 bend3]` (beam). A `<base>.ff` file next to the yaml supersedes it. | none |
| `omega` | float | Optional user SG measure, overriding the measured one. Solid: 3-D solid macro model only. Shell: the `n_model: 3` routes. | measured from the mesh |
| `aperiodic` | `1` | Optional, only for an SG that is not periodic: zero fluctuation on every node of the bounding-box faces instead of the periodic tie. Omit the key for a periodic SG. Shell: the `n_model: 3` 3-D shell SG. | periodic |
| `junction` | `off` \| `flag` \| `exclude` | Shell, `D` only: the junction-aware recovery tier (see {doc}`recovery`). | `flag` |
| `junction_bl` | float | Shell, `D` only: the junction boundary-layer radius, in units of the thickest wall there. | `1.0` |
| `junction_ang` | float, deg | Shell, `D` only: the tangent-grouping tolerance of the junction detector, between 0 and 90. | `1.0` |
| `junction_twist` | `hinged` \| `kirchhoff` | Shell, refined cross-section ring only: the twist condition of a wall that ends at a junction node. `kirchhoff` adds one constraint row per ending wall, imposing $2\gamma_{13} = 0$ at the node. | `hinged` |
```

{numref}`lst-ug-header-segment` to {numref}`lst-ug-header-beam-d` show three headers from the example folders: a
tapered blade segment that is not periodic along its axis, a cross-section homogenized to an equivalent solid with a
declared measure, and a solid beam SG that carries its own macro state for a dehomogenization.

% source: examples/OpenSG_shell/5_taper_segment_aperiodic_boundaries/BAR_URC_numEl_52_segment_12.yaml
```{code-block} yaml
:caption: Header of a shell segment SG (BAR_URC_numEl_52_segment_12.yaml).
:name: lst-ug-header-segment

n_model: 1      # 1 = beam, 2 = plate (opensg_solid), 3 = solid -- the macro model this SG homogenizes to
refined: 1      # 0 = classical (beam EB 4x4, KL wall); 1 = shear-refined (beam Timoshenko 6x6, RM wall)
msg: shell      # the ENGINE this SG belongs to (opensg_shell); `opensg <yaml>` dispatches on it
aperiodic: 1    # the two end cross-sections carry their own V0/V1 as Dirichlet data (no axial periodicity)
```

% source: examples/OpenSG_shell/3_get_solid_props_from_shell_cross_section/square_section_isotropic/square_tube_1Dshell.yaml
```{code-block} yaml
:caption: Header of a cross-section homogenized to a 3-D solid (square_tube_1Dshell.yaml).
:name: lst-ug-header-omega

n_model: 3      # 1 = beam, 2 = plate (opensg_solid), 3 = solid -- the macro model this SG homogenizes to
refined: 0      # 0 = classical (beam EB 4x4, KL wall); 1 = shear-refined (beam Timoshenko 6x6, RM wall)
msg: shell      # the ENGINE this SG belongs to (opensg_shell); `opensg <yaml>` dispatches on it
omega: 0.12     # user SG measure = the wall MATERIAL area 4*a*t (a=1.0, t=0.03), the drivers' cell_area
```

% source: examples/OpenSG-solid/8_get_beam_dehom_from_SG/RHC_SW_2UC_45.yaml
```{code-block} yaml
:caption: Header of a solid beam SG with a macro state (8_get_beam_dehom_from_SG/RHC_SW_2UC_45.yaml).
:name: lst-ug-header-beam-d

n_model: 1      # 1 = beam, 2 = plate, 3 = solid -- the macro model this SG homogenizes to
refined: 1      # 0 = classical (plate ABD / beam EB); 1 = shear-refined (plate ABDG / beam Timoshenko)
msg: solid      # the ENGINE this SG belongs to (opensg_solid); `opensg <yaml>` dispatches on it
epsilon_bar: [0.001, 0.0, 0.0, 0.0, 0.01, 0.0]   # macro beam strain for opensg_solid <yaml> D
```

:::{warning}
The header ends at the first mesh block key (`nodes`, `cells`, `mat_id`, `materials`, `elements`, `sets`, `sections`
or `elementOrientations`). A header key written below that line is not read. Keep the header at the top of the file.
:::

## What the header selects

`n_model` and `refined` pick the macro law. In the shell engine the mesh also picks the route: 2-node line elements
are a cross-section contour, 3- and 4-node elements a surface in space. {numref}`tab-ug-routes` lists every
combination and the file the homogenization writes; the file layout is described in {doc}`output_files`.

% source: src/opensg_shell/cli.py (--help text and main); src/opensg_solid/sg_homo.py (plate_homo_2d law titles)
```{table} Macro law and output file selected by the header; base is the yaml name without its extension.
:name: tab-ug-routes

| engine | `n_model` | `refined` | mesh | macro law | written to |
|---|---|---|---|---|---|
| shell | 1 | 0 | contour (2-node lines) | Euler-Bernoulli beam $4\times4$, `[eps11 kappa1 kappa2 kappa3]` | `<base>_EB.out` |
| shell | 1 | 1 | contour (2-node lines) | Timoshenko beam $6\times6$, `[eps11 gam12 gam13 kappa1 kappa2 kappa3]` | `<base>_Timo.out`, `<base>_ABDG.out` |
| shell | 1 | 1 | surface segment (4-node quads) | Timoshenko beam $6\times6$ of a tapered or aperiodic segment | `<base>_Timo.out` |
| shell | 3 | ignored | contour (2-node lines) | Cauchy continuum $6\times6$, `[11 22 33 23 13 12]` | `<base>_C3D.out`, `<base>_ABDG.out` |
| shell | 3 | ignored | surface (3-node triangles, 4-node quads) | Cauchy continuum $6\times6$, `[11 22 33 23 13 12]` | `<base>_C3D.out`, `<base>_ABDG.out` |
| solid | 1 | 0 / 1 | 2-D or 3-D SG | Euler-Bernoulli beam $4\times4$ / Timoshenko beam $6\times6$ | `<base>.out` |
| solid | 2 | 0 / 1 | 1-D, 2-D or 3-D SG | classical plate $6\times6$ `[N11 N22 N12 M11 M22 M12]` / Reissner-Mindlin $8\times8$ `[N11 N22 N12 M11 M22 M12 Q1 Q2]` | `<base>.out` |
| solid | 3 | ignored | 1-D, 2-D or 3-D SG | Cauchy continuum $6\times6$ and the engineering constants | `<base>.out` |
```

`<base>_ABDG.out` is the wall plate law of every section, the halfway point of the two-step shell reduction. The
segment route returns the Timoshenko $6\times6$ only, so a shell segment needs `refined: 1`. A dehomogenization (`D`)
is available for the shell cross-section ring with `n_model: 1` and `refined: 1`, and for the solid engine; see
{doc}`recovery`.

## Boundary treatment: periodic or aperiodic

An SG homogenized to an equivalent 3-D solid needs a constrained boundary. A free boundary admits affine fluctuation
fields that cancel five of the six macro strains at zero energy, and the law comes out rank one: only $C_{11}$ survives.
Periodicity is the default at every SG dimension. It ties opposite faces through the assembly map, never partially and
never per direction:

% source: docs/theory/rules/periodicity_in_solid_props.md
| SG | what is tied |
|---|---|
| 2-D SG | opposite face edges, and the corners |
| 3-D SG | all opposite faces, edges and corners |

`aperiodic: 1` replaces the tie with the boundary solution: zero translational fluctuation (Dirichlet) on every node of
the bounding-box faces. It is a kinematic upper bound on one cell and is faster than the periodic solve, as the
Schwarz-P shell SG of {doc}`../examples/shell_solid_3d_tpms` shows in {numref}`tab-ug-aperiodic`.

% source: docs/theory/rules/periodicity_in_solid_props.md (Schwarz-P TPMS shell, same yaml, same mesh)
```{table} Aperiodic against periodic results on the Schwarz-P shell SG, per wall thickness t.
:name: tab-ug-aperiodic

| t | normals | shears | couplings | time vs periodic |
|---|---|---|---|---|
| 0.0365 | +6.8 % | +3.6 % | -1.7 % | 72 s vs 96 s |
| 0.1293 | +10.8 % | +4.8 % | -8.0 % | 71 s vs 95 s |
```

:::{tip}
Use the periodic default for any property reported as the effective stiffness of a unit cell, and for every
comparison with a SwiftComp `.K`. Use `aperiodic: 1` when the SG is not periodic: blade segments, cut-outs, one-off
cells. A shell segment (`n_model: 1` on a surface mesh) is aperiodic by construction, because its two end
cross-sections carry their own warping as Dirichlet data; there the key only documents the case.
:::

Periodicity ties whatever cell it is given, so the cell must be a genuine unit cell. A wall lying on the cell boundary
is shared with the neighbour cell and is counted twice. Put the walls through the cell instead, one wall of each
family inside the cell.

## The SG measure

The homogenized energy is divided by the SG measure $\omega$, the measure of the domain the macro model does not
resolve. It is measured from the mesh:

% source: src/opensg_solid/io/sg_input.py (sg_omega); src/opensg_solid/sg_assembly.py (_solid_omega);
% src/opensg_shell/cli.py (sg_cell_area); examples/OpenSG-solid/3_get_plate_props_from_2D_SG/RHC_SW_2UC_45.out
- a 3-D SG homogenized to a 3-D solid: the volume of the node bounding box, the periodic unit cell including its holes;
- a 1-D or 2-D solid SG homogenized to a 3-D solid: the integrated material measure (length or area) of the elements;
- a plate: the in-plane span of the SG (a length for a 2-D SG, an area for a 3-D SG), for example `omega 35.248001`
  on the banner of `examples/OpenSG-solid/3_get_plate_props_from_2D_SG/RHC_SW_2UC_45.out`; a 1-D plate SG has
  $\omega = 1$;
- a beam from a 2-D SG: 1, the cross-section itself being the measure; from a 3-D SG: the span of the first SG
  coordinate;
- a shell cross-section homogenized to a 3-D solid: the area of the node bounding box, not the convex hull of the
  contour;
- a 3-D shell SG homogenized to a 3-D solid: the volume of the node bounding box.

% source: examples/OpenSG_shell/3_get_solid_props_from_shell_cross_section/square_section_isotropic/square_tube_1Dshell.yaml; examples/OpenSG_shell/3_get_solid_props_from_shell_cross_section/deo_cellular_solid/deo_cellular_+15_1Dshell.yaml
Set `omega:` only when the equivalent continuum occupies something other than the measured cell. The square tube of
{numref}`lst-ug-header-omega` declares its wall material area, `omega: 0.12`, and the cellular lattices of
{doc}`../examples/shell_solid_cellular` declare their periodic cell area, for example
`omega: 0.048637033051562736` in `deo_cellular_+15_1Dshell.yaml`.

(ug-laminate-reference)=
## The laminate reference of a shell SG: `--center`

The shell yaml carries no reference key. Which surface of the wall the node contour is, and therefore where the wall
ABD (the membrane, coupling and bending stiffness of the laminate), the plate-SG $z$ origin, the mass moments and the
recovery depths are referenced, is a run-time choice, made the same way on every shell route:

% source: docs/input_format.md ("The laminate reference (--center)"); src/opensg_shell/sg_reference.py
```{table} The two laminate references of the command line.
:name: tab-ug-reference

| run | the node contour is | fraction of thickness from the outer face |
|---|---|---|
| `opensg <yaml>` (default, `oml`) | the outer mold line; every laminate stacks inward from it, no shift | 0.0 |
| `opensg <yaml> --center` | the laminate mid-surface; the ABD is shifted by $t/2$ ($B' = B - \tfrac{t}{2}A$, $D' = D - tB + \tfrac{t^2}{4}A$) and recovery depths run from $-t/2$ to $+t/2$ about the contour | 0.5 |
```

% source: docs/input_format.md ("The laminate reference (--center)"); src/opensg_shell/pynumad/sg_mesh.py (emit_shell_yaml)
Where the contour sits is decided when the yaml is generated, and OpenSG never moves the nodes.
`emit_shell_yaml(..., reference="center")` and the OpenSG_io converter (`fraction=0.5`) move the skin nodes inward by
$t/2$ before writing, and record that only as a leading comment line. A mid-surface yaml
must therefore be run with `--center`; running it at the default shifts $B$ and $D$ by a parallel-axis term of the
wall thickness and moves every bending and torsion entry of the beam $6\times6$. The IEA-22 stations are mid-surface
meshes:

% source: docs/input_format.md ("Running it")
```bash
opensg iea_s10_shell.yaml --center     # the IEA stations are center-offset meshes
opensg iea_s10_shell.yaml --center --mesh   # also <base>_mesh.png + gmsh <base>.msh
```

The Python API takes the same choice as `ref=` (`build_rm_bundle(yaml, ref="center")`,
`beam_props(yaml, ref="center")`, `dehom_station(..., ref="center")`; default `"oml"`), and refuses an unknown name.
The diagnostic values `oml_flip` and `iml` (a full-thickness shift) exist in the API only. `--center` is refused for a
solid SG, whose materials carry no laminate to reference.

:::{note}
A `reference:` key left in an older yaml is ignored. The run prints a note when it agrees with the flag, and a
warning when it does not.
:::

## See also

- {doc}`coordinate_systems` for the order of `epsilon_bar` and of the stiffness entries.
- {doc}`../theory/rules/periodicity_in_solid_props` for the rank-one theorem and the aperiodic mode.
- {doc}`../examples/shell_taper_segment` for an aperiodic segment run end to end.
