# Tutorial

This tutorial walks through one complete OpenSG analysis of a composite wind-blade cross-section. It shows how the
input file is organized, how the effective Timoshenko beam properties are computed (homogenization), and how the 3-D
stress in the walls is recovered from a set of beam loads (dehomogenization). All files used here ship with the
repository in `examples/OpenSG_shell/1_get_beam_props_from_shell_cross_section/` and
`examples/OpenSG_shell/2_get_beam_dehom_from_shell_cross_section/`, so you can follow every step on your own machine.

## The design of a cross-section

```{figure} ../_static/shell_xsec_mesh.png
:width: 100%
:name: fig-tutorial-xsec-mesh

The cross-section contour of the IEA-22 MW blade at $r/R = 0.20$, coloured by layup.
```

% source: docs/tutorials/msg_shell_beam.md ("What a 1-D shell SG is"); examples/OpenSG_shell/1_get_beam_props_from_shell_cross_section/iea_s10_shell.yaml
The cross-section shown in {numref}`fig-tutorial-xsec-mesh` is station $r/R = 0.20$ of the IEA-22 MW reference
wind-turbine blade. It is a closed, airfoil-shaped skin with two carbon spar caps, a trailing-edge reinforcement, a
leading-edge reinforcement and three shear webs. The section spans about 7.2 m in $y_2$ and 2.5 m in $y_3$, and its
spar caps are 87.6 mm thick.

% source: docs/tutorials/msg_shell_beam.md ("What a 1-D shell SG is"); examples/OpenSG_shell/1_get_beam_props_from_shell_cross_section/iea_s10_shell.yaml
OpenSG models this thin-walled section as a 1-D shell structure gene (SG): the contour of the walls, meshed with
2-node line elements. Each element carries its laminate as material data, so the wall thickness never appears in the
mesh; it enters through the wall's plate law and through the recovery. The walls are modelled as Reissner-Mindlin
(RM) plates, and the homogenization returns the $6\times6$ Timoshenko stiffness matrix of the beam. The whole station
is described by 307 nodes and 310 elements.

The materials used in this cross-section are listed in {numref}`tab-tutorial-materials`. The layups are listed in
{numref}`tab-tutorial-layups`, and {numref}`tab-tutorial-sets` gives the wall region, the total thickness and the
number of elements of each layup.

% source: examples/OpenSG_shell/1_get_beam_props_from_shell_cross_section/iea_s10_shell.yaml (materials block)
```{list-table} Material properties
:name: tab-tutorial-materials
:header-rows: 2

* - Name
  - Density
  - $E_1$
  - $E_2$
  - $E_3$
  - $G_{12}$
  - $G_{13}$
  - $G_{23}$
  - $\nu_{12}$
  - $\nu_{13}$
  - $\nu_{23}$
* -
  - kg/m³
  - Pa
  - Pa
  - Pa
  - Pa
  - Pa
  - Pa
  -
  -
  -
* - gelcoat
  - 1235.0
  - 3440000000.0
  - 3440000000.0
  - 3440000000.0
  - 1323000000.0
  - 1323000000.0
  - 1323000000.0
  - 0.3
  - 0.3
  - 0.3
* - glass_triax
  - 1940.0
  - 28211400000.0
  - 16238800000.0
  - 15835500000.0
  - 8248220000.0
  - 3491240000.0
  - 3491240000.0
  - 0.497511
  - 0.18091
  - 0.27481
* - glass_uniax
  - 1940.0
  - 43700000000.0
  - 16500000000.0
  - 15450000000.0
  - 3265000000.0
  - 3495000000.0
  - 3480000000.0
  - 0.262
  - 0.264
  - 0.35
* - medium_density_foam
  - 130.0
  - 142500000.0
  - 142500000.0
  - 142500000.0
  - 54000000.0
  - 54000000.0
  - 54000000.0
  - 0.3194
  - 0.3194
  - 0.3194
* - carbon_uniax
  - 1600.0
  - 140000000000.0
  - 9100000000.0
  - 9100000000.0
  - 4131000000.0
  - 4131000000.0
  - 2689000000.0
  - 0.3133
  - 0.3133
  - 0.47066
* - glass_biax
  - 1940.0
  - 11023100000.0
  - 11023100000.0
  - 16047700000.0
  - 13231400000.0
  - 3487480000.0
  - 3487480000.0
  - 0.688074
  - 0.117173
  - 0.117173
```

% source: examples/OpenSG_shell/1_get_beam_props_from_shell_cross_section/iea_s10_shell.yaml (sections block)
```{list-table} Layups, outer ply first
:name: tab-tutorial-layups
:header-rows: 2

* - Region
  - Layup
  - Material
  - Ply thickness
  - Angle
* -
  -
  -
  - m
  - degree
* - Trailing-edge reinforcement
  - layup_0
  - gelcoat
  - 0.0005
  - 0.0
* -
  -
  - glass_triax
  - 0.00300294
  - 0.0
* -
  -
  - glass_uniax
  - 0.02609467
  - 0.0
* -
  -
  - glass_triax
  - 0.00300294
  - 0.0
* - Skin
  - layup_1
  - gelcoat
  - 0.0005
  - 0.0
* -
  -
  - glass_triax
  - 0.00300294
  - 0.0
* -
  -
  - medium_density_foam
  - 0.07060815
  - 0.0
* -
  -
  - glass_triax
  - 0.00300294
  - 0.0
* - Upper spar cap
  - layup_2
  - gelcoat
  - 0.0005
  - 0.0
* -
  -
  - glass_triax
  - 0.00300294
  - 0.0
* -
  -
  - carbon_uniax
  - 0.08105991
  - 0.0
* -
  -
  - glass_triax
  - 0.00300294
  - 0.0
* - Leading-edge reinforcement
  - layup_3
  - gelcoat
  - 0.0005
  - 0.0
* -
  -
  - glass_triax
  - 0.00300294
  - 0.0
* -
  -
  - glass_uniax
  - 0.02158284
  - 0.0
* -
  -
  - glass_triax
  - 0.00300294
  - 0.0
* - Lower spar cap
  - layup_4
  - gelcoat
  - 0.0005
  - 0.0
* -
  -
  - glass_triax
  - 0.00300294
  - 0.0
* -
  -
  - carbon_uniax
  - 0.07707452
  - 0.0
* -
  -
  - glass_triax
  - 0.00300294
  - 0.0
* - Shear webs
  - layup_5
  - glass_biax
  - 0.002
  - 0.0
* -
  -
  - medium_density_foam
  - 0.042
  - 0.0
* -
  -
  - glass_biax
  - 0.002
  - 0.0
```

% source: docs/tutorials/msg_shell_beam.md (layup table); thicknesses also in
% examples/OpenSG_shell/1_get_beam_props_from_shell_cross_section/abd/iea_s10_shell_abd.yaml
```{list-table} Wall regions of the cross-section
:name: tab-tutorial-sets
:header-rows: 2

* - Layup
  - Region
  - Total thickness
  - Elements
* -
  -
  - m
  -
* - layup_0
  - Trailing-edge reinforcement
  - 0.03260055
  - 25
* - layup_1
  - Skin, foam core
  - 0.07711403
  - 161
* - layup_2
  - Upper spar cap
  - 0.08756579
  - 13
* - layup_3
  - Leading-edge reinforcement
  - 0.02808872
  - 8
* - layup_4
  - Lower spar cap
  - 0.0835804
  - 14
* - layup_5
  - Shear webs (all three)
  - 0.046
  - 89
```

Every ply angle in this file is $0^\circ$. The biaxial and triaxial glass fabrics are given as smeared orthotropic
materials rather than as individual plies, so the material frame of each ply coincides with the wall frame.

## Steps of preparing the input file

An OpenSG analysis reads one YAML file. For a 1-D shell SG the file holds a short analysis header followed by six
blocks, listed in {numref}`tab-tutorial-blocks` in the order they appear in `iea_s10_shell.yaml`.

% source: docs/tutorials/msg_shell_beam.md ("The YAML blocks"); counts checked in iea_s10_shell.yaml
```{list-table} Blocks of the input file
:name: tab-tutorial-blocks
:header-rows: 1

* - Block
  - Content
  - This station
* - header
  - `n_model`, `refined`, `msg`: the analysis the file requests
  - 3 keys
* - `nodes`
  - `[y2 y3 0]` contour coordinates, m
  - 307 nodes
* - `elements`
  - 1-based two-node line connectivity
  - 310 elements
* - `sets: element`
  - named element sets, one per wall region
  - 6 sets
* - `sections`
  - per set: `elementSet` plus `layup` as `[material, thickness, angle]` rows, outer ply first
  - 6 sections
* - `elementOrientations`
  - per element, the flattened $3\times3$ frame $[e_1\;e_2\;e_3]$
  - 310 rows
* - `materials`
  - `name`, `density`, `elastic: {E, G, nu}` as 3-vectors
  - 6 materials
```

Two decisions come before the blocks are written. The first is the decomposition of the walls into regions of
uniform laminate: each region becomes one element set with one section, six in this example. The second is which
surface of the wall the node contour represents: the outer mold line (OML) or the laminate mid-surface. For a real
blade both decisions are made by the generator that writes the file from the blade definition (see
{doc}`../examples/blade_windio_pipeline`); `iea_s10_shell.yaml` was generated with the contour on the laminate
mid-surface.

### The analysis header

The first lines of the file state what the analysis should produce.

% source: examples/OpenSG_shell/1_get_beam_props_from_shell_cross_section/iea_s10_shell.yaml (lines 1 to 3)
```yaml
n_model: 1      # 1 = beam, 2 = plate (opensg_solid), 3 = solid -- the macro model this SG homogenizes to
refined: 1      # 0 = classical (beam EB 4x4, KL wall); 1 = shear-refined (beam Timoshenko 6x6, RM wall)
msg: shell      # the ENGINE this SG belongs to (opensg_shell); `opensg <yaml>` dispatches on it
```

`n_model: 1` asks for a beam model and `refined: 1` for its shear-refined form, so the walls are RM plates and the
result is the Timoshenko $6\times6$. `msg: shell` names the engine, `opensg_shell`, that the `opensg` command hands
the file to. Every header key has a default; {doc}`../user_guide/analysis_settings` lists them all.

### Nodes

% source: examples/OpenSG_shell/1_get_beam_props_from_shell_cross_section/iea_s10_shell.yaml (nodes block)
```yaml
nodes:
- [4.70852238 0.09722431 0.00000000]
- [4.64563825 0.11238283 0.00000000]
# ... 307 rows in all
```

Each row is one node, written as a bracketed, space-separated triple. The first two components are the cross-section
coordinates $(y_2, y_3)$ in metres; the third is the beam-axis coordinate, which is zero for a plane cross-section.

### Elements

% source: examples/OpenSG_shell/1_get_beam_props_from_shell_cross_section/iea_s10_shell.yaml (elements block)
```yaml
elements:
- [1 2]
- [2 3]
# ...
- [307 65]
```

Each row lists the two node ids of a line element, 1-based. Element $k$ in this list is element label $k$ in the
sets, and row $k$ of `elementOrientations` belongs to it. Walls meet by sharing a node: the last element, `[307 65]`,
is the end of a shear web and closes on node 65 of the upper spar cap.

### Element sets and sections

% source: examples/OpenSG_shell/1_get_beam_props_from_shell_cross_section/iea_s10_shell.yaml (sets and sections blocks)
```yaml
sets:
  element:
  - name: layup_2
    labels:
    - 64
    - 65
    # ... 13 labels in all, 64 to 76
sections:
- type: shell
  elementSet: layup_2
  layup:
  - - gelcoat
    - 0.0005
    - 0.0
  - - glass_triax
    - 0.00300294
    - 0.0
  - - carbon_uniax
    - 0.08105991
    - 0.0
  - - glass_triax
    - 0.00300294
    - 0.0
```

`sets` groups element labels by name, and `sections` gives each named group its laminate through the `elementSet`
field, which must equal a set `name`. Each `layup` row is `[material, thickness, angle]`: a material name defined in
`materials`, a ply thickness in metres, and a ply angle in degrees. The rows are ordered from the outer face inward,
and the laminate stacks along the wall normal $e_3$. Every element should appear in exactly one set: an element that
is in no set falls silently to the first section.

### Element orientations

% source: examples/OpenSG_shell/1_get_beam_props_from_shell_cross_section/iea_s10_shell.yaml (elementOrientations block)
```yaml
elementOrientations:
- [0.0, 0.0, 1.0, -0.9721541292349604, 0.2343423756204075, 0.0, -0.2343423756204075,
  -0.9721541292349604, 0.0]
```

Each row holds the nine direction cosines of the element frame, $[e_1\;e_2\;e_3]$, in the component order of the node
rows. $e_1$ is the beam axis, `[0, 0, 1]`, out of the plane of the section. $e_2$ is the wall tangent along the
contour, the in-plane ply direction. $e_3 = e_1 \times e_2$ is the wall normal, the direction in which the layup
stacks. The ply angle of the `layup` rows rotates the material about $e_3$, away from $e_1$, so an angle of
$0^\circ$ places the fibre along the beam axis. {doc}`../user_guide/coordinate_systems` gives the full convention.

```{figure} ../_static/shell_xsec_orient.png
:width: 100%
:name: fig-tutorial-xsec-orient

Per-element material axes of the cross-section: $e_2$ in blue, $e_3$ in green.
```

Plot the frames before trusting any result. {numref}`fig-tutorial-xsec-orient` is drawn by
`opensg_shell.auto_emit`, which the example driver calls. On this station the green $e_3$ arrows point into the
section everywhere on the skin, and the blue $e_2$ arrows run consistently around the closed skin and along each web.
An $e_3$ that flips between neighbouring elements means the layup is stacked in two directions at once, which leaves
the run complete but the stiffness wrong.

### Materials

% source: examples/OpenSG_shell/1_get_beam_props_from_shell_cross_section/iea_s10_shell.yaml (materials block)
```yaml
materials:
- name: carbon_uniax
  density: 1600.0
  elastic:
    E:
    - 140000000000.0
    - 9100000000.0
    - 9100000000.0
    G:
    - 4131000000.0
    - 4131000000.0
    - 2689000000.0
    nu:
    - 0.3133
    - 0.3133
    - 0.47066
```

Each material has a `name`, a `density` and an `elastic` block with three triples: $[E_1, E_2, E_3]$,
$[G_{12}, G_{13}, G_{23}]$ and $[\nu_{12}, \nu_{13}, \nu_{23}]$. Units follow the input; this file is in SI units (Pa,
kg/m³, m). The density feeds the mass per unit area of each wall and is not needed for the stiffness. The other
material formats are described in {doc}`../user_guide/materials_and_layups`.

### Laminate reference surface

The surface the contour represents is chosen on the command line, not in the file, and it applies the same way to
the homogenization and to the recovery.

% source: docs/input_format.md ("The laminate reference"); src/opensg_shell/sg_reference.py
| Run | Meaning | Fraction of thickness from the outer face |
|---|---|---|
| `opensg <yaml>` | the contour is the OML; every laminate stacks inward from it | 0.0 |
| `opensg <yaml> --center` | the contour is the laminate mid-surface; the wall law is shifted by $t/2$ and the recovery depths run from $-t/2$ to $+t/2$ about the contour | 0.5 |

OpenSG never moves the nodes, so a mid-surface contour such as this one must be run with `--center`. Running it
without the flag shifts the wall coupling and bending terms by a parallel-axis term of the wall thickness and changes
every bending and torsion entry of the beam matrix. The file still ends with a line `reference: center`; that key is
ignored, and the run prints a note saying so.

## Execution and results

### Homogenization

Once the input file is prepared, the cross-section is homogenized with one command:

% source: docs/tutorials/msg_shell_beam.md ("Run it", example 1)
```bash
cd examples/OpenSG_shell/1_get_beam_props_from_shell_cross_section
opensg iea_s10_shell.yaml --center
```

The run prints what it resolved from the header, then the law, then where it was stored:

% source: docs/index.md, section "Running OpenSG" (repository version at commit 0ef5e43)
```text
 ============================================================================
 OpenSG -- a multiscale structural analysis tool based on the Mechanics of
 Structure Genome (MSG), developed by the Multiscale Structural Mechanics
 Group led by Prof. Wenbin Yu at Purdue University
 ============================================================================

 input     : /.../1_get_beam_props_from_shell_cross_section/iea_s10_shell.yaml
 msg       : shell
 SG dim    : 2D
 analysis  : homogenization
 macro model: beam, shear-refined

Timoshenko Beam Stiffness Matrix  [eps11 gam12 gam13 kappa1 kappa2 kappa3]:
[[ 2.76606e+10 -1.72307e-05  4.66505e-06  5.40202e-06  1.66864e+09 -2.07730e+09]
 ...
Homogenization stored in iea_s10_shell_Timo.out
Time taken: 2.58 sec
```

`SG dim` is the dimension of the space the SG occupies, so a cross-section contour prints `2D`. Every run also prints a
`reference :` line after `macro model`; with `--center` it reads

% source: src/opensg_shell/cli.py (run banner) and src/opensg_shell/sg_reference.py (describe)
```text
 reference : center (contour = laminate mid-surface; ABD shifted by t/2, recovery depths -t/2..+t/2)
```

% source: examples/OpenSG_shell/1_get_beam_props_from_shell_cross_section/iea_s10_shell_Timo.out; src/opensg_shell/cli.py (run banner) and src/opensg_shell/sg_reference.py (reference_note)
Because the file still ends with `reference: center`, a `note` line follows it, saying that the key is ignored and
agrees with this run. The time depends on the machine; the `Time taken` footer of
the shipped `iea_s10_shell_Timo.out` reads 2.82 sec. The run writes the files of {numref}`tab-tutorial-homo-files`.

% source: docs/tutorials/msg_shell_beam.md ("What comes out", example 1); src/opensg/cli.py docstring (--mesh)
```{list-table} Files written by the homogenization
:name: tab-tutorial-homo-files
:header-rows: 1

* - File
  - Content
* - `iea_s10_shell_Timo.out`
  - the effective Timoshenko stiffness and compliance matrices, in the SwiftComp `.K` layout, with a `Time taken` footer
* - `iea_s10_shell_ABDG.out`
  - one $8\times8$ RM plate stiffness and compliance block per section, each headed by its layup
* - `abd/iea_s10_shell_abd.yaml`
  - a per-layup cache of the $8\times8$ wall law with the ply list, thickness and mass per area
* - `iea_s10_shell_mesh.png`, `iea_s10_shell.msh`
  - with `--mesh` only: the contour coloured by section, and the gmsh twin of the mesh
```

% source: src/opensg_shell/sg_homo.py (build_rm_bundle, abd cache); examples/OpenSG_shell/1_get_beam_props_from_shell_cross_section/abd/iea_s10_shell_abd.yaml
:::{note}
The `abd/` cache is reused when it already exists at the reference of the run (its `reference:` line reads `mid`
after a `--center` run). Delete `abd/iea_s10_shell_abd.yaml` after changing the layups so that it is rebuilt.
:::

The same homogenization is available from Python. The `ref=` argument is the Python form of the reference choice:
`ref="center"` corresponds to `--center`, and the default `ref="oml"` to a run without the flag.

% source: docs/tutorials/msg_shell_beam.md ("The one call")
```python
import numpy as np
from opensg_shell import build_rm_bundle, auto_emit

B = build_rm_bundle("iea_s10_shell.yaml", ref="center")   # RM 6-DOF ring + MSG wall G
C6 = np.asarray(B["Timo"])                  # (6, 6) Timoshenko stiffness
print(B["ref"], B["g_source"])              # "center msg"
print(np.diag(C6))
auto_emit("iea_s10_shell.yaml", out_png="iea_s10_orient.png")
```

The example folder also contains the driver `beam_homo_shell.py` (`python beam_homo_shell.py`), which calls
`build_rm_bundle` and writes the two figures of this page, `iea_s10_mesh.png` and `iea_s10_orient.png`.

% source: examples/OpenSG_shell/1_get_beam_props_from_shell_cross_section/beam_homo_shell.py; src/opensg_shell/sg_homo.py (build_rm_bundle, ref="oml")
:::{note}
As shipped, both example drivers call `build_rm_bundle(name + "_shell.yaml")` without `ref=`, so they build at the
default OML reference. They then rewrite `iea_s10_shell_Timo.out`, `iea_s10_shell_ABDG.out` and
`abd/iea_s10_shell_abd.yaml` with values that differ from {numref}`tab-tutorial-timo`. To reproduce the numbers on
this page, change that call to `build_rm_bundle(name + "_shell.yaml", ref="center")` in `beam_homo_shell.py` and
`beam_dehom_shell.py`.
:::

### Effective Timoshenko stiffness

The effective Timoshenko stiffness matrix is listed in {numref}`tab-tutorial-timo`, on the strain vector
$[\varepsilon_{11}\;\gamma_{12}\;\gamma_{13}\;\kappa_1\;\kappa_2\;\kappa_3]$: extension, the two transverse shears,
twist, and the two bending curvatures.

% source: examples/OpenSG_shell/1_get_beam_props_from_shell_cross_section/iea_s10_shell_Timo.out
```{list-table} Timoshenko stiffness matrix in the file iea_s10_shell_Timo.out (N, N·m, N·m²)
:name: tab-tutorial-timo
:header-rows: 1
:stub-columns: 1

* -
  - $\varepsilon_{11}$
  - $\gamma_{12}$
  - $\gamma_{13}$
  - $\kappa_1$
  - $\kappa_2$
  - $\kappa_3$
* - $\varepsilon_{11}$
  - 2.7660599E+010
  - -1.7230658E-005
  - 4.6650498E-006
  - 5.4020232E-006
  - 1.6686413E+009
  - -2.0773040E+009
* - $\gamma_{12}$
  - -1.7230658E-005
  - 7.1861326E+008
  - 5.7520370E+007
  - -1.0192044E+008
  - 2.3079953E-006
  - 5.2738749E-006
* - $\gamma_{13}$
  - 4.6650498E-006
  - 5.7520370E+007
  - 4.2174428E+008
  - -1.6217837E+008
  - 8.4816767E-006
  - 2.7931393E-007
* - $\kappa_1$
  - 5.4020232E-006
  - -1.0192044E+008
  - -1.6217837E+008
  - 2.4375653E+009
  - 1.6449557E-006
  - -8.1140553E-006
* - $\kappa_2$
  - 1.6686413E+009
  - 2.3079953E-006
  - 8.4816767E-006
  - 1.6449557E-006
  - 3.5143887E+010
  - -5.5571249E+009
* - $\kappa_3$
  - -2.0773040E+009
  - 5.2738749E-006
  - 2.7931393E-007
  - -8.1140553E-006
  - -5.5571249E+009
  - 6.8129724E+010
```

The diagonal follows the VABS ordering of {numref}`tab-tutorial-diag`.

% source: examples/OpenSG_shell/1_get_beam_props_from_shell_cross_section/iea_s10_shell_Timo.out
```{list-table} Diagonal stiffness terms
:name: tab-tutorial-diag
:header-rows: 1

* - Term
  - Entry
  - Value
  - Units
  - Meaning
* - $EA$
  - $C_{11}$
  - 2.7660599E+010
  - N
  - axial stiffness
* - $GA_2$
  - $C_{22}$
  - 7.1861326E+008
  - N
  - transverse shear, chordwise
* - $GA_3$
  - $C_{33}$
  - 4.2174428E+008
  - N
  - transverse shear, flapwise
* - $GJ$
  - $C_{44}$
  - 2.4375653E+009
  - N·m²
  - torsional stiffness about the mid-surface reference
* - $EI_2$
  - $C_{55}$
  - 3.5143887E+010
  - N·m²
  - flapwise bending
* - $EI_3$
  - $C_{66}$
  - 6.8129724E+010
  - N·m²
  - edgewise bending
```

% source: examples/OpenSG_shell/1_get_beam_props_from_shell_cross_section/iea_s10_shell_Timo.out
The off-diagonal terms carry the couplings of the section. $C_{15}$ = 1.6686413E+009 N·m and $C_{16}$ =
-2.0773040E+009 N·m are extension-bending couplings: the tension centre is away from the reference point.
$C_{56}$ = -5.5571249E+009 N·m² is the flap-edge bending coupling, so the principal bending axes are not the
$y_2$ and $y_3$ axes. $C_{24}$ = -1.0192044E+008 N·m and $C_{34}$ = -1.6217837E+008 N·m are the shear-twist
couplings that place the shear centre. The entries printed near $10^{-5}$ against diagonals near $10^{10}$ are
zeros to about fifteen digits: extension couples neither to transverse shear nor to twist on this section.

% source: examples/OpenSG_shell/1_get_beam_props_from_shell_cross_section/iea_s10_shell_ABDG.out (sections 1 and 2)
Each wall law in `iea_s10_shell_ABDG.out` is an $8\times8$ RM plate matrix whose last two rows hold the
transverse-shear block $G$. The foam-cored skin `layup_1` has $G$ diagonal terms 4.1457954E+006 and 4.1451801E+006
N/m, against 2.8674578E+008 and 2.1272143E+008 N/m for the carbon cap `layup_2`, so the soft core dominates the
shear compliance of the section.

### Recovery of the 3-D stress

The recovery takes the internal forces of the beam at this station and reconstructs the stress at every point of
every wall. Its input files are in the second example folder: the same `iea_s10_shell.yaml`, and the macro state
`iea_s10_shell.ff` next to it.

% source: examples/OpenSG_shell/2_get_beam_dehom_from_shell_cross_section/iea_s10_shell.ff
```yaml
# OpenSG dehomogenization macro state -- read by `opensg_shell iea_s10_shell.yaml D`
# The SG yaml describes the STRUCTURE; this file carries the LOADING.
u:     [0.0, 0.0, 0.0]      # macro displacement u1 u2 u3
theta: [0.0, 0.0, 0.0]      # macro rotation t1 t2 t3 [rad] (about y1 y2 y3)
C:     [[1.0, 0.0, 0.0],       # direction cosines of the macro frame
        [0.0, 1.0, 0.0],       #   (identity = unrotated; overrides theta)
        [0.0, 0.0, 1.0]]
# generalized macro forces, beam [F1 F2 F3 M1 M2 M3]
# IEA-22 station r/R = 0.20 (ff51_rmc_reform.dat row 10, RM center-ref BeamDyn)
FF:    [4.19268867e+04, 8.26739746e+03, 1.46733512e+06, 1.21914500e+06, -6.01849720e+07, 3.04264594e+05]
```

% source: examples/OpenSG_shell/2_get_beam_dehom_from_shell_cross_section/iea_s10_shell.ff and ff51_rmc_reform.dat
`FF` holds the beam forces and moments $[F_1\;F_2\;F_3\;M_1\;M_2\;M_3]$ in N and N·m, in the same order as the
stiffness matrix. They are row 10 of the spanwise table `ff51_rmc_reform.dat`, $\eta = r/R = 0.20$, and are referred
to the mid-surface reference, matching the `--center` homogenization. The load is dominated by the flapwise moment
$M_2$ = -6.01849720e+07 N·m and the flapwise shear $F_3$ = 1.46733512e+06 N. When there is no `.ff` file, the run
takes the macro strain from an `epsilon_bar:` key in the header instead.

The recovery is run with the `D` argument:

% source: docs/tutorials/msg_shell_beam.md ("Run it", example 2)
```bash
cd examples/OpenSG_shell/2_get_beam_dehom_from_shell_cross_section
opensg iea_s10_shell.yaml D --center
```

% source: docs/tutorials/msg_shell_beam.md ("The two-step chain"); examples/OpenSG_shell/2_get_beam_dehom_from_shell_cross_section/beam_dehom_shell.py
The `D` run homogenizes the section and then recovers the local fields in two steps. Along the contour, the macro beam
strain $C^{-1}\,FF$ gives the six shell strains of every element. Through the thickness, an RM plate SG per layup turns
those shell strains into the 3-D stress at nine depths per element, at $\zeta = (k + \tfrac12)/9$ with $\zeta = 0$
at the OML and $\zeta = 1$ at the inner mold line, giving $310 \times 9 = 2790$ recovery points. The stresses are
reported in the ply axes by default (`--material`); `--global` reports them in the wall axes. The console prints a
census of the wall junctions:

% source: docs/tutorials/msg_shell_beam.md ("Junctions in the recovery")
```text
 junctions : 97 detected (5 T, 91 L, 1 unresolved)
 stations  : 2790 total, 1032 near, 56 overlapped (50 duplicates)
```

and closes with `Local field files are computed and stored.` and one `Time taken` line. The files are listed in
{numref}`tab-tutorial-dehom-files`.

% source: src/opensg_shell/cli.py (dehom_write); docs/tutorials/msg_shell_beam.md ("What comes out", example 2)
```{list-table} Files written by the recovery
:name: tab-tutorial-dehom-files
:header-rows: 1

* - File
  - Content
* - `iea_s10_shell_dehom.txt`
  - 2790 rows, one per recovery point: `elem`, `zeta`, `y2`, `y3`, `z` from the reference surface, the six stresses `S11 S22 S33 S23 S13 S12` (Pa), the six strains `E11 E22 E33 2E23 2E13 2E12`, the displacements `u1 u2 u3` (m), and the junction columns `jflag` and `jdist`
* - `iea_s10_shell_dehom.vtk`
  - the recovery mesh, one quad per element and depth band, each carrying its own stress as cell data
* - `iea_s10_shell_dehom.junc`
  - every junction with its coordinates, topology, overlap area and the wall that owns it
```

The census and the two last columns come from the `junction:` key of the header, whose default `flag` marks the
recovery points where the columns of two crossing walls cover the same material, without changing any stress value.
{doc}`../user_guide/recovery` describes the three settings of this key.

```{figure} ../_static/shell_dehom_stress.png
:width: 90%
:name: fig-tutorial-dehom-stress

Recovered $\sigma_{11}$ (top) and $\sigma_{12}$ (bottom) in MPa at the 2790 recovery points of the cross-section.
```

{numref}`fig-tutorial-dehom-stress` is drawn by the driver `beam_dehom_shell.py` of the same folder, which runs the
two steps through the Python API, writes the stresses in MPa to `iea_s10_dehom_shell.txt` and, at the mid-surface
reference (`ref="center"` in its `build_rm_bundle` call), reports the peak as

% source: docs/tutorials/msg_shell_beam.md ("What comes out", example 2)
```text
peak |sigma11| = 318.18 MPa at (y2, y3) = (-0.677, 1.311), zeta = 0.06  [material frame]
```

% source: docs/tutorials/msg_shell_beam.md ("Reading the result")
The points sit slightly off the contour because each one is placed at its own depth $z$ along the wall normal. The
axial stress is confined almost entirely to the two spar caps, as a flap-dominated load requires
({numref}`tab-tutorial-extremes`): the upper cap is in compression and the lower cap in tension, at nearly equal
magnitude, and both extremes fall at the outermost station $\zeta = 0.056$, just inside the carbon ply. The in-plane
shear peaks in the trailing-edge and leading-edge regions, where the closed cell carries the torque as shear flow.

% source: docs/tutorials/msg_shell_beam.md ("Reading the result")
```{list-table} Extreme recovered stresses (material frame)
:name: tab-tutorial-extremes
:header-rows: 1

* - Quantity
  - Value (MPa)
  - Element
  - Layup
  - $(y_2, y_3)$ (m)
  - $\zeta$
* - minimum $\sigma_{11}$
  - -318.177
  - 75
  - `layup_2`
  - (-0.677, 1.311)
  - 0.056
* - maximum $\sigma_{11}$
  - +314.461
  - 150
  - `layup_4`
  - (-0.145, -1.249)
  - 0.056
* - maximum $\lvert\sigma_{12}\rvert$
  - 5.573
  - 11
  - `layup_0`
  - (3.895, 0.335)
  -
```

:::{warning}
Two stress components are not delivered by this recovery. $\sigma_{33}$ is zero to machine precision in the
first-order recovery, and $\sigma_{13}$ is not a valid recovery target on a cross-section ring, because its
transverse-shear row is untied by construction. A physical $\sigma_{13}$ needs a segment or 3-D shell SG.
:::

The comparison of this recovery with a VABS 2-D solid model of the same section, and the junction settings in
detail, are in {doc}`../examples/shell_beam_recovery`. The homogenization example on its own is
{doc}`../examples/shell_beam_cross_section`, and {doc}`running` lists every command line option.
