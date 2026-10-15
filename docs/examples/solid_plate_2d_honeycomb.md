# Honeycomb sandwich plate from a 2-D SG

## Problem description

% source: docs/tutorials/msg_solid_plate.md (section C); examples/OpenSG-solid/3_get_plate_props_from_2D_SG/RHC_SW_2UC_45.yaml (nodes, cells)
When a plate wall has in-plane microstructure, such as a honeycomb core, a corrugation or a truss,
the structure gene (SG) is a 2-D mesh of one periodic cell. This page uses the two-unit-cell
honeycomb sandwich cell `RHC_SW_2UC_45` ({numref}`fig-shc-mesh`): $\pm45^\circ$ face laminae
(materials 1 and 2) as thin horizontal bands at $y_2 = \pm23$ mm, and an isotropic core web
(material 3) folding between them over the in-plane period. The through-thickness coordinate is
$y_2$, spanning $\pm23.35$ mm; the in-plane coordinate $y_1$ runs from $-17.624$ to $+17.624$ mm.
The mesh has 4 251 nodes and 6 640 triangles, in millimetres with moduli in MPa.

The engine is `opensg_solid`, through `opensg_solid.sg_homo.plate_homo_2d` with the plate macro
model (`n_model: 2`). Opposite edges and corners of the cell are tied through the sparse periodic
map; periodic is the default and the correct treatment for a periodic medium. Two examples share
this SG:

- **Example 3** (`examples/OpenSG-solid/3_get_plate_props_from_2D_SG/`) computes the plate law:
  the classical $6\times6$ ABD matrix (extension $A$, extension-bending coupling $B$ and bending
  $D$ stiffness) with `refined: 0`, or the $8\times8$ Reissner-Mindlin (RM) ABDG
  with `refined: 1`.
- **Example 4** (`examples/OpenSG-solid/4_get_plate_dehom_from_2DSG/`) recovers the local 3-D
  strain and stress at every Gauss point of the cell for a given macro plate strain.

The plate law is checked against the cell's own geometry (the measured period and the mid-plane
symmetry), and a 1-D stack shipped in the same folder runs through the identical call.

```{figure} ../_static/rhc_2dsg_mesh.png
:width: 480px
:name: fig-shc-mesh

The two-unit-cell honeycomb sandwich SG `RHC_SW_2UC_45`, elements colored by material.
```

% source: examples/OpenSG-solid/3_get_plate_props_from_2D_SG/RHC_SW_2UC_45.yaml (materials:), examples/OpenSG-solid/3_get_plate_props_from_2D_SG/mat_override.yaml (roles)
```{list-table} Material properties (engineering constants, type 1)
:header-rows: 2
:name: tab-shc-materials

* - Material
  - Role
  - $E_1$
  - $E_2$
  - $E_3$
  - $G_{12}$
  - $G_{13}$
  - $G_{23}$
  - $\nu_{12}$
  - $\nu_{13}$
  - $\nu_{23}$
  - Angle
* -
  -
  - MPa
  - MPa
  - MPa
  - MPa
  - MPa
  - MPa
  -
  -
  -
  - °
* - 1
  - face ply
  - 108000.0
  - 8000.0
  - 8000.0
  - 4000.0
  - 4000.0
  - 3000.0
  - 0.32
  - 0.32
  - 0.30
  - 45.0
* - 2
  - face ply
  - 108000.0
  - 8000.0
  - 8000.0
  - 4000.0
  - 4000.0
  - 3000.0
  - 0.32
  - 0.32
  - 0.30
  - -45.0
* - 3
  - core
  - 69000.0
  - 69000.0
  - 69000.0
  - 26540.0
  - 26540.0
  - 26540.0
  - 0.30
  - 0.30
  - 0.30
  - 0.0
```

% source: src/opensg_solid/sg_homo.py (plate_homo_2d: the mass matrix is built only for n_model 1)
The plate route of `plate_homo_2d` reports stiffness only; the SG yaml carries no densities.

## Input

### Example 3: the SG yaml

The yaml is the whole problem, and every header key has a default. The header is the leading
scalar keys above the mesh blocks:

% source: examples/OpenSG-solid/3_get_plate_props_from_2D_SG/RHC_SW_2UC_45.yaml
```yaml
n_model: 2      # 1 = beam, 2 = plate, 3 = solid -- the macro model this SG homogenizes to
refined: 1      # 0 = classical (plate ABD / beam EB); 1 = shear-refined (plate ABDG / beam Timoshenko)
msg: solid      # the ENGINE this SG belongs to (opensg_solid); `opensg <yaml>` dispatches on it
```

% source: docs/tutorials/msg_solid_plate.md (section C, "The input you edit")
% source: src/opensg_solid/sg_homo.py (plate_homo_2d docstring, refined), src/opensg_solid/cli.py (refined=int(hdr.get("refined", 0)))
`refined` upgrades the macro law within the chosen `n_model`: for the plate, `0` is the classical
$6\times6$ ABD and `1` the shear-refined $8\times8$ ABDG. `0` is the default when the key is
absent; the shipped file asks for `1`. There is no `dim` and no `scale` key: the SG dimension is
inferred from the mesh and the measure $\omega$ is computed from it. `analysis:` defaults to `H`;
with `analysis: D` the header also carries `epsilon_bar:`, the $(6,)$ macro state, and the run
recovers the local 3-D fields right after the homogenization. Explicit arguments to
`plate_homo_2d` always override the header. With neither a `refined` argument nor a `refined:`
key, the Python call defaults to the Timoshenko $6\times6$ for `n_model=1` and to the classical
ABD for the plate; the command line defaults to classical for every model. The header keys are described in
{doc}`../user_guide/analysis_settings`.

The materials live in the SG yaml itself, as in a SwiftComp `.sc`. Each entry of the `materials:`
block defines one material either by its $6\times6$ elastic stiffness (`type: 2`, stored
pre-rotated, used as-is) or by the nine orthotropic engineering constants
`[E1, E2, E3, G12, G13, G23, nu12, nu13, nu23]` with the ply angle (`type: 1` plus `angle:`,
rebuilt and rotated in-code). This example ships the second form:

% source: examples/OpenSG-solid/3_get_plate_props_from_2D_SG/RHC_SW_2UC_45.yaml
```yaml
materials:
  1:
    type: 1
    engineering: [108000.0, 8000.0, 8000.0, 4000.0, 4000.0, 3000.0, 0.32, 0.32, 0.30]
    angle: 45.0
  2:
    type: 1
    engineering: [108000.0, 8000.0, 8000.0, 4000.0, 4000.0, 3000.0, 0.32, 0.32, 0.30]
    angle: -45.0
  3:
    type: 1
    engineering: [69000.0, 69000.0, 69000.0, 26540.0, 26540.0, 26540.0, 0.30, 0.30, 0.30]
    angle: 0.0
```

% source: docs/tutorials/msg_solid_plate.md (section C)
The units are the mesh's: this cell is in millimetres with moduli in MPa, so $108\times10^{3}$ MPa
is a 108 GPa fiber direction. The `material_param=` and `angles=` arguments of `plate_homo_2d`
remain available as an override that takes precedence over the file's blocks; example 4 uses
that route.

The solver reads `<name>.yaml`. A SwiftComp `.sc` is converted once with the packaged helper
(see {doc}`../user_guide/converters` for the command-line form):

% source: docs/tutorials/msg_solid_plate.md (section C)
```python
from opensg_solid.io.sc_to_yaml import convert

convert("RHC_SW_2UC_45.sc")   # writes RHC_SW_2UC_45.yaml + .msh
```

### Example 4: the macro plate strain

`plate_dehom_2dsg.py` takes the SG name, its own `material_param` and `angles` (the override
route), and the macro plate strain in the order `[e11, e22, 2e12, k11, k22, 2k12]`:

% source: examples/OpenSG-solid/4_get_plate_dehom_from_2DSG/plate_dehom_2dsg.py
```python
############### User Input #################################
n_model = 2                    # 1: Beam; 2: Plate; 3: 3D elastic
name = "RHC_SW_2UC_45"         # reads <name>.yaml

material_param = jnp.array([
    (108e3, 8e3, 8e3, 4e3, 4e3, 3e3, 0.32, 0.32, 0.30),
    (108e3, 8e3, 8e3, 4e3, 4e3, 3e3, 0.32, 0.32, 0.30),
    (69e3, 69e3, 69e3, 26.54e3, 26.54e3, 26.54e3, 0.30, 0.30, 0.30)])
angles = jnp.array([45.0, -45.0, 0.0])

# the MACRO plate strain [e11, e22, 2e12, k11, k22, 2k12] -- YOURS, from
# the plate solution at the point of interest:
epsilon_bar = jnp.array([0.0, 0.1, 0.0, 0.1, 0.0, 0.0])
############################################################
```

% source: examples/OpenSG-solid/4_get_plate_dehom_from_2DSG/plate_dehom_2dsg.py; docs/tutorials/msg_solid_plate.md (section D)
The shipped state combines a transverse extension $\varepsilon_{22} = 0.1$ with a bending
curvature $\kappa_{11} = 0.1$. This vector comes from your plate solution at the station of
interest; there is no meaningful default. The `RHC_SW_2UC_45.yaml` of example 4 carries the same
mesh in the named-material dialect (`Al69`, `HC_ply`, `HC_ply_3`) with element sets and
per-element frames (`elementOrientations`). The script's `material_param` and `angles` replace
its material blocks, row by row in the order of the yaml's material list.

:::{warning}
Run the example-4 yaml only through `plate_dehom_2dsg.py`. Its own material blocks are not bound to the element sets the way example 3 assigns the +45 ply, the -45 ply and the core, so a run without the override does not reproduce the example-3 cell.
:::

For the command-line recovery, the macro state is a separate `.ff` file next to the yaml. The SG
yaml describes the structure, this file carries the loading:

% source: examples/OpenSG-solid/3_get_plate_props_from_2D_SG/RHC_SW_2UC_45.ff
```yaml
# OpenSG dehomogenization macro state -- read by `opensg_solid RHC_SW_2UC_45.yaml D`
# The SG yaml describes the STRUCTURE; this file carries the LOADING.
u:     [0.0, 0.0, 0.0]      # macro displacement u1 u2 u3
theta: [0.0, 0.0, 0.0]      # macro rotation t1 t2 t3 [rad] (about y1 y2 y3)
C:     [[1.0, 0.0, 0.0],       # direction cosines of the macro frame
        [0.0, 1.0, 0.0],       #   (identity = unrotated; overrides theta)
        [0.0, 0.0, 1.0]]
# generalized macro forces, plate [N11 N22 N12 M11 M22 M12]
FF:    [1000, 0, 0, 0, 0, 0]
```

## Run

### Homogenization

```bash
cd examples/OpenSG-solid/3_get_plate_props_from_2D_SG
opensg RHC_SW_2UC_45.yaml
```

% source: examples/OpenSG-solid/3_get_plate_props_from_2D_SG/plate_homo_2dsg.py (L30-32); examples/OpenSG-solid/4_get_plate_dehom_from_2DSG/plate_dehom_2dsg.py (L41)
The unified `opensg` command reads `msg: solid` from the header and forwards the file unchanged;
`opensg_solid RHC_SW_2UC_45.yaml` and `python -m opensg_solid RHC_SW_2UC_45.yaml` are the same
entry point. `python plate_homo_2dsg.py` makes the identical call through the Python interface,
one line, `plate_homo_2d(name + ".yaml")`. The returned dict holds the selected law in `r["law"]`
with its header in `r["law_title"]`, and the classical $6\times6$ ABD in `r["C_eff"]`. Both print the law selected by the header under its
title, then two closing lines:

% source: src/opensg_solid/cli.py (homogenization branch), src/opensg_solid/sg_homo.py (law_title)
```text
Reissner-Mindlin Stiffness Matrix  [N11 N22 N12 M11 M22 M12 Q1 Q2]:
<the 8x8 law>
Homogenization stored in RHC_SW_2UC_45.out
Time taken: <seconds> sec
```

The run writes `RHC_SW_2UC_45.out` and the mesh figure `RHC_SW_2UC_45_mesh.png`.

### Dehomogenization

```bash
cd examples/OpenSG-solid/4_get_plate_dehom_from_2DSG
python plate_dehom_2dsg.py
```

The script homogenizes the cell, writes the plate ABD, recovers the fields with `dehom_fields`
(using the warping columns $V_0$ the homogenization already solved; nothing is re-solved), and
prints the ABD diagonal and the peak of each stress component.

The same recovery runs from the command line on the example-3 yaml, with the macro state of its
`.ff` (the forces are inverted through the plate law):

```bash
cd examples/OpenSG-solid/3_get_plate_props_from_2D_SG
opensg_solid RHC_SW_2UC_45.yaml D
```

It writes `RHC_SW_2UC_45_dehom.txt`, `.vtk`, `.SM`, `.EM` and `.U`, and closes with
`Local field files are computed and stored.` and the `Time taken` line.

:::{note}
The command-line recovery writes stress and strain in each element's ply axes by default
(`--material`); `--global` keeps the SG axes. A yaml with per-element frames
(`elementOrientations`) refuses `--material`: pass `--global`.
`plate_dehom_2dsg.py` writes its fields in the SG axes.
:::

## Result

### Plate stiffness

% source: examples/OpenSG-solid/3_get_plate_props_from_2D_SG/RHC_SW_2UC_45.out
The `.out` banner records the SG measure and the boundary treatment. The shipped file asks for
`refined: 1`, so the law is the $8\times8$ ({numref}`lst-shc-abdg`); the compliance follows in
the same layout and the file closes with `Time taken: 1.78 sec`.

% source: examples/OpenSG-solid/3_get_plate_props_from_2D_SG/RHC_SW_2UC_45.out
```{code-block} text
:caption: RM plate stiffness from RHC_SW_2UC_45.out, rows and columns [N11 N22 N12 M11 M22 M12 Q1 Q2] (A and G in N/mm, B in N, D in N·mm)
:name: lst-shc-abdg

 OpenSG msg-solid plate model, omega 35.248001, periodic

 The Effective Reissner-Mindlin Stiffness Matrix
 --------------------------------------------
     3.3585973E+005     7.7500043E+004    -1.4366892E+002     4.1488406E-001     1.4861903E+000     3.6602857E-002     0.0000000E+000     0.0000000E+000
     7.7500043E+004     1.1291139E+005    -4.7913151E+002     1.3829469E+000     4.9539677E+000     1.2200953E-001     0.0000000E+000     0.0000000E+000
    -1.4366892E+002    -4.7913151E+002     1.0551654E+005     6.5307011E-002     2.1872937E-001     4.1192792E+000     0.0000000E+000     0.0000000E+000
     4.1488406E-001     1.3829469E+000     6.5307010E-002     9.0367521E+007     3.9540465E+007    -8.5182653E+004     0.0000000E+000     0.0000000E+000
     1.4861903E+000     4.9539677E+000     2.1872937E-001     3.9540465E+007     5.7011050E+007    -2.6597664E+005     0.0000000E+000     0.0000000E+000
     3.6602857E-002     1.2200953E-001     4.1192792E+000    -8.5182653E+004    -2.6597664E+005     4.6026320E+007     0.0000000E+000     0.0000000E+000
     0.0000000E+000     0.0000000E+000     0.0000000E+000     0.0000000E+000     0.0000000E+000     0.0000000E+000     2.6974186E+004     8.6527237E-001
     0.0000000E+000     0.0000000E+000     0.0000000E+000     0.0000000E+000     0.0000000E+000     0.0000000E+000     8.6527237E-001     9.7898064E+001
```

% source: examples/OpenSG-solid/3_get_plate_props_from_2D_SG/RHC_SW_2UC_45.out; docs/tutorials/msg_solid_plate.md (section C)
The leading $6\times6$ block is the classical ABD that `refined: 0` reports on its own, and the
two extra rows carry the transverse shears $2.6974186\times10^{4}$ and $9.7898064\times10^{1}$
({numref}`tab-shc-diag`).

% source: examples/OpenSG-solid/3_get_plate_props_from_2D_SG/RHC_SW_2UC_45.out (diagonal); docs/tutorials/msg_solid_plate.md (section C)
```{list-table} Diagonal of the RM plate stiffness of the honeycomb cell
:header-rows: 1
:name: tab-shc-diag

* - Term
  - Value
  - Units
* - $A_{11}$
  - $3.3585973\times10^{5}$
  - N/mm
* - $A_{22}$
  - $1.1291139\times10^{5}$
  - N/mm
* - $A_{66}$
  - $1.0551654\times10^{5}$
  - N/mm
* - $D_{11}$
  - $9.0367521\times10^{7}$
  - N·mm
* - $D_{22}$
  - $5.7011050\times10^{7}$
  - N·mm
* - $D_{66}$
  - $4.6026320\times10^{7}$
  - N·mm
* - $G_{11}$
  - $2.6974186\times10^{4}$
  - N/mm
* - $G_{22}$
  - $9.7898064\times10^{1}$
  - N/mm
```

% source: docs/tutorials/msg_solid_plate.md (section C, "What the numbers mean")
$\omega = 35.248001$ is the cell's in-plane period, and it is directly checkable against the mesh:
the nodes run from $y_1 = -17.624$ to $+17.624$ mm. The membrane law is strongly anisotropic,
$A_{11}/A_{22} = 2.97$, while the bending law is much closer to isotropic,
$D_{11}/D_{22} = 1.59$: the two directions see the cell very differently in stretching and much
less differently in bending. The largest entry of the whole $B$ block is $4.95$, against $A$ terms
of order $10^{5}$ and $D$ terms of order $10^{7}$, which is what a cell symmetric about its
mid-plane should give and is the first health check to make on a run of your own.

#### The Reissner-Mindlin option

Setting `refined: 1` in the yaml header, or passing `refined=1` to `plate_homo_2d`
(`shear_refined=True` is the older spelling, plate only), runs the RM first-order warping ladder
and returns `r["G_msg"]` (the $2\times2$ transverse-shear block), `r["ABDG"]` (the same
$8\times8$ block form as {doc}`solid_plate_1d_layup`) and `r["A6_ladder"]`. When the ladder
produces a positive-definite fit, the `.out` is written from the $8\times8$ and its matrices are
titled `Reissner-Mindlin`; otherwise the run falls back to the $6\times6$ titled
`Classical Plate`.

% source: docs/tutorials/msg_solid_plate.md (section C, "The Reissner-Mindlin option")
```python
from opensg_solid.sg_homo import plate_homo_2d

r = plate_homo_2d("RHC_SW_2UC_45.yaml", n_model=2, refined=1)
print(r["ABDG"])
```

#### A 1-D stack through the same call

% source: examples/OpenSG-solid/3_get_plate_props_from_2D_SG/Plate_1D_SG_2UC_45.yaml
The folder also ships `Plate_1D_SG_2UC_45.yaml`, a 1-D SG of two-node line elements through a
stack ({numref}`fig-shc-1d`). The SG dimension follows from the element node count; nothing
declares it. Materials 1 and 2 are given by their pre-rotated $6\times6$ stiffness (`type: 2`)
and material 3 is isotropic (`type: 0`, `E: 69000.0`, `nu: 0.3`). With `refined: 0` it runs
through the identical call by changing `name`, and `plate_homo_2d` names its report after the
input basename ({numref}`lst-shc-1d`). `plate_homo_2d` reads 1-D, 2-D and 3-D SGs from the same
code path, so a stack of plies can be homogenized either here or through the dedicated
through-thickness route of {doc}`solid_plate_1d_layup`.

% source: examples/OpenSG-solid/3_get_plate_props_from_2D_SG/Plate_1D_SG_2UC_45.yaml (mat_id, nodes)
```{figure} ../_static/plate_2dsg_alt_mesh.png
:width: 480px
:name: fig-shc-1d

The 1-D SG `Plate_1D_SG_2UC_45`: face plies (materials 1 and 2) at both ends of the core
(material 3, from $y_3 = -10$ to $10$), nodes as dots.
```

% source: examples/OpenSG-solid/3_get_plate_props_from_2D_SG/Plate_1D_SG_2UC_45.out
```{code-block} text
:caption: Classical plate stiffness from Plate_1D_SG_2UC_45.out, rows and columns [N11 N22 N12 M11 M22 M12] (A in N/mm, B in N, D in N·mm)
:name: lst-shc-1d

 OpenSG msg-solid plate model, omega 1, periodic

 The Effective Classical Plate Stiffness Matrix
 --------------------------------------------
     1.6545290E+006     5.6099053E+005    -9.0949470E-013     4.0163286E-009     3.6670826E-009    -5.0931703E-011
     5.6099053E+005     1.6545290E+006    -9.0949470E-013     3.6670826E-009     4.1327439E-009    -5.0931703E-011
     5.5610167E-013     5.5610167E-013     6.4249681E+005    -5.8562379E-011    -5.8562379E-011     3.8084653E-009
     3.6987426E-009     4.3390269E-009    -3.0356497E-011     7.8672694E+007     3.9405411E+007    -2.5191121E+004
     4.5765616E-009     5.5660919E-009    -2.8992255E-011     3.9405411E+007     7.8672694E+007    -2.5191121E+004
    -1.1180735E-011    -1.1180735E-011     3.8667533E-009    -2.5191121E+004    -2.5191121E+004     3.1248589E+007
```

`Plate_1D_SG_2UC_45_plate_ABD.out`, shipped beside it, is a bare $6\times6$ under a single comment
line (`np.savetxt` format), not the timed `.K` layout.

### Recovered fields

% source: docs/tutorials/msg_solid_plate.md (section D, "What comes out"), examples/OpenSG-solid/4_get_plate_dehom_from_2DSG/plate_dehom_2dsg.py
```{list-table} Output files of plate_dehom_2dsg.py
:header-rows: 1
:name: tab-shc-out

* - File
  - Content
* - `RHC_SW_2UC_45_plate_ABD.out`
  - the plain-text $6\times6$ ABD, header `plate 6x6 [N11 N22 N12 M11 M22 M12] from the 2D SG RHC_SW_2UC_45`
* - `RHC_SW_2UC_45_dehom.txt`
  - one row per Gauss point: `GP_Index`, `Coord_X`, `Coord_Y`, then `Eps_xx Eps_yy Eps_zz Eps_yz Eps_xz Eps_xy` and `Sig_xx` through `Sig_xy`, in SwiftComp component order
* - `RHC_SW_2UC_45_dehom.vtk`
  - ASCII `UNSTRUCTURED_GRID` point cloud of the same 19 920 points, each strain and stress component as its own `SCALARS` array
* - `RHC_SW_2UC_45_dehom.SM`, `.EM`, `.U`
  - the same fields in the layout of {doc}`solid_plate_1d_recovery`: `x y z` then `S11 S22 S33 S12 S13 S23` (stress), `E11` through `E23` (strain), `U1 U2 U3` (displacement)
* - `RHC_SW_2UC_45.out`, `RHC_SW_2UC_45_mesh.png`
  - the timed law and the mesh figure, from the `plate_homo_2d` call the script makes first
```

% source: examples/OpenSG-solid/4_get_plate_dehom_from_2DSG/RHC_SW_2UC_45_plate_ABD.out
The diagonal of the shipped `RHC_SW_2UC_45_plate_ABD.out`, $3.35859732\times10^{5}$,
$1.12911393\times10^{5}$, $1.05516537\times10^{5}$, $9.03675213\times10^{7}$,
$5.70110504\times10^{7}$ and $4.60263195\times10^{7}$, matches the first six entries of
{numref}`tab-shc-diag` to the printed digits. `RHC_SW_2UC_45.out` is regenerated on every run
rather than read from the folder, so its `Time taken` footer is that run's.

Two conventions matter when reading the files. In the `.SM`, `.EM` and `.U` tables the SG
coordinates fill the last slots of the `x y z` triple, so a 2-D SG leaves the `x` column at zero
and puts $y_1$ in `y` and the through-thickness $y_2$ in `z`. And `.U` is the fluctuation-only
displacement, as its own header says, because a unit-state SG recovery has no macro displacement
to add.

% source: docs/tutorials/msg_solid_plate.md (section D, "Reading the result")
The first Gauss point of the recovery sits at $(y_1, y_2) = (17.54067,\ 23.29375)$ mm, right at the
top face of the cell, whose half-thickness is $23.35$ mm, and reports

$$\varepsilon_{11} = 2.329375, \qquad
\sigma_{11} = 8.755003\times10^{4},\quad \sigma_{22} = 7.102971\times10^{4},\quad
\sigma_{12} = -6.542276\times10^{4}$$

with the stresses in MPa. That $\varepsilon_{11}$ is exactly $\kappa_{11} y_2 = 0.1 \times 23.29375$,
the macro part of the strain reproduced to every printed digit, which is the cheapest check that
the strain state passed in is the strain state that arrived. The other components carry the
response of the cell: $\varepsilon_{22} = 0.2643$ and $\varepsilon_{33} = -0.9187$, and a large
$\sigma_{12}$ even though the applied macro shear strain $2\varepsilon_{12}$ is zero, consistent
with the off-axis face laminae at that height. {numref}`fig-shc-sxx` shows $\sigma_{xx}$ over the
whole cell. Local states of this kind, in the faces and in the web, are what a smeared plate model
cannot show, and the reason to run a dehomogenization at all.

```{figure} ../_static/pv_plate_dehom_sxx.png
:width: 420px
:name: fig-shc-sxx

The recovered $\sigma_{xx}$ (MPa) at the Gauss points of the honeycomb SG, displayed in ParaView
from the dehomogenization point cloud.
```

## Files

```{list-table} Files of examples 3 and 4
:header-rows: 1
:name: tab-shc-files

* - File
  - Role
* - `3_.../RHC_SW_2UC_45.yaml`
  - input: the SG, header (`n_model`, `refined`, `msg`) and materials
* - `3_.../RHC_SW_2UC_45.sc`, `.msh`
  - the SwiftComp source of the SG and its gmsh twin
* - `3_.../RHC_SW_2UC_45.ff`
  - input: the macro force state for `opensg_solid RHC_SW_2UC_45.yaml D`
* - `3_.../plate_homo_2dsg.py`
  - driver script (the Python form of `opensg RHC_SW_2UC_45.yaml`)
* - `3_.../mat_override.yaml`
  - the engineering-constant override (`material_param`, `angles`) of the same three materials
* - `3_.../RHC_SW_2UC_45.out`
  - output: the timed RM plate stiffness and compliance
* - `3_.../Plate_1D_SG_2UC_45.yaml`, `.sc`, `.msh`, `.ff`
  - the 1-D stack SG and its sources
* - `3_.../Plate_1D_SG_2UC_45.out`, `Plate_1D_SG_2UC_45_plate_ABD.out`
  - output of the 1-D stack: the timed classical law and a bare $6\times6$
* - `4_.../RHC_SW_2UC_45.yaml`
  - input: the same cell in the named-material dialect with per-element frames
* - `4_.../plate_dehom_2dsg.py`
  - driver script with the `epsilon_bar` input
* - `4_.../RHC_SW_2UC_45_plate_ABD.out`
  - output: the plain-text $6\times6$ ABD
* - `4_.../RHC_SW_2UC_45_dehom.txt`, `.vtk`, `.SM`, `.EM`, `.U`
  - written by the script: the recovered fields
```

In the table, `3_...` stands for `examples/OpenSG-solid/3_get_plate_props_from_2D_SG/` and `4_...`
for `examples/OpenSG-solid/4_get_plate_dehom_from_2DSG/`. The recovery keys of the yaml header
are described in {doc}`../user_guide/recovery`.
