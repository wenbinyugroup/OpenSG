# Sandwich plate stiffness from a through-thickness 1-D SG

## Problem description

% source: docs/tutorials/msg_solid_plate.md (section A)
This example computes the plate stiffness of a sandwich wall: graphite/epoxy face plies around a
thick PVC foam core, stacked as $[0/90/0/90/\text{core}]_s$ with a total thickness
$h = 0.1524$ m. Nothing varies in the plane of the wall, so the structure gene (SG) is
one-dimensional: a line mesh through the thickness with one 5-noded quartic element per ply
({numref}`fig-sp1-sg`). The engine is `opensg_solid`, through its through-thickness plate route
`rm_plate_msg`. The macro model is a Reissner-Mindlin (RM) plate: the run returns the
$8\times8$ ABDG law (the $6\times6$ ABD matrix (extension $A$, extension-bending coupling $B$ and
bending $D$ stiffness) plus the $2\times2$
transverse-shear block $G$), or the classical $6\times6$ ABD alone. The result is checked against
the section mass, the symmetry of the stack and a core-only estimate of the shear stiffness.

% source: docs/tutorials/msg_solid_plate.md (section A), examples/OpenSG-solid/1_get_plate_props_from_1DSG/layup_db.yaml
The stack is the $[0/90/0/90/\text{core}]_s$ sandwich of Nayak's Example 5. Each of the eight
graphite/epoxy plies is 0.9525 mm thick and the HEREX C70.130 PVC foam core is 144.78 mm thick.
The material properties are listed in {numref}`tab-sp1-materials` and the layup in
{numref}`tab-sp1-layup`, both copied from `layup_db.yaml`. Complete input files are in
`examples/OpenSG-solid/1_get_plate_props_from_1DSG/`.

```{figure} ../_static/plate_1dsg.png
:width: 360px
:name: fig-sp1-sg

The through-thickness 1-D SG of the sandwich: thin face plies at top and bottom, the foam core
between them, and the reference surface (dotted line) at mid-thickness.
```

% source: examples/OpenSG-solid/1_get_plate_props_from_1DSG/layup_db.yaml
```{list-table} Material properties
:header-rows: 2
:name: tab-sp1-materials

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
* - `ge` (graphite/epoxy face ply)
  - 1500.0
  - $128.0\times10^{9}$
  - $11.0\times10^{9}$
  - $11.0\times10^{9}$
  - $4.48\times10^{9}$
  - $4.48\times10^{9}$
  - $1.53\times10^{9}$
  - 0.25
  - 0.25
  - 0.25
* - `herex` (HEREX C70.130 PVC foam core)
  - 130.0
  - $103.63\times10^{6}$
  - $103.63\times10^{6}$
  - $103.63\times10^{6}$
  - $50.0\times10^{6}$
  - $50.0\times10^{6}$
  - $50.0\times10^{6}$
  - 0.32
  - 0.32
  - 0.32
```

% source: examples/OpenSG-solid/1_get_plate_props_from_1DSG/layup_db.yaml (layup:, bottom ply first)
```{list-table} Layup, bottom ply first
:header-rows: 2
:name: tab-sp1-layup

* - Ply
  - Material
  - Thickness
  - Angle
* -
  -
  - m
  - °
* - 1
  - `ge`
  - 0.0009525
  - 0.0
* - 2
  - `ge`
  - 0.0009525
  - 90.0
* - 3
  - `ge`
  - 0.0009525
  - 0.0
* - 4
  - `ge`
  - 0.0009525
  - 90.0
* - 5
  - `herex`
  - 0.1447800
  - 0.0
* - 6
  - `ge`
  - 0.0009525
  - 90.0
* - 7
  - `ge`
  - 0.0009525
  - 0.0
* - 8
  - `ge`
  - 0.0009525
  - 90.0
* - 9
  - `ge`
  - 0.0009525
  - 0.0
```

The SG solves the three warping (fluctuation) displacements at its nodes. Minimizing the 3-D
strain energy over the line mesh order by order gives the zeroth-order warping $V_0$, and the
plate stiffness follows as

$$A_6 = \big\langle(\Gamma_\varepsilon + \Gamma_h V_0)^{\mathsf T}\,C(z)\,
(\Gamma_\varepsilon + \Gamma_h V_0)\big\rangle ,$$

where $\langle\cdot\rangle$ is the through-thickness integral and $C(z)$ the ply stiffness
rotated by its fiber angle. With `model: 1` the run also identifies the $2\times2$
transverse-shear block of the mechanics of structure genome (MSG) $G$ from the second-order energy and assembles

$$\text{ABDG} = \begin{bmatrix} A_6 & 0 \\ 0 & G \end{bmatrix},$$

with the transverse shears uncoupled from the membrane and bending measures.

## Input

The whole problem lives in `layup_db.yaml`. The driver `1d_sg.py` reads that file and nothing
else, so a new laminate means editing the yaml, never the code ({numref}`tab-sp1-keys`).

% source: docs/tutorials/msg_solid_plate.md (section A, "The input you edit"), examples/OpenSG-solid/1_get_plate_props_from_1DSG/1d_sg.py
```{list-table} Keys of layup_db.yaml read by 1d_sg.py
:header-rows: 1
:name: tab-sp1-keys

* - Key
  - Meaning
  - Value in the shipped file
* - `model`
  - `0` = classical $6\times6$ ABD only, `1` = $8\times8$ ABD plus transverse shear
  - `1`
* - `fraction`
  - reference surface as a fraction of the thickness: `0` = bottom face, `0.5` = mid-surface, `1` = top
  - `0.5`
* - `mesh: n_per_layer`
  - SG elements per physical ply
  - `1`
* - `mesh: elem_order`
  - element order; `4` = 5-noded quartic
  - `4`
* - `materials`
  - per material `E: [E1, E2, E3]`, `G: [G12, G13, G23]`, `nu: [nu12, nu13, nu23]`, `rho`
  - `ge`, `herex`
* - `layup`
  - one entry per ply, bottom ply first: `{material, thickness, angle}`
  - 9 plies
```

The corresponding lines of the file (excerpt):

% source: examples/OpenSG-solid/1_get_plate_props_from_1DSG/layup_db.yaml
```yaml
model: 1                  # 0 = ABD only, 1 = ABD + transverse shear (8x8)
fraction: 0.5             # reference surface as a fraction of thickness:
                          #   0 = bottom/OML face, 0.5 = mid-surface, 1 = top

mesh:
  n_per_layer: 1          # SG elements per physical ply
  elem_order: 4           # 4 = 5-noded quartic (exact for the warping ladder)

materials:
  ge:
    full_name: graphite/epoxy face ply
    E:  [128.0e9, 11.0e9, 11.0e9]
    G:  [4.48e9, 4.48e9, 1.53e9]
    nu: [0.25, 0.25, 0.25]
    rho: 1500.0
  herex:
    full_name: HEREX C70.130 PVC foam core
    E:  [103.63e6, 103.63e6, 103.63e6]
    G:  [50.0e6, 50.0e6, 50.0e6]
    nu: [0.32, 0.32, 0.32]
    rho: 130.0

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

The `divisions` key of the core ply is not read by `1d_sg.py`: the SG's own through-thickness
discretization is the `mesh:` block. The same file also carries `V2`, a `plate:` block and a
`dehom:` block, which the recovery scripts of {doc}`solid_plate_1d_recovery` read.

:::{note}
`fraction` is a property of the mesh, not of the homogenization call. The generated node
coordinates are measured from the reference plane, so they run from $-\text{fraction}\cdot h$ to
$(1-\text{fraction})\cdot h$ and $x_3 = 0$ is the reference. The $B$ block of the ABD depends on
this choice, so homogenize, analyze and recover with the same `fraction`.
:::

## Run

```bash
cd examples/OpenSG-solid/1_get_plate_props_from_1DSG
python 1d_sg.py
```

A different database file can be passed as the single positional argument:

```bash
python 1d_sg.py my_other_layup.yaml
```

The script writes the SG mesh `1dsg.yaml` and its figure `1dsg.png`, homogenizes the SG and
writes the law to `<layup_db stem>_plate_homo.out`. Its last console line names the three files:

% source: examples/OpenSG-solid/1_get_plate_props_from_1DSG/1d_sg.py (final print)
```text
layup_db.yaml + 1dsg.yaml  ->  layup_db_plate_homo.out
```

% source: examples/OpenSG-solid/1_get_plate_props_from_1DSG/1dsg.yaml, docs/tutorials/msg_solid_plate.md (section A, "What comes out")
`1dsg.yaml` has an `sg:` header (`type: plate_1d`, `elem_order: 4`, `n_per_layer: 1`,
`reference_fraction: 0.5`, `thickness`, `n_ply: 9`), 37 nodes given as single $x_3$ coordinates
from $-0.0762$ to $+0.0762$ m, 9 five-noded elements, the material blocks with their densities,
and one element set plus one `sections:` entry per ply carrying its material, angle and thickness.

## Result

% source: examples/OpenSG-solid/1_get_plate_props_from_1DSG/layup_db_plate_homo.out
The `.out` file opens with a banner that records the run (plies, thickness, `fraction`, the
section mass $\rho h$ and the row order), then the matrix titled after `model`:
`Reissner-Mindlin` here, `Classical Plate` at `model: 0` ({numref}`lst-sp1-abdg`). The
effective compliance follows in the same layout, and the file closes with
`Time taken: 0.83 sec`. Engineering constants are not printed for a plate law.

% source: examples/OpenSG-solid/1_get_plate_props_from_1DSG/layup_db_plate_homo.out
```{code-block} text
:caption: RM plate stiffness from layup_db_plate_homo.out, rows and columns [e11 e22 g12 k11 k22 k12 2g13 2g23] (A and G in N/m, B in N, D in N·m)
:name: lst-sp1-abdg

 OpenSG msg-solid plate model from 1dsg.yaml: 9 plies, h = 0.152400 m, fraction = 0.5, rho*h = 30.251400 kg/m^2, rows [e11 e22 g12 k11 k22 k12 2g13 2g23]

 The Effective Reissner-Mindlin Stiffness Matrix
 --------------------------------------------
     5.4916502E+008     2.6417019E+007     1.5524611E-010     3.6182957E-006     3.6122933E-006     1.0244011E-024     0.0000000E+000     0.0000000E+000
     2.6417019E+007     5.4916502E+008     2.7287693E-008     3.5815597E-006     3.5901363E-006     9.4990440E-025     0.0000000E+000     0.0000000E+000
     1.5524611E-010     2.7287693E-008     4.1376600E+007     3.7505596E-025     4.0252328E-025     2.7871828E-010     0.0000000E+000     0.0000000E+000
     3.6038602E-006     3.6011174E-006     3.4553341E-025     3.0005458E+006     1.2565963E+005     8.4612092E-013     0.0000000E+000     0.0000000E+000
     3.6094994E-006     3.6190073E-006     3.8414598E-025     1.2565963E+005     2.9371144E+006     1.4872313E-010     0.0000000E+000     0.0000000E+000
     1.0909630E-024     1.1200459E-024     2.7871828E-010     8.4612092E-013     1.4872313E-010     2.0111708E+005     0.0000000E+000     0.0000000E+000
     0.0000000E+000     0.0000000E+000     0.0000000E+000     0.0000000E+000     0.0000000E+000     0.0000000E+000     7.7009772E+006    -9.7394806E-009
     0.0000000E+000     0.0000000E+000     0.0000000E+000     0.0000000E+000     0.0000000E+000     0.0000000E+000    -9.7394806E-009     7.5423712E+006
```

The code is unit-agnostic and follows the input, which is in metres and pascals here
({numref}`tab-sp1-diag`).

% source: examples/OpenSG-solid/1_get_plate_props_from_1DSG/layup_db_plate_homo.out (diagonal); docs/tutorials/msg_solid_plate.md (section A)
```{list-table} Diagonal of the RM plate stiffness
:header-rows: 1
:name: tab-sp1-diag

* - Term
  - Value
  - Units
* - $A_{11}$
  - $5.4916502\times10^{8}$
  - N/m
* - $A_{22}$
  - $5.4916502\times10^{8}$
  - N/m
* - $A_{66}$
  - $4.1376600\times10^{7}$
  - N/m
* - $D_{11}$
  - $3.0005458\times10^{6}$
  - N·m
* - $D_{22}$
  - $2.9371144\times10^{6}$
  - N·m
* - $D_{66}$
  - $2.0111708\times10^{5}$
  - N·m
* - $G_{11}$
  - $7.7009772\times10^{6}$
  - N/m
* - $G_{22}$
  - $7.5423712\times10^{6}$
  - N/m
```

% source: docs/tutorials/msg_solid_plate.md (section A, "What the numbers mean")
Three checks are worth making on any run of your own, and all three are visible here:

- **The section mass.** The banner's $\rho h = 30.2514$ kg/m² is
  $8 \times 0.0009525 \times 1500 + 0.14478 \times 130 = 11.43 + 18.8214$ kg/m², so the layup that
  reached the solver is the layup in the file.
- **The $B$ block.** The stack is symmetric about the mid-surface and `fraction: 0.5` puts the
  reference plane there, so membrane-bending coupling must vanish. The largest entry of the $B$
  block is $3.6\times10^{-6}$ against an $A_{11}$ of $5.5\times10^{8}$, fourteen orders down.
- **The transverse-shear block.** $G_{11} = 7.70\times10^{6}$ N/m sits 6 % above the core-only
  estimate $G_{\rm core} h_{\rm core} = 50\times10^{6} \times 0.14478 = 7.24\times10^{6}$ N/m, the
  expected magnitude for a sandwich whose shear is carried almost entirely by the foam. $A_{11}$,
  by contrast, is set by the stiff faces.

% source: docs/tutorials/msg_solid_plate.md (section A, "What it is validated against")
The construction is gated by `tests/rm_plate_1D/test_msg_rm_plate.py`, on analytic anchors and on
an independent plain-NumPy re-assembly of the same SG matrices. For an isotropic plate with
$\nu = 0$ the transverse-shear block comes out exactly $\tfrac56 G h$, the textbook
shear-correction factor, with the least-squares residual $U^*$ at machine zero. $A_6$ equals the
classical lamination theory `msg_materials.compute_ABD_matrix` at the matching reference plane,
$A_6(\text{fraction})$ equals the offset transform of $A_6(0)$, and the warping fields satisfy the
Euler-Lagrange equations and the gauge condition $\psi^{\mathsf T} H V = 0$ for $V_0$, $V_{11}$
and $V_{12}$ directly. The formulation is that of Yu, Hodges and Volovoi, *Computers & Structures*
81 (2003) 439-454; {doc}`../theory/rm_plate_1d_user_guide` documents the function interface.

## Files

```{list-table} Files of the example
:header-rows: 1
:name: tab-sp1-files

* - File
  - Role
* - `layup_db.yaml`
  - input: `model`, `fraction`, `mesh`, materials with densities, layup
* - `1d_sg.py`
  - driver script
* - `1dsg.yaml`
  - written: the generated 1-D SG mesh, read back for the homogenization
* - `1dsg.png`
  - written: the mesh figure, plies colored by material
* - `layup_db_plate_homo.out`
  - written: the timed plate stiffness and compliance
```

The recovery of the ply stresses from this SG is {doc}`solid_plate_1d_recovery`; the layup
database format is described in {doc}`../user_guide/materials_and_layups`.
