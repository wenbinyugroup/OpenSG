# Timoshenko beam properties of a blade cross-section

## Problem description

% source: docs/tutorials/msg_shell_beam.md ("What a 1-D shell SG is")
This example computes the Timoshenko beam stiffness of station $r/R = 0.20$ of the IEA-22 MW
reference wind-turbine blade from a one-dimensional shell structure gene (SG). The SG is the
contour of the cross-section: a chain of two-node line elements along the wall mid-surface, each
element carrying its laminate as material data. The wall thickness never appears in the mesh; it
lives inside the wall law. The section spans about 7.2 m in $y_2$ and 2.5 m in $y_3$, has spar caps
87.6 mm thick and three shear webs, and is described by 307 nodes and 310 elements
({numref}`fig-sbc-mesh`).

The engine is `opensg_shell` (`msg: shell` in the yaml header). Each wall is a Reissner-Mindlin
(RM) plate with an $8\times8$ ABDG law (extension, bending and transverse shear stiffness), and the
ring of walls is homogenized by the mechanics of structure genome (MSG) into the $6\times6$ Timoshenko beam stiffness (`n_model: 1`,
`refined: 1`). The 3-D stresses of the same station under a beam load, and their comparison with
a VABS 2-D solid model, are in {doc}`shell_beam_recovery`.

```{figure} ../_static/shell_xsec_mesh.png
:width: 95%
:name: fig-sbc-mesh

Cross-section contour of the IEA-22 station $r/R = 0.20$, coloured by layup.
```

The material properties are listed in {numref}`tab-sbc-materials` and the layups in
{numref}`tab-sbc-layups`. The biax and triax fabrics are smeared orthotropic materials and every
ply angle is $0^\circ$, so the material frame of each ply coincides with the wall frame.
`layup_2` and `layup_4` are the carbon spar caps on the upper and lower surfaces, `layup_1` is the
foam-cored skin that covers most of the contour, `layup_5` the three shear webs, `layup_0` the
trailing-edge reinforcement and `layup_3` the leading edge.

% source: examples/OpenSG_shell/1_get_beam_props_from_shell_cross_section/iea_s10_shell.yaml (materials block; E and G in Pa there, shown here in GPa)
```{list-table} Material properties
:header-rows: 2
:name: tab-sbc-materials

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
  - GPa
  - GPa
  - GPa
  - GPa
  - GPa
  - GPa
  -
  -
  -
* - gelcoat
  - 1235.0
  - 3.44
  - 3.44
  - 3.44
  - 1.323
  - 1.323
  - 1.323
  - 0.3
  - 0.3
  - 0.3
* - glass_triax
  - 1940.0
  - 28.2114
  - 16.2388
  - 15.8355
  - 8.24822
  - 3.49124
  - 3.49124
  - 0.497511
  - 0.18091
  - 0.27481
* - glass_uniax
  - 1940.0
  - 43.7
  - 16.5
  - 15.45
  - 3.265
  - 3.495
  - 3.48
  - 0.262
  - 0.264
  - 0.35
* - medium_density_foam
  - 130.0
  - 0.1425
  - 0.1425
  - 0.1425
  - 0.054
  - 0.054
  - 0.054
  - 0.3194
  - 0.3194
  - 0.3194
* - carbon_uniax
  - 1600.0
  - 140.0
  - 9.1
  - 9.1
  - 4.131
  - 4.131
  - 2.689
  - 0.3133
  - 0.3133
  - 0.47066
* - glass_biax
  - 1940.0
  - 11.0231
  - 11.0231
  - 16.0477
  - 13.2314
  - 3.48748
  - 3.48748
  - 0.688074
  - 0.117173
  - 0.117173
```

% source: examples/OpenSG_shell/1_get_beam_props_from_shell_cross_section/iea_s10_shell.yaml (sections and sets blocks); totals from docs/tutorials/msg_shell_beam.md
```{list-table} Layups (plies listed outer first, every ply angle 0°)
:header-rows: 2
:name: tab-sbc-layups

* - Layup
  - Plies: material (thickness)
  - Total thickness
  - Elements
* -
  - m
  - m
  -
* - `layup_0`
  - gelcoat (0.0005), glass_triax (0.00300294), glass_uniax (0.02609467), glass_triax (0.00300294)
  - 0.03260055
  - 25
* - `layup_1`
  - gelcoat (0.0005), glass_triax (0.00300294), medium_density_foam (0.07060815), glass_triax (0.00300294)
  - 0.07711403
  - 161
* - `layup_2`
  - gelcoat (0.0005), glass_triax (0.00300294), carbon_uniax (0.08105991), glass_triax (0.00300294)
  - 0.08756579
  - 13
* - `layup_3`
  - gelcoat (0.0005), glass_triax (0.00300294), glass_uniax (0.02158284), glass_triax (0.00300294)
  - 0.02808872
  - 8
* - `layup_4`
  - gelcoat (0.0005), glass_triax (0.00300294), carbon_uniax (0.07707452), glass_triax (0.00300294)
  - 0.0835804
  - 14
* - `layup_5`
  - glass_biax (0.002), medium_density_foam (0.042), glass_biax (0.002)
  - 0.046
  - 89
```

Complete input files are in `examples/OpenSG_shell/1_get_beam_props_from_shell_cross_section/`:
the SG `iea_s10_shell.yaml` and the driver script `beam_homo_shell.py`.

## Input

The yaml header selects the engine and the macro model:

% source: examples/OpenSG_shell/1_get_beam_props_from_shell_cross_section/iea_s10_shell.yaml (lines 1 to 3)
```yaml
n_model: 1      # 1 = beam, 2 = plate (opensg_solid), 3 = solid -- the macro model this SG homogenizes to
refined: 1      # 0 = classical (beam EB 4x4, KL wall); 1 = shear-refined (beam Timoshenko 6x6, RM wall)
msg: shell      # the ENGINE this SG belongs to (opensg_shell); `opensg <yaml>` dispatches on it
```

The rest of the file is a set of blocks, listed in {numref}`tab-sbc-blocks`.

% source: docs/tutorials/msg_shell_beam.md ("The YAML blocks"), counts checked against iea_s10_shell.yaml
```{list-table} Blocks of iea_s10_shell.yaml
:header-rows: 1
:name: tab-sbc-blocks

* - Block
  - Content
  - This station
* - `nodes`
  - `[y2 y3 0]` contour coordinates, metres
  - 307 nodes
* - `elements`
  - 1-based two-node line connectivity
  - 310 elements
* - `sets: element`
  - named element sets, one per wall region
  - 6 sets
* - `sections`
  - per set: `elementSet` plus `layup` as `[material, thickness, angle]` triples, outer ply first
  - 6 sections
* - `elementOrientations`
  - per element, the flattened $3\times3$ frame $[e_1\;e_2\;e_3]$
  - 310 rows
* - `materials`
  - `name`, `density`, `elastic: {E, G, nu}` as 3-vectors, with `G` = $[G_{12}, G_{13}, G_{23}]$ and `nu` = $[\nu_{12}, \nu_{13}, \nu_{23}]$
  - 6 materials
```

A section ties a layup to an element set. The upper spar cap reads:

% source: examples/OpenSG_shell/1_get_beam_props_from_shell_cross_section/iea_s10_shell.yaml (lines 978 to 992)
```yaml
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

Each row of `elementOrientations` holds $e_1$, $e_2$ and $e_3$ of one element. $e_1$ is the beam
axis (out of the page), $e_2$ (blue in {numref}`fig-sbc-orient`) runs along the contour, and $e_3$
(green) is the wall normal, which sets the outer mold line (OML) to inner mold line (IML) sense used
by the recovery. The first element reads:

% source: examples/OpenSG_shell/1_get_beam_props_from_shell_cross_section/iea_s10_shell.yaml (lines 1035 to 1037)
```yaml
elementOrientations:
- [0.0, 0.0, 1.0, -0.9721541292349604, 0.2343423756204075, 0.0, -0.2343423756204075,
  -0.9721541292349604, 0.0]
```

```{figure} ../_static/shell_xsec_orient.png
:width: 95%
:name: fig-sbc-orient

Per-element material axes on the contour: $e_2$ in blue, $e_3$ in green.
```

:::{tip}
Check that the green arrows point consistently into the section on the skin and consistently to
one side on each web before trusting any number. A wrong `elementOrientations` block is the most
common way to get a plausible-looking but wrong answer.
:::

The laminate reference surface is a run-time choice, not a yaml key. `opensg <yaml>` takes the
contour as the OML (laminates stack inward); `opensg <yaml> --center`, or
`build_rm_bundle(yaml, ref="center")` in Python, takes it as the laminate mid-surface. The choice
maps to a thickness fraction (`oml` to 0.0, `center` to 0.5) used for the wall law, the plate SG
origin, the ABD cache and the recovery depths. This station was generated on the mid-surface, so
every command below passes `--center`, and the quantities that depend on the reference axis ($GJ$
and the extension-bending couplings) are mid-surface values. The `reference: center` line at the
end of the yaml records this; the solver ignores it, and the CLI prints a note that it agrees with
the run.

The driver has a single user-input line:

% source: examples/OpenSG_shell/1_get_beam_props_from_shell_cross_section/beam_homo_shell.py
```python
name = "iea_s10"               # reads <name>_shell.yaml
```

## Run

```bash
cd examples/OpenSG_shell/1_get_beam_props_from_shell_cross_section
opensg iea_s10_shell.yaml --center
```

The header resolves the shell engine, which runs the RM ring and writes `iea_s10_shell_Timo.out`
and `iea_s10_shell_ABDG.out`. Add `--mesh` to also write `iea_s10_shell_mesh.png` and the gmsh mesh
`iea_s10_shell.msh`. The console shows the OpenSG banner and a short run header (input, engine, SG
dimension, analysis, macro model, reference), then the matrix under the title
`Timoshenko Beam Stiffness Matrix  [eps11 gam12 gam13 kappa1 kappa2 kappa3]:` and two closing
lines:

% source: src/opensg_shell/cli.py (the homogenization branch)
```text
Homogenization stored in iea_s10_shell_Timo.out
Time taken: <seconds> sec
```

The driver script produces the same kind of law through the Python interface and adds the two
figures of this page (`iea_s10_mesh.png` and, through `auto_emit`, `iea_s10_orient.png`).
`build_rm_bundle` itself never draws the orientation figure, so make the same `auto_emit` call
in your own driver.

```bash
python beam_homo_shell.py
```

The driver calls `build_rm_bundle(name + "_shell.yaml")`, whose default reference is `"oml"`; the
shipped `.out` files are mid-surface results. To reproduce them in Python, pass `ref="center"`:

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

`build_rm_bundle` returns a bundle rather than a matrix, because the recovery in
{doc}`shell_beam_recovery` reuses the identical discretization ({numref}`tab-sbc-bundle`).

% source: docs/tutorials/msg_shell_beam.md ("The one call")
```{list-table} Keys of the bundle returned by build_rm_bundle
:header-rows: 1
:name: tab-sbc-bundle

* - Key
  - Content
* - `Timo`
  - the $(6,6)$ Timoshenko stiffness
* - `V0`, `V1`
  - the zeroth- and first-order warping fields, $(6m,4)$ each, including the drilling $\omega_3$ boundary values
* - `corners`, `red_cells`, `rx3`, `re3`, `k22`
  - the contour geometry, connectivity, wall normals and curvature per element
* - `strip`
  - the prismatic strip actually assembled, as `(nodes, quads, h)`
* - `ax`, `cross`
  - the beam-axis index and the two cross-section axis indices
* - `layup_per_elem`, `layup_db`, `material_db`
  - the layup name of each element and the geometry-free plate SG databases
* - `frac`, `ref`, `g_source`
  - the reference fraction, its name, and the wall $G$ source
```

## Result

% source: examples/OpenSG_shell/1_get_beam_props_from_shell_cross_section/iea_s10_shell_Timo.out (footer); docs/tutorials/msg_shell_beam.md ("The numbers")
{numref}`lst-sbc-timo` is the stiffness block of `iea_s10_shell_Timo.out`. The file also holds the
compliance matrix and ends with `Time taken: 2.82 sec`, which covers the whole bundle build:
reading the yaml, the six per-layup MSG plate solves, the ring factorization and the
layup-database build.

% source: examples/OpenSG_shell/1_get_beam_props_from_shell_cross_section/iea_s10_shell_Timo.out
```{code-block} text
:caption: Timoshenko stiffness from iea_s10_shell_Timo.out, rows and columns [eps11 gam12 gam13 kappa1 kappa2 kappa3] (N, N·m, N·m²)
:name: lst-sbc-timo

 OpenSG msg-shell beam model [ext sh2 sh3 twist bend2 bend3]

 The Effective Timoshenko Beam Stiffness Matrix
 --------------------------------------------
     2.7660599E+010    -1.7230658E-005     4.6650498E-006     5.4020232E-006     1.6686413E+009    -2.0773040E+009
    -1.7230658E-005     7.1861326E+008     5.7520370E+007    -1.0192044E+008     2.3079953E-006     5.2738749E-006
     4.6650498E-006     5.7520370E+007     4.2174428E+008    -1.6217837E+008     8.4816767E-006     2.7931393E-007
     5.4020232E-006    -1.0192044E+008    -1.6217837E+008     2.4375653E+009     1.6449557E-006    -8.1140553E-006
     1.6686413E+009     2.3079953E-006     8.4816767E-006     1.6449557E-006     3.5143887E+010    -5.5571249E+009
    -2.0773040E+009     5.2738749E-006     2.7931393E-007    -8.1140553E-006    -5.5571249E+009     6.8129724E+010
```

% source: docs/tutorials/msg_shell_beam.md ("The numbers")
```{list-table} Diagonal of the Timoshenko stiffness
:header-rows: 1
:name: tab-sbc-diag

* - Term
  - Value
  - Units
  - Meaning
* - $EA$
  - $2.76606\times10^{10}$
  - N
  - axial stiffness
* - $GA_2$
  - $7.18613\times10^{8}$
  - N
  - transverse shear, chordwise
* - $GA_3$
  - $4.21744\times10^{8}$
  - N
  - transverse shear, flapwise
* - $GJ$
  - $2.43757\times10^{9}$
  - N·m²
  - torsional stiffness about the mid-surface reference
* - $EI_2$
  - $3.51439\times10^{10}$
  - N·m²
  - flapwise bending
* - $EI_3$
  - $6.81297\times10^{10}$
  - N·m²
  - edgewise bending
```

% source: docs/tutorials/msg_shell_beam.md ("The numbers")
The off-diagonal terms carry the couplings. The extension-bending entries $1.6686\times10^{9}$ and
$-2.0773\times10^{9}$ N·m place the tension centre away from the mid-surface reference point,
$-5.5571\times10^{9}$ N·m² is the flap-edge bending coupling (the principal bending axes are not
the $y_2$ and $y_3$ axes), and $-1.0192\times10^{8}$ and $-1.6218\times10^{8}$ N·m are the
shear-twist couplings that place the shear centre. Entries printed at $10^{-5}$ against diagonals
of $10^{10}$ are zeros to about fifteen digits: extension does not couple to transverse shear or
twist on this section.

`iea_s10_shell_ABDG.out` holds one $8\times8$ RM plate stiffness and compliance block per section,
headed by its layup, on the wall strains

$$\bar{\varepsilon} = [\varepsilon_{11}\;\varepsilon_{22}\;2\varepsilon_{12}\;\;
K_{11}\;K_{22}\;K_{12}+K_{21}\;\; 2\gamma_{13}\;2\gamma_{23}]^{T},
\qquad
\mathbf{ABDG} = \begin{bmatrix}A & B & 0\\ B & D & 0\\ 0 & 0 & G\end{bmatrix}.$$

% source: examples/OpenSG_shell/1_get_beam_props_from_shell_cross_section/iea_s10_shell_ABDG.out (sections 1 and 2)
The $2\times2$ transverse-shear block $G$ comes from the MSG second-order energy projection (the
least-squares construction of Eq. 61 in Yu, Hodges and Volovoi, *Computers & Structures*
81:439-454, 2003). There is no switch to set; the complementary-energy $G$ is used only as a
numerical fallback on a laminate whose MSG fit is not positive definite. The construction matters
most on the foam-cored walls: the $G$ diagonal of `layup_1` is 4.1457954E+006 and 4.1451801E+006
N/m, against 2.8674578E+008 and 2.1272143E+008 N/m for the carbon cap `layup_2`, so the soft core
dominates the shear compliance of the section.

:::{note}
The ring is a one-quad-deep prismatic strip whose extruded node row is mapped back onto the
contour, and every matrix is divided by the strip depth, so the result is per unit length. Each
node carries six degrees of freedom $[w_1\;w_2\;w_3\;\omega_1\;\omega_2\;\omega_3]$; the drilling
rotation $\omega_3$ is independent, so adjacent walls at a junction share a consistent rotation,
and its spurious mode is removed by one element-constant Lagrange multiplier per element
(`lam_space="elem"`, the inf-sup-stable choice). Transverse shear uses assumed-strain (MITC) tying of
the $\gamma_{23}$ row only (`shear="mitc4_g23"`), because on a prismatic strip $\gamma_{13}$ carries
no fluctuation gradient.
:::

## Files

% source: examples/OpenSG_shell/1_get_beam_props_from_shell_cross_section/ (folder listing) and beam_homo_shell.py
| File | Kind | Content |
|---|---|---|
| `iea_s10_shell.yaml` | input | the 1-D shell SG |
| `beam_homo_shell.py` | driver | builds the bundle, prints the $6\times6$, draws the figures |
| `iea_s10_shell.ff` | input | the macro state for the recovery in {doc}`shell_beam_recovery` |
| `iea_s10_shell_Timo.out` | output | Timoshenko stiffness and compliance, SwiftComp `.K` layout, timed |
| `iea_s10_shell_ABDG.out` | output | one $8\times8$ RM wall law and its compliance per section |
| `abd/iea_s10_shell_abd.yaml` | output | per-layup cache of the $8\times8$ wall law, plies, thickness and mass per area |
| `iea_s10_mesh.png`, `iea_s10_orient.png` | output | the contour by layup and the $e_2$/$e_3$ arrows (driver) |
| `iea_s10_shell_mesh.png`, `iea_s10_shell.msh` | output | mesh figure and gmsh mesh (with `--mesh`) |

:::{note}
The ABD cache is reused whenever its own `reference:` line matches the run (`mid` for
`--center`), and it records `g_source: msg`. Delete `abd/iea_s10_shell_abd.yaml` after you change a
layup or a material.
:::
