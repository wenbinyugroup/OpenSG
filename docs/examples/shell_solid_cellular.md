# Equivalent solid properties of cellular cross-sections

## Problem description

A prismatic cellular or lattice solid is described by the cross-section of one periodic cell. In
this example that cross-section is a one-dimensional shell structure gene (SG): the wall midlines
as two-node line elements, each wall carrying its laminate as material data. The engine
`opensg_shell` homogenizes it into the equivalent 3-D solid stiffness $C$ (`n_model: 3`), a
$6\times6$ matrix in the solid Voigt order
$[\bar\Gamma_{11}\;\bar\Gamma_{22}\;\bar\Gamma_{33}\;2\bar\Gamma_{23}\;2\bar\Gamma_{13}\;2\bar\Gamma_{12}]$
with axis 1 along the prismatic direction. The SG is driven by the six macro strains, so
periodicity in $y_2$ and $y_3$ is always on, on opposite edges and at the corners of the cell. The
wall law is the Reissner-Mindlin (RM) $8\times8$ ABDG law (ABD, the extension $A$, coupling $B$ and bending $D$
stiffness, plus the mechanics of structure genome (MSG) transverse-shear block $G$), so the walls carry the transverse shear that a classical (Kirchhoff) wall model discards.

% source: docs/tutorials/solid_props_shell_sg.md (Validation section, Deo and Yu 2023)
The four cases reproduce A. Deo and W. Yu, *Equivalent 3D properties of thin-walled composite
structures using mechanics of structure genome*, Mechanics of Advanced Materials and Structures
30(9), 2023, 1737-1748. Each result is compared with two columns of that paper: its own
thin-walled model (Deo MSG-TW, classical wall law) and its reference MSG solid model, computed with
SwiftComp (the equivalent of an OpenSG 2-D solid run). The cases are listed in
{numref}`tab-ssc-cases` and drawn in {numref}`fig-ssc-p15` to {numref}`fig-ssc-composite`.

% source: examples/OpenSG_shell/3_get_solid_props_from_shell_cross_section/*/make_*.py, *_1Dshell.yaml, *_C3D.out; docs/tutorials/solid_props_shell_sg.md
```{list-table} The four cases
:header-rows: 1
:name: tab-ssc-cases

* - Case (paper)
  - Folder
  - Geometry
  - Wall
  - Mesh
* - Cellular solid, $\theta = +15^\circ$ (paper Table 1)
  - `deo_cellular_solid`
  - $l = h = 10$ cm, cell area $\omega$ = 0.048637033051562736 m²
  - aluminium, $t = 5$ mm
  - 61 nodes, 60 elements
* - Re-entrant cellular solid, $\theta = -15^\circ$ (paper Table 2)
  - `deo_cellular_solid`
  - $l = h = 10$ cm, cell area $\omega$ = 0.028637033051562739 m²
  - aluminium, $t = 5$ mm
  - 61 nodes, 60 elements
* - Hierarchical square (paper Table 3)
  - `deo_hierarchical_square`
  - cell $2R$, interior square $2r$, four ligaments; $R = 10$ cm, $r = 5$ cm
  - aluminium, $t = 5$ mm
  - 120 nodes, 120 elements
* - Composite square (paper Fig. 13, Table 4)
  - `deo_composite_square`
  - cell $25.4\times25.4$ mm, horizontal walls at $y_3 = \pm5.84$ mm, one vertical wall through the centre
  - $[15]_8$, ply 0.127 mm, $t = 1.016$ mm
  - 113 nodes, 112 elements
```

```{figure} ../_static/deo_cellular_p15_mesh.png
:width: 45%
:name: fig-ssc-p15

Cellular solid, $\theta = +15^\circ$: the periodic double cell.
```

```{figure} ../_static/deo_cellular_m15_mesh.png
:width: 45%
:name: fig-ssc-m15

Re-entrant cellular solid, $\theta = -15^\circ$ (auxetic).
```

```{figure} ../_static/deo_hier_square_mesh.png
:width: 45%
:name: fig-ssc-hier

Hierarchical square: interior square with four ligaments to the cell edges.
```

```{figure} ../_static/deo_composite_square_mesh.png
:width: 45%
:name: fig-ssc-composite

Composite square: two horizontal walls and one vertical wall, all $[15]_8$ laminates.
```

% source: examples/OpenSG_shell/3_get_solid_props_from_shell_cross_section/deo_cellular_solid/deo_cellular_+15_1Dshell.yaml, examples/OpenSG_shell/3_get_solid_props_from_shell_cross_section/deo_hierarchical_square/deo_square_1Dshell.yaml, examples/OpenSG_shell/3_get_solid_props_from_shell_cross_section/deo_composite_square/deo_composite_square_1Dshell.yaml (materials blocks; E and G in Pa there, shown here in GPa)
```{list-table} Material properties
:header-rows: 2
:name: tab-ssc-materials

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
* - alu (cellular, hierarchical)
  - 2700.0
  - 68.9
  - 68.9
  - 68.9
  - 25.90226
  - 25.90226
  - 25.90226
  - 0.33
  - 0.33
  - 0.33
* - pl (composite square)
  - 1800.0
  - 141.96
  - 9.79
  - 9.79
  - 6.136
  - 6.136
  - 6.136
  - 0.42
  - 0.42
  - 0.42
```

Complete input files are in
`examples/OpenSG_shell/3_get_solid_props_from_shell_cross_section/`, one folder per case.

## Input

Each Deo case folder holds a generator script (`make_*.py`) that writes the SG yaml, the yaml itself,
and a driver (`*_props.py`) that runs it and writes the comparison table. The header of the
$+15^\circ$ cell reads:

% source: examples/OpenSG_shell/3_get_solid_props_from_shell_cross_section/deo_cellular_solid/deo_cellular_+15_1Dshell.yaml (lines 1 to 4)
```yaml
n_model: 3      # 1 = beam, 2 = plate (msg_solid), 3 = solid -- the macro model this SG homogenizes to
refined: 0      # 0 = classical (beam EB 4x4, KL wall); 1 = shear-refined (beam Timoshenko 6x6, RM wall)
msg: shell      # the ENGINE this SG belongs to (opensg_shell); `opensg <yaml>` dispatches on it
omega: 0.048637033051562736   # SG measure = the periodic cell area (2 l cos th) x 2 (h + l sin th)
```

% source: src/opensg_shell/cli.py (sg_cell_area); examples/OpenSG_shell/3_get_solid_props_from_shell_cross_section/deo_cellular_solid/make_deo_cellular_yamls.py (omega comment)
`refined:` is ignored for the solid macro model. The SG measure $\omega$ that divides the
stiffness is the `omega:` header key when the file declares one, and otherwise the measured
periodic cell, the node bounding box. The bounding box is not the convex hull of the contour; the
two differ on any non-convex cell, which is every honeycomb and lattice. The cellular yamls declare
$\omega$ because the node block is written to 12 decimals; the two square cells use the bounding
box. The single section and material of the aluminium cells read:

% source: examples/OpenSG_shell/3_get_solid_props_from_shell_cross_section/deo_cellular_solid/deo_cellular_+15_1Dshell.yaml (sections and materials blocks)
```yaml
sections:
- type: shell
  elementSet: layup_0
  layup:
  - - alu
    - 5.000000e-03
    - 0.0
materials:
- name: alu
  density: 2700.0
  elastic:
    E: [6.890000e+10, 6.890000e+10, 6.890000e+10]
    G: [2.590226e+10, 2.590226e+10, 2.590226e+10]
    nu: [0.330000, 0.330000, 0.330000]
```

The yamls end with `reference: center`: the contours are wall midlines. The solver ignores that line
(the CLI prints a note); the laminate reference is set at run time with `--center` on the command line or
`ref="center"` in `build_solid_bundle`, whose default is `"oml"`. Through the Python interface the
whole run is one call:

% source: docs/tutorials/solid_props_shell_sg.md ("Driver")
```python
from opensg_shell import build_solid_bundle

b = build_solid_bundle("cell_1Dshell.yaml", cell_area=A_cell)
C3D = b["C3D"]            # 6x6 equivalent solid stiffness
b["solve_time"]           # seconds; also written into <yaml>_C3D.out
```

## Run

```bash
cd examples/OpenSG_shell/3_get_solid_props_from_shell_cross_section/deo_cellular_solid
opensg deo_cellular_+15_1Dshell.yaml --center
```

The other yamls run the same way (`deo_cellular_-15_1Dshell.yaml`, `deo_square_1Dshell.yaml`,
`deo_composite_square_1Dshell.yaml`). The console prints the matrix under the title
`Cauchy Continuum Stiffness Matrix  [11 22 33 23 13 12]:` and closes with

% source: src/opensg_shell/cli.py (the homogenization branch)
```text
Homogenization stored in deo_cellular_+15_1Dshell_C3D.out
Time taken: <seconds> sec
```

The run writes `<base>_C3D.out` in the SwiftComp `.K` layout (the stiffness, its compliance and the
orthotropic-approximated engineering constants, with a `Time taken` footer) and `<base>_ABDG.out`,
the $8\times8$ RM wall law of each section. The drivers rebuild the comparison tables and the mesh
figures:

```bash
python deo_cellular_solid_props.py
```

% source: examples/OpenSG_shell/3_get_solid_props_from_shell_cross_section/deo_cellular_solid/deo_cellular_+15_1Dshell_ABDG.out (B block zero)
The tables below are mid-surface results: the `_ABDG.out` files hold $B = 0$ for the aluminium
wall. The drivers call `build_solid_bundle(yml, cell_area=area)` without `ref`, which selects the
OML default; add `ref="center"` to that call to match the `--center` run.

% source: the Time taken footers in examples/OpenSG_shell/3_get_solid_props_from_shell_cross_section/ of deo_cellular_+15_1Dshell_C3D.out, deo_cellular_-15_1Dshell_C3D.out, deo_square_1Dshell_C3D.out, deo_composite_square_1Dshell_C3D.out
The recorded solve times are 1.10 s ($+15^\circ$), 0.17 s ($-15^\circ$), 1.35 s (hierarchical
square) and 1.45 s (composite square).

## Result

{numref}`lst-ssc-c3d` is the stiffness of the $+15^\circ$ cell as written. The comparison tables
below are in MPa and give the error of each thin-walled column against SwiftComp.

% source: examples/OpenSG_shell/3_get_solid_props_from_shell_cross_section/deo_cellular_solid/deo_cellular_+15_1Dshell_C3D.out
```{code-block} text
:caption: Equivalent solid stiffness of the +15° cell from deo_cellular_+15_1Dshell_C3D.out, order [11 22 33 23 13 12] (Pa)
:name: lst-ssc-c3d

 OpenSG msg-shell equivalent 3D solid (cross-section SG), omega 0.048637033 (user)

 The Effective Cauchy Continuum Stiffness Matrix
 --------------------------------------------
     4.7338395E+009     1.0860078E+009     3.8063296E+008     5.0517856E-006     5.6734646E-005     4.1598012E-006
     1.0860078E+009     2.4442418E+009     8.4669081E+008    -2.4129503E-005    -2.7357740E-004    -1.9340510E-005
     3.8063296E+008     8.4669081E+008     3.0674240E+008     4.0313446E-005     4.4539606E-004     3.1929391E-005
     5.0517856E-006    -2.4129503E-005     4.0313446E-005     4.1992634E+006    -1.0574351E-002    -8.7310517E-005
     5.6734646E-005    -2.7357740E-004     4.4539606E-004    -1.0574351E-002     5.6260804E+008     4.8857888E-003
     4.1598012E-006    -1.9340510E-005     3.1929391E-005    -8.7310517E-005     4.8857888E-003     9.9377516E+008
```

### Cellular solid, $\theta = \pm15^\circ$

% source: examples/OpenSG_shell/3_get_solid_props_from_shell_cross_section/deo_cellular_solid/deo_cellular_compare.dat
```{list-table} Cellular solid, $\theta = +15^\circ$ (MPa)
:header-rows: 1
:name: tab-ssc-p15

* - Term
  - OpenSG (RM)
  - Deo MSG-TW
  - SwiftComp
  - OpenSG %
  - Deo MSG-TW %
* - $C_{11}$
  - 4733.84
  - 4736.90
  - 4678.90
  - +1.17
  - +1.24
* - $C_{12}$
  - 1086.01
  - 1089.40
  - 1105.50
  - -1.76
  - -1.46
* - $C_{13}$
  - 380.63
  - 381.81
  - 386.88
  - -1.61
  - -1.31
* - $C_{22}$
  - 2444.24
  - 2446.39
  - 2488.90
  - -1.79
  - -1.71
* - $C_{23}$
  - 846.69
  - 847.44
  - 860.89
  - -1.65
  - -1.56
* - $C_{33}$
  - 306.74
  - 306.99
  - 311.48
  - -1.52
  - -1.44
* - $C_{44}$
  - **4.20**
  - 4.32
  - 4.19
  - **+0.18**
  - +3.09
* - $C_{55}$
  - 562.61
  - 564.15
  - 573.11
  - -1.83
  - -1.56
* - $C_{66}$
  - 993.78
  - 997.52
  - 1000.10
  - -0.63
  - -0.26
```

% source: examples/OpenSG_shell/3_get_solid_props_from_shell_cross_section/deo_cellular_solid/deo_cellular_compare.dat
```{list-table} Re-entrant cellular solid, $\theta = -15^\circ$ (MPa)
:header-rows: 1
:name: tab-ssc-m15

* - Term
  - OpenSG (RM)
  - Deo MSG-TW
  - SwiftComp
  - OpenSG %
  - Deo MSG-TW %
* - $C_{11}$
  - 7505.26
  - 7507.40
  - 7352.90
  - +2.07
  - +2.10
* - $C_{12}$
  - 1090.52
  - 1094.10
  - 1075.20
  - +1.42
  - +1.76
* - $C_{13}$
  - -219.81
  - -220.53
  - -213.94
  - +2.74
  - +3.08
* - $C_{22}$
  - 4151.29
  - 4154.90
  - 4080.00
  - +1.75
  - +1.84
* - $C_{23}$
  - -846.69
  - -847.45
  - -821.77
  - +3.03
  - +3.12
* - $C_{33}$
  - 180.61
  - 180.75
  - 173.48
  - +4.11
  - +4.19
* - $C_{44}$
  - **2.47**
  - 2.68
  - 2.47
  - **+0.17**
  - +8.38
* - $C_{55}$
  - 331.26
  - 332.17
  - 346.01
  - -4.26
  - -4.00
* - $C_{66}$
  - 1687.82
  - 1694.20
  - 1706.40
  - -1.09
  - -0.71
```

% source: examples/OpenSG_shell/3_get_solid_props_from_shell_cross_section/deo_cellular_solid/deo_cellular_compare.dat; docs/tutorials/solid_props_shell_sg.md (C44 paragraph)
The re-entrant cell is auxetic, with negative $C_{13}$ and $C_{23}$. $C_{44}$ is the shear stiffness
in the plane of the cross-section. The paper flags it as the largest error of its thin-walled
model and attributes the error to the classical wall model, which drops the transverse shear of
the segments; it names a first-order shear wall model as future work. The RM wall law is that
model: the errors of $+3.09$ % and $+8.38$ % drop to $+0.18$ % and $+0.17$ %. Every other term
tracks the paper's own thin-walled column to a few tenths of a percent, so the two independent
thin-walled implementations agree where they share physics.

% source: docs/index.md ("Validation status"), docs/theory/rules/benchmarking_solid_props.md, docs/theory/rules/junction_corrections.md
:::{note}
$C_{44}$ has a closed form on a periodic cross cell, $C_{44} = \tfrac12 E'(t/L)^3$ for walls of
thickness $t$ at period $L$, and the shell route reproduces it to $+0.01$ % at $t/L = 0.0125$. Use
a cross cell, not a ring cell, to benchmark $C_{44}$: the periodic tie merges coincident ring walls
into one $2t$ wall in a solid model (bending $8t^3$), while the shell keeps two independent $t$
walls ($2t^3$), an exact factor of 4 by construction. See
{doc}`../theory/rules/benchmarking_solid_props`.
:::

### Hierarchical square

% source: examples/OpenSG_shell/3_get_solid_props_from_shell_cross_section/deo_hierarchical_square/deo_square_compare.dat
```{list-table} Hierarchical square (MPa)
:header-rows: 1
:name: tab-ssc-hier

* - Term
  - OpenSG (RM)
  - Deo MSG-TW
  - SwiftComp
  - OpenSG %
  - Deo MSG-TW %
* - $C_{11}$
  - 5183.48
  - 5186.39
  - 5099.03
  - +1.66
  - +1.71
* - $C_{12}$
  - 24.22
  - 24.83
  - 26.75
  - -9.47
  - -7.18
* - $C_{13}$
  - 24.22
  - 24.83
  - 26.75
  - -9.47
  - -7.18
* - $C_{22}$
  - 46.23
  - 47.13
  - 50.46
  - -8.39
  - -6.60
* - $C_{23}$
  - 27.15
  - 27.94
  - 30.59
  - -11.23
  - -8.66
* - $C_{33}$
  - 46.23
  - 47.13
  - 50.45
  - -8.37
  - -6.58
* - $C_{44}$
  - 3.20
  - 3.22
  - 3.39
  - -5.60
  - -5.01
* - $C_{55}$
  - 647.56
  - 650.00
  - 670.92
  - -3.48
  - -3.12
* - $C_{66}$
  - 647.56
  - 650.00
  - 670.93
  - -3.48
  - -3.12
```

% source: docs/tutorials/solid_props_shell_sg.md ("Hierarchical square")
This lattice is bending-dominated: its transverse channels are $t^3$-scale ($C_{22}\approx 50$ MPa
against $C_{11}\approx 5100$ MPa), so the load path is wall bending, not wall stretching. The RM
result is mesh-converged and its offset is physics. Refining from 10 to 40 elements per segment
moves $C_{23}$ by 1 %, and the same model with shear-rigid walls (the Kirchhoff limit) reproduces
the paper's classical column to 0.0 to 0.3 %, with $C_{12} = 24.83$ and $C_{44} = 3.22$ exactly.
The gap between the two thin-walled columns is therefore the wall transverse-shear compliance,
which is real at the $t/l = 0.1$ ligaments. Both thin-walled models miss the same thing against
SwiftComp: the finite rotational stiffness of the joint regions, which no midline model carries.
The smaller apparent error of the classical model is a partial cancellation of its artificial
shear rigidity against that missing joint stiffness.

### Composite square

% source: examples/OpenSG_shell/3_get_solid_props_from_shell_cross_section/deo_composite_square/deo_composite_square_compare.dat
```{list-table} Composite square (MPa)
:header-rows: 1
:name: tab-ssc-composite

* - Term
  - OpenSG (RM)
  - Deo MSG-TW
  - SwiftComp
  - OpenSG %
  - Deo MSG-TW %
* - $C_{11}$
  - 15263.83
  - 16133.90
  - 15518.11
  - **-1.64**
  - +3.97
* - $C_{12}$
  - 936.74
  - 1016.82
  - 1008.76
  - -7.14
  - +0.80
* - $C_{13}$
  - 468.37
  - 468.38
  - 459.01
  - +2.04
  - +2.04
* - $C_{15}$
  - 1191.85
  - -1192.81
  - -1135.27
  - -204.98
  - +5.07
* - $C_{16}$
  - 2383.70
  - 2589.74
  - 2535.23
  - -5.98
  - +2.15
* - $C_{22}$
  - 906.11
  - 983.56
  - 988.88
  - -8.37
  - -0.54
* - $C_{23}$
  - 0.00
  - 0.00
  - 20.44
  - -100.00
  - -100.00
* - $C_{26}$
  - 292.25
  - 317.09
  - 311.80
  - -6.27
  - +1.70
* - $C_{33}$
  - 453.06
  - 453.06
  - 456.25
  - -0.70
  - -0.70
* - $C_{35}$
  - 146.13
  - -146.06
  - -140.04
  - -204.35
  - +4.30
* - $C_{44}$
  - 0.94
  - 1.08
  - 1.16
  - -18.57
  - -6.90
* - $C_{55}$
  - 547.31
  - 547.32
  - 545.43
  - +0.35
  - +0.35
* - $C_{66}$
  - 1094.63
  - 1188.20
  - 1188.30
  - -7.88
  - -0.01
```

% source: docs/tutorials/solid_props_shell_sg.md ("Composite square")
$C_{13}$, $C_{33}$ and $C_{55}$ reproduce the paper's thin-walled column, and $C_{11}$ improves on it
against SwiftComp. $C_{15}$ and $C_{35}$ match the published magnitudes to 0.1 % with the opposite
sign, which is why their percentages read $-204.98$ % and $-204.35$ %. The ply-angle handedness conventions of
the two codes have not been reconciled, and that difference is the likely cause of the $-6$ to
$-8$ % group ($C_{12}$, $C_{16}$, $C_{22}$, $C_{26}$, $C_{66}$). $C_{23}$ is zero in both
thin-walled models, for the reason below.

### Why $C_{23}$ vanishes on axis-aligned lattices

% source: docs/tutorials/solid_props_shell_sg.md ("Why C23 vanishes on axis-aligned lattices"), docs/theory/rules/junction_corrections.md
For a wall at angle $\varphi$, the only rows of the macro strain operator with weight in both the
$\bar\Gamma_{22}$ and the $\bar\Gamma_{33}$ columns are the membrane $\varepsilon_{22}$ row
($\cos^2\varphi$, $\sin^2\varphi$) and the transverse-shear $2\gamma_{23}$ row
($-\sin\varphi\cos\varphi$, $+\sin\varphi\cos\varphi$), which give the per-length integrand

$$(A_{11}-G_{\rm msg})\,\sin^2\varphi\cos^2\varphi .$$

On a $0/90$ lattice every wall has $\sin^2\varphi\cos^2\varphi = 0$, so this $O(t)$ membrane path
vanishes identically and the whole of $C_{23}$ is the $O(t^2)$ junction contribution of the
wall-overlap blocks, which have zero measure on a midline mesh. Measured on a solid reference it
obeys $C_{23} \simeq C_{23}^{\rm 3D}\cdot A_{\rm junction}$ to within a few percent. Inclined-wall
cells are unaffected: the same code returns $C_{23} = E' t L/\sqrt{2}$ to 0.02 % on a $\pm45^\circ$
cell, which is why the cellular cases carry $C_{23}$ correctly and only the square lattices expose
the junction term.

:::{warning}
`build_solid_bundle` accepts optional junction corrections (`junction="census"`, `"micro"`,
`"microcell"`) that recover this term on stretch-dominated cells. They are off by default and do
not apply to bending-dominated lattices such as the hierarchical square. See
{doc}`../theory/rules/junction_corrections`.
:::

### Isotropic square tube

% source: examples/OpenSG_shell/3_get_solid_props_from_shell_cross_section/square_section_isotropic/square_tube_1Dshell.yaml (header) and square_tube_1Dshell_C3D.out
The folder `square_section_isotropic` holds one more input, a closed isotropic square tube
({numref}`fig-ssc-tube`). Its $\omega$ is the wall material area, which is not a property of the
node cloud, so the yaml declares it: `omega: 0.12`, that is $4at$ with $a = 1.0$ and $t = 0.03$.
With that measure the axial modulus of `square_tube_1Dshell_C3D.out` is
`E1  =     7.0000000E+010` and `nu12=     3.0000000E-001`, the modulus and Poisson ratio of the wall.

```{figure} ../_static/deo_square_tube_mesh.png
:width: 45%
:name: fig-ssc-tube

Isotropic square tube, side $a = 1.0$ m: the 1-D shell SG of `square_section_isotropic`.
```

## Files

% source: examples/OpenSG_shell/3_get_solid_props_from_shell_cross_section/ (folder listing) and the driver docstrings
| Folder | Inputs | Driver | Outputs |
|---|---|---|---|
| `deo_cellular_solid` | `deo_cellular_+15_1Dshell.yaml`, `deo_cellular_-15_1Dshell.yaml` (written by `make_deo_cellular_yamls.py`) | `deo_cellular_solid_props.py` | `*_1Dshell_C3D.out`, `*_1Dshell_ABDG.out`, `*_msg_shell.out` (the $6\times6$ in MPa), `deo_cellular_compare.dat`, `deo_cellular_+15_mesh.png`, `deo_cellular_-15_mesh.png` |
| `deo_hierarchical_square` | `deo_square_1Dshell.yaml` (written by `make_deo_square_yaml.py`) | `deo_square_solid_props.py` | `deo_square_1Dshell_C3D.out`, `deo_square_1Dshell_ABDG.out`, `deo_square_msg_shell.out`, `deo_square_compare.dat`, `deo_square_mesh.png` |
| `deo_composite_square` | `deo_composite_square_1Dshell.yaml` (written by `make_deo_composite_yaml.py`) | `deo_composite_square_props.py` | `deo_composite_square_1Dshell_C3D.out`, `deo_composite_square_1Dshell_ABDG.out`, `deo_composite_square_msg_shell.out`, `deo_composite_square_compare.dat`, `deo_composite_square_mesh.png` |
| `square_section_isotropic` | `square_tube_1Dshell.yaml` | `solid_props_msg_shell.py` | `square_tube_1Dshell_C3D.out`, `square_tube_1Dshell_ABDG.out`, `square_solid_msg_shell.out`, `square_tube_mesh.png` |

Each folder also holds a `<base>.ff` macro state (a solid stress `FF`). The equivalent-solid route
is homogenization only; the file is the macro state a recovery of this route will consume.
