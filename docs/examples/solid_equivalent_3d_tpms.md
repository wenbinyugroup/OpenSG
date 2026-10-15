# Equivalent solid properties of TPMS unit cells

## Problem description

This example computes the equivalent 3-D solid stiffness of two triply periodic minimal surface
(TPMS) unit cells: aluminium sheets around a Schwarz-P surface, of two thicknesses, meshed with
linear tetrahedra. The structure gene (SG) is a 3-D solid SG, the engine is `opensg_solid` and the
macro model is the 3-D continuum (`n_model = 3`). The result is the $6\times6$ effective Cauchy
continuum stiffness of the cell, its compliance and the orthotropic engineering constants.

The cells are periodic in all three directions, and periodic is the default boundary treatment:
opposite faces, edges and corners are tied through the periodic assembly map. The meshes come as
SwiftComp `.sc` decks, and SwiftComp's own results for the same decks (`Sample_1.sc.k`,
`Sample_2.sc.k`) are the benchmark. The page also compares the aperiodic boundary treatment and the
shell route on the same surface ({doc}`shell_solid_3d_tpms`).

% source: examples/OpenSG-solid/6_get_solid_props_from_3D_SG/sample_<n>/Sample_<n>.sc (meta line); solid_vs_shell_compare.dat (rho, t); sample_<n>/Sample_<n>.out (Time taken)
| Sample | Nodes | Tetrahedra | Sheet thickness (unit cell) | Relative density | Time taken |
|---|---|---|---|---|---|
| | | | | | sec |
| Sample_1 | 116851 | 545741 | t~0.128 | 0.3000 | 69.20 |
| Sample_2 | 54874 | 191957 | t~0.0365 | 0.0857 | 23.63 |

Sample_2 meshes the sheet whose midsurface is the shell SG of {numref}`fig-tpms-surface`, the
input of {doc}`shell_solid_3d_tpms`. No picture of the tetrahedral meshes is shipped.

```{figure} ../_static/schwarz_p_mesh.png
:width: 55%
:name: fig-tpms-surface

The Schwarz-P surface of the unit cell, as the triangulated shell SG of the shell route.
```

% source: examples/OpenSG-solid/6_get_solid_props_from_3D_SG/sample_1/run_sample_1.py; sample_1/Sample_1.sc (material block)
| Name | Type | $E$ | $\nu$ | $G$ |
|---|---|---|---|---|
| | | Pa | | Pa |
| aluminium | isotropic | 69.0e9 | 0.30 | $E/(2(1+\nu))$ |

The input files are in `examples/OpenSG-solid/6_get_solid_props_from_3D_SG/`: `sample_1/Sample_1.sc`
and `sample_2/Sample_2.sc`, each with its SwiftComp result `.sc.k` and its driver.

## Input

The drivers read the `.sc` decks directly. A deck opens with the model header lines and the meta
line (SG dimension, number of nodes, number of elements, number of materials), then lists the nodes
and the tetrahedra, and ends with the material block (id, type 0 for isotropic, an auxiliary
temperature/density line, then $E$ and $\nu$ in MPa) and the scale line:

% source: examples/OpenSG-solid/6_get_solid_props_from_3D_SG/sample_1/Sample_1.sc (first and last lines)
```text
0
0 0 0 0

3 116851 545741 1 0 0

1 0.161016949153 -0.03729663714677 0.466101694915
...
545741 1 15341 104464 75007 50306 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0

1 0 1
0 0
69000 0.3

1.0
```

The driver replaces the deck's material by the same aluminium in SI units, so the result is in Pa:

% source: examples/OpenSG-solid/6_get_solid_props_from_3D_SG/sample_1/run_sample_1.py
```python
S = "1"
SC = "Sample_%s.sc" % S
E, nu = 69.0e9, 0.30
G = E/(2*(1+nu))

r = plate_homo_2d(SC, material_param=jnp.array([(E, E, E, G, G, G,
                                                 nu, nu, nu)]),
                  angles=jnp.array([0.0]), n_model=3, workdir=".", plot=True,
                  boundary="periodic")   # SwiftComp .K digit-parity mode
```

% source: docs/tutorials/solid_props_3d_sg.md ("Solid route", important box); src/opensg_solid/cli.py (header keys omega, aperiodic)
:::{note}
The SG measure $\omega$ of a 3-D solid SG is the node bounding-box volume, the volume the equivalent
continuum occupies and the cell the periodic map ties, not the summed element (material) volume. Both
samples are unit cells, so $\omega = 1$ and the `.out` compares with the SwiftComp `.K` entry for
entry, with only the unit change (OpenSG writes Pa here, SwiftComp MPa). In a yaml, an `omega:`
header key overrides the measured value.
:::

## Run

```bash
cd examples/OpenSG-solid/6_get_solid_props_from_3D_SG/sample_1
python run_sample_1.py
```

`sample_2/run_sample_2.py` is the same driver with `S = "2"`. Each run writes the timed
SwiftComp-layout `Sample_<n>.out`, the per-unit-cell matrix `Sample_<n>_msg_solid.out`, the mesh
figure and the comparison table `Sample_<n>_compare.dat`, which it also prints. The `.out` opens with
the banner and closes with the timing line:

% source: examples/OpenSG-solid/6_get_solid_props_from_3D_SG/sample_1/Sample_1.out
```text
 OpenSG msg-solid 3D elastic model, omega 1, periodic
 ...
 Time taken: 69.20 sec
```

The command line reads a yaml. `opensg sc_to_yaml` writes `Sample_1.yaml` (and `Sample_1.msh`)
from the deck, with the deck's own material (`69000 0.3`, in the deck's MPa), and `opensg` runs it:

% source: docs/tutorials/solid_props_3d_sg.md ("Solid route"); `opensg sc_to_yaml --help`
```bash
opensg sc_to_yaml Sample_1.sc --n_model 3
opensg Sample_1.yaml
```

The two comparison scripts run from the example folder:

```bash
cd examples/OpenSG-solid/6_get_solid_props_from_3D_SG
python compare_boundary_modes.py
python solid_vs_shell_compare.py
```

`compare_boundary_modes.py` runs both samples with both boundary treatments, keeps each `.out` as
`Sample_<n>_<mode>.out` and writes `boundary_compare.dat`. `solid_vs_shell_compare.py` reads the two
`Sample_<n>_msg_solid.out` files and the `schwarz_p_3Dshell_C3D.out` of the shell example, and writes
`solid_vs_shell_compare.dat`.

## Result

{numref}`lst-tpms-s1` is the stiffness block of `Sample_1.out`; the file continues with the
compliance and the engineering constants ({numref}`tab-tpms-constants`).

% source: examples/OpenSG-solid/6_get_solid_props_from_3D_SG/sample_1/Sample_1.out
```{code-block} text
:caption: Effective Cauchy continuum stiffness of Sample_1 from Sample_1.out, Pa, order [11 22 33 23 13 12]
:name: lst-tpms-s1

 The Effective Cauchy Continuum Stiffness Matrix
 --------------------------------------------
     1.0190158E+010     5.6460685E+009     5.6460915E+009     1.7077286E+004     6.0850329E+004     5.3492520E+004
     5.6460685E+009     1.0190131E+010     5.6460124E+009     3.3690961E+004     9.7456199E+002    -1.0898598E+005
     5.6460915E+009     5.6460124E+009     1.0189958E+010    -1.1304849E+005     4.1008720E+004     1.0001640E+005
     1.7077286E+004     3.3690961E+004    -1.1304849E+005     4.5196402E+009     4.5040986E+004    -2.0992207E+004
     6.0850329E+004     9.7456198E+002     4.1008720E+004     4.5040986E+004     4.5196366E+009     5.0843711E+003
     5.3492520E+004    -1.0898598E+005     1.0001640E+005    -2.0992207E+004     5.0843711E+003     4.5197549E+009
```

% source: examples/OpenSG-solid/6_get_solid_props_from_3D_SG/sample_1/Sample_1.out; sample_2/Sample_2.out
```{list-table} Engineering constants (approximated as orthotropic), moduli in Pa
:header-rows: 1
:name: tab-tpms-constants

* - Sample
  - $E_1$
  - $E_2$
  - $E_3$
  - $G_{12}$
  - $G_{13}$
  - $G_{23}$
  - $\nu_{12}$
  - $\nu_{13}$
  - $\nu_{23}$
* - Sample_1
  - 6.1641278E+009
  - 6.1641815E+009
  - 6.1640078E+009
  - 4.5197549E+009
  - 4.5196366E+009
  - 4.5196402E+009
  - 3.5652386E-001
  - 3.5654253E-001
  - 3.5653028E-001
* - Sample_2
  - 9.7551133E+008
  - 9.7545861E+008
  - 9.7540496E+008
  - 1.1066643E+009
  - 1.1066684E+009
  - 1.1067345E+009
  - 4.1174435E-001
  - 4.1178978E-001
  - 4.1179268E-001
```

### Comparison with SwiftComp

$C_{11}$ of `Sample_1.out`, $1.0190158\times10^{10}$ Pa, is 10190.158 MPa against the `.K` value
1.0190158E+004; for Sample_2, $2.3031598\times10^{9}$ Pa against 2.3031598E+003. The driver
tabulates the nine resolved terms per unit cell, in MPa:

% source: examples/OpenSG-solid/6_get_solid_props_from_3D_SG/sample_1/Sample_1_compare.dat; sample_2/Sample_2_compare.dat
| Term | Sample_1 OpenSG | Sample_1 SwiftComp | % | Sample_2 OpenSG | Sample_2 SwiftComp | % |
|---|---|---|---|---|---|---|
| $C_{11}$ | 10190.158 | 10190.158 | -0.000 | 2303.160 | 2303.160 | -0.000 |
| $C_{12}$ | 5646.069 | 5646.068 | +0.000 | 1612.130 | 1612.130 | -0.000 |
| $C_{13}$ | 5646.092 | 5646.091 | +0.000 | 1612.141 | 1612.141 | +0.000 |
| $C_{22}$ | 10190.131 | 10190.131 | +0.000 | 2303.067 | 2303.067 | +0.000 |
| $C_{23}$ | 5646.012 | 5646.012 | +0.000 | 1612.120 | 1612.120 | +0.000 |
| $C_{33}$ | 10189.958 | 10189.958 | +0.000 | 2303.019 | 2303.018 | +0.000 |
| $C_{44}$ | 4519.640 | 4519.640 | -0.000 | 1106.734 | 1106.735 | -0.000 |
| $C_{55}$ | 4519.637 | 4519.637 | -0.000 | 1106.668 | 1106.668 | +0.000 |
| $C_{66}$ | 4519.755 | 4519.755 | -0.000 | 1106.664 | 1106.664 | -0.000 |

% source: docs/tutorials/solid_props_3d_sg.md ("Solid route")
Both samples match SwiftComp digit for digit, to seven significant figures on every entry, in the
`.out` itself. The law is cubic to five digits, and every coupling that cubic symmetry forbids sits
four to five orders below the diagonal: the check on the periodic tie in all three directions.

### Boundary treatment

`boundary="aperiodic"` in `plate_homo_2d` (or `aperiodic: 1` in a yaml header) maps the boundary
solution onto the bounding-box-face nodes as Dirichlet data, a kinematic upper bound for one cell.
Solid elements carry only translations, so every degree of freedom of those nodes is clamped. Same
decks, per unit cell, MPa:

% source: examples/OpenSG-solid/6_get_solid_props_from_3D_SG/boundary_compare.dat
| Term | Sample_1 aperiodic | periodic | % | Sample_2 aperiodic | periodic | % |
|---|---|---|---|---|---|---|
| $C_{11}$ | 12977.036 | 10190.158 | +27.349 | 2810.329 | 2303.160 | +22.021 |
| $C_{22}$ | 12977.030 | 10190.131 | +27.349 | 2810.363 | 2303.067 | +22.027 |
| $C_{33}$ | 12976.815 | 10189.958 | +27.349 | 2810.251 | 2303.019 | +22.025 |
| $C_{12}$ | 5423.565 | 5646.069 | -3.941 | 1548.627 | 1612.130 | -3.939 |
| $C_{13}$ | 5423.674 | 5646.092 | -3.939 | 1548.954 | 1612.141 | -3.919 |
| $C_{23}$ | 5423.151 | 5646.012 | -3.947 | 1549.049 | 1612.120 | -3.912 |
| $C_{44}$ | 5545.997 | 4519.640 | +22.709 | 1300.352 | 1106.734 | +17.494 |
| $C_{55}$ | 5546.044 | 4519.637 | +22.710 | 1300.151 | 1106.668 | +17.483 |
| $C_{66}$ | 5546.133 | 4519.755 | +22.709 | 1300.048 | 1106.664 | +17.474 |
| boundary nodes | 17610 | | | 4456 | | |
| solve | 30.2 s | 28.5 s | | 9.6 s | 8.4 s | |

% source: docs/tutorials/solid_props_3d_sg.md ("Boundary treatment on the solid route")
The bias is much larger than on the shell route (+7 %) because the clamped set is much larger: the
solid sheet meets the cube faces in 2-D patches, 17 610 of 116 851 nodes (15 %) for Sample_1
against 720 of 13 536 (5 %) for the shell traces. Unlike on the shell route, aperiodic is not faster
on the solid route, because the periodic tie merges only face pairs and the factorized size barely changes. Periodic is
the exact treatment for a periodic medium; use aperiodic only for an SG that is not periodic.

### Solid versus shell on the same surface

% source: docs/tutorials/solid_props_3d_sg.md ("Shell versus solid on the same surface"); examples/OpenSG_shell/4_get_solid_props_from_shell_3D_SG/schwarz_p_3Dshell.yaml (thickness); examples/OpenSG_shell/4_get_solid_props_from_shell_3D_SG/make_schwarz_yaml.py; examples/OpenSG-solid/6_get_solid_props_from_3D_SG/solid_vs_shell_compare.py
Sample_2 is the Schwarz-P sheet of the shell route: its midsurface area recovered from the tet mesh
(2.3448) matches the shell mesh (2.3533). The shell model runs at $t = 0.036457$, the Sample_2
volume-consistent thickness of `schwarz_p_3Dshell.yaml` (set in `make_schwarz_yaml.py`), and the
comparison script normalizes the shell row with a relative density of 0.0860 against the solid's
0.0857, so the two routes meet on one structure. The comparison script records the headline terms
with the density-normalized values $C/(E\rho)$ and the Zener ratio $2C_{44}/(C_{11}-C_{12})$:

% source: examples/OpenSG-solid/6_get_solid_props_from_3D_SG/solid_vs_shell_compare.dat
| Case | $\rho$ | $C_{11}$ | $C_{12}$ | $C_{44}$ | $C_{11}/(E\rho)$ | $C_{44}/(E\rho)$ | Zener |
|---|---|---|---|---|---|---|---|
| | | MPa | MPa | MPa | | | |
| Sample_1 solid (Schwarz-P, t~0.128) | 0.3000 | 10190.2 | 5646.1 | 4519.6 | 0.4923 | 0.2183 | 1.989 |
| Sample_2 solid (Schwarz-P, t~0.0365) | 0.0857 | 2303.2 | 1612.1 | 1106.7 | 0.3895 | 0.1872 | 3.203 |
| Schwarz-P shell t=0.0365 (msg_shell) | 0.0860 | 2225.3 | 1564.9 | 1041.0 | 0.3750 | 0.1754 | 3.152 |

Term by term against Sample_2, per unit cell, MPa:

% source: docs/tutorials/solid_props_3d_sg.md ("Shell versus solid on the same surface")
| Term | Shell SG (26 360 shell elements) | Solid SG (191 957 tets) | % |
|---|---|---|---|
| $C_{11}$ | 2225.3 | 2303.2 | -3.38 |
| $C_{22}$ | 2225.3 | 2303.1 | -3.38 |
| $C_{33}$ | 2225.0 | 2303.0 | -3.39 |
| $C_{12}$ | 1564.9 | 1612.1 | -2.93 |
| $C_{13}$ | 1565.0 | 1612.1 | -2.93 |
| $C_{23}$ | 1564.8 | 1612.1 | -2.94 |
| $C_{44}$ | 1041.0 | 1106.7 | -5.94 |
| $C_{55}$ | 1041.0 | 1106.7 | -5.93 |
| $C_{66}$ | 1040.9 | 1106.7 | -5.94 |
| $E_1$ | 932.96 | 975.51 | -4.36 |
| $E_2$ | 933.14 | 975.46 | -4.34 |
| $E_3$ | 932.85 | 975.40 | -4.36 |
| $G_{12}$ | 1040.93 | 1106.66 | -5.94 |
| $G_{13}$ | 1041.01 | 1106.67 | -5.93 |
| $G_{23}$ | 1040.98 | 1106.73 | -5.94 |
| $\nu_{12}$ | 0.41278 | 0.41174 | +0.25 |
| $\nu_{13}$ | 0.41305 | 0.41179 | +0.31 |
| $\nu_{23}$ | 0.41288 | 0.41179 | +0.26 |

The normal terms differ by a uniform -3 %, the shears by -6 % and the Poisson ratios by +0.3 %, with
cubic symmetry intact in both. Because the solid column equals SwiftComp digit for digit, these
percentages measure the thin-shell reduction itself: the surface reaches $t/R \approx 0.2$ near the
saddle necks, where a midsurface model slightly under-stiffens transverse shear and under-counts
material at the neck curvature. The shell reaches this accuracy with 26 k elements and 79 k degrees
of freedom, against 192 k tets and 165 k degrees of freedom.

% source: docs/tutorials/solid_props_3d_sg.md (note on the shell column); examples/OpenSG_shell/4_get_solid_props_from_shell_3D_SG/schwarz_p_3Dshell_C3D.out
:::{note}
The shell column above was computed with each triangle assembled as a four-node element with a
repeated node. The shipped `schwarz_p_3Dshell_C3D.out` of {doc}`shell_solid_3d_tpms` uses the
native three-node MITC3 triangle and gives $C_{11}$ = 2.2201948E+009 Pa, $C_{44}$ = 1.0093168E+009 Pa
and $E_1$ = 9.2565619E+008 Pa. A fresh run of `solid_vs_shell_compare.py` reads that file, so its
shell row differs from the row above.
:::

## Files

| File | Content |
|---|---|
| `sample_<n>/Sample_<n>.sc` | the TPMS SG deck: linear tets, aluminium |
| `sample_<n>/Sample_<n>.sc.k` | SwiftComp's effective stiffness, compliance and constants for the deck (MPa) |
| `sample_<n>/run_sample_<n>.py` | the periodic run and the comparison with the `.sc.k` |
| `sample_<n>/Sample_<n>.out` | stiffness, compliance and engineering constants, timed (Pa) |
| `sample_<n>/Sample_<n>_msg_solid.out` | the per-unit-cell $6\times6$ (Pa) with $\omega$ and the solve time |
| `sample_<n>/Sample_<n>_compare.dat` | OpenSG against SwiftComp, nine terms |
| `compare_boundary_modes.py` | both samples, aperiodic and periodic; writes `boundary_compare.dat` and `Sample_<n>_<mode>.out` |
| `solid_vs_shell_compare.py` | writes `solid_vs_shell_compare.dat` |
| `tpms_solid_props.py` | the same homogenization with $E$ = 70 GPa; writes `Sample_<n>_msg_solid.out` in the example folder |
| `sc_to_3dyaml.py` | converts each deck to a 3-D solid yaml `Sample_<n>_3Dsolid.yaml` ($E$ = 70 GPa) and checks that every tet has a nonzero volume |
| `Sample_<n>.ff` | a macro state for the SG, read by `opensg_solid Sample_<n>.yaml D` |
