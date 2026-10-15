# Equivalent solid properties of a Schwarz-P shell

## Problem description

This example computes the equivalent 3-D solid stiffness of a Schwarz-P triply periodic minimal
surface (TPMS) unit cell made of a thin aluminium sheet. The sheet is meshed on its surface with
shell elements, so the structure gene (SG) is a 3-D shell SG and the engine is `opensg_shell`. The
header asks for the solid macro model (`n_model: 3`); the result is the $6\times6$ Cauchy continuum
stiffness $C$ of the cell, its compliance and the orthotropic engineering constants. The wall
carries a Reissner-Mindlin (RM) plate law, so the sheet thickness is an input value, not a new mesh.

% source: examples/OpenSG_shell/4_get_solid_props_from_shell_3D_SG/schwarz_p_3Dshell_C3D.out (banner)
The mesh of {numref}`fig-schwarz-p-shell` fills a unit cube with 13536 nodes and 26360 three-node
triangles (MITC3, the three-node triangle with mixed interpolation of tensorial components). The surface is smooth: no shell edge is shared by more than two elements, so
there are no junction lines. By default opposite faces, edges and corners of the cell are tied by
periodicity in all three directions. The result is compared with the aperiodic boundary treatment
on the same yaml, and with the solid route on a tetrahedral mesh of the same surface
({doc}`solid_equivalent_3d_tpms`).

```{figure} ../_static/schwarz_p_mesh.png
:width: 60%
:name: fig-schwarz-p-shell

The Schwarz-P shell SG: the triangulated surface of the unit cell, axes $y_1$, $y_2$, $y_3$.
```

% source: examples/OpenSG_shell/4_get_solid_props_from_shell_3D_SG/schwarz_p_3Dshell.yaml
| Name | Density | $E_1$ | $E_2$ | $E_3$ | $G_{12}$ | $G_{13}$ | $G_{23}$ | $\nu_{12}$ | $\nu_{13}$ | $\nu_{23}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| | kg/m³ | Pa | Pa | Pa | Pa | Pa | Pa | | | |
| alu | 2700.0 | 6.900000e+10 | 6.900000e+10 | 6.900000e+10 | 2.653846e+10 | 2.653846e+10 | 2.653846e+10 | 0.300000 | 0.300000 | 0.300000 |

% source: examples/OpenSG_shell/4_get_solid_props_from_shell_3D_SG/schwarz_p_3Dshell.yaml
| Name | Layer | Material | Ply thickness | Orientation |
|---|---|---|---|---|
| | | | length unit of the nodes | ° |
| layup_0 | 1 | alu | 3.645700e-02 | 0.0 |

The cell is a unit cube, so the ply thickness is a fraction of the cell edge. Input files are in
`examples/OpenSG_shell/4_get_solid_props_from_shell_3D_SG/`: `schwarz_p_3Dshell.yaml`, the gmsh mesh
`shell_msh/schwarz_p_D2_shell.msh` it was converted from, and the driver scripts.

## Input

The header and the two blocks a user edits, verbatim; the mesh blocks `nodes`, `elements`,
`elementOrientations` and `sets` follow them in the file.

% source: examples/OpenSG_shell/4_get_solid_props_from_shell_3D_SG/schwarz_p_3Dshell.yaml
```yaml
n_model: 3      # 1 = beam, 2 = plate (opensg_solid), 3 = solid -- the macro model this SG homogenizes to
refined: 0      # 0 = classical (beam EB 4x4, KL wall); 1 = shear-refined (beam Timoshenko 6x6, RM wall)
msg: shell      # the ENGINE this SG belongs to (opensg_shell); `opensg <yaml>` dispatches on it
sections:
- type: shell
  elementSet: layup_0
  layup:
  - - alu
    - 3.645700e-02
    - 0.0
materials:
- name: alu
  density: 2700.0
  elastic:
    E: [6.900000e+10, 6.900000e+10, 6.900000e+10]
    G: [2.653846e+10, 2.653846e+10, 2.653846e+10]
    nu: [0.300000, 0.300000, 0.300000]
```

`make_schwarz_yaml.py` writes this yaml from the gmsh mesh through
`opensg_shell.helper.msh_to_yaml.convert`. The lines you change are the paths, the wall thickness
and the material:

% source: examples/OpenSG_shell/4_get_solid_props_from_shell_3D_SG/make_schwarz_yaml.py
```python
MSH = "shell_msh/schwarz_p_D2_shell.msh"
YAML = "schwarz_p_3Dshell.yaml"
T = 0.036457           # Sample_2, volume-consistent
E, nu = 69.0e9, 0.30
```

Two optional header keys change the analysis. `aperiodic: 1` replaces periodicity by the boundary
solution (below). `omega:` overrides the SG measure $\omega$, which by default is the node
bounding-box volume: the volume the equivalent continuum occupies and the cell the periodic map
ties. The `.out` is written per unit-cell volume, so its moduli compare entry by entry with the
SwiftComp `.K` of a solid cell.

## Run

From the example folder:

```bash
python make_schwarz_yaml.py
```

```bash
opensg schwarz_p_3Dshell.yaml
```

% source: examples/OpenSG_shell/4_get_solid_props_from_shell_3D_SG/schwarz_p_3Dshell_C3D.out
The command prints the law under the title `Cauchy Continuum Stiffness Matrix  [11 22 33 23 13 12]:`,
then `Homogenization stored in schwarz_p_3Dshell_C3D.out` and the `Time taken` line (12.02 sec for
the shipped file). It also writes `schwarz_p_3Dshell_ABDG.out`, the wall plate law. The same solve
through the Python API:

```python
from opensg_shell.sg_homo import shell_sg3d

r = shell_sg3d("schwarz_p_3Dshell.yaml")                        # boundary="periodic" (default)
r = shell_sg3d("schwarz_p_3Dshell.yaml", boundary="aperiodic")  # non-periodic SGs (explicit)
r["C3D"], r["n_junction_edges"], r["solve_time"]
```

`python schwarz_solid_props.py` runs the periodic case through the API and draws
`schwarz_p_mesh.png`. `python compare_boundary_modes.py` runs both boundary treatments, keeps each
`.out` as `schwarz_t<T>_<mode>.out` and appends a block to `boundary_compare.dat`.

## Result

The `.out` opens with a banner that records the mesh, the element, the boundary treatment and the
drilling treatment (one Lagrange multiplier per element):

% source: examples/OpenSG_shell/4_get_solid_props_from_shell_3D_SG/schwarz_p_3Dshell_C3D.out
```text
 OpenSG msg-shell equivalent 3D solid (3-D shell SG, 13536 nodes, 26360 elems [26360 tri3/MITC3], 0 junction edges, periodic in 3 dirs, 26360 element drilling multipliers, per unit cell 1)
```

% source: examples/OpenSG_shell/4_get_solid_props_from_shell_3D_SG/schwarz_p_3Dshell_C3D.out
```{list-table} Effective Cauchy continuum stiffness of the Schwarz-P shell cell, Pa, order [11 22 33 23 13 12]
:name: tab-schwarz-c3d
:header-rows: 1

* -
  - 11
  - 22
  - 33
  - 23
  - 13
  - 12
* - 11
  - 2.2201948E+009
  - 1.5653259E+009
  - 1.5652745E+009
  - 2.5188640E+005
  - -8.8448528E+004
  - 4.9935761E+004
* - 22
  - 1.5653259E+009
  - 2.2201678E+009
  - 1.5652141E+009
  - 3.4256118E+004
  - -2.6519047E+005
  - 1.1721896E+005
* - 33
  - 1.5652745E+009
  - 1.5652141E+009
  - 2.2201780E+009
  - 8.3591243E+004
  - -5.8670642E+004
  - 2.5572206E+005
* - 23
  - 2.5188640E+005
  - 3.4256118E+004
  - 8.3591243E+004
  - 1.0093168E+009
  - -6.4916473E+004
  - 9.6973947E+004
* - 13
  - -8.8448528E+004
  - -2.6519047E+005
  - -5.8670642E+004
  - -6.4916473E+004
  - 1.0092781E+009
  - -9.4144405E+004
* - 12
  - 4.9935761E+004
  - 1.1721896E+005
  - 2.5572206E+005
  - 9.6973947E+004
  - -9.4144405E+004
  - 1.0092871E+009
```

% source: examples/OpenSG_shell/4_get_solid_props_from_shell_3D_SG/schwarz_p_3Dshell_C3D.out
| $E_1$ | $E_2$ | $E_3$ | $G_{12}$ | $G_{13}$ | $G_{23}$ | $\nu_{12}$ | $\nu_{13}$ | $\nu_{23}$ |
|---|---|---|---|---|---|---|---|---|
| Pa | Pa | Pa | Pa | Pa | Pa | | | |
| 9.2565619E+008 | 9.2570448E+008 | 9.2577296E+008 | 1.0092870E+009 | 1.0092780E+009 | 1.0093168E+009 | 4.1355369E-001 | 4.1346885E-001 | 4.1341517E-001 |

The law is cubic: the three normal terms match, as do the three normal couplings and the three
shears, and the couplings that cubic symmetry forbids sit about four orders of magnitude below the
diagonal.

`schwarz_p_3Dshell_ABDG.out` holds the step-1 wall law, the $8\times8$ RM plate stiffness the
surface carries, rows $[\varepsilon_{11}\;\varepsilon_{22}\;2\varepsilon_{12}\;|\;\kappa_{11}\;\kappa_{22}\;\kappa_{12}+\kappa_{21}\;|\;2\gamma_{13}\;2\gamma_{23}]$
(with node coordinates in metres: A in N/m, B in N, D in N·m, the shear block in N/m):

% source: examples/OpenSG_shell/4_get_solid_props_from_shell_3D_SG/schwarz_p_3Dshell_ABDG.out
```text
 section 0: layup_0  layup [['alu', 0.036457, 0.0]]
 The Effective Reissner-Mindlin Plate Stiffness Matrix
 --------------------------------------------
     2.7643220E+009     8.2929659E+008     0.0000000E+000     5.0389443E+007     1.5116833E+007     0.0000000E+000     0.0000000E+000     0.0000000E+000
     8.2929659E+008     2.7643220E+009     0.0000000E+000     1.5116833E+007     5.0389443E+007     0.0000000E+000     0.0000000E+000     0.0000000E+000
     0.0000000E+000     0.0000000E+000     9.6751264E+008     0.0000000E+000     0.0000000E+000     1.7636304E+007     0.0000000E+000     0.0000000E+000
     5.0389443E+007     1.5116833E+007     0.0000000E+000     1.2246986E+006     3.6740959E+005     0.0000000E+000     0.0000000E+000     0.0000000E+000
     1.5116833E+007     5.0389443E+007     0.0000000E+000     3.6740959E+005     1.2246986E+006     0.0000000E+000     0.0000000E+000     0.0000000E+000
     0.0000000E+000     0.0000000E+000     1.7636304E+007     0.0000000E+000     0.0000000E+000     4.2864449E+005     0.0000000E+000     0.0000000E+000
     0.0000000E+000     0.0000000E+000     0.0000000E+000     0.0000000E+000     0.0000000E+000     0.0000000E+000     8.5193106E+008    -1.0506151E-003
     0.0000000E+000     0.0000000E+000     0.0000000E+000     0.0000000E+000     0.0000000E+000     0.0000000E+000    -1.0506151E-003     8.5193106E+008
```

The run uses the default laminate reference, the outer face of the wall on the mesh surface, so the
coupling block B is nonzero for this single ply. `opensg schwarz_p_3Dshell.yaml --center` references
the wall to its mid-surface instead.

### Boundary treatment

`boundary="aperiodic"` (or `aperiodic: 1` in the header) maps the boundary solution, for a unit cell
the affine macro field itself, onto the 720 bounding-box-face nodes of the 13536: the translational
fluctuations are zero there ($w_1=w_2=w_3=0$) and the rotations stay free. This fixes the rigid
modes and gives a kinematic upper bound on one cell.

% source: examples/OpenSG_shell/4_get_solid_props_from_shell_3D_SG/boundary_compare.dat (first block, t = 0.036457, and second block, t = 0.129330)
| term | t = 0.036457 aperiodic | periodic | % | t = 0.129330 aperiodic | periodic | % |
|---|---|---|---|---|---|---|
| $C_{11}$ | 2375.89 | 2225.31 | +6.767 | 9881.00 | 8916.81 | +10.813 |
| $C_{22}$ | 2375.92 | 2225.28 | +6.770 | 9881.00 | 8916.85 | +10.813 |
| $C_{33}$ | 2376.22 | 2225.03 | +6.795 | 9873.34 | 8916.17 | +10.735 |
| $C_{12}$ | 1537.47 | 1564.87 | -1.751 | 4921.09 | 5352.21 | -8.055 |
| $C_{13}$ | 1539.24 | 1564.95 | -1.643 | 4931.10 | 5352.71 | -7.877 |
| $C_{23}$ | 1538.99 | 1564.78 | -1.648 | 4931.00 | 5352.55 | -7.876 |
| $C_{44}$ | 1078.28 | 1040.98 | +3.583 | 4150.25 | 3961.63 | +4.761 |
| $C_{55}$ | 1078.32 | 1041.01 | +3.585 | 4150.26 | 3961.63 | +4.761 |
| $C_{66}$ | 1078.61 | 1040.93 | +3.620 | 4151.27 | 3961.51 | +4.790 |
| ndof | 81216 | 79056 | | 81216 | 79056 | |
| solve | 42.3 s | 66.5 s | | 42.2 s | 65.4 s | |

Values are per unit cell in MPa. The offset is the clamped-face boundary layer of a single cell,
the expected kinematic bound.
% source: src/opensg_shell/sg_homo.py (shell_sg3d docstring, boundary="aperiodic")
Clamping the rotations as well over-stiffens the cut edges further (+12 % instead of +7 % on
$C_{11}$). Report the periodic result for a periodic medium; request `aperiodic` only for an SG
that is not periodic.

% source: docs/tutorials/solid_props_3d_sg.md (note on the element); boundary_compare.dat (last block); banners of the schwarz_t*.out files
:::{note}
The two blocks above, the files `schwarz_t0.129330.out`, `schwarz_t0.129330_*.out`, `schwarz_t0.036414.out` and
`schwarz_t0.127481.out`, and the shell column of the next table were computed with each triangle
assembled as a four-node element with a repeated node; their banners do not list `tri3/MITC3`. The
last block of `boundary_compare.dat` repeats the t = 0.036457 comparison with the native MITC3
triangle: periodic $C_{11}$ = 2222.96 MPa, aperiodic 2376.36 MPa (+6.901 %), $C_{44}$ 1043.65 and
1082.48 MPa (+3.721 %), solves 10.5 s and 11.5 s (`schwarz_t0.036457_periodic.out`,
`schwarz_t0.036457_aperiodic.out`). These values, and those of the shipped
`schwarz_p_3Dshell_C3D.out` above, therefore differ from the tables in this section.
:::

### Shell versus solid on the same surface

Sample 2 of {doc}`solid_equivalent_3d_tpms` is a tetrahedral mesh of the same Schwarz-P sheet: its
midsurface area recovered from the tet mesh (2.3448) matches the shell mesh (2.3533), and the shell
row of `solid_vs_shell_compare.dat` (`Schwarz-P shell t=0.0365`) records a relative density of
0.0860 against the solid's 0.0857. Per unit cell, MPa:

% source: docs/tutorials/solid_props_3d_sg.md ("Shell versus solid on the same surface"); rho, C11, C12, C44 also in examples/OpenSG-solid/6_get_solid_props_from_3D_SG/solid_vs_shell_compare.dat
| term | shell SG (26 360 shell elements) | solid SG (191 957 tets) | % |
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

The solid column equals SwiftComp digit for digit, so the uniform -3 % on the normal terms, -6 % on
the shears and +0.3 % on the Poisson ratios measure the thin-shell reduction itself. The surface
reaches $t/R \approx 0.2$ near the saddle necks, where a midsurface model slightly under-stiffens
transverse shear. The shell reaches this accuracy with 26 k elements and 79 k degrees of freedom,
against 192 k tets and 165 k degrees of freedom.

## Files

| File | Content |
|---|---|
| `schwarz_p_3Dshell.yaml` | the 3-D shell SG (header, sections, materials, mesh) |
| `shell_msh/schwarz_p_D2_shell.msh` | gmsh 2.2 triangle mesh of the surface |
| `make_schwarz_yaml.py` | mesh to yaml, sets the thickness and the material |
| `schwarz_solid_props.py` | periodic run through the API, writes `schwarz_p_mesh.png` |
| `compare_boundary_modes.py` | aperiodic against periodic, appends to `boundary_compare.dat` |
| `schwarz_p_3Dshell_C3D.out` | stiffness, compliance and engineering constants, timed |
| `schwarz_p_3Dshell_ABDG.out` | the wall RM plate law and its compliance |
| `schwarz_t<T>[_<mode>].out` | per-thickness and per-mode copies of the `.out` |
| `boundary_compare.dat` | the boundary-treatment tables, one block per run |
| `schwarz_p_3Dshell.ff` | a macro state for this SG; the route is homogenization only |
