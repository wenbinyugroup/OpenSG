# Timoshenko beam properties and recovery of a honeycomb cross-section

## Problem description

These two examples model a composite honeycomb sandwich beam through a 2-D solid structure gene (SG)
of its cross-section, `RHC_SW_2UC_45.yaml` ({numref}`fig-rhc-beam-mesh`). The engine is
`opensg_solid` and the macro model is the beam (`n_model: 1`, `refined: 1`):

- `examples/OpenSG-solid/7_get_beam_props_from_SG` reduces the SG to the $6\times6$ Timoshenko
  stiffness in $[\epsilon_{11},\ \gamma_{12},\ \gamma_{13},\ \kappa_1,\ \kappa_2,\ \kappa_3]$. The
  classical Euler-Bernoulli $4\times4$ in $[\epsilon_{11},\ \kappa_1,\ \kappa_2,\ \kappa_3]$ comes out
  of the same zeroth-order warping solve $V_0$ and serves as a built-in cross-check.
- `examples/OpenSG-solid/8_get_beam_dehom_from_SG` recovers the 3-D strain and stress at every
  Gauss point of the SG under a macro beam strain (dehomogenization).

% source: docs/tutorials/msg_solid_beam_and_solid.md ("The example SG", "What it computes")
The SG has 4251 nodes and 6640 three-node triangles, 12 753 solve degrees of freedom (three per
node) plus four Lagrange rows that remove the rigid-body modes in the Karush-Kuhn-Tucker (KKT)
factorization. The first-order solve
$V_{1s}$ that adds the transverse shear reuses the same factorization. Coordinates are in mm and
moduli in MPa, so the stiffnesses come out in N, N·mm and N·mm². The beam route needs an SG of
dimension 2 or 3. The same SG is homogenized as a plate in {doc}`solid_plate_2d_honeycomb`.

```{figure} ../_static/rhc_2dsg_mesh.png
:width: 70%
:name: fig-rhc-beam-mesh

The honeycomb cross-section SG, elements coloured by material: the two face plies (mat 1 and mat 2) and the aluminium (mat 3).
```

% source: examples/OpenSG-solid/7_get_beam_props_from_SG/beam_homo_sg.py (User Input); mat_override.yaml
```{list-table} Material properties used by the drivers
:header-rows: 2
:name: tab-rhc-beam-materials

* - Id
  - Name
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
  - Angle
* -
  -
  - tonne/mm³
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
  - ply +45
  - 1.6e-9
  - 108e3
  - 8e3
  - 8e3
  - 4e3
  - 4e3
  - 3e3
  - 0.32
  - 0.32
  - 0.30
  - 45
* - 2
  - ply -45
  - 1.6e-9
  - 108e3
  - 8e3
  - 8e3
  - 4e3
  - 4e3
  - 3e3
  - 0.32
  - 0.32
  - 0.30
  - -45
* - 3
  - core
  - 2.7e-9
  - 69e3
  - 69e3
  - 69e3
  - 26.54e3
  - 26.54e3
  - 26.54e3
  - 0.30
  - 0.30
  - 0.30
  - 0
```

## Input

The yaml header selects the engine and the macro model:

% source: examples/OpenSG-solid/7_get_beam_props_from_SG/RHC_SW_2UC_45.yaml
```yaml
n_model: 1      # 1 = beam, 2 = plate, 3 = solid -- the macro model this SG homogenizes to
refined: 1      # 0 = classical (plate ABD / beam EB); 1 = shear-refined (plate ABDG / beam Timoshenko)
msg: solid      # the ENGINE this SG belongs to (opensg_solid); `opensg <yaml>` dispatches on it
```

The mesh follows as `nodes`, `cells` (0-based) and `mat_id`. The `materials` block holds the two
plies as SwiftComp type-2 blocks, $6\times6$ stiffness matrices already rotated to $\pm45^\circ$,
and the core as an isotropic type-0 block:

% source: examples/OpenSG-solid/7_get_beam_props_from_SG/RHC_SW_2UC_45.yaml (materials block, material 2 omitted)
```yaml
materials:
  1:
    type: 2
    aux: [0.0, 0.0]
    C:
    - [35696.6, 27696.6, 3251.1, 0.0, 0.0, -25368.7]
    - [27696.6, 35696.6, 3251.1, 0.0, 0.0, -25368.7]
    - [3251.1, 3251.1, 8917.8, 0.0, 0.0, -487.1]
    - [0.0, 0.0, 0.0, 3500.0, -500.0, 0.0]
    - [0.0, 0.0, 0.0, -500.0, 3500.0, 0.0]
    - [-25368.7, -25368.7, -487.1, 0.0, 0.0, 27958.5]
  3:
    type: 0
    aux: [0.0, 0.0]
    E: 69000.0
    nu: 0.3
```

The driver `beam_homo_sg.py` has no command-line arguments; its User Input block names the SG and
gives the materials:

% source: examples/OpenSG-solid/7_get_beam_props_from_SG/beam_homo_sg.py
```python
n_model = 1                    # 1: Beam; 2: Plate; 3: 3D elastic
name = "RHC_SW_2UC_45"         # reads <name>.yaml

material_param = jnp.array([
    (108e3, 8e3, 8e3, 4e3, 4e3, 3e3, 0.32, 0.32, 0.30),
    (108e3, 8e3, 8e3, 4e3, 4e3, 3e3, 0.32, 0.32, 0.30),
    (69e3, 69e3, 69e3, 26.54e3, 26.54e3, 26.54e3, 0.30, 0.30, 0.30)])
angles = jnp.array([45.0, -45.0, 0.0])
density = [1.6e-9, 1.6e-9, 2.7e-9]   # tonne/mm^3 per material (MPa-mm system):
                                     # CFRP plies, aluminum core -- edit to yours
############################################################

r = plate_homo_2d(name + ".yaml", material_param=material_param,
                    angles=angles, n_model=n_model, density=density)
```

% source: docs/tutorials/msg_solid_beam_and_solid.md ("The input you edit")
`material_param` is a $(n_{\rm mat}, 9)$ table of $[E_1, E_2, E_3, G_{12}, G_{13}, G_{23}, \nu_{12},
\nu_{13}, \nu_{23}]$ and `angles` is one rotation angle in degrees per material. Passing them
overrides the material blocks of the SG file. Since the type-2 blocks are pre-rotated, giving
`angles` together with the stored blocks would rotate them twice; rebuilding from engineering
constants and rotating in code is unambiguous. `mat_override.yaml` records the same values. Omit
both arguments to use the file's own blocks unchanged.

The recovery driver `beam_dehom_sg.py` repeats this block and adds the macro beam strain
$\bar{\epsilon} = [\epsilon_{11},\ \gamma_{12},\ \gamma_{13},\ \kappa_1,\ \kappa_2,\ \kappa_3]$, a
strain 6-vector, not a load. A load vector goes in through the other driver,
`dehom_fields(r, beam_force=[F1, F2, F3, M1, M2, M3])`; give exactly one of the two. The shipped
state is $\epsilon_{11} = 0.001$ with $\kappa_2 = 0.01$ mm⁻¹:

% source: examples/OpenSG-solid/8_get_beam_dehom_from_SG/beam_dehom_sg.py
```python
epsilon_bar = jnp.array([0.001, 0.0, 0.0, 0.0, 0.01, 0.0])
...
Gam, Sig, U = dehom_fields(r, epsilon_bar)
export_gauss(r, Gam, Sig, name + "_dehom", U_eqd=U, frame_kind="global")
```

% source: src/opensg_solid/sg_dehom.py (compute_fluctuations_gpu, dehom_fields)
For `n_model=1` (`refined: 1`) the recovery is the generalized-Timoshenko recovery of the msg-shell
route (`sg_dehom.compute_fluctuations_gpu`). The chain is seeded with the beam force $F$ itself (the
force you give, or $F = C_{\rm eff}\,\bar{\epsilon}$ for a strain), and the 1-D equilibrium of a
prismatic beam without distributed load gives its axial derivative exactly, $F' = R_1 F$ with
the fixed operator $R_1$ ($M_2' = F_3$, $M_3' = -F_2$, $F'' = 0$). The recovered field is the
zeroth-order (classical) part, the warping $V_0\,\epsilon$, plus the first-order correction, the
warping $V_1\,\epsilon'$ and the axial derivative $V_0\,\epsilon'$ of the zeroth-order one; the
correction carries the transverse shear, $\int\sigma_{12}\,dA = F_2$ and
$\int\sigma_{13}\,dA = F_3$. Nothing from the homogenization is recomputed: the result dict `r` carries the warping modes, the
material tables and the quadrature data, so you homogenize once and recover as often as you like.

% source: src/opensg_solid/sg_dehom.py (beam_expected_resultants); tests/unit/test_beam_dehom_timoshenko.py
:::{note}
Every beam state is a valid driver, the transverse shear included, and the recovery is linear
in the load: a pure $F_1$, $M_1$, $M_2$ or $M_3$ gives $F' = 0$ and the classical field exactly,
while $F_2$ and $F_3$ drive the first-order fields. On a section whose classical and shear
channels couple, the recovered $[F_1, M_1, M_2, M_3]$ under a shear force are the applied values
plus the coupling term of the Timoshenko reduction,
$\Delta F_{cl} = -A\,(I - QQ^{\mathsf T})\,A^{-1} B\,A^{-1} Q\,F_s$, with $A$ = `r["C_eff_EB"]`,
$B$ = `r["B_tim"]`, $F_s = [F_2, F_3]$ and $Q = [[0, 0], [0, 0], [0, -1], [1, 0]]$, so that
$Q F_s = [0, 0, -F_3, F_2]$. `sg_dehom.beam_expected_resultants(r, beam_force=F)` returns
$F + \Delta F_{cl}$, and `opensg <name>.yaml D` prints it.
$\Delta F_{cl}$ belongs to the recovery procedure, not to the section. VABS 4.1 recovers the same
field and the same excess on the same mesh, and the msg-shell ring recovery carries the same
term. It changes with the section origin: the same physical load on the same section meshed
about another origin gives another $\Delta F_{cl}$ and another field. A 3-D solution of the
prismatic beam closes every resultant at every section. The term is zero when the shear does
not couple to the classical channels, as on a doubly symmetric section. On the 0.4 m x 0.2 m
glass-epoxy test section of `tests/unit/test_beam_dehom_timoshenko.py` (+30 and -45 degree
halves, origin 0.148 m from the centre), a pure $F_3 = 1000$ N recovers $F_1 = -1218.6$ N.
The same load about the centre ($M_1 = 130$ N m there) recovers $F_1$ = 0 to round-off, and the
two stress fields differ by 37.9 % of max $|\sigma|$ (0.36 % under $F_2$). VABS 4.1, run on the
same section about the two origins, changes the same way: 36.3 % on its 3 x 3 Gauss points, equal
to OpenSG's change on the same points to $5\times10^{-11}$ of it, with the same
$F_1 = -1.21864\,F_3$ about the offset origin.
A `refined: 0` beam has no shear law: it recovers the classical field only and refuses a
nonzero $F_2$, $F_3$ or shear strain. The scope is a prismatic, untwisted beam without
distributed load.
:::

## Run

```bash
cd examples/OpenSG-solid/7_get_beam_props_from_SG
python beam_homo_sg.py
```

% source: examples/OpenSG-solid/7_get_beam_props_from_SG/beam_homo_sg.py (print statements)
The script prints the Timoshenko $6\times6$ under
`beam 6x6  [eps11 gam12 gam13 kappa1 kappa2 kappa3]  (Timoshenko):`, the Euler-Bernoulli $4\times4$
under `beam 4x4  [eps11 kappa1 kappa2 kappa3]  (EB, same run):` and the mass matrix, then:

```text
Homogenization stored in RHC_SW_2UC_45.out
Time taken: <seconds> sec
```

```bash
cd examples/OpenSG-solid/8_get_beam_dehom_from_SG
python beam_dehom_sg.py
```

The recovery driver prints the stiffness diagonal and the six maxima `max|Sig_xx|` to
`max|Sig_xy|`.

% source: docs/tutorials/msg_solid_beam_and_solid.md ("Run it", sections 1 and 2); src/opensg_solid/cli.py (analysis D)
The drivers exist because they override pre-rotated material blocks with engineering constants and
explicit angles, which the command line does not do. For a yaml whose own `type: 1` material blocks
carry their `angle:`, the recommended pattern, the same runs are one command each:

```bash
opensg RHC_SW_2UC_45.yaml
opensg RHC_SW_2UC_45.yaml D
```

`D` homogenizes, then recovers. The macro state comes from a `<name>.ff` next to the yaml when there
is one (a `0:` strain is used as given; a `1:`/`FF` beam force seeds the recovery chain directly,
$F' = R_1 F$), otherwise from `epsilon_bar:` in the header. Example 8 ships both, for the same state:

% source: examples/OpenSG-solid/8_get_beam_dehom_from_SG/RHC_SW_2UC_45.yaml; RHC_SW_2UC_45.ff
```yaml
epsilon_bar: [0.001, 0.0, 0.0, 0.0, 0.01, 0.0]   # macro beam strain for opensg_solid <yaml> D
```

```yaml
# generalized macro forces, beam  [F1 F2 F3 M1 M2 M3]
FF:    [9899.3087, 6.9807273, 0.0032515777, -31861.511, 21895742, -959.63908]
```

See {doc}`../user_guide/recovery` for the `.ff` format and the output frames.

## Result

### Timoshenko stiffness

% source: examples/OpenSG-solid/7_get_beam_props_from_SG/RHC_SW_2UC_45.out
```{code-block} text
:caption: Timoshenko stiffness from RHC_SW_2UC_45.out, rows and columns [eps11 gam12 gam13 kappa1 kappa2 kappa3] (N, N·mm, N·mm²)
:name: lst-rhc-beam-timo

 OpenSG msg-solid beam model, omega 1

 The Effective Timoshenko Beam Stiffness Matrix
 --------------------------------------------
     9.8992866E+006     6.9973096E+003     1.2577025E+001    -7.7065225E-001     2.2740513E+000    -8.6717879E-001
     6.9973096E+003     1.9074524E+006     5.9324366E+001    -3.4044008E+001    -1.6298116E+000    -6.0670321E-002
     1.2577025E+001     5.9324366E+001     9.3619028E+005    -8.4787042E+001    -9.3250225E-001     8.5948342E-001
    -7.7065138E-001    -3.4044008E+001    -8.4787042E+001     2.9774561E+008    -3.1862083E+006    -7.3057137E+003
     2.2740485E+000    -1.6298116E+000    -9.3250225E-001    -3.1862083E+006     2.1895745E+009    -9.5964309E+004
    -8.6717825E-001    -6.0670321E-002     8.5948342E-001    -7.3057137E+003    -9.5964309E+004     9.2899273E+008
```

The file continues with the compliance matrix and `The 6X6 Mass Matrix` built from the densities
(mass per unit length 4.8454171E-007 tonne/mm), and closes with `Time taken: 2.43 sec`. The banner
reports $\omega = 1$: for a beam on a 2-D SG the matrix is the total sectional stiffness, not a
per-area density, and no engineering-constants block is printed.

% source: docs/tutorials/msg_solid_beam_and_solid.md ("What the numbers mean", section 1); values in examples/OpenSG-solid/7_get_beam_props_from_SG/RHC_SW_2UC_45.out
| Entry | Strain | Value | Reads as |
|---|---|---|---|
| $C_{11}$ | $\epsilon_{11}$ | $9.8992866\times10^{6}$ N | axial stiffness |
| $C_{22}$ | $\gamma_{12}$ | $1.9074524\times10^{6}$ N | transverse shear stiffness, direction 2 |
| $C_{33}$ | $\gamma_{13}$ | $9.3619028\times10^{5}$ N | transverse shear stiffness, direction 3 |
| $C_{44}$ | $\kappa_1$ | $2.9774561\times10^{8}$ N·mm² | torsional stiffness |
| $C_{55}$ | $\kappa_2$ | $2.1895745\times10^{9}$ N·mm² | bending stiffness about axis 2 |
| $C_{66}$ | $\kappa_3$ | $9.2899273\times10^{8}$ N·mm² | bending stiffness about axis 3 |
| $C_{45}$ | $\kappa_1$, $\kappa_2$ | $-3.1862083\times10^{6}$ N·mm² | twist-bend coupling from the $\pm45^\circ$ plies |

% source: docs/tutorials/msg_solid_beam_and_solid.md ("What the numbers mean", section 1)
With couplings present, the engineering stiffnesses are the inverses of the compliance diagonal:
$1/S_{11} = 9.8993\times10^{6}$ against $C_{11} = 9.8993\times10^{6}$ and
$1/S_{44} = 2.9774\times10^{8}$ against $C_{44} = 2.9775\times10^{8}$, so on this section the
couplings shift the engineering values by less than $0.002\,\%$. $\lvert C_{45}\rvert$ is a factor
of 93 below $C_{44}$ and 687 below $C_{55}$, yet it is the physical signature of the $\pm45^\circ$
layup.

### Euler-Bernoulli cross-check

`r["C_eff_EB"]` comes from the same $V_0$ solve, so its four entries must reproduce the classical
diagonal of the Timoshenko matrix; the two differ only by the refinement that produces the shear
terms. They agree to at least five significant figures ({numref}`tab-rhc-eb-timo`). $C_{22}$ and
$C_{33}$ exist only in the Timoshenko matrix, which is the reason to run the refined model.

% source: EB column, docs/tutorials/msg_solid_beam_and_solid.md ("The built-in cross-check"); Timoshenko column, examples/OpenSG-solid/8_get_beam_dehom_from_SG/RHC_SW_2UC_45_beam_Timo.out
```{list-table} Classical terms of the Euler-Bernoulli 4x4 and the Timoshenko 6x6 (N, N·mm²)
:header-rows: 1
:name: tab-rhc-eb-timo

* - Term
  - Euler-Bernoulli $4\times4$
  - Timoshenko $6\times6$
* - extension
  - $9.89926177\times10^{6}$
  - $9.89928657\times10^{6}$
* - torsion
  - $2.97746\times10^{8}$
  - $2.97745606\times10^{8}$
* - bending 2
  - $2.18957447\times10^{9}$
  - $2.18957452\times10^{9}$
* - bending 3
  - $9.28992811\times10^{8}$
  - $9.28992729\times10^{8}$
```

### Recovered fields

`dehom_fields` returns `Gam` (strain), `Sig` (stress) and `U`, the fluctuation-only displacement
(the zeroth- plus first-order warping), each over elements and Gauss points. Units follow the SG
input, MPa here. {numref}`fig-rhc-beam-sxx` shows $\sigma_{xx}$ and
{numref}`tab-rhc-beam-max` the maxima of a run of the driver.

```{figure} ../_static/pv_beam_dehom_sxx.png
:width: 70%
:name: fig-rhc-beam-sxx

Recovered $\sigma_{xx}$ (MPa) at the Gauss points of the honeycomb SG under $\epsilon_{11} = 0.001$ and $\kappa_2 = 0.01$ mm⁻¹.
```

% source: docs/tutorials/msg_solid_beam_and_solid.md ("What the numbers mean", section 2)
```{list-table} Maximum absolute stress and strain per component
:header-rows: 1
:name: tab-rhc-beam-max

* - Component
  - $\max\lvert\sigma\rvert$ (MPa)
  - $\max\lvert\epsilon\rvert$
* - $xx$
  - $1.52933\times10^{4}$
  - $2.33938\times10^{-1}$
* - $yy$
  - $3.46016\times10^{3}$
  - $2.26517\times10^{-1}$
* - $zz$
  - $5.54000\times10^{2}$
  - $8.78725\times10^{-2}$
* - $yz$
  - $6.03413\times10^{2}$
  - $1.34434\times10^{-1}$
* - $xz$
  - $1.17652\times10^{3}$
  - $1.70442\times10^{-1}$
* - $xy$
  - $2.95939\times10^{3}$
  - $1.02868\times10^{-1}$
```

The strain state carries a small shear force through the law's couplings
($C_{\rm eff}\,\bar{\epsilon}$ has $F_2 = -6.98$ N and $F_3 = -3.3\times10^{-3}$ N), and its
first-order shear part is in these maxima.

% source: docs/tutorials/msg_solid_beam_and_solid.md ("What the numbers mean", section 2)
Check the axial field first, because beam kinematics predict it:
$\epsilon_{xx} \approx \bar{\epsilon}_{11} + \kappa_2\,y_3$ plus a warping correction. The first row
of `RHC_SW_2UC_45_dehom.txt` sits at $y_3 = 23.29375$ mm and reports
$\epsilon_{xx} = 2.339375\times10^{-1}$, against $0.001 + 0.01\times23.29375 = 0.2339375$, an exact
match and the table maximum, as it must be at the extreme fibre. The transverse and shear components
exist only because the cross-section is heterogeneous and warps; no beam theory supplies them. The
shipped state is a demonstration magnitude, not a service load: scale it to your own macro state
before reading the stresses as failure indices.

## Files

% source: docs/tutorials/msg_solid_beam_and_solid.md ("What comes out", sections 1 and 2)
| File | Content |
|---|---|
| `RHC_SW_2UC_45.yaml` | the 2-D SG (example 8 adds `epsilon_bar:` to the header) |
| `RHC_SW_2UC_45.sc`, `RHC_SW_2UC_45.msh` | the same mesh as a SwiftComp deck and as a gmsh mesh |
| `mat_override.yaml` | the engineering constants and angles the drivers use |
| `RHC_SW_2UC_45.ff` | a macro state for `opensg RHC_SW_2UC_45.yaml D`: a `0:` strain, or a `1:` force that seeds the chain directly |
| `beam_homo_sg.py` | example 7 driver: Timoshenko and Euler-Bernoulli stiffness, mass matrix |
| `beam_dehom_sg.py` | example 8 driver: homogenization, then recovery under `epsilon_bar` |
| `RHC_SW_2UC_45.out` | Timoshenko stiffness, compliance and mass matrix, timed; rewritten on every run |
| `RHC_SW_2UC_45_beam_Timo.out` | the bare $6\times6$ written by `np.savetxt` with one header line, ready for `np.loadtxt` |
| `RHC_SW_2UC_45_dehom.txt` | Gauss table: `GP_Index`, `Coord_X`, `Coord_Y`, six `Eps_*`, six `Sig_*`; 19 920 rows (6640 triangles, 3 points each) |
| `RHC_SW_2UC_45_dehom.vtk` | the exploded SG mesh as an ASCII unstructured grid for ParaView: 6640 triangles, each with its own corners and the mean of its 3 Gauss values as `CELL_DATA`: six `Eps_*`, six `Sig_*`, `mat_id`, `U1 U2 U3` and the vector `U` (the displacement, in the SG axes) |
| `RHC_SW_2UC_45_dehom_global.SM`, `_dehom_global.EM`, `_dehom.U` | stress and strain in the SG axes (the `_global` tag; the ply-axes files of `opensg <name>.yaml D` keep the bare `_dehom.SM`/`.EM` names) and the fluctuation displacement: `x y z` columns (the SG coordinates in the last `n_sg` of them), then components in the order 11 22 33 12 13 23 (strain shears engineering, `2E12 2E13 2E23`) |
| `RHC_SW_2UC_45_dehom_elemental_global.SM`, `_elemental_global.EM`, `_elemental.U` | the per-element means of the same fields, one row per element at its centroid |
| `RHC_SW_2UC_45_mesh.png` | the SG, elements coloured by material |

The `.txt` and `.vtk` files use the SwiftComp component order `xx yy zz yz xz xy`; the `.SM`, `.EM`
and `.U` files reorder to `11 22 33 12 13 23` on write. The recovery files are written by each run
and are not shipped. {doc}`../user_guide/output_files` describes every format.

`opensg RHC_SW_2UC_45.yaml D` runs the same recovery from the command line. It writes the ply-axes
fields by default (`--material`: each element's `orientation` frame first, then the block angle) as
`<name>_dehom.SM`/`.EM`, and the SG-axes fields with `--global` as `<name>_dehom_global.SM`/`.EM`;
it refuses `orientation` rows that are not proper rotations (orthonormal, right-handed) to the
precision of rows written with 6 decimals, and it ends a refused run with the one-line reason.
