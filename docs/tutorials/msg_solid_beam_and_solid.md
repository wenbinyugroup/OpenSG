# Beam and Equivalent-Solid Properties (msg-solid)

`opensg_solid` runs beam, plate and equivalent-3-D-solid homogenization through **one**
entry point, `opensg_solid.sg_homo.plate_homo_2d`. The structure gene (SG) — a 1-D, 2-D or
3-D mesh of the repeating heterogeneity — is the input; `n_model` selects which macro model
the SG is reduced to:

| `n_model` | macro model | `r["C_eff"]` | matrix title in the `.out` |
|---|---|---|---|
| 1 | beam | $6\times6$ Timoshenko, strains $[\epsilon_{11},\ \gamma_{12},\ \gamma_{13},\ \kappa_1,\ \kappa_2,\ \kappa_3]$ | `The Effective Timoshenko Stiffness Matrix` |
| 2 | plate | $6\times6$ ABD, $[N_{11}, N_{22}, N_{12}, M_{11}, M_{22}, M_{12}]$ | `The Effective Classical Plate Stiffness Matrix` |
| 3 | 3-D elastic | $6\times6$ solid law, Voigt $[11, 22, 33, 23, 13, 12]$ | `The Effective Stiffness Matrix` |

Every `plate_homo_2d` call writes a timed SwiftComp-`.K`-layout `<base>.out` (effective
stiffness, then effective compliance) and, unless you pass `plot=False`, a `<base>_mesh.png`
with the elements coloured by material. The 3-D solid law (`n_model=3`) is the only one that
also prints an engineering-constants block, because it is the only one whose macro law is an
ordinary material stiffness.

This page covers the **beam** route (homogenization and dehomogenization) and the
**equivalent-solid** route from a 2-D SG. The equivalent solid from a fully 3-D SG has its own
page, *Equivalent 3-D Solid Properties from a 3-D SG (TPMS)*.

## 1. Beam properties from a 2-D SG

### What it computes

`n_model=1` reduces a cross-sectional SG to the Timoshenko beam constitutive law, relating the
axial force, two transverse shear forces, torque and two bending moments to the six
generalized strains $[\epsilon_{11},\ \gamma_{12},\ \gamma_{13},\ \kappa_1,\ \kappa_2,\ \kappa_3]$.
Internally this is the Beam_solid KKT engine merged into `sg_assembly`/`sg_homo`: the
zeroth-order warping solve $V_0$ is run under four rigid-body Lagrange constraints, the
$l$-chain first-order solve $V_{1s}$ reuses the same factorization, and the pair is reduced to
the $6\times6$. The classical Euler–Bernoulli $4\times4$ in
$[\epsilon_{11},\ \kappa_1,\ \kappa_2,\ \kappa_3]$ falls out of the same $V_0$ solve and rides
along in `r["C_eff_EB"]` at no extra cost. The beam route requires an SG of dimension 2 or 3.

### The example SG

`examples/OpenSG-solid/7_get_beam_props_from_SG` homogenizes `RHC_SW_2UC_45.yaml`, a 2-D
composite honeycomb cross-section: 4251 nodes, 6640 three-node triangles, 12 753 solve DOFs
(unique periodic DOFs, three per master node; the KKT factorization carries four extra
Lagrange rows). Coordinates are in mm and moduli in MPa, so the resulting stiffnesses are in
N and N·mm².

![Honeycomb cross-section SG, elements coloured by material](../_static/rhc_2dsg_mesh.png)

### The input you edit

The runner has no command-line arguments — the SG file and the materials are the User Input
block at the top of the script. The three materials are the two $\pm45^\circ$ plies and the
aluminium core:

| id | material | $E_1, E_2, E_3$ (MPa) | $G_{12}, G_{13}, G_{23}$ (MPa) | $\nu_{12}, \nu_{13}, \nu_{23}$ | angle |
|---|---|---|---|---|---|
| 1 | ply | 108 000, 8 000, 8 000 | 4 000, 4 000, 3 000 | 0.32, 0.32, 0.30 | $+45^\circ$ |
| 2 | ply | 108 000, 8 000, 8 000 | 4 000, 4 000, 3 000 | 0.32, 0.32, 0.30 | $-45^\circ$ |
| 3 | aluminium core | 69 000, 69 000, 69 000 | 26 540, 26 540, 26 540 | 0.30, 0.30, 0.30 | none |

```python
import numpy as np
import jax.numpy as jnp

from opensg_solid.sg_homo import plate_homo_2d

n_model = 1                    # 1: Beam; 2: Plate; 3: 3D elastic
name = "RHC_SW_2UC_45"         # reads <name>.yaml

material_param = jnp.array([
    (108e3, 8e3, 8e3, 4e3, 4e3, 3e3, 0.32, 0.32, 0.30),
    (108e3, 8e3, 8e3, 4e3, 4e3, 3e3, 0.32, 0.32, 0.30),
    (69e3, 69e3, 69e3, 26.54e3, 26.54e3, 26.54e3, 0.30, 0.30, 0.30)])
angles = jnp.array([45.0, -45.0, 0.0])

r = plate_homo_2d(name + ".yaml", material_param=material_param,
                  angles=angles, n_model=n_model)
np.set_printoptions(precision=5)
print("beam 6x6  [eps11 gam12 gam13 kappa1 kappa2 kappa3]  (Timoshenko):")
print(r["C_eff"])
print("beam 4x4  [eps11 kappa1 kappa2 kappa3]  (EB, same run):")
print(r["C_eff_EB"])
```

`material_param` is a $(n_{\rm mat}, 9)$ table of engineering constants
$[E_1, E_2, E_3, G_{12}, G_{13}, G_{23}, \nu_{12}, \nu_{13}, \nu_{23}]$ and `angles` is one
rotation angle in degrees per material ($0.0$ = no rotation). Passing them **overrides** the
material blocks stored in the SG file. That override is the intended route here: SwiftComp
`.sc` type-2 material blocks hold **pre-rotated** ply stiffness matrices, so giving `angles`
together with the stored blocks would rotate them twice. Rebuilding from engineering
constants and rotating in code is unambiguous. The same values are recorded next to the
runner in `mat_override.yaml` for reference. Omit both arguments to use the file's own
material blocks unchanged.

### Run it

```bash
cd examples/OpenSG-solid/7_get_beam_props_from_SG
```

```bash
python beam_homo_sg.py
```

### What comes out

| file | contents |
|---|---|
| `RHC_SW_2UC_45.out` | the timed SwiftComp-layout report: `The Effective Timoshenko Stiffness Matrix` followed by `The Effective Timoshenko Compliance Matrix`, banner ` OpenSG msg-solid beam model, omega 1`, closing ` Time taken: 8.94 sec` |
| `RHC_SW_2UC_45_mesh.png` | the SG mesh, elements coloured by material, no title |

The banner reports $\omega = 1$: for a beam model on a 2-D SG the measure is unity, so the
$6\times6$ is the **total sectional stiffness**, not a per-area density. No engineering-constants
block is printed, because a beam stiffness is not a material stiffness.

### What the numbers mean

The shipped `RHC_SW_2UC_45.out` stiffness block reads:

```
     9.8992866E+006     6.9973096E+003     1.2577025E+001    -7.7065215E-001     2.2740495E+000    -8.6717860E-001
     6.9973096E+003     1.9074524E+006     5.9324366E+001    -3.4044009E+001    -1.6298115E+000    -6.0670378E-002
     1.2577025E+001     5.9324366E+001     9.3619028E+005    -8.4787043E+001    -9.3250231E-001     8.5948371E-001
    -7.7065184E-001    -3.4044009E+001    -8.4787043E+001     2.9774561E+008    -3.1862083E+006    -7.3057137E+003
     2.2740506E+000    -1.6298115E+000    -9.3250231E-001    -3.1862083E+006     2.1895745E+009    -9.5964309E+004
    -8.6717875E-001    -6.0670378E-002     8.5948371E-001    -7.3057137E+003    -9.5964309E+004     9.2899273E+008
```

| entry | conjugate pair | value | reads as |
|---|---|---|---|
| $C_{11}$ | $\epsilon_{11}$ | $9.8992866\times10^{6}$ N | axial (extensional) stiffness |
| $C_{22}$ | $\gamma_{12}$ | $1.9074524\times10^{6}$ N | transverse shear stiffness, direction 2 |
| $C_{33}$ | $\gamma_{13}$ | $9.3619028\times10^{5}$ N | transverse shear stiffness, direction 3 |
| $C_{44}$ | $\kappa_1$ | $2.9774561\times10^{8}$ N·mm² | torsional stiffness |
| $C_{55}$ | $\kappa_2$ | $2.1895745\times10^{9}$ N·mm² | bending stiffness about axis 2 |
| $C_{66}$ | $\kappa_3$ | $9.2899273\times10^{8}$ N·mm² | bending stiffness about axis 3 |
| $C_{45}$ | $\kappa_1$–$\kappa_2$ | $-3.1862083\times10^{6}$ N·mm² | twist–bend coupling from the $\pm45^\circ$ plies |

Read the diagonal only as a first orientation. When couplings are present the engineering
stiffnesses are the inverses of the **compliance** diagonal, which is why the same `.out`
prints the compliance: here $1/S_{11} = 9.8993\times10^{6}$ against $C_{11} = 9.8993\times10^{6}$
and $1/S_{44} = 2.9774\times10^{8}$ against $C_{44} = 2.9775\times10^{8}$, so on this section the
couplings shift the engineering values by less than $0.002\,\%$. The twist–bend term
$C_{45}$ is two to three orders below the diagonals it couples —
$\lvert C_{45}\rvert = 3.1862083\times10^{6}$ against $C_{44} = 2.9774561\times10^{8}$, a factor
of 93, and against $C_{55} = 2.1895745\times10^{9}$, a factor of 687 — but is the physically
meaningful signature of the $\pm45^\circ$ layup.

### The built-in cross-check: Euler–Bernoulli against Timoshenko

`r["C_eff_EB"]` is the classical $4\times4$ from the same $V_0$ solve. Its four entries must
reproduce the classical rows of the Timoshenko $6\times6$, since the two differ only by the
$l$-chain refinement that produces the shear terms. On this SG they agree to at least five
significant figures on every classical diagonal:

| term | EB $4\times4$ | Timoshenko $6\times6$ |
|---|---|---|
| extension | $9.89926177\times10^{6}$ | $9.89928657\times10^{6}$ |
| torsion | $2.97745363\times10^{8}$ | $2.97745606\times10^{8}$ |
| bending 2 | $2.18957447\times10^{9}$ | $2.18957452\times10^{9}$ |
| bending 3 | $9.28992811\times10^{8}$ | $9.28992729\times10^{8}$ |

The two transverse shear stiffnesses $C_{22}$ and $C_{33}$ exist **only** in the Timoshenko
$6\times6$; that is the entire reason to run `n_model=1` rather than read the EB block.

## 2. Beam dehomogenization: local fields under a macro beam state

### What it computes

Homogenization is only half of MSG. Dehomogenization takes a macro beam state and recovers
the pointwise 3-D strain and stress inside the SG, at every element Gauss point. For
`n_model=1` this runs the generalized-Timoshenko recovery ladder
(`sg_dehom.compute_fluctuations_gpu`): a successive-derivative product-rule chain
$F' = R\,F$ seeded with $F_{1d} = C_{\rm eff}^{-1}\bar{\epsilon}$, which builds the
shear-corrected states and the zeroth- and first-order warping displacement fields. Nothing
from the homogenization is recomputed — the result dict `r` carries the warping modes, the
material tables and the quadrature data, so you homogenize once and recover as often as you
like.

### The input you edit

`examples/OpenSG-solid/8_get_beam_dehom_from_SG` runs the same honeycomb SG. Beyond the
material block, the one new user input is `epsilon_bar`, the **macro beam strain**
$[\epsilon_{11},\ \gamma_{12},\ \gamma_{13},\ \kappa_1,\ \kappa_2,\ \kappa_3]$ — a strain
6-vector, not a load vector. The shipped driver is $\epsilon_{11}=0.001$ combined with
$\kappa_2 = 0.01$ mm⁻¹:

```python
import numpy as np
import jax.numpy as jnp

from opensg_solid.sg_homo import plate_homo_2d
from opensg_solid.sg_dehom import dehom_fields, export_gauss

n_model = 1                    # 1: Beam; 2: Plate; 3: 3D elastic
name = "RHC_SW_2UC_45"         # reads <name>.yaml

material_param = jnp.array([
    (108e3, 8e3, 8e3, 4e3, 4e3, 3e3, 0.32, 0.32, 0.30),
    (108e3, 8e3, 8e3, 4e3, 4e3, 3e3, 0.32, 0.32, 0.30),
    (69e3, 69e3, 69e3, 26.54e3, 26.54e3, 26.54e3, 0.30, 0.30, 0.30)])
angles = jnp.array([45.0, -45.0, 0.0])

# the MACRO beam state [eps11, gam12, gam13, kappa1, kappa2, kappa3]
epsilon_bar = jnp.array([0.001, 0.0, 0.0, 0.0, 0.01, 0.0])

r = plate_homo_2d(name + ".yaml", material_param=material_param,
                  angles=angles, n_model=n_model)
np.savetxt(name + "_beam_Timo.out", r["C_eff"], fmt="%16.8e",
           header="beam Timoshenko 6x6 [eps11 gam12 gam13 kappa1 kappa2"
                  " kappa3] (Beam_solid KKT engine) from the %dD SG %s"
                  % (r["n_sg"], name))

Gam, Sig, U = dehom_fields(r, epsilon_bar)
export_gauss(r, Gam, Sig, name + "_dehom", U_eqd=U)
for i, nm in enumerate(("xx", "yy", "zz", "yz", "xz", "xy")):
    print("  max|Sig_%s| = %.5e" % (nm, np.abs(Sig[..., i]).max()))
```

```{warning}
Only **classical-channel** states are valid drivers: extension, twist, bending and their
combinations. The recovery chain is kept verbatim from its validated reference, and its seed
is the compliance-scaled state $C_{\rm eff}^{-1}\bar{\epsilon}$, which makes the two transverse
shear entries `epsilon_bar[1]` and `epsilon_bar[2]` **recovery-inert** — a shear-only driver
recovers essentially zero stress. This is a documented deviation from the refined theory, not
a switch to flip.
```

### Run it

```bash
cd examples/OpenSG-solid/8_get_beam_dehom_from_SG
```

```bash
python beam_dehom_sg.py
```

### What comes out

`dehom_fields` returns three arrays shaped $(E, Q, \cdot)$ over elements and Gauss points:
`Gam` $(E,Q,6)$ strain, `Sig` $(E,Q,6)$ stress, and `U` $(E,Q,3)$ the **fluctuation-only**
displacement (for the beam route, the zeroth- plus first-order warping $w_1 + w_{1s}$; there is
no macro contribution in a unit-state SG recovery). `export_gauss` writes them out:

| file | contents |
|---|---|
| `RHC_SW_2UC_45.out` | the Timoshenko report written by `plate_homo_2d` itself, as in section 1; regenerated on every run |
| `RHC_SW_2UC_45_beam_Timo.out` | the bare $6\times6$ written by the script through `np.savetxt`, one commented header line — convenient to `np.loadtxt` |
| `RHC_SW_2UC_45_dehom.txt` | tab-separated Gauss table: `GP_Index`, `Coord_X`, `Coord_Y`, six `Eps_*`, six `Sig_*`; one header line plus 19 920 rows (6640 triangles $\times$ 3 quadrature points) |
| `RHC_SW_2UC_45_dehom.vtk` | the same cloud as an ASCII unstructured grid of vertex cells, each strain and stress component a `SCALARS` field — open in ParaView |
| `RHC_SW_2UC_45_dehom.SM` / `.EM` / `.U` | the OpenSG dehom files: stress, strain and fluctuation displacement in the `rm_plate_1D` layout, `x y z` columns then components printed in the order 11 22 33 12 13 23, with the SG coordinates in the last `n_sg` of the `x, y, z` slots; regenerated on every run |
| `RHC_SW_2UC_45_mesh.png` | the mesh figure |

The `.txt`/`.vtk` pair uses the SwiftComp component order `xx yy zz yz xz xy`; the
`.SM`/`.EM`/`.U` files reorder to `11 22 33 12 13 23` **on write** only. Units follow the SG
input system throughout — MPa here.

### What the numbers mean

The shipped table is the run of the driver above. Its component maxima are:

| component | $\max\lvert\sigma\rvert$ (MPa) | $\max\lvert\epsilon\rvert$ |
|---|---|---|
| $xx$ | $1.52933\times10^{4}$ | $2.33938\times10^{-1}$ |
| $yy$ | $3.46016\times10^{3}$ | $2.26520\times10^{-1}$ |
| $zz$ | $5.53998\times10^{2}$ | $8.78725\times10^{-2}$ |
| $yz$ | $6.03408\times10^{2}$ | $1.34434\times10^{-1}$ |
| $xz$ | $1.17652\times10^{3}$ | $1.70443\times10^{-1}$ |
| $xy$ | $2.95941\times10^{3}$ | $1.02868\times10^{-1}$ |

The axial field is the one to sanity-check first, because beam kinematics predict it directly:
$\epsilon_{xx} \approx \bar{\epsilon}_{11} + \kappa_2\,y_3$ plus a warping correction. The first
row of the shipped `.txt` sits at $y_3 = 23.29375$ mm and reports
$\epsilon_{xx} = 2.339375\times10^{-1}$, against $0.001 + 0.01\times23.29375 = 0.2339375$ — an
exact match, and the same value is the table maximum, as it must be at the extreme fibre. The
useful content of the recovery is everything *else*: the transverse and shear components,
which exist purely because the cross-section is heterogeneous and warps, and which no beam
theory can supply. Note that the shipped driver is a demonstration magnitude, not a service
load; scale it to your own macro state before reading the stresses as failure indices.

## 3. Equivalent 3-D solid properties from a 2-D SG

### What it computes

`n_model=3` reduces an SG to an equivalent **3-D solid** stiffness — the $6\times6$ material
law of the homogenized continuum, in Voigt order $[11, 22, 33, 23, 13, 12]$ with engineering
shears. On a **2-D** SG this is the generalized-plane-strain problem: the cell is periodic in
its two in-plane directions and uniform out of plane, six unit macro strains are applied, the
fluctuation field $V_0$ is solved for each, and the effective law is

$$\mathbf{D}_{\rm eff} = \frac{\mathbf{D}_{ee} + \mathbf{V}_0^{\mathsf T}\mathbf{D}_{he}}{\omega},$$

with $\omega$ the cell area. This is the classical micromechanics unit-cell problem, which
makes it the natural place to validate the route against published numbers.

### The example SG

`examples/OpenSG-solid/5_get_solid_props_from_SG` homogenizes a unidirectional
fibre/matrix square unit cell: 801 nodes, 1536 three-node triangles, 512 matrix elements and
1024 fibre elements, cell area $\omega = 1$.

![Unidirectional composite unit cell, fibre and matrix elements](../_static/udcomp_2dsg_mesh.png)

| material | $E_1, E_2, E_3$ (MPa) | $G_{12}, G_{13}, G_{23}$ (MPa) | $\nu_{12}, \nu_{13}, \nu_{23}$ | density |
|---|---|---|---|---|
| matrix | 4760, 4760, 4760 | $4760/(2\cdot1.37) = 1737.23$ each | 0.37, 0.37, 0.37 | 1200.0 |
| fibre | 276 000, 19 500, 19 500 | 70 000, 70 000, 5735 | 0.28, 0.28, 0.70 | 1800.0 |

### Step 1 — build the SG yaml from the gmsh mesh

`make_udcomp_yaml.py` is a one-time converter from `UDcomp_2D.msh` (gmsh 2.2 format) to
`UDcomp_2D.yaml`. It writes the nodes as `- [x y z]` rows, the triangles as 1-based
`- [n1 n2 n3]` rows, a constant `elementOrientations` frame ($e_1$ = axial $z$, $e_2$ = $+x$,
$e_3$ = $+y$), the two materials as engineering constants in MPa, and `sets.element` splitting
the mesh into the matrix and fibre phases by gmsh physical tag. The yaml is already checked
in, so this step is only needed if you remesh.

```bash
cd examples/OpenSG-solid/5_get_solid_props_from_SG
```

```bash
python make_udcomp_yaml.py
```

### Step 2 — homogenize

```bash
python solid_homo_2dsg.py
```

The User Input block is three lines: `n_model = 3`, `name = "UDcomp_2D"` (which reads
`<name>.yaml`), and the `paper` dictionary of reference constants to compare against.

```{note}
This runner is deliberately self-contained: it assembles constant-strain triangles, the
periodic master–slave node map (right$\to$left matching $y$, top$\to$bottom matching $x$, with
corner chains resolved) and the three pinned rigid translations in plain NumPy, so you can
read the entire `n_model=3` algebra in one 180-line file. It therefore writes its own
`_Deff.out` through `np.savetxt` rather than the SwiftComp-layout report, and it consumes the
`nodes`/`elements`/`elementOrientations`/`materials`/`sets` yaml dialect that
`make_udcomp_yaml.py` emits — not the `dim`/`nodes`/`cells`/`mat_id`/`materials` dialect that
`opensg_solid.sg_mesh.load_sg_input` parses. The engine path itself is exercised on the
honeycomb SG of section 1: `n_model=3` on `RHC_SW_2UC_45.yaml` is 6640 elements, 12 273 DOFs,
11.4 s cold and 8.1 s warm, and writes the full SwiftComp `.out` with its engineering-constants
block.
```

### What comes out

| file | contents |
|---|---|
| `UDcomp_2D_Deff.out` | the $6\times6$ $\mathbf{D}_{\rm eff}$ in MPa, Voigt order `[e11 e22 e33 2e23 2e13 2e12]`, with $\omega$ in the header |
| `UDcomp_2D_constants_vs_paper.dat` | the nine effective engineering constants next to the published values and the percentage difference |
| `UDcomp_2D_mesh.png` | the unit cell, elements coloured by phase |

### What the numbers mean, and what they were validated against

The nine constants are read off the compliance $\mathbf{S} = \mathbf{D}_{\rm eff}^{-1}$ as
$E_i = 1/S_{ii}$, $G_{ij} = 1/S_{kk}$ and $\nu_{ij} = -S_{ij}/S_{ii}$, and compared against
Table II Model-3 of the ASC-23 OpenSG-Solid paper, the same case as the FEniCS
`3DModel.ipynb` in `wenbinyugroup/OpenSG`. The shipped
`UDcomp_2D_constants_vs_paper.dat` reads:

| constant | this code | paper | % difference |
|---|---|---|---|
| $E_1$ (GPa) | 167.255337 | 167.255300 | $+0.0000$ |
| $E_2$ (GPa) | 11.413657 | 11.413600 | $+0.0005$ |
| $E_3$ (GPa) | 11.413657 | 11.413600 | $+0.0005$ |
| $G_{23}$ (GPa) | 3.095555 | 3.095500 | $+0.0018$ |
| $G_{13}$ (GPa) | 6.801763 | 6.801700 | $+0.0009$ |
| $G_{12}$ (GPa) | 6.801763 | 6.801700 | $+0.0009$ |
| $\nu_{12}$ | 0.311780 | 0.311780 | $-0.0000$ |
| $\nu_{13}$ | 0.311780 | 0.311780 | $-0.0000$ |
| $\nu_{23}$ | 0.575743 | 0.575740 | $+0.0006$ |

Every constant agrees with the published value to within $0.002\,\%$, which is the round-off
of the paper's own printed precision — the route is exact against its reference, not merely
close.

The stiffness itself, from `UDcomp_2D_Deff.out` (MPa, $\omega = 1$):

$$
\mathbf{D}_{\rm eff} =
\begin{bmatrix}
1.72654\times10^{5} & 8.65849\times10^{3} & 8.65849\times10^{3} & 0 & 0 & 0\\
8.65849\times10^{3} & 1.75072\times10^{4} & 1.02639\times10^{4} & 0 & 0 & 0\\
8.65849\times10^{3} & 1.02639\times10^{4} & 1.75072\times10^{4} & 0 & 0 & 0\\
0 & 0 & 0 & 3.09555\times10^{3} & 0 & 0\\
0 & 0 & 0 & 0 & 6.80176\times10^{3} & 0\\
0 & 0 & 0 & 0 & 0 & 6.80176\times10^{3}
\end{bmatrix}
$$

Two health checks are worth making a habit of. First, the symmetry the geometry demands is
recovered: $D_{22} = D_{33}$ and $D_{55} = D_{66}$ to all printed digits, and every
normal–shear coupling term in the shipped file is either exactly zero or below
$3.4\times10^{-2}$ MPa — six to seven orders under the diagonal, which is the numerical
signature of a correctly tied periodic cell. Second, the transverse plane is **not**
isotropic: a transversely isotropic material would satisfy
$D_{44} = (D_{22} - D_{23})/2 = 3621.7$ MPa, whereas the computed $D_{44}$ is 3095.6 MPa, a
$15\,\%$ gap. A square fibre array is tetragonal, so $G_{23}$ genuinely has to be
homogenized rather than inferred from $E_2$ and $\nu_{23}$ — which is precisely why a
unit-cell solve is needed instead of a rule of mixtures.

## 4. Equivalent 3-D solid properties from a 3-D SG

When the SG is periodic in all three directions — a TPMS unit cell, a lattice block, any
genuine 3-D microstructure — the same macro model is reached with
`plate_homo_2d(..., n_model=3, boundary="periodic")` on a tet or hex mesh; `"periodic"` is the
default on every route, and `boundary="aperiodic"` (zero fluctuation on the bounding-box-face
nodes) is the explicit request for SGs that are genuinely not periodic. The runnable cases
live in `examples/OpenSG-solid/6_get_solid_props_from_3D_SG`, where `tpms_solid_props.py`
homogenizes two TPMS samples in aluminium and `compare_boundary_modes.py`,
`solid_vs_shell_compare.py` and `tpms_all_compare.py` produce the comparison tables. That
route, its SwiftComp digit-for-digit benchmark and the head-to-head against the msg-shell
solver are covered in full by the page *Equivalent 3-D Solid Properties from a 3-D SG (TPMS)*.
