# Equivalent 3-D Solid Properties from a 3-D SG (TPMS)

A structure gene periodic in **all three** directions — a triply-periodic minimal surface (TPMS)
unit cell, a lattice block, any 3-D microstructure — homogenizes to the same equivalent 3-D
solid $6\times6$. OpenSG covers this from both sides:

| route | input | solver |
|---|---|---|
| **msg-solid** | 3-D solid SG (tets/hexes) | `opensg_solid.sg_homo.plate_homo_2d(..., n_model=3)` |
| **msg-shell** | 3-D **shell** SG (a thin sheet meshed as shell elements) | `opensg_shell.sg_homo.shell_sg3d(...)` |

Both routes default to **periodic**: opposite faces, edges and corners tied through the sparse
periodic assembly map. Both also take a `boundary` argument whose `"aperiodic"` option (the
boundary solution mapped onto the bounding-box nodes as Dirichlet data) is compared
head-to-head below — a single-cell kinematic upper bound, for SGs that genuinely are not
periodic.

```{note}
Runnable cases in this repository:
`examples/OpenSG-solid/6_get_solid_props_from_3D_SG/{sample_1,sample_2}/` for the solid route and
`examples/OpenSG_shell/4_get_solid_props_from_shell_3D_SG/` for the shell route. Each writes a
timed `.out` in SwiftComp format plus the mesh PNG.
```

## Solid route — matched to SwiftComp

Two TPMS samples supplied as SwiftComp `.sc` meshes (linear tets, unit cell), aluminium
$E=69$ GPa, $\nu=0.3$, with the vendor's own `.K` results as the benchmark. Values are per unit
cell in MPa; `%` is against the `.K`.

```{important}
**The SG measure $\omega$ of a 3-D solid SG is the node bounding-box volume** — the volume the
equivalent continuum occupies and the cell the periodic assembly map ties — **not** the summed
element (material) volume. Both samples are unit cells, so $\omega = 1$ and the `.out` banner
reads ` OpenSG msg-solid 3D elastic model, omega 1, periodic`.

The consequence is that the **`.out` file itself is now directly comparable with a SwiftComp
`.K`**, entry for entry, with only the unit change (OpenSG writes Pa, SwiftComp MPa). Under the
older material-volume convention the `.out` came out $1/\text{relative density}$ too stiff and
only the comparison scripts' rescale by $\omega$ brought the tables into agreement. Read
`sample_1/Sample_1.out`: its $C_{11} = 1.0190158\times10^{10}$ Pa is 10190.1580 MPa against the
`.K`'s 10190.1580. `sample_2/Sample_2.out` gives $2.3031598\times10^{9}$ Pa = 2303.1598 against
2303.1598. Nothing has to be rescaled to make that comparison.
```

```bash
opensg Sample_1.yaml
```

The sample runners `sample_1/run_sample_1.py` and `sample_2/run_sample_2.py` do the same
homogenization through the API and additionally parse the vendor `.K` and write the
`Sample_<n>_compare.dat` tables reproduced below.

**Sample 1** — 116 851 nodes, 545 741 tets, relative density 0.300, solve 69.2 s:

| term | OpenSG msg-solid | SwiftComp `.K` | % |
|---|---|---|---|
| $C_{11}$ | 10190.158 | 10190.158 | ±0.000 |
| $C_{12}$ | 5646.069 | 5646.068 | +0.000 |
| $C_{13}$ | 5646.092 | 5646.091 | +0.000 |
| $C_{22}$ | 10190.131 | 10190.131 | +0.000 |
| $C_{23}$ | 5646.012 | 5646.012 | +0.000 |
| $C_{33}$ | 10189.958 | 10189.958 | +0.000 |
| $C_{44}$ | 4519.640 | 4519.640 | −0.000 |
| $C_{55}$ | 4519.637 | 4519.637 | −0.000 |
| $C_{66}$ | 4519.755 | 4519.755 | −0.000 |

**Sample 2** — 54 874 nodes, 191 957 tets, relative density 0.0857, solve 23.6 s:

| term | OpenSG msg-solid | SwiftComp `.K` | % |
|---|---|---|---|
| $C_{11}$ | 2303.160 | 2303.160 | ±0.000 |
| $C_{12}$ | 1612.130 | 1612.130 | ±0.000 |
| $C_{13}$ | 1612.141 | 1612.141 | +0.000 |
| $C_{22}$ | 2303.067 | 2303.067 | +0.000 |
| $C_{23}$ | 1612.120 | 1612.120 | +0.000 |
| $C_{33}$ | 2303.019 | 2303.018 | +0.000 |
| $C_{44}$ | 1106.734 | 1106.735 | −0.000 |
| $C_{55}$ | 1106.668 | 1106.668 | +0.000 |
| $C_{66}$ | 1106.664 | 1106.664 | −0.000 |

Digit-for-digit on both samples, to seven significant figures on every entry — and, since the
$\omega$ change, digit-for-digit *in the `.out` itself*, not only in this table. The effective
law is cubic to five digits and every symmetry-forbidden coupling sits four to five orders
below the diagonal, which is the health check on the all-directions periodic tie.

### Boundary treatment on the solid route

`plate_homo_2d` takes the same `boundary` argument, defaulting to `"periodic"` on every route
(the sample runners above pass it explicitly to document the digit-parity gate). Solid
elements carry only
translations, so aperiodic clamps every DOF of the bounding-box-face nodes. Same `.sc`, per
unit cell, MPa:

| term | aperiodic (S1) | periodic (S1) | % | aperiodic (S2) | periodic (S2) | % |
|---|---|---|---|---|---|---|
| $C_{11}$ | 12977.0 | 10190.2 | +27.3 | 2810.3 | 2303.2 | +22.0 |
| $C_{12}$ | 5423.6 | 5646.1 | −3.9 | 1548.6 | 1612.1 | −3.9 |
| $C_{44}$ | 5546.0 | 4519.6 | +22.7 | 1300.4 | 1106.7 | +17.5 |
| solve | 30 s | 29 s | | 10 s | 8 s | |

The bias is much larger than the shell route's (+7 %) because the clamped set is much larger:
the solid sheet meets the cube faces in 2-D patches — 17 610 of 116 851 nodes (15 %) for
Sample 1 against 720 of 13 536 (5 %) for the shell traces — and on the solid route aperiodic
is **not** faster (the periodic tie merges only face pairs, so the factorized size barely
changes). Periodic is the exact treatment for a periodic medium — hence the default; aperiodic
is the right model only when the SG genuinely is not periodic. Full table:
`examples/OpenSG-solid/6_get_solid_props_from_3D_SG/boundary_compare.dat`.

## Shell route — the 3-D shell SG

`shell_sg3d` solves the same equivalent-continuum problem when the microstructure is a **thin
sheet**: the surface is meshed with shell elements and the wall carries the RM $8\times8$ law,
so thickness is a parameter rather than a remesh. It reuses the cross-section $\Gamma_e$ /
$\Gamma_h$ operators unchanged — they are geometry-general — and adds only the 3-D environment:
sparse assembly, the three-direction periodic map, drilling by penalty on the element-constant
residual, and a three-translation kernel. Shell edges shared by more than two elements are
detected as **junction lines**; a smooth TPMS has none.

```bash
opensg schwarz_p_3Dshell.yaml
```

```python
from opensg_shell.sg_homo import shell_sg3d

r = shell_sg3d("schwarz_p_3Dshell.yaml")                        # boundary="periodic" (default)
r = shell_sg3d("schwarz_p_3Dshell.yaml", boundary="aperiodic")  # non-periodic SGs (explicit)
r["C3D"], r["n_junction_edges"], r["solve_time"]
```

The SG measure $\omega$ of a 3-D SG is the **node bounding-box volume** — the volume the
equivalent continuum occupies and the cell the three-direction periodic map ties — and an
`omega:` header key overrides it. The `.out` is written per unit-cell volume, so its moduli
compare directly with a solid `.K`, exactly as the msg-solid route's now do.

The run writes two files. `schwarz_p_3Dshell_C3D.out` is the equivalent continuum law
(` The Effective Cauchy Continuum Stiffness Matrix`, its compliance and the
orthotropic-approximated engineering constants), closing with ` Time taken: 10.85 sec` on this
cell. `schwarz_p_3Dshell_ABDG.out` is the **step-1 wall plate law** — the $8\times8$
Reissner–Mindlin ABDG the surface actually carries, one block per section:

```text
 OpenSG msg-shell wall plate laws, one block per section
 rows [eps11 eps22 2eps12 | K11 K22 K12+K21 | 2g13 2g23]

 section 0: layup_0  layup [['alu', 0.036457, 0.0]]
 The Effective Reissner-Mindlin Plate Stiffness Matrix
```

For this single-ply aluminium wall $A_{11} = 2.7643220\times10^{9}$,
$A_{12} = 8.2929659\times10^{8}$, $A_{66} = 9.6751264\times10^{8}$ N/m,
$D_{11} = 3.0617465\times10^{5}$ N·m and both transverse-shear diagonals
$8.5193106\times10^{8}$ N/m. Every route that reduces a layup to a wall plate law now emits
this record, so the halfway point of the two-step reduction is always inspectable.

### Boundary treatment: aperiodic vs periodic, same yaml

`boundary="aperiodic"` is the boundary-solution treatment: for a unit cell the boundary
solution is the macro (affine) field itself, so mapping it onto the boundary nodes prescribes
zero **translational** warping fluctuation ($w_1=w_2=w_3=0$, Dirichlet) on every
bounding-box-face node, rotations left natural. That forbids the affine fluctuations that make
a *free* aperiodic SG rank-one, fixes the rigid modes (no Lagrange border), and is a kinematic
upper bound. On the Schwarz-P yaml (720 boundary nodes of 13 536), per unit cell, MPa:

| term | aperiodic | periodic | % | | aperiodic | periodic | % |
|---|---|---|---|---|---|---|---|
| | **t = 0.0365** | | | | **t = 0.1293** | | |
| $C_{11}$ | 2375.9 | 2225.3 | +6.8 | | 9881.0 | 8916.8 | +10.8 |
| $C_{12}$ | 1537.5 | 1564.9 | −1.8 | | 4921.1 | 5352.2 | −8.1 |
| $C_{44}$ | 1078.3 | 1041.0 | +3.6 | | 4150.3 | 3961.6 | +4.8 |
| ndof | 81 216 | 79 056 | | | 81 216 | 79 056 | |
| solve | 42.3 s | 66.5 s | | | 42.2 s | 65.4 s | |

The offset is the clamped-face boundary layer of a **single** cell — the expected kinematic
bound, not an error in either path (clamping the rotations too would double it). Aperiodic is
also ~35 % *faster* on both thicknesses: the Dirichlet rows leave the factorization and there
is no tie map or border, which more than pays for the 2 160 extra DOF that the untied faces
carry.
Periodic (the default) is what to report for a periodic medium; request `"aperiodic"` only for
SGs that genuinely are not periodic. Full table:
`examples/OpenSG_shell/4_get_solid_props_from_shell_3D_SG/boundary_compare.dat`.

### Shell versus solid on the same surface

Sample 2 is the Schwarz-P **sheet**: its midsurface area recovered from the tet mesh (2.3448)
matches the shell mesh (2.3533), and its thickness follows from its own mesh data as
$t = 2V/S_{\rm free} = 0.036547$. Running the shell model at that thickness gives relative
density 0.0860 against the solid's 0.0857 — a 0.35 % census match — and makes the two routes a
genuine head-to-head on one structure. Per unit cell, MPa:

| term | msg-shell (26 360 shell elems) | msg-solid (191 957 tets) | % |
|---|---|---|---|
| $C_{11}$ | 2225.3 | 2303.2 | −3.38 |
| $C_{22}$ | 2225.3 | 2303.1 | −3.38 |
| $C_{33}$ | 2225.0 | 2303.0 | −3.39 |
| $C_{12}$ | 1564.9 | 1612.1 | −2.93 |
| $C_{13}$ | 1565.0 | 1612.1 | −2.93 |
| $C_{23}$ | 1564.8 | 1612.1 | −2.94 |
| $C_{44}$ | 1041.0 | 1106.7 | −5.94 |
| $C_{55}$ | 1041.0 | 1106.7 | −5.93 |
| $C_{66}$ | 1040.9 | 1106.7 | −5.94 |

| constant | msg-shell | msg-solid | % |
|---|---|---|---|
| $E_1$ | 932.96 | 975.51 | −4.36 |
| $E_2$ | 933.14 | 975.46 | −4.34 |
| $E_3$ | 932.85 | 975.40 | −4.36 |
| $G_{12}$ | 1040.93 | 1106.66 | −5.94 |
| $G_{13}$ | 1041.01 | 1106.67 | −5.93 |
| $G_{23}$ | 1040.98 | 1106.73 | −5.94 |
| $\nu_{12}$ | 0.41278 | 0.41174 | +0.25 |
| $\nu_{13}$ | 0.41305 | 0.41179 | +0.31 |
| $\nu_{23}$ | 0.41288 | 0.41179 | +0.26 |

```{note}
The msg-shell column of the last three tables is the shipped
`solid_vs_shell_compare.dat` / `boundary_compare.dat`, produced before the **native tri3 /
MITC3** element landed — until then a triangle entered the assembly as a quad with its last
node repeated. The current `schwarz_p_3Dshell_C3D.out`, whose banner now reads
`26360 elems [26360 tri3/MITC3]`, gives $C_{11} = 2.2205758\times10^{9}$ Pa (2220.58 MPa),
$G_{12} = 1.0395467\times10^{9}$ Pa, $E_1 = 9.2745286\times10^{8}$ Pa and a solve time of
10.85 s — about 0.2 % below the shell column quoted here, and the most recent block of
`boundary_compare.dat` records the same shift (periodic $C_{11}$ 2220.58 against the 2225.31
in the table above). The comparison tables have not yet been regenerated on the new element;
read the 0.2 % as the element change and everything larger as the thin-shell reduction.
```

A uniform −3 % on the normal terms, −6 % on the shears and +0.3 % on Poisson, with cubic
symmetry intact in both. Because the solid column is digit-identical to SwiftComp, these
percentages measure the **thin-shell reduction itself**: this surface reaches $t/R \approx 0.2$
near the saddle necks, where a midsurface model slightly under-stiffens transverse shear and
under-counts material at the neck curvature. The shell reaches that accuracy with 26 k elements
and 79 k DOF against 192 k tets and 165 k DOF.

## Output format

Every OpenSG homogenization writes a SwiftComp-layout `.out` by default — effective stiffness,
effective compliance, and (for 3-D laws) the orthotropic-approximated engineering constants —
opening with an ` OpenSG <model>` banner and closing with ` Time taken: … sec`:

| model | file | matrix title |
|---|---|---|
| msg-solid beam, classical | `<base>.out` | `Euler-Bernoulli Beam` |
| msg-solid beam, shear-refined | `<base>.out` | `Timoshenko Beam` |
| msg-solid plate, classical / shear-refined | `<base>.out` | `Classical Plate` / `Reissner-Mindlin` |
| msg-solid 3-D | `<base>.out` | `Cauchy Continuum` |
| msg-shell beam (Timoshenko) | `<yaml>_Timo.out` | `Timoshenko Beam` |
| msg-shell beam (Kirchhoff–Love wall) | `<yaml>_EB.out` | `Euler-Bernoulli Beam` |
| msg-shell solid props (cross-section SG) | `<yaml>_C3D.out` | `Cauchy Continuum` |
| msg-shell 3-D shell SG | `<yaml>_C3D.out` | `Cauchy Continuum` |
| the step-1 **wall** law of any shell route | `<yaml>_ABDG.out` | `Reissner-Mindlin Plate`, one block per section |
