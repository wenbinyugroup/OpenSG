# Equivalent 3-D Solid Properties from a 3-D SG (TPMS)

A structure gene periodic in **all three** directions — a triply-periodic minimal surface (TPMS)
unit cell, a lattice block, any 3-D microstructure — homogenizes to the same equivalent 3-D
solid $6\times6$. OpenSG covers this from both sides:

| route | input | solver |
|---|---|---|
| **msg-solid** | 3-D solid SG (tets/hexes) | `opensg_solid.sg_homo.plate_homo_2d(..., n_model=3)` |
| **msg-shell** | 3-D **shell** SG (a thin sheet meshed as shell elements) | `opensg_shell.shell_sg3d.shell_sg3d(...)` |

The solid route is periodic: opposite **faces, edges and corners** tied through the sparse
periodic assembly map. The shell route takes a `boundary` argument — `"aperiodic"` (default:
the boundary solution mapped onto the bounding-box nodes as Dirichlet data) or `"periodic"`
(the same three-direction tie) — compared head-to-head below.

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

**Sample 1** — 116 851 nodes, 545 741 tets, relative density 0.300, solve 164.6 s:

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

**Sample 2** — 54 874 nodes, 191 957 tets, relative density 0.0857, solve 61.2 s:

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

Digit-for-digit on both samples, to seven significant figures on every entry. The effective law
is cubic to five digits and every symmetry-forbidden coupling sits four to five orders below the
diagonal, which is the health check on the all-directions periodic tie.

### Boundary treatment on the solid route

`plate_homo_2d` takes the same `boundary` argument; `None` resolves to `"aperiodic"` for the
3-D SG solid model and `"periodic"` for every other route (the sample runners above pass
`boundary="periodic"` explicitly to keep the digit-parity gate). Solid elements carry only
translations, so aperiodic clamps every DOF of the bounding-box-face nodes. Same `.sc`, per
unit cell, MPa:

| term | aperiodic (S1) | periodic (S1) | % | aperiodic (S2) | periodic (S2) | % |
|---|---|---|---|---|---|---|
| $C_{11}$ | 12977.0 | 10190.2 | +27.3 | 2810.3 | 2303.2 | +22.0 |
| $C_{12}$ | 5423.6 | 5646.1 | −3.9 | 1548.6 | 1612.1 | −3.9 |
| $C_{44}$ | 5546.0 | 4519.6 | +22.7 | 1300.4 | 1106.7 | +17.5 |
| solve | 146 s | 150 s | | 52 s | 52 s | |

The bias is much larger than the shell route's (+7 %) because the clamped set is much larger:
the solid sheet meets the cube faces in 2-D patches — 17 610 of 116 851 nodes (15 %) for
Sample 1 against 720 of 13 536 (5 %) for the shell traces — and on the solid route aperiodic
is **not** faster (the periodic tie merges only face pairs, so the factorized size barely
changes). Periodic is the exact treatment for a periodic medium; aperiodic is the right model
only when the SG genuinely is not periodic. Full table:
`examples/OpenSG-solid/6_get_solid_props_from_3D_SG/boundary_compare.dat`.

## Shell route — the 3-D shell SG

`shell_sg3d` solves the same equivalent-continuum problem when the microstructure is a **thin
sheet**: the surface is meshed with shell elements and the wall carries the RM $8\times8$ law,
so thickness is a parameter rather than a remesh. It reuses the cross-section $\Gamma_e$ /
$\Gamma_h$ operators unchanged — they are geometry-general — and adds only the 3-D environment:
sparse assembly, the three-direction periodic map, drilling by penalty on the element-constant
residual, and a three-translation kernel. Shell edges shared by more than two elements are
detected as **junction lines**; a smooth TPMS has none.

```python
from opensg_shell.shell_sg3d import shell_sg3d

r = shell_sg3d("schwarz_p_3Dshell.yaml")                        # boundary="aperiodic" (default)
r = shell_sg3d("schwarz_p_3Dshell.yaml", boundary="periodic")   # unit-cell benchmark mode
r["C3D"], r["n_junction_edges"], r["solve_time"]
```

The SG measure $\omega$ defaults to the **midsurface area** — the 3-D analogue of
$\omega = \text{perimeter}$ for a plane-section shell SG — and can be overridden. The `.out`
file is written per unit-cell volume so its moduli compare directly with a solid `.K`.

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
| solve | 72 s | 96 s | | | 71 s | 95 s | |

The offset is the clamped-face boundary layer of a **single** cell — the expected kinematic
bound, not an error in either path (clamping the rotations too would double it). Aperiodic is
~25 % *faster*: the Dirichlet rows leave the factorization and there is no tie map or border.
Use `boundary="periodic"` for anything benchmarked against SwiftComp `.K` or reported as the
effective law of a periodic medium; the aperiodic default is for SGs that genuinely are not
periodic. Full table: `examples/OpenSG_shell/4_get_solid_props_from_shell_3D_SG/boundary_compare.dat`.

### Shell versus solid on the same surface

Sample 2 is the Schwarz-P **sheet**: its midsurface area recovered from the tet mesh (2.3448)
matches the shell mesh (2.3533), and its thickness follows from its own mesh data as
$t = 2V/S_{\rm free} = 0.036547$. Running the shell model at that thickness gives relative
density 0.0860 against the solid's 0.0857 — a 0.35 % census match — and makes the two routes a
genuine head-to-head on one structure. Per unit cell, MPa:

| term | msg-shell (26 360 shell elems) | msg-solid (191 957 tets) | % |
|---|---|---|---|
| $C_{11}$ | 2231.1 | 2303.2 | −3.13 |
| $C_{22}$ | 2231.1 | 2303.1 | −3.12 |
| $C_{33}$ | 2230.9 | 2303.0 | −3.13 |
| $C_{12}$ | 1568.8 | 1612.1 | −2.69 |
| $C_{13}$ | 1568.9 | 1612.1 | −2.68 |
| $C_{23}$ | 1568.7 | 1612.1 | −2.69 |
| $C_{44}$ | 1043.6 | 1106.7 | −5.70 |
| $C_{55}$ | 1043.7 | 1106.7 | −5.69 |
| $C_{66}$ | 1043.6 | 1106.7 | −5.70 |

| constant | msg-shell | msg-solid | % |
|---|---|---|---|
| $E_1$ | 935.65 | 975.51 | −4.09 |
| $E_2$ | 935.83 | 975.46 | −4.06 |
| $E_3$ | 935.53 | 975.40 | −4.09 |
| $G_{12}$ | 1043.58 | 1106.66 | −5.70 |
| $G_{13}$ | 1043.66 | 1106.67 | −5.69 |
| $G_{23}$ | 1043.64 | 1106.73 | −5.70 |
| $\nu_{12}$ | 0.41275 | 0.41174 | +0.24 |
| $\nu_{13}$ | 0.41302 | 0.41179 | +0.30 |
| $\nu_{23}$ | 0.41286 | 0.41179 | +0.26 |

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

| model | file |
|---|---|
| msg-solid beam / plate / 3-D | `<base>.out` |
| msg-shell beam (Timoshenko) | `<yaml>_Timo.out` |
| msg-shell solid props (cross-section SG) | `<yaml>_C3D.out` |
| msg-shell 3-D shell SG | `<yaml>_C3D.out` |
