# User guide

The operational reference for the RM-OpenSG plate chain
(`opensg_jax.fe_jax.msg_rm_plate` + `segment_plate`).  Everything here is the
Sphinx mirror of the typeset PDF manual (`docs/rm_plate_manual/`).

## Conventions

- $x_1, x_2$ in-plane, $x_3$ through the thickness; **ply 1 is the bottom
  ply**, stacking in $+x_3$.
- The recovery coordinate $x_3$ is measured from the **reference surface**,
  chosen by `fraction`: `0` = bottom (OML) face, `0.5` = mid-surface, `1` =
  top (IML) face.  The $8\times8$ (its $B$-block in particular) **depends on
  this choice** — homogenize, analyze and dehomogenize with the same
  `fraction`.
- **3-D Voigt order** of every stress/strain 6-vector:
  $[11,\,22,\,33,\,23,\,13,\,12]$.
- **Plate-measure order** (rows/cols of the $8\times8$):
  $[\epsilon_{11}, \epsilon_{22}, \gamma_{12}, \kappa_{11}, \kappa_{22},
  \kappa_{12}, 2\gamma_{13}, 2\gamma_{23}]$, conjugate resultants
  $[N_{11}, N_{22}, N_{12}, M_{11}, M_{22}, M_{12}, Q_1, Q_2]$.
- **Fiber angle** $\theta$ (deg) rotates the material 1-axis about $+x_3$
  from $x_1$; ply stiffness $C(\theta) = R_\sigma C_\mathrm{mat}
  R_\sigma^\top$ with `msg_materials.rotation_6x6`.  Material-frame stress:
  `rotation_6x6(-theta) @ Sig`.
- Units: any consistent set (the code is unit-agnostic).

## Material input

```python
MATERIAL_DB = {
  "uni": {"E":  [E1, E2, E3],       # 1 = fiber, 2 = in-plane transverse, 3 = thickness
          "G":  [G12, G13, G23],    # ORDER MATTERS
          "nu": [nu12, nu13, nu23],
          "rho": 1600.0},
}
```

## Homogenization

```python
r = rm_plate_msg(thick, angles_deg, mat_names, material_db,
                 n_per_layer=1, elem_order=4, fraction=0.0)
```

`r["ABDG"]` is the $8\times8$ (or `None` if the transverse-shear energy fit
is not SPD — inspect the layup); `r["A6"]`, `r["G_msg"]` its blocks.
The same dictionary `r` drives every recovery call — homogenize once,
recover as often as needed.

### The SG YAML

```python
from opensg_jax.fe_jax.segment_plate import plate_sg_yaml, read_plate_sg_yaml
plate_sg_yaml("my_sg.yaml", layup, MATERIAL_DB, fraction=0.5)
inp = read_plate_sg_yaml("my_sg.yaml")   # -> thick/angles/mat_names/material_db/...
```

```yaml
sg: {type: plate_1d, elem_order: 4, n_per_layer: 1,
     reference_fraction: 0.5, thickness: 0.01, n_ply: 4}
nodes:            # one x3 per node, reference-surface origin, ascending
- [-0.005]
elements:         # 1-based node ids, (elem_order+1) per element
- [1, 2, 3, 4, 5]
materials:
- name: gr
  density: 1600.0
  elastic: {E: [...], G: [...], nu: [...]}   # same order as MATERIAL_DB
sets:
  element:
  - {name: ply_1, labels: [1]}
sections:         # one per ply, BOTTOM FIRST
- {elementSet: ply_1, material: gr, angle: 45.0, thickness: 0.002}
```

The reader cross-checks header vs. mesh and refuses inconsistent files (see
the error table below).

```{image} ../_img/rm_plate/sg_mesh_example.png
:width: 60%
:align: center
```

## Dehomogenization — the load cases

```python
Gam6, Sig6, ply_angle = msgrm_strain_at_depth(r, z, E6,
        dE1=None, dE2=None, dE11=None, dE12=None, dE22=None,
        qt6=None, qb6=None)
```

| # | load case | what you pass | what it recovers |
|---|---|---|---|
| 1 | uniform resultants | `E6 = S6 @ FF[:6]` (with `S6 = inv(A6)`) | classical in-plane stress |
| 2 | transverse shear $Q_1, Q_2$ | `dE1 = S6@[0,0,0,Q1,0,0]`, `dE2 = S6@[0,0,0,0,Q2,0]` | interface-continuous $\sigma_{13}, \sigma_{23}$ |
| 3 | second-order (bending peak) | `dE11` etc. — e.g. harmonic: $-p^2 E_s$ | in-plane beyond CLT |
| 4 | applied face pressure | `qt6/qb6 = [q, q,1, q,2, q,11, q,12, q,22]` per face | $\sigma_{33}$ faces machine-exact; $\sigma_{22}$ load content |
| 5 | $\sigma_{33}$ by equilibrium | integrate $-\sigma_{13,1}-\sigma_{23,2}$ from a face | the most accurate $\sigma_{33}$ route |

**The pure-resultant call (cases 1–2) is the general two-step
dehomogenization** — the blade/airfoil workflow, identical in spirit to the
standard VABS recovery.  Face pressures (case 4) are *inputs* the analyst
knows (like VABS distributed loads), not solution outputs; webs and interior
walls carry none, and blade aero pressure is a $10^{-4}$-level correction —
leave `None` unless pressure is a first-order load.

### Displacement recovery

$U_\alpha = u_\alpha^{2d} - x_3\, w^{2d}_{,\alpha} + w_\alpha(x_3)$, 
$U_3 = w^{2d} + w_3(x_3)$, with $w_i$ from `msgrm_warping_at_depth` (same
arguments as the stress call at that station).  **Compose with the Kirchhoff
tilt $-x_3 w_{,\alpha}$, never with $x_3\varphi_\alpha$** — the warping
already carries the mean shear tilt.

## Accuracy guidance

| component (chain) | $S{=}4$ | $S{=}10$ | thin |
|---|---|---|---|
| $\sigma_{13}$ (cross-ply) | 21 % | 4.4 % | 0.18 % at $S{=}50$ |
| $\sigma_{33}$ equilibrium | 5.5 % | 1.2 % | 0.05 % |
| $\sigma_{11}$ second-order | 27 % | 2.0 % | 0.04 % at $S{=}64$ |
| $\sigma_{12}$ angle-ply | 18 % | — | 0.07 % at $S{=}64$ (order 2.0) |
| $U_1$ | 30 % | 1.4 % | 0.02 % at $S{=}64$ |

All errors fall at $\mathcal{O}(S^{-2})$.  Soft-core sandwiches need larger
$S$ for the *in-plane* second-order recovery; their $\sigma_{13}$,
$\sigma_{33}$ and displacements stay accurate at $S=10$.

## Input errors

| message | cause / fix |
|---|---|
| `layup lists disagree: N materials, M thicknesses, K angles` | one entry per ply in all three lists |
| `ply thicknesses must be positive, got [...]` | zero/negative `thick` entry |
| `materials not in the database: NAME` | `mat_names` key missing from `MATERIAL_DB` |
| `FILE is a 'TYPE' SG, not a through-thickness plate_1d SG` | a shell/solid SG passed to the plate reader |
| `ply 'ply_k' declares thickness T but its elements span S` | hand-edited YAML: header vs. mesh mismatch |
| `plies carry different element counts [...]` | regenerate with `plate_sg_yaml` |
| `reference_fraction F puts the bottom face at A, but the first node is at B` | header/mesh mismatch on the reference surface |
| `r["ABDG"] is None` (no exception) | non-SPD shear fit — inspect the layup (degenerate placeholder materials are the usual cause) |
