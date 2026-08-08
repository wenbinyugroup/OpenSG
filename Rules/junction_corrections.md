# Rule — when junction corrections apply (and when they must stay off)

**Scope:** `junction=` in `build_solid_bundle` (`"census"`, `"micro"`, `"microcell"`) and any
future junction model.

## What a junction correction is for

A midline shell SG has no material at wall crossings: the $t_A \times t_B$ overlap block has
**zero measure**. Any effective property carried by that block is invisible to the wall
integrals. The corrections add it back as a 6x6 energy block on `Dee` — **no new elements, no
new DOFs, no change to $\Gamma_e$/$\Gamma_h$ or the fluctuation solve**. Default is OFF; a
flag-off run must stay bit-identical.

## Applies to — stretch-dominated cells

Valid where periodicity makes the wall-average strain equal the macro strain, i.e. straight
walls running **through** the cell (rings, I-beams, cross cells). There:

| effect | scaling | typical size |
|---|---|---|
| the missing $C_{23}$ on a 0/90 lattice | $O(t^2)$ | $C_{23} \simeq C^{3D}_{23}\!\cdot A_j$ |
| the $D_{33}$/$D_{13}$ deficit at thick walls | $O(t^2)$ | up to −28 % at $t/h = 0.2$ |

Validated: iso ring all normal terms 1.000–1.027; iso I-beam ≤4 % on every term for
$t/h = 0.03$–$0.2$; 8-ply symmetric and unbalanced laminates 1.000–1.025.

## Must stay OFF — bending-dominated lattices

If transverse load travels by **wall bending** rather than stretching, the walls shed the macro
strain by bowing and the patch environment assumption is void. The frozen-corner energy
($\sim E t^2$) then dwarfs the true $t^3$-scale stiffness.

Diagnostic: compare the shell's own transverse stiffness with the stretch census
$E' t L$. If it is orders below, the cell is bending-dominated — **do not apply a junction
correction.** Deo's hierarchical square ($C_{22} \approx 50$ MPa against $C_{11} \approx 5100$
MPa) blew up by +280…+525 % on $C_{12}$/$C_{22}$/$C_{23}$ when the correction was forced.

The rigid-block (rotational-spring) variant was implemented and **removed**: a rigid corner is
only an upper bound, and it over-stiffened the same lattice by +279 % on $C_{22}$ and +866 % on
$C_{44}$. A finite joint spring is the open path.

## Geometry facts that must be respected

- **Inclined walls need no junction term for $C_{23}$.** The membrane path
  $(A_{11}-G_{\rm msg})\sin^2\varphi\cos^2\varphi$ is $O(t)$ and the formulation carries it
  exactly (±45 cell: $E' t L/\sqrt 2$ to 0.02 %). Only $0/90$ lattices, where that factor
  vanishes identically, expose the $O(t^2)$ junction term. Deo's cellular honeycombs get
  $C_{23}$ right with no junction treatment for this reason.
- **Detect junctions on the periodically reduced mesh.** A ring's four L-corners merge into one
  X-crossing of stacked walls; the mirrored partner's frame is the 180° image, so **its ply
  angles negate** (a $-45$ ring merges to the antisymmetric $[+45/-45]$ laminate). Getting this
  registration wrong is invisible for isotropic walls and gutted the m45 $C_{23}$ by 4x.
- Ring cells are a poor $C_{44}$ benchmark: the periodic tie merges coincident walls into one
  $2t$ wall in the solid (bending $8t^3$) while the shell keeps two independent $t$ walls
  ($2t^3$) — an exact factor 4 by construction. Benchmark $C_{44}$ on a cross cell or against
  the frame closed form.
