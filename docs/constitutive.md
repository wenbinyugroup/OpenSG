# Constitutive Relations Recovered by OpenSG

Every OpenSG run answers the same question: *given a structure gene (SG) — a 1-D
through-thickness stack, a 2-D cross-section or cell, or a 3-D unit cell — what is the
constitutive law of the macroscopic model that replaces it?* The SG problem is solved once,
and its output is a small dense stiffness matrix relating **generalized forces** to
**generalized strains**. This page lists every such law the code produces, the exact ordering
of the strain and force components in each, the function that computes it, and the file it is
written to.

Two conventions hold throughout and are worth internalising before reading any matrix:

- **Units follow the input.** `write_sc_K` performs no unit conversion, so an SG given in
  metres and pascals returns N, N·m and Pa; an SG given in millimetres and MPa returns the
  corresponding mixed units.
- **Shears are engineering shears.** Off-diagonal strain components carry the factor of two
  ($2\varepsilon_{12}$, $2\bar\Gamma_{23}$, …), which is why the shear stiffnesses appear once,
  not four times, in the quadratic energy.

## The `.out` contract

Every homogenization writes a timed text file in the SwiftComp `.K` layout through the single
writer `opensg_solid.sg_homo.write_sc_K(path, C, solve_time, model, constants, name)`. The
file opens with an ` OpenSG <model>` banner, prints the effective stiffness, then its exact
inverse (the effective compliance), and closes with ` Time taken: <t> sec`. The `name`
argument is the only thing that changes the matrix title, and it identifies which macro law
you are looking at. **`name` is always the macro law's console title**, so the `.out` header
and the line the terminal prints are word-for-word the same:

| `name` passed to `write_sc_K` | title printed in the `.out` | macro law |
|---|---|---|
| `"Euler-Bernoulli Beam"` | `The Effective Euler-Bernoulli Beam Stiffness Matrix` | beam $4\times4$ (classical) |
| `"Timoshenko Beam"` | `The Effective Timoshenko Beam Stiffness Matrix` | beam $6\times6$ (shear-refined) |
| `"Classical Plate"` | `The Effective Classical Plate Stiffness Matrix` | plate ABD $6\times6$ |
| `"Reissner-Mindlin"` | `The Effective Reissner-Mindlin Stiffness Matrix` | plate ABDG $8\times8$ |
| `"Cauchy Continuum"` | `The Effective Cauchy Continuum Stiffness Matrix` | 3-D solid $6\times6$ |

Only the last of these also prints ` The Engineering Constants (Approximated as Orthotropic)`
— the `constants=True` branch, which is meaningful only for a 3-D elasticity law.

(`name=""` would give the anonymous ` The Effective Stiffness Matrix`; no macro law uses it.
The per-section *wall* laws inside a shell beam run are written by a different writer,
`opensg_shell.sg_homo.write_abdg_out`, which titles each block
`The Effective Reissner-Mindlin Plate Stiffness Matrix` — a wall plate law, not the macro law
of the run.)

## 1. Beam — the Timoshenko $6\times6$

The beam macro model carries six generalized strains: one axial extension, two transverse
shears, one twist and two bending curvatures. In code order these are

$$
\bar{\boldsymbol\epsilon} \;=\;
\left\{\;\bar\epsilon_{11}\;\;\gamma_{12}\;\;\gamma_{13}\;\;
\kappa_{1}\;\;\kappa_{2}\;\;\kappa_{3}\;\right\}^{\mathsf T},
$$

and the conjugate generalized forces are the axial force, the two transverse shear forces, the
torque and the two bending moments — the VABS ordering $\{F_1\;F_2\;F_3\;M_1\;M_2\;M_3\}$ that
the shipped load table `ff51_rmc_reform.dat` is written in (`# eta F1 F2 F3 M1 M2 M3 (RM
CENTER-ref BeamDyn, VABS order)`). The law is

$$
\begin{Bmatrix} F_1 \\ F_2 \\ F_3 \\ M_1 \\ M_2 \\ M_3 \end{Bmatrix}
=
\begin{bmatrix}
S_{11} & S_{12} & S_{13} & S_{14} & S_{15} & S_{16} \\
S_{12} & S_{22} & S_{23} & S_{24} & S_{25} & S_{26} \\
S_{13} & S_{23} & S_{33} & S_{34} & S_{35} & S_{36} \\
S_{14} & S_{24} & S_{34} & S_{44} & S_{45} & S_{46} \\
S_{15} & S_{25} & S_{35} & S_{45} & S_{55} & S_{56} \\
S_{16} & S_{26} & S_{36} & S_{46} & S_{56} & S_{66}
\end{bmatrix}
\begin{Bmatrix} \bar\epsilon_{11} \\ \gamma_{12} \\ \gamma_{13} \\
\kappa_{1} \\ \kappa_{2} \\ \kappa_{3} \end{Bmatrix}.
$$

The matrix is fully populated in general — a composite blade section couples extension to
bending and twist — but its diagonal carries the familiar engineering meaning

$$
\mathrm{diag}(\mathbf S) = \left[\;EA\;\;GA_2\;\;GA_3\;\;GJ\;\;EI_2\;\;EI_3\;\right],
$$

which is how the drivers label their printout (`LBL = ["EA", "GA2", "GA3", "GJ", "EI2",
"EI3"]` in `examples/OpenSG_shell/IEA_blade_beam/1_homo_beam_props.py`). For the IEA-22 blade
station at $r/R = 0.20$ that diagonal is
`[2.7661e+10 7.1861e+08 4.2174e+08 2.4376e+09 3.5144e+10 6.8130e+10]` in SI units.

### The classical $4\times4$ that rides along

The Timoshenko reduction is built on top of a zeroth-order (Euler–Bernoulli) solve, and that
intermediate law is a genuine classical beam stiffness in its own right. It drops the two
transverse shears:

$$
\begin{Bmatrix} F_1 \\ M_1 \\ M_2 \\ M_3 \end{Bmatrix}
=
\mathbf A_{\mathrm{EB}}
\begin{Bmatrix} \bar\epsilon_{11} \\ \kappa_1 \\ \kappa_2 \\ \kappa_3 \end{Bmatrix},
\qquad \mathbf A_{\mathrm{EB}} \in \mathbb R^{4\times4}.
$$

On the solid route it is returned as `r["C_eff_EB"]` alongside the Timoshenko `r["C_eff"]`
(see `_beam_homo_kkt` in `opensg_solid/sg_homo.py`, where it is formed as
$(D_{ee} + D_1^{V_0})/\omega$ before the first-order ladder runs). It is not written to the
`.out` — the `.out` always carries the $6\times6$.

### Which routes produce it

| route | entry point | SG input |
|---|---|---|
| msg-solid beam | `opensg_solid.sg_homo.plate_homo_2d(..., n_model=1)` | 2-D or 3-D solid SG (`n_sg >= 2`) |
| msg-shell ring | `opensg_shell.build_rm_bundle` (calls `ring_indep`) | 1-D shell SG (cross-section contour) |
| msg-shell segment | `opensg_shell.segment_timo_from_3dyaml` | 3-D shell segment yaml (tapered or prismatic) |

The msg-solid route runs the KKT beam driver: a zeroth-order $V_0$ solve under four
rigid-body Lagrange constraints, then a first-order $V_{1s}$ solve reusing the same
factorization, then the $6\times6$ reduction in `finalize_v1_and_compute_deff`. From
`examples/OpenSG-solid/7_get_beam_props_from_SG`:

```bash
python beam_homo_sg.py
```

which writes `RHC_SW_2UC_45.out` with the banner ` OpenSG msg-solid beam model, omega 1`.

The msg-shell ring route homogenizes the cross-section contour with the constrained 6-DOF
shell element (independent drilling $\omega_3$ enforced by element-constant Lagrange
multipliers, $\gamma_{23}$ tied by MITC) and takes its wall transverse shear from the MSG
least-squares plate solution. From `examples/OpenSG_shell/IEA_blade_beam`:

```bash
opensg iea_s10_shell.yaml
```

which writes `iea_s10_shell_Timo.out`, banner
` OpenSG msg-shell beam model [ext sh2 sh3 twist bend2 bend3]`, plus the per-section wall laws
in `iea_s10_shell_ABDG.out`. (`python 1_homo_beam_props.py` is the same homogenization inside
the blade pipeline, where it also emits the station figures.)

The segment route replaces axial periodicity by Dirichlet data: each end cross-section is
extracted topologically, solved on its own with `ring_indep`, and its $V_0$/$V_1$ fields are
imposed on the segment's boundary nodes; the segment energy is divided by the segment length
$L$. It writes `<yaml>_Timo.out` with the same `Timoshenko` title.

## 2. Plate — classical ABD $6\times6$ and Reissner–Mindlin ABDG $8\times8$

### Classical (Kirchhoff–Love) plate law

The classical plate model pairs three membrane stress resultants and three moment resultants
with three membrane strains and three curvatures:

$$
\begin{Bmatrix} N_{11} \\ N_{22} \\ N_{12} \\ M_{11} \\ M_{22} \\ M_{12} \end{Bmatrix}
=
\begin{bmatrix} \mathbf A & \mathbf B \\[2pt] \mathbf B^{\mathsf T} & \mathbf D \end{bmatrix}
\begin{Bmatrix} \varepsilon_{11} \\ \varepsilon_{22} \\ 2\varepsilon_{12} \\
\kappa_{11} \\ \kappa_{22} \\ \kappa_{12} \end{Bmatrix},
\qquad
\mathbf A, \mathbf B, \mathbf D \in \mathbb R^{3\times3}.
$$

The row order `[N11 N22 N12 M11 M22 M12]` is stated in the module header of
`opensg_solid/sg_homo.py` and echoed by the driver in
`examples/OpenSG-solid/3_get_plate_props_from_2D_SG`. The $\mathbf B$ block is the
extension–bending coupling and vanishes only for a laminate that is symmetric about the chosen
reference surface. Run it with

```bash
opensg RHC_SW_2UC_45.yaml
```

which writes `RHC_SW_2UC_45.out`, banner ` OpenSG msg-solid plate model, omega 35.248001,
periodic`, title `The Effective Classical Plate Stiffness Matrix` at `refined: 0`.
(`python plate_homo_2dsg.py` is the same call through the API.)

### Reissner–Mindlin plate law

The shear-refined model adds the transverse-shear pair. Its strain vector grows to eight
components and the resultant vector gains the two transverse shear forces $Q_1, Q_2$:

$$
\begin{Bmatrix} \mathbf N \\ \mathbf M \\ \mathbf Q \end{Bmatrix}
=
\begin{bmatrix}
\mathbf A & \mathbf B & \mathbf 0 \\[2pt]
\mathbf B^{\mathsf T} & \mathbf D & \mathbf 0 \\[2pt]
\mathbf 0 & \mathbf 0 & \mathbf G
\end{bmatrix}
\begin{Bmatrix} \boldsymbol\varepsilon \\ \boldsymbol\kappa \\ \boldsymbol\gamma \end{Bmatrix},
\qquad
\boldsymbol\gamma = \{\,2\gamma_{13}\;\;2\gamma_{23}\,\}^{\mathsf T},
\qquad
\mathbf Q = \{\,Q_1\;\;Q_2\,\}^{\mathsf T} .
$$

The zero off-diagonal blocks are structural, not an approximation of convenience: the code
assembles the $8\times8$ as `ABDG[:6,:6] = A6` and `ABDG[6:,6:] = G_msg`, so the in-plane law
and the transverse-shear law are computed by two separate variational problems and stacked.
The $2\times2$ $\mathbf G$ is obtained as $\mathbf G = \mathbf X^{-1}$ from the $\mathbf U^*$
least-squares reduction (a 78-equation / 27-unknown system) of the first-order warping ladder;
the relative residual of that fit is reported as `r["Ustar_rel"]`, and the block is returned as
`None` if $\mathbf X$ fails the SPD gate.

Two independent routes produce the $8\times8$:

**1-D SG route** — `opensg_solid.rm_plate_1D.msg_rm_plate.rm_plate_msg` solves the
through-thickness SG of a single laminate and returns `A6` (the $6\times6$ ABD), `G_msg`, and
`ABDG`. Which of the two is printed is selected by `model:` in the user's `layup_db.yaml`:
`model: 0` prints the classical $6\times6$, `model: 1` prints the full $8\times8$. From
`examples/OpenSG-solid/1_get_plate_props_from_1DSG`:

```bash
python 1d_sg.py
```

The input the user edits is `layup_db.yaml` alone — reference surface (`fraction: 0.5` = the
mid-surface), materials with densities, the stacking sequence bottom-ply-first, and the SG
discretisation (`n_per_layer`, `elem_order: 4` = the 5-noded quartic element that represents
the warping ladder exactly). The shipped case is the Nayak Example-5 sandwich,
$(0/90/0/90/\text{core})_s$, nine plies, $h = 0.1524$ m, graphite/epoxy faces over a HEREX
C70.130 PVC foam core. The run writes the SG mesh (`1dsg.yaml` plus the figure below) and the
law in `layup_db_plate_homo.out`, whose leading entries are $A_{11} = 5.4917\times10^{8}$ N/m,
$D_{11} = 3.0005\times10^{6}$ N·m, $G_{11} = 7.7010\times10^{6}$ N/m and
$G_{22} = 7.5424\times10^{6}$ N/m — the thin stiff faces carry the membrane and bending
channels while the thick compliant core sets the transverse shear.

![Through-thickness 1-D structure gene of the sandwich laminate, faces in blue and foam core in orange, with the mid-surface reference marked](_static/plate_1dsg.png)

**2-D / 3-D SG route** — `plate_homo_2d(..., n_model=2, shear_refined=True)` runs
`plate_shear_ladder` on the general SG assembly after the classical solve. It adds
`r["G_msg"]` ($2\times2$), `r["A6_ladder"]`, `r["X_shear"]` and `r["ABDG"]` to the result dict,
and the `.out` switches its title to `Reissner-Mindlin Plate` and its matrix to the $8\times8$.
Note the derivation status recorded in `plate_shear_ladder`: for `n_sg = 1` this *is* the Yu
(2005) 1-D formulation, gated against the closed-form anchors; for `n_sg = 2` or `3` it is the
SwiftComp-style extension, gated on the laminate-as-strip equivalence but **not** yet validated
against an external reference for SGs that are heterogeneous *in plane*.

## 3. Solid — the equivalent 3-D law

When the macro model is ordinary 3-D elasticity, the SG returns a $6\times6$ anisotropic
stiffness in the Voigt order $[11\;22\;33\;23\;13\;12]$ with engineering shear strains:

$$
\begin{Bmatrix}
\sigma_{11} \\ \sigma_{22} \\ \sigma_{33} \\ \sigma_{23} \\ \sigma_{13} \\ \sigma_{12}
\end{Bmatrix}
=
\mathbf C^{\,\mathrm{3D}}
\begin{Bmatrix}
\bar\Gamma_{11} \\ \bar\Gamma_{22} \\ \bar\Gamma_{33} \\
2\bar\Gamma_{23} \\ 2\bar\Gamma_{13} \\ 2\bar\Gamma_{12}
\end{Bmatrix},
\qquad \mathbf C^{\,\mathrm{3D}} \in \mathbb R^{6\times6}.
$$

On the msg-shell routes axis 1 is the prismatic (beam) direction and axes 2, 3 span the
cross-section, as recorded in `opensg_shell.sg_homo.GBAR_ORDER`.

This is the only law for which the `.out` also prints an engineering-constants block. It is
computed from the compliance $\mathbf S = (\mathbf C^{\,\mathrm{3D}})^{-1}$ by the
orthotropic-approximation formulas

$$
E_1 = \frac{1}{S_{11}},\quad E_2 = \frac{1}{S_{22}},\quad E_3 = \frac{1}{S_{33}},
\qquad
G_{12} = \frac{1}{S_{66}},\quad G_{13} = \frac{1}{S_{55}},\quad G_{23} = \frac{1}{S_{44}},
$$

$$
\nu_{12} = -\frac{S_{12}}{S_{11}},\qquad
\nu_{13} = -\frac{S_{13}}{S_{11}},\qquad
\nu_{23} = -\frac{S_{23}}{S_{22}} .
$$

The word *approximated* in the printed heading is load-bearing: these nine numbers are exact
only if $\mathbf C^{\,\mathrm{3D}}$ is genuinely orthotropic in the SG axes. A cell with
normal–shear coupling still prints them, and they will not reproduce the full matrix. The
matrix, not the constants block, is the result.

Three entry points produce it:

| entry point | SG | normalisation written to the `.out` |
|---|---|---|
| `plate_homo_2d(..., n_model=3)` | 1-D/2-D/3-D solid SG | $\mathbf C_{\mathrm{eff}} = (\bar{\mathbf D} + \mathbf D_1)/\omega$ |
| `opensg_shell.build_solid_bundle` | 1-D shell SG (cross-section contour) | $\mathbf C^{\,\mathrm{3D}} = \mathbf D_{\mathrm{eff}}/A_{\mathrm{cell}}$ |
| `opensg_shell.sg_homo.shell_sg3d` | 3-D shell SG (surface mesh) | $\mathbf D_{\mathrm{eff}}/V_{\mathrm{cell}}$ |

The last row carries a normalisation subtlety worth stating plainly: `shell_sg3d` *returns*
`C3D` divided by the SG measure (the midsurface area, the 3-D analog of the plane-section
perimeter), but *writes* the `.out` divided by the bounding-box cell volume, so the printed
moduli compare directly against a solid SwiftComp `.K` file.

Both `build_solid_bundle` and `shell_sg3d` **default** to periodicity — opposite faces, edges
and corners tied — because a free cell driven by six macro strains is rank one by construction:
every mode except $\bar\Gamma_{11}$ is cancelled at zero energy by an affine fluctuation that
only periodicity forbids. `shell_sg3d` and `plate_homo_2d` both also accept
`boundary="aperiodic"`, which instead prescribes zero fluctuation on the bounding-box-face
nodes; that is a kinematic (Dirichlet) treatment and therefore an upper bound on a single cell.
On the shell route the clamp is applied to the three translations only, leaving the rotational
DOFs natural, because clamping the rotations as well over-stiffens the cut edges. The shipped
head-to-head of the two treatments is
`examples/OpenSG_shell/4_get_solid_props_from_shell_3D_SG/compare_boundary_modes.py`.

The TPMS 3-D solid SG shipped in
`examples/OpenSG-solid/6_get_solid_props_from_3D_SG/sample_1` is the reference case for this
law:

```bash
opensg Sample_1.yaml
```

(`python run_sample_1.py` is the same homogenization through the API, plus the parse of the
vendor `.K` and the comparison table.)

Its `Sample_1.out` reports $E_1 = 6.1641278\times10^{9}$, $E_2 = 6.1641815\times10^{9}$,
$E_3 = 6.1640078\times10^{9}$ Pa, $G_{12} = 4.5197549\times10^{9}$ Pa and
$\nu_{12} = 0.35652386$ for TPMS Sample_1 in aluminium ($E = 69$ GPa, $\nu = 0.3$) — a cubic
cell, correctly returning three equal Young's moduli from three independent solves.

The banner reads ` OpenSG msg-solid 3D elastic model, omega 1, periodic`. **$\omega$ for a 3-D
solid SG is the node bounding-box volume** — the volume the equivalent continuum occupies and
the cell the periodic map ties — not the summed element (material) volume. This cell is a unit
cell, so $\omega = 1$ and the reported law is the per-unit-cell law. The practical consequence
is that the `.out` is now **directly** comparable with a SwiftComp `.K`: $C_{11} =
1.0190158\times10^{10}$ Pa is 10190.1580 MPa against the `.K`'s 10190.1580, with no rescale.
Under the older material-volume convention the file was $1/\text{relative density}$ too stiff
and only the comparison script's $C\,\omega$ rescale brought the tables into agreement. The
same nine constants appear in `Sample_1_periodic.out` one directory up, the copy saved by
`compare_boundary_modes.py`. (The sibling `tpms_solid_props.py` runs the same two samples at
$E = 70$ GPa and writes `Sample_1_msg_solid.out`, so its numbers will not match these.)

## 4. The wall law — per-section $8\times8$ ABDG

The msg-shell routes need a plate law for **each wall of the cross-section** before they can
solve the shell SG at all, and that intermediate is exported so it can be inspected
independently. `opensg_shell.sg_homo.write_abdg_out` writes one $8\times8$ stiffness and its
compliance per layup section to `<yaml>_ABDG.out`. Every route that reduces a layup to a wall
plate law calls it — the cross-section rings through `build_rm_bundle` and
`build_solid_bundle`, and the 3-D shell SG through `shell_sg3d` — so it appears next to every
shell `.out`, and the halfway point of the two-step reduction is always inspectable.

The file header states the row convention verbatim:

```
 OpenSG msg-shell wall plate laws, one block per section
 rows [eps11 eps22 2eps12 | K11 K22 K12+K21 | 2g13 2g23]
```

so the block structure is the same $[[\mathbf A,\mathbf B,\mathbf 0],
[\mathbf B^{\mathsf T},\mathbf D,\mathbf 0],[\mathbf 0,\mathbf 0,\mathbf G]]$ as in
section 2, with the shell operators' twisting measure $K_{12}+K_{21}$ in the sixth slot.
Each block is preceded by its section identity and full layup, for example

```
 section 0: layup_0  layup [['gelcoat', 0.0005, 0.0], ['glass_triax', 0.00300294, 0.0], ['glass_uniax', 0.02609467, 0.0], ['glass_triax', 0.00300294, 0.0]]
```

Two facts about how these blocks are built matter when comparing them against hand
calculations. First, the ABD is shifted by the parallel-axis rule to the reference surface the
YAML declares (`reference: center` gives the laminate mid-surface, `fraction = 0.5`), so the
$\mathbf B$ block depends on that choice. Second, the transverse-shear block $\mathbf G$ is
**always replaced** by the MSG least-squares result from `rm_plate_msg` — it is not a
shear-correction-factor estimate, and there is no switch that selects anything else.

## 5. Where each law is produced and written

| macro law | producing function | output file | title in the `.out` |
|---|---|---|---|
| Timoshenko beam $6\times6$ | `opensg_solid.sg_homo.plate_homo_2d(n_model=1)` → `_beam_homo_kkt` | `<sg>.out` | `The Effective Timoshenko Beam Stiffness Matrix` |
| Timoshenko beam $6\times6$ | `opensg_shell.build_rm_bundle` → `ring_indep` | `<yaml>_Timo.out` | `The Effective Timoshenko Beam Stiffness Matrix` |
| Timoshenko beam $6\times6$ | `opensg_shell.segment_timo_from_3dyaml` | `<yaml>_Timo.out` | `The Effective Timoshenko Beam Stiffness Matrix` |
| Euler–Bernoulli beam $4\times4$ | `_beam_homo_kkt` → `r["C_eff_EB"]` (`refined: 0`) | `<sg>.out` | `The Effective Euler-Bernoulli Beam Stiffness Matrix` |
| Classical plate ABD $6\times6$ | `plate_homo_2d(n_model=2)` | `<sg>.out` | `The Effective Classical Plate Stiffness Matrix` |
| RM plate ABDG $8\times8$ | `plate_homo_2d(n_model=2, shear_refined=True)` → `plate_shear_ladder` | `<sg>.out` | `The Effective Reissner-Mindlin Stiffness Matrix` |
| RM plate ABDG $8\times8$ | `opensg_solid.rm_plate_1D.msg_rm_plate.rm_plate_msg` → `r["ABDG"]` | `<layup_db>_plate_homo.out` | `The Effective Reissner-Mindlin Stiffness Matrix` |
| 3-D solid $6\times6$ | `plate_homo_2d(n_model=3)` | `<sg>.out` | `The Effective Cauchy Continuum Stiffness Matrix` + engineering constants |
| 3-D solid $6\times6$ | `opensg_shell.build_solid_bundle` → `ring_solid` | `<yaml>_C3D.out` | `The Effective Cauchy Continuum Stiffness Matrix` + engineering constants |
| 3-D solid $6\times6$ | `opensg_shell.sg_homo.shell_sg3d` | `<yaml>_C3D.out` | `The Effective Cauchy Continuum Stiffness Matrix` + engineering constants |
| Wall plate law $8\times8$, per section | `opensg_shell.sg_homo.write_abdg_out` | `<yaml>_ABDG.out` | `The Effective Reissner-Mindlin Plate Stiffness Matrix` (one block per section) |

## 6. Running the laws backwards: two-step recovery

Homogenization is only half of MSG. Each law above was obtained together with the fluctuation
(warping) fields that produced it, and those fields let the macro answer be pushed back down to
pointwise 3-D stress and strain inside the SG. The rule is *homogenize once, recover as often
as needed*: every field the recovery needs rides along in the result dict.

### The shell chain: beam force → plate resultants → 3-D stress

Verified end to end in `examples/OpenSG_shell/IEA_blade_beam/2_dehom_v2_vabs.py`, which runs

```bash
python 2_dehom_v2_vabs.py
```

The chain has two interfaces, and naming them is the clearest way to see why it works.

**Step 2a — beam to wall (`opensg_shell`).** The macro beam strain follows from inverting the
Timoshenko law, $\mathbf{st} = \mathbf C_6^{-1}\,\mathbf{FF}$, with `FF` the six beam
force/moment components in VABS order. `_macro_fields` recombines the ring warping modes
$V_0, V_1$ into the nodal fields $w$ and $w'$, and `_rm_shell_strain` then evaluates, per ring
element, the six shell strains
$\mathbf s_6 = [\varepsilon_{11}\;\varepsilon_{22}\;2\varepsilon_{12}\;\kappa_{11}\;
\kappa_{22}\;2\kappa_{12}]$ (with the transverse pair $[2\gamma_{13}\;2\gamma_{23}]$ alongside).
Multiplying by that wall's own MSG ABD gives the interface quantity handed to the plate SG:

$$
\mathbf F_6 \;=\; \mathbf A_6\,\mathbf s_6
\;=\; [\,N_{11}\;\;N_{22}\;\;N_{12}\;\;M_{11}\;\;M_{22}\;\;M_{12}\,]^{\mathsf T}.
$$

These are exactly the resultants of section 2 — the beam model has handed each wall a plate
problem. They are written to `<name>_elem_plate_forces.dat`, one row per ring element, with
columns `elem  y2(m) y3(m)  e11 e22 2e12 k11 k22 2k12  N11 N22 N12(N/m) M11 M22 M12(N)`.

**Step 2b — wall to 3-D (`opensg_solid.rm_plate_1D`).** For each layup, `rm_plate_msg` builds
the through-thickness SG once; `msgrm_strain_at_depth` (or its vectorized twin
`msgrm_strain_at_depth_batch`) then returns the 3-D Voigt strain and stress at any depth $z$.
Passing only the first strain gradients `dE1`/`dE2` selects the first-order recovery
(Yu Eq. 63); adding `dE11`/`dE12`/`dE22` selects the second-order recovery (Eq. 66), which is
what carries the transverse pair $\sigma_{33}/\sigma_{13}$ that first order cannot produce.
Output is in the ply material frame by default, since failure criteria live there.

The IEA-22 station at $r/R = 0.20$ is validated against a VABS 2-D solid solution of the same
section, sampled at every VABS Gauss point. From `iea_s10_report.txt`, RMS difference in MPa,
web / skin:

| component | first order | second order (V2) | VABS RMS |
|---|---|---|---|
| $\sigma_{11}$ | 43.073 / 9.529 | 43.073 / 9.529 | 44.440 / 96.676 |
| $\sigma_{22}$ | 0.612 / 0.482 | 0.612 / 0.482 | 0.704 / 0.998 |
| $\sigma_{12}$ | 6.799 / 1.255 | 6.798 / 1.255 | 22.026 / 6.009 |
| $\sigma_{33}$ | 0.081 / 0.049 | 0.081 / 0.049 | 0.081 / 0.049 |
| $\sigma_{13}$ | 0.394 / 0.192 | 0.394 / 0.192 | 0.394 / 0.193 |
| $\sigma_{23}$ | 0.035 / 0.028 | 0.035 / 0.028 | 0.035 / 0.028 |

Read the last column with the others: the dominant $\sigma_{11}$ is recovered to roughly 10 %
of its own RMS in the skin, while the transverse components $\sigma_{33}$, $\sigma_{13}$,
$\sigma_{23}$ show differences equal to the VABS RMS itself — i.e. the shell chain returns
essentially zero there. The gap in the transverse pair is the honest open item on this route,
and it is why the second-order drivers exist even though their payload at this near-constant
load state is small (V2 changes $\sigma_{11}$ by $2.0\times10^{-4}$ MPa RMS against a
first-order RMS of $8.9\times10^{1}$ MPa).

The recovered in-plane stress over the full section looks like this — the spar caps in
compression above and tension below under the flapwise moment, with the shear flow visible in
$\sigma_{12}$ around the leading edge and trailing-edge panels.

![Recovered sigma_11 and sigma_12 over the IEA-22 blade cross-section at r over R equal 0.20, plotted at the recovery points around the contour and webs](_static/shell_dehom_stress.png)

### The solid chain

For an SG homogenized through `plate_homo_2d`, recovery is a single call in
`opensg_solid.sg_dehom`:

```python
from opensg_solid.sg_homo import plate_homo_2d
from opensg_solid.sg_dehom import dehom_fields, export_gauss

r = plate_homo_2d("RHC_SW_2UC_45.yaml", n_model=2)
epsilon_bar = [0.001, 0.0, 0.0, 0.0, 0.01, 0.0]
Gamma, Sigma, U = dehom_fields(r, epsilon_bar)
export_gauss(r, Gamma, Sigma, "RHC_SW_2UC_45_dehom", U_eqd=U)
```

`epsilon_bar` is whichever macro state matches the model that was homogenized: the six plate
strains $[\varepsilon_{11}\;\varepsilon_{22}\;2\varepsilon_{12}\;\kappa_{11}\;\kappa_{22}\;
2\kappa_{12}]$ for `n_model=2`, the six macro strains of section 3 for `n_model=3`, or the six
Timoshenko beam strains of section 1 for `n_model=1`. The call returns, at every element Gauss
point, the local strain `Gamma` $(E, Q, 6)$, the local stress `Sigma` $(E, Q, 6)$ — both in
SwiftComp storage order $(xx, yy, zz, yz, xz, xy)$ — and the fluctuation-only displacement `U`
$(E, Q, 3)$. `plate_dehom_2d` is a two-output view of the same single pass, and `gauss_coords`
returns the physical coordinates of the recovery points.

`export_gauss` writes five files from one recovery: `<prefix>.txt` (the Gauss table),
`<prefix>.vtk` (a point cloud for ParaView), and the OpenSG dehom files `<prefix>.SM` (stress),
`<prefix>.EM` (strain) and `<prefix>.U` (fluctuation displacement). Note the printed component
order in the `.SM`/`.EM`/`.U` files is $11\;22\;33\;12\;13\;23$ — the reorder happens on write
only; storage stays SwiftComp order. Runnable drivers are
`examples/OpenSG-solid/4_get_plate_dehom_from_2DSG/plate_dehom_2dsg.py` for the plate model and
`examples/OpenSG-solid/8_get_beam_dehom_from_SG/beam_dehom_sg.py` for the beam model.
