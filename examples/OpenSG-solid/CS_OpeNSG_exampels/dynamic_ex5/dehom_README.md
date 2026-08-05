# The OpenSG-RM dehomogenization files

```
rm_dehom.py       ONE station:  1dsg.yaml + FF data in the code
                  -> <sg>_dehom.SM / .EM / .U / .vtk        (Gauss depths)
rm_dehom_dat.py   WHOLE plate:  an Abaqus shell field .dat
                  -> <dat>.SM / .EM / .U / _gauss.vtk       (40 x 40 x 45)
```

Both re-run the homogenization every call (`rm_plate_msg` re-solves the
nodal warping ladders V0/V11/V12 on the 1-D SG line — milliseconds), and both
write their `.vtk` **by default**, so `render_fields.py` / `render_deformed.py`
always have their input. `rm_dehom_dat.py` imports everything per-depth from
`rm_dehom.py`; it owns only what is per-`.dat` (parsers, driver fields, the
`jax.vmap`, the Q-rescale, the σ33 momentum route).

## Nomenclature — every symbol used in these files

| symbol | meaning | units |
|---|---|---|
| **SG** | Structure Gene: the 1-D through-thickness line mesh (`1dsg.yaml`) the plate is homogenized on — one 5-noded quartic element per ply | — |
| **FF** | the "force field" input: the section resultants at a station, i.e. what a shell/beam solution hands the recovery | — |
| `N11, N22, N12` | membrane (in-plane) force resultants, Abaqus `SF1 SF2 SF3` | N/m |
| `M11, M22, M12` | bending/twisting moment resultants, Abaqus `SM1 SM2 SM3` | N |
| `Q1, Q2` | transverse shear resultants, Abaqus `SF4 SF5` | N/m |
| `F6` | the FF 6-vector `[N11 N22 N12 M11 M22 M12]` | — |
| `dF1, dF2` | in-plane gradients of the resultants, d(F6)/dx and d(F6)/dy — the strain gradients follow as `E6,α = S6 @ dFα` | per-m |
| `A6` | the 6×6 plate stiffness (ABD) from the homogenization; `ABDG` adds the 2×2 MSG transverse-shear block | — |
| `S6` | `inv(A6)`: converts resultants to plate strains | — |
| `E6` | the plate strains `[e11 e22 g12 k11 k22 k12]` = `S6 @ R6` — the primary recovery driver | —, 1/m |
| `DE1, DE2` | in-plane gradients d(E6)/dx, d(E6)/dy — drive the first-order warping → **σ13, σ23** | 1/m, 1/m² |
| `D11, D12, D22` | second gradients E,11 E,12 E,22 — drive the **σ33** interior build-up | 1/m² … |
| `QT6` | the surface-load ladder `[q, q,1, q,2, q,11, q,12, q,22]` of the APPLIED pressure — the face-traction condition σ33 = −q | Pa, Pa/m … |
| `V0` | the classical (zeroth-order) warping solution: 3 dofs per SG node per plate strain | — |
| `V11, V12` | the first-gradient warping corrections (one per gradient direction) | — |
| `qt6` ladder | same as QT6, the load's own warping column | — |
| `fraction` | where the reference surface sits: 0 = bottom face, 0.5 = mid-plane, 1 = top | — |
| `z` (SG frame) | through-thickness coordinate with **0 at the reference surface** (the shell's node plane); the plate mid-plane is at `(0.5 − fraction)·h` | m |
| `h`, `H` | total laminate thickness (0.1524 m in ex5) | m |
| `E11 … E23` (`.EM`) | the recovered 3-D strain components, global frame | — |
| `S11 … S23` (`.SM`) | the recovered 3-D stress components, global frame | Pa |
| `U1 U2 U3` (`.U`) | the 3-D displacement `U_a = u_a − z·w,_a + warp_a`, `U3 = w + w3` | m |
| warping (`w1 w2 w3`) | the fluctuation displacement the SG adds on top of the plate kinematics — what a bare shell node cannot show | m |
| Gauss lattice | output stations: the 2×2 in-plane integration points × the 5-point Gauss depths of each SG element — all strictly inside a ply | — |
| Q-rescale | per-station scaling of the σa3 profiles so ∫σ13 dz = Q1 and ∫σ23 dz = Q2 exactly | — |
| momentum route | σ33 from s33,3 = ρü3 − σ13,1 − σ23,2 integrated from the free face (needs a `--hist` deflection history for ü3) | — |
| SG Voigt order | the internal component order (11, 22, 33, **23, 13, 12**) — files are written in the conventional order 11 22 33 12 13 23 | — |

## The FF data

```python
F6  = [N11, N22, N12, M11, M22, M12]    # [N/m, N/m, N/m, N, N, N]
Q2v = [Q1, Q2]                          # [N/m]
```

The plate strains driving the recovery are `E6 = inv(A6) @ F6` with the A6 of
the re-run homogenization — the FF never bypasses the section law.

**From resultant gradients to strain gradients.** If you know how the FF
varies in-plane (`dF1 = d(F6)/dx`, `dF2 = d(F6)/dy`), the strain gradients
follow through the same section law: the plate is uniform in-plane, so `S6 =
inv(A6)` is a CONSTANT matrix and differentiation commutes with it,

```
E6 = S6 @ F6      =>      E6,1 = S6 @ dF1,      E6,2 = S6 @ dF2
```

i.e. `DE1 = S6 @ dF1` and `DE2 = S6 @ dF2` — one matrix multiply each, no new
physics. (On a plate whose section varies in-plane, S6 would carry its own
gradient and this shortcut would gain a `dS6 @ F6` term.) Everything below
about "gradients" is about obtaining `dF1/dF2`; converting them to `DE1/DE2`
is always this one line.

**Q1/Q2 are deliberately NOT part of the 6-vector, and never become one.**
`F6` closes on its own through the 6×6 A6; the transverse shears are a
separate 2-vector because they play two entirely different roles:

1. there is **no shear slot in the 42-driver vector** `D = [E6, E6,1, E6,2,
   E6,11, E6,12, E6,22, qt6]` — in the asymptotic construction σ13/σ23 are
   produced by the *gradient* brackets (step 4 below), not by a shear-strain
   input. Q reaches the recovery (a) through the gradients via equilibrium,
   `M11,1 + M12,2 = Q1` (Recipe 1), and (b) through the per-station rescale
   `∫σ13 dz = Q1`;
2. the RM shear strain `γ = G⁻¹ Q` (the 2×2 block of the 8×8 ABDG) belongs
   to the **plate FE solve** — it is what the deck hands Abaqus as
   `*TRANSVERSE SHEAR STIFFNESS` so the shell deflects with the right shear
   compliance. It is not a recovery driver.

## The optional inputs, and WHERE THEY COME FROM given a single FF

This is the question that matters, because a single station has no
neighbours: **gradients cannot be computed from one FF value — they must come
from the physics around it.** Each optional input feeds exactly one part of
the 3-D state:

| input | feeds | without it |
|---|---|---|
| `DE1, DE2` — strain gradients | σ13, σ23 (the V11/V12 warping) | σ13 = σ23 ≡ 0 |
| `D11, D12, D22` — 2nd gradients | the σ33 interior distribution | σ33 loses its build-up |
| `QT6` — surface-load ladder | the face tractions (σ33 = −q on the loaded face) | σ33 = 0 at that face |

### Recipe 1 — one FF station: classical recovery (gradients zero)

**The equilibrium closure is REMOVED as a recipe** (it used to be here:
`dF1 = [0,0,0,Q1,0,Q2]`, `DE1 = S6 @ dF1` from `M11,1 = Q1`). Plate
equilibrium supplies only 5 equations for the 30 unknown gradient
components (the indeterminacy section below), so as a *general* closure it
smuggles in a 1-D modal assumption. The rule now, everywhere in the code
(`cli_rm_plate.py`, example 7, the single-station `rm_dehom.py`):

**one FF station → all gradients zero → classical recovery.** σ11/σ22/σ12
layer-by-layer exact for that FF; σ13 = σ23 = 0; σ33 from the applied-load
ladder only. Honest and well-defined — nothing is invented.

(Where the 1-D relation `M11,1 = Q1` IS exact — a known harmonic
cylindrical-bending mode — it belongs to Recipe 2 below as the analytical
mode derivative, not to a closure.)

### Recipe 2 — a known analytical mode

If the FF belongs to a known solution shape (our ex5 double-sine
q₀ sin(πx/a) sin(πy/b) is the textbook case), every field is that shape times
an amplitude, so the gradients are the mode's derivatives evaluated at the
station. At the plate centre:

```python
DE1 = DE2 = 0                    # sin peaks: first derivatives vanish
D11 = D22 = [-(pi/a)**2 * e for e in E6]     # mode-consistent
D12 = 0
QT6 = [q0, 0, 0, -(pi/a)**2*q0, 0, -(pi/b)**2*q0]
```

This is exactly what `compare_stress_top.py` does, and it is exact — no
finite differences anywhere.

### Recipe 3 — a line of FF stations (beam / blade span)

If the FF comes as a table along a line (BeamDyn stations, a GEBT solution),
finite-difference the NEIGHBOURING stations for the along-line gradient; the
transverse gradient stays zero unless a transverse line of stations exists
too. One FF row alone cannot do this — the table can.

### Recipe 4 — a surrounding 2-D field

The general answer, and what `rm_dehom_dat.py` automates: with resultants on
a grid, all gradients are finite differences of the grid and the user
supplies nothing. If you have any field solution — not just Abaqus — put it
on a grid and take the derivatives; the extras stop being inputs at all.

### The second gradients D11, D12, D22 — analytical formulas

Same commuting argument as the first gradients: they are the second
gradients of the RESULTANTS pushed through the constant section law,

```
D11 = E6,11 = S6 @ d2F11        d2F11 = d^2(F6)/dx^2
D12 = E6,12 = S6 @ d2F12        d2F12 = d^2(F6)/dx dy
D22 = E6,22 = S6 @ d2F22        d2F22 = d^2(F6)/dy^2
```

so the question is again about the resultants. **Differentiate the
equilibrium relations once more** and the second moment-gradients are pinned
by the FIRST gradients of Q — and those by the applied load:

$$M_{11,11} + M_{12,12} = Q_{1,1}\qquad M_{12,11} + M_{22,12} = Q_{2,1}$$
$$M_{11,12} + M_{12,22} = Q_{1,2}\qquad M_{12,12} + M_{22,22} = Q_{2,2}$$
$$Q_{1,1} + Q_{2,2} = -q \;\;\text{(the load you applied)}$$

**Single-FF closure (cylindrical bending, nothing varies along y):** the
chain collapses to one exact statement — $M_{11,11} = Q_{1,1} = -q$:

```python
d2F11 = [0, 0, 0, -q, 0, 0]      # M11,11 = -q : the applied pressure itself
D11 = S6 @ d2F11
D12 = D22 = 0                    # no y-variation
```

Note the ladder: **Q (in the FF) closed the first gradients; q (the load you
applied) closes the second.** Each order of the recovery is fed by one more
piece of information you already have.

**Known mode (ex5's double sine):** every resultant field is
$\hat F\sin(Px)\sin(Py)$, so pointwise

$$E_{6,11} = -P^2 E_6,\qquad E_{6,22} = -P^2 E_6,\qquad
E_{6,12} = +P^2\,\hat E\cos(Px)\cos(Py)$$

— the code's `d11 = d22 = -P*P*E6` (exact, no differencing), with $D_{12}=0$
at the plate centre by symmetry.

**Field data:** finite-difference the first gradients
(`d12 = grad(dE2)` in `rm_dehom_dat.py`). ex5 uses the mode-consistent form
for D11/D22 rather than the double FD precisely because differentiating twice
on a grid amplifies edge noise — that noise polluted σ33 near the boundary
until the mode-consistent form replaced it.

What they feed: the D-blocks multiply the $[\Gamma_{l\alpha}V_{1\beta}]$
brackets of step 4 — the **σ33 interior build-up**. Skip them and σ33 keeps
only its `qt6` face terms.

### The indeterminacy of equilibrium closure — and the FD answer

The closure recipes above deserve their honest boundary: plate equilibrium
supplies **five** equations ($N_{\alpha\beta,\beta}=0$ twice,
$M_{\alpha\beta,\beta}=Q_\alpha$ twice, $Q_{\alpha,\alpha}=-q$ once) against
**thirty** unknown gradient components (6×2 first + 6×3 second) — an
indeterminate system except under special symmetry (the cylindrical-bending
recipes close it only because nothing varies along y). The mode-consistent
second gradients likewise assume the double-sine solution.

The general answer needs no assumption at all: **the FE solution itself
carries the field.** Sample the strains at multiple points within each
element — the 2×2 integration points the deck already prints, a 40×40
lattice with true coordinates from the COORD table — and finite-difference
them. **This IS the pipeline method**: `rm_dehom_dat.py` computes all 30
gradient components by FD (first gradients, then gradients of the
gradients, mixed term symmetrised), `qt6` still closed-form because the
applied load is input, not closure. **σ33 is constitutive** — C(z)Γ(z)
with the FD drivers and the qt6 face-traction ladder; the equilibrium
(momentum-integral) σ33 route has been removed along with the closure
recipes. One honest note that removal carries: under strong inertia the
constitutive σ33 does not see the ρü₃ contribution the momentum route
integrated (measured at the ex5 snapshot: <1 % interior difference).

Tested side by side on the same ex5 field `.dat`, same operators:

| region | agreement FD vs mode-consistent |
|---|---|
| interior (4 ip trimmed per edge) | **< 1 % on every component** |
| whole field | 11–15 % in-plane, edge-band dominated |

So the two closures confirm each other where both are valid, and the FD
route's cost is exactly the known one: one-sided double-FD stencils in a
boundary band ~2 elements wide (the reason the mode form was introduced for
σ33 originally). For a general plate with no known mode, FD is the method;
trim or widen the stencil at edges.

### QT6 is never derivable from the FF

It is the APPLIED surface load and its derivatives at the station,
`[q, q,1, q,2, q,11, q,12, q,22]` — you know it because you applied it.
Uniform pressure: `[q, 0, 0, 0, 0, 0]`. Unloaded station: leave it `None`.

### The general input: the `.ff` station table + the `dehom:` yaml block

**Implemented** (`abq2ff.py` collects it from Abaqus, `rm_dehom_ff.py`
recovers from it — no Abaqus anywhere in the driver). The `.ff` is the
plate analog of a VABS `.glb`: one row per in-plane station, whitespace
separated, `#` comments, **any station geometry, any row order**:

```
x  y  N11 N22 N12 M11 M22 M12  Q1 Q2  [u1 u2 u3 wx wy]
```

- cols 3–8: the FF. `E6 = S6 · FF` per station; **every gradient driver
  comes from the stations themselves** (seconds only when `V2: 1`). One
  row → classical recovery. Two gradient paths, chosen automatically:
  - **lattice** — the coordinates form a complete rectangular lattice
    (structured quad mesh; row order irrelevant, the driver rebuilds it
    by `unique`/`searchsorted`): exact tensor-grid FD on the true
    non-uniform spacings. ~1 ms for 1600 stations.
  - **meshfree** — any other cloud (unstructured mesh, mixed element
    sizes): `mls_gradients` in the core — KD-tree k-nearest patches
    (k = 12) + distance-weighted local *quadratic* least squares; the
    derivatives are the fitted coefficients, firsts AND seconds from one
    batched 6×6 solve per station. Machine-exact for locally quadratic
    fields (pytest-gated); agrees with lattice FD to ~1e-4 in the ex5
    interior. ~30 ms for 1600 stations — the operator build (seconds) is
    the cost, never the gradients. The one-sided **edge band** caveat
    applies to both paths equally. `--scatter` forces this path.
    Scattered stations skip the structured-grid VTK (the `.SM/.EM/.U`
    tables carry per-station coordinates regardless).
- cols 9–10: Q1/Q2, used ONLY as the per-station rescale targets.
- cols 11–15 (optional): mid-surface `u1 u2 u3` and slopes `wx wy` for the
  `.U` displacement composition; absent → warping-only `.U`.
- any solver ("the local way") writes this table directly; `abq2ff.py` is
  just the Abaqus-side collector (ip SF/SM + COORD tables + nodal U
  bilinearly interpolated to the ip lattice).

The APPLIED face pressure is NOT in the table — loads are input, and they
live in `layup_db.yaml`:

```yaml
dehom:
  ff: Abaqus_results/ex5_shell_S8R_field.ff
  loads:
    top:  {q0: 68.9476e6, shape: sinsin}   # uniform | sin_x | sin_y | sinsin
    bottom: none                           # (a, b from the plate: block)
```

Each shape fixes the WHOLE 6-ladder analytically — `uniform →
[q0,0,0,0,0,0]`, `sin_x → q0[s, Pc, 0, -P²s, 0, 0]` with `P = π/a`,
`sinsin` → the ex5 double-sine ladder. Bottom-face loads raise
`NotImplementedError` for now (the 42-driver operator block carries qt6
only; split-face recovery is available through
`msgrm_strain_at_depth(qb6=...)` directly). A tabulated-load variant
(`top: {file: ...}`, 3-column `x y q` FD'd on its own grid or the full
8-column ladder) remains open for loads with no closed form.

## What you get with a bare FF (everything optional = None)

The classical ply stresses σ11, σ22, σ12 — layer by layer, exactly. The
shipped example in `rm_dehom.py` (N11 = 1 MN/m, M11 = 50 kN, all extras
None) shows the signature of that mode: S13 = S23 = 0 exactly, S33 at
machine zero. If those are the components you need, the extras exist for
you; if not, a bare FF is a complete classical recovery.

## The analytical chain — every equation from FF to 3-D stress

Notation: $z$ is the through-thickness coordinate (SG frame), $\langle\cdot\rangle=\int_{-fh}^{(1-f)h}(\cdot)\,dz$
the thickness integral, $C(z)$ the ply's rotated 6×6 stiffness at depth $z$,
and Voigt order (11, 22, 33, 23, 13, 12) throughout.

### 1. The kinematic ansatz (VAM)

The exact 3-D displacement is split into plate kinematics plus an unknown
through-thickness **fluctuation (warping)** $w_i$:

$$U_\alpha(x_1,x_2,z) = u_\alpha - z\,u_{3,\alpha} + w_\alpha(x_1,x_2,z),
\qquad U_3 = u_3 + w_3(x_1,x_2,z)$$

with the constraint $\langle w_i\rangle = 0$ (so $u_i$ *is* the plate
displacement, not redundant with the warping).

### 2. The 3-D strain in operator form

Substituting the ansatz into the 3-D strain definition and sorting by what
each term differentiates:

$$\Gamma(z) = \Gamma_\varepsilon(z)\,E_6 \;+\; \Gamma_h\,w \;+\;
\Gamma_{l1}\,w_{,1} \;+\; \Gamma_{l2}\,w_{,2}$$

- $E_6 = [\varepsilon_{11},\varepsilon_{22},2\varepsilon_{12},
  \kappa_{11},\kappa_{22},2\kappa_{12}]$ — the plate strains,
- $\Gamma_\varepsilon(z)$ maps them into 3-D strain (rows 11/22/12 get
  $\varepsilon + z\kappa$),
- $\Gamma_h = \partial_z$ acting on the warping (rows 33, 23, 13),
- $\Gamma_{l\alpha}$ place the in-plane warping gradients into the strain rows.

### 3. The warping ladder (what the homogenization solves)

Minimizing $\tfrac12\langle\Gamma^T C\,\Gamma\rangle$ order by order in the
small parameter $h/\ell$ gives a **ladder of 1-D problems on the SG line**,
all sharing the same operator $D_{hh}=\langle\Gamma_h^T C\,\Gamma_h\rangle$:

$$\text{order 0:}\quad D_{hh}\,V_0 = -\langle\Gamma_h^T C\,\Gamma_\varepsilon\rangle
\qquad\Rightarrow\; w^{(0)} = V_0\,E_6$$

$$\text{order 1:}\quad D_{hh}\,V_{1\alpha} =
-\big(\langle\Gamma_h^T C\,\Gamma_{l\alpha}\rangle
+ \langle\Gamma_{l\alpha}^T C\,\Gamma_h\rangle^T\big)V_0
- \langle\Gamma_{l\alpha}^T C\,\Gamma_\varepsilon\rangle
\qquad\Rightarrow\; w^{(1)} = V_{11}\,E_{6,1} + V_{12}\,E_{6,2}$$

$$\text{load:}\quad D_{hh}\,V_q = \text{(face-traction source)}
\qquad\Rightarrow\; w^{(q)} = V_q \cdot qt_6$$

These are the **nodal** arrays `V0_ns / V11_ns / V12_ns` (3 dofs per SG node
× 6 columns), solved fresh on every call. The homogenized stiffness falls out
of the same $V_0$:

$$A_6 = \big\langle(\Gamma_\varepsilon+\Gamma_h V_0)^T\,C\,
(\Gamma_\varepsilon+\Gamma_h V_0)\big\rangle$$

and the 2×2 transverse-shear block $G$ from the second-order energy
identification (Yu's least-squares construction).

### Provenance — what is Yu's and what is this project's

| term | source (equation numbers = Yu–Hodges–Volovoi, *Computers & Structures* 81 (2003) 439–454) |
|---|---|
| $V_0$, $A_6$, the ladder of §3 | the Reissner-like construction (IJSS 2002); its 1-D FEM form is C&S 2003 Eqs. (28)–(34) — the 5-noded element is *their* choice too |
| $V_{1\alpha}$ and the $\epsilon_{,\alpha}$ (DE1/DE2) recovery terms | **C&S 2003 Eq. (45)**: $V_1 = V_{11}\epsilon_{,1} + V_{12}\epsilon_{,2} + V_{1L}$, entering the recovery **Eqs. (62)–(63)** — where $\epsilon_{,\alpha}$ appears a second time through $V_{0,\alpha} = V_0\,\epsilon_{,\alpha}$. Verified here against Pagano to machine zero, including the raw-vs-detilted $\bar V_1$ convention (Yu's Section 5). |
| the second gradients D11/D12/D22 and $V_2$ | **C&S 2003 Eq. (66), FULLY IMPLEMENTED in the core** — `V21/V22/V23` per Eq. 64 in **two variants** (detilted-source for the in-plane rows, tilted-source for the through-thickness rows — the row-split fix of 2026-08-04) plus the **V2L load quintets** per face. Selected by the layup_db **`V2:` flag**, default 0 = Eq. 63 (the full second order overshoots thick in-plane face stress: ex5 σxx +8.67% → +0.32% at first order). `validate_v2.py` verdict after the fix: **σ33 with V2:1 beats first order for EVERY stack** — homogeneous 0.5–2.6%, [15/−15] 1.1%, [0/90/0] 1.2–5.6%, Yu case3 0.8–4.3%, sandwich 0.1–16.7% — with machine-exact face tractions. Set `V2: 1` when through-thickness accuracy on a thick section is the goal. |
| σ33 by through-thickness equilibrium (the momentum route) | Yu's route for σ33 — he never recovers it constitutively |
| the Q-rescale $\int\sigma_{a3}dz = Q_a$ | this project's dynamic-consistency rule, not Yu's |

**Why the dehom still needs $\epsilon_{,\alpha}$ as numbers even though "the
fluctuation is already solved":** the homogenization solves $V_{11}, V_{12}$
as *unit-response columns* — the warping per unit strain gradient. The
recovery (Eq. 63) is indeed direct, no solve — but it is a **contraction**,
$w = SV_0\,\epsilon + SV_{11}\,\epsilon_{,1} + SV_{12}\,\epsilon_{,2}$, and a
contraction needs both factors: the solved columns *and* the amplitudes
$\epsilon, \epsilon_{,\alpha}$ of your particular plate solution. With no
gradient values only the $V_0\epsilon$ term survives — the classical
recovery, zero transverse shear.

### 4. Assembling the 3-D strain — where every input lands

Insert $w = V_0 E_6 + V_{11}E_{6,1} + V_{12}E_{6,2} + V_q qt_6$ into step 2
and collect by driver:

$$\Gamma(z)=
\underbrace{[\Gamma_\varepsilon+\Gamma_h V_0]}_{\times\,E_6}
+\underbrace{[\Gamma_h V_{11}+\Gamma_{l1}V_0]}_{\times\,E_{6,1}\;(DE1)}
+\underbrace{[\Gamma_h V_{12}+\Gamma_{l2}V_0]}_{\times\,E_{6,2}\;(DE2)}$$
$$+\underbrace{[\Gamma_{l1}V_{11}]}_{\times\,E_{6,11}\;(D11)}
+\underbrace{[\Gamma_{l1}V_{12}+\Gamma_{l2}V_{11}]}_{\times\,E_{6,12}\;(D12)}
+\underbrace{[\Gamma_{l2}V_{12}]}_{\times\,E_{6,22}\;(D22)}
+\underbrace{[\ldots V_q\ldots]}_{\times\,qt_6\;(QT6)}$$

**This is the 42-driver linearity the code exploits**: 6 + 6 + 6 + 6 + 6 + 6
+ 6 = 42 columns. Now you can see *why* each optional input feeds what it
feeds: $\Gamma_h V_0 E_6$ has no 13/23 rows of its own — the transverse
shears live in the $\Gamma_{l\alpha}V_0$ and $\Gamma_h V_{1\alpha}$ terms,
which are **multiplied by the gradients**. Zero gradients ⇒ zero σ13/σ23.

### 5. Stress, and the two physical enforcements

$$\sigma(z) = C(z)\,\Gamma(z)$$

with $C(z)$ jumping at every ply interface — which is why stress is
double-valued there and the Gauss stations sit inside the plies. Then:

**Q-consistency** (shear-force rescale, per station):

$$\sigma_{13}(z)\;\leftarrow\;\sigma_{13}(z)\cdot
\frac{Q_1}{\int\sigma_{13}\,dz},\qquad
\sigma_{23}(z)\;\leftarrow\;\sigma_{23}(z)\cdot
\frac{Q_2}{\int\sigma_{23}\,dz}$$

so the profile *shape* comes from the gradients and the *carried force* is
exactly the FF's.

**σ33 by the momentum route** (with `--hist`, i.e. under inertia):

$$\sigma_{33}(z) = \int_{-fh}^{z}\big(\rho\,\ddot u_3
- \sigma_{13,1} - \sigma_{23,2}\big)\,dz'$$

integrated from the traction-free bottom face; the top face then closes at
$\sigma_{33} = -q$ automatically (global equilibrium). Without `--hist` the
constitutive $C(z)\Gamma(z)$ row is kept, with the $qt_6$ ladder carrying the
face condition.

### 6. Displacement recovery

$$U_\alpha(z) = u_\alpha - z\,u_{3,\alpha} + N(z)\,w^{nodes}_\alpha,
\qquad U_3(z) = u_3 + N(z)\,w^{nodes}_3$$

$N(z)$ the quartic Lagrange functions — at an SG node they collapse to the
solved dof itself, between nodes they interpolate quartically. The warping
$w_3$ IS the through-thickness compression a bare shell node cannot show
(1.24 % of the ex5 top deflection).

### 7. What the code actually computes

Because steps 4–6 are linear in the 42 drivers, both files precompute the
operators once per depth from **42 unit evaluations**:

$$T_E(z), T_S(z)\in\mathbb R^{6\times42},\quad T_W(z)\in\mathbb R^{3\times42}$$
$$\varepsilon^{3D}=T_E\,D,\qquad \sigma^{3D}=T_S\,D,\qquad w=T_W\,D,
\qquad D=[E_6,\,E_{6,1},\,E_{6,2},\,E_{6,11},\,E_{6,12},\,E_{6,22},\,qt_6]$$

and the whole plate is one `jax.vmap` of that contraction over the in-plane
stations — 400 or 1600 matrix-vector products, nothing re-solved.

## Output conventions (all three files + the VTK)

- columns: `x y z` + 6 components (order 11 22 33 12 13 23) or `U1 U2 U3`
- z is in the **SG frame**: 0 = the shell reference surface
- depths: the **Gauss points of each 5-noded SG element** (5 per ply, all
  strictly inside their ply — a material interface never appears in a file);
  `--lattice nodes` / `LATTICE = "nodes"` switches to the SG nodes, where
  the fluctuation dofs are read exactly but interface stress is single-sided
- stress/strain in the written `.SM`/`.EM`/VTK are in the **PLY MATERIAL
  frame** of each depth (failure criteria live there; matches Abaqus
  `*ORIENTATION` ply-local output directly). The whole chain COMPUTES in
  the plate/laminate frame — it must: the FD gradients, the operators and
  the Q-rescale integrate σ13/σ23 across plies — and rotates per depth
  only at write time (`material_rotations`); σ33/ε33 are invariant, the
  displacement `.U` stays global. The point API mirrors this:
  `msgrm_strain_at_depth` returns material frame by default,
  `frame="plate"` for anything that integrates/compares across plies.
  SG Voigt internally (11, 22, 33, 23, 13, 12), reordered on write
- single-station `.vtk` is a 1 × 1 × nz line; the field `.vtk` is the full
  structured grid the ParaView scripts consume
