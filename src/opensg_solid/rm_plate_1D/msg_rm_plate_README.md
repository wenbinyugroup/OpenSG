# msg_rm_plate.py — construction notes, conventions, and validation history

Companion document to `opensg_jax/fe_jax/msg_rm_plate.py`. The code keeps
only short pointers; every long rationale, historical note and validation
number lives here. Equation numbers = Yu, Hodges & Volovoi, *Computers &
Structures* 81 (2003) 439–454 (the same construction is IJSS 39:5185
Eqs. 49–55 and CMAME 191:5087 Eqs. 70–77).

## 0. Workflow — who calls what, and in which order

```
layup_db.yaml ──> rm_homo.load_layup_db / homogenize_layup_db     [thin wrapper]
      │                        │
      │     direct arrays      ▼
      └──> rm_plate_msg(thick, angles_deg, mat_names, material_db,
                        n_per_layer=1, elem_order=4, fraction)
                               │   one jitted kernel per (n_ply, n_per_layer,
                               │   elem_order) bucket; KKT factorized once;
                               │   the whole warping ladder solved per call
                               ▼
           r = { A6, G_msg, ABDG, X, H,
                 V0, V11/V12 (+bar/barD), V21/22/23 (+t), V1L/V2L per face,
                 node_x, elem_layer, C_layers, elem_order, angles, c1, c2 }
                               │
        drivers (the caller's job -- r never leaves the laminate):
          E6  = inv(A6) @ F6            F6 = [N11 N22 N12 M11 M22 M12]
          dE1, dE2                      FD / closed form / harmonic
          dE11, dE12, dE22              ONLY when second order is wanted (V2: 1)
          qt6, qb6                      the APPLIED face-load ladders
                               ▼
        msgrm_strain_at_depth(r, z, E6, dE1, dE2, [dE11, dE12, dE22],
                              qt6=, qb6=, frame=)
                                            ->  (Gam6, Sig6, ply_angle)
              frame="material" (DEFAULT): Gam/Sig in the ply axes of the
              layer containing z (Sig rotation_6x6(-ang) @ sig, Gam
              rotation_6x6(ang).T @ gam -- failure criteria live there)
              frame="plate": laminate frame -- REQUIRED wherever the result
              is integrated/differentiated ACROSS plies (build_ops, the
              Q-rescale, momentum integrals, global-frame comparisons)
        msgrm_warping_at_depth(r, z, ...)   ->  w (3,)  ALWAYS plate frame
              (it composes with the global plate displacement)
                               │
                               ▼
        the caller composes    U_a = u_a^2d - x3 w,a + w_a ,  U3 = w + w3
```

Rules the workflow encodes (details in the numbered sections below):

1. **Pass the RM measures R, never a pre-converted ε** — the detilted
   columns do the Eq.-50 conversion internally (sec. 3).
2. **First vs second order is the caller's driver choice**: dE1/dE2 only
   → Eq. 63; add dE11/dE12/dE22 → Eq. 66 with the row split (sec. 4). The
   pipeline exposes this as the layup_db `V2:` flag.
3. **qt6/qb6 are input, not closure** — the applied face pressures and
   their gradients; with them σ33 is constitutive with machine-exact faces
   (sec. 5).
4. **Homogenize per call, never cache** — the SG is a line; ms per laminate
   (batch with `rm_plate_msg_batch` for many).

Minimal example (one laminate, one station, second order on):

```python
from opensg_jax.fe_jax.msg_rm_plate import rm_plate_msg, msgrm_strain_at_depth
import numpy as np

r  = rm_plate_msg(thick, angles, mats, matdb, fraction=0.5)
S6 = np.linalg.inv(r["A6"])
E6 = S6 @ F6                                   # FF -> plate strains
gam, sig, ang = msgrm_strain_at_depth(
    r, z, E6, dE1, dE2, dE11, dE12, dE22,      # drivers, caller-supplied
    qt6=np.array([q, 0, 0, -p*p*q, 0, 0]))     # applied top-face ladder
```

Consumers in this repo:

| caller | role |
|---|---|
| `opensg_jax/fe_jax/rm_homo.py` | layup_db.yaml → `rm_plate_msg` + the homo `.out` |
| `opensg_jax/fe_jax/rm_dehom.py` | `load_sg` (re-runs the homo), `z_stations` (Gauss/nodes), `build_ops` (the 42-driver operators), file/VTK writers |
| `examples/opensg-rm_dynamic/ex5/rm_dehom_dat.py` | Abaqus field `.dat` → FD drivers → vmapped recovery (`V2:` flag) |
| `examples/opensg-rm_dynamic/ex5/compare_stress_top.py` | patch-fit drivers → single-station history |
| `examples/garg/pagano_bench.py`, `examples/yu2003/yu_bench.py` | statics / harmonic drivers → Pagano validation |
| `examples/opensg_rm_static/validate_v2.py` | THE recovery gate (run after any change here) |
| `tests/test_msg_rm_plate.py` | 11 unit tests (A6 vs classical ABD, gauges, closures) |

## 1. The construction ladder

| stage | what is solved | equations |
|---|---|---|
| constraints | both warping solves are the constrained minimization Eq. (34), solved DIRECTLY as a Lagrange-multiplier (KKT) saddle-point system — the multiplier of Eq. (35) and the gauge ⟨wᵢ⟩ = 0 of Eqs. (31)/(39) are exact by construction, no penalty, no post-hoc projection. One factorization serves every order (the 18 strain unit cases + the load columns reuse the same LU). | (31)–(39) |
| order 0 | V0 warping → A6 (classical ABD; must reproduce `compute_ABD_matrix` exactly). The multiplier vanishes identically here (kernelᵀ D_hε = 0, asserted in the test suite). The constraint row is still load-bearing: it selects the unique ⟨w⟩ = 0 gauge that propagates into G. | (39)–(40) |
| order 1 | V11/V12 columns driven by D̄ₐ = (D_hlₐ − D_hlₐᵀ)V0 − D_lₐε | (43)–(45) |
| order 2 (energy) | gradient-energy blocks B, C, D → H (12×12 over [ε,1; ε,2]) | (46)–(47) |
| RM projection | least-squares minimization of the residual U* over the shear compliance X = G⁻¹ (3 unknowns) + the relaxed constants c₁, c₂ (24; in-plane rows only — the w₃-shift columns are identically inert for monoclinic laminates) → G_msg | (56)–(61) |
| order 2 (recovery) | V21/V22/V23 (two variants, §4) + the V2L load quintets | (64)–(66) |

**Orders and exactness of the discretization.** With C constant per ply the
exact warping is piecewise polynomial of degree V0: 2, V1: 3, V2: 4 (the
paper's "piecewise, fourth-order polynomials" remark), so `elem_order = 4`
(the paper's own 5-noded choice) represents the whole ladder exactly:
measured σ33 closure 8.4e−5 with machine-zero face tractions vs 1.5e−2 for
cubic subdivision, at ~3× fewer dofs. A6 and G are identical for any
`elem_order ≥ 3`.

**Performance.** One traced function per (n_ply, n_per_layer, elem_order)
bucket, vmapped over elements inside (the beam-solid pattern) and over
laminates by `rm_plate_msg_batch`. Jit ≈ 1.7 s per bucket, then a few ms per
laminate batched (measured 12× the NumPy loop on the 10-wall st15 station,
83× at blade scale). The KKT solve keeps all load-case columns as one
matrix (level-3 BLAS); measured 15–25% ahead of two independent multi-RHS
solves.

## 2. The U* least squares

The system is "78 equations, 27 unknowns" (text after Eq. 57), rank 26:
truncated-SVD minimum-norm solve with `_LS_RCOND = 1e-12` (the null singular
value sits ~1e−16 of the largest; anything in 1e−14..1e−10 works). Column
equilibration is essential: the X-columns scale like A² (~1e13) and the
c-columns like S (~1e9) — a ~1e4 spread that pushes cond(A) to ~5e19;
unit-norm scaling brings it to ~8e15, moving only the redundant constants.

**Symmetrization rule (Eq. 60).** The DIAGONAL blocks carry 2 L₁ᵀD̄₁ as the
symmetric representative c₁ᵀS₁ + S₁ᵀc₁. Do not "correct" this to 2·c₁ᵀS₁:
the raw product is non-symmetric and, although the two agree inside the
quadratic form U* actually is, the Frobenius objective sees the
antisymmetric part — feeding the raw form lets the 24 constants chase a
residual with no energetic meaning (measured G error up to 14%, Ustar_rel
inflated up to 10×). The OFF-diagonal Ĉ is deliberately NOT symmetrized: it
sits in the bilinear 2R,1ᵀĈR,2 between two different vectors, where the
asymmetry is real.

## 3. The tilt / detilt rules (the heart of the recovery conventions)

The warping gauge is ONLY ⟨wᵢ⟩ = 0 — no first-moment constraint,
deliberately: the x₃-linear (tilt) content of the in-plane V1bar columns IS
the transverse shear deformation (Yu constrains the triad normal to the
deformed surface, so shear has exactly one home, the warping).

Yu states the recovery in CLASSICAL measures: a Reissner-like solver's
output R is converted by ε = R − Dₐγ,ₐ (Eq. 50) before driving the warping.
This code's public API takes the RM measures **R directly**; using DETILTED
columns in the Γₗ (value) terms performs that conversion implicitly —
numerically equivalent to the literal Eq.-50 route in the asymptotic regime
(0.043% vs 0.044% at S = 64 on Pagano caseA) and far better-behaved thick
(26.8% vs 2373% at S = 4, where the ε-substitution itself is no longer
asymptotic). **Callers must therefore pass R, never a pre-converted ε**
(that would double-correct).

Per-use rules, each validated by a controlled caseA sweep:

| use | columns | why / numbers |
|---|---|---|
| Γₕ (through-thickness derivative) | RAW V1bar | the tilt delivers the mean shear into σₐ₃ (0.18% at S = 50); detilted here would lose it |
| Γₗ (value, second-order in-plane rows) | DETILTED V1barD | raw tilt next to z·φ double-counts shear: caseA U1 85%/3.4% at S = 10/50 raw vs 2.1%/0.07%; in-plane second-order strain −76% raw vs ±2% detilted |
| displacement composition | MEAN-ZERO V11/V12 + the KIRCHHOFF z-term | Uₐ = uₐ − x₃w,ₐ + wₐ, U₃ = w + w₃. (a) the raw first-order tilt is exactly the mean-shear content the Kirchhoff term lacks (U1 1.41%/0.016% at S = 10/64; composing with x₃φ double-counts: 85%/2.1%); (b) u²ᵈ is DEFINED as the thickness average, so the added warping must average to zero — the relaxed constants would shift Uₐ bodily (a constant ~9% offset on the angle-ply U2) |
| w₃ | keeps its tilt | U₃ has no z·φ partner |

## 4. The V2 second-order recovery: two variants and the row split

Putting V = V0ε + V̄₁,ₐε,ₐ + V2 back into the energy and collecting the
V2-linear terms, everything pairing V2 with ε or ε,ₐ cancels through the
lower-order Euler–Lagrange equations (+ the gauge), leaving

    2Π₂* = V2ᵀ E V2 + 2V2ᵀ (D̄₂₁ ε,11 + D̄₂₂ ε,12 + D̄₂₃ ε,22)

with the Eq. (43)–(44) drivers one rung up (V0 → V̄₁ₐ in the skew part,
D_lₐε → D_lₐl_b V0 in the direct part; the mixed driver gets both orders):

    D̄₂₁ = (D_hl1 − D_hl1ᵀ) V̄₁₁ − D_l1l1 V0
    D̄₂₂ = (D_hl1 − D_hl1ᵀ) V̄₁₂ + (D_hl2 − D_hl2ᵀ) V̄₁₁ − (D_l1l2 + D_l1l2ᵀ) V0
    D̄₂₃ = (D_hl2 − D_hl2ᵀ) V̄₁₂ − D_l2l2 V0

V2's own energy content is O((h/l)⁴), below the model's resolution, so A6
and G are untouched; V2 exists purely to recover the through-thickness
fields at their leading order.

**Which V̄₁ enters the source — the 2026-08-04 falsification sweep** (gate:
`examples/opensg_rm_static/validate_v2.py`):

- raw V11/V12 (the literal E-L source, and Yu's "original V1" wording at his
  Eq. 64) — FALSIFIED: breaks the face-traction closure by the ±0.73q₀ tilt
  mode everywhere (single-ply σ33 2.55% → 37.7%);
- relaxed-untilted V11bar/V12bar — numerically identical to raw
  (E·kernel = 0): the TILT, not the constants, carries the cancellation.

The rule that survives: **the V2 source and the recovery's Γₗ columns must
match**, or the V2 natural BC fails to cancel the Γₗ face traction. The two
consistent (source, Γₗ) pairs split cleanly by output row family:

| pair | in-plane rows (11, 22, 12) | through-thickness rows (33, 23, 13) |
|---|---|---|
| detilted (V1barD, V1barD) | good | cross-ply interior σ33 O(1) wrong ([0/90/0] 19%, Yu case3 22%, sandwich ~470%) |
| tilted (V1bar, V1bar) | −76%-family z·φ double-count | excellent for every stack (caseA S50 0.05%, caseC S50 0.11%, yu1 1.14%) |

So the recovery **row-splits**: both V2 variants are solved (V21/22/23
detilted-source, V21t/22t/23t tilted-source — one shared LU, 18 extra RHS
columns) and the second-order stress takes rows (11, 22, 12) from the
detilted chain and rows (33, 23, 13) from the tilted chain — the same
convention-per-use precedent as raw-in-Γₕ / detilted-in-Γₗ. Gate results
after the split: σ33 with V2 beats first order for EVERY stack
(homogeneous 0.5–2.6%, [15/−15] 1.1%, [0/90/0] 1.2–5.6%, Yu case3
0.8–4.3%, sandwich 0.1–16.7%), faces machine-exact, `tests/` 11/11, ex5
first-order results bit-identical.

## 5. The load columns (V1L, V2L)

History: implemented+validated at d1b52ac, removed at b9d2658, reinstated
for the full-field recovery — σ33 DIRECT from the constitutive law, and the
load-driven ε33 content that σ22 and sandwich in-plane stresses need (this
is what VAPAS keeps).

Load pattern: unit normal traction on a FACE (σ33 = +1 there). Virtual work
⟨δw·τ⟩ on the face → a single entry at that face node's w₃ dof (shape
functions are 1 at their own node). TWO columns, top and bottom, so
split-face loads (Yu §6.1: s₃ = b₃ = p₀/2) compose linearly.

- First order (Eq.-45 load column, per unit local face pressure):
  E V1L + L = Hψ(ψᵀL) — more RHS on the same LU.
- Second order (the V2L quintets; the D̄₂ₖ collection with V1L as the lower
  rung): q,ₐ drivers (D_hlₐ − D_hlₐᵀ)V1L; q,ₐᵦ drivers −D_lₐl_b V1L.
- Signs: `solve_constrained` solves E V = −rhs with the load entering as
  external work; the face checks (σ33(top) = +q_t, σ33(bottom) = +q_b,
  machine-exact) fix both signs empirically.

In the recovery, qt6/qb6 = [q, q,1, q,2, q,11, q,12, q,22] of the LOCAL
face pressure per face. With them σ33 comes directly from the constitutive
law with machine-exact face values; left at None, every resultant-driven
chain is unchanged.

## 6. Validation

`tests/test_msg_rm_plate.py` (11 tests). The recovery conventions are gated
by `examples/opensg_rm_static/validate_v2.py` (the four Pagano statics +
homogeneous/mild-contrast/near-cross-ply probes) and the ex5 dynamic
benchmark (`examples/opensg-rm_dynamic/ex5/`, S8R vs C3D20). The
dimension-suffix notation of the internals (`_ns`, `_nn`, `_ss`, …) is
documented in the module docstring.
