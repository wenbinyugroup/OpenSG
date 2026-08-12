# Rule — the shell SEGMENT/RING element is 4-node only, and MITC3 is not what blocks it

**Scope:** `opensg_shell.sg_assembly.assemble_segment_indep`, `assemble_constraint`,
`build_C_Psi_segment6`, their `quad_ops_indep[_batch]` operator family, and the two
routes built on them — `sg_homo.ring_indep` (boundary rings) and
`sg_homo.segment_timo_from_3dyaml` (the tapered / aperiodic segment).

## The rule

These functions take **4-node quads only**. `require_quad_mesh` enforces it. Unlike
`shell_sg3d` — which does true mixed dispatch, `(ne3,18,18)` tri3 blocks and
`(ne4,24,24)` quad blocks scattered into one system — the segment family has no
triangle at all, and passing one is a hard error rather than a silent path.

**Do not "wire `tri_scheme_for` through" to fix this.** The transverse-shear scheme is
not what is missing.

## Why: the aliasing objection is real for quads and does NOT transfer to MITC3

`assemble_segment_indep` documents `shear="full"` as the production default because
"with the independent `omega_3`, Dvorkin-Bathe tying aliases the algebraic drilling
shear". That is true, but the mechanism is narrower than the sentence suggests.

Written on **one** frame `(a1, a2, n)` the operator rows are

```
2g13 = D1_a (n . w_a) + N_a (a2 . om_a)
2g23 = D2_a (n . w_a) - N_a (a1 . om_a)
DR   = N_a (n . om_a) - 1/2 [ D1_a (a2 . w_a) - D2_a (a1 . w_a) ]
```

The shear rows are blind to the drilling `n.om_a`; `DR` is blind to the tangential
rotation. They are exactly orthogonal projections of the same nodal `om`. The `om3`
column they appear to share is a **basis artifact** — `om3` is the `b3` *global*
component, a mixture of tangential and normal rotation — not a physical coupling.

A tied row is a linear combination of rows sampled at **different parent points**, and
`_surf_frame_batch` rebuilds `(a1, a2, n)` at each one. If the frame moves across the
element, the tied rotation coefficient `c_a = dBG/d(om_a)` picks up a component along
`n(gauss)` and drilling leaks into the shear. Measured `max |c_a.n| / |c_a|` at a Gauss
point, generic 3-D pose:

| element | full | mitc4_g23 | mitc4_wonly | mitc4_both |
|---|---|---|---|---|
| planar parallelogram | 8.5e-17 | 8.5e-17 | 8.5e-17 | 2.6e-16 |
| planar trapezoid (non-parallelogram) | 8.6e-17 | 1.8e-16 | 8.6e-17 | 3.6e-16 |
| warped, node 4 lifted 0.15 | 8.0e-17 | **4.4e-02** | 8.0e-17 | **4.4e-02** |
| warped, node 4 lifted 0.50 | 6.8e-17 | **1.7e-01** | 6.8e-17 | **1.7e-01** |

So the driver is quad **warp**, not the independence of `om3`; a planar quad of any
shape is clean, and `mitc4_wonly` is clean by construction (it keeps the rotation
columns at their full-integration values).

**On a triangle the mechanism cannot fire.** A tri3 is affine: `g_r = X2-X1` and
`g_s = X3-X1` are constant, so `tri_frame_batch` returns the *same* `(a1, a2, n)` and
the same metric at the three edge-midpoint tying points as at the Gauss point.
Measured frame drift `max|cos(tie) - cos(gauss)| = 0.000e+00` on all five reference
triangles (equilateral, right-isoceles, obtuse 143 deg, sliver ar 250, skew scaled).
Every MITC3 row is therefore a combination of rows whose rotation blocks lie in
`span{a1.om_a, a2.om_a}`, and `n.om_a` never enters:

| element | full | mitc3 |
|---|---|---|
| equilateral | 4.0e-17 | 1.0e-16 |
| right-isoceles | 1.0e-16 | 9.9e-17 |
| obtuse 143 deg | 4.2e-17 | 7.9e-17 |
| sliver ar 250 | 6.4e-17 | 1.4e-16 |
| skew scaled | 9.2e-17 | 2.9e-16 |

The companion objection — that tying an *algebraic* row de-penalizes the director
(hourglass) — also does not fire: the tri3 18x18 has exactly 9 zero eigenvalues with the
drilling untreated (6 rigid + 3 drilling), **identically under `full` and `mitc3`**, on
all five shapes, and the element-wise drilling multiplier removes one of them on every
shape under both schemes. MITC3 adds no spurious mode.

## Why, then: the operators do not exist

The segment assembler wants **ten** arrays per element. The triangle has neither half of
what is missing:

| block | what it is | quad | tri3 |
|---|---|---|---|
| `BDh`, `BGh`, `DRh` | Gamma_h (fluctuation) | `quad_ops_indep_batch` | `sg_homo.solid_fluct_ops_tri_batch` |
| `BDe`, `BGe`, `DRe` | Gamma_e, the **beam** macro `eb = [g11 k1 k2 k3]` (built from `Rn1`, `Rn2`, `swept`, `x2`, `x3`) | `quad_ops_indep_batch` | **absent** — `solid_macro_ops_tri_batch` is the 3-D SOLID 6-column drive, a different macro |
| `BDl`, `BGl`, `DRl` | Gamma_l, the `w'` chain-rule columns | `quad_ops_indep_batch` | **absent** |

One further mismatch:

* `assemble_constraint` and `build_C_Psi_segment6` need the same mixed dispatch, and
  `sg_mesh.extract` stores `seg_cells` as one rectangular array.

(The `DR` mismatch is gone: `solid_fluct_ops_[tri_]batch` no longer divides the drilling
row by `C33`, so both families now carry the same **undivided multiplied-through**,
frame-covariant residual `DR = om.n - 1/2 [ (w.a2),1 - (w.a1),2 ]`, and both enforce it
with an element-wise Lagrange multiplier — `shell_sg3d` included, which used to apply a
`drill_pen*A11*int DR^2` penalty to the divided row. The counting and the frame
objectivity of the new element are gated in
`src/opensg_shell/tests/test_tri3_mitc3.py`.)

Writing `tri_ops_indep_batch` is a **new element family**, not a wiring job.

## If a tri3 segment element is ever written, project all three blocks

The tied path today applies the assumed field to `Gamma_h` **alone**: `BGe` and `BGl`
stay at their Gauss values while `BGt` replaces `BGh`, so the tied energy is not the
energy of one assumed field. `BGe` genuinely varies inside an element (its columns carry
the interpolated `x2`, `x3`), measured in-element spread `(max-min)/max` of 2.000 on both
the prismatic and the BAR-URC segment meshes. A consistent element must apply the same
projection to `Gamma_e` and `Gamma_l` — the sibling of
`Rules/gamma_e_gamma_h_consistency.md`.

## Acceptance: the prismatic identity measures scheme CONSISTENCY, not the transfer

`examples/OpenSG_shell/5_taper_segment_aperiodic_boundaries/run_taper_segment.py`
reports a worst diagonal error of 0.00027 % (iso) / 0.00074 % (+-45). That residual is
**entirely** the ring/segment shear-scheme mismatch — `ring_indep` defaults to
`mitc4_g23`, the segment to `full` — not an error in the boundary V0/V1 Dirichlet
transfer:

| ring shear | segment shear | worst diagonal, iso | worst diagonal, +-45 |
|---|---|---|---|
| mitc4_g23 | full (shipped) | 0.000269 % | 0.000743 % |
| mitc4_g23 | mitc4_g23 | 0.000000 % | 0.000000 % |
| full | full | 0.000000 % | 0.000000 % |
| full | mitc4_g23 | -0.000269 % | -0.000743 % |

The sign flips with the mismatch. So the boundary transfer is exact to the printed
precision, and any future element change must keep the ring and the segment on the
**same** scheme to preserve the identity — that, not the absolute 0.0003 %, is the
invariant worth gating.
