# Bare-core homogenization — mesh-convergence study

Mesh convergence of the effective **Reissner–Mindlin (8×8, `refined: 1`) plate
law** of the bare sandwich core, at two relative densities, **ρ = 0.05** and
**ρ = 0.3**. The purpose is to fix the mesh at which the homogenized law — and
in particular the transverse-shear stiffness **G** — is mesh-independent, so
that the Case-1 Abaqus tiling is driven by a converged constitutive law rather
than by a locked one.

Core material is isotropic aluminium (E = 70 GPa, ν = 0.3, G = 26.923 GPa),
single material, periodic SG, `msg: solid`, `n_model: 2`, `refined: 1`.

## Files

| file | what |
|---|---|
| `conv_tables.txt` | the convergence tables (values + % change vs the finest tet10), the solver-certificate note, the assumed-order error table, and the tet4 locking table |
| `convergence_plots/conv_rho0.05.png` | A11 / D11 / G11 vs element count, normalized to the converged tet10 value, ρ = 0.05. Upper panel = tet4 scale; lower panel re-plots the tet10 pair alone, because on the tet4 scale (up to 1.77) the tet10 points sit on y = 1 and show nothing |
| `convergence_plots/conv_rho0.3.png` | the same for ρ = 0.3 |
| `convergence_plots/locking_tet4_vs_tet10.png` | combined tet4-vs-tet10 deviation from the converged law, log–log |
| `make_conv_study.py` | regenerates every table and plot above from the `.out` files |
| `SP_solid_rho<ρ>_n<knob>[_quad].out` | the effective-law files themselves |

The `_n<knob>` decimals are a gmsh sizing knob, **not** a mesh measure — the
knob is not monotone in mesh size and three different ρ = 0.3 knobs
(`n1.27429`, `n1.69905`, `n2.54857`) produce the *identical* 172,546-element
mesh. Every count in this study is read from the gmsh `.msh` header by
`make_conv_study.py`; nothing is inferred from a filename. `_quad` = tet10
(p-refinement of the same tet4 mesh, so element count and geometry are
identical and only the interpolation order changes); no suffix = tet4.
Throughout, **dofs = 3 × nodes** (three fluctuation-function components per
node).

## The mesh ladder

### ρ = 0.05

| mesh | type | elements | nodes | dofs |
|---|---|---:|---:|---:|
| `n0.424762` | tet4 | 59,939 | 19,692 | 59,076 |
| `n0.424762_quad` | tet10 | 59,939 | 119,002 | 357,006 |
| `n0.849524` | tet4 | 118,124 | 37,856 | 113,568 |
| **`n0.849524_quad`** | **tet10** | **118,124** | **230,477** | **691,431** |
| `n2.12381` | tet4 | 435,366 | 120,112 | 360,336 |
| `n2.83175` | tet4 | 893,397 | 221,877 | 665,631 |

### ρ = 0.3

| mesh | type | elements | nodes | dofs |
|---|---|---:|---:|---:|
| `n2.54857` | tet4 | 172,546 | 40,245 | 120,735 |
| `n2.54857_quad` | tet10 | 172,546 | 277,346 | 832,038 |
| `n5.09714` | tet4 | 458,133 | 96,306 | 288,918 |
| **`n5.09714_quad`** | **tet10** | **458,133** | **695,137** | **2,085,411** |
| `n12.7429` | tet4 | 2,191,358 | 408,404 | 1,225,212 |

Bold = the converged reference of that density.

**Excluded from the ladder.** `SP_solid_rho0.3_n2.54857_quad_amg`,
`_amgtight` and `_directref` are solver-certificate reruns of the *same*
172,546-element tet10 mesh with a different KSP/PC — they are not extra mesh
points and are reported separately (see below). Meshes present in the folder
but never homogenized (`rho0.05_n3.53968`, `rho0.3_n16.9905`, `n25.4857`,
`n12.7429_quad`) are not data points and do not appear.

## Verdict

### ρ = 0.3 — CONVERGED

Going from the 172,546-element tet10 to the 458,133-element tet10 (2.65× the
elements, 2.5× the dofs) moves **every** diagonal term of the law by at most
**0.0161 %**:

| | A11 | A66 | D11 | D66 | G11 | G22 |
|---|---:|---:|---:|---:|---:|---:|
| tet10 172,546 vs tet10 458,133 | +0.0138 % | +0.0076 % | +0.0161 % | +0.0079 % | **+0.0093 %** | **+0.0095 %** |

These are two-point differences, not measured errors — see the ρ = 0.05
discussion below for what that distinction costs. Here it costs nothing: even
under the most pessimistic assumed order (*p* = 1) the implied error of the
458,133-element tet10 is ≤ 0.042 % on every term (G11 0.024 %), so the verdict
holds whatever the true order turns out to be.

**The converged choice for G at ρ = 0.3 is `SP_solid_rho0.3_n5.09714_quad`** —
tet10, 458,133 elements, 695,137 nodes, 2,085,411 dofs. This is the law the
Case-1 Abaqus tiling should use. (`n2.54857_quad` already reproduces G to
0.009 % at one third of the cost — 360 s vs 1074 s — so it is a legitimate
cheaper stand-in, but the numbers quoted below are the finest ones.)

**Converged 8×8 law, ρ = 0.3** (`SP_solid_rho0.3_n5.09714_quad`; A in N/m,
D in N·m, G in N/m):

```
A11 = 6.3343147E+09      D11 = 4.3564861E+08      G11 = 3.1929382E+09
A22 = 6.3343140E+09      D22 = 4.3564878E+08      G22 = 3.1929371E+09
A66 = 4.4503654E+09      D66 = 3.0346802E+08
A12 = 2.6519996E+09      D12 = 1.8966977E+08
```

The law is orthotropic and very nearly square-symmetric (A11/A22 and G11/G22
agree to 7 significant figures); every membrane–bending and in-plane–shear
coupling is at or below 1e-6 of the corresponding diagonal, i.e. numerically
zero.

### ρ = 0.05 — CONVERGED to ≈0.3–0.7 % on G, ≈0.06–0.24 % on A and D

| | A11 | A66 | D11 | D66 | G11 | G22 |
|---|---:|---:|---:|---:|---:|---:|
| tet10 59,939 vs tet10 118,124 (difference) | +0.0360 % | +0.0539 % | +0.0498 % | +0.0598 % | **+0.1744 %** | **+0.1738 %** |
| implied error of the finer, p = 2 | 0.063 % | 0.094 % | 0.087 % | 0.105 % | **0.305 %** | **0.304 %** |
| implied error of the finer, p = 1 | 0.142 % | 0.212 % | 0.196 % | 0.236 % | **0.687 %** | **0.685 %** |

Converged reference `SP_solid_rho0.05_n0.849524_quad` (tet10, 118,124
elements, 230,477 nodes, 691,431 dofs):

```
A11 = 5.0145741E+08      D11 = 2.7528393E+07      G11 = 1.7417036E+08
A22 = 5.0145737E+08      D22 = 2.7528370E+07      G22 = 1.7417095E+08
A66 = 4.1993228E+08      D66 = 2.5553780E+07
A12 = 3.3315644E+08      D12 = 2.4611401E+07
```

This verdict is weaker than the ρ = 0.3 one and should be read as such: the
tet10 pair spans only 1.97× in element count (against 2.65× at ρ = 0.3), and G
is still moving at the 0.17 % level where at ρ = 0.3 it had already settled to
0.009 %. **If G at ρ = 0.05 is ever needed to better than 0.1 %, run
`n2.12381_quad` (435,366 tet10) and re-check.** For the present purpose even
the pessimistic 0.69 % is below any other error in the chain.

Two things this number is *not*, and they matter if G at ρ = 0.05 is ever
quoted quantitatively:

* **0.1744 % is a difference between two meshes, not the error of the finer
  one.** With only two tet10 points per density there is no third point, so
  the observed order of convergence cannot be measured and no genuine
  Richardson extrapolation exists. Assuming an order *p*, the error of the
  finer mesh is `diff / (r_h^p − 1)` with `r_h = (N_fine/N_coarse)^(1/3) =
  1.2537` — which is 0.305 % at *p* = 2 and 0.687 % at *p* = 1, i.e. 2–4× the
  raw difference. **Quote 0.3–0.7 % for G at ρ = 0.05, not 0.17 %.** At
  ρ = 0.3 (`r_h = 1.3847`) the same construction gives ≤ 0.042 % on every term
  even at *p* = 1, so that verdict is safe under any assumed order. The full
  table is in `conv_tables.txt`.
* **G is not slower in *rate* than A11/D11 — it is larger in *magnitude*.** No
  rate is measurable on the tet10 legs at all. On the tet4 legs, which do have
  3–4 points, the observed order over the finest interval is ≈2.0 for G at
  both densities, against 1.57 (A11) and 1.74 (D11) at ρ = 0.05 — G is if
  anything the *steeper* one. What makes G the last term to become usable at
  ρ = 0.05 is that it starts at +77 % where A11 starts at +15 %.

### tet4 locking — the headline result

tet4 converges to the law **from above** at both densities: the linear
tetrahedron is too stiff, and it is worst in exactly the term this study
cares about, the transverse shear G.

At a **matched element count** (identical mesh, tet4 vs its own tet10
p-refinement — a pure interpolation-order comparison with no geometry
difference):

| ρ | mesh | elements | A11 | A66 | D11 | D66 | **G11** | G22 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | `n0.424762` | 59,939 | +15.22 % | +26.83 % | +23.82 % | +32.93 % | **+77.00 %** | +77.02 % |
| 0.05 | `n0.849524` | 118,124 | +12.34 % | +24.09 % | +19.04 % | +28.82 % | **+65.18 %** | +65.21 % |
| 0.3 | `n2.54857` | 172,546 | +3.432 % | +1.297 % | +4.299 % | +1.516 % | **+2.566 %** | +2.602 % |
| 0.3 | `n5.09714` | 458,133 | +1.656 % | +0.660 % | +2.089 % | +0.759 % | **+1.343 %** | +1.340 % |

**At ρ = 0.05, tet4 over-predicts the transverse shear stiffness by 65 % on
the same mesh where tet10 is converged.** At ρ = 0.3 the same comparison gives
+2.6 %. Locking scales with ligament slenderness, so the low-density core is
where it is fatal.

Refining tet4 does not rescue it economically. Comparing at roughly **matched
dofs** instead:

* ρ = 0.05: tet4 `n2.83175` (893,397 elements, 665,631 dofs) against tet10
  `n0.849524_quad` (118,124 elements, 691,431 dofs) — essentially the same
  problem size — still leaves **G11 +16.33 %** and A11 +3.16 %.
* ρ = 0.3: tet4 `n12.7429` (2,191,358 elements, 1,225,212 dofs), i.e. 4.8× the
  element count of the converged tet10 and still only 59 % of its dofs, is
  **G11 +0.445 %**, A11 +0.572 %, D11 +0.712 %.

Conclusion: **use tet10 for this core.** tet4 at any affordable mesh is not an
acceptable substitute at ρ = 0.05, and costs ~5× the elements to reach 0.5 %
at ρ = 0.3.

## Solver-certificate note

**Two** independent reruns of the ρ = 0.3, 172,546-element tet10 mesh with a
different linear solver, plus one archived copy of the reference run:

| rerun | max abs rel. diff, **diagonal** | max abs rel. diff, off-diagonal (full 8×8) | worst-moving entry |
|---|---:|---:|---|
| `_amg` (default tol) | 0.000e+00 % | 6.725e-02 % | `K[3,5]` = −8.30187E+01, 1.3e−08 of A11 |
| `_amgtight` (tightened tol) | 0.000e+00 % | 5.488e-04 % | `K[0,5]` = 1.91339E+02, 3.0e−08 of A11 |
| `_directref` | 0.000e+00 % | 0.000e+00 % | — *(see caveat)* |

⚠ **`_directref` is not an independent solve.** It is byte-for-byte the same
file as `SP_solid_rho0.3_n2.54857_quad.out` — same md5
(`1b3df55b2572440d9a2d3621719b7dcd`), same `Time taken: 360.5 sec` line, which
two separate solves could not produce. It is an archived copy of the reference
run, so its `0.000e+00` row is a self-comparison and carries no information.
The certificate rests on `_amg` and `_amgtight` only.

Every diagonal term is identical to all 8 printed digits under both real
reruns, and so are the two physically significant couplings **A12 and D12**
(max abs rel. diff exactly 0). What moves is only the numerically-zero
couplings — entries 7–8 orders of magnitude below the diagonal, zero by
symmetry for this orthotropic core, i.e. pure solver noise. Those hold **≈3.2
significant digits at the default tolerance and ≈5.3 when it is tightened**.
See the caveat below for why that is the expected structure and not a defect.

## Caveat — the CG tolerance affects diagonal and off-diagonal terms differently

Verified by audit this session: **the error in the effective law is second
order in the CG tolerance for the diagonal / energy terms (A11, A66, D11, D66,
G11, G22 — the terms this study reports) but only first order for the
off-diagonal couplings.** The reason is that each right-hand-side column of
the fluctuation solve carries its own Krylov space: a diagonal entry is a
Rayleigh quotient of a column against *itself*, so the first-order error term
cancels and what survives is O(ε²); an off-diagonal entry pairs *two different*
columns whose residuals are not orthogonal to one another, leaving the O(ε)
term intact.

Practically, the diagonal terms are converged far below the digits printed
here, while the numerically-zero coupling terms hold **≈3.2 significant digits
at the default tolerance**, rising to **≈5.3 when it is tightened**. The
solver-certificate table above is a direct measurement of exactly this:
diagonals (and A12, D12) identical to all 8 printed digits, the zero couplings
wandering in the 4th significant figure at default tolerance and the 6th when
tightened.

That measurement also *tests* the first-order claim rather than merely
asserting it. Tightening moved the worst coupling from 6.725e-02 % to
5.488e-04 %, a factor of **122.5**. If the tolerance was tightened by 100×
(1e-8 → 1e-10) the implied exponent is `log(122.5)/log(100) = 1.04` — first
order, as claimed. Caveat on that inference: **none of the `.out` files record
the solver or the tolerance actually used**, so the 100× ratio is assumed from
the run recipe, not read from an artifact. If the tightening was 10× the
implied exponent would be 2.09 instead. Worth writing the KSP type and `rtol`
into the `.out` header so this stops being an inference.

**This does not affect the convergence verdict**, which rests entirely on
diagonal terms. It does mean that if a coupling term is ever needed
quantitatively (it is numerically zero for this core, so it is not needed
here), the tolerance must be tightened rather than assumed.

## How to regenerate

All commands run in this folder, on `msg.ecn.purdue.edu`:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate opensg_2_0
export PYTHONPATH=$HOME/OpenSG-2.0/src
```

**1 — tet4 → tet10 (p-refinement of an existing mesh).** Element count and
geometry are unchanged; only the interpolation order rises. See
`msh_p_refine.py`:

```python
from opensg import helper
helper.linear_msh_to_quad("SP_solid_rho0.3_n5.09714.msh")   # -> ..._quad.msh
```

**2 — mesh → SG yaml.**

```bash
opensg msh_to_yaml SP_solid_rho0.3_n5.09714_quad.msh --mat1 Al --n_model 2 --refined 1
```

**3 — homogenize.** The `.out` next to the yaml is the effective 8×8 law:

```bash
opensg_solid SP_solid_rho0.3_n5.09714_quad.yaml
```

The solver is chosen automatically from the dof count. **For the large meshes
force the algebraic-multigrid path explicitly** — the direct solver will
exhaust memory well before 2 M dofs:

```bash
opensg_solid SP_solid_rho0.3_n5.09714_quad.yaml --solver amg
```

Wall times actually observed (single node, `Time taken` line of each `.out`):
ρ = 0.3 tet10 172,546 el = 360 s direct / 1325 s amg; ρ = 0.3 tet10 458,133 el
= 1074 s; ρ = 0.3 tet4 2,191,358 el = 349 s; ρ = 0.05 tet10 118,124 el =
1730 s.

**4 — rebuild the tables and plots** from whatever `.out` files are present:

```bash
python make_conv_study.py
```

## Provenance

`SP_solid_rho0.3_n5.09714_quad.out`, `SP_solid_rho0.3_n12.7429.out` and
`SP_solid_rho0.05_n0.424762_quad.out` were recovered from the Google Drive
mirror and from `~/claude_tmp/conv_results` respectively, where they had been
left by earlier runs, and copied into this folder so the study is
self-contained. Each was checked against a second independent copy and agrees
digit-for-digit. Every other `.out` was already here; all of them were
verified identical to the Drive mirror before use. Re-verified independently:
all three md5s still match their source copies.

**The `.out` files are self-contained; the meshes are not.**
`SP_solid_rho0.05_n0.424762_quad.msh` (9.2 MB) — which supplies the element,
node and dof counts for one of the only *two* ρ = 0.05 tet10 points — lives in
`~/claude_tmp/mshdirect`, not in this folder, and `make_conv_study.py` reaches
it through `EXTRA_MSH`. If that scratch directory is cleaned the script now
aborts with a `FATAL:` message rather than silently dropping the row (it used
to `continue`, which would have quietly reduced ρ = 0.05 to a single tet10
point — and no convergence evidence — while leaving every caveat in this file
unchanged). The `.msh` files are too large to commit; only the `.out` files,
scripts, tables and plots are in git.
