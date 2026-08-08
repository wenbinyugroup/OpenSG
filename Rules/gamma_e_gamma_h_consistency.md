# Rule — Γe must be Γh evaluated on the macro embedding

**Scope:** every MSG operator pair in `opensg_shell` (`solid_props.py`,
`segment_indep.py`, `shell_sg3d.py`) and any new macro drive added later.

## The rule

$\Gamma_e$ is **not** an independent object. It is $\Gamma_h$ applied to the macro
displacement embedding, at 1:1 scale:

```
Gamma_e = Gamma_h  on   w_i -> Gbar_ij y_j ,  om_i -> 0
```

A row of $\Gamma_e$ may only contain what that same row of $\Gamma_h$ produces when the
fluctuation is replaced by the affine macro field. Write the contraction with the **tensor**
macro strain, i.e. off-diagonal (shear) components at engineering/2:

```
row = C_3i Eps_ij X_jalpha        with Eps_23 = (2*Gbar_23)/2
```

## Why (the bug this rule exists to prevent)

`solid_macro_ops_batch` originally stuffed the **engineering** shear (`2G23`) into both
off-diagonal slots of the macro strain matrix. The diagonal columns were then the correct
one-sided tensor evaluation while the pair-sum columns were **2x** too large — one row,
internally inconsistent between its own column groups. Energy is quadratic in the drive, so the
surviving shear residual came out exactly **4x**: relaxed `C44` was 4.00x its closed form on the
periodic cross cell.

The full-Bond (engineering pair-sum) row is *tensorially* correct but **inadmissible here**: the
half it adds is a fiber-tilt rotation that has the opposite sign on the two wall families, so it
would have to jump by `g` at every junction, while the nodal rotation DOF is single-valued. With
$\omega^e = 0$ — the embedding the curvature rows already assume — the tensor contraction is the
only consistent choice.

## How to check a new row

1. Halving or doubling a $\Gamma_e$ row changes that channel's relaxed stiffness by **4x**, not
   2x. A factor-4 discrepancy in one channel is this bug's signature.
2. Membrane rows are safe either way (the $\Gamma_h$ membrane operator already carries both
   transposed terms), so a mistake hides there and only shows in shear.
3. On axis-aligned cells (`sin·cos = 0`) the diagonal columns vanish, so "halve the whole row"
   and "halve the pair-sum columns" agree. **Verify on an inclined-wall cell** (±45) before
   believing either.
4. Validate against a closed form, not another code: periodic cross cell
   `C44 = 0.5·E'·(t/L)^3`; frame lattice `D44 = 2·kv·kh/(kv+kh)`, `k = 6·E'·I/span`.

## Never touch Γh to fix a Γe symptom

The $\Gamma_h$ rows in `solid_props.py` are bit-identical to the validated
`segment_indep.py` ones behind the Timoshenko pipeline (GA22/GA33 within 1 % of VABS).
Rescaling $\Gamma_h$ can reproduce a target number while silently breaking every beam result.
Fix the drive.
