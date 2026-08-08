# Rule — every OpenSG run writes a timed SwiftComp-format `.out`

**Scope:** every homogenization entry point in both codes. This is written **in the source**,
not in example scripts, so no run can forget it.

## Format

`opensg_solid.sg_homo.write_sc_K(path, C, solve_time=..., model=..., constants=...)` is the one
writer. It emits the SwiftComp `.K` layout so results drop straight into existing comparison
tooling:

```
 OpenSG <model>, omega <value>

 The Effective Stiffness Matrix
 --------------------------------------------
     <6 columns, Fortran E+00x>
 ...
 The Effective Compliance Matrix
 --------------------------------------------
 ...
 The Engineering Constants (Approximated as Orthotropic)     <- 3-D laws only
 ----------------------------------------------------------
  E1  = ...  nu23= ...

 Time taken: <x.xx> sec
```

- ` OpenSG …` banner **first line**, naming the model and the SG measure $\omega$.
- ` Time taken: X sec` **last line**. Never report a solve without it.
- Engineering-constants block only for a 3-D law (`constants=False` for beam/plate).
- The writer is size-generic — a beam 6x6 and a plate matrix format identically.

## File names — no `_K` suffix

| model | file |
|---|---|
| msg-solid beam / plate / 3-D (`plate_homo_2d`) | `<base>.out` |
| msg-shell beam, Timoshenko (`build_rm_bundle`) | `<yaml>_Timo.out` |
| msg-shell solid props, cross-section SG (`build_solid_bundle`) | `<yaml>_C3D.out` |
| msg-shell 3-D shell SG (`shell_sg3d`) | `<yaml>_C3D.out` |

One file per model — do not reintroduce a plain matrix dump alongside the SwiftComp one. When
two routes can run on the same YAML, the suffix disambiguates (`_Timo` vs `_C3D`); a collision
silently overwrites.

## Normalization must be stated, not assumed

The banner carries $\omega$ because the same stiffness has several legitimate normalizations and
a bare matrix is ambiguous:

- **cross-section SG** — $\omega$ = cell area (`cell_area`, or the convex hull if not given).
- **3-D shell SG** — the returned `C3D` uses $\omega$ = **midsurface surface area** (the 3-D
  analogue of $\omega$ = perimeter for a plane section); the `.out` is written **per unit-cell
  volume** so its moduli compare directly with a solid `.K`.
- **3-D solid SG** — `C_eff` is per material volume; multiply by $\omega$ for per-cell.

A whole-matrix ratio that is *uniform* across every entry (e.g. exactly 0.5000) is a
normalization difference, **not** a physics discrepancy. Check that before debugging.
