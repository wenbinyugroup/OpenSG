# Rule — every OpenSG run writes a timed SwiftComp-format `.out`

**Scope:** every homogenization entry point in both codes. This is written **in the source**,
not in example scripts, so no run can forget it.

## Format

`opensg_solid.sg_homo.write_sc_K(path, C, solve_time=..., model=..., constants=..., name=...)`
is the one writer. It emits the SwiftComp `.K` layout so results drop straight into existing
comparison tooling:

```
 OpenSG <model>, omega <value>

 The Effective <Name> Stiffness Matrix
 --------------------------------------------
     <6 columns, Fortran E+00x>
 ...
 The Effective <Name> Compliance Matrix
 --------------------------------------------
 ...
 The 6X6 Mass Matrix                                         <- beam route, mass= given
 ========================================================

 The Engineering Constants (Approximated as Orthotropic)     <- 3-D laws only
 ----------------------------------------------------------
  E1  = ...  nu23= ...

 Time taken: <x.xx> sec
```

- ` OpenSG …` banner **first line**, naming the model and the SG measure $\omega$.
- ` Time taken: X sec` **last line**. Never report a solve without it.
- Engineering-constants block only for a 3-D law (`constants=False` for beam/plate).
- The writer is size-generic — a beam 6x6 and a plate matrix format identically.

## `<Name>` — ONE table, and it is the console title

`name` must be the macro law's **console title**, so ` The Effective <Name> Stiffness Matrix`
in the file reads the same as the line the terminal printed. There are exactly five:

| macro law | `<Name>` |
|---|---|
| 3-D solid (Cauchy continuum) 6x6 | `Cauchy Continuum` |
| beam 6x6, shear-refined | `Timoshenko Beam` |
| beam 4x4, classical | `Euler-Bernoulli Beam` |
| plate ABD 6x6, classical | `Classical Plate` |
| plate ABDG 8x8, shear-refined | `Reissner-Mindlin` |

`name=""` gives the anonymous ` The Effective Stiffness Matrix`. **No macro law may use it** —
do not add a sixth spelling, and do not reintroduce the old `Timoshenko` /
`Reissner-Mindlin Plate` forms. The per-section *wall* laws are a different writer,
`opensg_shell.sg_homo.write_abdg_out`, and keep `The Effective Reissner-Mindlin Plate
Stiffness Matrix`, one block per section.

## File names — no `_K` suffix

| model | file |
|---|---|
| msg-solid beam / plate / 3-D (`plate_homo_2d`) | `<base>.out` |
| msg-shell beam, Timoshenko (`build_rm_bundle`, `segment_timo_from_3dyaml`) | `<yaml>_Timo.out` |
| msg-shell beam, classical Kirchhoff–Love wall | `<yaml>_EB.out` |
| msg-shell solid props, cross-section SG (`build_solid_bundle`) | `<yaml>_C3D.out` |
| msg-shell 3-D shell SG (`shell_sg3d`) | `<yaml>_C3D.out` |
| the step-1 **wall** plate law of any shell route (`write_abdg_out`) | `<yaml>_ABDG.out` |

One file per model — do not reintroduce a plain matrix dump alongside the SwiftComp one. When
two routes can run on the same YAML, the suffix disambiguates (`_Timo` vs `_C3D`); a collision
silently overwrites. `_ABDG.out` is not a second copy of the macro law: it is the wall law the
reduction passed through, and every route that builds one now emits it.

## Normalization must be stated, not assumed

The banner carries $\omega$ because the same stiffness has several legitimate normalizations and
a bare matrix is ambiguous. In every case $\omega$ is **measured from the mesh** unless the yaml
declares an `omega:` header key, which wins:

- **cross-section SG** — $\omega$ = the measured periodic cell, i.e. the node **bounding box**
  (the assembly map ties opposite box faces). Not the convex hull of the contour: the two
  differ on any non-convex cell, which is every honeycomb and lattice. Declare `omega:` when
  the equivalent continuum occupies something else, e.g. the wall *material* area of a closed
  tube.
- **3-D shell SG** — $\omega$ = the node bounding-box **volume**, the volume the equivalent
  continuum occupies and the same cell the three-direction periodic map ties. The returned
  `C3D`, the CLI print and `<base>_C3D.out` are therefore one and the same matrix.
- **3-D solid SG** — $\omega$ = the node bounding-box **volume**, *not* the summed element
  (material) volume. This is the convention change that made the `.out` directly comparable
  with a SwiftComp `.K`: under the old material-volume rule the file came out
  $1/\text{relative density}$ too stiff and only a rescale by $\omega$ in the comparison
  script matched the vendor table. Nothing downstream should rescale a 3-D solid `.out`.

A whole-matrix ratio that is *uniform* across every entry (e.g. exactly 0.5000) is a
normalization difference, **not** a physics discrepancy. Check that before debugging.
