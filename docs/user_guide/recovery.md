# Specifications for recovery

Homogenization replaces the structure gene (SG) by a macro law, and it keeps the fluctuation
(warping) fields that produced the law. Recovery, or dehomogenization, runs that construction
backwards: given the macro state at a point of the beam, plate or solid, it returns the local
3-D strain, stress and displacement inside the SG. Nothing is re-solved; the recovery reuses the
fields the homogenization stored.

`opensg <sg.yaml> D` (or `analysis: D` in the yaml header) homogenizes and then recovers in one
pass. The yaml describes the structure only. The loading lives in a second file, `<base>.ff`,
next to the yaml, or in the `epsilon_bar:` header key when there is no `.ff`. Three recovery
routes exist:

% source: src/opensg_shell/cli.py (dehom_write); src/opensg_solid/cli.py; docs/tutorials/msg_solid_plate.md
| route | SG and macro model | chain | files written |
|---|---|---|---|
| msg-shell | 1-D cross-section shell SG, beam, `refined: 1` | two steps: beam force to wall strains, then the through-thickness plate SG of each layup | `<base>_dehom.txt`, `.vtk`, `.junc` |
| msg-solid | 1-D, 2-D or 3-D solid SG; beam, plate or 3-D solid | one step on the stored warping fields | `<base>_dehom.txt`, `.vtk`, `.SM`, `.EM`, `.U` and the per-element files |
| plate field | through-thickness SG of `layup_db.yaml` | `rm_dehom_ff.py` over a table of plate stations | `<ff stem>_ff.SM`, `.EM`, `.U`, `_ff_gauss.vtk` |

The other msg-shell routes (the equivalent solid, the classical beam and the 3-D shell segment)
have no recovery, and `D` stops with a message that says so. The two chains are derived in
{doc}`../theory/constitutive`; the files are described in {doc}`output_files`.

## The macro state file

`<base>.ff` is a yaml mapping of the keys below. It must carry a force (`1:`, or the
older spelling `FF:`) or a strain (`0:`); a file with neither is not read as a macro state.

% source: src/opensg_solid/cli.py (read_ff_state)
| key | type | meaning | default |
|---|---|---|---|
| `u` | 3 floats | macro displacement $u_1, u_2, u_3$ | `[0, 0, 0]` |
| `theta` | 3 floats | macro rotation about $y_1, y_2, y_3$, in rad | `[0, 0, 0]` |
| `C` | 3 rows of 3 | direction cosines of the macro frame; overrides `theta` | the small rotation built from `theta` |
| `1:` or `FF:` | 6 floats | generalized macro forces, in the order of the model's stiffness | none |
| `0:` | 6 floats | generalized macro strain, used directly (msg-solid only) | none |

The component order follows the macro model:

% source: src/opensg_solid/cli.py (read_ff_state and the header docstring)
| macro model | `1:` or `FF:` | `0:` and `epsilon_bar` |
|---|---|---|
| beam | $[F_1\;F_2\;F_3\;M_1\;M_2\;M_3]$ | `[ext sh2 sh3 twist bend2 bend3]` |
| plate | $[N_{11}\;N_{22}\;N_{12}\;M_{11}\;M_{22}\;M_{12}]$ | `[e11 e22 2e12 k11 k22 2k12]` |
| 3-D solid | $[S_{11}\;S_{22}\;S_{33}\;S_{23}\;S_{13}\;S_{12}]$ | the six macro strains in the order $[11\;22\;33\;23\;13\;12]$, engineering shears |

The shell recovery example ships this file beside `iea_s10_shell.yaml`:

% source: examples/OpenSG_shell/2_get_beam_dehom_from_shell_cross_section/iea_s10_shell.ff
```yaml
# OpenSG dehomogenization macro state -- read by `opensg_shell iea_s10_shell.yaml D`
# The SG yaml describes the STRUCTURE; this file carries the LOADING.
u:     [0.0, 0.0, 0.0]      # macro displacement u1 u2 u3
theta: [0.0, 0.0, 0.0]      # macro rotation t1 t2 t3 [rad] (about y1 y2 y3)
C:     [[1.0, 0.0, 0.0],       # direction cosines of the macro frame
        [0.0, 1.0, 0.0],       #   (identity = unrotated; overrides theta)
        [0.0, 0.0, 1.0]]
# generalized macro forces, beam [F1 F2 F3 M1 M2 M3]
# IEA-22 station r/R = 0.20 (ff51_rmc_reform.dat row 10, RM center-ref BeamDyn)
FF:    [4.19268867e+04, 8.26739746e+03, 1.46733512e+06, 1.21914500e+06, -6.01849720e+07, 3.04264594e+05]
```

% source: docs/tutorials/msg_shell_beam.md
```bash
cd examples/OpenSG_shell/2_get_beam_dehom_from_shell_cross_section
opensg iea_s10_shell.yaml D --center
```

The forces must be referred to the same axis as the stiffness; this load table is
center-referenced, which is why the run carries `--center`.

### Which state is used

- **msg-solid:** `0:` first (`macro state: 0: (global strain, used directly)`), then `1:` or
  `FF:`, inverted through the effective stiffness
  (`macro state: 1:/FF (force, inverted through C_eff)`), then `epsilon_bar:`.
- **msg-shell:** the force, giving the macro beam strain $C_6^{-1}\,FF$ through the Timoshenko law,
  then `epsilon_bar:`. The shell recovery does not read `0:`.
- With none of them, the run stops with the message below.

% source: src/opensg_solid/cli.py (main); src/opensg_shell/cli.py
```text
analysis D needs the macro state: either <base>.ff (u, theta, C, FF) or `epsilon_bar:` in the yaml header
```

The recovered displacement adds the macro rigid motion of the `.ff` to the warping, in the
recovery form of VABS (the Variational Asymptotic Beam Section code),

% source: src/opensg_solid/cli.py (main); src/opensg_shell/cli.py (dehom_write)
$$U_i = u_i + (C_{ij} - \delta_{ij})\,y_j + w_i .$$

:::{note}
A file that carries only the older `FF:` key is upgraded in place on the first `D` run: the
reader appends a `1:` line with the same six values and prints
`note: <base>.ff upgraded in place (legacy FF: -> 1:)`.
:::

### The `epsilon_bar:` header key

Without a `.ff`, the macro strain comes from the yaml header, in the strain order of the table
above. Example 8 carries one, although the `.ff` in the same folder takes precedence:

% source: examples/OpenSG-solid/8_get_beam_dehom_from_SG/RHC_SW_2UC_45.yaml
```yaml
epsilon_bar: [0.001, 0.0, 0.0, 0.0, 0.01, 0.0]   # macro beam strain for opensg_solid <yaml> D
```

### Refined-recovery drivers

A shear-refined plate run (`refined: 1`) of the solid engine reads further optional `.ff` keys.
Each one adds a term to the recovery; a run that cannot use a key prints a note and skips it,
for example:

% source: src/opensg_solid/cli.py (main)
```text
note: <base>.ff carries strain derivatives, but the yaml ran classical (refined: 0) -- V2 recovery skipped; set `refined: 1` to use them
```

% source: src/opensg_solid/cli.py (read_ff_state, main)
| key | type | meaning | default |
|---|---|---|---|
| `Q` | 2 floats | transverse shear resultants $Q_1, Q_2$; with the two strain derivatives below, turns on the rescale of the recovered $\sigma_{13}, \sigma_{23}$ to them | none |
| `deps_dx1`, `deps_dx2` | 6 floats each | $\partial/\partial x_1$ and $\partial/\partial x_2$ of the plate strain measures `[e11 e22 2e12 k11 k22 2k12]`; drive the Eq. 63 term | none |
| `qt6`, `qb6` | 6 floats each | top and bottom face pressure ladders $[q\;q_{,1}\;q_{,2}\;q_{,11}\;q_{,12}\;q_{,22}]$, $q$ positive pushing into the face | none |
| `q_reaction` | `uniform` or `tau` | how the pressure load column reacts its net face force inside the cell; `tau` follows the cell's own $\sigma_{xz}$ path, for cells heterogeneous in plane | `uniform` |
| `d2eps_dx1dx1`, `d2eps_dx1dx2`, `d2eps_dx2dx2` | 6 floats each | second derivatives of the plate measures; drive the Eq. 64 to 66 second-order recovery of 2-D plate SGs | none |
| `d3eps_dx1dx1dx1` to `d3eps_dx2dx2dx2`, `d4eps_dx1dx1dx1dx1` to `d4eps_dx2dx2dx2dx2` | 6 floats each | third and fourth derivatives of the plate measures; feed the Eq. 50 conversion of the derivative drivers to the classical measure | none |

## Output frame

The recovery computes in the SG axes. A flag on the `D` command picks the frame of the written
stress and strain:

% source: src/opensg_solid/cli.py; src/opensg_shell/cli.py
| flag | msg-solid | msg-shell |
|---|---|---|
| `--material` (default) | each element's ply axes, $\theta = +$`angle` about the thickness axis | each element's ply axes |
| `--global` | the SG axes; the files are named `<base>_dehom_global.SM` and `_global.EM` | the wall (plate) axes |

The file headers name the frame. The displacement file `.U` is never rotated. On the solid
engine the material frame is defined for block-angle yamls only (`angle:` on the material
blocks). Recover a yaml whose elements carry their own frames (`elementOrientations`) with
`--global`: `--material` rotates by the block angle alone and does not compose it with the
element frame.

## Recovered fields

**msg-solid.** Strain, stress and displacement at every element Gauss point. The `.txt` table
keeps the SwiftComp storage order $(xx, yy, zz, yz, xz, xy)$; the `.SM` and `.EM` files
print `11 22 33 12 13 23`, reordered on write, and the `.U` file prints `U1 U2 U3`. The SG coordinates fill the last $n_{\rm sg}$
slots of the `x y z` columns, leading slots zero, and $z = 0$ is the reference surface. Units
follow the SG input. From Python the same recovery is two calls on the homogenization result,
with `epsilon_bar` in the strain order of the table above:

% source: docs/theory/constitutive.md (the solid chain)
```python
from opensg_solid.sg_homo import plate_homo_2d
from opensg_solid.sg_dehom import dehom_fields, export_gauss

r = plate_homo_2d("RHC_SW_2UC_45.yaml", n_model=2)
epsilon_bar = [0.001, 0.0, 0.0, 0.0, 0.01, 0.0]
Gamma, Sigma, U = dehom_fields(r, epsilon_bar)
export_gauss(r, Gamma, Sigma, "RHC_SW_2UC_45_dehom", U_eqd=U)
```

**msg-shell.** One column of 9 ply-interior recovery points per element, hung off the element
mid-arc along the inward wall normal, at $\zeta = (k + \tfrac12)/9$ with $\zeta = 0$ at the outer
mold line and $\zeta = 1$ at the inner one. The depth $z$ is measured from the reference surface.
The IEA station gives $310 \times 9 = 2790$ points ({numref}`fig-rec-shell-stress`).

```{figure} ../_static/shell_dehom_stress.png
:width: 90%
:name: fig-rec-shell-stress

Recovered $\sigma_{11}$ and $\sigma_{12}$ over the IEA-22 blade cross-section at $r/R = 0.20$,
plotted at the recovery points around the contour and webs.
```

### Junctions in the shell recovery

Within about a wall thickness of a wall crossing, the columns of two walls cover the same
material twice. Three header keys of the msg-shell `D` run measure this and say what to do:

% source: src/opensg_shell/cli.py (the junction: header key, dehom_write)
| `junction:` | behaviour |
|---|---|
| `off` | no extra columns, no sidecar, no census |
| `flag` | the default, and non-mutating: columns `jflag` and `jdist` appended at the end of `<base>_dehom.txt`, the same two added to the `.vtk` as `CELL_DATA`, the sidecar `<base>_dehom.junc` written and the census printed; no field value changes |
| `exclude` | as `flag`, plus `NaN` in the 15 field columns of every duplicate (`jflag` 3) row; the row stays, so row $i$ is still element $\lfloor i/9 \rfloor$ at depth $i \bmod 9$; the `.vtk` values are never `NaN` |

% source: src/opensg_shell/sg_dehom_junction.py
| `jflag` | meaning |
|---|---|
| 0 | clean: outside every junction's boundary layer |
| 1 | near: inside some boundary layer, inside no other wall's material |
| 2 | overlapped: inside another wall's material, and this wall owns the block |
| 3 | duplicate: inside another wall's material, and another wall owns it |
| 4 | reserved for a `patch` tier that is not implemented; never written |

`jdist` is the distance to the nearest junction node in units of the thickest wall there.
`junction_bl:` (default `1.0`) is the boundary-layer radius in those units, and `junction_ang:`
(default `1.0`, degrees) is the tangent-grouping tolerance of the detector; a value of 5 to 15
lists the wall crossings only.

:::{warning}
`junction:` is a recovery bookkeeping tier. It is not the homogenization junction correction,
which is described in {doc}`../theory/rules/junction_corrections`.
:::

## Plate field recovery from a station table

The 1-D plate SG of `layup_db.yaml` recovers a whole plate field with
`examples/OpenSG-solid/2_get_plate_stress_from_1DSG/rm_dehom_ff.py`. Its `.ff` is a different
file: a whitespace table with `#` comments and one row per in-plane station, in any order and on
any station geometry. Any solver can write it; the shipped table was collected from an Abaqus
S8R shell field:

% source: examples/OpenSG-solid/2_get_plate_stress_from_1DSG/ex5_shell_S8R_field.ff
```text
# plate dehom FF station table (collected from ex5_shell_S8R_field.dat)
# x[m] y[m] N11 N22 N12[N/m] M11 M22 M12[N] Q1 Q2[N/m] u1 u2 u3[m] wx wy[-]
# 40 x 40 rectangular station lattice, x-fastest
```

Ten columns are required (`x y N11 N22 N12 M11 M22 M12 Q1 Q2`); columns 11 to 15 add the
mid-surface displacement and slopes; without them the `.U` file carries the warping alone. Every
gradient is computed from the stations: exact finite
differences when the stations form a complete rectangular lattice, otherwise a meshfree patch fit
over the $k = 12$ nearest neighbours (`--scatter` forces it). The applied face pressure is input,
not solver output, so it lives in `layup_db.yaml`:

% source: examples/OpenSG-solid/2_get_plate_stress_from_1DSG/layup_db.yaml
```yaml
dehom:
  ff: Abaqus_results/ex5_shell_S8R_field.ff
  loads:
    top:                    # the ex5 double-sine pressure
      q0: 68.9476e6         #   amplitude [Pa]
      shape: sinsin         #   uniform | sin_x | sin_y | sinsin
    bottom: none            # traction-free
```

Each `shape` fixes the six-term pressure ladder analytically, differentiated with the side
lengths `a` and `b` of the `plate:` block of the same database.

% source: docs/tutorials/msg_solid_plate.md
```bash
python rm_dehom_ff.py ex5_shell_S8R_field.ff
```

:::{warning}
The shipped `dehom: ff:` path names an `Abaqus_results` folder that is not shipped; the table sits
in the example folder itself. Name the table on the command line, as above, or correct the key.
`--db` points at a database elsewhere. Bottom-face loads raise `NotImplementedError`.
By default `rm_dehom_ff.py` reads the `layup_db.yaml` and `1dsg.yaml` beside it in
`examples/OpenSG-solid/2_get_plate_stress_from_1DSG/`, not the copy in example 1; a database
named with `--db` needs its own `1dsg.yaml` beside it.
:::

The run prints the gradient path it chose, `lattice FD (40 x 40)` for the shipped table. The
output is named after the table (`ex5_shell_S8R_field_ff.SM`, `.EM`, `.U` and, on a lattice,
`_ff_gauss.vtk`), at the Gauss depths of each SG element (five per ply, 45 in all), with stress
and strain in the ply material frame and $z$ measured from the reference surface. The text files
share a header that names the station set, here
`1600 .ff stations [lattice FD (40 x 40)] x the 45 SG-element Gauss depths`. The single-station driver
`rm_dehom.py` in the same folder takes the resultants from a `USER INPUT` block instead.

## See also

- {doc}`output_files` for the columns and headers of every recovery file.
- {doc}`converters` for `opensg ff_to_glb`, which turns the same `.ff` into a SwiftComp `.glb`.
- {doc}`../examples/shell_beam_recovery` and {doc}`../examples/solid_plate_1d_recovery` for the
  complete runs.
