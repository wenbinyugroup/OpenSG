# Stress recovery in a blade cross-section

## Problem description

% source: docs/tutorials/msg_shell_beam.md ("What a 1-D shell SG is", "The YAML blocks")
Homogenization gives the beam its stiffness; recovery (dehomogenization) does the inverse. Given
the internal forces of the beam at one station, this example reconstructs the 3-D stress at every
depth of every wall, which is what a failure criterion needs. The structure gene (SG) is the
1-D shell SG of {doc}`shell_beam_cross_section` ({numref}`fig-sbc-mesh`): station $r/R = 0.20$ of
the IEA-22 MW blade, 307 nodes, 310 line elements, six layups, mid-surface reference (`--center`).
The engine is `opensg_shell` and the macro model is the Timoshenko beam.

% source: docs/tutorials/msg_shell_beam.md ("The two-step chain")
The recovery has two steps.

1. **Along the contour.** The macro beam strain is $\mathrm{st} = C_6^{-1}FF$. The bundle's warping
   fields turn it into the nodal warping $w = V_0\,\mathrm{st}_m + V_1\,\mathrm{st}_{c1}$ and its axial
   derivative $w' = V_0\,\mathrm{st}_{c1} + V_1\,\mathrm{st}_{c2}$, and the same element operators
   that assembled the stiffness give, per element, the six shell strains
   $[\varepsilon_{11}\;\varepsilon_{22}\;2\varepsilon_{12}\;K_{11}\;K_{22}\;2K_{12}]$ at mid-arc, with
   their span and arc gradients. This step is the energy-consistent adjoint of the homogenization,
   not a separate post-processing model. The arc gradient is a finite difference of nodal averages
   that never crosses a change of layup, where the strains legitimately jump.
2. **Through the thickness.** For each layup, `rm_plate_msg` builds a 1-D mechanics of structure genome (MSG) Reissner-Mindlin
   (RM) plate SG, and `msgrm_strain_at_depth` evaluates the first-order recovery of Eq. (63) at a
   depth $z$ from the reference surface. It returns the 3-D strain, the 3-D stress
   $[\sigma_{11}\;\sigma_{22}\;\sigma_{33}\;\sigma_{23}\;\sigma_{13}\;\sigma_{12}]$ and the ply angle.
   Both functions come from `opensg_solid.rm_plate_1D.msg_rm_plate`, shared with the plate examples.

% source: docs/tutorials/msg_shell_beam.md ("The two-step chain"); examples/OpenSG_shell/2_get_beam_dehom_from_shell_cross_section/beam_dehom_shell.py (n_depth, zeta)
Nine ply-interior stations per element, at $\zeta = (k + \tfrac12)/9$ with $\zeta = 0$ at the outer
mold line (OML) and $\zeta = 1$ at the inner mold line (IML), give $310 \times 9 = 2790$ recovery
points. The result is compared with a VABS 2-D solid model of the identical section. Complete input
files are in `examples/OpenSG_shell/2_get_beam_dehom_from_shell_cross_section/`.

## Input

The beam load lives in `iea_s10_shell.ff` next to the yaml. The SG yaml describes the structure,
this file carries the loading:

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

% source: docs/tutorials/msg_shell_beam.md (Example 2, "Input the user edits"); examples/OpenSG_shell/2_get_beam_dehom_from_shell_cross_section/ff51_rmc_reform.dat (header line)
`FF` is in the VABS order and the SI units of the $6\times6$ (N, N·m): a 60 MN·m flapwise moment
with a 1.47 MN flapwise shear, which dominates everything that follows. The forces must be referred
to the same axis as the stiffness; this load is center-referenced, matching the `--center` run.
The row comes from `ff51_rmc_reform.dat`, a plain table with one row per spanwise station, the
first column $\eta = r/R$ and the six resultants after it, under the header line
`# eta F1 F2 F3 M1 M2 M3 (RM CENTER-ref BeamDyn, VABS order)`. When there is no `.ff`, the
recovery takes the macro beam strain from `epsilon_bar:` in the yaml header instead.

Three optional header keys control how the recovery treats wall junctions
({numref}`tab-sbr-junction-keys`). At a wall crossing, the through-thickness columns of two walls
cover the same material twice, with two different wall laws, which is how a spar-cap ply stress can
be reported at a web station. The keys measure that hazard; they do not repair it.

% source: docs/tutorials/msg_shell_beam.md ("Junctions in the recovery"), src/opensg_shell/cli.py (header keys)
```{list-table} Junction keys of the recovery
:header-rows: 1
:name: tab-sbr-junction-keys

* - Key
  - Values
  - Default
  - Meaning
* - `junction:`
  - `off`, `flag`, `exclude`
  - `flag`
  - `off`: no extra columns, no sidecar, no census. `flag`: appends the columns `jflag` and `jdist` at the end of `<base>_dehom.txt`, adds them as `.vtk` `CELL_DATA` scalars, writes the `<base>_dehom.junc` sidecar and prints the census; no field value changes. `exclude`: as `flag`, plus `NaN` in the 15 field columns of every duplicate (`jflag` 3) row, so the cloud partitions the material instead of double-covering it.
* - `junction_bl:`
  - float
  - `1.0`
  - $k_{\rm bl}$, the boundary-layer radius in units of the junction's thickest wall (a modelling choice, widen it to be conservative)
* - `junction_ang:`
  - float, degrees
  - `1.0`
  - tangent-grouping tolerance of the junction detector; 5 to 15 lists the wall crossings only (it moves the advisory `near` count, not the overlap flags)
```

:::{note}
Under `exclude` the row itself stays: row $i$ is still element $i/\!/n_{\rm depth}$ at depth
$i\%n_{\rm depth}$. The `.vtk` values are never set to `NaN`; threshold the mesh on `jflag` instead.
A fourth tier, `patch`, raises an error if requested. This `junction:` key is a recovery
bookkeeping tier, not the homogenization junction correction of `build_solid_bundle`; see
{doc}`../theory/rules/junction_corrections`.
:::

The driver script `beam_dehom_shell.py` has this user-input block:

% source: examples/OpenSG_shell/2_get_beam_dehom_from_shell_cross_section/beam_dehom_shell.py
```python
name = "iea_s10"                  # reads <name>_shell.yaml
ff_file = "ff51_rmc_reform.dat"   # beam FF per station (# eta F1 F2 F3 M1 M2 M3)
station_row = 10                  # row 10 -> eta = r/R = 0.20 (iea_s10)
n_depth = 9                       # through-thickness recovery points per element
stress_frame = "material"         # "material" (ply axes) | "plate" (wall axes)
```

## Run

```bash
cd examples/OpenSG_shell/2_get_beam_dehom_from_shell_cross_section
opensg iea_s10_shell.yaml D --center
```

% source: examples/OpenSG_shell/2_get_beam_dehom_from_shell_cross_section/iea_s10_shell_Timo.out (footer); src/opensg_shell/cli.py (D frame flags; --global selects frame "plate")
The `D` argument (or `analysis: D` in the header) runs the homogenization and then the recovery
in one pass, with the macro state of `iea_s10_shell.ff`. The homogenization part is the one of
{doc}`shell_beam_cross_section`; the copy of `iea_s10_shell_Timo.out` in this folder ends with
`Time taken: 2.91 sec`. Stresses are written in each element's
ply axes (`--material`, the default) or in the wall (plate) axes (`--global`, the driver's
`stress_frame = "plate"`). At the default
`junction: flag` the run prints the junction census; for this station it reads

% source: docs/tutorials/msg_shell_beam.md ("Junctions in the recovery")
```text
 junctions : 97 detected (5 T, 91 L, 1 unresolved)
 stations  : 2790 total, 1032 near, 56 overlapped (50 duplicates)
```

and the run closes with

% source: src/opensg_shell/cli.py (the dehomogenization branch)
```text
Local field files are computed and stored.
Time taken: <seconds> sec
```

% source: docs/tutorials/msg_shell_beam.md ("Junctions in the recovery")
At the default 1° tolerance, most of the 97 junctions are discretized contour curvature and only a
handful are genuine wall crossings, which is why the `near` count is large while the overlap counts
stay small.

The driver script performs the same two-step recovery through the Python interface, adds the
figures and prints the peak stress; `write_dehom_vtk.py` re-runs it and writes the exploded
through-thickness mesh as legacy VTK:

```bash
python beam_dehom_shell.py
python write_dehom_vtk.py
```

% source: docs/tutorials/msg_shell_beam.md ("What comes out")
```text
peak |sigma11| = 318.18 MPa at (y2, y3) = (-0.677, 1.311), zeta = 0.06  [material frame]
```

The driver builds the bundle with `build_rm_bundle(name + "_shell.yaml")`, whose default reference
is `"oml"`; the numbers on this page are for the mid-surface reference, so pass `ref="center"` to
reproduce them. The two steps are these calls of the driver (excerpt):

% source: examples/OpenSG_shell/2_get_beam_dehom_from_shell_cross_section/beam_dehom_shell.py (excerpt; section comments shortened, loop abbreviated)
```python
from opensg_shell import build_rm_bundle, ring_wall_strains
from opensg_solid.rm_plate_1D.msg_rm_plate import rm_plate_msg, msgrm_strain_at_depth

B = build_rm_bundle(name + "_shell.yaml")
FF = np.loadtxt(ff_file)[station_row, 1:]

# step 1: RM shell section strains
F = ring_wall_strains(B, beam_force_vabs=FF)
s6mid, dE1e, dE2e, s6n = F["s6mid"], F["dE1"], F["dE2"], F["s6n"]

# step 2: MSG-RM through-thickness recovery
warpM = {ln: rm_plate_msg(i["thick"], i["angles"], i["mat_names"], mdb, fraction=frac)
         for ln, i in ldb.items()}
zeta = (np.arange(n_depth) + 0.5) / n_depth            # ply-interior stations, 0=OML..1=IML
...
        z = (zeta[k] - frac) * h                        # depth from the reference surface
        Gam, Sig, ply = msgrm_strain_at_depth(warpM[ln], z, s6, dE1e[e], dE2e[e],
                                              frame=stress_frame)
```

For a single point, the lower-level calls are `_macro_fields(B, beam_force_vabs=FF)`, which
returns `st, st_m, aA, aB`, and `_rm_shell_strain(B, e, xi, st_m, aA, aB)`, which returns the six
shell strains `s6` and the two transverse shears `s2` of element `e` at arc coordinate `xi`.

## Result

% source: docs/tutorials/msg_shell_beam.md ("Reading the result")
{numref}`fig-sbr-cloud` shows the full recovery-point cloud. Each point is displaced from the
mid-surface by its own depth $z$ along the wall normal, which is why the cap points reach
$y_3 = 1.311$ m while the contour stops at $1.273$ m. {numref}`fig-sbr-pv-s11` and
{numref}`fig-sbr-pv-s12` show the same fields on the exploded through-thickness mesh.

```{figure} ../_static/shell_dehom_stress.png
:width: 90%
:name: fig-sbr-cloud

Recovered $\sigma_{11}$ (top) and $\sigma_{12}$ (bottom) at the 2790 recovery points, MPa.
```

```{figure} ../_static/pv_shell_dehom_s11.png
:width: 95%
:name: fig-sbr-pv-s11

$\sigma_{11}$ (MPa) on the through-thickness recovery mesh, one cell per element and depth band.
```

```{figure} ../_static/pv_shell_dehom_s12.png
:width: 95%
:name: fig-sbr-pv-s12

$\sigma_{12}$ (MPa) on the through-thickness recovery mesh.
```

% source: docs/tutorials/msg_shell_beam.md ("Reading the result")
The axial stress is confined almost entirely to the two spar caps, as a flap-dominated load state
requires ({numref}`tab-sbr-extremes`). The upper cap is in compression and the lower cap in
tension at nearly equal magnitude, and both extremes fall at the outermost recovery station
$\zeta = 0.056$, which for these layups lies just inside the carbon uniaxial ply: that is where a
maximum-stress check on the carbon belongs. The in-plane shear peaks in the trailing-edge and
leading-edge regions, where the closed cell carries the torque as shear flow.

% source: docs/tutorials/msg_shell_beam.md ("Reading the result")
```{list-table} Stress extremes of the recovery
:header-rows: 1
:name: tab-sbr-extremes

* - Quantity
  - Value (MPa)
  - Where
* - minimum $\sigma_{11}$
  - $-318.177$
  - element 75, `layup_2`, $(y_2,y_3) = (-0.677, 1.311)$, $\zeta = 0.056$
* - maximum $\sigma_{11}$
  - $+314.461$
  - element 150, `layup_4`, $(y_2,y_3) = (-0.145, -1.249)$, $\zeta = 0.056$
* - maximum $\lvert\sigma_{12}\rvert$
  - $5.573$
  - element 11, `layup_0`, $(y_2,y_3) = (3.895, 0.335)$
```

The folder `examples/OpenSG_shell/IEA_blade_beam/` compares this recovery with a VABS 2-D solid
run of the identical section (`iea_s10.sg.SM`) and records the root-mean-square (rms) differences
next to the rms of the VABS field ({numref}`tab-sbr-vabs`).

% source: examples/OpenSG_shell/IEA_blade_beam/iea_s10_report.txt
```{list-table} Recovered stress against VABS (rms over each region or path)
:header-rows: 1
:name: tab-sbr-vabs

* - Component
  - Region
  - rms difference (MPa)
  - rms of the VABS field (MPa)
* - $\sigma_{11}$
  - skin
  - 9.529
  - 96.676
* - $\sigma_{11}$
  - web
  - 43.073
  - 44.440
* - $\sigma_{12}$
  - web
  - 6.799
  - 22.026
* - $\sigma_{11}$
  - spar-cap through-thickness path
  - 38.355
  - 285.457
* - $\sigma_{11}$
  - cap-left junction path
  - 1.265
  - 286.837
```

% source: docs/tutorials/msg_shell_beam.md ("What this route recovers, and what it does not"); examples/OpenSG_shell/IEA_blade_beam/iea_s10_report.txt (V2 payload, s11 row)
The in-plane field is sound where the load is: about 10 % rms on the skin, 13 % along the cap
thickness and 0.4 % along the reference junction path. The web $\sigma_{11}$ difference is the size
of the field itself because the webs carry almost no axial stress in this load state. The
second-order recovery rung (Eq. 66) changes $\sigma_{11}$ by an rms of 1.950e-04 MPa against a
first-order rms of 8.851e+01 MPa, so the first-order chain is the right default for this class of
section.

% source: examples/OpenSG_shell/IEA_blade_beam/iea_s10_report.txt and iea_s10_q_consistency.txt; S33 column statistics from docs/tutorials/msg_shell_beam.md
:::{warning}
Two components are not delivered by this route. $\sigma_{33}$ is machine zero in the first-order
recovery (rms 3.626e-13 MPa against a VABS rms of 5.576e-02 MPa); over the 2790 points of
`iea_s10_dehom_shell.txt` its median is $1.96\times10^{-13}$ and its maximum $9.15\times10^{-13}$
MPa. $\sigma_{13}$ is recovered at rms 6.306e-03 MPa against a VABS rms of 2.389e-01 MPa: on a
prismatic ring the $\gamma_{13}$ row is untied by construction and cannot serve as a recovery
target. `iea_s10_q_consistency.txt` shows that the tied $\gamma_{23}$ channel closes its equilibrium
identity $\int \sigma_{23}\,dz = (G_{\text{msg}}\,s_2)_2$ to 1.05 % median error while the
$\gamma_{13}$ channel does not. A physical $\sigma_{13}$ needs a surface-quad route (a segment or
3-D shell SG, where both rows are tied).
:::

% source: docs/tutorials/msg_shell_beam.md ("Junctions in the recovery")
With `junction:` at `flag` or `exclude`, each recovery row carries the flag of
{numref}`tab-sbr-jflag` and the distance $\texttt{jdist} = |p - J| / \max_i t_i$ to the nearest
junction node $J$ (infinite when the section has no junction), so `jflag >= 1` is exactly
`jdist <= k_bl`. Ownership at a crossing follows the homogenization's own patch mesh: at a T, the
through-family owns the block; at an X, two through-families mitre by $|s_i|/t_j$ against
$|s_j|/t_i$; at an L, two ending families mitre on the normalized-band diagonal; with more than
two families the nearest midline wins and the junction is reported unresolved. The sidecar
`iea_s10_shell_dehom.junc` lists every junction with its coordinates, topology, overlap area
$A_j = \sum t_i t_j / |\sin\theta_{ij}|$ and ownership decision.

% source: docs/tutorials/msg_shell_beam.md ("Junctions in the recovery")
```{list-table} Values of jflag
:header-rows: 1
:name: tab-sbr-jflag

* - `jflag`
  - Meaning
* - 0
  - clean, outside every junction's boundary layer
* - 1
  - near, inside some $r_{\rm bl}$ but inside no other wall's material
* - 2
  - overlapped, and this wall owns the block
* - 3
  - duplicate, inside another wall's material, and another wall owns it
* - 4
  - reserved for `patch`; never emitted
```

## Files

% source: examples/OpenSG_shell/2_get_beam_dehom_from_shell_cross_section/ (folder listing, driver docstrings) and docs/tutorials/msg_shell_beam.md ("What comes out")
| File | Kind | Content |
|---|---|---|
| `iea_s10_shell.yaml` | input | the 1-D shell SG (same as {doc}`shell_beam_cross_section`) |
| `iea_s10_shell.ff` | input | macro state: `u`, `theta`, `C`, `FF` |
| `ff51_rmc_reform.dat` | input | beam forces of 51 spanwise stations (driver input) |
| `beam_dehom_shell.py` | driver | two-step recovery, figures, peak stress |
| `write_dehom_vtk.py` | driver | exploded-mesh VTK of the driver's recovery |
| `iea_s10_shell_Timo.out`, `iea_s10_shell_ABDG.out` | output | the homogenization that precedes the recovery |
| `iea_s10_shell_dehom.txt` | output (`D`) | 2790 rows: `elem`, `zeta`, `y2`, `y3`, `z`, six stresses (Pa), six strains, three displacements, then `jflag` and `jdist` |
| `iea_s10_shell_dehom.vtk` | output (`D`) | one quad per element and depth band, each with its own station's stress as `CELL_DATA` (never node-averaged), plus `jflag` and `jdist` |
| `iea_s10_shell_dehom.junc` | output (`D`) | junction census, coordinates, topology, overlap area and ownership |
| `iea_s10_dehom_shell.txt` | output (driver) | 2790 rows: `elem`, `y2`, `y3`, `z`, `zeta`, ply angle (deg), six stresses (MPa); the header repeats frame, $\eta$ and `FF` |
| `iea_s10_dehom_stress.png` | output (driver) | the section stress cloud, $\sigma_{11}$ and $\sigma_{12}$ |
| `iea_s10_cap_tt.png` | output (driver) | $\sigma_{11}$ and $\sigma_{12}$ against $\zeta$ through the arc-centre element of the peak-stress cap |
| `iea_s10_dehom_shell.vtk` | output (`write_dehom_vtk.py`) | QUAD cells with `CELL_DATA` S11 S22 S33 S23 S13 S12 (MPa) |
