# IEA-22 blade: beam properties and stress recovery

## Problem description

This example takes the IEA-22 wind-turbine blade from its cross-sections to a spanwise beam table
and back to the stress in the blade wall. Each span station is a 1-D shell structure gene (SG): the
section contour meshed with two-node wall elements, each carrying a laminate. The engine is
`opensg_shell`: the Reissner-Mindlin (RM) shell ring is homogenized to the Timoshenko $6\times6$ in
the VABS strain order $[\varepsilon_{11}\;\gamma_{12}\;\gamma_{13}\;\kappa_1\;\kappa_2\;\kappa_3]$,
whose diagonal is $[EA\;GA_2\;GA_3\;GJ\;EI_2\;EI_3]$. Given the sectional forces of a beam run, a
two-step recovery returns the 3-D stress through each wall: the ring gives the plate forces of every
element, and the through-thickness RM plate SG of `opensg_solid.rm_plate_1D` gives $\sigma(z)$.

Four scripts in `examples/OpenSG_shell/IEA_blade_beam/` cover the chain:

| Script | Computes | Compared with |
|---|---|---|
| `1_homo_beam_props.py` | Timoshenko $6\times6$ of station `iea_s10` | |
| `2_dehom_v2_vabs.py` | wall stress at `iea_s10`, first and second order | VABS on the same section meshed as a 2-D solid |
| `3_q_consistency_check.py` | thickness integral of the recovered transverse shear | the wall's own shear resultant |
| `4_sweep_51_stations.py` | $6\times6$ and peak wall stress at all 51 stations | |

% source: examples/OpenSG_shell/IEA_blade_beam/iea_s10_shell.yaml; iea_s10_elem_plate_forces.dat (310 rows)
Station `iea_s10` lies at $r/R = 0.20$: a ring of 310 wall elements ({numref}`fig-iea-s10-ring`),
6 layups built from 6 materials. Lengths are in m and moduli in Pa, so the results are in SI.

```{figure} ../_static/shell_xsec_mesh.png
:width: 90%
:name: fig-iea-s10-ring

The `iea_s10` cross-section contour, elements coloured by layup.
```

% source: examples/OpenSG_shell/IEA_blade_beam/iea_s10_shell.yaml
| Name | Density | $E_1$ | $E_2$ | $E_3$ | $G_{12}$ | $G_{13}$ | $G_{23}$ | $\nu_{12}$ | $\nu_{13}$ | $\nu_{23}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| | kg/m³ | Pa | Pa | Pa | Pa | Pa | Pa | | | |
| gelcoat | 1235.0 | 3440000000.0 | 3440000000.0 | 3440000000.0 | 1323000000.0 | 1323000000.0 | 1323000000.0 | 0.3 | 0.3 | 0.3 |
| glass_triax | 1940.0 | 28211400000.0 | 16238800000.0 | 15835500000.0 | 8248220000.0 | 3491240000.0 | 3491240000.0 | 0.497511 | 0.18091 | 0.27481 |
| glass_uniax | 1940.0 | 43700000000.0 | 16500000000.0 | 15450000000.0 | 3265000000.0 | 3495000000.0 | 3480000000.0 | 0.262 | 0.264 | 0.35 |
| medium_density_foam | 130.0 | 142500000.0 | 142500000.0 | 142500000.0 | 54000000.0 | 54000000.0 | 54000000.0 | 0.3194 | 0.3194 | 0.3194 |
| carbon_uniax | 1600.0 | 140000000000.0 | 9100000000.0 | 9100000000.0 | 4131000000.0 | 4131000000.0 | 2689000000.0 | 0.3133 | 0.3133 | 0.47066 |
| glass_biax | 1940.0 | 11023100000.0 | 11023100000.0 | 16047700000.0 | 13231400000.0 | 3487480000.0 | 3487480000.0 | 0.688074 | 0.117173 | 0.117173 |

Layups list the outer ply first as [material, ply thickness in m]; every ply angle is 0.0:

% source: examples/OpenSG_shell/IEA_blade_beam/iea_s10_shell.yaml
| Layup | Ply 1 (outer) | Ply 2 | Ply 3 | Ply 4 |
|---|---|---|---|---|
| layup_0 | gelcoat, 0.0005 | glass_triax, 0.00300294 | glass_uniax, 0.02609467 | glass_triax, 0.00300294 |
| layup_1 | gelcoat, 0.0005 | glass_triax, 0.00300294 | medium_density_foam, 0.07060815 | glass_triax, 0.00300294 |
| layup_2 | gelcoat, 0.0005 | glass_triax, 0.00300294 | carbon_uniax, 0.08105991 | glass_triax, 0.00300294 |
| layup_3 | gelcoat, 0.0005 | glass_triax, 0.00300294 | glass_uniax, 0.02158284 | glass_triax, 0.00300294 |
| layup_4 | gelcoat, 0.0005 | glass_triax, 0.00300294 | carbon_uniax, 0.07707452 | glass_triax, 0.00300294 |
| layup_5 | glass_biax, 0.002 | medium_density_foam, 0.042 | glass_biax, 0.002 | |

The station yamls are produced from the windIO blade by OpenSG_io, and the sectional forces come
from a BeamDyn run; {doc}`blade_windio_pipeline` describes both hand-offs.

## Input

`iea_s10_shell.yaml` holds `nodes`, `elements`, `sets`, `sections`, `elementOrientations` and
`materials` (the format is explained in {doc}`../getting_started/tutorial`). Its last line,
`reference: center`, records that the contour is the laminate mid-surface.

% source: src/opensg_shell/sg_reference.py (reference_note); src/opensg_shell/cli.py (docstring)
The analysis ignores that key (`build_rm_bundle` never reads it; the command line only prints a
note, or a warning when it disagrees with the run): the reference is a run-time choice, `--center`
on the command line or `ref="center"` in `build_rm_bundle`.

`ff51_rmc_reform.dat` is the beam-force (FF) table, one row per station, forces in N and moments in
N·m, about the mid-surface reference axis and in VABS order:

% source: examples/OpenSG_shell/IEA_blade_beam/ff51_rmc_reform.dat (header and row 10)
```text
# eta F1 F2 F3 M1 M2 M3 (RM CENTER-ref BeamDyn, VABS order)
2.00000000e-01 4.19268867e+04 8.26739746e+03 1.46733512e+06 1.21914500e+06 -6.01849720e+07 3.04264594e+05
```

`ff51_rmc_reform.yaml` is the same table with its convention written next to the data
(`opensg_shell.helper.ff_to_yaml.convert` writes it, `station_ff` reads one station back):

% source: examples/OpenSG_shell/IEA_blade_beam/ff51_rmc_reform.yaml
```yaml
convention:
  order: [F1, F2, F3, M1, M2, M3]
  frame: RM center-ref
  units: {force: N, moment: N.m}
stations:
- row: 10
  eta: 0.2
  FF: [41926.8867, 8267.39746, 1467335.12, 1219145.0, -60184972.0, 304264.594]
```

The converter is two calls:

% source: docs/tutorials/windio_blade_pipeline.md (step 4); src/opensg_shell/helper/ff_to_yaml.py
```python
from opensg_shell.helper.ff_to_yaml import convert, station_ff

d = convert("ff51_rmc_reform.dat")                      # writes ff51_rmc_reform.yaml
FF, eta = station_ff("ff51_rmc_reform.yaml", eta=0.20)  # (6,) resultants, and 0.2
```

% source: docs/tutorials/windio_blade_pipeline.md (step 4); ff51_rmc_reform.dat row 0
At $\eta = 0$ the axial and in-plane entries print as $\sim10^{-6}$ N, a numerically zero root row,
not a physical load.

The user-input block at the top of each script, verbatim:

% source: examples/OpenSG_shell/IEA_blade_beam/1_homo_beam_props.py, 2_dehom_v2_vabs.py, 3_q_consistency_check.py, 4_sweep_51_stations.py
```python
# 1_homo_beam_props.py
name = "iea_s10"                  # IEA-22 station at r/R = 0.20

# 2_dehom_v2_vabs.py
name = "iea_s10"                     # IEA-22 station at r/R = 0.20
ff_file = "ff51_rmc_reform.dat"      # BeamDyn FF (RM center-ref, VABS order)
station_row = 10                     # row 10 -> eta = 0.20
vabs_sm = "iea_s10.sg.SM"            # the VABS 2-D solid reference cloud
# Through-thickness paths: two are GENERATED from the ring itself (clean
# single walls -- spar-cap centre and a web), so they are frame-consistent
# with the recovery and the VABS cloud by construction.  The station's own
# reference path is added when present; its printed nearest-VABS distance is
# the frame check (mm-level = same frame, 100s of mm = a foreign mesh frame).
jpath_file = "iea_s10.lp_sparcap_left_thickness.coords"

# 3_q_consistency_check.py
name = "iea_s10"
ff_file = "ff51_rmc_reform.dat"
station_row = 10                  # eta = r/R = 0.20
n_z = 41                          # integration points through each wall

# 4_sweep_51_stations.py
yaml_dir = os.environ.get(                       # the 51 station yamls (OpenSG_io output)
    "OPENSG_IEA51_DIR", os.path.join("data", "iea_all_stations", "shell51", "1d_yaml"))
ff_file = "ff51_rmc_reform.dat"
n_depth = 9                  # through-thickness recovery points per element
stations = range(51)         # s00 .. s50
```

`yaml_dir` is the folder holding `iea_s00_shell.yaml` to `iea_s50_shell.yaml`, the OpenSG_io
output; point it at your own copy.

% source: examples/OpenSG_shell/IEA_blade_beam/1_homo_beam_props.py, 2_dehom_v2_vabs.py, 3_q_consistency_check.py, 4_sweep_51_stations.py
:::{warning}
The shipped outputs are at the mid-surface reference: the header of `iea_s10_beam_Timo.out` ends
with `reference=center`. `build_rm_bundle` takes the reference as its `ref` argument, default
`"oml"`; the IEA stations need `ref="center"`. The four scripts call
`build_rm_bundle(name + "_shell.yaml")` (script 4: `build_rm_bundle(y)`) without `ref`, so they run
at the default `"oml"`; add `ref="center"` to that call to reproduce the shipped outputs.
:::

## Run

From the example folder:

```bash
python 1_homo_beam_props.py
python 2_dehom_v2_vabs.py
python 3_q_consistency_check.py
python 4_sweep_51_stations.py
```

The homogenization through the API, if you drive it yourself:

```python
import numpy as np
from opensg_shell import build_rm_bundle

B = build_rm_bundle("iea_s10_shell.yaml", ref="center")
C6 = np.asarray(B["Timo"])        # (6,6) Timoshenko stiffness
print(B["ref"], B["g_source"])    # 'center'  'msg'
```

The same mesh with the header `n_model: 1`, `refined: 1` is
the input of {doc}`shell_beam_cross_section`, where `opensg iea_s10_shell.yaml --center` writes an
`iea_s10_shell_Timo.out` with the same stiffness.

## Result

### Beam properties of station iea_s10

% source: examples/OpenSG_shell/IEA_blade_beam/iea_s10_beam_Timo.out
```{list-table} Timoshenko stiffness of iea_s10, reference=center ($EA$, $GA_2$, $GA_3$ in N; $GJ$, $EI_2$, $EI_3$ in N·m²)
:name: tab-iea-s10-timo
:header-rows: 1

* -
  - $\varepsilon_{11}$
  - $\gamma_{12}$
  - $\gamma_{13}$
  - $\kappa_1$
  - $\kappa_2$
  - $\kappa_3$
* - $\varepsilon_{11}$
  - 2.76605989e+10
  - -1.72306578e-05
  - 4.66504982e-06
  - 5.40202321e-06
  - 1.66864132e+09
  - -2.07730400e+09
* - $\gamma_{12}$
  - -1.72306578e-05
  - 7.18613262e+08
  - 5.75203697e+07
  - -1.01920441e+08
  - 2.30799528e-06
  - 5.27387488e-06
* - $\gamma_{13}$
  - 4.66504982e-06
  - 5.75203697e+07
  - 4.21744284e+08
  - -1.62178373e+08
  - 8.48167671e-06
  - 2.79313932e-07
* - $\kappa_1$
  - 5.40202321e-06
  - -1.01920441e+08
  - -1.62178373e+08
  - 2.43756534e+09
  - 1.64495570e-06
  - -8.11405526e-06
* - $\kappa_2$
  - 1.66864132e+09
  - 2.30799528e-06
  - 8.48167671e-06
  - 1.64495570e-06
  - 3.51438873e+10
  - -5.55712487e+09
* - $\kappa_3$
  - -2.07730400e+09
  - 5.27387488e-06
  - 2.79313932e-07
  - -8.11405526e-06
  - -5.55712487e+09
  - 6.81297238e+10
```

% source: examples/OpenSG_shell/IEA_blade_beam/iea_s10_shell_Timo.out
The extension-bending terms show that the tension centre is off the reference axis, the
flap-edge term that the bending axes are rotated, and the shear-twist terms that the shear centre is
off the axis too. The entries near $10^{-5}$ against diagonals of $10^{10}$ are numerically zero:
extension does not couple to shear or twist in this section. `iea_s10_shell_Timo.out` holds the same
matrix in the SwiftComp layout with its compliance and ` Time taken: 2.83 sec`;
`iea_s10_shell_ABDG.out` holds the $8\times8$ wall law of each layup.

### Recovery against VABS

% source: docs/tutorials/windio_blade_pipeline.md (step 5, gauss-point count)
`2_dehom_v2_vabs.py` recovers the stress at the 166 302 gauss points of the VABS reference cloud
`iea_s10.sg.SM` and differences it. Root-mean-square (rms) difference in MPa, web / skin:

% source: examples/OpenSG_shell/IEA_blade_beam/iea_s10_report.txt
| component | first order | second order (V2) | VABS rms |
|---|---|---|---|
| $\sigma_{11}$ | 43.073 / 9.529 | 43.073 / 9.529 | 44.440 / 96.676 |
| $\sigma_{22}$ | 0.612 / 0.482 | 0.612 / 0.482 | 0.704 / 0.998 |
| $\sigma_{12}$ | 6.799 / 1.255 | 6.798 / 1.255 | 22.026 / 6.009 |
| $\sigma_{33}$ | 0.081 / 0.049 | 0.081 / 0.049 | 0.081 / 0.049 |
| $\sigma_{13}$ | 0.394 / 0.192 | 0.394 / 0.192 | 0.394 / 0.193 |
| $\sigma_{23}$ | 0.035 / 0.028 | 0.035 / 0.028 | 0.035 / 0.028 |

% source: examples/OpenSG_shell/IEA_blade_beam/iea_s10_report.txt
The in-plane skin stress is good: $\sigma_{11}$ differs by 9.529 MPa rms against a VABS rms of
96.676 MPa, and by 1.265 MPa against 286.837 MPa on the station's own reference thickness path,
whose points sit 0.00 mm from the nearest VABS gauss point. The two paths built from the ring sit
2.44 mm and 2.43 mm off, and that offset dominates their larger differences (38.355 MPa on the spar
cap). The web axial stress is the weak spot: its difference, 43.073, is close to the VABS rms of
44.440. The transverse differences equal the VABS rms, the signature of a recovered field near zero.
The second-order rung carries no payload at this load state: it moves $\sigma_{11}$ by
1.950e-04 MPa rms against a first-order rms of 8.851e+01, $\sigma_{33}$ by 5.190e-05 MPa against a
VABS rms of 5.576e-02, and $\sigma_{13}$ and $\sigma_{23}$ by 0.000e+00. The first-order
$\sigma_{33}$ rms is 3.626e-13 MPa, machine zero.

### Q-consistency of the transverse shear

The thickness integral of a recovered transverse shear must equal the wall's shear resultant,
$I_a = \int_h \sigma_{a3}\,\mathrm{d}z = Q_a = (G_{\rm msg}\,s_2)_a$ with
$s_2 = [2\gamma_{13}\;2\gamma_{23}]$; the ratio $I/Q = 1$ is consistent.

% source: examples/OpenSG_shell/IEA_blade_beam/iea_s10_q_consistency.txt
| ratio | median | mean | min | max | verdict |
|---|---|---|---|---|---|
| $I_{13}/Q_1$ | 5.228e-01 | -6.972e+00 | -1.299e+03 | 2.260e+01 | target invalid |
| $I_{23}/Q_2$ | 9.895e-01 | 1.096e+00 | -1.217e+01 | 1.194e+01 | target valid |

% source: examples/OpenSG_shell/IEA_blade_beam/iea_s10_q_consistency.txt
$\sigma_{23}$ passes with a 1.05 % Q-consistency error (wall $Q_2$ rms 8.9686 N/m, recovered
$I_{23}$ rms 8.3085 N/m). $\sigma_{13}$ is not measurable in a cross-section model: the ring is
homogenized with `shear="mitc4_g23"` (MITC: mixed interpolation of tensorial components), $\gamma_{13}$ stays untied and carries the flat-wall drilling
mode, so $Q_1$ (rms 9.9064e+02 N/m, about 110 times $Q_2$) is not a physical wall shear. On this
one-quad-deep prismatic strip the $\gamma_{13}$ tying reproduces the raw row exactly: the report
measures `rms |Q1(g23) - Q1(both)| = 0.0000e+00 N/m`, so `mitc4_g23` and `mitc4_both` give the same
result here. On a genuine surface quad, the 3-D segment of {doc}`shell_taper_segment` or a 3-D
shell SG, the $\gamma_{13}$ row is not linear along the axis, and that is where a physical
$\gamma_{13}$ is obtained, not by rescaling this ring against $Q_1$.

:::{note}
Trust $\sigma_{11}$, $\sigma_{22}$ and $\sigma_{12}$ from this pipeline, and $\sigma_{23}$ to about
one percent in amplitude. Do not use the cross-section route's $\sigma_{13}$ or $\sigma_{33}$ as a
design quantity.
:::

### The 51-station sweep

% source: docs/tutorials/windio_blade_pipeline.md (step 3); 4_sweep_51_stations.py
All 51 stations complete; a station whose yaml is missing is skipped with a printed message, and a
station that raises is caught and reported, so one bad section cannot stop the sweep;
`iea51_beam_props.dat` holds one row per station, $\eta = r/R$ from 0.00 to 1.00 in steps of 0.02,
followed by the six diagonal terms. Selected rows:

% source: examples/OpenSG_shell/IEA_blade_beam/iea51_beam_props.dat
| $\eta$ | $EA$ [N] | $GA_2$ [N] | $GA_3$ [N] | $GJ$ [N·m²] | $EI_2$ [N·m²] | $EI_3$ [N·m²] |
|---|---|---|---|---|---|---|
| 0.00 | 6.44275512e+10 | 7.96306621e+09 | 8.36135930e+09 | 1.27413308e+11 | 2.97295270e+11 | 2.19213722e+11 |
| 0.20 | 2.76605989e+10 | 7.18613262e+08 | 4.21744284e+08 | 2.43756534e+09 | 3.51438873e+10 | 6.81297238e+10 |
| 0.50 | 2.15596413e+10 | 5.34263266e+08 | 2.02128533e+08 | 4.95011412e+08 | 7.26524060e+09 | 1.82791086e+10 |
| 0.80 | 9.28038409e+09 | 3.90243799e+08 | 7.82870454e+07 | 6.27349440e+07 | 5.76401363e+08 | 2.07536066e+09 |
| 1.00 | 2.65581077e+08 | 1.56743220e+07 | 1.88135914e+06 | 5.15174420e+04 | 3.04364786e+05 | 1.60428557e+06 |

The $\eta = 0.20$ row is the diagonal of the single-station run, digit for digit. The table holds the
diagonal only; a beam model needs the full $6\times6$ of each station's `_Timo.out`.

```{figure} ../_static/iea51_beam_props.png
:width: 95%
:name: fig-iea51-beam-props

Spanwise Timoshenko diagonal of the IEA-22 blade, six log-scale panels against $r/R$.
```

% source: docs/tutorials/windio_blade_pipeline.md (step 3); GJ values from iea51_beam_props.dat
The root stiffness drops by an order of magnitude in the first 10 % of span as the circular root
becomes an aerofoil, and $GJ$ falls from 1.27413308e+11 to 5.15174420e+04 N·m² along the blade.
The sweep also recovers the stress at `n_depth = 9` depths per wall element, in the ply frame, and
records the peak $|\sigma|$ per component over the section in `iea51_peak_stress.dat` (MPa):

% source: examples/OpenSG_shell/IEA_blade_beam/iea51_peak_stress.dat
| $\eta$ | $\sigma_{11}$ | $\sigma_{22}$ | $\sigma_{33}$ | $\sigma_{23}$ | $\sigma_{13}$ | $\sigma_{12}$ |
|---|---|---|---|---|---|---|
| 0.00 | 1.49271417e+02 | 1.53636678e+00 | 4.13972884e-13 | 5.39458803e-03 | 4.12768831e-02 | 2.24186053e+00 |
| 0.16 | 3.25793641e+02 | 4.14543750e+00 | 8.98726285e-13 | 3.91335971e-02 | 3.51941097e-02 | 7.01052829e+00 |
| 0.20 | 3.18177433e+02 | 3.08795353e+00 | 9.13627446e-13 | 1.57149557e-02 | 3.45770096e-02 | 5.66031219e+00 |
| 0.54 | 2.14382707e+02 | 1.62953363e+00 | 8.16304237e-13 | 1.74810053e-02 | 8.23774603e-02 | 5.64234617e+01 |
| 1.00 | 4.88165692e-01 | 2.71026227e-02 | 2.03726813e-15 | 4.70632777e-05 | 5.13380478e-02 | 7.89278421e+00 |

% source: docs/tutorials/windio_blade_pipeline.md ("The same recovery, all 51 stations")
The peak axial stress of the blade in this load case is 325.79 MPa at $\eta = 0.16$; the peak
in-plane shear is 56.42 MPa at $\eta = 0.54$, where the shear webs take over from the caps. The
$\sigma_{33}$ column is round-off, the first-order limit noted above.

```{figure} ../_static/iea51_sweep.png
:width: 95%
:name: fig-iea51-sweep

Left: peak recovered $\sigma_{11}$ and $\sigma_{12}$ along the blade. Right: wall time per station for homogenization and dehomogenization.
```

% source: docs/tutorials/windio_blade_pipeline.md ("The same recovery, all 51 stations")
The axial peak climbs out of the root, crests near $\eta \approx 0.16$ and decays to the tip; the
shear peak rises in two steps, near $\eta \approx 0.35$ and $\eta \approx 0.46$. Both steps of the
analysis take about one second per station; the first station adds a few seconds of compilation.

## Files

| File | Content |
|---|---|
| `iea_s10_shell.yaml` | the 1-D shell SG of station `iea_s10` |
| `ff51_rmc_reform.dat`, `ff51_rmc_reform.yaml` | beam forces at the 51 stations |
| `iea_s10.sg.SM`, `iea_s10.lp_sparcap_left_thickness.coords` | VABS stress cloud and reference thickness path |
| `iea_s10_shell_Timo.out`, `iea_s10_shell_ABDG.out`, `abd/iea_s10_shell_abd.yaml` | beam law, wall laws and the per-station ABD yaml (`build_rm_bundle`) |
| `iea_s10_beam_Timo.out` | the bare $6\times6$ with its strain order and reference (script 1) |
| `iea_s10_report.txt`, `iea_s10_elem_plate_forces.dat` | VABS comparison; 310 rows: `elem`, `y2 y3`, the six shell strains, and the plate forces $N_{11}$ $N_{22}$ $N_{12}$ [N/m], $M_{11}$ $M_{22}$ $M_{12}$ [N] (script 2) |
| `iea_s10_q_consistency.txt` | Q-consistency report (script 3) |
| `iea51_beam_props.dat`, `iea51_peak_stress.dat`, `iea51_sweep.npz` | spanwise diagonal, peak stress, and the arrays `eta`, `props`, `peak`, `t_homo`, `t_dehom`, `stations` (script 4) |

The scripts also write figures and the full stress cloud (`iea_s10_ring_mesh.png`,
`iea_s10_dehom_v2.txt` and `.npz`, `iea_s10_v2_section.png`, `iea_s10_parity.png`,
`iea_s10_thickness_paths.png`, `iea_s10_q_consistency.png`, `iea51_beam_props.png`,
`iea51_sweep.png`), which are not shipped in the folder.

% source: docs/tutorials/windio_blade_pipeline.md (step 5 outputs); 2_dehom_v2_vabs.py
`iea_s10_dehom_v2.txt` has 166 302 rows, one per VABS gauss point (`y2 y3 elem z`, then the six
recovered V2 stresses and the six VABS stresses, all MPa), and `iea_s10_dehom_v2.npz` adds `C6`,
`FF`, `s6mid`, `F6mid` and the strain gradients.
