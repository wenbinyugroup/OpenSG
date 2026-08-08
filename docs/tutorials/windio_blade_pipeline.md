# Full Blade Pipeline: windIO to Beam Properties, BeamDyn and Recovery

A wind-turbine blade is designed as a 3-D laminated shell but analysed, aeroelastically, as a
1-D beam. This page walks the whole round trip on the IEA-22 blade: a windIO blade description
becomes one 1-D shell structure gene per span station, each station is homogenized to a
Timoshenko $6\times6$, the spanwise table of those matrices is what a 1-D beam solver
(BeamDyn/OpenFAST) needs, the beam run returns sectional forces station by station, and those
forces drive the dehomogenization back down to 3-D stress in the wall.

Three separate tools are involved, and the boundary between them matters:

| stage | tool | who runs it |
|---|---|---|
| windIO blade → per-station 1-D shell SG yaml | **OpenSG_io** (separate repository) | you, once per blade |
| 1-D shell SG → Timoshenko $6\times6$ | **OpenSG-2.0**, `opensg_shell` | `1_homo_beam_props.py`, `4_sweep_51_stations.py` |
| spanwise $6\times6$ table → blade dynamics, sectional loads | **BeamDyn / OpenFAST** | you, outside OpenSG |
| sectional loads + $6\times6$ → 3-D wall stress | **OpenSG-2.0**, `opensg_shell` + `opensg_solid.rm_plate_1D` | `2_dehom_v2_vabs.py`, `4_sweep_51_stations.py` |
| correctness check on the recovered transverse shear | **OpenSG-2.0** | `3_q_consistency_check.py` |

OpenSG never runs the beam dynamics. It produces the sectional constitutive law that the beam
solver needs, and it consumes the sectional loads that the beam solver returns.

```{mermaid}
flowchart TD
    W["windIO blade yaml<br/>(IEA-22, v1 or v2)"] --> IO["OpenSG_io<br/>build_cross_section / emit_opensg_yaml"]
    IO --> Y["iea_s00_shell.yaml .. iea_s50_shell.yaml<br/>1-D shell SG per station"]
    IO --> PV["emit_prevabs -> PreVABS XML<br/>2-D solid route (VABS reference)"]
    Y --> H["opensg_shell.build_rm_bundle<br/>RM ring homogenization"]
    H --> T["Timoshenko 6x6 per station<br/>&lt;yaml&gt;_Timo.out + iea51_beam_props.dat"]
    T --> BD["BeamDyn / OpenFAST<br/>(run by the user)"]
    BD --> FF["ff51_rmc_reform.dat<br/>eta F1 F2 F3 M1 M2 M3"]
    FF --> D["two-step dehom<br/>_macro_fields -> _rm_shell_strain -> rm_plate_msg"]
    H --> D
    D --> S["3-D wall stress per station<br/>iea51_peak_stress.dat, &lt;name&gt;_dehom_v2.txt"]
    D --> Q["3_q_consistency_check.py<br/>I = integral(sigma_a3 dz) vs Q = G_msg s2"]
```

## The worked example

Everything below runs in `examples/OpenSG_shell/IEA_blade_beam/`. The single-station scripts use
`iea_s10`, the IEA-22 station at $r/R = 0.20$: a ring of **310 wall elements**, **6 layups**
(`layup_0` … `layup_5`) built from **6 materials** (`gelcoat`, `glass_triax`, `glass_uniax`,
`medium_density_foam`, `carbon_uniax`, `glass_biax`). The sweep script runs all 51 stations.
Run the scripts from inside that directory; the package holds no path assumptions, the example
owns every path.

## 1. windIO blade → per-station 1-D shell SG yamls

`OpenSG_io` (repository <https://github.com/bagla0/OpenSG_io>, documentation
<https://bagla0.github.io/OpenSG_io/>) is the input layer. It reads a windIO blade
description — auto-detecting windIO v1 and v2 — interpolates the outer shape, the layup stacks
and the material database to a requested span fraction, and writes the cross-section in the two
forms an MSG analysis can use: the **1-D shell SG yaml** that `opensg_shell` reads directly, and
a **PreVABS XML** that meshes to a 2-D solid cross-section (the route that produces the VABS
reference used in step 5). It also reads PreVABS XML and OpenFAST ElastoDyn/BeamDyn blade data,
so published beam properties can be pulled in as a validation reference.

Install it alongside OpenSG-2.0:

```bash
git clone --recurse-submodules https://github.com/bagla0/OpenSG_io
```

```bash
pip install -r requirements.txt
```

That file lists numpy, scipy, pyyaml, meshio, gmsh, tetgen and pyvista. A windIO **v2** blade —
the IEA-22 file used below is one — additionally needs the `windIO` package itself, which the
converter imports only on the v2 branch; a v1 blade needs nothing beyond the requirements file.

```bash
python scripts/fetch_prevabs.py
```

The command line writes a set of stations in one call:

```bash
python scripts/opensg_io.py BAR_URC.yaml out --name bar --stations 0.3 0.5 0.7 --solid
```

The Python API is what you script a 51-station blade with:

```python
from opensg_io import load_blade, build_cross_section, emit_opensg_yaml, emit_prevabs

blade = load_blade("IEA-22-280-RWT.yaml")      # windIO v1 or v2, auto-detected
cs = build_cross_section(blade, r=0.5)         # one span fraction
emit_opensg_yaml(cs, "shell_r050.yaml")        # the 1-D shell SG for opensg_shell
emit_prevabs(cs, "prevabs_r050", name="r050")  # PreVABS XML -> 2-D solid route
```

### What the 1-D shell SG yaml contains

`iea_s10_shell.yaml` is the reference for the format. Its top-level keys, in file order:

| key | content |
|---|---|
| `nodes` | contour node coordinates `[y2 y3 0]`, space-separated inside the brackets, in metres, one per line: the first two entries are the in-plane cross-section coordinates and the third is the beam axis, identically 0 (`sg_mesh.load_ring_ref` returns `ax = 2`, `cross = [0, 1]`) |
| `elements` | two-node line segments along the section contour |
| `sets` | `element` sets, one per layup, holding the element labels |
| `sections` | `type: shell`, `elementSet: layup_k`, and the `layup` as `[material, thickness (m), angle (deg)]` triples, outermost ply first |
| `elementOrientations` | one $3\times3$ frame per element, flattened; the first row is $e_1$, out of the section plane (the beam axis) |
| `materials` | `density` (kg/m³) and orthotropic `elastic: {E, G, nu}` triples, in Pa |
| `reference` | `center` — which surface the section geometry refers to |

The `reference` field is load-bearing and is the single source of truth: `build_rm_bundle` reads
it when no override is passed, and it fixes the through-thickness offset `frac` used by the ring
laminate law, the plate SG's `z_ref`, the emitted ABD and the recovery depth conversion alike.

| `reference` | `frac` | meaning |
|---|---|---|
| `center` | 0.5 | mid-surface (what the IEA stations use) |
| `oml` | 0.0 | outer mould line |
| `oml_flip` | 1.0 | outer mould line, reversed stacking |
| `iml` | 1.0 | inner mould line |

## 2. Each station → a Timoshenko $6\times6$

**What it computes.** The RM shell ring homogenization: a 6-DOF-per-node shell element with an
element-wise drilling Lagrange multiplier, $\gamma_{23}$ tied by MITC, and the MSG (Yu-2002
least-squares) wall transverse-shear law $G$. The ring warping problem is solved for the four
Euler–Bernoulli modes $V_0$ and the Timoshenko correction $V_1$, and the energy is condensed to
the beam $6\times6$ in the VABS strain order
$[\,\varepsilon_{11}\;\gamma_{12}\;\gamma_{13}\;\kappa_1\;\kappa_2\;\kappa_3\,]$, whose diagonal
is $[\,EA\;\;GA_2\;\;GA_3\;\;GJ\;\;EI_2\;\;EI_3\,]$.

**What you edit.** One line at the top of `1_homo_beam_props.py`:

```python
name = "iea_s10"                  # IEA-22 station at r/R = 0.20
```

**Command.**

```bash
python 1_homo_beam_props.py
```

**The API**, if you drive it yourself:

```python
import numpy as np
from opensg_shell import build_rm_bundle

B = build_rm_bundle("iea_s10_shell.yaml")
C6 = np.asarray(B["Timo"])        # (6,6) Timoshenko stiffness
print(B["ref"], B["g_source"])    # 'center'  'msg'
```

`build_rm_bundle(shell_yaml, ref=None, shear="mitc4_g23", g_source="msg")` returns a bundle, not
just a matrix: `Timo`, the warping modes `V0`/`V1`, the ring geometry (`corners`, `red_cells`,
`re3`, `k22`, `strip`), `layup_per_elem`, the geometry-free `layup_db`/`material_db`, and
`frac`/`ref`/`g_source`. Step 5 consumes all of it — do not throw the bundle away.

**What comes out.**

| file | written by | content |
|---|---|---|
| `iea_s10_shell_Timo.out` | `build_rm_bundle` | the timed SwiftComp `.K` layout: banner ` OpenSG msg-shell beam model [ext sh2 sh3 twist bend2 bend3]`, `The Effective Timoshenko Stiffness Matrix`, the effective compliance, and a `Time taken` footer |
| `iea_s10_shell_ABDG.out` | `build_rm_bundle` | per section, the wall law: `The Effective Reissner-Mindlin Plate Stiffness Matrix` (the $8\times8$ ABD plus the MSG $\mathbf{G}$) and its compliance |
| `abd/iea_s10_shell_abd.yaml` | `build_rm_bundle` | the per-station ABD yaml at the same reference, emitted once and cached; carries `mass_per_area` per section |
| `iea_s10_beam_Timo.out` | the script | the bare $6\times6$ as text, with the strain order and reference in the header |
| `iea_s10_ring_mesh.png` | the script | the real computed ring, elements coloured by layup |

**What the numbers mean.** The matrix from `iea_s10_beam_Timo.out`, in SI (the yaml is metres and
pascals, so $EA$, $GA_2$, $GA_3$ are in N and $GJ$, $EI_2$, $EI_3$ in N·m²):

| | $\varepsilon_{11}$ | $\gamma_{12}$ | $\gamma_{13}$ | $\kappa_1$ | $\kappa_2$ | $\kappa_3$ |
|---|---|---|---|---|---|---|
| $\varepsilon_{11}$ | 2.766060e+10 | −1.723e−05 | 4.665e−06 | 5.402e−06 | 1.668641e+09 | −2.077304e+09 |
| $\gamma_{12}$ | −1.723e−05 | 7.186133e+08 | 5.752037e+07 | −1.019204e+08 | 2.308e−06 | 5.274e−06 |
| $\gamma_{13}$ | 4.665e−06 | 5.752037e+07 | 4.217443e+08 | −1.621784e+08 | 8.482e−06 | 2.793e−07 |
| $\kappa_1$ | 5.402e−06 | −1.019204e+08 | −1.621784e+08 | 2.437565e+09 | 1.645e−06 | −8.114e−06 |
| $\kappa_2$ | 1.668641e+09 | 2.308e−06 | 8.482e−06 | 1.645e−06 | 3.514389e+10 | −5.557125e+09 |
| $\kappa_3$ | −2.077304e+09 | 5.274e−06 | 2.793e−07 | −8.114e−06 | −5.557125e+09 | 6.812972e+10 |

Read it as physics rather than as six numbers. The extension–bending entries
$1.67\times10^{9}$ and $-2.08\times10^{9}$ say the section's tension centre is not at the
reference axis; $-5.56\times10^{9}$ couples flap and edge bending, so the bending axes are not
the reference axes; the shear–twist entries $-1.02\times10^{8}$ and $-1.62\times10^{8}$ locate
the shear centre away from it too. The entries printed at $10^{-5}$ against diagonals of
$10^{10}$ are numerically zero: extension does not couple to transverse shear or twist in this
section, and the solver reproduces that to fifteen orders of magnitude.

## 3. The spanwise table, and what BeamDyn does with it

`4_sweep_51_stations.py` repeats step 2 for every station of the blade (and, in the same loop,
the recovery of step 5). What you edit:

```python
yaml_dir = os.path.expanduser(
    "~/OpenSG-TW-claude/examples/data/iea_all_stations/shell51/1d_yaml")
ff_file = "ff51_rmc_reform.dat"
n_depth = 9                  # through-thickness recovery points per element
stations = range(51)         # s00 .. s50
```

`yaml_dir` is where `iea_s00_shell.yaml … iea_s50_shell.yaml` live — the OpenSG_io output of
step 1. Point it at your own directory.

```bash
python 4_sweep_51_stations.py
```

A station whose yaml is missing is skipped with a printed message, and any station that raises
is caught and reported so one bad section cannot kill the sweep. In the shipped run, **49 of the
51 stations completed**: `iea51_beam_props.dat` and `iea51_peak_stress.dat` carry
$\eta = 0.00\ldots0.94$ in steps of $0.02$ plus $\eta = 1.00$; $\eta = 0.96$ and $0.98$ are
absent.

`iea51_beam_props.dat` holds one row per station, `eta` followed by the six diagonal terms.
Selected rows, verbatim:

| $\eta = r/R$ | $EA$ [N] | $GA_2$ [N] | $GA_3$ [N] | $GJ$ [N·m²] | $EI_2$ [N·m²] | $EI_3$ [N·m²] |
|---|---|---|---|---|---|---|
| 0.00 | 6.44275512e+10 | 7.96306621e+09 | 8.36135930e+09 | 1.27413308e+11 | 2.97295270e+11 | 2.19213722e+11 |
| 0.20 | 2.76605989e+10 | 7.18613262e+08 | 4.21744284e+08 | 2.43756534e+09 | 3.51438873e+10 | 6.81297238e+10 |
| 0.50 | 2.15596413e+10 | 5.34263266e+08 | 2.02128533e+08 | 4.95011412e+08 | 7.26524060e+09 | 1.82791086e+10 |
| 0.80 | 9.28038409e+09 | 3.90243799e+08 | 7.82870454e+07 | 6.27349440e+07 | 5.76401363e+08 | 2.07536066e+09 |
| 1.00 | 2.65581077e+08 | 1.56743220e+07 | 1.88135914e+06 | 5.15174420e+04 | 3.04364786e+05 | 1.60428557e+06 |

The $\eta = 0.20$ row is digit-for-digit the diagonal of the standalone `iea_s10` run in step 2 —
the sweep and the single-station script are the same computation.

![Spanwise Timoshenko diagonal of the IEA-22 blade, six log-scale panels against r/R](../_static/iea51_beam_props.png)

The figure is `iea51_beam_props.png`: each of the six diagonal terms on a log axis against
$r/R$. The root region collapses by an order of magnitude in the first 10 % of span as the
circular root transitions to an aerofoil, the mid-span decays smoothly as the section thins, and
the tip drops off a cliff — $GJ$ falls from $1.27\times10^{11}$ to $5.15\times10^{4}$ N·m²
across the blade, six and a half decades.

### Handing the table to BeamDyn

Per the OpenFAST/BeamDyn documentation, BeamDyn (the geometrically exact beam module of
OpenFAST) represents the blade by a sectional $6\times6$ **stiffness** matrix and a sectional
$6\times6$ **mass** matrix given at a list of normalized span stations in its blade input file.
Nothing in OpenSG-2.0 reads or writes that file; the implementation to look at is OpenSG_io's
`openfast_io.read_beamdyn_blade` and `openfast_io.write_beamdyn_blade`. Two practical points:

- **The station list must line up with BeamDyn's quadrature points.** The BeamDyn documentation
  states that with trapezoidal quadrature (`quadrature = 2`) the quadrature points are the
  stations you supply, so a 51-station OpenSG table maps one-to-one onto a 51-station BeamDyn
  blade file and no interpolation of the $6\times6$ is needed.
- **Use the full matrix, not the table's diagonal.** `iea51_beam_props.dat` records only the six
  diagonal terms, for plotting and inspection. The couplings shown in step 2 are precisely what a
  composite blade beam model exists to carry, so take the full $6\times6$ from each station's
  `<station>_Timo.out` (or from `B["Timo"]` in your own driver).

Be clear about what OpenSG supplies here: **the stiffness matrix only**. These scripts compute no
sectional mass matrix; the only mass quantity produced is `mass_per_area` per laminate in the
emitted ABD yaml. The sectional inertia and the aeroelastic run itself are yours, outside
OpenSG-2.0. The BeamDyn input assembly need not be done by hand, though: OpenSG_io ships
`openfast_io.timo_to_beamdyn`, which permutes an OpenSG Timoshenko $6\times6$ into BeamDyn DOF
order, and `openfast_io.write_beamdyn_blade`, which writes a BeamDyn blade input from a list of
etas and homogenized stiffness matrices. Its `M_list` argument is optional and zero-filled when
omitted, so supplying the sectional mass matrices remains your responsibility.

## 4. The loads come back: the FF table

A BeamDyn or GEBT run returns, at every station, the sectional force and moment resultants the
blade actually carries in a given load case. That table is the input to recovery.
`ff51_rmc_reform.dat` is such a table, 51 rows, one per station:

```
# eta F1 F2 F3 M1 M2 M3 (RM CENTER-ref BeamDyn, VABS order)
2.00000000e-01 4.19268867e+04 8.26739746e+03 1.46733512e+06 1.21914500e+06 -6.01849720e+07 3.04264594e+05
```

Forces in N, moments in N·m, `eta` = $r/R$ from 0.00 to 1.00 in steps of 0.02. The frame and the
component order are not decoration: the resultants must be expressed about the same reference
axis and in the same order as the $6\times6$ that produced them, or the round trip does not
close. Here both are `RM center-ref` and VABS order, matching `reference: center` in the station
yamls. At $\eta = 0$ the axial and in-plane entries print as $\sim10^{-6}$ N — a numerically
zero root row, not a physical load.

A converter turns the table into the annotated yaml form and pulls one station out of it:

```python
from opensg_shell.helper.ff_to_yaml import convert, station_ff

d = convert("ff51_rmc_reform.dat")                      # writes ff51_rmc_reform.yaml
FF, eta = station_ff("ff51_rmc_reform.yaml", eta=0.20)  # (6,) resultants, and 0.2
```

`ff51_rmc_reform.yaml` records the convention explicitly next to the data, which is what makes
the file self-describing months later:

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

## 5. Dehomogenization: from beam forces back to 3-D wall stress

**What it computes.** A two-step chain, because a blade wall is a laminate inside a section:

$$\text{beam } FF \;\xrightarrow[\text{RM shell ring}]{}\; \text{per-element plate forces}
\;\xrightarrow[\text{RM plate SG}]{}\; \sigma(z)\ \text{through the wall.}$$

*Step 2a, the section* (`opensg_shell`). The macro beam strain is $\bar\varepsilon = C_6^{-1} FF$;
`_macro_fields` recombines the RM warping from it, and `_rm_shell_strain` evaluates, per ring
element, the six shell strains
$s_6 = [\,\varepsilon_{11}\;\varepsilon_{22}\;2\varepsilon_{12}\;\kappa_{11}\;\kappa_{22}\;
2\kappa_{12}\,]$. That wall's own MSG ABD turns them into the **plate reaction forces**
$F_6 = A_6 s_6 = [\,N_{11}\;N_{22}\;N_{12}\;M_{11}\;M_{22}\;M_{12}\,]$ — the interface quantity
handed down to the plate SG.

*Step 2b, the wall* (`opensg_solid.rm_plate_1D`). `rm_plate_msg` builds the through-thickness
MSG-RM structure gene for each layup once, and `msgrm_strain_at_depth` (or its vectorized twin
`msgrm_strain_at_depth_batch`) evaluates the 3-D strain and stress at depth $z$, by default
rotated into the ply material frame where failure criteria live.

Recovery needs strain **gradients**, and they come from the ring itself: the span gradient
$\partial E/\partial x_1$ from the macro recovery ladder (`BDe @ st_cl1 + BDh @ aB`), and the arc
gradient $\partial E/\partial x_2$ by nodal differencing of $s_6$ between the two elements
sharing a node — never across a layup change, because that jump is physical. Passing the second
gradients as well selects the second-order (V2) recovery, which is the rung that can carry the
transverse pair $\sigma_{33}$, $\sigma_{13}$ that first order cannot.

**What you edit** in `2_dehom_v2_vabs.py`:

```python
name = "iea_s10"                     # IEA-22 station at r/R = 0.20
ff_file = "ff51_rmc_reform.dat"      # BeamDyn FF (RM center-ref, VABS order)
station_row = 10                     # row 10 -> eta = 0.20
vabs_sm = "iea_s10.sg.SM"            # the VABS 2-D solid reference cloud
jpath_file = "iea_s10.lp_sparcap_left_thickness.coords"
```

```bash
python 2_dehom_v2_vabs.py
```

**What comes out.**

| file | content |
|---|---|
| `iea_s10_report.txt` | the validation report: FF, the Timo diagonal, rms differences against VABS split web/skin, the V2 payload, and the through-thickness path tables |
| `iea_s10_elem_plate_forces.dat` | 310 rows, one per ring element: `y2 y3`, the six shell strains, and the six plate reaction forces $N_{11} N_{22} N_{12}$ [N/m], $M_{11} M_{22} M_{12}$ [N] |
| `iea_s10_dehom_v2.txt` | 166 302 rows, one per VABS gauss point: `y2 y3 elem z`, then the six recovered V2 stresses and the six VABS stresses, all MPa |
| `iea_s10_dehom_v2.npz` | the same fields plus `C6`, `FF`, `s6mid`, `F6mid` and every strain gradient, for re-plotting without re-solving |
| `iea_s10_v2_section.png` | section contours, VABS beside RM V2, for $\sigma_{11}$, $\sigma_{12}$, $\sigma_{13}$, $\sigma_{33}$ |
| `iea_s10_parity.png` | parity clouds, first order and V2 against VABS |
| `iea_s10_thickness_paths.png` | through-thickness profiles on a spar-cap wall, a web, and the station's own reference path |

**What it was validated against.** `iea_s10.sg.SM` is the VABS stress cloud of the *same*
cross-section meshed as a 2-D solid — the independent reference, 166 302 gauss points. The
recovery is projected onto it and differenced. From `iea_s10_report.txt`, rms difference in MPa,
web / skin, with the VABS rms alongside so each number has a scale:

| component | first order (web / skin) | V2 (web / skin) | VABS rms (web / skin) |
|---|---|---|---|
| $\sigma_{11}$ | 43.073 / 9.529 | 43.073 / 9.529 | 44.440 / 96.676 |
| $\sigma_{22}$ | 0.612 / 0.482 | 0.612 / 0.482 | 0.704 / 0.998 |
| $\sigma_{12}$ | 6.799 / 1.255 | 6.798 / 1.255 | 22.026 / 6.009 |
| $\sigma_{33}$ | 0.081 / 0.049 | 0.081 / 0.049 | 0.081 / 0.049 |
| $\sigma_{13}$ | 0.394 / 0.192 | 0.394 / 0.192 | 0.394 / 0.193 |
| $\sigma_{23}$ | 0.035 / 0.028 | 0.035 / 0.028 | 0.035 / 0.028 |

Read this honestly, because the table says three different things at once.

*The in-plane skin stress is good.* $\sigma_{11}$ in the skin differs by 9.53 MPa rms against a
VABS rms of 96.68 MPa, and $\sigma_{12}$ by 1.26 against 6.01. On the station's own reference
thickness path — the one whose points sit **0.00 mm** from the nearest VABS gauss point, i.e.
exactly in the reference frame — the $\sigma_{11}$ difference is **1.265 MPa against a VABS rms
of 286.837 MPa**, better than half a percent. The two paths generated from the ring geometry sit
2.4 mm from the nearest VABS point, and their larger differences (38.355 MPa rms on the spar cap)
are dominated by that offset rather than by the recovery.

*The web is the weak spot.* For $\sigma_{11}$ in the web the difference, 43.07, is essentially
the VABS rms itself, 44.44 — the web axial stress is not being reproduced. The same is true of
the three transverse components everywhere: the difference equals the VABS rms to three digits,
which is what you get when the recovered field is (near) zero. This is the known transverse
recovery gap of the shell-to-plate route, not a tuning issue.

*The second-order rung is inert at this load state.* The report's V2 payload block measures what
V2 changes rather than a rounded difference: $\sigma_{11}$ moves by 1.950e−04 MPa rms against a
first-order rms of 8.851e+01, $\sigma_{33}$ by 5.190e−05 MPa against a VABS rms of 5.576e−02,
and $\sigma_{13}$ and $\sigma_{23}$ by exactly 0.000e+00. For this constant-force blade state
the V2 drivers carry no payload; the first-order $\sigma_{33}$ rms is 3.603e−13 MPa, machine
zero.

### The same recovery, all 51 stations

`4_sweep_51_stations.py` runs the first-order recovery at `n_depth = 9` depths per wall element
(mid-bin, $\zeta = (k+\tfrac12)/9$, converted to $z = (\zeta - \texttt{frac})\,h$) and records
the peak $|\sigma|$ per component over the whole section, in the ply frame, into
`iea51_peak_stress.dat`:

| $\eta = r/R$ | $\sigma_{11}$ | $\sigma_{22}$ | $\sigma_{33}$ | $\sigma_{23}$ | $\sigma_{13}$ | $\sigma_{12}$ |
|---|---|---|---|---|---|---|
| 0.00 | 1.49271417e+02 | 1.53636678e+00 | 4.13972884e−13 | 5.39458803e−03 | 4.12768841e−02 | 2.24186054e+00 |
| 0.16 | 3.25793641e+02 | 4.14543750e+00 | 8.97794962e−13 | 3.91335971e−02 | 3.51941097e−02 | 7.01052831e+00 |
| 0.20 | 3.18177433e+02 | 3.08795353e+00 | 9.12696123e−13 | 1.57149557e−02 | 3.45770096e−02 | 5.66031217e+00 |
| 0.54 | 2.14382707e+02 | 1.62953363e+00 | 8.13510269e−13 | 1.74810053e−02 | 8.23774625e−02 | 5.64234617e+01 |
| 1.00 | 4.88165692e−01 | 2.71026227e−02 | 2.03726813e−15 | 4.70632777e−05 | 5.13380478e−02 | 7.89278421e+00 |

All values are MPa. The peak axial stress of the whole blade in this load case is
**325.79 MPa at $\eta = 0.16$**, and the peak in-plane shear is **56.42 MPa at $\eta = 0.54$**,
where the shear webs take over from the caps. The $\sigma_{33}$ column is $10^{-13}$ MPa
throughout — identically zero to round-off, the first-order limit noted above, so do not read it
as a physical through-thickness stress.

![Peak recovered wall stress and per-station wall time along the IEA-22 blade](../_static/iea51_sweep.png)

`iea51_sweep.png` shows both halves of the sweep. Left, the spanwise peaks of $\sigma_{11}$ and
$\sigma_{12}$: the axial peak climbs steeply out of the root, crests near $\eta \approx 0.16$ and
decays to nothing at the tip, while the shear peak rises in two distinct steps, near
$\eta \approx 0.35$ and $\eta \approx 0.46$. Right, the recorded per-station wall time,
homogenization against dehomogenization — of order ten seconds and of order one second
respectively, with a scatter of much faster homogenizations. The same arrays are saved to
`iea51_sweep.npz` (`eta`, `props`, `peak`, `t_homo`, `t_dehom`, `stations`).

## The correctness check on transverse shear

`3_q_consistency_check.py` tests an identity that any recovered transverse shear must satisfy,
wall by wall: the thickness integral of the recovered stress must equal the wall's own
transverse-shear resultant, which the RM shell already knows.

$$I_a = \int_{h} \sigma_{a3}\,\mathrm{d}z \;\;\overset{!}{=}\;\; Q_a = (G_{\rm msg}\, s_2)_a ,
\qquad s_2 = [\,2\gamma_{13}\;\;2\gamma_{23}\,] .$$

The plate pipeline enforces this by construction — shape from the strain gradients, amplitude
from the resultant, $\sigma_{a3} \mathrel{*}= Q/I$. The shell-to-plate chain never uses $s_2$, so
nothing pins the amplitude, and the ratio $r = I/Q$ measures how far off it is: $r = 1$ is
consistent.

```python
name = "iea_s10"
ff_file = "ff51_rmc_reform.dat"
station_row = 10                  # eta = r/R = 0.20
n_z = 41                          # integration points through each wall
```

```bash
python 3_q_consistency_check.py
```

Outputs `iea_s10_q_consistency.txt` and `iea_s10_q_consistency.png` (recovered $|I_{13}|$ against
wall $|Q_1|$ on log axes with the consistency line, plus the histogram of
$\log_{10}(I_{13}/Q_1)$). The measured result:

| ratio | median | mean | min | max | verdict |
|---|---|---|---|---|---|
| $I_{13}/Q_1$ | 5.228e−01 | −6.974e+00 | −1.299e+03 | 2.260e+01 | target invalid |
| $I_{23}/Q_2$ | 9.895e−01 | 1.096e+00 | −1.217e+01 | 1.194e+01 | target valid |

**$\sigma_{23}$ passes**: the median $I_{23}/Q_2$ is 0.9895, a **1.05 % Q-consistency error**,
with wall $Q_2$ rms 8.9686 N/m against recovered $I_{23}$ rms 8.3085 N/m.

**$\sigma_{13}$ is not measurable in a cross-section model**, and the script explains why rather
than papering over it. This ring is homogenized with `shear="mitc4_g23"`: $\gamma_{23}$ is tied
by MITC, $\gamma_{13}$ is raw and carries the flat-wall drilling mode, so $Q_1$ built from it is
not a physical wall shear and cannot serve as a recovery target. The fingerprint is in the
magnitudes — $Q_1$ rms is $9.9064\times10^{2}$ N/m against $Q_2$ rms $8.9686$ N/m, a factor of
about 110, which is implausible for a thin-walled section that carries its load as in-plane shear
flow. The ring is a one-quad-deep prismatic strip, so the $\gamma_{13}$ tying interpolates along
a direction in which the row is already linear and reproduces it exactly: the script measures
`rms |Q1(g23) - Q1(both)| = 0.0000e+00 N/m` and the same for $Q_2$, so `mitc4_g23` and
`mitc4_both` are bit-identical here even though both flags branch. A physical $\gamma_{13}$ needs
the surface-quad route (the 3-D segment SG, where both rows are tied), not a rescale of this ring
against a contaminated $Q_1$.

```{note}
The rule that follows: trust $\sigma_{11}$, $\sigma_{22}$ and $\sigma_{12}$ from this pipeline
and treat $\sigma_{23}$ as good to about one percent in amplitude; do not use the cross-section
route's $\sigma_{13}$ or $\sigma_{33}$ as a design quantity.
```

## Where the tool ends

| produced by OpenSG_io | produced by OpenSG-2.0 | produced by BeamDyn/OpenFAST |
|---|---|---|
| per-station 1-D shell SG yamls; PreVABS XML for the 2-D solid route; OpenFAST blade data read back as a reference | the Timoshenko $6\times6$ per station and its `.out`; the RM wall law `ABDG.out`; the recovered 3-D wall stress and its validation against VABS; the Q-consistency diagnostic | the blade dynamics, and the per-station sectional resultants that come back as the FF table |

The chain closes only because the conventions match at each hand-off: the same reference surface
(`center`), the same component order (VABS), the same units (SI). Those three lines are the ones
to check first when a recovered stress field looks wrong.
