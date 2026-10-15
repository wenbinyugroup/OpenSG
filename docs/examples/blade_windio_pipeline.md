# Wind-turbine blade from windIO to BeamDyn and back

## Problem description

A wind-turbine blade is designed as a 3-D laminated shell but analysed, aeroelastically, as a 1-D beam.
This example walks the whole round trip on the IEA-22 blade, `IEA-22-280-RWT.yaml` (windIO version
2.0): the windIO file becomes one 1-D shell structure gene (SG) per span station, each station is
homogenized to a Timoshenko $6\times6$ stiffness and a $6\times6$ mass matrix, the spanwise table of
those matrices becomes a BeamDyn blade, the BeamDyn static solve returns the sectional resultants
station by station, and those resultants drive the dehomogenization back to the 3-D stress in the
wall.

Each station is a ring of two-node shell elements along the section contour and the shear webs, each
element carrying a laminate ({numref}`fig-windio-ring`). The engine is `opensg_shell` with its blade
package `opensg_shell.pynumad`. The Reissner-Mindlin (RM) shell ring homogenization of the
Mechanics of Structure Genome (MSG) gives the Timoshenko $6\times6$ in the VABS strain order $[\varepsilon_{11}\;\gamma_{12}\;\gamma_{13}\;\kappa_1\;\kappa_2\;\kappa_3]$, whose
diagonal is $[EA\;GA_2\;GA_3\;GJ\;EI_2\;EI_3]$. The beam properties are verified against VABS `.K`
files of the same blade meshed as 2-D solid sections.

```{figure} ../_static/shell_xsec_mesh.png
:width: 95%
:name: fig-windio-ring

An IEA-22 ring mesh coloured by layup (station $r/R = 0.20$ of {doc}`shell_beam_cross_section`); step 1 writes the same plot for every station as `cross_sections/<tag>_mesh.png`.
```

Seven numbered scripts in `examples/OpenSG_shell/windio/` cover the chain. The station tag is
`<prefix>_rXXXX` with `XXXX = round(r * 1000)`.

% source: examples/OpenSG_shell/windio/README.md and the script docstrings
| Step | Script | Reads | Writes |
|---|---|---|---|
| 1 | `1_windio_to_cross_sections.py` | `IEA-22-280-RWT.yaml` | `cross_sections/<tag>_shell.yaml`, mesh and orientation PNGs |
| 2 | `2_get_beam_props_from_shell_cross_section.py` | station yamls | `props/<tag>.K` (VABS `.K` layout: mass and Timoshenko $6\times6$) |
| 3 | `3_convert_to_beamdyn_props.py` | station `.K` files | `beamdyn/<prefix>_bd_props.inp` |
| 4 | `4_prepare_beamdyn_input.py` | windIO file and `.K` list | `beamdyn/<prefix>_bd_{primary,driver}.inp` (nodal loads) |
| 5 | `5_run_beamdyn_extract_ff.py` | driver input | `ff/<tag>.ff` (u, theta, DCM, F, M in the VABS frame) |
| 6 | `6_dehom_stress_recovery.py` | station yaml and `.ff` | `recovery/<tag>.{SM,EM,U}` and a stress PNG |
| 7 | `7_verify_blade_timo_vs_K.py` | windIO file and `props/*.K` | one PASS or FAIL line per station |

```{mermaid}
flowchart TD
    W["IEA-22-280-RWT.yaml<br/>windIO v1 or v2"] --> S1["1: cross_sections/&lt;tag&gt;_shell.yaml"]
    S1 --> S2["2: props/&lt;tag&gt;.K<br/>Timoshenko + mass 6x6"]
    S2 --> S3["3: beamdyn/iea22_bd_props.inp"]
    W --> S4["4: iea22_bd_primary.inp, iea22_bd_driver.inp"]
    S3 --> S4
    S4 --> S5["5: beamdyn_driver, ff/&lt;tag&gt;.ff"]
    S5 --> S6["6: recovery/&lt;tag&gt;.SM .EM .U"]
    S1 --> S6
    S2 --> S7["7: in-memory Blade vs props/*.K"]
```

### Stations

`stations = "airfoil"` uses the blade's own airfoil positions, 16 on IEA-22; an integer `N` puts `N`
uniform stations $r = i/(N-1)$ (51 matches the VABS 51-station reference set), and any explicit list
of $r$ works too. Step 1 records the stations:

% source: examples/OpenSG_shell/windio/iea22_stations.dat
| $r$ | chord [m] | twist [windIO unit] | nodes | elements | sets | webs |
|---|---|---|---|---|---|---|
| 0.0000 | 5.8000 | 12.000000 | 580 | 583 | 4 | 3 |
| 0.0200 | 5.8000 | 11.999024 | 580 | 583 | 5 | 3 |
| 0.0487 | 6.1820 | 11.985147 | 545 | 548 | 6 | 3 |
| 0.0665 | 6.4552 | 11.959527 | 512 | 515 | 6 | 3 |
| 0.0835 | 6.6851 | 11.915293 | 471 | 474 | 6 | 3 |
| 0.1022 | 6.8884 | 11.829706 | 428 | 431 | 6 | 3 |
| 0.1104 | 6.9598 | 11.778809 | 414 | 417 | 6 | 3 |
| 0.1364 | 7.1256 | 11.508292 | 372 | 375 | 6 | 3 |
| 0.1556 | 7.1925 | 11.164676 | 346 | 349 | 6 | 3 |
| 0.1967 | 7.2055 | 9.501771 | 315 | 318 | 6 | 3 |
| 0.2470 | 7.0368 | 7.035050 | 302 | 305 | 6 | 3 |
| 0.3993 | 5.8405 | 3.366355 | 290 | 293 | 6 | 3 |
| 0.5336 | 4.5414 | 0.393329 | 274 | 277 | 7 | 3 |
| 0.7389 | 2.9301 | -4.174710 | 259 | 262 | 6 | 3 |
| 0.9800 | 1.2502 | -5.957699 | 241 | 244 | 6 | 3 |
| 1.0000 | 0.4000 | -4.952606 | 241 | 244 | 6 | 3 |

### Materials

The blade layers use six materials of the windIO `materials` block, copied as the file states them: a
scalar for an isotropic material, the lists $[E_1, E_2, E_3]$, $[G_{12}, G_{13}, G_{23}]$ and
$[\nu_{12}, \nu_{13}, \nu_{23}]$ for an orthotropic one.

% source: examples/OpenSG_shell/windio/IEA-22-280-RWT.yaml (materials)
| Name | orth | Density [kg/m³] | $E$ [Pa] | $G$ [Pa] | $\nu$ |
|---|---|---|---|---|---|
| gelcoat | 0 | 1235.0 | 3440000000.0 | 1323000000.0 | 0.3 |
| glass_uniax | 1.0 | 1940.0 | [43700000000.0, 16500000000.0, 15450000000.0] | [3265000000.0, 3495000000.0, 3480000000.0] | [0.262, 0.264, 0.35] |
| carbon_uniax | 1.0 | 1600.0 | [140000000000.0, 9100000000.0, 9100000000.0] | [4131000000.0, 4131000000.0, 2689000000.0] | [0.3133, 0.3133, 0.47066] |
| glass_biax | 1.0 | 1940.0 | [11023100000.0, 11023100000.0, 16047700000.0] | [13231400000.0, 3487480000.0, 3487480000.0] | [0.688074, 0.117173, 0.117173] |
| glass_triax | 1.0 | 1940.0 | [28211400000.0, 16238800000.0, 15835500000.0] | [8248220000.0, 3491240000.0, 3491240000.0] | [0.497511, 0.18091, 0.27481] |
| medium_density_foam | 0.0 | 130.0 | 142500000.0 | 54000000.0 | 0.3194 |

### Conventions

- The section origin is the windIO reference axis $x_1$ (the chordwise `section_offset_y` or
  `pitch_axis` shift). The $6\times6$, the BeamDyn loads and the recovery all share it.
- The shell reference surface of the blade route is the outer mold line (OML) by default; the
  laminate mid-surface is an option for matching a 2-D solid section (step 1).
- Cross-section matrices are in the VABS order and frame (1 = axial). The BeamDyn files are in the
  IEC blade frame, reached by the beam-axis swap `B = [[0,0,1],[0,-1,0],[1,0,0]]`.
- BeamDyn runs trapezoidal quadrature, so its output nodes coincide with the property stations; the
  `.ff` resultants are the section-local (`FxL`, `MxL`) channels.
- `.SM` and `.EM` hold Gauss-point stress and strain in the material frame, VABS column order
  `x2 x3 s11 s12 s13 s22 s23 s33`; `.U` holds `id x2 x3 u1 u2 u3`, the total displacement.
- All files are in SI units.

## Input

The windIO file is the only input: the blade's `reference_axis`, `outer_shape` and `structure` under
`components`, plus the `airfoils` and `materials` blocks. Every script opens with a `User Input` block,
shown with each step below. Run the scripts in order from inside `examples/OpenSG_shell/windio/` with
OpenSG installed ({doc}`../getting_started/installation`); each reads the folder the previous step
wrote.

## Run

### Step 1: windIO blade to station cross-sections

`1_windio_to_cross_sections.py` reads the blade (windIO v1 `outer_shape_bem` or v2 `outer_shape`),
builds the cross-section at every station and writes one 1-D shell SG yaml per station, the ring plot
`<tag>_mesh.png`, the $e_1/e_2/e_3$ orientation plot `<tag>_orient.png`, and `iea22_stations.dat`.

% source: examples/OpenSG_shell/windio/1_windio_to_cross_sections.py
```python
windio = "IEA-22-280-RWT.yaml"     # windIO blade file (v1 or v2)
prefix = "iea22"                   # station files = <prefix>_rXXXX_shell.yaml
stations = "airfoil"               # "airfoil" | int N (uniform, e.g. 51) | [0.2, 0.5, ...]
mesh_size = 0.01                   # element arc length / chord
reference = "oml"                  # shell reference surface: "oml" (default) |
                                   # "center" (laminate mid-surface -- choose it
                                   # to match a 2-D solid / VABS section)
```

```bash
python 1_windio_to_cross_sections.py
```

Each `cross_sections/<tag>_shell.yaml` carries the header `msg: shell`, `refined: 1` and holds,
in file order:

% source: src/opensg_shell/pynumad/sg_mesh.py (emit_shell_yaml); docs/tutorials/windio_blade_pipeline.md (step 1)
| Key | Content |
|---|---|
| `nodes` | contour node coordinates `[y2 y3 0]` in m, origin on the reference axis; the third entry is the beam axis, identically 0 |
| `elements` | two-node line segments along the contour and the webs |
| `sets` | `element` sets `layup_k`, one per laminate, holding the element labels |
| `sections` | `type: shell`, `elementSet: layup_k`, and the `layup` as `[material, thickness (m), angle (deg)]` triples, outermost ply first |
| `elementOrientations` | one $3\times3$ frame per element, flattened; the first row is $e_1$ = `[0, 0, 1]`, the beam axis |
| `materials` | `density` (kg/m³) and the `elastic` constants, in Pa |

The surface the contour was generated on is recorded only as a leading comment, never as a key:

% source: src/opensg_shell/pynumad/sg_mesh.py (emit_shell_yaml)
```text
# contour placed on the outer mold line (OML) -- the default laminate reference
# contour placed on the laminate mid-surface (skin offset 0.5 t inward from the OML) -- run opensg with --center
```

The laminate reference is a run-time choice. `opensg <yaml>` takes the contour as the OML
(`frac` = 0.0), `opensg <yaml> --center` (`beam_props(yaml, ref="center")`) as the laminate
mid-surface (`frac` = 0.5). That choice fixes the through-thickness offset used by the ring laminate
law, the plate SG's `z_ref`, the emitted ABD matrix (extension, coupling and bending stiffness of the
wall) and the recovery depth conversion alike ({doc}`../user_guide/analysis_settings`).

:::{warning}
`reference = "center"` moves the contour to the mid-surface, but steps 2 and 6 call `beam_props` and
`dehom_to_files` without `ref`, that is at the OML. A center-offset station set needs `ref="center"`
in both calls. `tests/msg_shell_windio/test_vabs_k_beam_props.py` and `test_windio_dehom.py` call
`beam_props(..., ref="center")`, and the latter passes that bundle to the recovery.
:::

#### OpenSG_io, the separate input layer

The scripts of this folder read the windIO file with `opensg_shell.pynumad` and need nothing else.
OpenSG_io (repository <https://github.com/bagla0/OpenSG_io>, documentation
<https://bagla0.github.io/OpenSG_io/>) is the separate input-layer repository that produced the 51
station yamls of {doc}`blade_iea_beam`. It reads a windIO blade, auto-detecting v1 and v2,
interpolates the outer shape, the layup stacks and the material database to a requested span
fraction, and writes the cross-section both as the 1-D shell SG yaml and as PreVABS XML, the 2-D
solid route of the VABS reference. It also reads PreVABS XML and OpenFAST ElastoDyn and BeamDyn blade
data, so published beam properties can serve as a reference. Install it alongside OpenSG:

% source: docs/tutorials/windio_blade_pipeline.md (step 1)
```bash
git clone --recurse-submodules https://github.com/bagla0/OpenSG_io
```

```bash
pip install -r requirements.txt
```

That file lists numpy, scipy, pyyaml, meshio, gmsh, tetgen and pyvista. A windIO v2 blade also needs
the `windIO` package, which the converter imports only on the v2 branch.

```bash
python scripts/fetch_prevabs.py
```

The command line writes a set of stations in one call, and the Python API scripts a full blade:

```bash
python scripts/opensg_io.py BAR_URC.yaml out --name bar --stations 0.3 0.5 0.7 --solid
```

```python
from opensg_io import load_blade, build_cross_section, emit_opensg_yaml, emit_prevabs

blade = load_blade("IEA-22-280-RWT.yaml")      # windIO v1 or v2, auto-detected
cs = build_cross_section(blade, r=0.5)         # one span fraction
emit_opensg_yaml(cs, "shell_r050.yaml")        # the 1-D shell SG for opensg_shell
emit_prevabs(cs, "prevabs_r050", name="r050")  # PreVABS XML -> 2-D solid route
```

### Step 2: beam properties of every station

`2_get_beam_props_from_shell_cross_section.py` homogenizes each station yaml with the production RM
ring (`build_rm_bundle`): a 6-DOF-per-node shell element with an element-wise drilling Lagrange
multiplier, $\gamma_{23}$ tied by mixed interpolation of tensorial components (MITC), and the MSG wall
transverse-shear law $G$. The ring warping problem is solved for the four Euler-Bernoulli modes $V_0$
and the Timoshenko correction $V_1$, and the energy is condensed to the beam $6\times6$. The mass
$6\times6$ is integrated over the same ring from the through-thickness density moments of each
laminate.

% source: examples/OpenSG_shell/windio/2_get_beam_props_from_shell_cross_section.py
```python
prefix = "iea22"                   # station tag prefix from step 1
```

```bash
python 2_get_beam_props_from_shell_cross_section.py
```

Each station writes `props/<tag>.K` in the VABS `.K` layout (mass blocks, mass centre, Timoshenko
stiffness and compliance, tension and shear centres, principal values, `Time taken`). The spanwise
tables are `iea22_timo_by_r.dat` ($r$ and the flattened $6\times6$, row-major) and
`iea22_mass_by_r.dat` ($r$, mass per span, flattened mass $6\times6$). `build_rm_bundle` also writes
`<base>_Timo.out` and `<base>_ABDG.out` beside each yaml, and `beam_props` inserts the mass block
into that `_Timo.out` ({doc}`../user_guide/output_files`). The console prints one line per station
with the six diagonal terms and the mass per span `mu`.

### Step 3: BeamDyn blade properties

`3_convert_to_beamdyn_props.py` reads the stiffness and mass $6\times6$ of every station `.K`
(`read_k_file` also accepts native VABS `.K` files), rotates both to the BeamDyn blade frame by the
block-wise similarity of the beam-axis swap (`vabs_to_beamdyn`), and writes the BeamDyn blade
properties file (`write_bd_props`).

```bash
python 3_convert_to_beamdyn_props.py
```

The file opens with the station count and the damping block, then lists, per station, $\eta$, the
$6\times6$ stiffness and the $6\times6$ mass:

% source: examples/OpenSG_shell/windio/beamdyn/iea22_bd_props.inp
```text
 ------- BEAMDYN V1.00.* INDIVIDUAL BLADE INPUT FILE --------------------------
 OpenSG msg-shell blade sectional properties
 ---------------------- BLADE PARAMETERS --------------------------------------
16   station_total    - Number of blade input stations (-)
 1   damp_type        - Damping type: 0: no damping; 1: damped
```

% source: src/opensg_shell/pynumad/sg_beamdyn.py (write_bd_primary)
BeamDyn, the geometrically exact beam module of OpenFAST, represents the blade by a sectional
$6\times6$ stiffness and a sectional $6\times6$ mass given at normalized span stations. With
trapezoidal quadrature (`quadrature = 2`) every station supplied is a quadrature point, so the
OpenSG table maps one-to-one onto the BeamDyn blade stations; `refine` (step 4) adds `refine - 1`
interpolated quadrature points between consecutive stations, which a sparse set such as the 16
airfoil stations needs to converge. Step 3 passes the full matrices, not their diagonals. The
station scripts of {doc}`blade_iea_beam` compute no sectional mass matrix (their only mass quantity
is `mass_per_area` per laminate in the emitted ABD yaml); this route integrates it in step 2.

:::{note}
OpenSG_io offers the same hand-over from its side: `openfast_io.timo_to_beamdyn` permutes an OpenSG
Timoshenko $6\times6$ into BeamDyn DOF order, `openfast_io.write_beamdyn_blade` writes a BeamDyn blade
input from a list of etas and stiffness matrices (its `M_list` argument is optional and zero-filled
when omitted, so the mass matrices stay your responsibility), and `openfast_io.read_beamdyn_blade`
reads one back.
:::

### Step 4: BeamDyn primary and driver inputs

`4_prepare_beamdyn_input.py` writes a primary input with a straight, untwisted reference line and
trapezoidal quadrature, and a static driver input that applies a uniform flapwise OML surface traction
$p$ as nodal loads about the reference axis $x_1$: $F_{x,i} = p\,P_i\,\Delta s_i$ (flap, $P$ the OML
perimeter) and $M_{z,i} = p\,\left(\int x_2\,\mathrm{d}s\right)_i\,\Delta s_i$ (torsion), with
$\Delta s_i$ the tributary span bin of station $i$ (half a bin at the ends).

% source: examples/OpenSG_shell/windio/4_prepare_beamdyn_input.py
```python
windio = "IEA-22-280-RWT.yaml"     # windIO blade file (same as step 1)
prefix = "iea22"                   # station tag prefix
p_traction = 1500.0                # flapwise OML surface traction [Pa]
refine = 3                         # trapezoidal quadrature refinement (sparse-station
                                   # sets need >1 to converge; must match step 5)
```

```bash
python 4_prepare_beamdyn_input.py
```

% source: examples/OpenSG_shell/windio/beamdyn/iea22_bd_primary.inp
The primary input sets `quadrature` 2, `refine` 3, `order_elem` 10 and 69 key points from 0 to
1.38204e+02 m. The driver is a static solve without gravity, with 16 point loads:

% source: examples/OpenSG_shell/windio/beamdyn/iea22_bd_driver.inp
```text
Non-dim blade-span eta   Fx          Fy            Fz           Mx           My           Mz
(-)                      (N)         (N)           (N)          (N-m)        (N-m)        (N-m)
0.0000 37763.20551 0 0 0 0 1.170177284
0.0200 92520.30414 0 0 0 0 192.5657357
0.0490 89318.48617 0 0 0 0 3626.353092
0.0660 67690.54664 0 0 0 0 15718.00732
0.0830 70521.10148 0 0 0 0 28499.49867
0.1020 50620.21299 0 0 0 0 27946.63077
0.1100 62717.39048 0 0 0 0 38473.73569
0.1360 82247.92983 0 0 0 0 63522.06751
0.1560 106859.0298 0 0 0 0 86707.48597
0.1970 153599.6942 0 0 0 0 140249.6234
0.2470 327595.4522 0 0 0 0 343961.5104
0.3990 379518.1878 0 0 0 0 403172.4418
0.5340 343475.8184 0 0 0 0 301283.9792
0.7390 286856.4522 0 0 0 0 153778.8218
0.9800 70595.16161 0 0 0 0 16080.23068
1.0000 1730.856643 0 0 0 0 165.0140051
```

### Step 5: run BeamDyn and extract the .ff files

`5_run_beamdyn_extract_ff.py` runs `beamdyn_driver` (conda-forge `openfast`) and maps the converged
nodal channels to the VABS frame. Station $i$ is output node $i \cdot \mathrm{refine} + 1$ of the
$(n-1) \cdot \mathrm{refine} + 1$ quadrature nodes, so `refine` must match step 4.

% source: examples/OpenSG_shell/windio/5_run_beamdyn_extract_ff.py
```python
prefix = "iea22"                   # station tag prefix
refine = 3                         # trapezoidal refinement (MUST match step 4)
beamdyn_exe = os.path.expanduser("~/miniconda3/bin/beamdyn_driver")
ld_library = os.path.expanduser("~/miniconda3/lib")   # None if beamdyn_driver is self-contained
```

```bash
python 5_run_beamdyn_extract_ff.py
```

Each station gets one `.ff` file, the glb-type recovery input: the displacement, the
Wiener-Milenkovic rotation parameters, the exact direction cosine matrix (DCM) of the deformed triad,
and the section-local force and moment resultants ($F_1$ axial, $M_1$ torsion), values first and
labels inline:

% source: examples/OpenSG_shell/windio/ff/iea22_r0197.ff
```text
# OpenSG station FF (BeamDyn -> VABS frame)  eta = 0.197000

-0.000765691337 -0.027464943 0.317219704 # u1 u2 u3 [m];
0.00507813599 -0.026433371 -0.00223995186 # th1 th2 th3 (Wiener-Milenkovic);

0.9996481618418236 -0.002306755494465712 0.02642406867521203 # C 3x3;
0.002172535481788995 0.9999845989797325 0.005107043462147566
-0.02643544241816159 -0.00504783938261199 0.999637777748382

37176.3008 7185.15283 1406400.88 # F1 F2 F3 [N]
1263782.38 -57353936 259608.797 # M1 M2 M3 [N-m]
```

The script also writes the table `ff/iea22_ff.dat` (`# eta F1 F2 F3 M1 M2 M3`). The macro-state file of the yaml route and the `.glb`
converter are described in {doc}`../user_guide/recovery` and {doc}`../user_guide/converters`.

### Step 6: stress recovery

`6_dehom_stress_recovery.py` drives a two-step RM dehomogenization over the full cross-section with
each station's `.ff`:

$$\text{beam } FF \;\xrightarrow[\text{RM shell ring}]{}\; \text{section strains and gradients}
\;\xrightarrow[\text{RM plate SG}]{}\; \sigma(z)\ \text{through the wall.}$$

Step one recovers the six RM shell strains of every ring element and their gradients: along the span
from the macro recovery, along the contour by nodal differencing between elements of the same layup.
Step two evaluates the MSG-RM through-thickness stress and strain of the 1-D plate SG of each laminate
at `n_depth` ply-interior points, $\zeta = (k + 0.5)/n_\text{depth}$ from the OML ($\zeta = 0$) to the
inner surface ($\zeta = 1$), plus the total displacement (beam motion and RM warping).

% source: examples/OpenSG_shell/windio/6_dehom_stress_recovery.py
```python
prefix = "iea22"                   # station tag prefix
n_depth = 9                        # through-thickness recovery points per element
stress_frame = "material"          # "material" (ply axes) | "plate" (wall axes)
png_station = 0.2                  # r of the station to plot (nearest .ff)
```

```bash
python 6_dehom_stress_recovery.py
```

Each station writes `recovery/<tag>.SM`, `.EM` and `.U` (VABS layout, one row per recovery point,
that is elements times `n_depth` rows), and the console prints the peak $|\sigma_{11}|$ in MPa with
its location and the largest displacement. The station nearest `png_station` also gets
`<tag>_dehom_stress.png`, the $\sigma_{11}$ and $\sigma_{12}$ clouds in MPa. The `.SM` file opens
with:

% source: src/opensg_shell/pynumad/sg_recovery.py (write_vabs_sm)
```text
 Stresses for each Gauss point in material coordinate system for load case #
           1
```

### Step 7: verification of the stored .K files

`7_verify_blade_timo_vs_K.py` reads the blade into the editable `Blade` object (`opensg.read`),
recomputes every station of `props/` fully in memory (`opensg_shell.pynumad.sg_homo.timo`, no yaml,
no `.out`), and compares both the stiffness and the mass $6\times6$ with the stored
`iea22_rXXXX.K`. Each tag is mapped back to the blade's true airfoil station before solving.

% source: examples/OpenSG_shell/windio/7_verify_blade_timo_vs_K.py
```python
BLADE_YAML = "IEA-22-280-RWT.yaml"
K_FOLDER = "props"
RTOL = 1e-8                        # max-entry-relative pass tolerance
```

```bash
python 7_verify_blade_timo_vs_K.py
```

A station passes when `max|dK|rel` and `max|dM|rel`, the largest absolute difference over the largest
file entry, are both below `RTOL`. The script prints one PASS or FAIL line per station and the count of
verified stations; any failure fails the final assertion with a nonzero exit code.
`tests/msg_shell_windio/test_blade_vs_k.py` runs it as a unit test.

## Result

### Beam properties of one station

% source: examples/OpenSG_shell/windio/props/iea22_r0197.K
```{list-table} Timoshenko stiffness of station iea22_r0197 ($r$ = 0.1967), reference=oml. $EA$, $GA_2$, $GA_3$ in N; $GJ$, $EI_2$, $EI_3$ in N·m²; extension-bending and shear-twist terms in N·m.
:name: tab-windio-r0197-timo
:header-rows: 1

* -
  - $\varepsilon_{11}$
  - $\gamma_{12}$
  - $\gamma_{13}$
  - $\kappa_1$
  - $\kappa_2$
  - $\kappa_3$
* - $\varepsilon_{11}$
  - 2.7951877290E+10
  - -2.1429882787E-05
  - 7.5073214293E-06
  - 3.9565286981E-06
  - 1.6652547691E+09
  - -2.1591112090E+09
* - $\gamma_{12}$
  - -2.1429882787E-05
  - 7.1994847930E+08
  - 5.6004311457E+07
  - -1.0523909703E+08
  - -1.2834156226E-05
  - 2.3471526150E-05
* - $\gamma_{13}$
  - 7.5073214293E-06
  - 5.6004311457E+07
  - 4.0450355825E+08
  - -1.5914652014E+08
  - 5.8601607326E-06
  - -2.9831378863E-06
* - $\kappa_1$
  - 3.9565286981E-06
  - -1.0523909703E+08
  - -1.5914652014E+08
  - 2.3898886919E+09
  - -2.6053118617E-06
  - -7.3885502608E-06
* - $\kappa_2$
  - 1.6652547691E+09
  - -1.2834156226E-05
  - 5.8601607326E-06
  - -2.6053118617E-06
  - 3.5857652277E+10
  - -5.7907296931E+09
* - $\kappa_3$
  - -2.1591112090E+09
  - 2.3471526150E-05
  - -2.9831378863E-06
  - -7.3885502608E-06
  - -5.7907296931E+09
  - 7.0001845499E+10
```

The entries near $10^{-5}$ against diagonals of $10^{10}$ are numerically zero. The same file holds
the mass matrix about the section origin (kg/m, kg, kg·m):

% source: examples/OpenSG_shell/windio/props/iea22_r0197.K
```text
 The 6X6 Mass Matrix
 ========================================================

    7.7354024400E+02    0.0000000000E+00    0.0000000000E+00    0.0000000000E+00    5.0881542123E+01   -5.4550855829E+02
    0.0000000000E+00    7.7354024400E+02    0.0000000000E+00   -5.0881542123E+01    0.0000000000E+00    0.0000000000E+00
    0.0000000000E+00    0.0000000000E+00    7.7354024400E+02    5.4550855829E+02    0.0000000000E+00    0.0000000000E+00
    0.0000000000E+00   -5.0881542123E+01    5.4550855829E+02    4.5177211634E+03    0.0000000000E+00    0.0000000000E+00
    5.0881542123E+01    0.0000000000E+00    0.0000000000E+00    0.0000000000E+00    6.4946932711E+02   -1.3819247507E+02
   -5.4550855829E+02    0.0000000000E+00    0.0000000000E+00    0.0000000000E+00   -1.3819247507E+02    3.8682518362E+03
```

and the section summary:

% source: examples/OpenSG_shell/windio/props/iea22_r0197.K
| Quantity | Value |
|---|---|
| mass per unit span [kg/m] | 7.7354024400E+02 |
| mass centre $(x_2, x_3)$ [m] | 7.0521031392E-01, 6.5777498350E-02 |
| tension centre $(x_2, x_3)$ [m] | 7.7243871192E-02, 5.9575775602E-02 |
| shear centre $(x_2, x_3)$ [m] | -3.7726143995E-01, 1.1682895688E-01 |
| principal $EI_{22}$, $EI_{33}$ [N·m²] | 3.4842272968E+10, 7.0751237855E+10 |
| principal bending axes rotation [deg] | 9.1912292631 |
| principal $GA_{22}$, $GA_{33}$ [N] | 3.9485559888E+08, 7.2959643867E+08 |
| time taken [s] | 2.11 |

### Spanwise table

The diagonal of every station from `iea22_timo_by_r.dat`, with the mass per span from
`iea22_mass_by_r.dat` ($r$ is the tag value):

% source: examples/OpenSG_shell/windio/iea22_timo_by_r.dat, iea22_mass_by_r.dat
| $r$ | $EA$ [N] | $GA_2$ [N] | $GA_3$ [N] | $GJ$ [N·m²] | $EI_2$ [N·m²] | $EI_3$ [N·m²] | $\mu$ [kg/m] |
|---|---|---|---|---|---|---|---|
| 0.000 | 6.56947306e+10 | 7.93568968e+09 | 8.32639022e+09 | 1.27181830e+11 | 3.03422342e+11 | 2.23307552e+11 | 4.02872517e+03 |
| 0.020 | 6.56981394e+10 | 7.93565097e+09 | 8.32695240e+09 | 1.27187498e+11 | 3.03426728e+11 | 2.23317600e+11 | 4.02887897e+03 |
| 0.049 | 3.32172327e+10 | 3.27098476e+09 | 3.21472420e+09 | 4.74864060e+10 | 1.50654734e+11 | 1.22512917e+11 | 1.79579833e+03 |
| 0.066 | 3.03500218e+10 | 2.18188475e+09 | 2.06993714e+09 | 2.80758184e+10 | 1.24281281e+11 | 1.22491488e+11 | 1.45765254e+03 |
| 0.083 | 2.97373843e+10 | 1.55955124e+09 | 1.44450955e+09 | 1.77856517e+10 | 1.03700807e+11 | 1.25501583e+11 | 1.26747766e+03 |
| 0.102 | 2.90573970e+10 | 1.06188918e+09 | 9.77076140e+08 | 1.07057792e+10 | 8.22334580e+10 | 1.20541699e+11 | 1.08599951e+03 |
| 0.110 | 2.85385420e+10 | 9.16007354e+08 | 8.38113036e+08 | 8.76635400e+09 | 7.42845034e+10 | 1.13203044e+11 | 1.01336941e+03 |
| 0.136 | 2.80407281e+10 | 7.26342612e+08 | 5.95442860e+08 | 5.18325778e+09 | 5.54148120e+10 | 1.00821006e+11 | 8.92845519e+02 |
| 0.156 | 2.80085281e+10 | 7.32126032e+08 | 5.03457037e+08 | 3.80043607e+09 | 4.60518509e+10 | 8.98434286e+10 | 8.44928692e+02 |
| 0.197 | 2.79518773e+10 | 7.19948479e+08 | 4.04503558e+08 | 2.38988869e+09 | 3.58576523e+10 | 7.00018455e+10 | 7.73540244e+02 |
| 0.247 | 2.75452413e+10 | 6.99968333e+08 | 3.60803341e+08 | 1.90729106e+09 | 2.86576255e+10 | 6.27196360e+10 | 7.29336585e+02 |
| 0.399 | 2.48813487e+10 | 6.01279743e+08 | 2.48690065e+08 | 9.40122896e+08 | 1.45627433e+10 | 3.38966942e+10 | 5.81134082e+02 |
| 0.534 | 2.06482921e+10 | 5.13998221e+08 | 1.57355017e+08 | 3.79853710e+08 | 5.67376781e+09 | 1.48313856e+10 | 4.33808597e+02 |
| 0.739 | 1.22847674e+10 | 4.26630552e+08 | 8.25872590e+07 | 9.15006400e+07 | 1.05613212e+09 | 3.47609292e+09 | 2.42736112e+02 |
| 0.980 | 9.41774098e+08 | 7.04420992e+07 | 8.84473611e+06 | 2.34640513e+06 | 1.15752971e+07 | 6.19251384e+07 | 3.07069270e+01 |
| 1.000 | 2.68357738e+08 | 1.52435594e+07 | 1.53118064e+06 | 4.70974291e+04 | 3.06822170e+05 | 1.61203105e+06 | 7.61475112e+00 |

The $r$ = 0.197 row is the diagonal of `props/iea22_r0197.K` above. Each row of the file holds the full
$6\times6$.

### Sectional resultants

Selected rows of `ff/iea22_ff.dat`, the section-local resultants of the shipped BeamDyn run in VABS
order (N, N·m). In the VABS frame the flapwise traction appears as $F_3$ and $M_2$ and the torque as
$M_1$; the root values of $F_1$, $F_2$ and $M_3$ are numerical zeros.

:::{note}
The shipped `beamdyn/iea22_bd_props.inp` and `ff/` files were written from an earlier `props/` set
and do not match `props/*.K` digit for digit; run steps 3 to 5 again to regenerate them from the
current `.K` files.
:::

% source: examples/OpenSG_shell/windio/ff/iea22_ff.dat
| $\eta$ | $F_1$ | $F_2$ | $F_3$ | $M_1$ | $M_2$ | $M_3$ |
|---|---|---|---|---|---|---|
| 0.000 | 5.14781448e-07 | 3.49841489e-09 | 2.22362975e+06 | 1.27953700e+06 | -1.08268168e+08 | 5.54819621e-08 |
| 0.197 | 3.71763008e+04 | 7.18515283e+03 | 1.40640088e+06 | 1.26378238e+06 | -5.73539360e+07 | 2.59608797e+05 |
| 0.534 | 5.22623711e+04 | 1.83595684e+04 | 4.73125562e+05 | 3.49066906e+05 | -1.32230240e+07 | 4.74558875e+05 |
| 1.000 | 4.82637909e+02 | 1.96856735e+02 | 1.65050684e+03 | 1.58452408e+02 | -3.07811975e+00 | -4.59672050e+01 |

### Verification against VABS

`vabs_K/` holds VABS 2-D solid `.K` references from the 51-station IEA-22 set: `iea_s10.sg.K`,
`iea_s25.sg.K` and `iea_s35.sg.K`, at $\eta$ = 0.2, 0.5 and 0.7. Two unit tests build the stations at
exactly those $r$ and gate the Timoshenko diagonal and the mass terms. Each gate is the measured worst
relative difference over the three stations, padded by about 1.5 (about 1.7 for the center-reference
mass terms). These are shell-against-2-D-solid modelling tolerances, not numerical parity: VABS
solves a different model. The gates are relative tolerances, the measured values percentages.

% source: examples/pynumad/test_timo_pynumad.py (OML route), tests/msg_shell_windio/test_vabs_k_beam_props.py (center reference)
| Term | OML route: measured worst [%] | OML route: gate (relative) | Center reference: measured worst [%] | Center reference: gate (relative) |
|---|---|---|---|---|
| $EA$ | 1.52 | 0.025 | 1.45 | 0.025 |
| $GA_2$ | 2.54 | 0.040 | 3.33 | 0.050 |
| $GA_3$ | 14.33 | 0.22 | 2.05 | 0.035 |
| $GJ$ | 2.99 | 0.045 | 2.23 | 0.035 |
| $EI_2$ | 1.37 | 0.020 | 0.14 | 0.005 |
| $EI_3$ | 2.45 | 0.040 | 2.70 | 0.040 |
| mass per span $\mu$ | 2.26 | 0.040 | 1.46 | 0.025 |
| $i_{11}$ | 2.53 | 0.040 | 1.22 | 0.025 |
| $i_{22}$ | 2.31 | 0.040 | 0.63 | 0.025 |
| $i_{33}$ | 2.57 | 0.040 | 1.45 | 0.025 |

The OML route (`blade.timo`, the reference surface of this pipeline) also keeps the mass centres within
1.0 % of the chord. Its wide $GA_3$ band absorbs the one-sided OML laminate offset on the thin aft
panels, which grows outboard (6.2, 12.4 and 14.3 % at the three stations). The center-reference yaml
route, whose contour sits on the mid-surface as in the solid section, stays within 3.33 % on every
term. The center-reference test `tests/msg_shell_windio/test_vabs_k_beam_props.py` runs with the
rest of that folder from the repository root:

% source: tests/msg_shell_windio/test_vabs_k_beam_props.py, examples/pynumad/test_timo_pynumad.py
```bash
pytest tests/msg_shell_windio -q
```

The OML-route benchmark `test_timo_pynumad.py` runs from `examples/pynumad/`, as a unit test or as
a script that prints the full comparison table:

```bash
pytest test_timo_pynumad.py -q
```

```bash
python test_timo_pynumad.py
```

The recovery is gated in `tests/msg_shell_windio/test_windio_dehom.py`: the recovered $\sigma_{11}$
cloud must integrate back to the applied resultants ($F_1 = \int \sigma_{11}\,\mathrm{d}A$,
$M_2 = \int \sigma_{11} x_3\,\mathrm{d}A$, $M_3 = -\int \sigma_{11} x_2\,\mathrm{d}A$), and under the
51-station forces of `ff51_rmc_reform.dat` the 99th percentile of $|\sigma_{11}|$ must match the VABS
`.SM` of the same station within 0.20 relative. The point-by-point comparison of the recovered stress
with VABS is in {doc}`blade_iea_beam`.

### Where the tool ends

| Produced by OpenSG | Produced by BeamDyn/OpenFAST | Produced by OpenSG_io (optional) |
|---|---|---|
| the station yamls; the Timoshenko and mass $6\times6$ per station and the spanwise tables; the BeamDyn input files; the `.ff` extraction; the recovered 3-D wall stress, strain and displacement | the static blade solution and the sectional resultants at every station | station yamls and PreVABS XML for the 2-D solid route; OpenFAST blade data read back as a reference |

The chain closes only because the conventions match at each hand-off: the same reference axis and
surface, the same component order (VABS), the same units (SI). Check those first when a recovered
stress looks wrong.

## Other routes to the same properties

Steps 1 and 2 have a fused terminal route that runs fully in memory, without station yamls: every
station's VABS-layout `.out` record plus the spanwise `<stem>_{stations,timo_by_r,mass_by_r}.dat`
tables under `cross_sections/`. The single-station bypass prints the Timoshenko $6\times6$ and stores
the `<tag>.out` record only.

% source: examples/OpenSG_shell/windio/README.md
```bash
opensg gen_windio_cs IEA-22-280-RWT.yaml
```

```bash
opensg windio_st IEA-22-280-RWT.yaml 0.1967
```

Both take `--mesh-size`, `--center` (mid-surface instead of the OML), `--out` and `--prefix`;
`gen_windio_cs` also takes `--stations` ({doc}`../user_guide/converters`). A pyNuMAD blade, a
windIO-v1 dialect that places the trailing- and leading-edge reinforcements by width, runs through
`opensg pynumad <blade.yaml> [st-id] [flags]`, where `st-id` is a 0-based station index, a span $r$
with a decimal point, or `all` (the default). A plain windIO reader would drop those reinforcements,
which on IEA-15 is up to -84 % edgewise stiffness and -19 % mass per span. The editable `Blade` class
of step 7 drives design studies in memory:

% source: examples/pynumad/README.md
```python
import opensg

blade = opensg.read("IEA-15-240-RWT.yaml")   # windIO v1/v2 or pyNuMAD dialect
K, M = blade(4)                              # one station: Timoshenko + mass 6x6
Ks, Ms = blade()                             # full spanwise sweep (two lists)
```

`blade.section(st)` edits one station's resolved stacks: in `blade_section_edit_demo.py` the spar cap
thickened by a factor 1.3 at station 4 gives $EA$ +25.4 %, $EI_2$ +24.4 % and $EI_3$ +1.3 %. On
IEA-15, `--center` moves $EA$ by about -1.3 % and the mass by -1.7 %.
`opensg pynumad IEA-15-240-RWT.yaml 4` ($r$ = 0.3288) stores `IEA-15-240-RWT_r0329.out`, which ends
with a cross-check against the blade file's own `elastic_properties_mb`. WISDEM computes those from
continuous thicknesses while OpenSG quantizes every layer to whole plies, so the two legitimately
differ where a layer is near a half-ply boundary:

% source: examples/pynumad/IEA-15-240-RWT_r0329.out
| Term | OpenSG | file | difference |
|---|---|---|---|
| $GA_3$ | 1.802099E+08 | 2.033162E+08 | -11.36 % |
| $GA_2$ | 3.628222E+08 | 3.337354E+08 | +8.72 % |
| $EA$ | 2.410208E+10 | 2.160810E+10 | +11.54 % |
| $EI_3$ | 2.890696E+10 | 2.818818E+10 | +2.55 % |
| $EI_2$ | 1.299092E+10 | 1.180563E+10 | +10.04 % |
| $GJ$ | 5.136618E+08 | 5.313826E+08 | -3.33 % |
| $\mu$ | 4.947839E+02 | 4.690433E+02 | +5.49 % |

## Files

| File | Content |
|---|---|
| `IEA-22-280-RWT.yaml` | the windIO v2 blade |
| `1_windio_to_cross_sections.py` to `7_verify_blade_timo_vs_K.py` | the seven scripts |
| `iea22_stations.dat`, `iea22_timo_by_r.dat`, `iea22_mass_by_r.dat` | station table and spanwise $6\times6$ tables (steps 1 and 2) |
| `props/iea22_rXXXX.K` | per-station mass and Timoshenko $6\times6$, 16 files (step 2) |
| `beamdyn/iea22_bd_props.inp`, `iea22_bd_primary.inp`, `iea22_bd_driver.inp` | BeamDyn inputs (steps 3 and 4) |
| `beamdyn/iea22_bd_driver.out`, `iea22_bd_primary.inp.sum` | BeamDyn output and summary (step 5) |
| `ff/iea22_rXXXX.ff`, `ff/iea22_ff.dat` | per-station recovery inputs and their table (step 5) |
| `vabs_K/iea_s10.sg.K`, `iea_s25.sg.K`, `iea_s35.sg.K`, `iea_s10.sg` | VABS 2-D solid references |

`cross_sections/` (step 1) and `recovery/` (step 6) are regenerated by the scripts and not shipped.
