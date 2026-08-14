# windIO blade -> beam properties -> BeamDyn -> stress recovery (msg-shell)

The full OpenSG shell pipeline from a windIO blade file, standalone (no external
wrapper).  Core modules: `src/opensg_shell/windio/` (`sg_windio`, `sg_props`,
`sg_beamdyn`, `sg_recovery`).

| step | script | in | out |
|------|--------|----|-----|
| 1 | `1_windio_to_cross_sections.py` | `IEA-22-280-RWT.yaml` | `cross_sections/<tag>_shell.yaml` + mesh/orient PNGs |
| 2 | `2_get_beam_props_from_shell_cross_section.py` | station yamls | `props/<tag>.K` (VABS .K layout: mass 6x6 + Timoshenko 6x6) |
| 3 | `3_convert_to_beamdyn_props.py` | station `.K` | `beamdyn/<prefix>_bd_props.inp` |
| 4 | `4_prepare_beamdyn_input.py` | windIO + `.K` list | `beamdyn/<prefix>_bd_{primary,driver}.inp` (nodal loads) |
| 5 | `5_run_beamdyn_extract_ff.py` | driver input | `ff/<tag>.ff` (u, theta, DCM, F, M -- VABS frame) |
| 6 | `6_dehom_stress_recovery.py` | yaml + `.ff` | `recovery/<tag>.{SM,EM,U}` + stress PNG |

Station tag = `<prefix>_rXXXX`, `XXXX = round(r * 1000)`.

Step 1 also has a terminal route -- all stations, PreVABS XML byproduct ON by
default, mesh + orientation PNGs and a station table under `cross_sections/`:

```bash
opensg gen_windio_cs IEA-22-280-RWT.yaml
```

(`gen_windio_cs --help` for `--stations`, `--mesh-size`, `--reference`,
`--no-xml`, `--out`, `--prefix`.)  The emitted yamls carry `refined: 1` and
`reference: center` in the header, so `opensg <station yaml>` runs the
shear-refined RM -> Timoshenko route directly.

Station count is free: `stations = "airfoil"` uses the blade's own windIO airfoil
positions (IEA-22: 16); `stations = 51` puts 51 uniform stations `r = i/50` (the
grid of the VABS 51-station reference set); any explicit list of `r` works too.

Run (server env `opensg_2_0`):

```bash
PY=~/miniconda3/envs/opensg_2_0/bin/python
$PY 1_windio_to_cross_sections.py
$PY 2_get_beam_props_from_shell_cross_section.py
$PY 3_convert_to_beamdyn_props.py
$PY 4_prepare_beamdyn_input.py
$PY 5_run_beamdyn_extract_ff.py     # needs beamdyn_driver (conda-forge openfast)
$PY 6_dehom_stress_recovery.py
```

Conventions

- Section origin = the windIO **reference axis x1** (chordwise `section_offset_y` /
  `pitch_axis` shift); the 6x6, the BeamDyn loads, and the recovery all share it.
- Shell reference surface = laminate **center** (mid-surface), recorded in the yaml's
  `reference` field (single source of truth for homogenization and dehom).
- Cross-section matrices are VABS order/frame (1 = axial); BeamDyn files are written
  in the IEC blade frame via the beam-axis swap `B = [[0,0,1],[0,-1,0],[1,0,0]]`.
- BeamDyn runs TRAPEZOIDAL quadrature so output nodes coincide with the property
  stations; `.ff` resultants are the section-LOCAL (`FxL`/`MxL`) channels.
- `.SM`/`.EM` are Gauss-point stress/strain in the MATERIAL frame, VABS column order
  `x2 x3 s11 s12 s13 s22 s23 s33`; `.U` is `id x2 x3 u1 u2 u3` total displacement.

Verification: `vabs_K/` holds VABS 2-D-solid `.K` references (51-station IEA-22 set,
stations s10/s25/s35 = eta 0.2/0.5/0.7).  The unit test
`tests/msg_shell_windio/test_vabs_k_beam_props.py` runs steps 1-2 at those stations
and gates the Timoshenko diagonals and the mass blocks against VABS:

```bash
$PY -m pytest tests/msg_shell_windio -q
```
