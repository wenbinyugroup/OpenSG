# pyNuMAD blade -> OpenSG cross-sections and beam properties

The pyNuMAD blade dialect, SEPARATE from the windIO example (`examples/windio`).
pyNuMAD exports (here `IEA-15-240-RWT.yaml` from sandialabs/pyNuMAD
`examples/example_data`) look like windIO v1 but place the TE/LE
reinforcements with width-based forms (`start/end/midpoint_nd_arc` +
`width` [m]) that the plain windio reader would drop from the laminate --
on this blade that is up to -84 % edgewise stiffness and -19 % mass per
span.  The `opensg_shell.pynumad` package resolves those placements against
the section perimeter and never drops material silently.

## The command

```bash
opensg pynumad <blade.yaml> <st-id> [flags]
```

`st-id` selects the station, one of:

- a 0-BASED integer index into the blade file's OWN spanwise stations
  (`0` = root; an out-of-range id prints the station list),
- a span fraction r between 0 and 1 -- any token WITH a decimal point
  (e.g. `0.3288`),
- `all` -- every station of the file.

Output names derive from the blade file stem: station tag =
`<stem>_rXXXX`, `XXXX = round(r * 1000)` (there is no prefix flag).

## One station

```bash
opensg pynumad IEA-15-240-RWT.yaml 4
```

prints the Timoshenko 6x6 and the 6x6 mass matrix at station 4
(r = 0.3288) and stores EXACTLY three files:
`IEA-15-240-RWT_r0329_shell.yaml` + `IEA-15-240-RWT_r0329_shell_Timo.out`
+ the VABS-layout `IEA-15-240-RWT_r0329.out` (mass blocks, stiffness /
compliance, centers, and the cross-check block against the file's own
`elastic_properties_mb` -- the pyNuMAD/WISDEM 6x6, 3 = axial frame).
No ABDG record, no abd/ cache, no PNGs.  Opt extra artifacts in:

- `--xml`   also write the PreVABS XML byproduct (`xml/<tag>/`)
- `--view`  also write the layup-colored mesh PNG + e1/e2/e3 orientation PNG

## All stations

```bash
opensg pynumad IEA-15-240-RWT.yaml all
```

homogenizes EVERY station: prints the Timoshenko 6x6 per station and
stores, under `cross_sections/`, the per-station 1-D shell SG yaml +
VABS-layout `.out` (+ PreVABS XML, `--no-xml` to skip) + mesh/orientation
PNGs, plus three spanwise tables: `<stem>_stations.dat`,
`<stem>_timo_by_r.dat`, `<stem>_mass_by_r.dat`.

## Flags and conventions

- `--reference` DEFAULTS TO `oml` for this dialect (the outer mold line is
  the shell reference surface; `--reference center` for the laminate
  mid-surface, the windio route's default).
- `--mesh-size` (default 0.01) = target element arc length / chord;
  `--out` redirects the output folder (default: current directory for one
  station, `cross_sections/` for `all`).
- Section origin = the pitch axis (`pitch_axis * chord` chordwise shift);
  emitted headers carry `refined: 1` (RM -> Timoshenko); twist is passed
  through in the file's own unit (radians here).  Stations where a web's
  layers have zero thickness (the circular root, the tip) get no web,
  matching the file.
