# pyNuMAD blade -> OpenSG cross-sections and beam properties

The pyNuMAD blade dialect, SEPARATE from the windIO example (`examples/windio`).
pyNuMAD exports (here `IEA-15-240-RWT.yaml` from sandialabs/pyNuMAD
`examples/example_data`) look like windIO v1 but place the TE/LE
reinforcements with width-based forms (`start/end/midpoint_nd_arc` +
`width` [m]) that the plain windio reader would drop from the laminate --
on this blade that is up to -84 % edgewise stiffness and -19 % mass per
span.  The `opensg_shell.pynumad` package resolves those placements against
the section perimeter and never drops material silently.

One command, st-id = a 0-based index into the blade file's OWN spanwise
stations, a span `r` in [0, 1] (any token with a decimal point), or `all`:

```bash
opensg pynumad IEA-15-240-RWT.yaml 4 --prefix iea15
```

prints the Timoshenko 6x6 at station 4 (r = 0.3288), the cross-check table
against the file's own `elastic_properties_mb` (pyNuMAD/WISDEM 6x6, 3 =
axial frame), and stores `<tag>_shell.yaml` + `<tag>_shell_Timo.out` +
the VABS-layout `<tag>.K` (nothing else: no ABDG record, no abd/ cache,
no PNGs on the single-station route).  Add `--xml` (PreVABS XML byproduct)
and/or `--view` (mesh + orientation PNGs) to opt extra artifacts in.

```bash
opensg pynumad IEA-15-240-RWT.yaml all --prefix iea15
```

generates every station: 1-D shell SG yaml (+ PreVABS XML byproduct,
`--no-xml` to skip) + mesh/orientation PNGs + `iea15_stations.dat` under
`cross_sections/`.

Conventions match the windIO route: section origin = the pitch axis
(`pitch_axis * chord` chordwise shift), shell reference surface = laminate
center, emitted headers carry `refined: 1` (RM -> Timoshenko), twist is
passed through in the file's own unit (radians here).  Stations where a
web's layers have zero thickness (the circular root, the tip) get no web,
matching the file.
