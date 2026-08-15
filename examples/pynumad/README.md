# pyNuMAD blade -> OpenSG cross-sections and beam properties

The blade route of OpenSG. pyNuMAD exports (here `IEA-15-240-RWT.yaml`
from sandialabs/pyNuMAD `examples/example_data`) look like windIO v1 but
place the TE/LE reinforcements with width-based forms
(`start/end/midpoint_nd_arc` + `width` [m]) that a plain windIO reader
would drop from the laminate -- on this blade that is up to -84 % edgewise
stiffness and -19 % mass per span.  The `opensg_shell.pynumad` package
resolves those placements against the section perimeter and never drops
material silently.  windIO v1 and v2 blades run through the same route.

## The command

```bash
opensg pynumad <blade.yaml> [st-id] [flags]
```

`st-id` selects the station and is OPTIONAL -- omitted, it defaults to
`all` (the full sweep).  It is one of:

- a 0-BASED integer index into the blade file's OWN spanwise stations
  (`0` = root; an out-of-range id prints the station list),
- a span fraction r between 0 and 1 -- any token WITH a decimal point
  (e.g. `0.3288`),
- `all` -- every station of the file.

One station prints the Timoshenko 6x6 and the mass 6x6 and stores exactly
ONE file: the VABS-layout `<stem>_rXXXX.out` record (`XXXX =
round(r * 1000)`; stiffness / compliance, centers, mass blocks, and the
cross-check table against the file's own `elastic_properties_mb` when the
block is present).  `all` sweeps every station -- one `.out` per station
plus three spanwise tables: `<stem>_stations.dat`,
`<stem>_timo_by_r.dat`, `<stem>_mass_by_r.dat`.  Everything runs in
memory; no yaml, XML or PNG byproducts.

**Output location: the directory you run the command from.**  There is no
output-folder flag -- `cd` to where you want the records and run it there.

## User inputs (flags)

- `--mesh-size H` -- target element arc length / chord (default 0.01).
- `--center` -- put the shell on the laminate **mid-surface** instead of the
  OML.  Omit it and the reference is the **OML** (always the default): the
  airfoil contour in the blade yaml IS the outer mold line, so the shell
  sits on the geometry the file states with the laminate hanging inward.
  Use `--center` when matching a 2-D solid / VABS section.  The surface
  used is named in the `.out` banner.
- `--xml` -- ALSO write the PreVABS XML byproduct per station
  (`{tag}.xml` + `{tag}.dat` + `materials.xml` under `xml/<tag>/` **in the
  blade yaml's own directory**) -- the cross-check input for the
  XML -> prevabs -> 2-D-solid pathway.

That is the whole flag surface.

## The Blade class (optimization workflow)

```python
import opensg

blade = opensg.read("IEA-15-240-RWT.yaml")   # windIO v1/v2 or pyNuMAD dialect
K, M = blade(4)                              # one station: Timoshenko + mass 6x6
Ks, Ms = blade()                             # full spanwise sweep (two lists)
```

Everything computes from the Blade's CURRENT state, fully in memory.
The user-editable inputs:

```python
blade.scale_layer_thickness("Spar_Cap_SS", 1.2)   # spanwise layer move
blade.set_material("glass_triax", E=[...])        # any material constant
blade.chord, blade.twist, blade.offset            # geometry tables (edit in place)
blade.layers, blade.webs, blade.materials, blade.airfoils
blade.update_blade()                              # re-sync after add/remove edits
R = blade.timo(4)                                 # R["Timo"], R["Mass"], R["info"]
rows = blade.timo_all()                           # sweep + spanwise .dat tables
```

Value edits propagate immediately; after adding/removing layers, webs,
materials or airfoils call `update_blade()`.  `blade.timo(st,
artifacts=True)` additionally writes the station yaml + `.out` record into
`blade.workdir` (the handoff to the opensg_shell SG homo/dehom engine).
See `blade_optimization_demo.py` for the design-sweep pattern.

### Reference surface

`blade.timo()` -- the optimization API -- always uses the **OML**, because
the airfoil contour in the blade yaml IS the outer mold line: the shell
sits on the geometry the file states, with the laminate hanging inward.

The terminal route lets you choose per run:

```bash
opensg pynumad IEA-15-240-RWT.yaml 4             # OML (always the default)
opensg pynumad IEA-15-240-RWT.yaml 4 --center    # laminate mid-surface
```

`--center` displaces every skin node inward by half the local laminate
thickness along the averaged inward normal (connectivity unchanged, webs
re-chained between the moved attachment points) AND re-references the ABD
and the mass moments to that same surface, so geometry and wall law stay
consistent.  On IEA-15 it moves EA by about -1.3% and mass by -1.7%.

From Python, either driver takes it directly:

```python
from opensg_shell.pynumad import station_timo
P = station_timo("IEA-15-240-RWT.yaml", 4, reference="center")
```

## Per-section edits (Section)

A **Section** is one station's resolved layup (the pyNuMAD 12-region
stacks plus the webs), returned as an editable object and registered as a
live override -- the next `blade(st)` computes from the edit, `reset()`
returns to the definition.  Unlike `scale_layer_thickness` (a spanwise
move on the definition), a Section edit touches exactly one station:

```python
S = blade.section(4)                  # {region: [[mat, t, ang], ...]} + S.webs
S.scale_thickness(1.3, region="LP_SPAR")
S.set_ply("HP_SPAR", 2, thickness=0.12)      # ply 2 = the CarbonUD cap ply
S.add_ply("LP_TE_REINF", "glass_triax", 0.002, angle=45.0)
S.scale_thickness(1.1, web=0)                # webs by index
K2, M2 = blade(4)                            # honors the edits
S.reset()                                    # back to the definition
```

An untouched Section reproduces the baseline digit-for-digit (gated in
`tests/msg_shell_windio/test_blade_class.py`); edited thicknesses are used
verbatim, with no ply re-quantization.  The un-edited route quantizes every
layer to whole plies, exactly as pyNuMAD's stack database does (the
realistic manufactured layup) -- so cross-checks against the file's
`elastic_properties_mb`, which WISDEM computes from the CONTINUOUS
thickness distributions, legitimately differ where a layer is near a
half-ply boundary (most visibly the sub-ply tip spar caps).
`blade_section_edit_demo.py` shows the
spar-cap move: x1.3 at station 4 gives EA +25.4 %, flapwise EI2 +24.4 %,
edgewise EI3 +1.3 %.

## Verification

`test_timo_pynumad.py` (also a pytest unit test) benchmarks `blade.timo()`
on the IEA-22 blade against TRUE VABS 2-D solid sections
(`examples/OpenSG_shell/windio/vabs_K/iea_s{10,25,35}.sg.K`): Timoshenko
diagonals and the mass matrix, at the measured OML-shell-vs-solid
tolerance bands.  `7_verify_blade_timo_vs_K.py` (in
`examples/OpenSG_shell/windio`) gates the same route against the
committed 16-station `.K` dataset at solver precision.
