---
name: prevabs-xml-to-2dyaml
description: Turn a PreVABS cross-section XML into a VABS .sg mesh and then into an OpenSG 2-D solid YAML carrying per-element material orientations. Use when a 2-D solid SG (with elementOrientations) is needed from a cross-section definition — a thin-walled tube, box, or airfoil section — including the CENTER (mid-thickness) layup-reference construction, the prevabs invocation, the .sg -> YAML conversion, and the verification checks.
---

# PreVABS XML → 2-D solid YAML

Produces, from a cross-section description:

| file | what it is |
|---|---|
| `<name>.xml` + `materials.xml` | PreVABS input (geometry + layup, material DB) |
| `<name>.sg` | VABS-format 2-D mesh: nodes, connectivity, per-element layup group and contour angle θ1, per-group fibre angle θ3 |
| `<name>_2Dsolid.yaml` | OpenSG 2-D solid SG: `nodes`, `elements`, `sets.element`, `materials`, **`elementOrientations`** |

Voigt/frame conventions downstream: axis 1 = beam axis (z in the YAML node triples), 2/3 = cross-section.

## 1. Write the PreVABS XML

**CENTER (mid-thickness) layup reference — the key construction.** PreVABS always
builds a layup to *one side* of its baseline; there is no "center" option. To get a
wall centred on a required midline, offset the midline **outward by t/2** and use that
as the baseline, then lay the plies **inward**:

```
midline square side a          ->  baseline = outer profile, side a + t
single ply of thickness t      ->  wall spans (a - t) .. (a + t) about the midline
```

Layup side is *relative to the direction of travel* along the baseline: listing the
points **clockwise** (negative signed area) makes `direction="right"` point **into**
the section, i.e. inward. Repeat the first point last to close the contour.

```xml
<cross_section name="square_tube" format="1">
  <include><material>materials</material></include>
  <analysis><model>1</model></analysis>          <!-- 1 = Timoshenko beam -->
  <general>
    <mesh_size>0.0075</mesh_size>                <!-- ~ t/4 -> 4 elements through t -->
    <element_type>linear</element_type>
  </general>
  <baselines>
    <point name="otl">-0.515  0.515</point>      <!-- midline +-0.5 offset by t/2 -->
    <point name="otr"> 0.515  0.515</point>
    <point name="obr"> 0.515 -0.515</point>
    <point name="obl">-0.515 -0.515</point>
    <baseline name="bl_wall" type="straight">
      <points>otl,otr,obr,obl,otl</points>       <!-- clockwise, closed -->
    </baseline>
  </baselines>
  <layups>
    <layup name="lyp_wall">
      <layer lamina="la_m45">-45:1</layer>       <!-- angle:n_plies -->
    </layup>
  </layups>
  <component name="wall">
    <segment name="sg_wall">
      <baseline>bl_wall</baseline>
      <layup direction="right">lyp_wall</layup>  <!-- right of travel = inward -->
    </segment>
  </component>
</cross_section>
```

`materials.xml` — the ply angle lives in the `<layup>`, NOT here; a `<lamina>` is the
un-rotated material plus its thickness. Keep units consistent (SI: m, Pa, kg/m³):

```xml
<materials>
  <material name="m45" type="orthotropic">
    <density>1600</density>
    <elastic>
      <e1>142.0e9</e1><e2>9.8e9</e2><e3>9.8e9</e3>
      <g12>6.0e9</g12><g13>6.0e9</g13><g23>4.8e9</g23>
      <nu12>0.30</nu12><nu13>0.30</nu13><nu23>0.42</nu23>
    </elastic>
  </material>
  <lamina name="la_m45"><material>m45</material><thickness>0.03</thickness></lamina>
</materials>
```

## 2. Run PreVABS

Canonical flags (as `OpenSG_io/scripts/opensg_io.py:_run_solid_from_xml` uses them) —
`--vabs` writes the `.sg`, `--hm` builds the homogenization mesh. Run **in the XML's
own directory**; PreVABS writes `<name>.sg`, `<name>.sg.mat`, `<name>.log`,
`<name>.debug.log` beside it:

```bash
cd <dir with the xml>
<prevabs> -i <name>.xml --vabs --hm
```

Binaries in use:
- Linux (server, used by the OpenSG_io wrapper):
  `~/OpenSG_io/third_party/prevabs_bin/prevabs-v2.1.0-preview.20260508.3-linux-rhel9-x64/prevabs`
- Windows local:
  `C:\Users\bagla0\OneDrive - purdue.edu\2026_195\PreVABS\prevabs-v2.0.1-windows-x64-full\prevabs-v2.0.1-windows-x64-full\prevabs.exe`

Check `<name>.log` for `OK  prevabs finished`. `<name>.debug.log` is verbose; the
`<nu23> not specified` warnings come from the shipped `MaterialDB.xml`, not from your
materials file — ignore them.

## 3. Convert .sg → 2-D solid YAML

```bash
python ~/OpenSG_io/scripts/convert_sg_to_yaml.py <name>.sg <name>_2Dsolid.yaml
```

It writes two files: the named one (θ1 **and** θ3 — the correct frame) and a
`_t1only.yaml` sibling (θ1 only). **Use the θ1+θ3 file**; the t1-only variant drops the
fibre rotation and is kept only for comparison.

How the frame is built (`frame()` in the converter): θ1 is the per-element contour
angle setting the ply plane in the cross-section; θ3 is the per-group fibre angle
rotating the fibre about the ply normal:

```
e1 (fibre)      = [sin θ3 cos θ1, sin θ3 sin θ1, cos θ3]
e2              = [cos θ3 cos θ1, cos θ3 sin θ1, -sin θ3]
e3 (ply normal) = [-sin θ1, cos θ1, 0]
```

so `e3` is always in-plane (z-component 0) and `e1`'s z-component is `cos θ3` — for a
±45° ply that is ±0.7071.

The emitted YAML uses **flat** material keys (`m["E"]`, `m["G"]`, `m["nu"]`, `m["rho"]`)
— *not* `m["elastic"]["E"]`. Materials are named `Material_1`, `Material_2`, … in .sg
phase order; the converter's `NAMES` dict only affects console printing, so a
single-material section printing "gelcoat" is cosmetic.

## 4. Verify before using the YAML

Always check, and report:

- **counts**: `len(nodes)`, `len(elements)`, `len(elementOrientations)` — the last two must match.
- **no inverted elements**: signed area `det > 0` for every element.
- **wall area** vs the analytical value. For a square annulus with mitred corners,
  `A = 4·a·t = (a+t)² − (a−t)²` — these coincide, and PreVABS reproduced 0.12000000 exactly for a = 1, t = 0.03.
- **orthonormal frames**: `max|eᵢ·eⱼ − δᵢⱼ|` ~ 1e-30.
- **fibre angle**: `arccos(|e1_z|)` should equal the ply angle magnitude (45° for ±45).
- **ply normal in-plane**: `max|e3_z| = 0`.
- **bbox** matches the outer profile (±(a+t)/2), confirming the CENTER construction.

Render two PNGs: the actual mesh (elements with edges, equal aspect, `y2`/`y3` labels,
**no figure title**) and the e1/e2/e3 orientation-arrow plot — this project wants the
orientation triad plot for every oriented mesh.

`tests/08072026_square_shell_mesh/check_2dyaml.py` is a working implementation of all
of the above; copy it as the starting point.

## Worked example in this repo

`tests/08072026_square_shell_mesh/` — square tube, midline side a = 1, t = 0.03,
single orthotropic ply at −45°:

```
prevabs/square_tube.xml, prevabs/materials.xml   ->  prevabs/square_tube.sg
convert_sg_to_yaml.py                            ->  square_tube_2Dsolid.yaml
check_2dyaml.py                                  ->  square_tube_2Dsolid.png
```

Result: 3228 nodes, 5384 linear triangles, area 0.12000000 (exact), 0 inverted,
frames orthonormal to 7.5e-33, e1_z = 0.7071 everywhere (45° from the beam axis),
e3_z = 0.

## Gotchas

- Getting the layup on the wrong side is the most common error: the section comes out
  offset by t. Check the bbox against the intended outer profile.
- `mesh_size` ≈ t/4 gives ~4 elements through the thickness; the soft bending-dominated
  stiffness terms need at least that.
- If gmsh inside PreVABS fails with "unable to recover the edge", the geometry has a
  too-thin or degenerate region — coarsen `mesh_size` or fix the offending baseline.
- Run commands over ssh with single-quoted strings from PowerShell (`\"` for embedded
  quotes) or write a script file and run it by path; PowerShell mangles quotes otherwise.
- Filter run output through `grep -v -i "leaked\|nanobind"` — basix prints a harmless
  leak dump at exit.
