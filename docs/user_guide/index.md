# Guide for preparing input files

In OpenSG, a structure gene (SG) is the smallest piece of a structure that carries its heterogeneity: a laminate
through the thickness, a beam cross-section, or a unit cell. Each SG is described by one yaml file, and the file is the
whole problem: it states the analysis it asks for, the finite-element mesh of the SG, the material of every element and
the material frame it is oriented in. The command `opensg <sg.yaml>` reads the file and runs it; there is no separate
control file.

Two dialects of the SG yaml exist, one for each engine of OpenSG, as listed in {numref}`tab-ug-dialects`. The shell
dialect describes a laminated surface: the mesh is the wall of a thin-walled structure, and each group of elements
carries a layup. The solid dialect describes a meshed domain: every element is solid material with its own stiffness.

% source: src/opensg_solid/sg_mesh.py (sg_dialect, resolve_msg); docs/input_format.md
```{table} The two SG yaml dialects and the engine that reads each.
:name: tab-ug-dialects

| dialect | engine | SG | mesh blocks | example input |
|---|---|---|---|---|
| shell | `opensg_shell` | 1-D shell SG (a cross-section contour of 2-node lines) or 3-D shell SG (a surface of 3-node triangles and 4-node quads) | `nodes`, `elements`, `sets`, `sections`, `elementOrientations`, `materials` | `examples/OpenSG_shell/1_get_beam_props_from_shell_cross_section/iea_s10_shell.yaml` |
| solid | `opensg_solid` | 1-D, 2-D or 3-D SG of solid elements | `nodes`, `cells`, `mat_id`, `materials`; or `nodes`, `elements`, `sets`, `elementOrientations`, `materials`; or, for the generated 1-D plate SG, `nodes`, `elements`, `sets`, `sections`, `materials` without `elementOrientations` | `examples/OpenSG-solid/3_get_plate_props_from_2D_SG/RHC_SW_2UC_45.yaml` |
```

## Anatomy of an SG yaml

Every SG yaml has two parts, in this order:

- **The header**: a few scalar keys at the top of the file (`msg`, `n_model`, `refined`, ...). It is the analysis request
  the file makes about itself: which engine owns the file, which macro model the SG is homogenized to, and whether the
  run is a homogenization or a dehomogenization. Every key has a default.
- **The mesh blocks**: the node coordinates, the element connectivity, the grouping of elements, the material (and, in
  the shell dialect, the laminate) of each group, and the material frame of each element.

{numref}`lst-ug-shell-layout` shows the layout of a shell SG yaml, with one row kept from each block of the IEA-22
blade cross-section used in the tutorial.

% source: examples/OpenSG_shell/1_get_beam_props_from_shell_cross_section/iea_s10_shell.yaml (abridged)
```{code-block} yaml
:caption: Layout of a shell SG yaml (iea_s10_shell.yaml, one row per block).
:name: lst-ug-shell-layout

n_model: 1      # 1 = beam, 2 = plate (opensg_solid), 3 = solid -- the macro model this SG homogenizes to
refined: 1      # 0 = classical (beam EB 4x4, KL wall); 1 = shear-refined (beam Timoshenko 6x6, RM wall)
msg: shell      # the ENGINE this SG belongs to (opensg_shell); `opensg <yaml>` dispatches on it
nodes:
- [4.70852238 0.09722431 0.00000000]
elements:
- [1 2]
sets:
  element:
  - name: layup_0
    labels:
    - 1
sections:
- type: shell
  elementSet: layup_0
  layup:
  - - gelcoat
    - 0.0005
    - 0.0
elementOrientations:
- [0.0, 0.0, 1.0, -0.9721541292349604, 0.2343423756204075, 0.0, -0.2343423756204075,
  -0.9721541292349604, 0.0]
materials:
- name: gelcoat
  density: 1235.0
  elastic:
    E:
    - 3440000000.0
    - 3440000000.0
    - 3440000000.0
    G:
    - 1323000000.0
    - 1323000000.0
    - 1323000000.0
    nu:
    - 0.3
    - 0.3
    - 0.3
```

{numref}`lst-ug-solid-layout` shows the same for a solid SG yaml, a two-unit-cell honeycomb plate SG. Here the material
of each element is a material id, and each material block carries its own engineering constants and ply angle.

% source: docs/input_format.md (excerpt of examples/OpenSG-solid/3_get_plate_props_from_2D_SG/RHC_SW_2UC_45.yaml)
```{code-block} yaml
:caption: Layout of a solid SG yaml (RHC_SW_2UC_45.yaml, abridged).
:name: lst-ug-solid-layout

n_model: 2      # 1 = beam, 2 = plate, 3 = solid -- the macro model this SG homogenizes to
refined: 1      # 0 = classical (plate ABD / beam EB); 1 = shear-refined (plate ABDG / beam Timoshenko)
msg: solid      # the ENGINE this SG belongs to (opensg_solid); `opensg <yaml>` dispatches on it
nodes:
- [17.6240005, 23.3500004, 0.0]
- [-17.6240005, 23.3500004, 0.0]
cells:
- [0, 66, 345]
- [345, 3, 0]
mat_id: [1, 1, 1, 1, 1]
materials:
  1:
    type: 1
    engineering: [108000.0, 8000.0, 8000.0, 4000.0, 4000.0, 3000.0, 0.32, 0.32, 0.30]
    angle: 45.0
  3:
    type: 1
    engineering: [69000.0, 69000.0, 69000.0, 26540.0, 26540.0, 26540.0, 0.30, 0.30, 0.30]
    angle: 0.0
```

## Which engine reads the file

The `msg:` header key names the engine: `msg: shell` sends the file to `opensg_shell`, `msg: solid` to
`opensg_solid`. Without the key, the mesh blocks decide: a file with `cells` or `mat_id` is solid, a file with
`elements`, `sections` and `elementOrientations` together is shell, and any other file with `elements` is solid. A
`msg:` that contradicts the mesh blocks is an error, and so is a file that has neither the key nor a recognizable mesh
block. The three commands below run the same solid SG; `opensg` reads `msg: solid` from the header and forwards the
file to the solid engine.

% source: docs/input_format.md ("The solid SG YAML")
```bash
opensg RHC_SW_2UC_45.yaml
opensg_solid RHC_SW_2UC_45.yaml
python -m opensg_solid RHC_SW_2UC_45.yaml
```

:::{note}
A file with no header still runs. A solid SG then runs as a classical plate homogenization, and a shell SG as a
classical beam homogenization with the node contour taken as the outer mold line.
:::

## Where each part is described

- Header keys (`msg`, `n_model`, `refined`, `analysis`, `epsilon_bar`, `omega`, `aperiodic`, the `junction` keys) and
  the `--center` run flag: {doc}`analysis_settings`.
- Node coordinates, the material frame in `elementOrientations`, the ply angle, and the order of strains and stiffness
  entries in every law: {doc}`coordinate_systems`.
- `nodes`, `elements`, `cells`, `sets`, the SG dimension and the finite elements of each route: {doc}`mesh_and_elements`.
- `materials`, `sections`, `mat_id`, the material library and the through-thickness `layup_db.yaml`:
  {doc}`materials_and_layups`.
- The macro state of a dehomogenization (`epsilon_bar` and the `<base>.ff` table): {doc}`recovery`.
- Building an SG yaml from a gmsh `.msh`, an Abaqus `.inp` or a SwiftComp `.sc`, and writing a `.sc` back:
  {doc}`converters`.
- The `.out` files and the other files a run writes: {doc}`output_files`.

```{toctree}
:maxdepth: 1

analysis_settings
coordinate_systems
mesh_and_elements
materials_and_layups
recovery
converters
output_files
```
