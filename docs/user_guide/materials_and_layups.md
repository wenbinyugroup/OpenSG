# Materials and layups

OpenSG reads every constitutive input from the structure gene (SG) yaml itself: nothing about a
material lives in code or on the command line. The two engines spell materials differently,
because they describe different things.

- A **shell SG** (`msg: shell`) is a laminated surface. Its `materials:` list names each
  material once, and each entry of `sections:` stacks named materials into a layup, one row per
  ply with a thickness and a ply angle. OpenSG integrates every layup to a wall plate law before
  it solves the SG.
- A **solid SG** (`msg: solid`) is a meshed domain. Every element carries one material, given as
  isotropic constants, as the nine engineering constants plus a ply angle, or as a stored
  $6\times6$ stiffness. There is no layup: a laminate is a stack of elements with their own
  materials.

Two further inputs complete the picture. A plain laminate homogenized through its thickness
needs no mesh at all: the database `layup_db.yaml` is the whole input of the 1-D plate SG
driver. A gmsh mesh carries no material, so `opensg msh_to_yaml` fills the solid material blocks
by name from a material library.

## Elastic constants

A material given by elastic constants uses the nine orthotropic engineering constants. The shell
dialect, the solid list spelling and the material library write them as three triples in this
order:

% source: src/opensg_shell/kernels/msg_materials.py (build_stiffness_6x6); docs/input_format.md
| key | components |
|---|---|
| `E` | $E_1$, $E_2$, $E_3$ |
| `G` | $G_{12}$, $G_{13}$, $G_{23}$ |
| `nu` | $\nu_{12}$, $\nu_{13}$, $\nu_{23}$ |

OpenSG assembles the compliance in the solid Voigt order $[11\;22\;33\;23\;13\;12]$ with
engineering shears and inverts it to the $6\times6$ stiffness:

% source: src/opensg_shell/kernels/msg_materials.py (build_stiffness_6x6)
$$
S_{11} = \frac{1}{E_1},\; S_{22} = \frac{1}{E_2},\; S_{33} = \frac{1}{E_3},\;
S_{44} = \frac{1}{G_{23}},\; S_{55} = \frac{1}{G_{13}},\; S_{66} = \frac{1}{G_{12}},\;
S_{12} = -\frac{\nu_{12}}{E_1},\; S_{13} = -\frac{\nu_{13}}{E_1},\; S_{23} = -\frac{\nu_{23}}{E_2}.
$$

Axes 1, 2, 3 are the material axes $e_1, e_2, e_3$ of {doc}`coordinate_systems`. The ply angle
rotates the material about the layer normal, away from $e_1$: about $e_3$, the wall normal, in
the shell dialect, and about the SG thickness axis $y_3$ in the solid dialect, so `angle: 0.0`
places the fibre along $e_1$.

:::{note}
Units follow the input; OpenSG performs no unit conversion. The SI examples give moduli in Pa,
thicknesses in m and densities in kg/m³; the honeycomb and micromechanics examples are in MPa and
mm. Keep one unit system per file.
:::

## Shell materials and sections

% source: docs/input_format.md (sets and sections, materials); src/opensg_shell/sg_materials.py
| key | type | meaning | default |
|---|---|---|---|
| `materials[].name` | string | the name a layup row refers to | required |
| `materials[].density` | float | mass density; feeds the section mass per unit area, not the stiffness | 0.0 |
| `materials[].elastic.E`, `.G`, `.nu` | 3 floats each | the triples of the table above | required |
| `sets.element[].name` | string | name of an element set | required |
| `sets.element[].labels` | list of int | 1-based element labels of the set | required |
| `sections[].elementSet` | string | the set this section applies to; must equal a set `name` | required |
| `sections[].layup` | list of `[material, thickness, angle_deg]` | the plies, **outer face first** | required |
| `sections[].type` | string | `shell`; descriptive only, not read | none |

The excerpt below is abridged from
`examples/OpenSG_shell/1_get_beam_props_from_shell_cross_section/iea_s10_shell.yaml`, where
`layup_0` lists 25 labels (elements 1 to 12 and 209 to 221) and there are six sets, six sections
and six materials in all.

% source: examples/OpenSG_shell/1_get_beam_props_from_shell_cross_section/iea_s10_shell.yaml
```yaml
sets:
  element:
  - name: layup_0
    labels:
    - 1
    - 2
    - 3
sections:
- type: shell
  elementSet: layup_0
  layup:
  - - gelcoat
    - 0.0005
    - 0.0
  - - glass_triax
    - 0.00300294
    - 0.0
  - - glass_uniax
    - 0.02609467
    - 0.0
  - - glass_triax
    - 0.00300294
    - 0.0
materials:
- name: glass_triax
  density: 1940.0
  elastic:
    E:
    - 28211400000.0
    - 16238800000.0
    - 15835500000.0
    G:
    - 8248220000.0
    - 3491240000.0
    - 3491240000.0
    nu:
    - 0.497511
    - 0.18091
    - 0.27481
```

Each layup row stacks inward along $e_3$ from the outer face, and the sum of the row thicknesses
is the wall thickness. Which surface of that wall the node contour is (the outer mold line by
default, the mid-surface with `--center`) is a run-time choice described in
{doc}`analysis_settings` ({numref}`fig-mat-shell-sections`).

```{figure} ../_static/shell_xsec_mesh.png
:width: 80%
:name: fig-mat-shell-sections

The IEA-22 blade cross-section at $r/R = 0.20$, elements coloured by section (layup).
```

:::{warning}
Put every element in exactly one set. An element that appears in no set is assigned to the first
section (section 0) without a message.
:::

## Solid materials

The solid dialect has two spellings of the same information.

### Material mapping with `mat_id`

The canonical solid yaml (`nodes`, `cells`, `mat_id`, `materials`) gives one material id per cell
in `mat_id` (1-based) and a mapping from that id to a material block. The block's `type` selects
how the $6\times6$ is built, with the SwiftComp material codes:

% source: docs/input_format.md (the solid SG yaml); src/opensg_solid/io/sg_input.py (_material_from_dict_entry)
| `type` | fields | meaning |
|---|---|---|
| 0 | `E`, `nu` | isotropic; $G = E/2(1+\nu)$ |
| 1 | `engineering` | the nine constants $E_1\,E_2\,E_3\,G_{12}\,G_{13}\,G_{23}\,\nu_{12}\,\nu_{13}\,\nu_{23}$ |
| 2 | `C` | the $6\times6$ stiffness stored directly, in the Voigt order above |

% source: src/opensg_solid/io/sg_input.py; src/opensg_solid/sg_dehom.py (material_frame_fields); docs/input_format.md
| optional key | meaning | default |
|---|---|---|
| `angle` | ply angle in degrees about the thickness axis $y_3$, applied when the stiffness is built from constants | 0.0 |
| `density` | mass density; used by the beam route's $6\times6$ mass matrix | none |
| `aux` | the auxiliary line of a converted `.sc` (temperature and density), carried through; the elastic path does not use it | none |

Type 1 with `angle` is the recommended form, because the ply data stays visible in one place.
The honeycomb sandwich of example 3 writes its two face laminae and its core this way ({numref}`fig-mat-solid-materials`):

% source: examples/OpenSG-solid/3_get_plate_props_from_2D_SG/RHC_SW_2UC_45.yaml
```yaml
materials:
  1:
    type: 1
    engineering: [108000.0, 8000.0, 8000.0, 4000.0, 4000.0, 3000.0, 0.32, 0.32, 0.30]
    angle: 45.0
  2:
    type: 1
    engineering: [108000.0, 8000.0, 8000.0, 4000.0, 4000.0, 3000.0, 0.32, 0.32, 0.30]
    angle: -45.0
  3:
    type: 1
    engineering: [69000.0, 69000.0, 69000.0, 26540.0, 26540.0, 26540.0, 0.30, 0.30, 0.30]
    angle: 0.0
```

The same cell converted from SwiftComp carries pre-rotated type 2 plies and a type 0 core
(material 2, the $-45^\circ$ ply, is left out of the excerpt):

% source: examples/OpenSG-solid/7_get_beam_props_from_SG/RHC_SW_2UC_45.yaml
```yaml
materials:
  1:
    type: 2
    aux: [0.0, 0.0]
    C:
    - [35696.6, 27696.6, 3251.1, 0.0, 0.0, -25368.7]
    - [27696.6, 35696.6, 3251.1, 0.0, 0.0, -25368.7]
    - [3251.1, 3251.1, 8917.8, 0.0, 0.0, -487.1]
    - [0.0, 0.0, 0.0, 3500.0, -500.0, 0.0]
    - [0.0, 0.0, 0.0, -500.0, 3500.0, 0.0]
    - [-25368.7, -25368.7, -487.1, 0.0, 0.0, 27958.5]
  3:
    type: 0
    aux: [0.0, 0.0]
    E: 69000.0
    nu: 0.3
```

```{figure} ../_static/rhc_2dsg_mesh.png
:width: 80%
:name: fig-mat-solid-materials

The two-unit-cell honeycomb sandwich SG, elements coloured by material: the $\pm45^\circ$ face
laminae (materials 1 and 2) and the isotropic core web (material 3).
```

:::{note}
A type 2 `C` is normally an already rotated ply, so give it no `angle`. The sign convention of
`angle: a` matches an Abaqus `*Orientation` rotation `3, a` about the thickness axis.
:::

### Material list with element sets

The mesh spelling written by `opensg msh_to_yaml` (`nodes`, `elements`, `elementOrientations`,
`sets`, `materials`) lists the materials by name, each with `density` and an `elastic` block of
the three triples, and binds them to the `sets: element:` entries:

% source: examples/OpenSG-solid/5_get_solid_props_from_SG/UDcomp_2D.yaml
```yaml
materials:
- name: matrix
  density: 1200.0
  elastic:
    E: [4760.000000, 4760.000000, 4760.000000]
    G: [1737.226277, 1737.226277, 1737.226277]
    nu: [0.370000, 0.370000, 0.370000]
- name: fiber
  density: 1800.0
  elastic:
    E: [276000.000000, 19500.000000, 19500.000000]
    G: [70000.000000, 70000.000000, 5735.000000]
    nu: [0.280000, 0.280000, 0.700000]
sets:
  element:
  - name: matrix
    labels:
    - 1
    - 2
```

The binding rules:

- There must be exactly one set per material.
- Sets are matched to materials by **name** when every set name is a material name, and by
  **position** otherwise.
- `labels: all` assigns the whole SG to one material.
- Every element must belong to a set; an element without a material is an error.
- An optional `angle:` (degrees about $y_3$) is applied first and then composed with the
  element frame of `elementOrientations`.
- A single value in `E`, `G` or `nu` is repeated for all three components.
- The constants may also sit flat on the entry (`name`, `E`, `G`, `nu`, `rho`) instead of under
  `elastic:`; both forms are read as a type 1 block.

:::{tip}
The Python call `plate_homo_2d(path, material_param=..., angles=...)` overrides the file's
material blocks with engineering constants and angles. The drivers of examples 4, 7 and 8
(`plate_dehom_2dsg.py`, `beam_homo_sg.py`, `beam_dehom_sg.py`) set `material_param` and `angles`
in their user-input block and use this route.
:::

## The material library

The packaged library `src/opensg_solid/io/materials.yaml` uses the list spelling above, plus
optional `aliases:` for each card and informational `layups:` entries that no solver route reads.
A ply angle is never stored in a card: it is given per use. The units are per card, as each card
was taken from the example it is named after.

% source: src/opensg_solid/io/materials.yaml
| name | aliases | units | density | `E` | `G` | `nu` |
|---|---|---|---|---|---|---|
| `Al` | `alu`, `aluminum` | Pa, kg/m³ | 2700.0 | `[7.0e+10, 7.0e+10, 7.0e+10]` | `[26923076923.076923, 26923076923.076923, 26923076923.076923]` | `[0.3, 0.3, 0.3]` |
| `Al69` | `hc_core`, `core_al` | MPa, mm | 0.0 | `[69000.0, 69000.0, 69000.0]` | `[26538.461538461539, 26538.461538461539, 26538.461538461539]` | `[0.3, 0.3, 0.3]` |
| `HC_ply` | `lam_HC` | MPa, mm | 0.0 | `[108000.0, 8000.0, 8000.0]` | `[4000.0, 4000.0, 3000.0]` | `[0.32, 0.32, 0.30]` |
| `ge` | | Pa, kg/m³ | 1500.0 | `[1.28e+11, 1.1e+10, 1.1e+10]` | `[4.48e+9, 4.48e+9, 1.53e+9]` | `[0.25, 0.25, 0.25]` |
| `herex` | | Pa, kg/m³ | 130.0 | `[1.0363e+8, 1.0363e+8, 1.0363e+8]` | `[5.0e+7, 5.0e+7, 5.0e+7]` | `[0.32, 0.32, 0.32]` |
| `ud_matrix` | `epoxy` | MPa, mm | 1200.0 | `[4760.0, 4760.0, 4760.0]` | `[1737.2262773722625, 1737.2262773722625, 1737.2262773722625]` | `[0.37, 0.37, 0.37]` |
| `ud_fiber` | | MPa, mm | 1800.0 | `[276000.0, 19500.0, 19500.0]` | `[70000.0, 70000.0, 5735.0]` | `[0.28, 0.28, 0.70]` |
| `yu2003` | `yu` | psi | 1.0 | `[2.5e+7, 1.0e+6, 1.0e+6]` | `[5.0e+5, 5.0e+5, 2.0e+5]` | `[0.25, 0.25, 0.25]` |
| `pagano` | | Pa | 1.0 | `[2.5e+10, 1.0e+9, 1.0e+9]` | `[5.0e+8, 5.0e+8, 2.0e+8]` | `[0.25, 0.25, 0.25]` |
| `as4` | | Pa | 1.0 | `[1.81e+11, 1.03e+10, 1.03e+10]` | `[7.17e+9, 7.17e+9, 3.5e+9]` | `[0.28, 0.28, 0.28]` |
| `garg_face` | | Pa | 1.0 | `[1.31e+11, 1.034e+10, 1.034e+10]` | `[6.205e+9, 6.205e+9, 3.0e+9]` | `[0.22, 0.22, 0.22]` |
| `garg_core` | | Pa | 1.0 | `[5.776e+8, 5.776e+8, 5.776e+8]` | `[1.079e+8, 1.079e+8, 1.079e+8]` | `[0.0025, 0.0025, 0.0025]` |
| `iso_taper` | | Pa, kg/m³ | 1800.0 | `[7.0e+10, 7.0e+10, 7.0e+10]` | `[26923076923.076923, 26923076923.076923, 26923076923.076923]` | `[0.3, 0.3, 0.3]` |
| `ani_taper` | | Pa, kg/m³ | 1800.0 | `[3.7e+10, 9.0e+9, 9.0e+9]` | `[4.0e+9, 4.0e+9, 4.0e+9]` | `[0.3, 0.3, 0.3]` |

### Selecting a card with `--mat<K> NAME:angle`

`opensg msh_to_yaml` takes one flag per gmsh physical tag $K$, written `--mat<K> NAME[:ANGLE]`:

% source: src/opensg/cli.py (msh_to_yaml docstring)
```bash
opensg msh_to_yaml SP_solid.msh --mat1 Al --n_model 2 --refined 1
opensg msh_to_yaml mesh.msh --mat1 Al69 --mat2 HC_ply:-45 --mat3 HC_ply:45 --n_model 2
```

- `NAME` is case-insensitive and may be an alias; an unknown name stops the conversion with the
  list of available materials.
- `:ANGLE` is a trailing number, the ply angle in degrees about $y_3$. Without it the angle is 0.
  A colon followed by anything other than a number stays part of the name.
- A tag with no flag keeps `FILL_IN` placeholders, and `opensg` refuses to run the yaml until
  they are replaced. A mesh without physical tags takes its sets from the elementary tags, and
  `--mat0` alone fills the whole mesh.

The library is found in this order: the `--materials PATH` argument, `materials.yaml` in the
current directory, `materials.yaml` next to the `.msh`, and the packaged file. To use your own
cards, write a file with the same schema in one of those places. An `.xml` library is refused.
The flags themselves are described in {doc}`converters`.

## The through-thickness layup database

For a laminate, the 1-D SG is one element stack through the thickness, and `layup_db.yaml` is the
whole user input of `1d_sg.py` (`examples/OpenSG-solid/1_get_plate_props_from_1DSG`). The script
generates the SG mesh `1dsg.yaml` and its figure `1dsg.png` ({numref}`fig-mat-layup-db`), homogenizes it, and writes
`layup_db_plate_homo.out`.

% source: docs/input_format.md (layup_db.yaml); examples/OpenSG-solid/1_get_plate_props_from_1DSG/1d_sg.py; examples/OpenSG-solid/2_get_plate_stress_from_1DSG/rm_dehom_ff.py
| key | meaning | default |
|---|---|---|
| `model` | `0` prints the classical $6\times6$ ABD (extension, coupling and bending stiffness), `1` the shear-refined $8\times8$ ABDG; the `.out` title follows | required |
| `fraction` | the reference surface as a fraction of the total thickness: `0` = bottom (outer mold line) face, `0.5` = mid-surface, `1` = top | required |
| `mesh.n_per_layer` | SG elements per physical ply | 1 |
| `mesh.elem_order` | element order $p$, $p+1$ nodes per element; 1 to 4, and `4` (the five-noded quartic) is exact for the warping ladder | 4 |
| `materials.<name>` | `E` (3), `G` (3), `nu` (3), `rho`, optional `full_name` | required |
| `layup` | the plies, **bottom ply first**, each `{material, thickness, angle}`; `divisions` is read only by the Abaqus 3-D deck generator | angle 0.0 |
| `V2` | recovery order, read only by `rm_dehom_ff.py`: `1` = second order (Yu Eq. 66), `0` = first order (Eq. 63) | 1 |
| `plate`, `dehom` | the plate geometry and loads, and the recovery input, read only by the recovery scripts; see {doc}`recovery` | none |

The shipped file also carries `name: 1d_sg`, which `1d_sg.py` does not read: the mesh is always
written as `1dsg.yaml` and `1dsg.png` beside the script, and the `.out` beside the database.

The shipped database is the Nayak sandwich, a nine-ply $(0/90/0/90/\text{core})_s$ stack of
graphite/epoxy faces on a HEREX C70.130 PVC foam core:

% source: examples/OpenSG-solid/1_get_plate_props_from_1DSG/layup_db.yaml
```yaml
model: 1
fraction: 0.5
mesh:
  n_per_layer: 1
  elem_order: 4
materials:
  ge:
    full_name: graphite/epoxy face ply
    E:  [128.0e9, 11.0e9, 11.0e9]
    G:  [4.48e9, 4.48e9, 1.53e9]
    nu: [0.25, 0.25, 0.25]
    rho: 1500.0
  herex:
    full_name: HEREX C70.130 PVC foam core
    E:  [103.63e6, 103.63e6, 103.63e6]
    G:  [50.0e6, 50.0e6, 50.0e6]
    nu: [0.32, 0.32, 0.32]
    rho: 130.0
layup:
  - {material: ge,    thickness: 0.0009525, angle:  0.0}
  - {material: ge,    thickness: 0.0009525, angle: 90.0}
  - {material: ge,    thickness: 0.0009525, angle:  0.0}
  - {material: ge,    thickness: 0.0009525, angle: 90.0}
  - {material: herex, thickness: 0.1447800, angle:  0.0, divisions: 8}
  - {material: ge,    thickness: 0.0009525, angle: 90.0}
  - {material: ge,    thickness: 0.0009525, angle:  0.0}
  - {material: ge,    thickness: 0.0009525, angle: 90.0}
  - {material: ge,    thickness: 0.0009525, angle:  0.0}
```

% source: examples/OpenSG-solid/1_get_plate_props_from_1DSG/1d_sg.py (docstring)
```bash
python 1d_sg.py
python 1d_sg.py my_other_layup.yaml
```

The second form reads another database. The `.out` banner echoes the numbers that mattered, which
is the quickest check that the file was read as intended:

% source: examples/OpenSG-solid/1_get_plate_props_from_1DSG/layup_db_plate_homo.out
```text
 OpenSG msg-solid plate model from 1dsg.yaml: 9 plies, h = 0.152400 m, fraction = 0.5, rho*h = 30.251400 kg/m^2, rows [e11 e22 g12 k11 k22 k12 2g13 2g23]
```

The generated `1dsg.yaml` records the same decisions in its header, and its `elements` rows are
the five-node quartics, `- [1, 2, 3, 4, 5]`:

% source: examples/OpenSG-solid/1_get_plate_props_from_1DSG/1dsg.yaml
```yaml
sg: {type: plate_1d, elem_order: 4, n_per_layer: 1, reference_fraction: 0.5, thickness: 0.15239999999999998,
  n_ply: 9}
```

```{figure} ../_static/plate_1dsg.png
:width: 70%
:name: fig-mat-layup-db

The through-thickness 1-D SG generated from `layup_db.yaml`: face plies and foam core, with the
reference surface at mid-thickness.
```

:::{note}
`fraction` is a property of the generated mesh: the node coordinates run from
$-\text{fraction}\cdot h$ to $(1-\text{fraction})\cdot h$, so $x_3 = 0$ is the reference surface.
The coupling block $B$ depends on it, so homogenize, analyze and recover with the same value.
:::

:::{warning}
PyYAML reads a number written like `128.0e9` as a string. `1d_sg.py` converts every material
number with `float()`, so the shipped file is safe; a script of your own that reads the database
must do the same.
:::

## See also

- {doc}`coordinate_systems` for the material frame and the angle sign.
- {doc}`mesh_and_elements` for the mesh blocks that the material blocks attach to.
- {doc}`converters` for `msh_to_yaml`, `inp_to_yaml` and `sc_to_yaml`, which write material blocks.
- {doc}`../examples/solid_plate_1d_layup` for the full `layup_db.yaml` run.
