# Geometry and meshes

An SG yaml describes the structure gene (SG) by its finite-element mesh: the node coordinates, the element
connectivity, and named groups of elements that carry a laminate or a material. OpenSG reads the mesh as it is
written. It never moves a node and has no geometry key of its own: the SG dimension, the beam axis of a segment and the
SG measure are all read from the mesh.

The mesh blocks come from a generator script (for example `make_schwarz_yaml.py` in
`examples/OpenSG_shell/4_get_solid_props_from_shell_3D_SG/`), from a converter that reads a gmsh, Abaqus or SwiftComp
mesh ({doc}`converters`), or from your own script. This page describes the blocks that carry geometry and topology
(`nodes`, `elements`, `cells`, `sets`), the kinds of SG each engine accepts, and the finite element each route builds.
The material side of the file (`materials`, `sections`, `mat_id`) is described in {doc}`materials_and_layups`, and the
material frame of each element in {doc}`coordinate_systems`.

## The SG dimension

There is no `dim:` key. The dimension is read from the mesh, differently in the two dialects:

- **Shell dialect**: the node count of the elements decides. 2-node line elements form a 1-D shell SG, the contour of
  a cross-section. 3- and 4-node elements form a surface in space; its nodes carry three coordinates, but the SG itself
  is two-dimensional.
- **Solid dialect**: the node columns decide. The SG coordinates always occupy the leading columns of `nodes`, and the
  dimension is the highest column that varies: a 1-D SG varies in column 0 only, a 2-D SG in columns 0 and 1. A column
  that never varies is padding.

{numref}`tab-mesh-kinds` lists the kinds of SG and the macro models each reaches through the `n_model` header key
({doc}`analysis_settings`).

% source: src/opensg_solid/sg_mesh.py (sg_dim, load_sg_input, _cell_basis); src/opensg_shell/cli.py (mesh_kind, main);
% src/opensg_shell/sg_homo.py (shell_sg3d, segment_timo_from_3dyaml); src/opensg_solid/io/sg_input.py (sg_omega)
```{table} Kinds of SG, their elements and the macro models they reach.
:name: tab-mesh-kinds

| SG | engine | elements | `n_model` | example input |
|---|---|---|---|---|
| 1-D shell SG: cross-section contour | `opensg_shell` | 2-node lines | 1, 3 | `examples/OpenSG_shell/1_get_beam_props_from_shell_cross_section/iea_s10_shell.yaml` |
| 3-D shell SG: surface unit cell | `opensg_shell` | 3-node triangles and 4-node quads, mixed or not | 3 | `examples/OpenSG_shell/4_get_solid_props_from_shell_3D_SG/schwarz_p_3Dshell.yaml` |
| 3-D shell segment: tapered or aperiodic piece of a beam | `opensg_shell` | 4-node quads | 1, with `refined: 1` | `examples/OpenSG_shell/5_taper_segment_aperiodic_boundaries/BAR_URC_numEl_52_segment_12.yaml` |
| 1-D solid SG: laminate through the thickness | `opensg_solid` | lines of 2 to 5 nodes | 2, 3 | `examples/OpenSG-solid/1_get_plate_props_from_1DSG/1dsg.yaml`, generated from `layup_db.yaml` |
| 2-D solid SG: cross-section or unit cell | `opensg_solid` | 3- and 6-node triangles, 4- and 9-node quads | 1, 2, 3 | `examples/OpenSG-solid/3_get_plate_props_from_2D_SG/RHC_SW_2UC_45.yaml` |
| 3-D solid SG: unit cell | `opensg_solid` | 4- and 10-node tetrahedra, 8-node hexahedra | 1, 2, 3 | `examples/OpenSG-solid/6_get_solid_props_from_3D_SG/sample_1/Sample_1.sc` |
```

## Mesh blocks

{numref}`tab-mesh-keys` lists the blocks that carry the geometry and topology. Row $k$ of `elements` (or `cells`) is
element $k$ everywhere else in the file: label $k$ in `sets`, entry $k$ of `mat_id` and row $k$ of
`elementOrientations`.

% source: src/opensg_shell/sg_mesh.py (_row, load_ring_ref, subdomain_ids, extract); src/opensg_shell/sg_homo.py
% (shell_sg3d); src/opensg_solid/io/sg_input.py (_tokens, read_opensg_yaml, _mat_id_from_sets);
% src/opensg_solid/sg_mesh.py (load_sg_input); docs/input_format.md
```{table} Mesh blocks of an SG yaml.
:name: tab-mesh-keys

| key | dialect | one row | numbering | default |
|---|---|---|---|---|
| `nodes` | both | the coordinates of one node: three numbers (one for the generated 1-D plate SG) | row $k$ is node $k$ | required |
| `elements` | shell | the node ids of one element: 2 (contour), 3 or 4 (surface) | 1-based node ids | required |
| `elements` | solid, mesh spelling | the node ids of one element | 1-based node ids | required, or `cells` |
| `cells` | solid, canonical spelling | the node ids of one element | 0-based node ids | required, or `elements` |
| `mat_id` | solid, canonical spelling | one material id per cell | 1-based material ids | required with `cells` |
| `sets: element:` | both | `name` and `labels` of one element group | 1-based element labels; `labels: all` (solid) is the whole SG | shell: required; solid: optional for one material |
| `elementOrientations` | both | nine direction cosines, `[e1 e2 e3]` | row $k$ is element $k$ | shell: required; solid: no frame, material `angle:` only |
```

A row is written in one of two flow styles: the bracketed, space-separated row `- [4.70852238 0.09722431 0.00000000]`
(one yaml scalar) or the comma list `- [17.6240005, 23.3500004, 0.0]`. The cross-section loader, the 3-D shell SG
loader and the solid mesh spelling read both. The segment route and the canonical solid spelling (`cells`, `mat_id`)
read the rows as plain arrays and need the comma list. The excerpts below show the connectivity of one file of each
kind.

% source: examples/OpenSG_shell/1_get_beam_props_from_shell_cross_section/iea_s10_shell.yaml;
% examples/OpenSG_shell/4_get_solid_props_from_shell_3D_SG/schwarz_p_3Dshell.yaml;
% examples/OpenSG-solid/3_get_plate_props_from_2D_SG/RHC_SW_2UC_45.yaml (rows abridged)
```yaml
# 1-D shell SG (iea_s10_shell.yaml): 2-node lines, 1-based
elements:
- [1 2]
- [2 3]
# 3-D shell SG (schwarz_p_3Dshell.yaml): 3-node triangles, 1-based
elements:
- [1 121 127]
- [7 1 127]
# 2-D solid SG (RHC_SW_2UC_45.yaml): 3-node triangles, 0-based, one material id per cell
cells:
- [0, 66, 345]
- [345, 3, 0]
mat_id: [1, 1, 1, 1, 1]
```

:::{tip}
Write the `nodes` and `elements` rows of a cross-section (1-D shell SG) yaml space-separated. The loader takes either
style, but the orientation plotter `opensg_shell.auto_emit` (`src/opensg_shell/kernels/orient_plot.py`) reads each
row as one string and does not draw a comma list.
:::

% source: src/opensg_shell/sg_mesh.py (load_ring_ref, extract); src/opensg_shell/sg_homo.py (shell_sg3d); examples/OpenSG_shell/5_taper_segment_aperiodic_boundaries/BAR_URC_numEl_52_segment_12.yaml
:::{note}
The shell cross-section and segment loaders also accept 0-based element ids: they shift the ids only when the
smallest one is 1. The segment of {numref}`fig-mesh-segment` is written 0-based (`- [0, 1, 53, 52]`). The 3-D shell
SG route reads its ids as 1-based.
:::

### Element sets

`sets: element:` groups element labels under a name. In the shell dialect each set is the element group of one
`sections` entry; in the solid mesh spelling each set is the element group of one material. The pairing rules are
described with the blocks they pair to, in {doc}`materials_and_layups`. Two rules depend on the route:

- The cross-section ring pairs a set with the section whose `elementSet` equals its `name`. The segment route pairs
  them by position: the $k$-th set takes the $k$-th section, so keep the two lists in the same order.
- The 3-D shell SG homogenizes with one wall law. A file with more than one entry in `sections` is refused.

## Node order in an element

Connectivity rows follow the gmsh node order: the corners first, cyclic around a quad, then the mid-side nodes. The
solid engine permutes each row into its own (basix) order before the assembly. {numref}`tab-mesh-node-order` lists
the order of the higher-order elements.

% source: src/opensg_solid/sg_homo.py (_to_basix_order); src/opensg_solid/sg_mesh.py (laminate_to_sg)
```{table} Node order of the higher-order solid elements in a connectivity row.
:name: tab-mesh-node-order

| element | nodes in the row |
|---|---|
| 6-node triangle | 3 corners, then the mid-sides of edges 12, 23, 13 |
| 9-node quad | 4 corners in cyclic order, then the mid-sides of edges 12, 23, 34, 41, then the centre |
| 10-node tetrahedron | 4 corners, then the mid-sides of edges 12, 23, 13, 14, 34, 24 |
| line of 3 to 5 nodes (`cells`) | the two end nodes, then the interior nodes from left to right |
```

:::{warning}
A wrong node order is not reported. A 10-node tetrahedron with its mid-sides in another order changes the sign of
the Jacobian determinant inside every element, and the run still completes.
:::

## Finite elements of the shell routes

The cross-section SG never assembles a line element directly. The contour is extruded one quad deep along the beam
axis, with the mean contour element length as the extrusion length, giving quads `[a, b, m+b, m+a]`; the top row of
nodes is mapped onto the bottom row, so the field is exactly span-invariant. The element is a 4-node bilinear quad
integrated at $2\times2$ Gauss points, and the 1-D cross-section problem is its span-invariant limit. The same strip
serves the beam (`n_model: 1`) and the equivalent solid (`n_model: 3`) of a cross-section.

% source: docs/input_format.md ("Finite elements used by each route"); src/opensg_shell/sg_homo.py (ring_indep,
% ring_solid, shell_sg3d, segment_timo_from_3dyaml); src/opensg_shell/kernels/msg_hermite.py;
% src/opensg_shell/sg_assembly.py (TRI_NDOF); src/opensg_shell/cli.py (main)
```{table} Elements of the shell routes.
:name: tab-mesh-shell-elements

| route | header and mesh | element | degrees of freedom (DOF) | transverse shear | drilling $\omega_3$ |
|---|---|---|---|---|---|
| cross-section ring | `n_model: 1` with `refined: 1`, or `n_model: 3`; 2-node lines | 4-node quad of the one-element strip | 6 per node `[w1, w2, w3, om1, om2, om3]`, 24 per element | MITC (mixed interpolation of tensorial components) tying of $\gamma_{23}$ only (`mitc4_g23`) | one Lagrange multiplier per element |
| classical cross-section | `n_model: 1`, `refined: 0`; 2-node lines | Hermite C1 cubic along the contour arc | value and arc slope of $w_1, w_2, w_3$: 6 per node, 12 per element | none (Kirchhoff-Love wall) | none |
| 3-D shell SG | `n_model: 3`; 3- and 4-node elements | native 3-node triangle and 4-node quad, assembled as two batches | 6 per node: 18 per triangle, 24 per quad | MITC3 on a triangle, MITC4 on a quad | one Lagrange multiplier per element |
| segment | `n_model: 1`, `refined: 1`; 4-node elements | 4-node quad | 6 per node, 24 per element | displacement-based (`full`) | one Lagrange multiplier per element |
```

Under span invariance $\gamma_{13}$ is algebraic in the directors and carries no fluctuation gradient, so the ring ties
only $\gamma_{23}$. The independent $\omega_3$ is tied back by the drilling residual
$DR = C_{33}\omega_3 + C_{3b}\omega_b - S/2$, which stays finite on a flat wall where $C_{33} = n \cdot b_3$ vanishes.
A second ring element, `sg_assembly.ring_general` (5 DOF per node `[w1, w2, w3, om1, om2]`, 20 per element),
eliminates $\omega_3 = S/(2C_{33}) - (C_{3b}/C_{33})\,\omega_b$ instead; it divides by $C_{33}$ and is available
through the Python API for comparison.

% source: docs/input_format.md ("The 1-D shell SG YAML"); examples/OpenSG_shell/1_get_beam_props_from_shell_cross_section/iea_s10_shell.yaml
```{figure} ../_static/shell_xsec_mesh.png
:width: 90%
:name: fig-mesh-ring

1-D shell SG of the IEA-22 blade at $r/R = 0.2$: 307 nodes and 310 line elements in six element sets, `layup_0` to
`layup_5`.
```

The 3-D shell SG is a surface cell whose facets carry a wall law, as in {numref}`fig-mesh-schwarz`. A triangle is a
native 3-node element: the collapsed quad `[n1 n2 n3 n3]` is refused, and a mesh may mix triangles and quads. The
banner of the `.out` names the elements that were assembled:

% source: examples/OpenSG_shell/4_get_solid_props_from_shell_3D_SG/schwarz_p_3Dshell_C3D.out
```text
 OpenSG msg-shell equivalent 3D solid (3-D shell SG, 13536 nodes, 26360 elems [26360 tri3/MITC3], 0 junction edges, periodic in 3 dirs, 26360 element drilling multipliers, per unit cell 1)
```

% source: docs/input_format.md ("The 3-D shell SG YAML")
```{figure} ../_static/schwarz_p_mesh.png
:width: 60%
:name: fig-mesh-schwarz

3-D shell SG of a Schwarz-P cell: 13536 nodes and 26360 triangles, one aluminium layup.
```

A segment is a 4-node quad surface between two cross-sections, as in {numref}`fig-mesh-segment`. The route finds the
two end cross-sections from the topology: a mesh edge used by exactly one quad is a free edge, and the connected
components of the free-edge graph are the ends. The beam axis is the coordinate direction between the two end
centroids. Each end is solved as a ring, and its warping is imposed on the segment as Dirichlet data. The route also
writes each end as a runnable 1-D shell SG, for example `boundary_BAR_URC_numEl_52_segment_12_boundaries_L.yaml` and
`..._R.yaml` next to the segment yaml. A mesh with fewer than two free-edge loops, or with a triangle, is refused.

```{figure} ../_static/aperiodic_segment_bar_urc.png
:width: 70%
:name: fig-mesh-segment

Tapered shell segment of the BAR-URC blade with the frame of every element: $e_1$ (red) along the beam axis, $e_2$
(blue) along the wall, $e_3$ (black) normal to it.
```

## Finite elements of the solid routes

The solid engine maps the SG dimension and the node count of an element onto a Lagrange element
({numref}`tab-mesh-solid-elements`); this table is the list of supported elements, and any other combination is
refused with `unsupported <dim>D element with <n> nodes`. Every node carries the three translations. The basis
functions are equispaced Lagrange polynomials, integrated with a quadrature of degree 6 for degree $> 1$ and degree 2
otherwise.

% source: src/opensg_solid/sg_mesh.py (_cell_basis); src/opensg_solid/sg_mixed.py (fe_tables);
% src/opensg_solid/io/msh_to_yaml.py (_ORDER_OF)
```{table} Elements of the solid engine.
:name: tab-mesh-solid-elements

| SG dimension | nodes per element | cell | Lagrange degree | gmsh name |
|---|---|---|---|---|
| 1 | 2 | `interval` | 1 | |
| 1 | 3 | `interval` | 2 | |
| 1 | 4 | `interval` | 3 | |
| 1 | 5 | `interval` | 4 | |
| 2 | 3 | `triangle` | 1 | tri3 |
| 2 | 4 | `quadrilateral` | 1 | quad4 |
| 2 | 6 | `triangle` | 2 | tri6 |
| 2 | 9 | `quadrilateral` | 2 | quad9 |
| 3 | 4 | `tetrahedron` | 1 | tet4 |
| 3 | 8 | `hexahedron` | 1 | hex8 |
| 3 | 10 | `tetrahedron` | 2 | tet10 |
```

An SG may mix element types, for example triangles and quads. Each type is assembled as its own batch over the
shared nodes. A mixed SG runs the plate and 3-D solid macro models with the direct solver; the beam route
(`n_model: 1`) and the iterative solvers take one element type and refuse a mixed mesh.

{numref}`fig-mesh-rhc` shows a 2-D solid SG of a plate. Column 0 of `nodes` (horizontal in the figure) is the in-plane
periodic direction, and column 1 (vertical), the last SG coordinate, is the thickness direction.

% source: docs/input_format.md ("The solid SG YAML")
```{figure} ../_static/rhc_2dsg_mesh.png
:width: 70%
:name: fig-mesh-rhc

2-D solid SG of a two-unit-cell honeycomb sandwich plate: 4251 nodes and 6640 triangles, coloured by material.
```

% source: docs/input_format.md ("Solid route"); examples/OpenSG-solid/1_get_plate_props_from_1DSG/1d_sg.py; src/opensg_solid/sg_mesh.py (load_sg_input)
The 1-D solid SG of a laminate is generated, not drawn. `1d_sg.py` reads the stacking sequence of `layup_db.yaml` and
writes `1dsg.yaml`, one element stack through the thickness ({numref}`fig-mesh-1dsg`). That file is the layup
spelling of the solid dialect: one coordinate per node, `elements` listed in sequence (`- [1, 2, 3, 4, 5]` for the
five-node quartic), and one set and one `sections` entry per ply. Its reader checks the ply thicknesses against the
element spans and rebuilds the mesh from the plies. The example driver `1d_sg.py` homogenizes the generated file with
`rm_plate_1D.msg_rm_plate.rm_plate_msg`, which carries its own Lagrange basis; `opensg 1dsg.yaml` instead rebuilds the
mesh from the plies with `opensg_solid.sg_mesh.laminate_to_sg`, whose elements follow {numref}`tab-mesh-solid-elements`
and the ends-first line order of {numref}`tab-mesh-node-order`. The header records the generation:

% source: examples/OpenSG-solid/1_get_plate_props_from_1DSG/1dsg.yaml
```yaml
msg: solid
sg: {type: plate_1d, elem_order: 4, n_per_layer: 1, reference_fraction: 0.5, thickness: 0.15239999999999998,
  n_ply: 9}
```

```{figure} ../_static/plate_1dsg.png
:width: 45%
:name: fig-mesh-1dsg

1-D solid SG of the nine-ply $(0/90/0/90/\text{core})_s$ sandwich, one quartic element per ply; the dotted line is
the mid-surface reference.
```

:::{tip}
Look at the mesh before reading the result. `opensg <sg.yaml> --mesh` runs the analysis and also writes
`<base>_mesh.png` (elements coloured by section or material) and the gmsh 2.2 twin `<base>.msh`; see
{doc}`output_files`.
:::

## See also

- {doc}`coordinate_systems` for the meaning of the node columns and the element frames.
- {doc}`materials_and_layups` for `sections`, `materials`, `mat_id` and the pairing of sets.
- {doc}`converters` for building the mesh blocks from gmsh, Abaqus or SwiftComp files.
- {doc}`../theory/architecture` for the pipeline each route runs on these elements.
- {doc}`../theory/rules/tri3_in_the_segment_route` for why the segment route takes quads only.
