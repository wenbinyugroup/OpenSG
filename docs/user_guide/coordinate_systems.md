# Coordinate systems and conventions

Three frames meet in a structure gene (SG) yaml. The SG coordinates are the node coordinates written in `nodes`. The material frame of
each element is a triad of unit vectors $e_1, e_2, e_3$, written row by row in `elementOrientations`. The ply frame
follows from the material frame by the ply angle, a rotation about $e_3$ that the layup (shell dialect) or the material
block (solid dialect) carries. On top of these, every macro law has a fixed order of strains and stiffness entries.

All of these conventions are fixed in the code and shared by the two engines. This page states them in the order a
file is written: coordinates, frames, angles, and the order of the results.

## SG coordinates

Node rows carry three components, except the generated 1-D through-thickness plate SG, whose rows carry one. Which
column means what depends on the SG and on the macro model, as listed in {numref}`tab-ug-sg-axes`.

% source: src/opensg_shell/sg_mesh.py (load_ring: ax = 2, cross = [0, 1]); src/opensg_solid/sg_materials.py
% (elem_rotation_from_yaml); docs/input_format.md ("Sign and order conventions")
```{table} Meaning of the node columns.
:name: tab-ug-sg-axes

| SG | node columns | remarks |
|---|---|---|
| 1-D shell SG; 2-D solid SG of a beam or an equivalent solid | $(y_2, y_3, x_1)$: the two cross-section coordinates, then the beam axis | the third column is zero for a plane cross-section, and the beam axis $x_1$ is out of the plane |
| 2-D solid SG of a plate | column 0: the in-plane periodic direction; column 1: the thickness direction | the last SG coordinate is the thickness direction |
| 1-D solid SG of a plate | the single coordinate is the thickness direction | runs from $-h/2$ to $+h/2$ when the reference is the mid-surface |
| 3-D solid SG; 3-D shell SG | $(y_1, y_2, y_3)$ | all three columns |
```

The solver indexes the beam axes as $(1, 2, 3) = (x_1, x_2, x_3)$ with $x_1$ the beam axis, while a cross-section yaml
writes every 3-vector as $(y_2, y_3, x_1)$. The solid engine therefore cycles each vector $[a, b, c] \to [c, a, b]$
before it uses a yaml frame. A frame fed to the solver without that cycle points $e_1$ along $y_3$: the run completes
and looks plausible, but a square tube comes out with $EI_2 \neq EI_3$.

% source: docs/input_format.md ("Sign and order conventions")
For a plate SG the in-plane measure follows from the columns. The two-unit-cell honeycomb `RHC_SW_2UC_45.yaml` reports
$\omega = 35.248$, the span of column 0 (the in-plane periodic direction), not its larger extent in column 1.

:::{note}
The SG coordinates always occupy the leading columns, and the solid engine takes the SG dimension as the highest
column that varies. A 1-D SG varies in column 0 only, a 2-D SG in columns 0 and 1.
:::

## Material frame of an element

Each row of `elementOrientations` is nine direction cosines forming three stacked unit vectors, in the same component
order as the node rows:

```text
[ e1x e1y e1z | e2x e2y e2z | e3x e3y e3z ]
```

On a cross-section SG, shell or solid, the roles are fixed:

- $e_1$ is the beam axis, the prismatic direction. For a plane cross-section meshed in $(y_2, y_3)$ it is out of
  plane: `e1 = [0, 0, 1]`.
- $e_2$ is the in-plane ply direction, the wall tangent along the contour.
- $e_3 = e_1 \times e_2$ is the wall normal, in plane, and the direction the layup stacks along from the reference
  surface.

The first element of the IEA-22 cross-section shows the pattern: $e_1$ out of plane, $e_2$ and $e_3$ in plane.

% source: examples/OpenSG_shell/1_get_beam_props_from_shell_cross_section/iea_s10_shell.yaml
```yaml
elementOrientations:
- [0.0, 0.0, 1.0, -0.9721541292349604, 0.2343423756204075, 0.0, -0.2343423756204075,
  -0.9721541292349604, 0.0]
```

On a 3-D shell SG the triad follows each facet: $e_3$ is the facet normal (right-hand rule on the node order), $e_2$
the first edge and $e_1 = e_2 \times e_3$, which is how the gmsh converter of {doc}`converters` writes it. A solid SG
may omit `elementOrientations`; the gmsh converter writes every element with the constant default frame, $e_1$ out of
plane:

% source: examples/OpenSG-solid/4_get_plate_dehom_from_2DSG/RHC_SW_2UC_45.yaml; src/opensg/cli.py (msh_to_yaml); src/opensg_solid/io/sg_input.py
% (frames_are_noop)
```yaml
elementOrientations:
- [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
```

:::{warning}
This default row is the identity in the solver's order, so it leaves every material unrotated. The row
`[1, 0, 0, 0, 1, 0, 0, 0, 1]` is not: in yaml component order it is a cyclic permutation of the axes.
:::

% source: src/opensg_shell/sg_mesh.py (frame_report)
The triad must be orthonormal and right-handed. `opensg_shell.sg_mesh.frame_report` checks unit norms,
$\max|e_i \cdot e_j| < 10^{-6}$ and $e_1 \times e_2 \cdot e_3 > 0.99$, and returns a pass flag with a one-line summary
ending in `OK` or `CHECK`. Its axial test $e_1 \cdot \hat{x} > 0.999$ and its mean of $e_3 \cdot (-\hat{r})$ are
specific to a cylinder along $x$, and the second is only reported, not part of the pass flag: a frame whose $e_3$ has
flipped outward still returns `OK`, so read that number. For a cross-section yaml the check below states the rule
directly:

% source: docs/theory/rules/orientation_e1_out_of_plane.md
```python
ori = np.array(d["elementOrientations"], float)
e1, e3 = ori[:, 0:3], ori[:, 6:9]
assert np.allclose(np.abs(e1[:, 2]), 1.0)   # e1 out of plane
assert np.allclose(e3[:, 2], 0.0)           # e3 in plane
```

### The orientation figure

Mesh handling can renumber nodes and elements, so every run should come with a figure of the frames. The
cross-section entry points do not draw it; the example driver
`examples/OpenSG_shell/1_get_beam_props_from_shell_cross_section/beam_homo_shell.py` calls `opensg_shell.auto_emit`,
and a driver of your own should do the same. The call draws once per file and process and never raises. The segment
route draws its own frame figures, for example `orient_BAR_URC_numEl_52_segment_12_boundaries_segment.png` and the
two end rings `orient_BAR_URC_numEl_52_segment_12_boundaries_ring_L.png` and `..._ring_R.png` in
`examples/OpenSG_shell/5_taper_segment_aperiodic_boundaries/`.
{numref}`fig-ug-orient` shows the result for the IEA-22 station at $r/R = 0.2$: $e_2$ in blue, $e_3$ in green, drawn
at element centroids; $e_1$ points out of the page and is not drawn.

```{figure} ../_static/shell_xsec_orient.png
:width: 95%
:name: fig-ug-orient

Material frames of the IEA-22 $r/R = 0.2$ shell cross-section: $e_2$ (blue) tangent to every wall, $e_3$ (green)
pointing inward on the skin.
```

Read it as a physical check. On this station the $e_3$ arrows point inward everywhere on the skin, as an outward-first
layup requires, and the $e_2$ arrows run consistently around the closed skin loop and along each of the three shear
webs. An $e_3$ that flips between neighbouring skin elements, or a web whose $e_2$ opposes its neighbours, stacks the
layup in two directions at once: visible in the figure in seconds, invisible in the resulting $6\times6$.

## Ply angle and its sign

The ply angle is not built into $e_1$. It lives in the third entry of each `layup` row, `[material, thickness,
angle_deg]`, in the shell dialect, and in the `angle:` entry of a material block, in degrees, in the solid dialect.
It rotates the material about $e_3$ (the thickness axis $y_3$ of a solid plate SG), away from $e_1$, so `angle: 0.0`
places the fibre along $e_1$. On a cross-section that is the beam axis: a unidirectional spar cap carries `angle: 0.0`,
and a $\pm45$ ply carries `45.0` and `-45.0`.

The rotation is right-handed about $e_3$: the fibre of a ply at angle $\theta$ lies along
$\cos\theta\,e_1 + \sin\theta\,e_2$, the VABS ply-angle convention. Both engines build the rotated stiffness with the
same $6\times6$ rotation, which leaves the Voigt-33 row untouched and mixes the 11, 22 and 12 rows. The reduced ply
stiffness of the transverse-shear construction (`_ply_Q_and_G`) follows the same rule; its first entry is

% source: docs/input_format.md ("Material orientation"); src/opensg_shell/kernels/msg_materials.py (rotation_6x6);
% src/opensg_solid/sg_materials.py (rotate_C_matrix); src/opensg_solid/io/sg_input.py (vabs_angles_to_frame);
% src/opensg_shell/kernels/msg_transverse_shear.py (_ply_Q_and_G)
$$\bar{Q}_{11} = Q_{11}\cos^4\theta + 2(Q_{12} + 2Q_{66})\sin^2\theta\cos^2\theta + Q_{22}\sin^4\theta .$$

A per-element frame of a solid SG can itself carry a rotation about the layer normal. That rotation and the material
`angle:` are both rotations about $y_3$, so they add.

% source: docs/theory/rules/orientation_e1_out_of_plane.md
:::{warning}
A yaml whose $e_1$ already contains the fibre rotation ($|e_{1z}| < 1$, for example $e_{1z} = 0.7071$ for a $\pm45$ ply)
carries a fibre frame, not the section frame. Combined with a non-zero ply angle it applies the rotation twice. The
PreVABS converter of OpenSG_io writes two files from one `.sg`: `<name>.yaml` with the fibre frame, and
`<name>_t1only.yaml` with $e_1$ on the beam axis. Use the `_t1only` file and give the ply angle in the material.
:::

## Order of strains and stiffness entries

{numref}`tab-ug-orders` lists the order of the strain vector, and so of the rows and columns, of every law OpenSG
writes. $\mathbf{A}$, $\mathbf{B}$, $\mathbf{D}$ are the membrane, coupling and bending blocks of a plate (ABD) and
$\mathbf{G}$ is its transverse-shear block.

% source: docs/input_format.md ("Sign and order conventions"); src/opensg_solid/sg_homo.py (law titles);
% examples/OpenSG_shell/4_get_solid_props_from_shell_3D_SG/schwarz_p_3Dshell_ABDG.out
```{table} Row order of the macro laws.
:name: tab-ug-orders

| law | rows | remarks |
|---|---|---|
| Cauchy continuum $6\times6$ (3-D solid) | `[11 22 33 23 13 12]` | the SwiftComp order with engineering shears: the compliance has $S_{44} = 1/G_{23}$, $S_{55} = 1/G_{13}$, $S_{66} = 1/G_{12}$; also the order of a `type: 2` material `C` |
| Timoshenko beam $6\times6$ | $[\varepsilon_{11}\;\gamma_{12}\;\gamma_{13}\;\kappa_1\;\kappa_2\;\kappa_3]$ | diagonal $[EA\;GA_2\;GA_3\;GJ\;EI_2\;EI_3]$, the VABS order; axis 1 is the beam axis |
| Euler-Bernoulli beam $4\times4$ | `[eps11 kappa1 kappa2 kappa3]` | the classical beam |
| classical plate $6\times6$ (ABD) | $[\varepsilon_{11}\;\varepsilon_{22}\;2\varepsilon_{12}\;\kappa_{11}\;\kappa_{22}\;\kappa_{12}]$ | resultants `[N11 N22 N12 M11 M22 M12]` |
| Reissner-Mindlin plate $8\times8$ (ABDG) | `[e11 e22 g12 k11 k22 k12 2g13 2g23]` | resultants `[N11 N22 N12 M11 M22 M12 Q1 Q2]` |
| shell wall law $8\times8$ (`<base>_ABDG.out`) | `[eps11 eps22 2eps12 \| K11 K22 K12+K21 \| 2g13 2g23]` | one block per section |
```

The macro state `epsilon_bar` of a dehomogenization follows the same orders: `[e11 e22 2e12 k11 k22 2k12]` for a plate
or a solid, `[ext sh2 sh3 twist bend2 bend3]` for a beam. The Reissner-Mindlin law has the block structure

$$\mathbf{ABDG} = \begin{bmatrix} \mathbf{A} & \mathbf{B} & 0 \\ \mathbf{B} & \mathbf{D} & 0 \\ 0 & 0 & \mathbf{G} \end{bmatrix}.$$

Moving the reference surface by $z_0$ moves the membrane strain to $m + z_0 \kappa$, so the ABD transforms as
$\mathbf{ABD}_{\text{new}} = T^{-\mathsf{T}}\,\mathbf{ABD}\,T^{-1}$ with $T = [[I, z_0 I], [0, I]]$
(`shift_abd_reference`). Use it rather than reversing the layup: reversal flips $e_3$ and breaks agreement with the
material frame. The transverse-shear block $\mathbf{G}$ does not depend on the reference.

## See also

- {doc}`analysis_settings` for the laminate reference of a shell SG (`--center`).
- {doc}`materials_and_layups` for the layup rows and the material blocks that carry the ply angle.
- {doc}`../theory/rules/orientation_e1_out_of_plane` for the frame rule and its failure modes.
- {doc}`../theory/constitutive` for the constitutive relations behind each law.
