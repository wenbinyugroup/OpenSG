# The solid segment route

The solid segment route gives the beam properties of a 3-D solid beam segment with aperiodic ends. The segment is any
piece of a beam you mesh in solid elements: prismatic or tapered, from a CAD model or from a generator. OpenSG returns
the Timoshenko $6\times6$, the Euler-Bernoulli $4\times4$ and the $6\times6$ mass matrix of the segment, and the
Timoshenko $6\times6$ of each end section.

## The header

A solid yaml selects the route with three header keys and a 3-D mesh.

% source: src/opensg_solid/sg_homo.py (plate_homo_2d), src/opensg_solid/sg_segment.py
```{code-block} yaml
:caption: Header of a solid beam segment.
:name: lst-ug-segment-header

msg: solid      # the solid engine
n_model: 1      # the beam model
refined: 1      # 1 = Timoshenko 6x6 in <base>.out, 0 = Euler-Bernoulli 4x4
aperiodic: 1    # the ends carry their own warping: the solid segment route
```

The mesh blocks are those of any solid yaml in the mesh dialect: `nodes`, `elements`, `sets` paired with the
`materials` by name, one `orientation` row per element, and a `density` in every material block. The same header on a
2-D beam SG (a cross-section) is refused with a message that names this route. Drop `aperiodic: 1` to run a
cross-section.

## Axes, ends and elements

- **Axes.** Mesh $z$ is the beam axis $x_1$, mesh $x$ is $x_2$ and mesh $y$ is $x_3$. This is the convention of the
  2-D beam SG, whose nodes are $[y_2\ y_3\ 0]$, extended to 3-D.
- **Frames.** Each `orientation` row lists the axes $e_1$, $e_2$, $e_3$ with their mesh components $(x, y, z)$, the
  spelling of the 2-D beam SG. A material block's `angle:` is applied as on the other routes.
- **Ends.** The two end sections are the element faces that lie in the planes $z = z_{min}$ and $z = z_{max}$. A hex8
  face is a quad4, a tet4 face a tri3 and a tet10 face a tri6. Each face keeps the material and the frame of its
  element. A segment with no element face in one of the end planes is refused.
- **Elements.** hex8, tet4, mixed hex8 and tet4, and tet10 for curved geometry. A tet10 mesh takes no linear
  elements. The node order is gmsh's.
- **Origin.** The nodes are used as given. The law is referred to the line $x = y = 0$ of the mesh: the strains of the
  $6\times6$ are the beam strains of that line, and the mass moments are taken about it.

## What the route computes

Each end section is solved as a 2-D beam SG: its Timoshenko $6\times6$ and its warping fields. The warping is pinned
as on the 2-D beam route: zero area mean of the displacement and of the rotation of the section. The 3-D segment is
then solved with the end warping as Dirichlet data on the end-face nodes, and the Timoshenko reduction gives the
segment's $6\times6$. Every energy term is divided by the segment length $L = z_{max} - z_{min}$, so the law is per unit
length of the segment.

:::{note}
How the end warping is pinned does not change an end section's law, but it changes the law of a tapered segment,
because the two ends then carry different rigid parts. The route pins it by the area means, as FEniCS OpenSG v1.2
does. Pinning it to the nodal rigid modes instead moved the test segments by $10^{-4}$ to $2\times10^{-2}$ of the
largest entry.
:::

The mass matrix integrates $\rho$, $\rho x_2$, $\rho x_3$, $\rho x_3^2$, $\rho x_2^2$ and $\rho x_2 x_3$ over the
segment volume and divides them by $L$. It is the mass per unit length averaged over the segment, about the line
$x = y = 0$, in the layout of the 2-D beam route (VABS). A material without `density` counts as density 0. A segment
with no density at all gets the zero matrix, and the log says so.

The route solves with SuperLU (`scipy.sparse.linalg.splu`) whatever `--solver` names. An iterative solver is refused.

## What the route writes

% source: src/opensg_solid/sg_segment.py (segment_homo, write_ends_out)
```{table} Files and log lines of the solid segment route. base is the yaml name without its extension.
:name: tab-ug-segment-out

| output | content |
|---|---|
| `<base>.out` | the law in the layout of the 2-D beam route: the Timoshenko $6\times6$ (`refined: 1`) or the Euler-Bernoulli $4\times4$ (`refined: 0`), its compliance, and the 6X6 mass matrix. The banner reads `msg-solid beam segment model, omega L, aperiodic ends at z ... and z ...` |
| `<base>_ends.out` | each end section's Timoshenko $6\times6$, its compliance and its section mass matrix, one block per end |
| `<base>_mesh.png` | the mesh, elements coloured by material, as on the other solid routes |
| ` route     :` | one log line with the length, both end planes and the faces of each end, for example `solid segment, length 2, ends z = 0 (16 quad4) and z = 2 (16 quad4)` |
| ` reference :` | the line the law is referred to |
| ` mass      :` | the mass per unit length and the mass centre, or the note that no density was given |
```

A dehomogenization (`opensg <segment.yaml> D`) is refused with one line: the route has no recovery yet.

## Python

`plate_homo_2d` runs the route from the header, or from its arguments for a file without them:

```python
from opensg_solid.sg_homo import plate_homo_2d

r = plate_homo_2d("segment.yaml")                       # the header selects the route
r = plate_homo_2d("mesh.yaml", n_model=1, boundary="aperiodic", refined=1)
r["C_eff"], r["C_eff_EB"], r["Mass"]                    # 6x6, 4x4, 6x6 per unit length
[e["C_eff"] for e in r["ends"]]                         # each end section's 6x6
```

## Checks

The route is tested in `tests/unit/test_solid_segment.py` on small generated segments.

- A prismatic hex8, tet4 or mixed segment equals its end section to $10^{-14}$ of the largest entry, and each end
  section equals OpenSG's 2-D beam route on the same face mesh to $2\times10^{-15}$.
- A prismatic tet10 segment differs from its end section by the discretization: $6.3\times10^{-4}$ on the test mesh,
  smaller on finer meshes. The isotropic $EA$, $EI$ and their couplings are exact.
- On two tapered tubes the segment law equals FEniCS OpenSG v1.2 to $6\times10^{-14}$ at the same quadrature, and to
  $0.22\times10^{-6}$ at the higher degree FEniCS picks for the non-affine hexes. The mass matrices agree to
  $2\times10^{-14}$.
- With the nodal pinning of the JAX port the route comes from, the segment and end laws equal the port's to
  $4\times10^{-15}$, prismatic and tapered.
- The mass matrix of a tapered segment equals its closed form, and rotated frame rows equal the block-angle twin.

## Where the algorithm comes from

The route is FEniCS OpenSG v1.2's tapered solid segment (`compute_stiffness` with `Taper=True`, `compute_timo_boun` and
`get_mass_solid` with `Taper=True`), through its JAX port in the OpenSG-TW-claude repository
(`opensg_jax/fe_jax/solid_taper.py`). The mass matrix has $-i_{23}$ in its product-of-inertia entries, as the 2-D beam
route and VABS do. FEniCS v1.2 writes $+i_{23}$ there. The two agree for a section symmetric about either axis.

## See also

- {doc}`analysis_settings` for the header keys and the other routes.
- {doc}`coordinate_systems` for the frame rows and the order of the beam strains.
- {doc}`output_files` for the `.out` layout.
