# Equivalent solid properties of a unidirectional composite

## Problem description

This example computes the equivalent 3-D solid stiffness of a unidirectional fibre/matrix composite
from a square unit cell. The structure gene (SG) is two-dimensional: a mesh of the cell cross-section,
periodic in its two in-plane directions and uniform along the fibres. The engine is `opensg_solid`
and the macro model is the 3-D continuum (`n_model = 3`), so the cell reduces to the $6\times6$
effective stiffness $\mathbf{D}_{\rm eff}$ in Voigt order $[11, 22, 33, 23, 13, 12]$ with engineering
shears, and to the nine engineering constants.

On a 2-D SG this is the classical micromechanics unit-cell problem (generalized plane strain). Six
unit macro strains are applied, the periodic fluctuation $\mathbf{V}_0$ is solved for each, and the
effective law is

$$\mathbf{D}_{\rm eff} = \frac{\mathbf{D}_{ee} + \mathbf{V}_0^{\mathsf T}\mathbf{D}_{he}}{\omega},$$

with $\omega$ the cell area. The nine constants are compared with Table II Model-3 of the ASC-23
OpenSG-Solid paper, the same case as the FEniCS `3DModel.ipynb` in `wenbinyugroup/OpenSG`.

% source: docs/tutorials/msg_solid_beam_and_solid.md ("The example SG"); examples/OpenSG-solid/5_get_solid_props_from_SG/UDcomp_2D.yaml
The mesh of {numref}`fig-udcomp-mesh` has 801 nodes and 1536 three-node triangles, 512 matrix
elements and 1024 fibre elements, and a cell area $\omega = 1$. Every element carries the same
frame: $e_1$ along the fibre axis $z$, $e_2 = +x$, $e_3 = +y$. The input files are in
`examples/OpenSG-solid/5_get_solid_props_from_SG/`: the SG `UDcomp_2D.yaml`, the gmsh mesh
`UDcomp_2D.msh` it is converted from, and the two scripts.

```{figure} ../_static/udcomp_2dsg_mesh.png
:width: 55%
:name: fig-udcomp-mesh

The unidirectional composite unit cell in the $y_2$-$y_3$ plane, fibre elements in red and matrix elements in blue.
```

% source: examples/OpenSG-solid/5_get_solid_props_from_SG/UDcomp_2D.yaml (materials block)
```{list-table} Material properties
:header-rows: 2
:name: tab-udcomp-materials

* - Name
  - Density
  - $E_1$
  - $E_2$
  - $E_3$
  - $G_{12}$
  - $G_{13}$
  - $G_{23}$
  - $\nu_{12}$
  - $\nu_{13}$
  - $\nu_{23}$
* -
  -
  - MPa
  - MPa
  - MPa
  - MPa
  - MPa
  - MPa
  -
  -
  -
* - matrix
  - 1200.0
  - 4760
  - 4760
  - 4760
  - 1737.226277
  - 1737.226277
  - 1737.226277
  - 0.37
  - 0.37
  - 0.37
* - fiber
  - 1800.0
  - 276000
  - 19500
  - 19500
  - 70000
  - 70000
  - 5735
  - 0.28
  - 0.28
  - 0.70
```

% source: examples/OpenSG-solid/5_get_solid_props_from_SG/UDcomp_2D.yaml (sets.element); make_udcomp_yaml.py (PHASES)
| Element set | Material | Elements | gmsh physical tag |
|---|---|---|---|
| matrix | matrix | 512 | 0 |
| fiber | fiber | 1024 | 1 |

The density is carried in the yaml but does not enter the stiffness.

## Input

`UDcomp_2D.yaml` opens with the engine line and then holds the mesh, the frames, the materials and
the element sets (the long blocks are cut here):

% source: examples/OpenSG-solid/5_get_solid_props_from_SG/UDcomp_2D.yaml
```yaml
msg: solid      # the ENGINE this SG belongs to (opensg_solid); `opensg <yaml>` dispatches on it
nodes:
- [-0.500000000000 -0.500000000000 0.000000000000]
# ... 801 node rows
elements:
- [5 161 65]
# ... 1536 triangles, 1-based node numbers
elementOrientations:
- [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
# ... one row per element
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
    # ... 512 labels, then the fiber set with 1024 labels
```

The yaml is written once from the gmsh mesh by `make_udcomp_yaml.py` through
`opensg_solid.io.msh_to_yaml.convert`. The script owns the modelling choices the mesh cannot supply,
the materials (`G` is $E/(2\cdot1.37)$ for the matrix) and the mapping of gmsh physical tags to
phases:

% source: examples/OpenSG-solid/5_get_solid_props_from_SG/make_udcomp_yaml.py
```python
MATERIALS = [
    {"name": "matrix", "density": 1200.0,
     "E": [4760.0, 4760.0, 4760.0],
     "G": [4760.0/2/1.37]*3,
     "nu": [0.37, 0.37, 0.37]},
    {"name": "fiber", "density": 1800.0,
     "E": [276000.0, 19500.0, 19500.0],
     "G": [70000.0, 70000.0, 5735.0],
     "nu": [0.28, 0.28, 0.70]},
]
PHASES = [("matrix", 0), ("fiber", 1)]      # (set name, gmsh physical tag)
```

The driver `solid_homo_2dsg.py` has a three-line User Input block: the macro model, the SG name and
the reference constants to compare with.

% source: examples/OpenSG-solid/5_get_solid_props_from_SG/solid_homo_2dsg.py
```python
n_model = 3                    # 1: Beam; 2: Plate; 3: 3D elastic
name = "UDcomp_2D"             # reads <name>.yaml
paper = {"E1": 167.2553, "E2": 11.4136, "G12": 6.8017, "G23": 3.0955,
         "nu12": 0.31178, "nu23": 0.57574}      # ASC-23 Table II Model-3
```

% source: docs/tutorials/msg_solid_beam_and_solid.md (note in section 3)
:::{note}
The driver is self-contained: it assembles constant-strain triangles, the periodic master-slave
node map (right to left matching $y$, top to bottom matching $x$, corner chains resolved) and the
three pinned rigid translations in plain NumPy, so the whole `n_model = 3` algebra reads in one
file. It writes its own `_Deff.out` through `np.savetxt` instead of the SwiftComp-layout report.
The engine call `plate_homo_2d(..., n_model=3)` runs the same model on any solid SG; on the
honeycomb SG of {doc}`solid_beam_2d` it takes 6640 elements, 12 273 degrees of freedom, 11.4 s cold and 8.1 s
warm, and writes the full SwiftComp-layout `.out` with its engineering-constants block.
:::

## Run

```bash
cd examples/OpenSG-solid/5_get_solid_props_from_SG
python make_udcomp_yaml.py
python solid_homo_2dsg.py
```

The yaml is checked in, so the first command is needed only after remeshing. The second prints the
mesh census, the cell area, $\mathbf{D}_{\rm eff}$ and the nine constants next to the paper values,
and closes with:

% source: examples/OpenSG-solid/5_get_solid_props_from_SG/solid_homo_2dsg.py (closing print statements)
```text
Homogenization stored in UDcomp_2D_Deff.out (+ UDcomp_2D_constants_vs_paper.dat)
Time taken: <seconds> sec
```

## Result

{numref}`lst-udcomp-deff` is the effective stiffness written by the driver.

% source: examples/OpenSG-solid/5_get_solid_props_from_SG/UDcomp_2D_Deff.out
```{code-block} text
:caption: Effective solid stiffness from UDcomp_2D_Deff.out, MPa, order [e11 e22 e33 2e23 2e13 2e12]
:name: lst-udcomp-deff

# D_eff (MPa), order [e11 e22 e33 2e23 2e13 2e12]; omega=1.00000000
  1.72654423e+05   8.65848751e+03   8.65848751e+03  -5.55862753e-06   0.00000000e+00   0.00000000e+00
  8.65848751e+03   1.75072478e+04   1.02638997e+04  -3.30098243e-02   0.00000000e+00   0.00000000e+00
  8.65848751e+03   1.02638997e+04   1.75072478e+04   3.29882610e-02   0.00000000e+00   0.00000000e+00
 -5.55862753e-06  -3.30098243e-02   3.29882610e-02   3.09555490e+03   0.00000000e+00   0.00000000e+00
  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00   6.80176267e+03  -1.95914939e-08
  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00  -1.95914939e-08   6.80176267e+03
```

The constants are read off the compliance $\mathbf{S} = \mathbf{D}_{\rm eff}^{-1}$ as
$E_i = 1/S_{ii}$, $G_{ij} = 1/S_{kk}$ and $\nu_{ij} = -S_{ij}/S_{ii}$ ({numref}`tab-udcomp-paper`).

% source: examples/OpenSG-solid/5_get_solid_props_from_SG/UDcomp_2D_constants_vs_paper.dat
```{list-table} Effective engineering constants against ASC-23 Table II Model-3
:header-rows: 1
:name: tab-udcomp-paper

* - Constant
  - OpenSG
  - Paper
  - Difference (%)
* - $E_1$ (GPa)
  - 167.255337
  - 167.255300
  - +0.0000
* - $E_2$ (GPa)
  - 11.413657
  - 11.413600
  - +0.0005
* - $E_3$ (GPa)
  - 11.413657
  - 11.413600
  - +0.0005
* - $G_{23}$ (GPa)
  - 3.095555
  - 3.095500
  - +0.0018
* - $G_{13}$ (GPa)
  - 6.801763
  - 6.801700
  - +0.0009
* - $G_{12}$ (GPa)
  - 6.801763
  - 6.801700
  - +0.0009
* - $\nu_{12}$
  - 0.311780
  - 0.311780
  - -0.0000
* - $\nu_{13}$
  - 0.311780
  - 0.311780
  - -0.0000
* - $\nu_{23}$
  - 0.575743
  - 0.575740
  - +0.0006
```

Every constant agrees with the published value to within $0.002\,\%$, the round-off of the paper's
printed precision.

% source: docs/tutorials/msg_solid_beam_and_solid.md ("What the numbers mean, and what they were validated against")
Two checks are worth making a habit of. First, the symmetry the geometry demands is recovered:
$D_{22} = D_{33}$ and $D_{55} = D_{66}$ to all printed digits, and every normal-shear coupling term
is either exactly zero or below $3.4\times10^{-2}$ MPa, six to seven orders under the diagonal. That
is the numerical signature of a correctly tied periodic cell. Second, the transverse plane is not
isotropic: a transversely isotropic material would satisfy $D_{44} = (D_{22} - D_{23})/2 = 3621.7$
MPa, whereas the computed $D_{44}$ is 3095.6 MPa, a $15\,\%$ gap. A square fibre array is tetragonal,
so $G_{23}$ has to be homogenized rather than inferred from $E_2$ and $\nu_{23}$.

### A second SG: square thin-walled section

% source: examples/OpenSG-solid/5_get_solid_props_from_SG/square_section/README.md
The subfolder `square_section/` homogenizes a square tube cross-section (wall midline side
$a = 1.0$, wall thickness $t = 0.03$, one orthotropic ply at $-45^\circ$) to the equivalent 3-D
solid. The SG `square_tube_2Dsolid.yaml` has 3228 nodes and 5384 linear triangles, with
`elementOrientations` that put $e_1$ out of plane. The material, in Pa, is
E = [142.0e9, 9.8e9, 9.8e9], G = [6.0e9, 6.0e9, 4.8e9], nu = [0.30, 0.30, 0.42], rho = 1600. Here
$\omega$ is the material area, 0.12. `python solid_props_new_architecture.py` runs `plate_homo_2d`
with `n_model = 3`. `solid_props_old_architecture.py` is an independent single-script driver that
reads the same mesh as `square_tube.sc` and needs its own environment (see the folder's
`README.md`); `python compare_architectures.py` tabulates the two.

% source: examples/OpenSG-solid/5_get_solid_props_from_SG/square_section/square_solid_compare.dat
| Constant | `plate_homo_2d` | single script | Difference (%) |
|---|---|---|---|
| $E_1$ (Pa) | 1.42000000e+11 | 1.42000000e+11 | +0.0000 |
| $E_2$ (Pa) | 5.11796756e+09 | 5.11796686e+09 | +0.0000 |
| $E_3$ (Pa) | 5.11790821e+09 | 5.11790757e+09 | +0.0000 |
| $G_{23}$ (Pa) | 9.64591556e+06 | 9.64624458e+06 | -0.0034 |
| $G_{13}$ (Pa) | 3.14483963e+09 | 3.14483951e+09 | +0.0000 |
| $G_{12}$ (Pa) | 3.14483389e+09 | 3.14483378e+09 | +0.0000 |
| $\nu_{12}$ | 3.00000000e-01 | 3.00003705e-01 | -0.0012 |
| $\nu_{13}$ | 3.00000000e-01 | 3.00003888e-01 | -0.0013 |
| $\nu_{23}$ | 2.67504869e-02 | 2.67543067e-02 | -0.0143 |

% source: examples/OpenSG-solid/5_get_solid_props_from_SG/square_section/README.md ("The one capability difference")
This comparison runs the ply unrotated (`USE_ORIENTATIONS = False`), the only setting a `.sc` can
express, and $E_1$ = 142 GPa exactly, as it must be when the fibres run along the beam axis. With
`USE_ORIENTATIONS = True` the driver applies the per-element frames of the yaml with the real
$-45^\circ$ ply and gives $E_1$ = 1.473164e10 Pa, the analytical $-45^\circ$ off-axis lamina
modulus (14.73 GPa).

## Files

| File | Content |
|---|---|
| `UDcomp_2D.msh` | gmsh 2.2 mesh of the unit cell |
| `make_udcomp_yaml.py` | mesh to yaml, sets the materials and the phases |
| `UDcomp_2D.yaml` | the 2-D SG: nodes, triangles, frames, materials, element sets |
| `solid_homo_2dsg.py` | the `n_model = 3` driver |
| `UDcomp_2D_Deff.out` | $\mathbf{D}_{\rm eff}$ in MPa, Voigt order `[e11 e22 e33 2e23 2e13 2e12]`, $\omega$ in the header |
| `UDcomp_2D_constants_vs_paper.dat` | the nine constants, the published values and the difference |
| `UDcomp_2D_mesh.png` | the unit cell, elements coloured by phase |
| `square_section/` | the square-tube SG, its two drivers, `.out` files and `square_solid_compare.dat` |
