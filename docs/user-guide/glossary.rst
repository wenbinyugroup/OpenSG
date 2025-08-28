.. _glossary:


Glossary
============

This page contains definitions for various terminology and abbreviations
used throughout opensg documentation and code. 

Terminology
-----------

ABD Matrix: A 6x6 matrix used in classical lamination theory to relate generalized forces and moments to generalized strains and curvatures. The matrix has the structure [[A, B], [B, D]] where A is the membrane stiffness, B is the coupling stiffness, and D is the bending stiffness.

Boundary Condition: A condition imposed on facets or boundaries of a mesh, 
such as Dirichlet boundary conditions (specifying function values on the boundary) 
or Neumann boundary conditions (specifying derivative values).

Cell: A fundamental element of a mesh (e.g., a triangle in 2D, tetrahedron in 3D) 
that divides the domain into discrete parts for FEA computations.

Composite Laminate: A structural material made up of multiple layers (plies) of fiber-reinforced material, each potentially having different fiber orientations and material properties.

Composite Layup: See Composite Laminate.

Connectivity: The relationship between different entities in a mesh 
(e.g., between vertices and cells, cells and facets). 
Connectivity maps define how entities are connected within a mesh.

Dehomogenization: The process of recovering local stress and strain fields from homogenized beam properties using fluctuating functions. This allows analysis of detailed stress distributions within the original 3D structure.

Eigenvalue Analysis: Mathematical technique used to determine natural frequencies, buckling loads, or other critical values by solving generalized eigenvalue problems of the form Ax = λBx.

Euler-Bernoulli Beam Theory: A classical beam theory that assumes plane sections remain plane and perpendicular to the neutral axis after deformation. Neglects shear deformation effects.

Facet: A boundary entity of a cell that is one dimension lower than the cell itself.

Fluctuating Functions: Mathematical functions that represent the deviation from average behavior in homogenization theory. Used to capture local effects within the Structure Genome.

Frame: Local coordinate system defined at each point in the mesh, typically consisting of orthogonal unit vectors (e1, e2, e3) that define material orientations.

Gamma Functions: Mathematical operators (gamma_h, gamma_e, gamma_l, gamma_d) used in MSG formulations to relate displacement fields to strain measures for different beam theories.

Homogenization: The process of computing effective structural properties (stiffness matrices, mass properties) from detailed 3D finite element analysis of representative cross-sections.

Layup: The sequence and orientation of composite layers (plies) that make up a laminated structure, including thickness, material, and fiber angle for each layer.

Local Frame: A coordinate system attached to specific geometric entities (edges, faces) that defines the orientation of materials and the direction of structural axes.

Mesh: A collection of cells (e.g., triangles, quadrilaterals in 2D, or tetrahedrons, 
hexahedrons in 3D) representing a discretized computational domain. 
A dolfinx.mesh.Mesh object encapsulates both the geometry and topology of this mesh.

Nullspace: In finite element analysis, the mathematical space containing rigid body motions that must be constrained to ensure a unique solution to the structural problem.

Segment: A portion of a blade mesh that is analyzed as a separate entity. Wind turbine blades are typically divided into multiple segments along their span for analysis.

Shell Element: A finite element type used for modeling thin-walled structures like wind turbine blade skins.

Solid Element: A finite element type used for modeling three-dimensional volumetric structures.

Stiffness Matrix: A matrix that relates applied forces to resulting displacements in a structural system. Can be 4x4 for Euler-Bernoulli beams or 6x6 for Timoshenko beams.

Structure Gene (SG): The fundamental building block in MSG theory - a representative cross-section of the structure that captures all geometric and material complexity in 1D, 2D, or 3D form.

Timoshenko Beam Theory: An enhanced beam theory that includes the effects of shear deformation and rotary inertia, providing more accurate results for thick beams or shells.

Topology: In dolfinx, refers to the structural organization of the mesh, including 
the number and types of entities (vertices, edges, faces, cells) and their connectivity.


Abbreviations
-------------

ABD: A-B-D matrix (membrane-bending-coupling stiffness matrix)

BECAS: BEam Cross section Analysis Software (alternative cross-sectional analysis tool)

DOF: Degrees of Freedom

DOLFINx: The next-generation finite element library (part of FEniCS)

EB: Euler-Bernoulli (beam theory)

EPS: Eigenvalue Problem Solver (from SLEPc library)

FEA: Finite Element Analysis

FEniCS: Open-source finite element computing platform

GEP: Generalized Eigenvalue Problem

GMSH: 3D finite element mesh generator

MPI: Message Passing Interface (for parallel computing)

MSG: Mechanics of Structure Genome

MSH: GMSH mesh file format

PETSc: Portable Extensible Toolkit for Scientific Computation

SG: Structure Gene

SLEPc: Scalable Library for Eigenvalue Problem Computations

UFL: Unified Form Language (part of FEniCS for expressing finite element forms)

VABS: Variational Asymptotic Beam Sectional Analysis (beam cross-section analysis tool)

YAML: Yet Another Markup Language (data serialization format used for input files) 


Variable Name Conventions
--------------------------

**Timoshenko Beam Model (Tapered Shell Wind Blade Segment)**

**Input:** .yaml File

**Output:** Timoshenko Stiffness Matrix:

* Boundary Timoshenko Stiffness (Deff_l/Deff_r)
* Entire Blade Segment Timoshenko Stiffness (Deff_srt)

**Byproduct and intermediate step of computation:**

* Boundary Euler-Bernoulli Stiffness (D_effEB_l/D_effEB_r)
* Entire Blade Segment Euler-Bernoulli Stiffness (D_eff)

**Input Variables/Data:**

``section_id``
    Constant - Segment_id of wind blade to be computed

``mesh``
    mesh data of entire wind blade (WB) segment

``subdomains``
    layup id of mesh elements (a.k.a. physical domain of mesh)

``boundaries``
    facet id of boundary elements of WB mesh (Not useful in computation)

``x_min, x_max``
    minimum/maximum x (beam axis) coordinate to define boundary mesh

``tdim, fdim``
    mesh topology/facet dimension (For quad mesh: tdim=2, fdim=1)

``facets_left/right``
    Left/Right boundary facet id of WB mesh (In WB numbering) - useful in generating boundary mesh

``mesh_r/mesh_l``
    left and right boundary mesh (Generated in dolfinx as submesh)

``entity_mapl/mapr``
    facet id of left/right boundary (In boundary mesh numbering)

``num_cells``
    number of quadrilateral mesh elements

``o_cell_idx``
    Original mesh element connectivity (SG_Shell.msh). Note: dolfinx does the renumbering of vertex and mesh connectivity

``VV``
    functionspace (Discontinuous galerkin) to store orientation data

``frame ([EE1,EE2,N])``
    Local Orientation frame (from .yaml) stored as ufl function
    
    * [EE1,EE2,N]: EE1-> curvilinear tangent direction (along beam axis), EE2-> Circumferential tangent Direction (CCW about +x (beam) axis), N -> Inward normal direction

``material_parameters``
    Stores anisotropic material properties of layups (Same for all WB Segments)

**Layup and Material Data:**

``nphases``
    number of layups in the WB segment

``nlay``
    [nphases,1] - number of layers in each layup of WB segment

``matid``
    [nphases,nlay,1] - Contains thickness data of layup id- nlay

``thick``
    [nphases,nlay,1]

``conn3``
    [num_cells,4] - facet connectivity matrix for each element

``subdomains_l``
    layup id of left boundary mesh (Output for subdomains.values[:] arranged in boundary dofs)

``frame_l``
    Local orientation frame (orthogonal) for each left boundary element
    
    * [E1l,E2l,Nl]: E1l-> beam axis (+x dir), E2l-> Circumferential tangent Direction (CCW about +x (beam) axis), E3l -> Inward normal direction

``boundary_facets_left``
    [num_facets_left_boundary,1] - facet id connectivity of boundary mesh (In mesh_l based numbering)

``D_effEB_l/D_effEB_r``
    Euler-Bernoulli matrix of left/right boundary of WB Segment

``Deff_l/Deff_r``
    Timoshenko Stiffness matrix of left/right boundary of WB Segment

**Function Spaces and Solution Variables:**

``ndofs_WBmesh``
    [3*len(np.arange(\*V.dofmap.index_map.local_range)),1] - total dofs of WB segment mesh (Note: V=> functionspace of WB mesh)

``V0_l/V0_r``
    [ndofs_leftmesh,4] - Fluctuating function solution after solving 4 load cases (for EB Stiffness/a.k.a zeroth order)

``V1_l/V1_r``
    [ndofs_leftmesh,4] - Fluctuating function solution after solving 4 load cases (for Timo Stiffness/ a.k.a first order)

``e_l``
    Local orientation Frame for left boundary mesh interpolated in V_l (Serendipidity) functionspace

``e_r``
    Local orientation Frame for right boundary mesh interpolated in V_r (Serendipidity) functionspace

``e``
    Local orientation Frame for WB Segment mesh interpolated in V (Serendipidity) functionspace

``V_l/V_r``
    functionspace of left/right boundary mesh (UFL)

``V``
    functionspace for defining WB segment mesh (UFL)

``dvl, v_l``
    Trial and Test Functions for left boundary (V_l) (UFL)

``dvr, v_r``
    Trial and Test Functions for right boundary (V_r) (UFL)

``dv, v_``
    Trial and Test Functions for WB mesh (V) (UFL)

``x_l, x_r``
    Spatial Coordinate Data for left/right boundary (UFL)

``dx_l``
    Measure for left boundary mesh with subdomains_l assigned

``dx``
    Measure for WB mesh with subdomains assigned (Used in defining weak form \*dx(i) mean integration over mesh elements with layup id i)

``nullspace_l``
    contain nullspace vector for used as constraint to block Rigid body motion when solving variation form (ksp solver) over left boundary mesh

``ndofs_leftmesh``
    [3*len(np.arange(\*V.dofmap.index_map.local_range)),1] - total dofs of left boundary mesh

**Assembly Matrices and Weak Forms:**

``A_l``
    [ndofs_leftmesh,ndofs_leftmesh] - Global Assembly Coefficient matrix (in ufl form) for left boundary
    
    * Can print matrix as array by A_l[:,:]
    * Used in solving 4 load cases where A_l w_l = F_l (similar to Ax=b)
    * Unknown w_l is stored in V0_l[:,p] for load case p

``A``
    [ndofs_WBmesh,ndofs_WBmesh] - Global Assembly matrix for WB mesh

``a``
    Bilinear weak form for WB mesh - a(dv,v_)

``F2``
    Linear weak form for WB mesh - F2(v_)
    
    * Can compare with weak form to solve as a(dv,v\_) w = F2(v\_), where w.vector[ndofs_WBmesh,1] is unknown dofs value stored in V0[ndofs_WBmesh,4]

``F``
    [ndofs_WBmesh,1] - Global Assembled Right hand vector for EB case

``b``
    [ndofs_WBmesh, 4] - Equivalent to Right hand vector F for Timo case

``v0``
    Stores the solved V0[ndofs_WBmesh,p] in ufl function form for each load case

``v2a``
    UFL function defined over V (functionspace_WB mesh) - Output by v2a.vector[:]

``bc``
    Dirichlet bc (known ufl function storing boundary solutions, dofs of boundary need to be constrained)

**Stiffness Matrix Components:**

``Dhe``
    [ndofs_WBmesh,4] - < gamma_h.T ABD gamma_e>

``Dle``
    [ndofs_WBmesh,4] - < gamma_l.T ABD gamma_e>

``Dll``
    [ndofs_WBmesh,ndofs_WBmesh] - < gamma_l.T ABD gamma_l>

``Dhl``
    [ndofs_WBmesh,ndofs_WBmesh] - < gamma_h.T ABD gamma_l>

``D_ee``
    [4,4] - < gamma_e.T ABD gamma_e>

``D1``
    [4,4] - < V0.T -Dhe> Information matrix from fluctuating function data

``V0``
    [ndofs_WBmesh, 4] - Solve [A w =F] Fluctuating function after solving 4 load cases on WB mesh (EB)

``V1s``
    [ndofs_WBmesh, 4] - Solve [A w =b] Fluctuating function after solving 4 load cases on WB mesh (Timo)

``D_eff``
    [4,4] - Euler-Bernoulli Stiffness matrix of entire WB segment

``Deff_srt``
    [6,6] - Timoshenko Stiffness matrix of entire WB segment

.. note::
   For Boundary Timo Solving, Dhe, Dle, Dll, Dhl, Dee use (dofs_leftmesh/dofs_rightmesh) in place of dofs_WBmesh