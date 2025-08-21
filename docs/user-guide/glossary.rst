.. _glossary:


Glossary
============

This page contains definitions for various terminology and abbreviations
used throughout opensg documentation and code. 

Terminology
-----------

!!WIP!!

Mesh: A collection of cells (e.g., triangles, quadrilaterals in 2D, or tetrahedrons, 
hexahedrons in 3D) representing a discretized computational domain. 
A dolfinx.mesh.Mesh object encapsulates both the geometry and topology of this mesh.

Cell: A fundamental element of a mesh (e.g., a triangle in 2D, tetrahedron in 3D) 
that divides the domain into discrete parts for FEA computations.

Facet: A boundary entity of a cell that is one dimension lower than the cell itself

Boundary Condition: A condition imposed on facets or boundaries of a mesh, 
such as Dirichlet boundary conditions (specifying function values on the boundary) 
or Neumann boundary conditions (specifying derivative values).

Connectivity: The relationship between different entities in a mesh 
(e.g., between vertices and cells, cells and facets). 
Connectivity maps define how entities are connected within a mesh.

Topology: In dolfinx, refers to the structural organization of the mesh, including 
the number and types of entities (vertices, edges, faces, cells) and their connectivity.


Abbreviations
-------------
