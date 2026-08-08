from .basis_quadrature import *
from .utils import *

import jax.numpy as jnp
from dataclasses import dataclass
import numpy as np
import jax.experimental.sparse as jsparse
from functools import partial

from numba import njit

from typing import Any

import igl
from scipy.spatial.distance import cdist

@njit
def uniform_quad_grid(n_rows: int, n_cols: int, bbox):
    """
    Creates a uniform grid of quadrilaters with a specified extent for both x and y.

    Parameters
    ----------
    n_rows  : int, number of rows of vertices
    n_cols  : int, number of columns of vertices
    bbox     : array with shape (D, 2)

    Returns
    ----------
    vertices    : dense 2d-array with shape (# verts, 3)
    cells       : dense 2d-array with shape (# elements, 3)
    """

    # Create the grid coordinates
    x_start = bbox[0, 0]
    x_stop = bbox[0, 1]
    y_start = bbox[1, 0]
    y_stop = bbox[1, 1]

    # Create the vertices matrix
    V = np.zeros((n_rows * n_cols, 3), dtype=np.float64)
    for i in range(n_rows):
        x = i / float(n_rows - 1) * (x_stop - x_start) + x_start
        for j in range(n_cols):
            y = j / float(n_cols - 1) * (y_stop - y_start) + y_start
            V[i * n_cols + j, :] = [x, y, 0.]

    # Create the faces matrix (defining quadrilaterals)
    F = np.zeros(((n_rows - 1) * (n_cols - 1), 4), dtype=np.int64)
    for i in range(n_rows - 1):
        for j in range(n_cols - 1):
            f = i *  (n_cols - 1) +  j
            F[f, :] = [i * n_cols + j, (i + 1) * n_cols + j, i * n_cols + j + 1, (i + 1) * n_cols + j + 1]

    return (V, F)



@njit
def uniform_tri_grid(n_rows: int, n_cols: int):
    """
    Creates a uniform grid of triangles with an extent for both x and y of [0, 1].

    Parameters
    ----------
    n_rows  : int, number of rows of vertices
    n_cols  : int, number of columns of vertices

    Returns
    ----------
    vertices    : dense 2d-array with shape (# verts, 3)
    cells       : dense 2d-array with shape (# elements, 3)
    """

    # Create the grid coordinates
    x_start = 0.0
    x_stop = 1.0
    y_start = 0.0
    y_stop = 1.0

    # Create the vertices matrix
    V = np.zeros((n_rows * n_cols, 3), dtype=np.float64)
    for i in range(n_rows):
        x = i / float(n_rows - 1) * (x_stop - x_start) + x_start
        for j in range(n_cols):
            y = j / float(n_cols - 1) * (y_stop - y_start) + y_start
            V[i * n_cols + j, :] = [x, y, 0.]

    # Create the faces matrix (defining triangles)
    F = np.zeros((2 * (n_rows - 1) * (n_cols - 1), 3), dtype=np.int64)
    for i in range(n_rows - 1):
        for j in range(n_cols - 1):
            f = i * 2 * (n_cols - 1) + 2 * j
            F[f, :] = [i * n_cols + j, (i + 1) * n_cols + j, i * n_cols + j + 1]
            f += 1
            F[f, :] = [
                (i + 1) * n_cols + j,
                (i + 1) * n_cols + j + 1,
                i * n_cols + j + 1,
            ]

    return (V, F)


def refine_tri_mesh(
    vertices: np.ndarray[Any, np.dtype[np.float32|np.float64]],
    cells: np.ndarray[Any, np.dtype[np.uint64]],
    number_of_subdivisions: int,
) -> tuple[
    np.ndarray[Any, np.dtype[np.float32|np.float64]],
    np.ndarray[Any, np.dtype[np.uint64]],
]:
    """
    Given a triangle mesh, this subdivides each triangle uniformly to create a more refined mesh
    without changing the coarse features of the mesh.

    Parameters
    ----------
    vertices    : dense 2d-array with shape (# verts, 3)
    cells       : dense 2d-array with shape (# elements, 3)

    Returns
    -------
    refined_vertices    : dense 2d-array with shape (# verts, 3)
    refined_cells       : dense 2d-array with shape (# elements, 3)

    """
    return igl.upsample(V=vertices, F=cells, number_of_subdivs=number_of_subdivisions)


def find_tri_mesh_boundary_verts(
    cells: np.ndarray[Any, np.dtype[np.uint64]] | np.ndarray[Any, np.dtype[np.int64]]
) -> np.ndarray[Any, np.dtype[np.uint64]] | np.ndarray[Any, np.dtype[np.int64]]:
    """
    Given a triangle mesh, this finds the vertices along the boundary of the mesh.

    Parameters
    ----------
    vertices    : dense 2d-array with shape (# verts, 3)
    cells       : dense 2d-array with shape (# elements, 3)

    Returns
    -------
    boundary_verts    : dense 1d-array with shape (# boundary verts,)
    """

    boundary_line_segments = igl.boundary_facets(cells)[0]
    return np.unique(boundary_line_segments)


@njit
def mesh_to_jax_helper(
    vertices: np.ndarray[Any, np.dtype[np.float32|np.float64]],
    cells: np.ndarray[Any, np.dtype[np.uint64]],
) -> np.ndarray[Any, np.dtype[np.float32|np.float64]]:
    x_n = np.zeros((cells.shape[0], cells.shape[1], vertices.shape[1]), dtype=vertices.dtype)
    for i in range(cells.shape[0]):
        for j in range(cells.shape[1]):
            # Note: [0:2] is added to ignore z-coordinate if one exists.
            x_n[i, j] = vertices[cells[i, j]][0:vertices.shape[1]]
    return x_n


# @timer()
def mesh_to_jax(
    vertices: np.ndarray[Any, np.dtype[np.float32|np.float64]],
    cells: np.ndarray[Any, np.dtype[np.uint64]],
) -> jnp.ndarray:
    """
    Given the vertex coordinates and list of triangles as a triplet of vertex indices,
    this returns a 3-dimensional arry describing the triangles in terms of vertices.

    Returns
    -------
    ```
    [ [[t0_v0_x, t0_v0_y],
        [t0_v1_x, t0_v1_y],
        [t0_v2_x, t0_v2_y]],
        ...,
        [[tN_v0_x, tN_v0_y],
        [tN_v1_x, tN_v1_y],
        [tN_v2_x, tN_v2_y]]
    ]
    ```
    """
    # 1. Transfer the SMALL, compressed arrays to the GPU first
    j_vertices = jnp.array(vertices)
    j_cells = jnp.array(cells)
    
    # 2. Let the GPU instantly expand the coordinates 
    # j_vertices[j_cells] automatically creates an array of shape (E, N, D)
    x_end = j_vertices[j_cells]
    return x_end


@njit
def get_n_cells_per_vert_helper(
    vertices: np.ndarray[Any, np.dtype[np.float32]],
    cells: np.ndarray[Any, np.dtype[np.uint64]],
) -> np.ndarray[Any, np.dtype[np.uint64]]:
    n_cells_per_vert = np.zeros((vertices.shape[0],), dtype=np.uint64)
    for i in range(cells.shape[0]):
        for j in range(cells.shape[1]):
            n_cells_per_vert[cells[i, j]] += 1
    return n_cells_per_vert


# @timer()
def get_n_cells_per_vert(
    vertices: np.ndarray[Any, np.dtype[np.floating]],
    cells: np.ndarray[Any, np.dtype[np.uint64]],
) -> jnp.ndarray:
    """
    Returns an array that describes the number of cells connected to each vertex.
    """
    return jnp.array(get_n_cells_per_vert_helper(vertices, cells))


@njit
def build_row_ind(
    n_vertices: int, cells: np.ndarray[Any, np.dtype[np.uint64]]
) -> np.ndarray[Any, np.dtype[np.uint64]]:
    """
    Creates row offset map for the compressed sparse row (CSR) format.
    """
    csr_row_ind = np.zeros((n_vertices + 1,), dtype=np.uint64)
    one = np.uint64(1)
    for i in range(cells.shape[0]):
        for j in range(cells.shape[1]):
            csr_row_ind[np.uint64(cells[i, j]) + one] += 1
    return np.cumsum(csr_row_ind)


@njit
def build_col_ind_and_data(
    csr_row_ind: np.ndarray[Any, np.dtype[np.uint64]],
    cells: np.ndarray[Any, np.dtype[np.uint64]],
) -> tuple[np.ndarray[Any, np.dtype[np.uint64]], np.ndarray[Any, np.dtype[np.float32|np.float64]]]:
    """
    Create column offset map and data for the compressed sparse row (CSR) format.
    """
    n_verts_per_cell = cells.shape[1]
    curr_col_offset = np.zeros((csr_row_ind.shape[0],), dtype=np.uint64)
    csr_col_ind = np.zeros((csr_row_ind[-1],), dtype=np.uint64)
    csr_data = np.zeros((csr_row_ind[-1],))
    for cell_index, cell in enumerate(cells):
        for i, vert_index in enumerate(cell):
            index = csr_row_ind[vert_index] + curr_col_offset[vert_index]
            col = cell_index * n_verts_per_cell + i
            csr_col_ind[index] = col
            csr_data[index] = 1.0
            curr_col_offset[vert_index] += 1
    return (csr_col_ind, csr_data)


# @timer()
def mesh_to_sparse_assembly_map(
    n_vertices: int,
    cells: np.ndarray[Any, np.dtype[np.uint64]],
) -> jsparse.BCSR:
    """
    Builds a compressed sparse row (CSR) matrix that serves as a map between two representations
    for vectors: 1) globally assembled format and 2) batched element-node format.

    NOTE: A map can only be created for a single type of cell, so different cell types need
    to be organized into separate batches.

    Parameters
    ----------
    vertices    : dense 2d-array with shape (# verts, N_x)
    cells       : dense 2d-array with shape (# elements, V)

    Returns
    -------
    assembly_map : sparse 3d-array with shape (# batches, # verts, # elements * V)
    """

    # Create row offset map for the compressed sparse row (CSR) format
    csr_row_ind = build_row_ind(n_vertices, cells)
    csr_col_ind, csr_data = build_col_ind_and_data(csr_row_ind, cells)

    # Make a batch of 1
    csr_row_ind = csr_row_ind.reshape(1, csr_row_ind.shape[0])
    csr_col_ind = csr_col_ind.reshape(1, csr_col_ind.shape[0])
    csr_data = csr_data.reshape(1, csr_data.shape[0])

    return jsparse.BCSR(
        (csr_data, csr_col_ind, csr_row_ind),
        shape=(1, n_vertices, cells.shape[0] * cells.shape[1]),
        indices_sorted=True,
        unique_indices=True,
    )


@partial(jax.jit,static_argnames = ["E","V","U"])
def transform_global_to_element_node(
    assembly_map: jsparse.BCSR, v_g: jnp.ndarray, E: int, V: int, U: int
):
    """
    Transforms a vector that represents a global assembled vector into the element-node representation.

    TODO: change this to transform into batches (keep batch info in Dimensions)
    """
    return jsparse.bcsr_dot_general(
        assembly_map,
        v_g.reshape(1, v_g.shape[0], v_g.shape[1]),
        dimension_numbers=(((1,), (1,)), ((0,), (0,))),
    ).reshape(E, V, U)


@partial(jax.jit, static_argnames=["U"])
def transform_global_unraveled_to_element_node(
    periodic_cells: jnp.ndarray, v_g: jnp.ndarray, U: int =3
):
    """
    Transforms a vector that represents a global assembled vector that is unraveled into the
    element-node representation.

    TODO: change this to transform into batches (keep batch info in Dimensions)
    """
    return v_g.reshape(-1, U)[periodic_cells]


@partial(jax.jit, static_argnames=["V"])
def transform_element_node_to_global_unraveled_nosum(
    periodic_cells: jnp.ndarray, v_en: jnp.ndarray, V:int
):
    """
    Returns:
        Flat 1D array of summed global contributions: (num_nodes * ndof,)
    """
    ndof = v_en.shape[-1]
    v_g_nodes = jnp.zeros((num_nodes, ndof), dtype=v_en.dtype)
    v_g_nodes = v_g_nodes.at[periodic_cells].add(v_en)
    return v_g_nodes.ravel()
    
   # n_cell_per_vert = (
   #     assembly_map.indptr[0, 1:] - assembly_map.indptr[0, :-1]
   # )
   # v_g = jsparse.bcsr_dot_general(
    #    assembly_map,
    #    v_en.reshape(1, v_en.shape[0] * v_en.shape[1], v_en.shape[2]),
    #    dimension_numbers=(((2,), (1,)), ((0,), (0,))),
   # )
   # return (v_g / n_cell_per_vert[jnp.newaxis, :, jnp.newaxis]).reshape(
   #     np.prod(v_g.shape)
   # )
@jax.jit
def transform_element_node_to_global_unraveled_sum_sparse(
    assembly_map: jsparse.BCSR, v_en: jnp.ndarray
):
    """Original sparse scatter to guarantee deterministic summation for Krylov solvers."""
    v_g = jsparse.bcsr_dot_general(
        assembly_map,
        v_en.reshape(1, v_en.shape[0] * v_en.shape[1], v_en.shape[2]),
        dimension_numbers=(((2,), (1,)), ((0,), (0,))),
    )
    return v_g.reshape(np.prod(v_g.shape))
    

@partial(jax.jit, static_argnames=["num_nodes"])
def transform_element_node_to_global_unraveled_sum(
    periodic_cells: jnp.ndarray, v_en: jnp.ndarray, num_nodes:int, U: int=3, 
):
    """
    TODO document
    """
    ndof = v_en.shape[-1]
    v_g_nodes = jnp.zeros((num_nodes, ndof), dtype=v_en.dtype)
    v_g_nodes = v_g_nodes.at[periodic_cells].add(v_en)
    return v_g_nodes.ravel()
    v_g = jsparse.bcsr_dot_general(
        assembly_map,
        v_en.reshape(1, v_en.shape[0] * v_en.shape[1], v_en.shape[2]),
        dimension_numbers=(((2,), (1,)), ((0,), (0,))),
    )
    return v_g.reshape(np.prod(v_g.shape))

@jax.jit
def transform_element_node_to_global_sum(
    assembly_map: jsparse.BCSR, v_en: jnp.ndarray
):
    """
    TODO document
    """
    v_g = jsparse.bcsr_dot_general(
        assembly_map,
        v_en.reshape(1, v_en.shape[0] * v_en.shape[1], v_en.shape[2]),
        dimension_numbers=(((2,), (1,)), ((0,), (0,))),
    )
    return v_g.reshape(np.prod(v_g.shape)//v_en.shape[2],v_en.shape[2])

@jax.jit
def transform_element_node_to_global_nosum(
    assembly_map: jsparse.BCSR, v_en: jnp.ndarray
):
    """
    TODO document
    """
    n_cell_per_vert = (
        assembly_map.indptr[0, 1:] - assembly_map.indptr[0, :-1]
    )
    v_g = jsparse.bcsr_dot_general(
        assembly_map,
        v_en.reshape(1, v_en.shape[0] * v_en.shape[1], v_en.shape[2]),
        dimension_numbers=(((2,), (1,)), ((0,), (0,))),
    )
    return (v_g / n_cell_per_vert[jnp.newaxis, :, jnp.newaxis]).reshape(np.prod(v_g.shape)//v_en.shape[2],v_en.shape[2])
    
# Added by Akshat for periodic assembly map

def mesh_to_periodic_sparse_assembly_map(
    V: int,
    cells: jnp.ndarray,
    points: jnp.ndarray,
    n_model: int,
    ndof_per_node: int = 3,
    atol=1e-6,
):
    dof_map_np = np.array(periodic_map(points, n_model, atol))
    master_nodes = dof_map_np[:][::ndof_per_node] // ndof_per_node
    master_nodes = master_nodes.astype(np.uint64) 

    # --- THE NEW COMPRESSION STEP ---
    # 1. Find the strictly unique master nodes (This is exactly your reduced set)
    unique_masters = np.unique(master_nodes)
    
    # 2. Create a translation array: Full Node ID -> Reduced Node ID
    full_to_reduced = np.full(V, -1, dtype=np.int32)
    full_to_reduced[unique_masters] = np.arange(len(unique_masters), dtype=np.int32)
    
    # 3. Map the original cells: Slave -> Master -> Reduced
    # master_nodes[cells] gets the master ID. 
    # full_to_reduced[...] converts it to the 0-indexed reduced space.
    reduced_periodic_cells = full_to_reduced[master_nodes[cells]]
    # --------------------------------

    return jnp.array(reduced_periodic_cells, dtype=jnp.int32), dof_map_np
        
def periodic_map(points, n_model, atol=1e-6, ndof_per_node=3):
    min_xyz = np.min(points, axis=0)
    max_xyz = np.max(points, axis=0) 
    num_nodes = len(points)
    dof_map = np.arange(num_nodes)
    n_sg=points.shape[1]
    if n_sg == 1 and n_model==3:
        left_points = np.isclose(points[:, 0], min_xyz[0], atol=1e-6).nonzero()[0]
        right_points = np.isclose(points[:, 0], max_xyz[0], atol=1e-6).nonzero()[0]

        # Calculate Box Dimensions for 1D
        Lx = max_xyz[0] - min_xyz[0]
        
        def map_boundary(slaves, masters, shift_vec):
            if len(slaves) == 0: return # Handle empty sets if boundary is empty
            
            slave_pts = points[slaves]
            master_pts = points[masters]
            target_pos = slave_pts - shift_vec
            
            dists = cdist(target_pos, master_pts)
            nearest_idx = np.argmin(dists, axis=1)
            min_dists = dists[np.arange(len(dists)), nearest_idx]
            
            if not np.all(min_dists < atol):
                raise ValueError("Geometric mismatch on periodic boundary")
                
            # Update the global map
            dof_map[slaves] = masters[nearest_idx]

        # 1. Map Right -> Left (Shift x by Lx)
        map_boundary(right_points, left_points, np.array([Lx]))
        
    elif n_sg==2:
        left_points = np.isclose(points[:, 0], min_xyz[0], atol=1e-6).nonzero()[0]
        right_points = np.isclose(points[:, 0], max_xyz[0], atol=1e-6).nonzero()[0]
        bottom_points = np.isclose(points[:, 1], min_xyz[1], atol=1e-6).nonzero()[0]
        top_points = np.isclose(points[:, 1], max_xyz[1], atol=1e-6).nonzero()[0]

        # Calculate Box Dimensions
        Lx = max_xyz[0] - min_xyz[0]
        Ly = max_xyz[1] - min_xyz[1]
        def map_boundary(slaves, masters, shift_vec):
            slave_pts = points[slaves]
            master_pts = points[masters]
            target_pos = slave_pts - shift_vec
            dists = cdist(target_pos, master_pts)
            nearest_idx = np.argmin(dists, axis=1)
            min_dists = dists[np.arange(len(dists)), nearest_idx]
            
            if not np.all(min_dists < atol):
                raise ValueError("Geometric mismatch on periodic boundary")
                
            # Update the global map
            dof_map[slaves] = masters[nearest_idx]
        if n_model==3: 
            # 1. Map Right -> Left (Shift x by -Lx)
            map_boundary(right_points, left_points, np.array([Lx, 0.0]))
            # 2. Map Top -> Bottom (Shift y by -Ly)
            map_boundary(top_points, bottom_points, np.array([0.0, Ly]))
        elif n_model==2:
            # 2. Map Top -> Bottom (Shift y by -Ly)
            map_boundary(right_points, left_points, np.array([Lx, 0.0]))
            
        #  while True:
        #      new_map = dof_map[dof_map]
        #      if np.array_equal(new_map, dof_map):
        #          break # Everything is fully resolved!
        #      dof_map = new_map    
        for _ in range(n_sg):
            dof_map = dof_map[dof_map]
    elif n_sg==3:
        left_points = np.isclose(points[:, 0], min_xyz[0], atol=1e-6).nonzero()[0]
        right_points = np.isclose(points[:, 0], max_xyz[0], atol=1e-6).nonzero()[0]
        bottom_points = np.isclose(points[:, 1], min_xyz[1], atol=1e-6).nonzero()[0]
        top_points = np.isclose(points[:, 1], max_xyz[1], atol=1e-6).nonzero()[0]
        back_points = np.isclose(points[:, 2], min_xyz[2], atol=1e-6).nonzero()[0]
        front_points = np.isclose(points[:, 2], max_xyz[2], atol=1e-6).nonzero()[0]
        num_nodes = len(points)
        dof_map = np.arange(num_nodes)
    
        # Calculate Box Dimensions for 3D
        Lx = max_xyz[0] - min_xyz[0]
        Ly = max_xyz[1] - min_xyz[1]
        Lz = max_xyz[2] - min_xyz[2]
        
        def map_boundary(slaves, masters, shift_vec):
            if len(slaves) == 0: return # Handle empty sets if boundary is empty
            
            slave_pts = points[slaves]
            master_pts = points[masters]
            target_pos = slave_pts - shift_vec

            dists = cdist(target_pos, master_pts)
            nearest_idx = np.argmin(dists, axis=1)
            min_dists = dists[np.arange(len(dists)), nearest_idx]
            
            if not np.all(min_dists < atol):
                raise ValueError(f"Geometric mismatch on periodic boundary with shift {shift_vec}")
                
            dof_map[slaves] = masters[nearest_idx]
        if n_model==3:
             # 1. Map Right -> Left (Shift X)
             map_boundary(right_points, left_points, np.array([Lx, 0.0, 0.0]))
             # 2. Map Top -> Bottom (Shift Y)
             map_boundary(top_points, bottom_points, np.array([0.0, Ly, 0.0]))
             # 3. Map Front -> Back (Shift Z)
             map_boundary(front_points, back_points, np.array([0.0, 0.0, Lz]))
          
        elif n_model==2:
             # 2. Map Right -> Left (Shift X)
             map_boundary(right_points, left_points, np.array([Lx, 0.0, 0.0]))
             # 3. Map Top -> Bottom (Shift Y)
             map_boundary(top_points, bottom_points, np.array([0.0, Ly, 0.0]))
            
        else:
             # 2. Map Right -> Left (Shift X)
             map_boundary(right_points, left_points, np.array([Lx, 0.0, 0.0]))
             
        for _ in range(n_sg):
            dof_map = dof_map[dof_map]
        
    node_periodic_map= jnp.array(dof_map)
    master_nodes = node_periodic_map  
    dof_offsets = jnp.arange(ndof_per_node)
    master_dof_indices = master_nodes[:, None] * ndof_per_node + dof_offsets[None, :]    
    return master_dof_indices.flatten()