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
            V[i * n_cols + j, :] = [x, y, 0.0]

    # Create the faces matrix (defining quadrilaterals)
    F = np.zeros(((n_rows - 1) * (n_cols - 1), 4), dtype=np.int64)
    for i in range(n_rows - 1):
        for j in range(n_cols - 1):
            f = i * (n_cols - 1) + j
            F[f, :] = [
                i * n_cols + j,
                (i + 1) * n_cols + j,
                i * n_cols + j + 1,
                (i + 1) * n_cols + j + 1,
            ]

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
            V[i * n_cols + j, :] = [x, y, 0.0]

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
    vertices: np.ndarray[Any, np.dtype[np.float32 | np.float64]],
    cells: np.ndarray[Any, np.dtype[np.uint64]],
    number_of_subdivisions: int,
) -> tuple[
    np.ndarray[Any, np.dtype[np.float32 | np.float64]],
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
    cells: np.ndarray[Any, np.dtype[np.uint64]] | np.ndarray[Any, np.dtype[np.int64]],
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
    vertices: np.ndarray[Any, np.dtype[np.float32 | np.float64]],
    cells: np.ndarray[Any, np.dtype[np.uint64]],
) -> np.ndarray[Any, np.dtype[np.float32 | np.float64]]:
    x_end = np.zeros(
        (cells.shape[0], cells.shape[1], vertices.shape[1]), dtype=vertices.dtype
    )
    for i in range(cells.shape[0]):
        for j in range(cells.shape[1]):
            x_end[i, j] = vertices[cells[i, j]]
    return x_end


# @timer()
def mesh_to_jax(
    vertices: np.ndarray[Any, np.dtype[np.float32 | np.float64]],
    cells: np.ndarray[Any, np.dtype[np.uint64]],
) -> jnp.ndarray:
    """
    Given the vertex coordinates and list of connectivity as a list of vertex indices,
    this returns a 3-dimensional arry describing the elements in terms of vertices.

    Returns
    -------
    ```
    For example, in 2D the result would be:
    [ [[e0_v0_x, e0_v0_y],
        [e0_v1_x, e0_v1_y],
        [e0_v2_x, e0_v2_y]],
        ...,
        [[eN_v0_x, eN_v0_y],
        [eN_v1_x, eN_v1_y],
        [eN_v2_x, eN_v2_y]]
    ]
    ```
    """
    return jnp.array(mesh_to_jax_helper(vertices, cells))


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

@struct.dataclass
class AssemblyMap:
    """
    Container for the information required to perform EN-V and V-EN transformations via indexing rather than sparse matmul.

    Fields
    ---------
    indices: Array of shape (EN,) whose entries are the indices from a vertex-based array corresponding to each element-node. 
        Equal to the row_indices array of the sparse array approach, with entries sorted in EN order (such that the col_indices array would be exactly jnp.arange(EN))
    shape: tuple with entries (V,EN), equal to the shape of a sparse array whose matmuls produce the desired transformations
    """

    indices: jnp.ndarray
    shape: tuple[int] = struct.field(pytree_node=False)  

@partial(jax.jit,static_argnames = "n_vertices")
def mesh_to_sparse_assembly_map(
    n_vertices: int,
    cells: jnp.ndarray,
):
    """
    Generates an array of indices to convert between vertex-labeled values and element-node-labeled values
    """
    VtoEN_indices = jnp.searchsorted(
        jnp.arange(n_vertices),
        cells,
        method="scan_unrolled",
    )
    return AssemblyMap(indices=VtoEN_indices, shape=(n_vertices, np.prod(cells.shape)))


@jax.jit
def transform_global_to_element_node(
    assembly_map: AssemblyMap, v_g: jnp.ndarray
):
    """
    Transforms a vector that represents a global assembled vector into the element-node representation.

    TODO: change this to transform into batches (keep batch info in Dimensions)
    """
    return v_g.at[assembly_map.indices, :].get(mode="drop", fill_value=0)


@jax.jit
def transform_global_unraveled_to_element_node(
    assembly_map: AssemblyMap, v_g: jnp.ndarray
):
    """
    Transforms a vector that represents a global assembled vector that is unraveled into the
    element-node representation.

    TODO: change this to transform into batches (keep batch info in Dimensions)
    """
    V = assembly_map.shape[0]
    U = v_g.shape[0] // V
    return (
        v_g.reshape((V, U)).at[assembly_map.indices, :].get(mode="drop", fill_value=0)
    )

@jax.jit
def transform_element_node_to_global_unraveled_nosum(
    assembly_map: AssemblyMap, v_en: jnp.ndarray
):
    """
    TODO document
    """
    U = v_en.shape[2]
    V, EN = assembly_map.shape
    v_g = jnp.zeros((V, U)).at[assembly_map.indices, ...].set(v_en, mode="drop")
    return v_g.reshape(np.prod(v_g.shape))


@jax.jit
def transform_element_node_to_global_unraveled_sum(
    assembly_map: AssemblyMap, v_en: jnp.ndarray
):
    """
    TODO document
    """
    U = v_en.shape[2]
    V, EN = assembly_map.shape
    v_g = jnp.zeros((V, U)).at[assembly_map.indices, ...].add(v_en, mode="drop")
    return v_g.flatten()


@jax.jit
def transform_element_node_to_global_sum(
    assembly_map: AssemblyMap, v_en: jnp.ndarray
):
    """
    TODO document
    """
    U = v_en.shape[2]
    V, EN = assembly_map.shape
    v_g = jnp.empty((V, U)).at[assembly_map.indices, ...].add(v_en, mode="drop")
    return v_g


@jax.jit
def transform_element_node_to_global_nosum(
    assembly_map: AssemblyMap, v_en: jnp.ndarray
):
    """
    TODO document
    """
    U = v_en.shape[2]
    V, EN = assembly_map.shape
    v_g = jnp.zeros((V, U)).at[assembly_map.indices, ...].set(v_en, mode="drop")
    return v_g


# ---------------------------------------------------------------------------
# Periodic boundary conditions (verbatim from the installed fe_jax core).
# Opposite bounding-box faces are paired: right -> left shifted by Lx,
# top -> bottom by Ly (and front -> back by Lz in 3-D); the repeated
# dof_map = dof_map[dof_map] resolves corner/edge chains.
# ---------------------------------------------------------------------------
from scipy.spatial.distance import cdist


def periodic_map(points, tol=1e-6, ndof_per_node=3):
    min_xyz = np.min(points, axis=0)
    max_xyz = np.max(points, axis=0)
    num_nodes = len(points)
    dof_map = np.arange(num_nodes)
    p = points.shape[1]
    if p == 2:
        left_points = np.isclose(points[:, 0], min_xyz[0], atol=1e-6).nonzero()[0]
        right_points = np.isclose(points[:, 0], max_xyz[0], atol=1e-6).nonzero()[0]
        bottom_points = np.isclose(points[:, 1], min_xyz[1], atol=1e-6).nonzero()[0]
        top_points = np.isclose(points[:, 1], max_xyz[1], atol=1e-6).nonzero()[0]

        Lx = max_xyz[0] - min_xyz[0]
        Ly = max_xyz[1] - min_xyz[1]

        def map_boundary(slaves, masters, shift_vec):
            slave_pts = points[slaves]
            master_pts = points[masters]
            target_pos = slave_pts - shift_vec
            dists = cdist(target_pos, master_pts)
            nearest_idx = np.argmin(dists, axis=1)
            min_dists = dists[np.arange(len(dists)), nearest_idx]

            if not np.all(min_dists < tol):
                raise ValueError("Geometric mismatch on periodic boundary")

            dof_map[slaves] = masters[nearest_idx]

        map_boundary(right_points, left_points, np.array([Lx, 0.0]))
        map_boundary(top_points, bottom_points, np.array([0.0, Ly]))

    elif p == 3:
        left_points = np.isclose(points[:, 0], min_xyz[0], atol=1e-6).nonzero()[0]
        right_points = np.isclose(points[:, 0], max_xyz[0], atol=1e-6).nonzero()[0]
        bottom_points = np.isclose(points[:, 1], min_xyz[1], atol=1e-6).nonzero()[0]
        top_points = np.isclose(points[:, 1], max_xyz[1], atol=1e-6).nonzero()[0]
        back_points = np.isclose(points[:, 2], min_xyz[2], atol=1e-6).nonzero()[0]
        front_points = np.isclose(points[:, 2], max_xyz[2], atol=1e-6).nonzero()[0]
        num_nodes = len(points)
        dof_map = np.arange(num_nodes)

        Lx = max_xyz[0] - min_xyz[0]
        Ly = max_xyz[1] - min_xyz[1]
        Lz = max_xyz[2] - min_xyz[2]

        def map_boundary(slaves, masters, shift_vec):
            if len(slaves) == 0:
                return

            slave_pts = points[slaves]
            master_pts = points[masters]
            target_pos = slave_pts - shift_vec

            dists = cdist(target_pos, master_pts)
            nearest_idx = np.argmin(dists, axis=1)
            min_dists = dists[np.arange(len(dists)), nearest_idx]

            if not np.all(min_dists < tol):
                raise ValueError(
                    f"Geometric mismatch on periodic boundary with shift {shift_vec}")

            dof_map[slaves] = masters[nearest_idx]

        map_boundary(right_points, left_points, np.array([Lx, 0.0, 0.0]))
        map_boundary(top_points, bottom_points, np.array([0.0, Ly, 0.0]))
        map_boundary(front_points, back_points, np.array([0.0, 0.0, Lz]))

    for _ in range(p):
        dof_map = dof_map[dof_map]

    node_periodic_map = jnp.array(dof_map)
    master_nodes = node_periodic_map
    dof_offsets = jnp.arange(ndof_per_node)
    master_dof_indices = master_nodes[:, None] * ndof_per_node + dof_offsets[None, :]
    return master_dof_indices.flatten()


def dof_map_full(points, tol=1e-6, ndof_per_node=3):
    dof_map_disp = periodic_map(points, tol)
    lambda_indices = jnp.arange(len(dof_map_disp), 3*points.shape[0]+3)
    return jnp.concatenate([dof_map_disp, lambda_indices])


def mesh_to_periodic_sparse_assembly_map(V, cells, points, ndof_per_node=3,
                                         tol=1e-6):
    dof_map_np = np.array(dof_map_full(points, tol))
    master_nodes = dof_map_np[:-3][::ndof_per_node] // ndof_per_node
    master_nodes = master_nodes.astype(np.uint64)

    periodic_cells = master_nodes[np.array(cells, dtype=np.uint64)]

    return jnp.array(periodic_cells, dtype=jnp.int32), dof_map_np


def periodic_node_map(points, tol=1e-6):
    """Node-level master map (n_nodes,) from the core periodic_map -- the form
    the shell ring SG needs, where a node carries 6 DOFs rather than 3."""
    return np.asarray(periodic_map(np.asarray(points), tol,
                                   ndof_per_node=1), dtype=int)
