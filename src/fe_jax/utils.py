from .profiling import *
from .np_types import *

import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils

import numpy as np

from functools import partial
from typing import Any, Literal

import inspect


def is_required(fn, arg_name) -> bool:
    """Helper function to query if an argument is required by a function given the argument name.
    """
    s = inspect.signature(fn)
    return arg_name in s.parameters.keys()


def array_slice(
    array: np.ndarray[Any, np.dtype], axis: int, start: int, end: int
) -> np.ndarray[Any, np.dtype]:
    """
    Extracts a slice of an array along a given axis.

    Parameters
    ----------
    array  : array to be sliced
    axis   : index of axis to slice along
    start  : start index of slice
    end    : end index of slice, supports negative indices

    Returns
    ----------
    sliced array : may be a shallow view of original data
    """
    return array[(slice(None),) * (axis % array.ndim) + (slice(start, end),)]


def shard_across_local_devices(
    array: np.ndarray[Any, np.dtype] | jnp.ndarray, axis: int = 0
) -> tuple[jnp.ndarray, ...]:
    """
    Shards an array across all local devices by splitting along given axis with no overlap
    between slices.

    Note: the array shape does not have to be divisible by the number of local devices but
    there will be two arrays returned (sharded portion, remainder).

    Parameters
    ----------
    array  : array to be sharded across local devices
    axis   : index of axis to shard along, default is 0

    Returns
    ----------
    tuple of arrays
     * [0] sharded portion of array, may be a shallow view of original data
     * [1] remainder portion of array on device 0, may be a shallow view of original data
    """

    mesh_shape = [1] * len(array.shape)
    mesh_shape[axis] = jax.local_device_count()  # type: ignore
    mesh_shape = tuple(mesh_shape)  # type: ignore
    devices = mesh_utils.create_device_mesh(mesh_shape)

    axis_names = ("i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t")
    axis_names = axis_names[0 : len(array.shape)]  # type: ignore
    mesh = jax.sharding.Mesh(devices, axis_names=axis_names)

    spec_axis_names = [None] * len(axis_names)
    spec_axis_names[axis] = axis_names[axis]  # type: ignore
    spec_axis_names = tuple(spec_axis_names)  # type: ignore
    sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(*spec_axis_names)
    )

    slices = slice_for_local_sharding(array=array, axis=axis)
    shard_slice = slices[0]

    # Using numpy array_split is critical since it provides a view (no copy)
    # instead of a copy.
    shard_subslices = np.array_split(shard_slice, jax.local_device_count(), axis=axis)
    # print('post array_split', get_current_pid_host_memory())
    # For most cases, this operation will be no-copy as well if destination is host memory.
    local_arrays = [
        jax.device_put(slice_, device)
        for slice_, device in zip(shard_subslices, jax.local_devices())
    ]
    # Note: make_array_from_process_local_data does not work for some reason.  I think I am
    #       not understanding the API.
    sharded_slice = jax.make_array_from_single_device_arrays(
        shape=shard_slice.shape, sharding=sharding, arrays=local_arrays
    )
    # sharded_slice.block_until_ready()
    # print('post make_array...', get_current_pid_host_memory())

    if array.shape[axis] % jax.local_device_count() > 0:
        remainder_slice = shard_slice[1]
        return (sharded_slice, jax.device_put(remainder_slice, jax.local_devices()[0]))

    else:
        return (sharded_slice,)
    

def slice_for_local_sharding(array: np.ndarray[Any, np.dtype] | jnp.ndarray, axis: int = 0):
    """
    TODO document
    """

    # If a JAX arary is past in, cast it to a NumPy array. This should be no-copy if the array
    # is already on host memory. However, it will copy the data to host if the array was on
    # GPU memory.
    # TODO verify expected behavior if array is on GPU memory
    array = np.asarray(array)

    array_shard_end = (
        int(array.shape[axis] // jax.local_device_count()) * jax.local_device_count()
    )

    shard_slice = array_slice(array=array, axis=axis, start=0, end=array_shard_end)

    if array.shape[axis] % jax.local_device_count() > 0:
        remainder_slice = array_slice(
            array=array, axis=axis, start=array_shard_end, end=-1
        )
        return (shard_slice, remainder_slice)

    else:
        return (shard_slice,)



def build_dirichlet_arrays(
    point_indices: NPArray_1_D_int64,
    component: int,
    value: float,
) -> tuple[NPArray_2_D2_uint64, NPArray_1_D_float64]:
    """
    Builds arrays that describe dirichlet boundary conditions.

    Returns tuple of:
      Array that is (# of constrained DoFs, 2) with each row being (point index, component of solution)
      Array with shape (# of constrained DoFs,) with each row being the value
    """
    dirichlet_bcs = np.zeros(shape=(point_indices.shape[0], 2), dtype=np.uint64)
    dirichlet_bcs[-1][:, 0] = point_indices
    dirichlet_bcs[-1][:, 1] = component
    dirichlet_values = value * np.ones(
        shape=(dirichlet_bcs[-1].shape[0],), dtype=np.float64
    )
    return (dirichlet_bcs, dirichlet_values)


def build_dirichlet_arrays_from_lists(
    point_indices: list[NPArray_1_D_int64],
    components: list[int],
    values: list[float],
) -> tuple[NPArray_2_D2_uint64, NPArray_1_D_float64]:
    """
    Builds arrays that describe dirichlet boundary conditions.

    Returns tuple of:
      Array that is (# of constrained DoFs, 2) with each row being (point index, component of solution)
      Array with shape (# of constrained DoFs,) with each row being the value
    """
    dirichlet_bcs = []
    dirichlet_values = []
    for i in range(len(point_indices)):
        dirichlet_bcs.append(
            np.zeros(shape=(point_indices[i].shape[0], 2), dtype=np.uint64)
        )
        dirichlet_bcs[-1][:, 0] = point_indices[i]
        dirichlet_bcs[-1][:, 1] = components[i]
        dirichlet_values.append(
            values[i] * np.ones(shape=(dirichlet_bcs[-1].shape[0],))
        )
    return (np.vstack(dirichlet_bcs), np.concat(dirichlet_values))



def rank2_tensor_to_voigt(tensor: jnp.ndarray) -> jnp.ndarray:
    """
    Converts 2nd rank tensor to Voigt notation.

    Parameters
    ----------
    tensor   : dense 4d-array with shape (..., N_qp, N_x, N_x)

    Returns
    -------
    tensor   : dense 3d-array with shape (..., N_qp, N_eps)
    """
    if tensor.shape[-1] == 1:  # 1D
        return tensor
    elif tensor.shape[-1] == 2:  # 2D
        voigt = tensor[..., [0, 1, 0], [0, 1, 1]]
        return voigt.at[..., 2].multiply(2.0)
    elif tensor.shape[-1] == 3:  # 3D
        voigt = tensor[..., [0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]]
        return voigt.at[..., 3:].multiply(2.0)
    else:
        raise RuntimeError(
            "The tensor must be 1D, 2D or 3D to convert to Voigt notation."
        )


def rank2_voigt_to_tensor(voigt: jnp.ndarray) -> jnp.ndarray:
    """
    Converts Voigt notation array to a 2nd rank tensor.

    Parameters
    ----------
    voigt   : dense 3d-array with shape (N_e, N_qp, N_eps)

    Returns
    -------
    tensor   : dense 4d-array with shape (N_e, N_qp, N_x, N_x)
    """
    
    if voigt.shape[-1] == 1:  # 1D
        return voigt
    elif voigt.shape[-1] == 3:  # 2D
        # 0  1  2
        # xx yy xy
        return voigt[..., [0, 2, 2, 1]].reshape((*voigt.shape[:-1], 2, 2))
    elif voigt.shape[-1] == 6:  # 3D
    #elif voigt.shape[2] == 6:  # 3D
        # 0  1  2  3  4  5
        # xx yy zz yz xz xy
        return voigt[..., [0, 5, 4, 5, 1, 3, 4, 3, 2]].reshape(
            (*voigt.shape[:-1], 3, 3)
        )
    else:
        raise RuntimeError("Invalid Voigt notation size.")
