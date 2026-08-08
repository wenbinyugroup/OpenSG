import numpy as np
import numpy.typing as npt
from typing import Any, Literal

# Nmaing convention: NPArray_<rank>_<shape>_<dtype>
# Note: D = dynamic

# Integers
NPArray_1_D_int64 = np.ndarray[tuple[int], np.dtype[np.int64]]
NPArray_2_DD_int64 = np.ndarray[tuple[int, int], np.dtype[np.int64]]
NPArray_2_D2_int64 = np.ndarray[tuple[int, Literal[2]], np.dtype[np.int64]]

NPArray_1_D_uint64 = np.ndarray[tuple[int], np.dtype[np.uint64]]
NPArray_2_DD_uint64 = np.ndarray[tuple[int, int], np.dtype[np.uint64]]
NPArray_2_D2_uint64 = np.ndarray[tuple[int, Literal[2]], np.dtype[np.uint64]]

# Floats
NPArray_1_D_float64 = np.ndarray[tuple[int], np.dtype[np.float64]]
NPArray_2_DD_float64 = np.ndarray[tuple[int, int], np.dtype[np.float64]]
NPArray_2_D2_float64 = np.ndarray[tuple[int, Literal[2]], np.dtype[np.float64]]