import numpy as np


def generate_boundary_markers(xmin, xmax):
    def is_left_boundary(x):
        return np.isclose(x[0], xmin)
    def is_right_boundary(x):
        return np.isclose(x[0], xmax)
    return is_left_boundary, is_right_boundary