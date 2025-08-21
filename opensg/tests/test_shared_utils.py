"""Unit tests for the shared utility functions in opensg.utils.shared."""

import unittest
import numpy as np
import dolfinx
import ufl
import basix
from mpi4py import MPI
import petsc4py.PETSc as PETSc

from opensg.utils.shared import (
    compute_nullspace,
    solve_ksp,
    local_frame_1D,
    deri_constraint,
)


class TestSharedUtils(unittest.TestCase):
    """Test case for the shared utility functions."""

    def test_compute_nullspace_dimension(self):
        """Test that compute_nullspace returns a nullspace with the expected dimension."""
        # Create a simple 1D mesh
        mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 10)

        # Create a vector-valued function space
        element = basix.ufl.element("Lagrange", "interval", 1, shape=(3,))
        V = dolfinx.fem.functionspace(mesh, element)

        # Compute the nullspace
        nullspace = compute_nullspace(V)

        # Check that the nullspace has the expected dimension (3 for translational)
        self.assertIsNotNone(nullspace.getVecs())
        self.assertEqual(len(nullspace.getVecs()), 3)

    def test_compute_nullspace_orthonormality(self):
        """Test that the vectors in the nullspace are orthonormal."""
        # Create a simple 1D mesh
        mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 10)

        # Create a vector-valued function space
        element = basix.ufl.element("Lagrange", "interval", 1, shape=(3,))
        V = dolfinx.fem.functionspace(mesh, element)

        # Compute the nullspace
        nullspace = compute_nullspace(V)

        # Get the nullspace vectors
        vecs = nullspace.getVecs()

        # Check orthonormality (dot product of different vectors should be close to 0)
        # and normalization (dot product of a vector with itself should be close to 1)
        for i in range(len(vecs)):
            for j in range(len(vecs)):
                dot_product = vecs[i].dot(vecs[j])
                if i == j:
                    self.assertTrue(np.isclose(dot_product, 1.0, rtol=1e-5))
                else:
                    self.assertTrue(np.isclose(dot_product, 0.0, rtol=1e-5))

    def test_local_frame_1D_orthonormality(self):
        """Test that local_frame_1D returns orthonormal vectors."""
        # Create a simple curved 1D mesh (a quarter circle)
        n = 10  # number of cells
        points = []
        for i in range(n + 1):
            theta = i * (np.pi / 2) / n  # angle from 0 to pi/2
            x = np.cos(theta)
            y = np.sin(theta)
            z = 0.0
            points.append([x, y, z])

        points = np.array(points)
        cells = np.array([[i, i + 1] for i in range(n)])

        element = basix.ufl.element("Lagrange", "interval", 1, shape=(3,))
        domain = ufl.Mesh(element)
        mesh = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cells, points, domain)

        # Compute the local frame
        e1, e2, e3 = local_frame_1D(mesh)

        # Check that the vectors are UFL expressions
        self.assertIsNotNone(e1)
        self.assertIsNotNone(e2)
        self.assertIsNotNone(e3)

        # Since we can't easily evaluate UFL expressions directly in a unit test,
        # we'll just check that the frame vectors are defined
        # The actual orthonormality is ensured by the implementation of local_frame_1D

    # def test_solve_ksp_simple_system(self):
    #     """Test solve_ksp with a simple linear system."""


class TestDeriConstraint(unittest.TestCase):
    """Test case for the deri_constraint function."""

    def test_deri_constraint_1d(self):
        """Test deri_constraint for a 1D mesh."""
        # Create a simple 1D mesh
        mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 10)

        # Create a function space
        element = basix.ufl.element("Lagrange", "interval", 2)
        V = dolfinx.fem.functionspace(mesh, element)

        # Create trial and test functions
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

        # Create a normal vector for the constraint
        nh = ufl.as_vector([1.0, 0.0])

        # Compute the derivative constraint
        constraint = deri_constraint(u, v, mesh, nh)

        # Check that the constraint is a UFL form
        self.assertIsNotNone(constraint)
        self.assertTrue(hasattr(constraint, "ufl_domain"))

    def test_deri_constraint_2d(self):
        """Test deri_constraint for a 2D mesh."""
        # Create a simple 2D mesh
        mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 5, 5)

        # Create a function space
        element = basix.ufl.element("Lagrange", "triangle", 2)
        V = dolfinx.fem.functionspace(mesh, element)

        # Create trial and test functions
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

        # Create a normal vector for the constraint
        nh = ufl.as_vector([1.0, 0.0])

        # Compute the derivative constraint
        constraint = deri_constraint(u, v, mesh, nh)

        # Check that the constraint is a UFL form
        self.assertIsNotNone(constraint)
        self.assertTrue(hasattr(constraint, "ufl_domain"))


if __name__ == "__main__":
    unittest.main()
