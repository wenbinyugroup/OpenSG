"""Tests for the dehom core (rm_plate.rm_dehom): the meshfree
MLS patch-gradient estimator that generalizes the lattice FD to arbitrary
station clouds.

Covered
  quadratic exactness   a random scattered cloud carrying fields quadratic
                        in (x, y): every first AND second derivative must
                        come back machine-exact (the fit reproduces any
                        quadratic regardless of the weights)
  lattice consistency   on a Gauss-point lattice with a quadratic field the
                        meshfree estimator and np.gradient FD agree at the
                        interior stations (both exact there)
  guard                 < 6 stations raises (classical recovery territory)

Run:  pytest tests/test_rm_dehom.py
"""
import os
import sys

import numpy as np
import pytest

CC = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))  # repo root
sys.path.insert(0, os.path.join(CC, "src"))

from opensg_solid.rm_plate_1D.rm_dehom import mls_gradients


def _quadratic_fields(x, y):
    """Two independent quadratics + their exact derivative tuples."""
    F = np.stack([1.0 + 2.0 * x - 3.0 * y + 0.7 * x * x - 1.1 * x * y
                  + 0.4 * y * y,
                  -0.5 + 0.3 * x + 1.9 * y - 0.2 * x * x + 0.8 * x * y
                  - 1.3 * y * y], axis=1)
    dFx = np.stack([2.0 + 1.4 * x - 1.1 * y, 0.3 - 0.4 * x + 0.8 * y], axis=1)
    dFy = np.stack([-3.0 - 1.1 * x + 0.8 * y, 1.9 + 0.8 * x - 2.6 * y], axis=1)
    dxx = np.stack([1.4 + 0 * x, -0.4 + 0 * x], axis=1)
    dxy = np.stack([-1.1 + 0 * x, 0.8 + 0 * x], axis=1)
    dyy = np.stack([0.8 + 0 * x, -2.6 + 0 * x], axis=1)
    return F, (dFx, dFy, dxx, dxy, dyy)


def test_mls_quadratic_exact_on_random_cloud():
    rng = np.random.default_rng(7)
    x = rng.uniform(0.0, 2.0, 300)
    y = rng.uniform(0.0, 1.0, 300)
    F, exact = _quadratic_fields(x, y)
    got = mls_gradients(x, y, F, k=12)
    for g, e, nm in zip(got, exact, ("dFx", "dFy", "dxx", "dxy", "dyy")):
        rel = np.max(np.abs(g - e)) / max(np.max(np.abs(e)), 1e-30)
        assert rel < 1e-8, "%s: max rel %.2e" % (nm, rel)


def test_mls_matches_lattice_fd_interior():
    """On the 2x2-Gauss lattice of a 6x6-element plate both estimators are
    exact for a quadratic -> they must agree at the interior stations."""
    gp = 1.0 / (2.0 * np.sqrt(3.0))
    cols = np.concatenate([[e + 0.5 - gp, e + 0.5 + gp] for e in range(6)])
    X, Y = np.meshgrid(cols, cols)
    x, y = X.reshape(-1), Y.reshape(-1)
    F, _ = _quadratic_fields(x, y)
    gx, gy, *_ = mls_gradients(x, y, F, k=12)
    Fg = F.reshape(12, 12, 2)
    fdx = np.gradient(Fg, cols, axis=1)
    fdy = np.gradient(Fg, cols, axis=0)
    interior = ((x > cols[1]) & (x < cols[-2])
                & (y > cols[1]) & (y < cols[-2]))
    for a, b, nm in ((gx, fdx.reshape(-1, 2), "dFx"),
                     (gy, fdy.reshape(-1, 2), "dFy")):
        rel = (np.max(np.abs(a[interior] - b[interior]))
               / max(np.max(np.abs(b[interior])), 1e-30))
        assert rel < 1e-8, "%s: MLS vs lattice FD interior rel %.2e" % (nm, rel)


def test_mls_needs_six_stations():
    x = np.arange(4.0)
    with pytest.raises(ValueError):
        mls_gradients(x, x, np.ones((4, 1)))


if __name__ == "__main__":
    test_mls_quadratic_exact_on_random_cloud()
    test_mls_matches_lattice_fd_interior()
    test_mls_needs_six_stations()
    print("all tests passed")
