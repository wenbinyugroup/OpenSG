"""VABS 4.1 benchmark of the solid-beam (n_model 1) D route on VABS's own
example isorect: an isotropic 1 x 2 rectangle of two quad9 elements.

The fixture (fixtures/vabs_isorect/) is the section, isorect.yaml, and
VABS's numbers on the same mesh, vabs_isorect.json, with their provenance:
the Timoshenko 6x6 of the .K and, under each unit load F1, F2, F3, M1, M2,
M3, the beam-frame stresses s11, s12, s13 of the .S (VABS's 3 x 3 Gauss
points) and of the .ELE (element averages).  OpenSG homogenizes the yaml
once and recovers each load from both beam drivers of dehom_fields: the
force F (beam_force, the `1:` line of a .ff) and the strain C^-1 F, with
C the 6x6 OpenSG computes (epsilon_bar, the `0:` line).  The first-order
term of the generalized-Timoshenko recovery is driven by the
x1-derivative of the classical strains.  That derivative comes only from
the x1-derivative of the beam force, F' = [0, 0, 0, 0, F3, -F2], so on
any section F1, M1, M2 and M3 do not reach the term.  On this section,
doubly symmetric with the origin at its centroid, F2 and F3 produce no
classical strain: their stresses come from that term alone, and a
recovery without it returns no stress there and misses VABS by the whole
field.

Both codes solve the same mesh with the same element, and no OpenSG
setting is changed.  The elements are straight-sided rectangles, so VABS's
3 x 3 rule and OpenSG's own 4 x 4 rule both integrate the stiffness and
every element integral of the stress exactly, and the stress inside an
element is a biquadratic polynomial.  OpenSG's field is therefore known
exactly at VABS's points: it is the biquadratic through its 16 Gauss
values (the fit residual is round-off, checked).

On this isotropic section s22, s23 and s33 vanish in both codes: VABS
prints round-off (at most 8.1e-15 in absolute value under the unit
loads), so the fixture keeps only s11, s12 and s13, and the test checks
that OpenSG's s22, s23 and s33 are round-off too (at most 1e-12 of the
field's largest value; measured 7.6e-16).  The test therefore checks
where OpenSG keeps 11, 12 and 13 in its component order, but not the
order of the other three; that needs an anisotropic section.

The gates are VABS's print precision.  VABS prints 11 significant digits,
so half a unit in the last digit is at most 5e-11 of the largest value
printed; every tolerance against VABS is twice that, 1e-10.  Measured: the
6x6 8.8e-12 of max|K|; the element averages <= 2.4e-12 and the Gauss-point
stresses <= 2.5e-11 of the field's largest value.
"""
import json
import os
import shutil
import struct

import numpy as np
import pytest

assert struct.calcsize("P") == 8

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

from opensg_solid.sg_dehom import dehom_fields  # noqa: E402
from opensg_solid.sg_homo import plate_homo_2d  # noqa: E402

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures",
                   "vabs_isorect")
with open(os.path.join(FIX, "vabs_isorect.json")) as f:
    REF = json.load(f)
LOADS = list(REF["loads"])                  # F1 F2 F3 M1 M2 M3
# OpenSG stores the stress as [11 22 33 23 13 12] (1 = the beam axis x1);
# the fixture keeps VABS's [11 12 13]
TO_VABS = [0, 5, 4]
ZERO = [1, 2, 3]                            # 22 33 23: zero on this section
TOL = 1e-10                                 # twice VABS's print half-unit


@pytest.fixture(scope="module")
def isorect(tmp_path_factory):
    """OpenSG's homogenization of the section, run on a copy in a temporary
    folder so that nothing is written next to the fixture."""
    d = str(tmp_path_factory.mktemp("vabs_isorect"))
    path = shutil.copy(os.path.join(FIX, "isorect.yaml"), d)
    return plate_homo_2d(path, plot=False, workdir=d, recovery=True)


def test_vabs_isorect_timoshenko_6x6(isorect):
    """The Timoshenko 6x6 equals VABS's .K to its print precision
    (measured 8.8e-12 of max|K|; diagonal terms 2.6e-11 relative or
    less)."""
    C = np.asarray(isorect["C_eff"], float)
    K = np.asarray(REF["K_timoshenko"]["value"], float)
    err = np.abs(C - K).max() / np.abs(K).max()
    print("6x6: max|C - K| / max|K| = %.3e" % err)
    assert err <= TOL


def _biquadratic(xy, values, centre, half, at):
    """Fit the biquadratic polynomial to `values` (n, 3) at the points `xy`
    (n, 2) by least squares, exact when the values are biquadratic, and
    evaluate it at the points `at`.  The coordinates are scaled to the
    element, u = (x - centre) / half.

    Out: (the values at `at`, the largest fit residual)."""
    def basis(p):
        u = (np.asarray(p, float) - centre) / half
        return np.stack([u[:, 0] ** i * u[:, 1] ** j
                         for i in range(3) for j in range(3)], axis=1)
    A = basis(xy)
    coef = np.linalg.lstsq(A, values, rcond=None)[0]
    return basis(at) @ coef, np.abs(A @ coef - values).max()


@pytest.mark.parametrize("driver", ["force", "strain"])
@pytest.mark.parametrize("load", LOADS)
def test_vabs_isorect_unit_load_stress(isorect, load, driver):
    """Under each unit load, driven by the force or by the strain, OpenSG's
    beam-frame s11, s12 and s13 equal VABS's, as element averages (.ELE)
    and at VABS's 3 x 3 Gauss points (.S), and its s22, s23 and s33 are
    round-off (1e-12 or less), all relative to the field's largest value.
    The averages alone cannot see M3 (s11 is odd in x2 in each element) nor
    the components odd in x2 (s13 of F2 and M1, s12 of F3): they average to
    zero.  The Gauss points do.  Measured, Gauss points: F1 1.8e-15, F2
    7.1e-12, F3 4.4e-12, M1 2.5e-11, M2 2.3e-11, M3 8.6e-12; element
    averages 2.4e-12 (M1) or less; s22, s23 and s33 7.6e-16 (M3) or less;
    the same for both drivers."""
    r = isorect
    F = np.asarray(REF["loads"][load], float)
    if driver == "force":                             # the `1:` line of a .ff
        _, S, _ = dehom_fields(r, beam_force=F)
    else:                                             # the `0:` line, C^-1 F
        eps = np.linalg.solve(np.asarray(r["C_eff"], float), F)
        _, S, _ = dehom_fields(r, epsilon_bar=eps)
    S = np.asarray(S, float)                          # (E, Q, 6), OpenSG order
    V_gp = np.asarray(REF["stress_gauss"]["value"][load], float)
    V_el = np.asarray(REF["stress_element"]["value"][load], float)
    scale = np.abs(V_gp).max()
    # s22, s23 and s33 are zero in exact arithmetic on this section
    e_zero = np.abs(S[..., ZERO]).max() / scale
    S = S[..., TO_VABS]                               # (E, Q, 3), VABS order
    # element averages with OpenSG's own quadrature (dV_q = weight x |J|)
    dV = np.asarray(r["dehomo_data"]["dV_q"], float)
    avg = (S * dV[..., None]).sum(axis=1) / dV.sum(axis=1)[:, None]
    e_avg = np.abs(avg - V_el).max(axis=0) / scale
    # the stress at VABS's points, from each element's biquadratic field
    x_end = np.asarray(r["x_end"], float)[..., :2]     # (E, 9, 2) nodes
    xy = np.einsum("qn,end->eqd", np.asarray(r["phi_qn"], float), x_end)
    lo, hi = x_end.min(axis=1), x_end.max(axis=1)
    pts = np.asarray(REF["gauss_points"]["x2x3"], float)
    at = np.full_like(V_gp, np.nan)
    fit = 0.0
    for e in range(len(x_end)):
        inside = np.all((pts >= lo[e]) & (pts <= hi[e]), axis=1)
        assert inside.sum() == 9                # VABS's 3 x 3 points
        at[inside], res = _biquadratic(xy[e], S[e], 0.5 * (lo[e] + hi[e]),
                                       0.5 * (hi[e] - lo[e]), pts[inside])
        fit = max(fit, res)
    assert not np.isnan(at).any()
    e_pt = np.abs(at - V_gp).max(axis=0) / scale
    print("%s %s: field max %.6g; of it, element averages %.2e, Gauss points"
          " %.2e (s11 s12 s13: %s), s22 s23 s33 %.1e, fit residual %.1e"
          % (load, driver, scale, e_avg.max(), e_pt.max(),
             " ".join("%.1e" % v for v in e_pt), e_zero, fit / scale))
    assert fit <= 1e-12 * scale                 # the field is biquadratic
    assert e_zero <= 1e-12                      # round-off
    assert e_avg.max() <= TOL
    assert e_pt.max() <= TOL
