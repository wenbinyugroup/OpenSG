"""statics_fsdt.py -- STATICS-only resultants and the standalone FSDT chain with the
Whitney-1973 shear correction.  NO Pagano anywhere in this module: everything below
follows from equilibrium of the problem statement and ply stiffnesses alone.

===============================================================================
1.  THE STATICS OF Q1 AND M11 (complete derivation)
===============================================================================
Free body of the strip 0 <= x <= a under q(x) = q0 sin(p x), p = pi/a, unit width,
simply supported.  Sectional equilibrium of a slice [x, x+dx]:

    vertical:   dQ1/dx = -q(x)                                              (i)
    moment:     dM11/dx = Q1(x)                                             (ii)

Integrate (i):    Q1(x) = (q0/p) cos(p x) + C1
Integrate (ii):   M11(x) = (q0/p^2) sin(p x) + C1 x + C2

Boundary conditions of the simple supports: M11(0) = M11(a) = 0.
  M11(0) = 0  ->  C2 = 0
  M11(a) = 0  ->  C1 a = 0  ->  C1 = 0        (sin(pi) = 0)

    Q1(x)  = (q0/p)  cos(p x)      ->  at x = 0 :   Q1 = q0/p
    M11(x) = (q0/p^2) sin(p x)     ->  at x = a/2:  M11 = q0/p^2

The end reactions are Q1(0) = -Q1(a) = q0/p: the shear resultant at the support IS
the support reaction.  Statically determinate -- no material property entered.

===============================================================================
2.  THE WHITNEY (JAM 40 (1973) 302-304) k1^2, HIS EQS. (3)-(7) IN FULL
===============================================================================
Under CYLINDRICAL BENDING with constant shear Q1, classical lamination gives the
in-plane stress gradient in each ply m (his Eq. (3); Qb11 = the plane-stress reduced
stiffness of the ply, rotated):

    d(sigma_11)/dx (z) = Qb11(z) (B11 - A11 z) Q1 / D                       (3)
    (A11, B11, D11) = int Qb11(z) (1, z, z^2) dz ,   D = A11 D11 - B11^2

3-D equilibrium sigma_11,1 + sigma_13,3 = 0 (his Eq. (4)) integrated from the free
bottom face gives the EQUILIBRIUM SHEAR SHAPE (his Eq. (5); the interface constants
a_m arise automatically from the cumulative integral):

    sigma_13(z) = ghat(z) Q1,      ghat(z) = (1/D) int_{-h/2}^{z} Qb11 (B11 - A11 z') dz'

ghat(-h/2) = 0 by construction and ghat(+h/2) = (A11 B11 - B11 A11)/D = 0 -- both
faces traction-free automatically.  Its resultant: int ghat dz = 1 (carries Q1).

His Eq. (7): equate the FSDT shear energy to the complementary energy of that shape,

    Q1^2 / (2 k1^2 A55)  =  (1/2) int  [ghat(z) Q1]^2 / C55(z)  dz
    ------------------------------------------------------------------
    k1^2  =  1 / ( A55 * int ghat(z)^2 / C55(z) dz ),   A55 = int C55(z) dz

Homogeneous section -> ghat is the parabola 3/(2h)(1-(2z/h)^2) -> k1^2 = 5/6
(Reissner), exactly as Whitney states.  His 4-ply cross-ply example (same 25:1
material ratios) gives k1^2 = 0.5952 -- same regime as the values printed here.

===============================================================================
3.  THE STANDALONE FSDT CHAIN (what the k is then used for)
===============================================================================
    gamma = Q1 / (k1^2 A55)                     (Whitney Eq. (1), constitutive law)
    sigma_13(z) = C55(z) gamma                  (layerwise-constant staircase)
    sigma_33(z) = 0  identically                (plane-stress plies -- FSDT has none)
    w(a/2) = q0/(p^4 D11) + q0/(p^2 k1^2 A55)   (what the k actually improves)

Whitney uses the good equilibrium shape ghat ONLY to calibrate the scalar k -- the
shape is then discarded, which is why no k can repair the staircase profile.
"""
import os
import sys

import numpy as np

# repo root = the first ancestor holding opensg_jax/ (location-robust: this
# module also runs from the examples/RM_OpenSG_pagano copy)
_CC = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_CC, "src", "opensg_solid", "rm_plate_1D")):
    _up = os.path.dirname(_CC)
    if _up == _CC:
        raise RuntimeError("rm_plate repo root not found above " + __file__)
    _CC = _up
sys.path.insert(0, os.path.join(_CC, "src"))
from opensg_solid.rm_plate_1D.msg_materials import rotated_stiffness_6x6     # noqa: E402


def statics_resultants(q0, a, x):
    """Q1(x) and M11(x) of section 1 above (statics only, no material property).

    Variables: q0 = load amplitude of q(x) = q0 sin(px) [Pa]; a = span [m];
    x = station [m]; p = pi/a the wavenumber [1/m].  Returns
    (Q1, M11) = ((q0/p) cos(px), (q0/p^2) sin(px)) [N/m, N].
    """
    p = np.pi / a
    return (q0 / p) * np.cos(p * x), (q0 / p ** 2) * np.sin(p * x)


def _ply_props(thick, angles_deg, mat_names, material_db, npts=200):
    """Fine z-grid with the plane-stress reduced Qb11(z) and C55(z).

    Variables
    ---------
    thick, angles_deg, mat_names, material_db
                laminate definition (ply thicknesses [m], fibre angles [deg],
                material names, and the {name: {E, G, nu}} database)
    npts        sample points per ply on the returned grid
    keep, drop  Voigt index sets of the plane-stress condensation: keep the
                in-plane rows [11, 22, 12] = (0, 1, 5), condense out the
                transverse rows [33, 23, 13] = (2, 3, 4)
    bot         (nply+1,) ply interface heights measured from the MID-surface [m]
    C           (6, 6) rotated ply stiffness (OpenSG Voigt order)
    Q3          (3, 3) plane-stress reduced stiffness
                Q = C_kk - C_kd C_dd^{-1} C_dk; its [0, 0] entry is Qb11
    zz          the per-ply z-samples; z_all/Qb11/C55 the concatenated grids

    Returns (z, Qb11(z), C55(z)) as flat arrays of length nply * npts.
    """
    keep = np.array([0, 1, 5]); drop = np.array([2, 3, 4])
    z_all, Qb11, C55 = [], [], []
    bot = np.concatenate([[0.0], np.cumsum(thick)]) - 0.5 * float(np.sum(thick))
    for k, (t, ang, m) in enumerate(zip(thick, angles_deg, mat_names)):
        C = np.asarray(rotated_stiffness_6x6(material_db[m]["E"], material_db[m]["G"],
                                             material_db[m]["nu"], ang))
        Q3 = (C[np.ix_(keep, keep)]
              - C[np.ix_(keep, drop)] @ np.linalg.solve(C[np.ix_(drop, drop)],
                                                        C[np.ix_(drop, keep)]))
        zz = np.linspace(bot[k], bot[k + 1], npts)
        z_all.append(zz)
        Qb11.append(np.full(npts, Q3[0, 0]))
        C55.append(np.full(npts, C[4, 4]))
    return np.concatenate(z_all), np.concatenate(Qb11), np.concatenate(C55)


def whitney_k1sq(thick, angles_deg, mat_names, material_db):
    """Whitney Eq. (7), computed exactly as derived in the module docstring.

    Variables
    ---------
    z, Qb11, C55    the fine grid from _ply_props (z from the mid-surface [m])
    bot             ply interface heights from the mid-surface [m]
    keepv, dropv    the same condensation index sets as in _ply_props
    C, Q3, z1, z2   per-ply rotated stiffness, its plane-stress reduction, and the
                    ply's bottom/top heights
    A11, B11, D11   laminate moments int Qb11 (1, z, z^2) dz, accumulated with the
                    EXACT closed-form per-ply integrals (Qb11 is ply-constant), not
                    quadrature -- trapezoid on z^2 is not exact
    D               the bending-gradient denominator A11 D11 - B11^2 of Eq. (3)
    integrand       the Eq.-(3) stress gradient shape Qb11 (B11 - A11 z)/D; its
                    SIGN is the one that makes int ghat dz = +1 (the flipped sign
                    was a real bug: it integrated to -1)
    ghat            the Eq.-(5) equilibrium shear shape: cumulative trapezoid of
                    integrand from the free bottom face; checked ghat(top) = 0
                    (machine) and int ghat dz = 1 (1e-4: trapezoid on a piecewise
                    quadratic at npts = 200 per ply)
    A55             int C55 dz [N/m]
    k1sq            Eq. (7): 1 / (A55 int ghat^2 / C55 dz); homogeneous -> 5/6

    Returns (k1sq, A55, ghat_fn) where ghat_fn(z_query) evaluates the equilibrium
    shear shape (resultant 1) -- exposed so callers can inspect the Eq.-(5) shape.
    """
    z, Qb11, C55 = _ply_props(thick, angles_deg, mat_names, material_db)
    # exact per-ply moments (Qb11 constant in each ply): A = sum Qb t,
    # B = sum Qb (z2^2 - z1^2)/2, D = sum Qb (z2^3 - z1^3)/3
    bot = np.concatenate([[0.0], np.cumsum(thick)]) - 0.5 * float(np.sum(thick))
    keepv = np.array([0, 1, 5]); dropv = np.array([2, 3, 4])
    A11 = B11 = D11 = 0.0
    for k, (t, ang, m) in enumerate(zip(thick, angles_deg, mat_names)):
        C = np.asarray(rotated_stiffness_6x6(material_db[m]["E"], material_db[m]["G"],
                                             material_db[m]["nu"], ang))
        Q3 = (C[np.ix_(keepv, keepv)]
              - C[np.ix_(keepv, dropv)] @ np.linalg.solve(C[np.ix_(dropv, dropv)],
                                                          C[np.ix_(dropv, keepv)]))
        z1, z2 = bot[k], bot[k + 1]
        A11 += Q3[0, 0] * (z2 - z1)
        B11 += Q3[0, 0] * (z2 ** 2 - z1 ** 2) / 2.0
        D11 += Q3[0, 0] * (z2 ** 3 - z1 ** 3) / 3.0
    D = A11 * D11 - B11 ** 2
    # his Eq. (5): cumulative integral of the Eq.-(3) gradient, zero at the bottom face
    integrand = Qb11 * (B11 - A11 * z) / D
    ghat = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1])
                                            * np.diff(z))])
    # checks the derivation promises: free top face, unit resultant
    # (quadrature tolerance: ghat is exact per node -- linear integrand -- but
    #  int ghat dz is trapezoid on a piecewise quadratic, error ~ (1/npts)^2)
    assert abs(ghat[-1]) < 1e-9 * np.max(np.abs(ghat))
    assert abs(np.trapezoid(ghat, z) - 1.0) < 1e-4
    A55 = float(np.trapezoid(C55, z))
    k1sq = 1.0 / (A55 * float(np.trapezoid(ghat ** 2 / C55, z)))
    return k1sq, A55, lambda zq: np.interp(zq, z, ghat)


def fsdt_s13(zc, thick, angles_deg, mat_names, material_db, Q1):
    """The standalone FSDT transverse shear (section 3): the constitutive staircase
    C55(z) Q1 / (k1^2 A55).  FSDT sigma_33 is identically zero and is not returned."""
    k1sq, A55, _ = whitney_k1sq(thick, angles_deg, mat_names, material_db)
    bot = np.concatenate([[0.0], np.cumsum(thick)]) - 0.5 * float(np.sum(thick))
    ply = np.clip(np.searchsorted(bot[1:-1], zc, side="left"), 0, len(thick) - 1)
    C55 = np.array([float(rotated_stiffness_6x6(material_db[m]["E"], material_db[m]["G"],
                                                material_db[m]["nu"], x)[4, 4])
                    for m, x in zip(mat_names, angles_deg)])
    return C55[ply] * Q1 / (k1sq * A55), k1sq, A55


if __name__ == "__main__":
    # self-checks: homogeneous -> exactly 5/6; cross-ply k in Whitney's reported regime
    iso = {"iso": {"E": [70e9] * 3, "G": [26.9e9] * 3, "nu": [0.3] * 3, "rho": 1.0}}
    k, _, _ = whitney_k1sq([0.01], [0.0], ["iso"], iso)
    print("homogeneous k1^2 = %.6f  (Reissner 5/6 = %.6f)" % (k, 5 / 6))
    pag = {"pagano": {"E": [25e9, 1e9, 1e9], "G": [0.5e9, 0.5e9, 0.2e9],
                      "nu": [0.25] * 3, "rho": 1.0}}
    k, A55, _ = whitney_k1sq([1 / 30] * 3, [0.0, 90.0, 0.0], ["pagano"] * 3, pag)
    print("[0/90/0] Pagano-material k1^2 = %.6f  (Whitney's 4-ply example: 0.5952)" % k)
