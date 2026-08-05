"""segment_indep.py -- EXPERIMENTAL: omega_3 as an INDEPENDENT 6th DOF.

The general operator (segment_element_general) ELIMINATES the drilling omega_3 via
  omega_3 = S/(2 C33) - (C3b/C33) omega_b ,   C33 = n.b3 ,
which is singular on flat walls where C33=n.b3 == 0 (the GA3-carrier walls / shear web).
Here omega_3 is kept as a genuine nodal DOF (per node [w1,w2,w3,om1,om2,om3]) and appears
DIRECTLY -- no 1/C33 anywhere:

  curvature:  Lambda_a = omega_3|a + x_{1;a} omega_3'          (was the eliminated Lambda)
  shear:      2g13 += C23 om3 = x_{3;2} om3 ,  2g23 += -C13 om3 = -x_{3;1} om3
  membrane:   unchanged (no omega_3)

The in-plane symmetry that DEFINED omega_3 is re-imposed in its FINITE (undivided) form
as a penalty on the drilling residual
  DR = C33 om3 + C3b om_b - S/2   (= C33 (om3 - om3_eliminated), finite even at C33=0),
  S/2 = 1/2[ k1(x11 Rn2 - x12 Rn1) + w'_i(x11 x_{i;2}-x12 x_{i;1}) + (w_{i|1}x_{i;2}-w_{i|2}x_{i;1}) ].
On healthy walls (C33~1) a large penalty pins om3 to its eliminated value (recovering the
general result); where C33=0 the residual drops om3 (constrains om_b instead) and om3 is set
by its own curvature stiffness -- no singularity.  PEN sweeps the penalty weight.

Boundary: reuse the validated 5-DOF rings (ring_general) for dofs [0..4]; om3 (dof 5) is left
FREE at the boundary (natural BC).
"""
import numpy as np
from .segment_element import _bilinear, dirichlet_solve
from .segment_element_general import _surf_frame

NDOF6 = 6


def quad_ops_indep(X, e3m, xi, eta, k22, cross, ax, kg=0.0):
    """8-strain 6-DOF operators + drilling-residual DR at (xi,eta).
    Returns BDe(6,4),BDh(6,24),BDl(6,24), BGe(2,4),BGh(2,24),BGl(2,24),
            DRe(4),DRh(24),DRl(24), dA."""
    N, D1, D2, dA, c = _surf_frame(X, e3m, xi, eta, cross, ax)
    x11, x12 = c["x11"], c["x12"]
    xi1 = np.array([x11, c["x21"], c["x31"]])            # x_{i;1}
    xi2 = np.array([x12, c["x22"], c["x32"]])            # x_{i;2}
    yv = np.array([c["y1"], c["y2"], c["y3"]])           # C_{3i}
    x2, x3 = c["x2"], c["x3"]
    x21, x31, x22, x32 = c["x21"], c["x31"], c["x22"], c["x32"]
    y1, y2, y3 = c["y1"], c["y2"], c["y3"]
    Rn1 = x2 * x31 - x3 * x21
    Rn2 = x2 * x32 - x3 * x22
    n = NDOF6

    BDe = np.zeros((6, 4)); BDh = np.zeros((6, 4 * n)); BDl = np.zeros((6, 4 * n))
    BGe = np.zeros((2, 4)); BGh = np.zeros((2, 4 * n)); BGl = np.zeros((2, 4 * n))
    DRe = np.zeros(4); DRh = np.zeros(4 * n); DRl = np.zeros(4 * n)

    # ---------------- MEMBRANE (unchanged; no omega_3) ----------------
    BDe[0] = [x11**2, x11 * Rn1, x11**2 * x3, -(x11**2) * x2]
    BDe[1] = [x12**2, x12 * Rn2, x12**2 * x3, -(x12**2) * x2]
    BDe[2] = [2 * x11 * x12, x11 * Rn2 + x12 * Rn1, 2 * x11 * x12 * x3, -2 * x11 * x12 * x2]
    for a in range(4):
        o = n * a
        for i in range(3):
            BDh[0, o + i] += xi1[i] * D1[a]
            BDl[0, o + i] += x11 * xi1[i] * N[a]
            BDh[1, o + i] += xi2[i] * D2[a]
            BDl[1, o + i] += x12 * xi2[i] * N[a]
            BDh[2, o + i] += xi1[i] * D2[a] + xi2[i] * D1[a]
            BDl[2, o + i] += (x11 * xi2[i] + x12 * xi1[i]) * N[a]

    # ---------------- CURVATURE (non-Lambda parts as general; Lambda = DIRECT om3) --------
    BDe[3] = [0.0, x11 * x12, x11 * c["x22"], x11 * c["x32"]]
    BDe[4] = [0.0, -x12 * x11, -x12 * x21, -x12 * x31]
    BDe[5] = [0.0, x12 * x12 - x11 * x11, x12 * x22 - x11 * x21, x12 * x32 - x11 * x31]
    for a in range(4):
        o = n * a
        # k11 : x_{b;2} x11 om'_b + x_{b;2} om_{b|1}
        BDl[3, o + 3] += x11 * x12 * N[a]; BDl[3, o + 4] += x11 * x22 * N[a]
        BDh[3, o + 3] += x12 * D1[a];      BDh[3, o + 4] += x22 * D1[a]
        # k22 : -(x_{b;1} x12 om'_b + x_{b;1} om_{b|2})
        BDl[4, o + 3] += -x12 * x11 * N[a]; BDl[4, o + 4] += -x12 * x21 * N[a]
        BDh[4, o + 3] += -x11 * D2[a];      BDh[4, o + 4] += -x21 * D2[a]
        # k12 : om'_b(x12 x_{b;2}-x11 x_{b;1}) + (x_{b;2}om_{b|2}-x_{b;1}om_{b|1})
        BDl[5, o + 3] += (x12 * x12 - x11 * x11) * N[a]; BDl[5, o + 4] += (x12 * x22 - x11 * x21) * N[a]
        BDh[5, o + 3] += x12 * D2[a] - x11 * D1[a];      BDh[5, o + 4] += x22 * D2[a] - x21 * D1[a]
        # Lambda contributions via DIRECT om3 (dof 5):  L_a = om3|a + x_{1;a} om3'
        #   k11 += x32 L1 ; k22 += -x31 L2 ; k12 += x32 L2 - x31 L1
        BDh[3, o + 5] += x32 * D1[a];               BDl[3, o + 5] += x32 * x11 * N[a]
        BDh[4, o + 5] += -x31 * D2[a];              BDl[4, o + 5] += -x31 * x12 * N[a]
        BDh[5, o + 5] += x32 * D2[a] - x31 * D1[a]; BDl[5, o + 5] += (x32 * x12 - x31 * x11) * N[a]

    # ---------------- TRANSVERSE SHEAR (pre-elimination: NO 1/C33; DIRECT om3) ------------
    swept = x2 * y3 - x3 * y2                            # k1 coeff without 1/C33 (h33=0)
    BGe[0] = np.array([x11 * y1, x11 * swept, x11 * y1 * x3, -x11 * y1 * x2])
    BGe[1] = np.array([x12 * y1, x12 * swept, x12 * y1 * x3, -x12 * y1 * x2])
    for a in range(4):
        o = n * a
        for i in range(3):
            BGh[0, o + i] += yv[i] * D1[a]               # C3i w_{i|1}
            BGh[1, o + i] += yv[i] * D2[a]               # C3i w_{i|2}
            BGl[0, o + i] += x11 * yv[i] * N[a]          # chain rule
            BGl[1, o + i] += x12 * yv[i] * N[a]
        BGh[0, o + 3] += x12 * N[a]; BGh[0, o + 4] += x22 * N[a]    # C_2a om_a
        BGh[1, o + 3] += -x11 * N[a]; BGh[1, o + 4] += -x21 * N[a]  # -C_1a om_a
        BGh[0, o + 5] += x32 * N[a]                     # +C23 om3
        BGh[1, o + 5] += -x31 * N[a]                    # -C13 om3

    # ---------------- DRILLING RESIDUAL DR = C33 om3 + C3b om_b - S/2 (finite) -------------
    # twist column: -(x11 Rn2 - x12 Rn1)/2 == +(x2 y2 + x3 y3)/2 by the direction-cosine
    # identity y2 = x12 x31 - x11 x32, y3 = x11 x22 - x12 x21 (Omega3_new simplification;
    # verified to 1e-16 in verify_indep_shear).  Kept in Rn form to match eq. A-derivation.
    DRe[1] = -0.5 * (x11 * Rn2 - x12 * Rn1)
    for a in range(4):
        o = n * a
        for i in range(3):
            DRh[o + i] += -0.5 * (xi2[i] * D1[a] - xi1[i] * D2[a])          # -1/2 (w_{i|1}x_{i;2}-w_{i|2}x_{i;1})
            DRl[o + i] += -0.5 * (x11 * xi2[i] - x12 * xi1[i]) * N[a]       # -1/2 w'_i(x11 x_{i;2}-x12 x_{i;1})
        DRh[o + 3] += y1 * N[a]; DRh[o + 4] += y2 * N[a]; DRh[o + 5] += y3 * N[a]  # C3b om_b + C33 om3
    return BDe, BDh, BDl, BGe, BGh, BGl, DRe, DRh, DRl, dA


# standard Dvorkin-Bathe MITC4 tying points (same as segment_element_general):
#   gamma_13 (row 0): sampled at (-1,0),(+1,0), linear in xi
#   gamma_23 (row 1): sampled at (0,-1),(0,+1), linear in eta
_TIE_G13 = [(-1.0, 0.0), (1.0, 0.0)]
_TIE_G23 = [(0.0, -1.0), (0.0, 1.0)]


def _mitc_shear_indep(X, e3m, xi, eta, k22, cross, ax, kg=0.0, scheme="mitc4_g23"):
    """Tied (assumed-strain) transverse-shear BGh rows (2 x 4*NDOF6) at (xi,eta).

    With the INDEPENDENT drilling omega_3 the gamma_13 row is ALGEBRAIC in omega_3 on
    flat walls (x_{3;2} omega_3 -- the role omega_2 plays prismatically), so tying it
    removes the director penalization (hourglass; GA3 -29/-47% on the thin square).
    Schemes:
      'mitc4_wonly' : FIELD-CONSISTENT partial tying -- tie BOTH rows at the standard
                      Dvorkin-Bathe points but keep every ROTATION column (om1,om2,om3)
                      at its full-integration (Gauss-point) value.  Removes the
                      w-gradient/rotation order mismatch (the locking pairing) without
                      de-penalizing the algebraic director content.  The general-case
                      safeguard.
      'mitc4_g23'   : tie only the gamma_23 row (prismatic-element analogue).
      'mitc4_both'  : naive full tying (ablation only -- aliases the drilling shear).
    """
    ops = quad_ops_indep(X, e3m, xi, eta, k22, cross, ax, kg)
    r23 = [quad_ops_indep(X, e3m, tx, te, k22, cross, ax, kg)[4][1:2, :] for (tx, te) in _TIE_G23]
    g23 = 0.5 * (1.0 - eta) * r23[0] + 0.5 * (1.0 + eta) * r23[1]
    if scheme in ("mitc4_both", "mitc4_wonly"):
        r13 = [quad_ops_indep(X, e3m, tx, te, k22, cross, ax, kg)[4][0:1, :] for (tx, te) in _TIE_G13]
        g13 = 0.5 * (1.0 - xi) * r13[0] + 0.5 * (1.0 + xi) * r13[1]
    else:
        g13 = ops[4][0:1, :]
    BGt = np.vstack([g13, g23])
    if scheme == "mitc4_wonly":
        rot = np.zeros(BGt.shape[1], bool)
        for a in range(4):
            rot[NDOF6 * a + 3:NDOF6 * a + 6] = True
        BGt[:, rot] = ops[4][:, rot]          # rotations stay fully integrated
    return BGt


# ---------------- BATCHED operator evaluation (vectorized over elements) ----------------
# The scalar quad_ops_indep above is kept for external diagnostics/verification;
# the assembly below evaluates the same closed-form operators for ALL elements at
# once per quadrature point -- the per-element Python loop dominated the shell
# wall time (5760 quad_ops_indep calls ~ 5 s of the 7 s square-thin solve).


def _surf_frame_batch(Xe, e3e, xi, eta, cross, ax):
    """Batched _surf_frame: Xe (ne,4,3), e3e (ne,3) -> N (4,), D1/D2 (ne,4),
    dA (ne,), direction-cosine dict of (ne,) arrays.  Same algebra per element."""
    N, dNx, dNe = _bilinear(xi, eta)
    Jxi = np.einsum('a,eaj->ej', dNx, Xe)
    Jeta = np.einsum('a,eaj->ej', dNe, Xe)
    a2 = Jxi / np.linalg.norm(Jxi, axis=1, keepdims=True)
    a1 = Jeta - np.sum(Jeta * a2, axis=1, keepdims=True) * a2
    a1 = a1 / np.linalg.norm(a1, axis=1, keepdims=True)
    n = np.cross(a1, a2)
    flip = np.sum(n * e3e, axis=1) < 0.0
    n[flip] = -n[flip]; a1[flip] = -a1[flip]
    G11 = np.sum(a1 * Jxi, axis=1); G12 = np.sum(a1 * Jeta, axis=1)
    G21 = np.sum(a2 * Jxi, axis=1); G22 = np.sum(a2 * Jeta, axis=1)
    # [D1; D2] = inv(G^T) [dNx; dNe] with G = [[G11,G12],[G21,G22]]
    det = G11 * G22 - G12 * G21
    D1 = (G22[:, None] * dNx[None, :] - G21[:, None] * dNe[None, :]) / det[:, None]
    D2 = (-G12[:, None] * dNx[None, :] + G11[:, None] * dNe[None, :]) / det[:, None]
    dA = np.linalg.norm(np.cross(Jxi, Jeta), axis=1)
    x = np.einsum('a,eaj->ej', N, Xe)
    c = dict(
        x11=a1[:, ax], x21=a1[:, cross[0]], x31=a1[:, cross[1]],
        x12=a2[:, ax], x22=a2[:, cross[0]], x32=a2[:, cross[1]],
        y1=n[:, ax], y2=n[:, cross[0]], y3=n[:, cross[1]],
        x2=x[:, cross[0]], x3=x[:, cross[1]],
    )
    return N, D1, D2, dA, c


def quad_ops_indep_batch(Xe, e3e, xi, eta, cross, ax):
    """Batched quad_ops_indep (same rows; k22/kg do not enter these operators).
    Returns BDe (ne,6,4), BDh/BDl (ne,6,24), BGe (ne,2,4), BGh/BGl (ne,2,24),
    DRe (ne,4), DRh/DRl (ne,24), dA (ne,)."""
    N, D1, D2, dA, c = _surf_frame_batch(Xe, e3e, xi, eta, cross, ax)
    ne = Xe.shape[0]
    x11, x12 = c["x11"], c["x12"]
    xi1 = np.stack([x11, c["x21"], c["x31"]], axis=1)     # (ne,3) x_{i;1}
    xi2 = np.stack([x12, c["x22"], c["x32"]], axis=1)     # (ne,3) x_{i;2}
    yv = np.stack([c["y1"], c["y2"], c["y3"]], axis=1)    # (ne,3) C_{3i}
    x2, x3, y1 = c["x2"], c["x3"], c["y1"]
    Rn1 = x2 * c["x31"] - x3 * c["x21"]
    Rn2 = x2 * c["x32"] - x3 * c["x22"]
    swept = x2 * c["y3"] - x3 * c["y2"]
    z = np.zeros(ne)

    BDe = np.stack([
        np.stack([x11**2, x11 * Rn1, x11**2 * x3, -(x11**2) * x2], 1),
        np.stack([x12**2, x12 * Rn2, x12**2 * x3, -(x12**2) * x2], 1),
        np.stack([2 * x11 * x12, x11 * Rn2 + x12 * Rn1,
                  2 * x11 * x12 * x3, -2 * x11 * x12 * x2], 1),
        np.stack([z, x11 * x12, x11 * c["x22"], x11 * c["x32"]], 1),
        np.stack([z, -x12 * x11, -x12 * c["x21"], -x12 * c["x31"]], 1),
        np.stack([z, x12**2 - x11**2, x12 * c["x22"] - x11 * c["x21"],
                  x12 * c["x32"] - x11 * c["x31"]], 1),
    ], axis=1)
    BGe = np.stack([
        np.stack([x11 * y1, x11 * swept, x11 * y1 * x3, -x11 * y1 * x2], 1),
        np.stack([x12 * y1, x12 * swept, x12 * y1 * x3, -x12 * y1 * x2], 1),
    ], axis=1)

    # DOF blocks laid out (ne, row, node a, dof) -> reshape (ne, row, 24)
    B = np.zeros((ne, 6, 4, NDOF6))
    B[:, 0, :, 0:3] = D1[:, :, None] * xi1[:, None, :]
    B[:, 1, :, 0:3] = D2[:, :, None] * xi2[:, None, :]
    B[:, 2, :, 0:3] = D2[:, :, None] * xi1[:, None, :] + D1[:, :, None] * xi2[:, None, :]
    B[:, 3, :, 3:6] = D1[:, :, None] * xi2[:, None, :]
    B[:, 4, :, 3:6] = -D2[:, :, None] * xi1[:, None, :]
    B[:, 5, :, 3:6] = D2[:, :, None] * xi2[:, None, :] - D1[:, :, None] * xi1[:, None, :]
    BDh = B.reshape(ne, 6, 24)

    Bl = np.zeros((ne, 6, 4, NDOF6))
    Bl[:, 0, :, 0:3] = N[None, :, None] * (x11[:, None] * xi1)[:, None, :]
    Bl[:, 1, :, 0:3] = N[None, :, None] * (x12[:, None] * xi2)[:, None, :]
    Bl[:, 2, :, 0:3] = N[None, :, None] * (x11[:, None] * xi2 + x12[:, None] * xi1)[:, None, :]
    Bl[:, 3, :, 3:6] = N[None, :, None] * (x11[:, None] * xi2)[:, None, :]
    Bl[:, 4, :, 3:6] = N[None, :, None] * (-x12[:, None] * xi1)[:, None, :]
    Bl[:, 5, :, 3:6] = N[None, :, None] * (x12[:, None] * xi2 - x11[:, None] * xi1)[:, None, :]
    BDl = Bl.reshape(ne, 6, 24)

    Bg = np.zeros((ne, 2, 4, NDOF6))
    Bg[:, 0, :, 0:3] = D1[:, :, None] * yv[:, None, :]
    Bg[:, 1, :, 0:3] = D2[:, :, None] * yv[:, None, :]
    Bg[:, 0, :, 3:6] = N[None, :, None] * xi2[:, None, :]
    Bg[:, 1, :, 3:6] = N[None, :, None] * (-xi1)[:, None, :]
    BGh = Bg.reshape(ne, 2, 24)

    Bgl = np.zeros((ne, 2, 4, NDOF6))
    Bgl[:, 0, :, 0:3] = N[None, :, None] * (x11[:, None] * yv)[:, None, :]
    Bgl[:, 1, :, 0:3] = N[None, :, None] * (x12[:, None] * yv)[:, None, :]
    BGl = Bgl.reshape(ne, 2, 24)

    DRe = np.zeros((ne, 4))
    DRe[:, 1] = -0.5 * (x11 * Rn2 - x12 * Rn1)
    Dr = np.zeros((ne, 4, NDOF6))
    Dr[:, :, 0:3] = -0.5 * (D1[:, :, None] * xi2[:, None, :] - D2[:, :, None] * xi1[:, None, :])
    Dr[:, :, 3:6] = N[None, :, None] * yv[:, None, :]
    DRh = Dr.reshape(ne, 24)
    Drl = np.zeros((ne, 4, NDOF6))
    Drl[:, :, 0:3] = N[None, :, None] * (-0.5 * (x11[:, None] * xi2 - x12[:, None] * xi1))[:, None, :]
    DRl = Drl.reshape(ne, 24)
    return BDe, BDh, BDl, BGe, BGh, BGl, DRe, DRh, DRl, dA


def _tie_rows_batch(Xe, e3e, cross, ax):
    """Shear rows at the Dvorkin-Bathe tying points, once per element set (they do
    not depend on the quadrature point; the scalar path re-evaluated them per GP)."""
    return {
        "g13m": quad_ops_indep_batch(Xe, e3e, -1.0, 0.0, cross, ax)[4][:, 0:1, :],
        "g13p": quad_ops_indep_batch(Xe, e3e, 1.0, 0.0, cross, ax)[4][:, 0:1, :],
        "g23m": quad_ops_indep_batch(Xe, e3e, 0.0, -1.0, cross, ax)[4][:, 1:2, :],
        "g23p": quad_ops_indep_batch(Xe, e3e, 0.0, 1.0, cross, ax)[4][:, 1:2, :],
    }


def _shear_batch(xi, eta, scheme, BGh_gauss, tie):
    """Batched tied shear rows at (xi, eta) -- mirrors _mitc_shear_indep."""
    g23 = 0.5 * (1.0 - eta) * tie["g23m"] + 0.5 * (1.0 + eta) * tie["g23p"]
    if scheme in ("mitc4_both", "mitc4_wonly"):
        g13 = 0.5 * (1.0 - xi) * tie["g13m"] + 0.5 * (1.0 + xi) * tie["g13p"]
    else:
        g13 = BGh_gauss[:, 0:1, :]
    BGt = np.concatenate([g13, g23], axis=1)
    if scheme == "mitc4_wonly":
        rot = np.zeros(24, bool)
        for a in range(4):
            rot[NDOF6 * a + 3:NDOF6 * a + 6] = True
        BGt = BGt.copy()
        BGt[:, :, rot] = BGh_gauss[:, :, rot]
    return BGt


def _d_scale(D_by):
    """Characteristic ABD stiffness magnitude = max |diag(D)| over layups.  The drilling
    penalty is set to beta*this so it is dimensionally commensurate with the elastic
    stiffness (enforces DR=0 without the ill-conditioning of an over-large absolute pen)."""
    keys = D_by.keys() if isinstance(D_by, dict) else range(len(D_by))
    return max(float(np.max(np.abs(np.diag(np.asarray(D_by[k]))))) for k in keys)


def assemble_segment_indep(nodes, quads, subdom, e3s, D_by, G_by, k22_e, cross, ax,
                           kg_e=None, pen=None, pen_beta=0.1, dof_map=None,
                           shear="full", sparse=False):
    """6-DOF assembly with the finite drilling-residual penalty pen*DR^2.
    pen defaults to pen_beta * max|diag(D)| (D-scaled; robust across materials/thickness).
    shear: 'full' (default) integrates both transverse-shear rows untied -- with the
    INDEPENDENT omega_3 both rows carry algebraic drilling content that Dvorkin-Bathe
    assumed-strain interpolation ALIASES (square thin: 'mitc4_both' -31/-50% GA3,
    'mitc4_g23' -15/-30% GA2, vs 'full' -4..-6%), and no transverse-shear locking is
    observed untied (circle: full==tied to 0.2%).  Tied variants kept for the ablation."""
    if pen is None:
        pen = pen_beta * _d_scale(D_by)
    if dof_map is None:
        dof_map = np.arange(len(nodes))
    dof_map = np.asarray(dof_map, int)
    nodes = np.asarray(nodes, float); quads = np.asarray(quads, int)
    ne = len(quads)
    Nn = int(np.max(dof_map)) + 1; ndof = NDOF6 * Nn
    Xe = nodes[quads]; e3e = np.asarray(e3s, float)
    sd = np.asarray(subdom, int)
    keys = sorted(set(int(s) for s in sd))
    Darr = np.stack([np.asarray(D_by[k], float) for k in keys])
    Garr = np.stack([np.asarray(G_by[k], float) for k in keys])
    pos = {k: i for i, k in enumerate(keys)}
    sdi = np.array([pos[int(s)] for s in sd])
    De = Darr[sdi]; Gm = Garr[sdi]                      # (ne,6,6), (ne,2,2)

    g = (NDOF6 * dof_map[quads])[:, :, None] + np.arange(NDOF6)[None, None, :]
    g = g.reshape(ne, 24)
    tie = None if shear == "full" else _tie_rows_batch(Xe, e3e, cross, ax)

    Ehh = np.zeros((ne, 24, 24)); Ehe = np.zeros((ne, 24, 4)); Dee = np.zeros((4, 4))
    Ehl = np.zeros((ne, 24, 24)); Ell = np.zeros((ne, 24, 24)); Ele = np.zeros((ne, 24, 4))
    gpv = 1.0 / np.sqrt(3.0)
    for (xi, eta) in [(-gpv, -gpv), (gpv, -gpv), (gpv, gpv), (-gpv, gpv)]:
        BDe, BDh, BDl, BGe, BGh, BGl, DRe, DRh, DRl, dA = quad_ops_indep_batch(
            Xe, e3e, xi, eta, cross, ax)
        BGt = BGh if shear == "full" else _shear_batch(xi, eta, shear, BGh, tie)
        w = dA[:, None, None]
        DB = np.einsum('eij,ejb->eib', De, BDh)
        GB = np.einsum('eij,ejb->eib', Gm, BGt)
        DBe = np.einsum('eij,ejb->eib', De, BDe)
        GBe = np.einsum('eij,ejb->eib', Gm, BGe)
        DBl = np.einsum('eij,ejb->eib', De, BDl)
        GBl = np.einsum('eij,ejb->eib', Gm, BGl)
        Ehh += w * (np.einsum('eia,eib->eab', BDh, DB) + np.einsum('eia,eib->eab', BGt, GB)
                    + pen * DRh[:, :, None] * DRh[:, None, :])
        Ehe += w * (np.einsum('eia,eib->eab', BDh, DBe) + np.einsum('eia,eib->eab', BGt, GBe)
                    + pen * DRh[:, :, None] * DRe[:, None, :])
        Dee += np.einsum('e,eab->ab', dA,
                         np.einsum('eia,eib->eab', BDe, DBe) + np.einsum('eia,eib->eab', BGe, GBe)
                         + pen * DRe[:, :, None] * DRe[:, None, :])
        Ehl += w * (np.einsum('eia,eib->eab', BDh, DBl) + np.einsum('eia,eib->eab', BGt, GBl)
                    + pen * DRh[:, :, None] * DRl[:, None, :])
        Ell += w * (np.einsum('eia,eib->eab', BDl, DBl) + np.einsum('eia,eib->eab', BGl, GBl)
                    + pen * DRl[:, :, None] * DRl[:, None, :])
        Ele += w * (np.einsum('eia,eib->eab', BDl, DBe) + np.einsum('eia,eib->eab', BGl, GBe)
                    + pen * DRl[:, :, None] * DRe[:, None, :])

    Dhe = np.zeros((ndof, 4)); Dle = np.zeros((ndof, 4))
    np.add.at(Dhe, g.reshape(-1), Ehe.reshape(-1, 4))
    np.add.at(Dle, g.reshape(-1), Ele.reshape(-1, 4))
    if sparse:
        # COO triplets from the per-element 24x24 blocks (dense ndof x ndof would be
        # TB at ~1e6 DOF); duplicates summed by tocsr().  Dhe/Dle stay dense (thin).
        from scipy.sparse import coo_matrix
        rr = np.broadcast_to(g[:, :, None], (ne, 24, 24)).ravel()
        cc = np.broadcast_to(g[:, None, :], (ne, 24, 24)).ravel()
        Dhh = coo_matrix((Ehh.ravel(), (rr, cc)), shape=(ndof, ndof)).tocsr()
        Dhl = coo_matrix((Ehl.ravel(), (rr, cc)), shape=(ndof, ndof)).tocsr()
        Dll = coo_matrix((Ell.ravel(), (rr, cc)), shape=(ndof, ndof)).tocsr()
        return Dhh, Dhe, Dee, Dhl, Dll, Dle
    Dhh = np.zeros((ndof, ndof)); Dhl = np.zeros((ndof, ndof)); Dll = np.zeros((ndof, ndof))
    rc = (g[:, :, None], g[:, None, :])
    np.add.at(Dhh, rc, Ehh)
    np.add.at(Dhl, rc, Ehl)
    np.add.at(Dll, rc, Ell)
    return Dhh, Dhe, Dee, Dhl, Dll, Dle


def assemble_constraint(nodes, quads, subdom, e3s, k22_e, cross, ax, kg_e=None,
                        lam_space="elem", dof_map=None, sparse=False):
    """Weak drilling-constraint operators for the Lagrange multiplier field.

    lam_space='elem' (default): one PIECEWISE-CONSTANT multiplier per element,
      <DR>_e = 0 -- the inf-sup-stable choice (an equal-order nodal multiplier
      over-constrains under refinement: square thin GA2/GA3 drift -1/-2% ->
      -17/-9% from NC=24 to NC=96, the classical LBB failure of equal-order
      multiplier spaces; the element-constant space removes the drift).
    lam_space='node': one multiplier per node, <N_a DR> = 0 (kept for the ablation).
    lam_space='elem_nofold': element-constant multipliers, EXCLUDING elements adjacent
      to a FOLD line (nodes where incident-element normals disagree > 30 deg).  The
      drilling constraint is derived on a smooth surface patch; across a slope
      discontinuity the C0-shared fields cannot satisfy both walls' symmetry rows
      simultaneously, and the growing number of fold-line rows (~1/h) produces the
      linear-in-1/h over-constraint drift seen on the square (GA2 -1 -> -17%).
    dof_map (optional): node -> dof-node index (wrapped prismatic strip, as in the
    ring SG).  With a dof_map the local index vector contains REPEATED dofs, so the
    accumulation uses np.add.at (fancy-index += silently drops duplicates).
    Returns G (P x 6Nd) on w_s, Gl (P x 6Nd) on w_s', Ge (P x 4) on eb."""
    Nn = len(nodes)
    if dof_map is None:
        dof_map = np.arange(Nn)
    Nd = int(np.max(dof_map)) + 1; M = NDOF6 * Nd
    skip = np.zeros(len(quads), bool)
    if lam_space == "elem_nofold":
        # per-element unit normal from the two diagonals
        nrm = np.cross(nodes[quads[:, 2]] - nodes[quads[:, 0]],
                       nodes[quads[:, 3]] - nodes[quads[:, 1]])
        nrm /= (np.linalg.norm(nrm, axis=1)[:, None] + 1e-30)
        node_ref = [[] for _ in range(Nn)]
        for q, quad in enumerate(quads):
            for nd in quad:
                node_ref[int(nd)].append(q)
        fold_node = np.zeros(Nn, bool)
        for nd in range(Nn):
            qs = node_ref[nd]
            for a in range(len(qs)):
                for c in range(a + 1, len(qs)):
                    if abs(float(nrm[qs[a]] @ nrm[qs[c]])) < np.cos(np.radians(30.0)):
                        fold_node[nd] = True
        skip = np.array([any(fold_node[int(nd)] for nd in quad) for quad in quads])
    if lam_space.startswith("elem"):
        row_of = -np.ones(len(quads), int)
        row_of[~skip] = np.arange(int((~skip).sum()))
        P = int((~skip).sum())
    else:
        P = Nd
    Ge = np.zeros((P, 4))
    gpv = 1.0 / np.sqrt(3.0)
    gp = [(-gpv, -gpv), (gpv, -gpv), (gpv, gpv), (-gpv, gpv)]
    if lam_space.startswith("elem"):
        # batched: accumulate the element-integrated DR rows for all elements at
        # once (the scalar per-element loop dominated the constraint assembly)
        nodes = np.asarray(nodes, float); quadsA = np.asarray(quads, int)
        ne = len(quadsA)
        Xe = nodes[quadsA]; e3e = np.asarray(e3s, float)
        gloc = (NDOF6 * np.asarray(dof_map, int)[quadsA])[:, :, None] \
            + np.arange(NDOF6)[None, None, :]
        gloc = gloc.reshape(ne, 24)
        Gel = np.zeros((ne, 24)); Glel = np.zeros((ne, 24)); Geel = np.zeros((ne, 4))
        for (xi, eta) in gp:
            _, _, _, _, _, _, DRe, DRh, DRl, dA = quad_ops_indep_batch(
                Xe, e3e, xi, eta, cross, ax)
            Gel += DRh * dA[:, None]
            Glel += DRl * dA[:, None]
            Geel += DRe * dA[:, None]
        keep = ~skip
        rows = (row_of if lam_space == "elem_nofold" else np.arange(ne))[keep]
        np.add.at(Ge, rows, Geel[keep])
        if sparse:
            from scipy.sparse import coo_matrix
            Rk = np.broadcast_to(rows[:, None], (int(keep.sum()), 24)).ravel()
            Ck = gloc[keep].ravel()
            G = coo_matrix((Gel[keep].ravel(), (Rk, Ck)), shape=(P, M)).tocsr()
            Gl = coo_matrix((Glel[keep].ravel(), (Rk, Ck)), shape=(P, M)).tocsr()
            return G, Gl, Ge
        G = np.zeros((P, M)); Gl = np.zeros((P, M))
        np.add.at(G, (rows[:, None], gloc[keep]), Gel[keep])
        np.add.at(Gl, (rows[:, None], gloc[keep]), Glel[keep])
        return G, Gl, Ge
    G = np.zeros((P, M)); Gl = np.zeros((P, M))
    for q, quad in enumerate(quads):
        if skip[q]:
            continue
        X = nodes[quad]; k22 = float(k22_e[q]); kg = float(kg_e[q]) if kg_e is not None else 0.0
        gloc = np.array([NDOF6 * int(dof_map[nd]) + c for nd in quad for c in range(NDOF6)])
        for (xi, eta) in gp:
            N, _, _ = _bilinear(xi, eta)
            _, _, _, _, _, _, DRe, DRh, DRl, dA = quad_ops_indep(X, e3s[q], xi, eta, k22, cross, ax, kg)
            for a in range(4):
                nd = int(dof_map[quad[a]])
                np.add.at(G[nd], gloc, N[a] * DRh * dA)
                np.add.at(Gl[nd], gloc, N[a] * DRl * dA)
                Ge[nd] += N[a] * DRe * dA
    return G, Gl, Ge


def build_C_Psi_segment6(nodes, quads, cross):
    """6-DOF rigid-body kernel/constraints (om3 column = 0: drilling is not a rigid mode)."""
    nodes = np.asarray(nodes, float); quads = np.asarray(quads, int)
    Nn = len(nodes); ndof = NDOF6 * Nn
    C = np.zeros((4, ndof)); Psi = np.zeros((ndof, 4))
    gpv = 1.0 / np.sqrt(3.0)
    qp = [(-gpv, -gpv), (gpv, -gpv), (gpv, gpv), (-gpv, gpv)]
    Xe = nodes[quads]
    node_w = np.zeros(Nn)
    for (xi, eta) in qp:
        Nn_, dNx, dNe = _bilinear(xi, eta)
        dA = np.linalg.norm(np.cross(np.einsum('a,eaj->ej', dNx, Xe),
                                     np.einsum('a,eaj->ej', dNe, Xe)), axis=1)
        np.add.at(node_w, quads, dA[:, None] * Nn_[None, :])
    base = NDOF6 * np.arange(Nn)
    for cc in range(4):
        C[cc, base + cc] = node_w
    y2 = nodes[:, cross[0]]; y3 = nodes[:, cross[1]]
    Psi[base + 0, 0] = 1.0
    Psi[base + 1, 1] = 1.0
    Psi[base + 2, 2] = 1.0
    Psi[base + 1, 3] = -y3; Psi[base + 2, 3] = y2; Psi[base + 3, 3] = -1.0
    return C, Psi
