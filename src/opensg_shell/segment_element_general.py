"""
segment_element_general.py -- GENERAL 3D-taper Reissner-Mindlin surface operators
==================================================================================
Implements the GENERAL RM shell strains and curvature strains of the OpenSG-RM
paper (Overleaf "OpenSG: RM to Timoshenko beam", Shell Strains section + Appendix
A.3-A.8), of which the prismatic cross-section operator (segment_element._quad_ops
/ msg_rm_timo._elem_BD_BG_BL) is the x_{1;1}=1, x_{1;2}=0 SIMPLIFIED form.

NOTATION (paper -> code)
------------------------
  zeta_1, zeta_2   surface coordinates along the in-plane axial tangent a1 and
                   hoop tangent a2 (unit, orthonormal).  d/dzeta_alpha of the
                   shape functions: D1a, D2a  (from the 2x2 metric).
  x_{i;alpha}      direction cosines  = i-th GLOBAL component of a_alpha, with
                   the global index i mapped to (1=beam axis, 2,3=cross coords):
                     x11=a1.b1  x21=a1.b2  x31=a1.b3     (b = beam frame)
                     x12=a2.b1  x22=a2.b2  x32=a2.b3
  y_i = C_{3i}     normal direction cosines: y1=n.b1, y2=n.b2, y3=n.b3, with
                   n = a1 x a2 (right-handed; inward for a CCW hoop).  In the
                   prismatic limit  (y2,y3) = (-xdot3, +xdot2)  so  C33 -> xdot2,
                   reproducing the paper's 1/(2 xdot2) drilling factors.
  w_{i|alpha}      SG-surface derivative of the fluctuation w_i along zeta_alpha
  w'_i, omega'_b   x1-(macro)-derivative fields (the Gamma_l argument)
  kappa_{1i}       beam curvatures (kappa_1 twist, kappa_2, kappa_3 bending);
                   beam strain vector eb = [gamma_11, k1, k2, k3]
  Rn1, Rn2         swept-area measures (A.4):
                     Rn1 = x2*x_{3;1} - x3*x_{2;1},  Rn2 = x2*x_{3;2} - x3*x_{2;2}

DOF per node: [w1, w2, w3, omega_1, omega_2]  (w_i global; omega_beta = the two
RM shear rotations, beta a GLOBAL index 1,2 as in A.3/A.8 x_{beta;alpha} pairing).

DRILLING (A.3, A.5, A.6):
  omega_3   = S/(2 C33) - (C3b/C33) omega_b                              (A.3)
      S     = k1 (x11 Rn2 - x12 Rn1) + w'_i (x11 x_{i;2} - x12 x_{i;1})
              + (w_{i|1} x_{i;2} - w_{i|2} x_{i;1})
  omega'_3  = (w'_{i|1} x_{i;2} - w'_{i|2} x_{i;1})/(2 C33) - (C3b/C33) omega'_b
              [ kappa'_1 and w'' terms neglected, order eps^3 -- A.5 ]
  Lambda_a  = omega'_3 x_{1;a} + omega_{3|a}                             (A.6)
  omega_{3|a}: differentiate the omega_3 OPERATOR along zeta_a:
      * shape-function part: KEPT where it stays first-order
        (the -(C3b/C33) omega_{b|a} term);  the w_{i|1|a}, w_{i|2|a} pieces are
        SECOND SG derivatives -- NOT RESOLVABLE in C0 RM -> DROPPED (this is the
        established RM decision: no second derivative of the fluctuations).
      * coefficient part: contour (zeta_2) derivatives of the direction cosines
        via the initial curvature k22 of the wall (Frenet along the hoop):
            (x_{i;2})_{;2} = k22 * y_i ,   (y_i)_{;2} = -k22 * x_{i;2} ,
            (x_{i;1})_{;2} = 0 (+geodesic terms, dropped on flat-ish quads),
            (x_i)_{;a}     = x_{i;a}      (coordinates),
        and all (.)_{;1} coefficient derivatives (taper rate of the frame) are
        higher-order -> dropped, marked below.

CURVATURE STRAINS (A.8), with Lambda as above:
  k^s_11      =  x11 x_{i;2} k_{1i}  +  x_{b;2} x11 omega'_b
               + x_{b;2} omega_{b|1} +  x_{3;2} Lambda_1
  k^s_22      = -[ x12 x_{i;1} k_{1i} + x_{b;1} x12 omega'_b
               + x_{b;1} omega_{b|2} + x_{3;1} Lambda_2 ]
  k^s_12+21   =  k_{1i}(x12 x_{i;2} - x11 x_{i;1})
               + omega'_b (x12 x_{b;2} - x11 x_{b;1})
               + (x_{b;2} omega_{b|2} - x_{b;1} omega_{b|1})
               + x_{3;2} Lambda_2 - x_{3;1} Lambda_1

MEMBRANE STRAINS (paper, Shell Strains):
  e_11   = x11^2 g11 + x11(x2 x31 - x3 x21) k1 + x11^2 (x3 k2 - x2 k3)
           + x_{i;1} w_{i|1} + x11 x_{i;1} w'_i
  e_22   = x12^2 g11 + x12(x2 x32 - x3 x22) k1 + x12^2 (x3 k2 - x2 k3)
           + x_{i;2} w_{i|2} + x12 x_{i;2} w'_i
  2e_12  = 2 x11 x12 g11 + [x2(x11 x32 + x12 x31) - x3(x11 x22 + x12 x21)] k1
           + 2 x11 x12 (x3 k2 - x2 k3)
           + (x_{i;1} w_{i|2} + x_{i;2} w_{i|1}) + (x11 x_{i;2} + x12 x_{i;1}) w'_i

TRANSVERSE SHEAR (paper Eq.12 = Shell-Strains 2eps13,2eps23 -- the FULL GENERAL
drilling-ELIMINATED form; NOT the prismatic reduction).  omega_3 is substituted
through C33=C^ab_33 (eq:om3), so the /(2 C33) factors ARE the drilling; there is
NO Lambda/curvature term in the shear (Lambda enters ONLY the curvature rows
k11/k22/k12).  With C^ab of Eq.1 (C31=y1, C33=y3, C32=y2, C23=x_{3;2}, C13=x_{3;1},
C_{2a}=x_{a;2}, C_{1a}=x_{a;1}, C_{3a}=y_a, C_{3i}=y_i, y_{j;a}=delta_{ja}):
  2g_13 = x11 C31 e11                                          # first term = C31 x11
        + [x2((x32^2 x11 - x32 x31 x12)/2C33 + C33 x11)
          -x3((x32 x22 x11 - x32 x21 x12)/2C33 + C32 x11)] k1  # swept-area
        + x11 C31 x3 k2 - x11 C31 x2 k3
        + (x32 x_{i;2}/2C33 + y_i) w_{i|1} - (x32 x_{i;1}/2C33) w_{i|2}
        + (C_{2a} - C23 C_{3a}/C33) om_a
        + x11 (x32 x_{i;2}/2C33 + y_i) w'_i                     # Gamma_l (chain rule)
  2g_23 : x11<->x12, x32<->x31, w_{i|1}<->w_{i|2}; om coeff = -C_{1a} + C13 C_{3a}/C33.
The prismatic reduction (x12=x31=y1=0, y3=xd2, x32=xd3, y2=-xd3) recovers eq:prism
2g13 = omega_2/xd2 + k1(x2(xd2+xd3^2/2xd2)+x3 xd3/2) - wd1 xd3/2xd2 + [w' pair];
the FULL code is certified term-by-term against Eq.12 + the Appendix curvature
strains by verify_strains_paper.py (<=1e-5).  1/C33 is Tikhonov-regularized
(C33_EPS) so the drilling elimination stays well-posed on flat/folded walls.
"""
import numpy as np
from .segment_element import _bilinear, dirichlet_solve, compute_k22, build_C_Psi_segment  # noqa

# beam-strain order [g11, k1, k2, k3]; DOF order per node [w1,w2,w3,om1,om2]
NDOF = 5
OM3_SIGN = +1.0   # sign convention of omega_3 (A.3); +1 confirmed by the kappa_11
                  # prismatic identity: x_{b;2}x11 om'_b + x_{3;2}(-C3b/C33)om'_b
                  # = (xd2 + xd3^2/xd2) om'_2 = om'_2/xd2  (eq:prism)
LAMBDA_ON = 1.0   # ablation switch: Lambda_alpha drilling in the kappa rows
# GDRILL_ON: the x_{3;a} omega_3 drilling coupling in the transverse-shear (gamma)
# rows.  DEFAULT OFF.  On a SMOOTH section (circle) it is a negligible stabilization
# (<0.5% on every 6x6 term), but on a FOLDED/flat-walled section it is a
# drilling-DOF-at-folds pathology: across a 90-deg corner the drilling rotation of
# one wall reads as the out-of-plane rotation of its neighbour, so this term adds a
# SPURIOUS torsional stiffness that does not converge -- on the square tube it
# inflates GJ by +1280% (and grows with mesh) whereas GDRILL_ON=0 gives GJ within
# +14% of Bredt with EA/GA/EI unchanged.  Found via the flat-walled (k22=0) square.
GDRILL_ON = 0.0
# Tikhonov regularization scale for the drilling denominator 1/C33 (C33=y3=a3.b3).
# 1/C33 -> C33/(C33^2 + C33_EPS^2): identity where C33 is O(1), smoothly ->0 as C33->0.
# On the CIRCLE C33 crosses zero only at isolated points (regularization negligible);
# on the SQUARE whole walls have C33=0 and this cleanly drops the ill-posed drilling term.
C33_EPS = 0.1
# --- DIAGNOSTIC transverse-shear TAPER ablations (default 1.0 = production, no effect) ---
# Each scales a taper-activated piece of the transverse-shear operator so we can localize
# the thin-square GA3 (C33) taper deficit.  NOT for production use.
SH_ABL_BGE = 1.0     # entire macro (beam-strain) shear coupling BGe(2,4)  [=0 prismatically except k1 swept]
SH_ABL_BGL = 1.0     # entire w'(Gamma_l) transverse-shear block BGl(2,20)
SH_ABL_Y1 = 1.0      # only the y1 = n.b1 (axial-normal) taper couplings (y1=0 prismatically)
KG_ABL = 1.0         # scale the hoop geodesic curvature kg (taper) fed to the curvature operator
G_SHEAR_SCALE = 1.0  # scale the shell transverse-shear material G in the segment assembly
DRILL_TOL = 0.0      # >0: drop the omega_3 drilling (Lambda + shear 1/C33) on |C33|=|n.b3|<tol walls


def _surf_frame(X, e3_mat, xi, eta, cross, ax):
    """Geometric surface frame + in-plane derivative operator at (xi, eta).

    a2 = hoop tangent (J_xi direction), a1 = in-plane AXIAL tangent (J_eta
    Gram-Schmidt'ed against a2 -- TILTED by the taper, this is where x21,x31
    come from), n = a1 x a2 sign-matched to the stored material normal e3_mat.
    Returns (N, D1, D2, dA, cosines dict).
    """
    N, dNx, dNe = _bilinear(xi, eta)
    Jxi = dNx @ X; Jeta = dNe @ X
    a2 = Jxi / np.linalg.norm(Jxi)
    a1 = Jeta - (Jeta @ a2) * a2
    a1 = a1 / np.linalg.norm(a1)
    n = np.cross(a1, a2)
    if n @ e3_mat < 0.0:                      # keep the mesh's inward/outward choice
        n = -n; a1 = -a1                      # flip a1 too -> right-handed (a1,a2,n)
    # in-plane derivatives: [d/dz1; d/dz2] = (G^T)^-1 [d/dxi; d/deta]
    G = np.array([[a1 @ Jxi, a1 @ Jeta], [a2 @ Jxi, a2 @ Jeta]])
    d = (np.linalg.inv(G.T) @ np.vstack([dNx, dNe]))      # (2,4): rows = [D1; D2]
    dA = np.linalg.norm(np.cross(Jxi, Jeta))
    x = N @ X
    c = dict(
        x11=a1[ax], x21=a1[cross[0]], x31=a1[cross[1]],
        x12=a2[ax], x22=a2[cross[0]], x32=a2[cross[1]],
        y1=n[ax],   y2=n[cross[0]],   y3=n[cross[1]],
        x2=x[cross[0]], x3=x[cross[1]],
    )
    return N, d[0], d[1], dA, c


FLOOR33 = 5e-2   # drilling-denominator floor for C33=n.b3 (diagnostic-tunable)


def _c33_floor(y3, floor=None):
    """C33 = n.b3 appears in the drilling denominators; floor it away from zero
    exactly like the validated 1-D code floors 1/xdot2 (vertical walls)."""
    floor = FLOOR33 if floor is None else floor
    return y3 if abs(y3) > floor else (floor if y3 >= 0 else -floor)


def _omega3_ops(N, D1, D2, c, k22):
    """Drilling operators (A.3/A.5): row vectors over the element DOFs.

        omega_3      = OM3h . w  + OM3l . w'  + OM3e . eb          (A.3)
        omega'_3     =             OM3pl . w' (same row as OM3h)   (A.5)
                       + the -(C3b/C33) omega'_b algebraic part
    Returns (OM3h(20,), OM3l(20,), OM3e(4,), OM3p_l(20,)).
    """
    x11, x12 = c["x11"], c["x12"]
    xi1 = np.array([x11, c["x21"], c["x31"]])            # x_{i;1}
    xi2 = np.array([x12, c["x22"], c["x32"]])            # x_{i;2}
    yv = np.array([c["y1"], c["y2"], c["y3"]])           # C_{3i}
    C33 = _c33_floor(c["y3"])
    Rn1 = c["x2"] * c["x31"] - c["x3"] * c["x21"]        # (A.4)
    Rn2 = c["x2"] * c["x32"] - c["x3"] * c["x22"]

    OM3h = np.zeros(4 * NDOF); OM3l = np.zeros(4 * NDOF)
    OM3pl = np.zeros(4 * NDOF); OM3e = np.zeros(4)
    #   S = k1 (x11 Rn2 - x12 Rn1) + w'_i (x11 x_{i;2} - x12 x_{i;1})
    #       + (w_{i|1} x_{i;2} - w_{i|2} x_{i;1})              -> /(2 C33)
    OM3e[1] = (x11 * Rn2 - x12 * Rn1) / (2 * C33)
    for a in range(4):
        o = NDOF * a
        for i in range(3):
            OM3h[o + i] += (xi2[i] * D1[a] - xi1[i] * D2[a]) / (2 * C33)
            OM3l[o + i] += (x11 * xi2[i] - x12 * xi1[i]) * N[a] / (2 * C33)
            #   omega'_3 (A.5): (w'_{i|1} x_{i;2} - w'_{i|2} x_{i;1})/(2C33)
            OM3pl[o + i] += (xi2[i] * D1[a] - xi1[i] * D2[a]) / (2 * C33)
        #   -(C3b/C33) omega_b   (algebraic; beta = GLOBAL 1,2 -> dofs om1,om2)
        OM3h[o + 3] += -(yv[0] / C33) * N[a]
        OM3h[o + 4] += -(yv[1] / C33) * N[a]
        OM3pl[o + 3] += -(yv[0] / C33) * N[a]             # -(C3b/C33) omega'_b
        OM3pl[o + 4] += -(yv[1] / C33) * N[a]
    return OM3_SIGN*OM3h, OM3_SIGN*OM3l, OM3_SIGN*OM3e, OM3_SIGN*OM3pl


def _lambda_ops(N, D1, D2, c, k22, kg, alpha):
    """Lambda_alpha = omega'_3 x_{1;alpha} + omega_{3|alpha}  (paper Eq. omega3alpha),
    with the geometric coefficient derivatives modeled by the DARBOUX frame of the
    hoop coordinate line (verified against exact FD geometry, verify_strains_paper):

      along zeta_2 (hoop):   (a1)_{;2} = kg a2
                             (a2)_{;2} = -k22 n - kg a1
                             (n)_{;2}  = +k22 a2         [surface torsion ~ 0]
      along zeta_1 (axial):  frame derivatives = 0 (straight generators of a
                             linear taper -- verified exactly axially invariant);
                             COORDINATE derivatives survive: (x_i)_{;1} = x_{i;1},
                             so Rn2_{;1} = x21 x32 - x31 x22 != 0 under taper.

    k22 = hoop normal curvature (d e3/ds . e2), kg = hoop GEODESIC curvature
    (d e1/ds . e2; 0 prismatic/flat wall, ~ taper x 1/R on a tapered curved wall).

    DROPPED (documented): w_{i|1|alpha}, w_{i|2|alpha} (2nd SG derivatives --
    C0-RM cannot resolve) and the kappa'_11 (eb-derivative) term.
    Returns (Lh(20,), Ll(20,), Le(4,)) acting on (w, w', eb).
    """
    x11, x12 = c["x11"], c["x12"]
    xi1 = np.array([x11, c["x21"], c["x31"]])
    xi2 = np.array([x12, c["x22"], c["x32"]])
    yv = np.array([c["y1"], c["y2"], c["y3"]])
    C33 = _c33_floor(c["y3"])
    x2, x3 = c["x2"], c["x3"]
    Rn1 = x2 * xi1[2] - x3 * xi1[1]
    Rn2 = x2 * xi2[2] - x3 * xi2[1]
    x1a = x11 if alpha == 1 else x12                      # x_{1;alpha}
    Da = D1 if alpha == 1 else D2

    # geometric coefficient derivatives (.)_{;alpha} from the Darboux model
    if alpha == 1:
        xi1_a = np.zeros(3); xi2_a = np.zeros(3); yv_a = np.zeros(3)
        x2_a, x3_a = xi1[1], xi1[2]                       # (x_i)_{;1} = x_{i;1}
    else:
        xi1_a = kg * xi2
        xi2_a = -k22 * yv - kg * xi1
        yv_a = k22 * xi2
        x2_a, x3_a = xi2[1], xi2[2]                       # (x_i)_{;2} = x_{i;2}
    C33_a = yv_a[2]
    Rn1_a = x2_a * xi1[2] + x2 * xi1_a[2] - x3_a * xi1[1] - x3 * xi1_a[1]
    Rn2_a = x2_a * xi2[2] + x2 * xi2_a[2] - x3_a * xi2[1] - x3 * xi2_a[1]
    tw = 2.0 * C33
    dinv = -(2.0 * C33_a) / (tw * tw)                     # d/dz_alpha [1/(2C33)]

    OM3h, OM3l, OM3e, OM3pl = _omega3_ops(N, D1, D2, c, k22)
    Lh = np.zeros(4 * NDOF); Ll = np.zeros(4 * NDOF); Le = np.zeros(4)
    sgn = OM3_SIGN                                        # every Lambda term carries omega_3's sign

    # ---- t1: kappa_1 coefficient-derivative group (paper line 882-883) ----
    Le[1] += sgn * ((xi1_a[0] * Rn2 + x11 * Rn2_a - xi2_a[0] * Rn1 - x12 * Rn1_a) / tw
                    + (x11 * Rn2 - x12 * Rn1) * dinv)

    # ---- t4 + omega' parts: x_{1;alpha} * omega'_3  (OM3pl carries the
    #      w'_{i|1}, w'_{i|2} shape parts and -(C3b/C33) omega'_b) ----
    Ll += x1a * OM3pl                                     # (OM3pl already sign-scaled)

    for a in range(4):
        o = NDOF * a
        # ---- t8 shape: -(C3b/C33) omega_{b|alpha} ----
        Lh[o + 3] += -sgn * (yv[0] / C33) * Da[a]
        Lh[o + 4] += -sgn * (yv[1] / C33) * Da[a]
        # ---- t8 coefficient: -om_b (y_{b;alpha} C33 - C33_alpha y_b)/C33^2 ----
        for b in range(2):
            Lh[o + 3 + b] += -sgn * ((yv_a[b] * C33 - C33_a * yv[b]) / (C33 * C33)) * N[a]
        for i in range(3):
            # ---- t3: w'_{i|alpha} (first derivative of the independent w' field) ----
            Ll[o + i] += sgn * (x11 * xi2[i] - x12 * xi1[i]) * Da[a] / tw
            # ---- t5: w'_i coefficient-derivative group (paper line 887-888) ----
            Ll[o + i] += sgn * ((xi1_a[0] * xi2[i] + x11 * xi2_a[i]
                                 - xi2_a[0] * xi1[i] - x12 * xi1_a[i]) / tw
                                + (x11 * xi2[i] - x12 * xi1[i]) * dinv) * N[a]
            # ---- t7: w_{i|1}, w_{i|2} coefficient-derivative group (line 890) ----
            Lh[o + i] += sgn * ((xi2_a[i] / tw + xi2[i] * dinv) * D1[a]
                                - (xi1_a[i] / tw + xi1[i] * dinv) * D2[a])
    return Lh, Ll, Le


def quad_ops_general(X, e1m, e2m, e3m, xi, eta, k22, cross=(1, 2), ax=None, kg=0.0):
    """GENERAL RM operators at (xi,eta):  returns
        BDe (6,4)  macro map        Gamma_eps (D-block rows)
        BDh (6,20) fluctuation      Gamma_h
        BDl (6,20) x1-derivative    Gamma_l
        BGe (2,4), BGh (2,20), BGl (2,20)   transverse-shear blocks
        dA  surface Jacobian
    Rows: [e11, e22, 2e12, k11, k22, k12+21] and [2g13, 2g23]."""
    if ax is None:
        ax = [j for j in range(3) if j not in cross][0]
    kg = KG_ABL * kg                                     # diagnostic geodesic-curvature ablation
    N, D1, D2, dA, c = _surf_frame(X, e3m, xi, eta, cross, ax)
    # DRILL_TOL: on gauss points where the drilling denominator C33=n.b3 nearly vanishes
    # (a whole flat wall, e.g. the GA3-carrier walls / shear web), DROP the drilling
    # rather than floor 1/C33.  dmask kills the omega_3 (Lambda) curvature contribution
    # AND the shear 1/C33 there; keeps the regular (non-drilling) strains intact.
    dmask = 0.0 if (DRILL_TOL > 0.0 and abs(c["y3"]) < DRILL_TOL) else 1.0
    lam = LAMBDA_ON * dmask
    x11, x12 = c["x11"], c["x12"]
    xi1 = np.array([x11, c["x21"], c["x31"]])
    xi2 = np.array([x12, c["x22"], c["x32"]])
    yv = np.array([c["y1"], c["y2"], c["y3"]])
    x2, x3 = c["x2"], c["x3"]
    Rn1 = x2 * c["x31"] - x3 * c["x21"]
    Rn2 = x2 * c["x32"] - x3 * c["x22"]

    BDe = np.zeros((6, 4)); BDh = np.zeros((6, 4 * NDOF)); BDl = np.zeros((6, 4 * NDOF))
    BGe = np.zeros((2, 4)); BGh = np.zeros((2, 4 * NDOF)); BGl = np.zeros((2, 4 * NDOF))

    # ================= MEMBRANE (general; paper Shell Strains) =================
    # e11 = x11^2 g11 + x11(x2 x31 - x3 x21)k1 + x11^2(x3 k2 - x2 k3)
    BDe[0] = [x11**2, x11 * Rn1, x11**2 * x3, -(x11**2) * x2]
    # e22 = x12^2 g11 + x12 Rn2 k1 + x12^2 (x3 k2 - x2 k3)
    BDe[1] = [x12**2, x12 * Rn2, x12**2 * x3, -(x12**2) * x2]
    # 2e12 = 2 x11 x12 g11 + (x11 Rn2 + x12 Rn1) k1 + 2 x11 x12 (x3 k2 - x2 k3)
    BDe[2] = [2 * x11 * x12, x11 * Rn2 + x12 * Rn1, 2 * x11 * x12 * x3, -2 * x11 * x12 * x2]
    for a in range(4):
        o = NDOF * a
        for i in range(3):
            BDh[0, o + i] += xi1[i] * D1[a]                       # x_{i;1} w_{i|1}
            BDl[0, o + i] += x11 * xi1[i] * N[a]                  # x11 x_{i;1} w'_i
            BDh[1, o + i] += xi2[i] * D2[a]                       # x_{i;2} w_{i|2}
            BDl[1, o + i] += x12 * xi2[i] * N[a]                  # x12 x_{i;2} w'_i
            BDh[2, o + i] += xi1[i] * D2[a] + xi2[i] * D1[a]      # mixed
            BDl[2, o + i] += (x11 * xi2[i] + x12 * xi1[i]) * N[a]

    # ================= CURVATURES (A.8, Lambda per A.6) ========================
    L1h, L1l, L1e = _lambda_ops(N, D1, D2, c, k22, kg, alpha=1)
    L2h, L2l, L2e = _lambda_ops(N, D1, D2, c, k22, kg, alpha=2)

    # k11 = x11 x_{i;2} k_{1i} + x_{b;2} x11 om'_b + x_{b;2} om_{b|1} + x32 Lambda_1
    BDe[3] = [0.0, x11 * x12, x11 * c["x22"], x11 * c["x32"]]
    BDe[3] += lam *c["x32"] * L1e
    for a in range(4):
        o = NDOF * a
        BDl[3, o + 3] += x11 * c["x12"] * N[a]                    # x_{1;2} x11 om'_1
        BDl[3, o + 4] += x11 * c["x22"] * N[a]                    # x_{2;2} x11 om'_2
        BDh[3, o + 3] += c["x12"] * D1[a]                         # x_{1;2} om_{1|1}
        BDh[3, o + 4] += c["x22"] * D1[a]                         # x_{2;2} om_{2|1}
    BDh[3] += lam *c["x32"] * L1h; BDl[3] += lam *c["x32"] * L1l

    # k22 = -[ x12 x_{i;1} k_{1i} + x_{b;1} x12 om'_b + x_{b;1} om_{b|2} + x31 Lambda_2 ]
    BDe[4] = [0.0, -x12 * x11, -x12 * c["x21"], -x12 * c["x31"]]
    BDe[4] += -lam *c["x31"] * L2e
    for a in range(4):
        o = NDOF * a
        BDl[4, o + 3] += -x12 * c["x11"] * N[a]
        BDl[4, o + 4] += -x12 * c["x21"] * N[a]
        BDh[4, o + 3] += -c["x11"] * D2[a]
        BDh[4, o + 4] += -c["x21"] * D2[a]
    BDh[4] += -lam *c["x31"] * L2h; BDl[4] += -lam *c["x31"] * L2l
    # NOTE prismatic check: x12=0, x31=0 -> k22 = -x_{1;1} om_{1|2} = -omdot_1  (eq:prism)
    # [the VALIDATED 1-D code carries +omdot_1 with its own compensating sign
    #  convention for omega_1; the prismatic-identity test decides the sign map]

    # k12+21 = k_{1i}(x12 x_{i;2} - x11 x_{i;1}) + om'_b(x12 x_{b;2} - x11 x_{b;1})
    #          + (x_{b;2} om_{b|2} - x_{b;1} om_{b|1}) + x32 Lambda_2 - x31 Lambda_1
    BDe[5] = [0.0,
              x12 * x12 - x11 * x11,
              x12 * c["x22"] - x11 * c["x21"],
              x12 * c["x32"] - x11 * c["x31"]]
    BDe[5] += lam *(c["x32"] * L2e - c["x31"] * L1e)
    for a in range(4):
        o = NDOF * a
        BDl[5, o + 3] += (x12 * c["x12"] - x11 * c["x11"]) * N[a]
        BDl[5, o + 4] += (x12 * c["x22"] - x11 * c["x21"]) * N[a]
        BDh[5, o + 3] += c["x12"] * D2[a] - c["x11"] * D1[a]
        BDh[5, o + 4] += c["x22"] * D2[a] - c["x21"] * D1[a]
    BDh[5] += lam *(c["x32"] * L2h - c["x31"] * L1h)
    BDl[5] += lam *(c["x32"] * L2l - c["x31"] * L1l)

    # ============ TRANSVERSE SHEAR (GENERAL, paper Shell-Strains 2eps13/2eps23) ==========
    # EXACT drilling-ELIMINATED general transverse shear from the RM paper.  omega_3 is
    # substituted algebraically through C33^ab -- the x_{3;2}/(2C33), x_{3;1}/(2C33) factors
    # ARE that substitution (so NO separate drilling boost is needed).  Using C^ab (Eq.1):
    #   C31=y1, C33=y3, C32=y2, C23=x_{3;2}=x32, C13=x_{3;1}=x31,
    #   C_{2a}=x_{a;2}, C_{1a}=x_{a;1}, C_{3a}=y_a, C_{3i}=y_i, and y_{j;a}=delta_{ja}:
    #  2eps13 = x11 C31 e11
    #         + [ x2( (x32^2 x11 - x32 x31 x12)/(2C33) + C33 x11 )
    #            -x3( (x32 x22 x11 - x32 x21 x12)/(2C33) + C32 x11 ) ] k1
    #         + x11 C31 x3 k2  - x11 C31 x2 k3
    #         + ( x32 x_{i;2}/(2C33) + y_i ) w_{i|1}  - ( x32 x_{i;1}/(2C33) ) w_{i|2}
    #         + ( C_{2a} - C23 C_{3a}/C33 ) om_a
    #         + x11 ( x32 x_{i;2}/(2C33) + y_i ) w'_i         # Gamma_l  (== eq:prism underlined)
    #  2eps23 : x11<->x12, x32<->x31, w_{i|1}<->w_{i|2}; om coeff = -C_{1a} + C13 C_{3a}/C33.
    # PRISMATIC (x12=x31=y1=0, y3=xd2, x32=xd3, y2=-xd3) reproduces eq:prism 2gamma_13/23
    # EXACTLY: omega_2/xd2, the k1 swept term x2(xd2+xd3^2/2xd2)+x3 xd3/2, the -wd1 xd3/2xd2,
    # and the w' pair -w'_2 xd3/2 + w'_3 (xd2 + xd3^2/2xd2).  (C32=y2 fixed by this identity.)
    # C33 = y3 = a3.b3 is the drilling-elimination denominator (Eq. om3).  It VANISHES on
    # flat walls whose hoop tangent aligns with b3 (a whole square wall has y3=0, not just
    # isolated points as on a circle), so a plain 1/C33 -- even magnitude-floored -- injects
    # a spurious drilling stiffness there (GJ blow-up = the drilling-at-folds artifact).
    # Tikhonov regularization  1/C33 -> C33/(C33^2 + eps^2)  equals 1/C33 where C33 is
    # healthy and SMOOTHLY -> 0 as C33 -> 0, dropping the ill-posed drilling term exactly at
    # the singularity (recovering the well-behaved no-drilling shear on flat walls).
    invc33 = dmask * c["y3"] / (c["y3"] ** 2 + C33_EPS ** 2)  # regularized 1/C33 (sign-preserving; dmask drops it on flat walls)
    h33 = 0.5 * invc33                                    # regularized 1/(2 C33)
    y1, y2 = SH_ABL_Y1 * c["y1"], c["y2"]; x31, x32 = c["x31"], c["x32"]; x21, x22 = c["x21"], c["x22"]
    yv_s = yv.copy(); yv_s[0] = SH_ABL_Y1 * yv[0]        # y1-ablated normal for the shear w-flux terms
    k1_13 = (x2 * ((x32 * x32 * x11 - x32 * x31 * x12) * h33 + c["y3"] * x11)
             - x3 * ((x32 * x22 * x11 - x32 * x21 * x12) * h33 + y2 * x11))
    k1_23 = (x2 * ((x31 * x31 * x12 - x31 * x32 * x11) * h33 + c["y3"] * x12)
             - x3 * ((x31 * x21 * x12 - x31 * x22 * x11) * h33 + y2 * x12))
    BGe[0] = np.array([x11 * y1, k1_13, x11 * y1 * x3, -x11 * y1 * x2])
    BGe[1] = np.array([x12 * y1, k1_23, x12 * y1 * x3, -x12 * y1 * x2])
    for a in range(4):
        o = NDOF * a
        for i in range(3):
            a13 = x32 * xi2[i] * h33 + yv_s[i]               # 2eps13 w_{i|1} coeff
            b13 = -x32 * xi1[i] * h33                        # 2eps13 w_{i|2} coeff
            a23 = x31 * xi1[i] * h33 + yv_s[i]               # 2eps23 w_{i|2} coeff
            b23 = -x31 * xi2[i] * h33                        # 2eps23 w_{i|1} coeff
            BGh[0, o + i] += a13 * D1[a] + b13 * D2[a]
            BGh[1, o + i] += a23 * D2[a] + b23 * D1[a]
            # w' CHAIN RULE: the total zeta_alpha derivative of a fluctuation splits
            # micro + macro, d(w_i)/dzeta_alpha -> w_{i|alpha} + x_{1;alpha} w'_i, so the
            # w'_i coefficient = x11*(w_{i|1} coeff) + x12*(w_{i|2} coeff)  [BOTH halves;
            # verified against eq:prism 2gamma_13 underlined terms and verify_strains_paper]
            BGl[0, o + i] += (x11 * a13 + x12 * b13) * N[a]
            BGl[1, o + i] += (x12 * a23 + x11 * b23) * N[a]
        BGh[0, o + 3] += (x12 - x32 * y1 * invc33) * N[a]    # 2eps13 om_1: C_21 - C23 C31/C33
        BGh[0, o + 4] += (x22 - x32 * y2 * invc33) * N[a]    #          om_2: C_22 - C23 C32/C33
        BGh[1, o + 3] += (-x11 + x31 * y1 * invc33) * N[a]   # 2eps23 om_1: -C_11 + C13 C31/C33
        BGh[1, o + 4] += (-x21 + x31 * y2 * invc33) * N[a]   #          om_2: -C_12 + C13 C32/C33
    BGe = SH_ABL_BGE * BGe                                    # diagnostic block ablations (default 1.0)
    BGl = SH_ABL_BGL * BGl
    return BDe, BDh, BDl, BGe, BGh, BGl, dA


# ------------------------------------------------------------------ MITC tying
# Assumed transverse-shear schemes for the QUAD element (Dvorkin-Bathe MITC4
# tying points: g23 sampled at (0,-1),(0,+1) linear in eta; g13 sampled at
# (-1,0),(+1,0) linear in xi).  Selected by name so the element stays general:
#   'mitc4_both' : tie BOTH shear rows (default -- mirrors the validated 1-D
#                  mitc_both scheme)
#   'mitc4_g23'  : tie only the hoop-locking-prone g23 row, g13 full
#   'reduced'    : single-point (centre) evaluation of both rows
#   'full'       : no treatment (exhibits transverse-shear LOCKING thin)
# TRIANGLE HOOK: if a 3-node element type is added, register its MITC3 tying
# (edge-midpoint tangential sampling, Lee-Bathe) under 'mitc3' here -- the
# assembly only calls shear_rows_general(scheme, ...).
# Standard Dvorkin-Bathe MITC4 tying (xi = r = hoop, eta = s = axial in these meshes):
#   e_rt (r-transverse shear = hoop-normal = 2*gamma_23): sampled at (0,+-1),
#        LINEAR in s (eta);
#   e_st (s-transverse shear = axial-normal = 2*gamma_13): sampled at (+-1,0),
#        LINEAR in r (xi).
# NOTE: the tapered square's GA2!=GA3 asymmetry is NOT the tying -- it is the
# general transverse-shear STRAIN expression (2g13, 2g23) being incomplete under
# taper (the "prismatic-consistent minimal generalization" flagged in the module
# docstring); tying an incomplete g13 merely exposes it.  Fix belongs in the strain
# rows (BGe/BGh/BGl), not the tying points.
_TIE_G23 = [(0.0, -1.0), (0.0, 1.0)]     # gamma_23 (row 1): sample along eta
_TIE_G13 = [(-1.0, 0.0), (1.0, 0.0)]     # gamma_13 (row 0): sample along xi
SHEAR_SCHEMES = ("mitc4_both", "mitc4_g23", "mitc4_cov", "reduced", "full")


def _metric_at(X, e3_mat, xi, eta):
    """G = [[a1.g_r, a1.g_s],[a2.g_r, a2.g_s]] (g_r=Jxi, g_s=Jeta) with the SAME
    (a1,a2) frame + e3-sign convention as _surf_frame.  Then the COVARIANT transverse
    shears are cov = G.T @ [2g13; 2g23], and physical = (G.T)^-1 @ cov."""
    N, dNx, dNe = _bilinear(xi, eta)
    Jxi = dNx @ X; Jeta = dNe @ X
    a2 = Jxi / np.linalg.norm(Jxi)
    a1 = Jeta - (Jeta @ a2) * a2
    a1 = a1 / np.linalg.norm(a1)
    if np.cross(a1, a2) @ e3_mat < 0.0:
        a1 = -a1
    return np.array([[a1 @ Jxi, a1 @ Jeta], [a2 @ Jxi, a2 @ Jeta]])


def _mitc_shear_general(X, e1m, e2m, e3m, xi, eta, k22, cross, ax, scheme="mitc4_both", kg=0.0):
    if scheme == "full":
        return quad_ops_general(X, e1m, e2m, e3m, xi, eta, k22, cross, ax, kg)[4]
    if scheme == "reduced":
        return quad_ops_general(X, e1m, e2m, e3m, 0.0, 0.0, k22, cross, ax, kg)[4]
    if scheme == "mitc4_cov":
        # RIGOROUS Dvorkin-Bathe: tie the COVARIANT shears (metric-consistent on a
        # DISTORTED quad), not the physical rows.  e_rt=cov row0 (along g_r), tied at
        # (0,+-1) linear in eta;  e_st=cov row1 (along g_s), tied at (+-1,0) linear in xi.
        def cov(tx, te):
            rows = quad_ops_general(X, e1m, e2m, e3m, tx, te, k22, cross, ax, kg)[4]
            return _metric_at(X, e3m, tx, te).T @ rows            # (2,Ndof) = [e_rt; e_st]
        A, Cc = cov(0.0, -1.0), cov(0.0, 1.0)
        ert = 0.5 * (1.0 - eta) * A[0:1, :] + 0.5 * (1.0 + eta) * Cc[0:1, :]
        Dp, Bp = cov(-1.0, 0.0), cov(1.0, 0.0)
        est = 0.5 * (1.0 - xi) * Dp[1:2, :] + 0.5 * (1.0 + xi) * Bp[1:2, :]
        return np.linalg.solve(_metric_at(X, e3m, xi, eta).T, np.vstack([ert, est]))
    # gamma_23 (row 1): tied at (0,+-1), linear in eta
    r23 = [quad_ops_general(X, e1m, e2m, e3m, tx, te, k22, cross, ax, kg)[4][1:2, :]
           for (tx, te) in _TIE_G23]
    g23 = 0.5 * (1.0 - eta) * r23[0] + 0.5 * (1.0 + eta) * r23[1]
    if scheme == "mitc4_g23":
        g13 = quad_ops_general(X, e1m, e2m, e3m, xi, eta, k22, cross, ax, kg)[4][0:1, :]
    else:                                                  # 'mitc4_both'
        # gamma_13 (row 0): tied at (+-1,0), linear in xi
        r13 = [quad_ops_general(X, e1m, e2m, e3m, tx, te, k22, cross, ax, kg)[4][0:1, :]
               for (tx, te) in _TIE_G13]
        g13 = 0.5 * (1.0 - xi) * r13[0] + 0.5 * (1.0 + xi) * r13[1]
    return np.vstack([g13, g23])


# -------------------------------------------------------------------- assembly
def assemble_segment_general(nodes, quads, subdom, e1s, e2s, e3s, D_by, G_by,
                             k22_e, cross=(1, 2), dof_map=None, shear="mitc4_both",
                             kg_e=None):
    """Assemble the six MSG blocks with the GENERAL RM taper operators:
        Dhh = <Gh' C Gh>   Dhe = <Gh' C Ge>   Dee = <Ge' C Ge>
        Dhl = <Gh' C Gl>   Dll = <Gl' C Gl>   Dle = <Gl' C Ge>
    each block carrying BOTH the classical (6x6 ABD, D) and the transverse-shear
    (2x2, G) energies; the shear h-rows are MITC-tied.

    dof_map (optional): node -> dof-node index.  Mapping a prismatic strip's top
    node row onto its bottom row makes the fields SPAN-INVARIANT, i.e. the exact
    1-D cross-section SG of the general operator (used for the boundary rings so
    boundary and segment share ONE parametrization)."""
    ax = [j for j in range(3) if j not in cross][0]
    if dof_map is None:
        dof_map = np.arange(len(nodes))
    Nn = int(np.max(dof_map)) + 1; ndof = NDOF * Nn
    Dhh = np.zeros((ndof, ndof)); Dhe = np.zeros((ndof, 4)); Dee = np.zeros((4, 4))
    Dhl = np.zeros((ndof, ndof)); Dll = np.zeros((ndof, ndof)); Dle = np.zeros((ndof, 4))
    gp = [(-1 / np.sqrt(3), -1 / np.sqrt(3)), (1 / np.sqrt(3), -1 / np.sqrt(3)),
          (1 / np.sqrt(3), 1 / np.sqrt(3)), (-1 / np.sqrt(3), 1 / np.sqrt(3))]
    for q, quad in enumerate(quads):
        X = nodes[quad]; k22 = float(k22_e[q])
        kg = float(kg_e[q]) if kg_e is not None else 0.0
        D = D_by[int(subdom[q])] if not isinstance(D_by, dict) or int(subdom[q]) in D_by else D_by[subdom[q]]
        G = G_SHEAR_SCALE * G_by[int(subdom[q])]
        g = np.concatenate([[NDOF * int(dof_map[n]) + c for c in range(NDOF)] for n in quad])
        gij = (g[:, None], g[None, :])
        for (xi, eta) in gp:
            BDe, BDh, BDl, BGe, BGh, BGl, dA = quad_ops_general(
                X, e1s[q], e2s[q], e3s[q], xi, eta, k22, cross, ax, kg)
            BGt = _mitc_shear_general(X, e1s[q], e2s[q], e3s[q], xi, eta, k22, cross, ax, shear, kg)
            w = dA  # unit gauss weights on [-1,1]^2 (2x2)
            # np.add.at (NOT fancy-index +=): with a dof_map the index vector g
            # contains REPEATED dofs (wrapped strip) and fancy-index += silently
            # drops duplicate contributions.
            np.add.at(Dhh, gij, (BDh.T @ D @ BDh + BGt.T @ G @ BGt) * w)
            np.add.at(Dhe, g, (BDh.T @ D @ BDe + BGt.T @ G @ BGe) * w)
            Dee += (BDe.T @ D @ BDe + BGe.T @ G @ BGe) * w
            np.add.at(Dhl, gij, (BDh.T @ D @ BDl + BGt.T @ G @ BGl) * w)
            np.add.at(Dll, gij, (BDl.T @ D @ BDl + BGl.T @ G @ BGl) * w)
            np.add.at(Dle, g, (BDl.T @ D @ BDe + BGl.T @ G @ BGe) * w)
    return Dhh, Dhe, Dee, Dhl, Dll, Dle


# ---------------------------------------------------- general-consistent RING SG
def ring_general(rx, rcells, rsub, re3, D_by, G_by, k22_edge, ax, cross, h=None,
                 shear="mitc4_both"):
    """Boundary cross-section SG solved with the SAME general operator as the
    segment (one parametrization end-to-end): the ring is extruded into a
    one-quad-deep PRISMATIC strip whose top node row is DOF-MAPPED onto the
    bottom row -- fields are then exactly span-invariant, i.e. the operator's
    own prismatic (eq:prism) reduction.  Returns C6 (6,6), V0 (5m,4), V1 (5m,4).
    """
    from scipy.sparse import coo_matrix
    import jax.numpy as jnp
    from .fe_jax.msg_solver import (solve_fluctuation_field,
                                              prepare_v1_rhs, finalize_v1_and_compute_deff)
    from .fe_jax.msg_rm_timo import build_C_Psi
    import pypardiso
    m = len(rx)
    if h is None:                                       # strip depth ~ hoop spacing
        h = float(np.mean(np.linalg.norm(rx[rcells[:, 1]] - rx[rcells[:, 0]], axis=1)))
    ez = np.zeros(3); ez[ax] = 1.0
    nodes = np.vstack([rx, rx + h * ez])                # (2m,3) prismatic extrusion
    dof_map = np.concatenate([np.arange(m), np.arange(m)])   # top row == bottom row
    quads = np.array([[a, b, m + b, m + a] for a, b in rcells], dtype=int)
    e3q = np.asarray(re3)
    e1q = np.tile(ez, (len(quads), 1)); e2q = e3q       # e1m/e2m unused by the op
    Dhh, Dhe, Dee, Dhl, Dll, Dle = assemble_segment_general(
        nodes, quads, rsub, e1q, e2q, e3q, D_by, G_by, np.asarray(k22_edge), cross,
        dof_map=dof_map, shear=shear)
    Dhh, Dhe, Dee, Dhl, Dll, Dle = [np.asarray(M) / h for M in (Dhh, Dhe, Dee, Dhl, Dll, Dle)]
    C, Psi = build_C_Psi(rx[:, cross], rcells, p=1)     # 4 rigid modes / constraints
    # GENERAL-op rigid twist: omega_beta are GLOBAL components, so the twist
    # kernel carries om_1 = +1 (rotation about the beam axis); the validated
    # code's kernel uses om_1 = -1 (its own internal convention) -> flip.
    Psi[3::NDOF, 3] *= -1.0
    Dc = C.T
    V0, D1, A_aug = solve_fluctuation_field(coo_matrix(Dhh), -Dhe, Dc)
    Deff = Dee + np.asarray(D1)
    bb, DhlV0, DhlTV0Dle, V0DllV0 = prepare_v1_rhs(
        jnp.array(V0), jnp.array(Dhl), jnp.array(Dll), jnp.array(Dle),
        jnp.array(Psi), jnp.array(Dc))
    n = Dhh.shape[0]
    R_v1 = np.concatenate([np.asarray(bb), np.zeros((4, np.asarray(bb).shape[1]))], axis=0)
    V_aug = pypardiso.spsolve(A_aug, R_v1)
    C6, *_ = finalize_v1_and_compute_deff(
        jnp.array(V_aug[:n, :]), jnp.array(V0), jnp.array(Deff),
        V0DllV0, DhlV0, DhlTV0Dle, jnp.array(Psi), jnp.array(Dc))
    return np.asarray(C6), np.asarray(V0), np.asarray(V_aug[:n, :])
