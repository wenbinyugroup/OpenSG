"""Shell element operators and assembly for the 3D-SG shell segment: the
production 6-DOF drilling element, the 5-DOF general taper element, the
prismatic MITC4 element, MITC assumed-strain tying, k22/kg initial-curvature
corrections, Dirichlet factorizations, and the six-block MSG energy assembly.
Merged from: segment_element.py, segment_element_general.py, segment_indep.py.

========================================================================
Section 1 -- segment_element.py
========================================================================
PART 2: the 2-D MITC4 Reissner-Mindlin SHELL element for the 3D-SG segment.

Design principle (verified against the FEniCS shell operators gamma_h/gamma_l
and the validated 1-D RM msg_rm_timo):
  * DOF/node = [w1,w2,w3, omega1,omega2]  (5), bilinear quad -> 20 DOF/elem.
  * Gamma_h (BDq) = the FULL surface gradient of the warping in the local frame
    (axial d/dx1 = grad.e1, hoop d/ds = grad.e2), with RM rotation-based bending.
    When the field is span-invariant (d/dx1 -> 0) BDq reduces EXACTLY to the 1-D
    BDq -- that is the prismatic self-check.
  * transverse shear (BGq rows [2eps13, 2eps23]) uses MITC4 tying
    (Dvorkin-Bathe / Chapelle-Bathe sec 8.2):
        gamma23 (xi-shear)  tied at (xi=0, eta=+/-1), linear in eta;
        gamma13 (eta-shear) tied at (xi=+/-1, eta=0), linear in xi.
  * Gamma_l (BLq) = the 1-D cross-section shear-warping operator (== FEniCS
    gamma_l for this frame).
  * macro map Gamma_e = _macro_BD (reused from msg_rm).

The segment is solved with the boundary rings' warping imposed as DIRICHLET BCs
(not periodicity): rigid-body modes are fixed by the boundary, exactly like the
FEniCS compute_stiffness.  For a PRISMATIC cylinder the interior warping must be
span-invariant, i.e. equal to the ring warping at every axial station -- the
acceptance gate you asked for.

========================================================================
Section 2 -- segment_element_general.py
========================================================================
GENERAL 3D-taper Reissner-Mindlin surface operators

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

========================================================================
Section 3 -- segment_indep.py
========================================================================
Six-DOF shell segment element with an independent drilling rotation omega_3.

Per-node DOFs are [w1, w2, w3, om1, om2, om3].  The general operator
(segment_element_general) ELIMINATES the drilling omega_3 via
  omega_3 = S/(2 C33) - (C3b/C33) omega_b ,   C33 = n.b3 ,
which is singular on flat walls where C33 = n.b3 == 0.  Here omega_3 is a
genuine nodal DOF and appears DIRECTLY -- no 1/C33 anywhere:

  curvature:  Lambda_a = omega_3|a + x_{1;a} omega_3'
  shear:      2g13 += C23 om3 = x_{3;2} om3 ,  2g23 += -C13 om3 = -x_{3;1} om3
  membrane:   unchanged (no omega_3)

The in-plane symmetry that DEFINED omega_3 is re-imposed in its FINITE (undivided)
form via the drilling residual
  DR = C33 om3 + C3b om_b - S/2   (= C33 (om3 - om3_eliminated), finite even at C33=0),
  S/2 = 1/2[ k1(x11 Rn2 - x12 Rn1) + w'_i(x11 x_{i;2}-x12 x_{i;1}) + (w_{i|1}x_{i;2}-w_{i|2}x_{i;1}) ],
enforced as a penalty (assemble_segment_indep) or a Lagrange-multiplier
constraint (assemble_constraint).  On healthy walls (C33~1) this pins om3 to its
eliminated value; where C33=0 the residual constrains om_b instead and om3 is
set by its own curvature stiffness -- no singularity.

Shear scheme constraint: with the independent omega_3, both transverse-shear
rows carry algebraic drilling content that Dvorkin-Bathe assumed-strain tying
ALIASES -- 'full' (untied) integration is the default; tied variants are ablation.

Boundary: reuse the validated 5-DOF rings (ring_general) for dofs [0..4]; om3
(dof 5) is left FREE at the boundary (natural BC).

Public entry points: assemble_segment_indep, assemble_constraint,
build_C_Psi_segment6, quad_ops_indep.
"""

import numpy as np
from collections import defaultdict
from .fe_jax.msg_rm import _macro_BD


# =========================================================================
# from segment_element.py
# =========================================================================


def compute_curvatures(centroids, e1s, e2s, e3s, elems):
    """Full initial-curvature tensor per element, Yu thesis eq 5.10, by finite-
    differencing the flat-facet frame across neighbours (0 on flat facets):
      k_ab = b3,a . b_b / A_a  and geodesic k_a3 = b1,a . b2 / A_a, discretely
      axial  (neighbour along e1):  k11=de3.e1/dx1, k12=de3.e2/dx1, k13=de1.e2/dx1
      hoop   (neighbour along e2):  k21=de3.e1/ds,  k22=de3.e2/ds,  k23=de1.e2/ds
    Returns dict {k11,k12,k21,k22,k13,k23} of per-element arrays (Ne,).
    For a prismatic circle -> (0,0,0,-1/R,0,0)."""
    node2el = defaultdict(list)
    for ei, el in enumerate(elems):
        for n in el:
            node2el[int(n)].append(ei)
    centroids = np.asarray(centroids); e1s = np.asarray(e1s); e2s = np.asarray(e2s); e3s = np.asarray(e3s)
    ne = len(elems)
    K = {k: np.zeros(ne) for k in ("k11", "k12", "k21", "k22", "k13", "k23")}
    for ei in range(ne):
        neigh = {ej for n in elems[ei] for ej in node2el[int(n)] if ej != ei}
        ax, hp = [], []
        for ej in neigh:
            disp = centroids[ej] - centroids[ei]; nd = float(np.linalg.norm(disp))
            if nd < 1e-12:
                continue
            d1 = float(np.dot(disp, e1s[ei])); d2 = float(np.dot(disp, e2s[ei]))
            de3, de1 = e3s[ej] - e3s[ei], e1s[ej] - e1s[ei]
            rec = (float(np.dot(de3, e1s[ei])), float(np.dot(de3, e2s[ei])), float(np.dot(de1, e2s[ei])))
            if abs(d1) > 0.5 * nd:                             # axial-direction neighbour
                ax.append((d1,) + rec)
            elif abs(d2) > 0.5 * nd:                           # hoop-direction neighbour
                hp.append((d2,) + rec)
        if ax:
            K["k11"][ei] = np.mean([a[1] / a[0] for a in ax])
            K["k12"][ei] = np.mean([a[2] / a[0] for a in ax])
            K["k13"][ei] = np.mean([a[3] / a[0] for a in ax])
        if hp:
            K["k21"][ei] = np.mean([h[1] / h[0] for h in hp])
            K["k22"][ei] = np.mean([h[2] / h[0] for h in hp])
            K["k23"][ei] = np.mean([h[3] / h[0] for h in hp])
    return K


def compute_k22(centroids, e2s, e3s, elems, flat_tol=1e-3, k22_max=None, cop_tol=0.5):
    """Per-element hoop curvature k22 = d e3/ds . e2 (== -1/R for a circle with inward
    e3), estimated from hoop-aligned neighbours -- elements sharing a node whose
    centroid displacement is mostly along e2.  Works for the 1-D boundary contour
    (line elements) and the 2-D segment (quads); junction/isolated elements -> 0.

    Guards (the initial-curvature terms must never explode):
      - COPLANARITY filter (cop_tol): only neighbours whose e3 is within ~60 deg of
        this element's e3 count.  Smooth curvature is a SMALL angle between adjacent
        normals (well under 30 deg even on a coarse circle), so this keeps the real
        curvature; a sharp FOLD (square corner, web junction) has near-perpendicular
        normals (e3.e3 ~ 0) and is a discontinuity, NOT curvature -- counting it would
        inject a spurious curvature spike that breaks section symmetry (square GA2!=GA3)
        and blows up the boundary-ring torsion.  A fold's mechanics live in the mesh
        connectivity, not a k22 term;
      - per-neighbour estimates combined with the MEDIAN (robust to a stray branch);
      - FLAT elements snap to EXACTLY zero: |k22| < flat_tol (1e-3 ~ R > 1 km);
      - optional |k22| <= k22_max cap for sharp trailing-edge spikes."""
    node2el = defaultdict(list)
    for ei, el in enumerate(elems):
        for n in el:
            node2el[int(n)].append(ei)
    centroids = np.asarray(centroids); e2s = np.asarray(e2s); e3s = np.asarray(e3s)
    k22 = np.zeros(len(elems))
    for ei in range(len(elems)):
        neigh = {ej for n in elems[ei] for ej in node2el[int(n)] if ej != ei}
        ks = []
        for ej in neigh:
            if float(np.dot(e3s[ej], e3s[ei])) < cop_tol:  # non-coplanar fold -> not curvature
                continue
            disp = centroids[ej] - centroids[ei]; nd = float(np.linalg.norm(disp))
            if nd < 1e-12:
                continue
            ds = float(np.dot(disp, e2s[ei]))
            if abs(ds) < 0.5 * nd:                         # keep only hoop-direction neighbours
                continue
            ks.append(float(np.dot(e3s[ej] - e3s[ei], e2s[ei])) / ds)
        k = float(np.median(ks)) if ks else 0.0
        if abs(k) < flat_tol:
            k = 0.0                                        # flat element: exactly zero
        elif k22_max is not None:
            k = float(np.clip(k, -k22_max, k22_max))
        k22[ei] = k
    return k22


def compute_kg(centroids, e1s, e2s, e3s, elems, flat_tol=1e-3, cop_tol=0.5):
    """Per-element GEODESIC curvature of the hoop coordinate line:
    kg = d(e1)/ds . e2 -- the in-surface rotation rate of the axial tangent along
    the hoop.  Zero for a prismatic tube and on any FLAT wall; ~ (taper rate)/R on
    a tapered curved wall (the hoop circles of a cone are not surface geodesics).
    Same neighbour estimate/guards as compute_k22 (hoop-aligned neighbours,
    coplanarity filter, median, flat snap-to-zero)."""
    node2el = defaultdict(list)
    for ei, el in enumerate(elems):
        for n in el:
            node2el[int(n)].append(ei)
    centroids = np.asarray(centroids); e1s = np.asarray(e1s)
    e2s = np.asarray(e2s); e3s = np.asarray(e3s)
    kg = np.zeros(len(elems))
    for ei in range(len(elems)):
        neigh = {ej for n in elems[ei] for ej in node2el[int(n)] if ej != ei}
        ks = []
        for ej in neigh:
            if float(np.dot(e3s[ej], e3s[ei])) < cop_tol:  # non-coplanar fold
                continue
            disp = centroids[ej] - centroids[ei]; nd = float(np.linalg.norm(disp))
            if nd < 1e-12:
                continue
            ds = float(np.dot(disp, e2s[ei]))
            if abs(ds) < 0.5 * nd:                         # hoop-direction neighbours only
                continue
            ks.append(float(np.dot(e1s[ej] - e1s[ei], e2s[ei])) / ds)
        k = float(np.median(ks)) if ks else 0.0
        kg[ei] = 0.0 if abs(k) < flat_tol else k
    return kg


# --------------------------------------------------------------- quad kinematics
def _bilinear(xi, eta):
    """4-node bilinear N and parametric derivatives at (xi,eta) in [-1,1]^2.
    Corner order matches the mesh winding [ (j,k),(j,k+1),(j+1,k+1),(j+1,k) ] ->
    reference corners [(-1,-1),(1,-1),(1,1),(-1,1)] (xi ~ hoop, eta ~ axial)."""
    xc = np.array([-1., 1., 1., -1.]); ec = np.array([-1., -1., 1., 1.])
    N = 0.25 * (1 + xc * xi) * (1 + ec * eta)
    dNx = 0.25 * xc * (1 + ec * eta)
    dNe = 0.25 * ec * (1 + xc * xi)
    return N, dNx, dNe


def _quad_ops(X, e1, e2, e3, xi, eta, k22, cross=(1, 2), full_curvature=False):
    """Return BDq(6,20), BGq(2,20), BLq(6,20) and geometry at (xi,eta).
    `cross` = the two cross-section coordinate indices (beam axis is the third):
    (1,2) for axis=x (cylinder), (0,1) for axis=z (BAR-URC).

    X (4,3) node coords; e1/e2/e3 unit frame.  In-frame derivatives d/dx1 (axial,
    along e1) and d/ds (hoop, along e2) are obtained from the 2x2 metric
    G = [[e1.Jxi, e1.Jeta],[e2.Jxi, e2.Jeta]] :  [d/dx1; d/ds] = G^-1 [dN/dxi; dN/deta].
    dA = |Jxi x Jeta|.

    NOTE (RM has NO second derivative): the curvature rows use ONLY first derivatives
    of the rotation fluctuations omega1,omega2 (that is what makes RM a SECOND-ORDER
    variational problem, vs Kirchhoff's fourth-order).  The appendix Lambda_alpha
    (omega_3-drilling) term would need a SECOND derivative of w -- because omega_3 is
    itself a first-derivative combination of w that is then differentiated -- which is
    the Kirchhoff-ELIMINATION path.  RM avoids it by keeping omega1,omega2 as
    independent DOFs and simply dropping the omega_3 drilling contribution (it is
    epsilon^2-smaller).  The `full_curvature` argument is retained for API
    compatibility but is a NO-OP: there is no second derivative to add in RM.
    """
    N, dNx, dNe = _bilinear(xi, eta)
    Jxi = dNx @ X                       # dX/dxi (3,)
    Jeta = dNe @ X                      # dX/deta (3,)
    # chain rule: [df/dxi; df/deta] = G^T [df/dx1; df/ds]  with
    # G = [[e1.Jxi, e1.Jeta],[e2.Jxi, e2.Jeta]]  ->  [df/dx1; df/ds] = (G^T)^-1 [df/dxi; df/deta].
    # (Using G^-1 instead of (G^T)^-1 swaps the hoop/axial lengths for a non-square element.)
    G = np.array([[e1 @ Jxi, e1 @ Jeta],
                  [e2 @ Jxi, e2 @ Jeta]])
    # per-node in-frame derivatives: rows = nodes, cols = [d/dx1, d/ds]
    d = (np.linalg.inv(G.T) @ np.vstack([dNx, dNe])).T   # (4,2): d[a,0]=dNa/dx1, d[a,1]=dNa/ds
    dA = np.linalg.norm(np.cross(Jxi, Jeta))
    x = N @ X
    x2, x3 = x[cross[0]], x[cross[1]]              # cross-section coords
    t2, t3 = e2[cross[0]], e2[cross[1]]            # hoop tangent (in-plane) -- as in 1-D
    n2, n3 = t3, -t2                               # 1-D in-plane normal convention

    BDq = np.zeros((6, 20)); BGq = np.zeros((2, 20)); BLq = np.zeros((6, 20))
    for a in range(4):
        o = 5 * a
        D1, Ds, Na = d[a, 0], d[a, 1], N[a]       # d/dx1, d/ds, shape value
        # --- Gamma_h : membrane (full surface gradient) ---
        BDq[0, o+0] += D1                                   # eps11 = dw1/dx1        (axial; ->0 span-inv)
        BDq[1, o+1] += t2*Ds; BDq[1, o+2] += t3*Ds          # eps22 = d(t.w)/ds
        BDq[2, o+0] += Ds                                   # 2eps12 = dw1/ds
        BDq[2, o+1] += t2*D1; BDq[2, o+2] += t3*D1          # 2eps12 += d(t.w)/dx1   (axial; ->0)
        # --- Gamma_h : bending (RM, FIRST derivative of rotations -- no 2nd derivative) ---
        BDq[3, o+4] += D1                                   # k11 = domega2/dx1      (axial; ->0)
        BDq[4, o+3] += Ds                                   # k22 = domega1/ds
        BDq[5, o+4] += Ds; BDq[5, o+0] += 0.5*k22*Ds        # 2k12 = domega2/ds + 0.5 k22 dw1/ds
        BDq[5, o+3] += D1                                   # 2k12 += domega1/dx1    (axial; ->0)
        # --- transverse shear (pre-tying; MITC4 assembles the tied form) ---
        BGq[0, o+4] += Na                                   # 2eps13 = omega2
        BGq[0, o+1] += n2*D1; BGq[0, o+2] += n3*D1          # 2eps13 += d(n.w)/dx1   (axial; ->0)
        BGq[1, o+1] += n2*Ds; BGq[1, o+2] += n3*Ds; BGq[1, o+3] += -Na  # 2eps23 = d(n.w)/ds - omega1
        # --- Gamma_l : shear-warping surrogate (== 1-D BLq / FEniCS gamma_l) ---
        BLq[0, o+0] += Na
        BLq[2, o+1] += t2*Na; BLq[2, o+2] += t3*Na
        BLq[5, o+1] += 2*t3*Ds - 0.5*k22*t2*Na
        BLq[5, o+2] += -2*t2*Ds - 0.5*k22*t3*Na
    return BDq, BGq, BLq, (x2, x3, t2, t3, dA)


# tying points (Dvorkin-Bathe MITC4): row1 (gamma23, xi-shear) at (0,-1),(0,+1);
#                                      row0 (gamma13, eta-shear) at (-1,0),(+1,0)
_TIE = {"g23": [(0.0, -1.0), (0.0, 1.0)], "g13": [(-1.0, 0.0), (1.0, 0.0)]}


def _mitc_shear(X, e1, e2, e3, xi, eta, k22, cross=(1, 2)):
    """Assumed (tied) transverse-shear 2x20 operator BGb at (xi,eta)."""
    (A23), (B23) = [_quad_ops(X, e1, e2, e3, tx, te, k22, cross)[1][1:2, :] for (tx, te) in _TIE["g23"]]
    (A13), (B13) = [_quad_ops(X, e1, e2, e3, tx, te, k22, cross)[1][0:1, :] for (tx, te) in _TIE["g13"]]
    g23 = 0.5*(1.0 - eta)*A23 + 0.5*(1.0 + eta)*B23      # linear in eta
    g13 = 0.5*(1.0 - xi)*A13 + 0.5*(1.0 + xi)*B13        # linear in xi
    return np.vstack([g13, g23])


# ------------------------------------------------------------------- assembly
def assemble_segment(nodes, quads, subdom, e1s, e2s, e3s, D_by, G_by, k22_e, cross=(1, 2),
                     full_curvature=False):
    """Assemble Dhh, Dhe, Dee, Dhl, Dll, Dle for the 2-D quad segment.

    nodes (Nn,3), quads (Ne,4) 0-based, subdom (Ne,), e{1,2,3}s (Ne,3),
    D_by/G_by keyed by subdomain (layup) id, k22_e = per-QUAD hoop curvature (Ne,),
    cross = cross-section coord indices.  2x2 Gauss on the D/Gamma_l energy; the
    transverse-shear G-energy uses MITC4-tied BGb at the same points.
    """
    Nn = len(nodes); ndof = 5 * Nn
    Dhh = np.zeros((ndof, ndof)); Dhe = np.zeros((ndof, 4)); Dee = np.zeros((4, 4))
    Dhl = np.zeros((ndof, ndof)); Dll = np.zeros((ndof, ndof)); Dle = np.zeros((ndof, 4))
    gp = 1.0 / np.sqrt(3.0)
    quad_pts = [(-gp, -gp), (gp, -gp), (gp, gp), (-gp, gp)]
    for e, quad in enumerate(quads):
        X = nodes[quad]                                     # (4,3)
        e1, e2, e3 = e1s[e], e2s[e], e3s[e]
        D = D_by[int(subdom[e])]; G = G_by[int(subdom[e])]; k22 = float(k22_e[e])
        g = np.concatenate([[5*n, 5*n+1, 5*n+2, 5*n+3, 5*n+4] for n in quad])
        for (xi, eta) in quad_pts:
            BDq, BGq, BLq, geo = _quad_ops(X, e1, e2, e3, xi, eta, k22, cross, full_curvature)
            x2, x3, t2, t3, dA = geo
            BGb = _mitc_shear(X, e1, e2, e3, xi, eta, k22, cross)
            BDe = _macro_BD(x2, x3, t2, t3, k22)
            BGe = np.zeros((2, 4))
            Dhh[np.ix_(g, g)] += (BDq.T @ D @ BDq + BGb.T @ G @ BGb) * dA
            Dhe[g] += (BDq.T @ D @ BDe + BGb.T @ G @ BGe) * dA
            Dee += (BDe.T @ D @ BDe + BGe.T @ G @ BGe) * dA
            Dhl[np.ix_(g, g)] += BDq.T @ D @ BLq * dA
            Dll[np.ix_(g, g)] += BLq.T @ D @ BLq * dA
            Dle[g] += BLq.T @ D @ BDe * dA
    return Dhh, Dhe, Dee, Dhl, Dll, Dle


# --------------------------------------- rigid kernel + constraints (segment EB)
def build_C_Psi_segment(nodes, quads, cross=(1, 2)):
    """4 rigid-body modes (3 translations + twist) and the conjugate <.>=0
    constraints, integrated over the 2-D segment area -- the 2-D analogue of
    msg_rm_timo.build_C_Psi, so the element-agnostic msg_solver KKT solve applies
    unchanged.  Psi twist uses the node's cross-section coords (y,z)."""
    Nn = len(nodes); ndof = 5 * Nn
    C = np.zeros((4, ndof)); Psi = np.zeros((ndof, 4))
    gp = 1.0 / np.sqrt(3.0)
    qp = [(-gp, -gp), (gp, -gp), (gp, gp), (-gp, gp)]
    for quad in quads:
        X = nodes[quad]
        for (xi, eta) in qp:
            N, dNx, dNe = _bilinear(xi, eta)
            dA = np.linalg.norm(np.cross(dNx @ X, dNe @ X))
            for a, nd in enumerate(quad):
                for c in range(4):                    # <w1>=<w2>=<w3>=<omega1>=0
                    C[c, 5 * nd + c] += N[a] * dA
    for nd in range(Nn):
        y2, y3 = nodes[nd, cross[0]], nodes[nd, cross[1]]
        Psi[5*nd+0, 0] = 1.0                          # w1 translation
        Psi[5*nd+1, 1] = 1.0                          # w2 translation
        Psi[5*nd+2, 2] = 1.0                          # w3 translation
        Psi[5*nd+1, 3] = -y3; Psi[5*nd+2, 3] = y2; Psi[5*nd+3, 3] = -1.0   # twist
    return C, Psi


# ---------------------------------------------- Dirichlet (boundary) segment solve
def dirichlet_solve(K, RHS, bdofs, bvals):
    """Solve K u = RHS with u[bdofs] = bvals (columns = load cases).

    Partitioned: K_ii u_i = RHS_i - K_ib bvals ; u[bdofs]=bvals.  Returns u (ndof,nc).
    """
    ndof = K.shape[0]; nc = RHS.shape[1]
    free = np.setdiff1d(np.arange(ndof), bdofs)
    u = np.zeros((ndof, nc)); u[bdofs] = bvals
    Kii = K[np.ix_(free, free)]
    rhs = RHS[free] - K[np.ix_(free, bdofs)] @ bvals
    u[free] = np.linalg.solve(Kii, rhs)
    return u


def dirichlet_factor(K, bdofs):
    """Factorize the interior block of dirichlet_solve once for reuse across load
    sets sharing the SAME boundary dof set (the V0 and V1 solves use the same K
    and bdofs; two full dense factorizations are one too many).  Same LAPACK
    getrf/getrs path as np.linalg.solve."""
    from scipy.linalg import lu_factor
    ndof = K.shape[0]
    free = np.setdiff1d(np.arange(ndof), bdofs)
    return dict(lu=lu_factor(K[np.ix_(free, free)]),
                free=free, Kib=K[np.ix_(free, bdofs)], ndof=ndof)


def dirichlet_solve_fac(fac, RHS, bdofs, bvals):
    """dirichlet_solve on a dirichlet_factor() context (back-substitution only)."""
    from scipy.linalg import lu_solve
    nc = RHS.shape[1]
    u = np.zeros((fac["ndof"], nc)); u[bdofs] = bvals
    u[fac["free"]] = lu_solve(fac["lu"], RHS[fac["free"]] - fac["Kib"] @ bvals)
    return u


def dirichlet_factor_sparse(K, bdofs):
    """Sparse analogue of dirichlet_factor: factorize the free-free block of a
    scipy-sparse K once with SuperLU, for the V0/V1 solves at large DOF where the
    dense free block would not fit in memory.

    SuperLU (partial pivoting) is the SINGLE solver used for ALL sections, skin or
    webbed.  PARDISO/pypardiso is deliberately NOT used: its static pivoting silently
    returns a wrong factorization on the ill-conditioned reduced block at a web/skin
    T-junction (verified -- it even made iterative refinement diverge), and the mesh
    does not tell us in advance whether a section has webs.  SuperLU is correct for
    every mesh."""
    from scipy.sparse.linalg import splu
    ndof = K.shape[0]
    free = np.setdiff1d(np.arange(ndof), bdofs)
    Kcsr = K.tocsr()
    Kii = Kcsr[free][:, free].tocsc()
    Kib = Kcsr[free][:, bdofs]
    return dict(kind="splu", lu=splu(Kii), Kib=Kib, free=free, ndof=ndof)


def dirichlet_solve_sparse(fac, RHS, bdofs, bvals):
    """dirichlet_solve on a dirichlet_factor_sparse() context (back-substitution).

    The PARDISO path is wrapped in Python-level iterative refinement: PARDISO's static-
    pivoting factorization is only APPROXIMATE on the ill-conditioned reduced block from a
    web/skin T-junction, so a single solve is wrong (the "sparse broken on webs" bug).
    Refining with the same (fast) factorization drives the residual to machine level, giving
    ONE robust+fast solver for skin and webbed sections alike.  (If the block were truly
    singular the residual would stall -- it does not; it converges.)"""
    nc = RHS.shape[1]
    u = np.zeros((fac["ndof"], nc)); u[bdofs] = bvals
    rhs = np.asarray(RHS)[fac["free"]] - fac["Kib"] @ bvals
    if fac["kind"] == "pardiso":
        u[fac["free"]] = fac["solver"].solve(fac["Kii"], rhs)
    else:
        u[fac["free"]] = fac["lu"].solve(rhs)
    return u


# =========================================================================
# from segment_element_general.py
# =========================================================================

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


# =========================================================================
# from segment_indep.py
# =========================================================================

NDOF6 = 6


def quad_ops_indep(X, e3m, xi, eta, k22, cross, ax, kg=0.0):
    """Evaluate the 8-strain 6-DOF operators and drilling-residual rows at one
    parent point of one element.

    In:
      X: (4,3) float, element node coordinates.
      e3m: (3,) float, reference material normal (orients the surface frame).
      xi, eta: float, parent coordinates in [-1,1].
      k22: float, unused here (signature parity with the general element).
      cross: (2,) int, cross-section coordinate indices.
      ax: int, beam-axis coordinate index.
      kg: float, unused here (signature parity).
    Out:
      BDe (6,4), BDh (6,24), BDl (6,24): membrane/curvature operators on the
        beam strains eb, fluctuation w_s, and axial derivative w_s'.
      BGe (2,4), BGh (2,24), BGl (2,24): transverse-shear operators.
      DRe (4,), DRh (24,), DRl (24,): drilling-residual rows.
      dA: float, area Jacobian.
    """
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
    # twist column: -(x11 Rn2 - x12 Rn1)/2 == +(x2 y2 + x3 y3)/2 (direction-cosine identity); kept in Rn form
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


def _mitc_shear_indep(X, e3m, xi, eta, k22, cross, ax, kg=0.0, scheme="mitc4_g23"):
    """Tied (assumed-strain) transverse-shear BGh rows at (xi, eta).

    With the INDEPENDENT drilling omega_3 the gamma_13 row is ALGEBRAIC in omega_3
    on flat walls, so naive tying de-penalizes the director (hourglass).
    Schemes:
      'mitc4_wonly' : tie BOTH rows at the Dvorkin-Bathe points but keep every
                      ROTATION column (om1,om2,om3) at its full-integration value.
      'mitc4_g23'   : tie only the gamma_23 row (prismatic-element analogue).
      'mitc4_both'  : naive full tying (ablation only -- aliases the drilling shear).

    In:
      X: (4,3) float, element node coordinates.
      e3m: (3,) float, reference material normal.
      xi, eta: float, parent coordinates in [-1,1].
      k22, kg: float, passed through to quad_ops_indep (unused there).
      cross: (2,) int, cross-section coordinate indices.
      ax: int, beam-axis coordinate index.
      scheme: str, 'mitc4_wonly' | 'mitc4_g23' | 'mitc4_both'.
    Out:
      BGt: (2, 24) tied transverse-shear operator rows on w_s.
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
# The scalar quad_ops_indep above is kept for external diagnostics/verification.


def _surf_frame_batch(Xe, e3e, xi, eta, cross, ax):
    """Batched _surf_frame: surface frame, surface derivatives, and direction
    cosines for all elements at one parent point (same algebra per element).

    In:
      Xe: (ne,4,3) float, element node coordinates.
      e3e: (ne,3) float, reference material normals (fix the frame sign).
      xi, eta: float, parent coordinates in [-1,1].
      cross: (2,) int, cross-section coordinate indices.
      ax: int, beam-axis coordinate index.
    Out:
      N: (4,) shape functions; D1, D2: (ne,4) surface derivatives;
      dA: (ne,) area Jacobians; c: dict of (ne,) direction-cosine arrays
        (x11..x32, y1..y3, x2, x3).
    """
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
    """Batched quad_ops_indep: same operator rows for all elements at once
    (k22/kg do not enter these operators).

    In:
      Xe: (ne,4,3) float, element node coordinates.
      e3e: (ne,3) float, reference material normals.
      xi, eta: float, parent coordinates in [-1,1].
      cross: (2,) int, cross-section coordinate indices.
      ax: int, beam-axis coordinate index.
    Out:
      BDe (ne,6,4), BDh/BDl (ne,6,24), BGe (ne,2,4), BGh/BGl (ne,2,24),
      DRe (ne,4), DRh/DRl (ne,24), dA (ne,).
    """
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

    # ---- Gamma_h (fluctuation): DOF blocks (ne, row, node a, dof) -> (ne, row, 24)
    B = np.zeros((ne, 6, 4, NDOF6))
    B[:, 0, :, 0:3] = D1[:, :, None] * xi1[:, None, :]                      # eps11 = X_i1 w_i,1
    B[:, 1, :, 0:3] = D2[:, :, None] * xi2[:, None, :]                      # eps22 = X_i2 w_i,2
    B[:, 2, :, 0:3] = D2[:, :, None] * xi1[:, None, :] + D1[:, :, None] * xi2[:, None, :]   # 2eps12 = X_i1 w_i,2 + X_i2 w_i,1
    B[:, 3, :, 3:6] = D1[:, :, None] * xi2[:, None, :]                      # K11 = X_i2 om_i,1
    B[:, 4, :, 3:6] = -D2[:, :, None] * xi1[:, None, :]                     # K22 = -X_i1 om_i,2
    B[:, 5, :, 3:6] = D2[:, :, None] * xi2[:, None, :] - D1[:, :, None] * xi1[:, None, :]   # K12+K21 = X_i2 om_i,2 - X_i1 om_i,1
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
    Bg[:, 0, :, 0:3] = D1[:, :, None] * yv[:, None, :]                      # 2g13 = C_3i w_i,1 ...
    Bg[:, 1, :, 0:3] = D2[:, :, None] * yv[:, None, :]                      # 2g23 = C_3i w_i,2 ...
    Bg[:, 0, :, 3:6] = N[None, :, None] * xi2[:, None, :]                   # ... + X_i2 om_i
    Bg[:, 1, :, 3:6] = N[None, :, None] * (-xi1)[:, None, :]                # ... - X_i1 om_i
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
    """Evaluate the shear operator rows at the four Dvorkin-Bathe tying points,
    once per element set (they do not depend on the quadrature point).

    In:
      Xe: (ne,4,3) float, element node coordinates.
      e3e: (ne,3) float, reference material normals.
      cross: (2,) int, cross-section coordinate indices.
      ax: int, beam-axis coordinate index.
    Out:
      dict with keys 'g13m','g13p','g23m','g23p', each (ne,1,24): BGh row 0
      at (-1,0)/(+1,0) and row 1 at (0,-1)/(0,+1).
    """
    return {
        "g13m": quad_ops_indep_batch(Xe, e3e, -1.0, 0.0, cross, ax)[4][:, 0:1, :],
        "g13p": quad_ops_indep_batch(Xe, e3e, 1.0, 0.0, cross, ax)[4][:, 0:1, :],
        "g23m": quad_ops_indep_batch(Xe, e3e, 0.0, -1.0, cross, ax)[4][:, 1:2, :],
        "g23p": quad_ops_indep_batch(Xe, e3e, 0.0, 1.0, cross, ax)[4][:, 1:2, :],
    }


def _shear_batch(xi, eta, scheme, BGh_gauss, tie):
    """Batched tied transverse-shear rows at (xi, eta); mirrors _mitc_shear_indep.

    In:
      xi, eta: float, parent coordinates in [-1,1].
      scheme: str, 'mitc4_wonly' | 'mitc4_g23' | 'mitc4_both'.
      BGh_gauss: (ne,2,24) untied shear rows at this Gauss point.
      tie: dict from _tie_rows_batch.
    Out:
      BGt: (ne,2,24) tied shear rows; for 'mitc4_wonly' the rotation columns
      stay at their full-integration values.
    """
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
    """Characteristic ABD stiffness magnitude: max |diag(D)| over all layups.

    In:
      D_by: dict or sequence of (6,6) ABD matrices, keyed/indexed by subdomain.
    Out:
      float, max absolute diagonal entry over all layups.
    Note: the drilling penalty is set to beta*this so it is dimensionally
    commensurate with the elastic stiffness (avoids penalty ill-conditioning).
    """
    keys = D_by.keys() if isinstance(D_by, dict) else range(len(D_by))
    return max(float(np.max(np.abs(np.diag(np.asarray(D_by[k]))))) for k in keys)


def assemble_segment_indep(nodes, quads, subdom, e3s, D_by, G_by, k22_e, cross, ax,
                           kg_e=None, pen=None, pen_beta=0.1, dof_map=None,
                           shear="full", sparse=False):
    """Assemble the 6-DOF segment energy blocks with the finite drilling-residual
    penalty pen*DR^2.

    In:
      nodes: (Nn,3) float, node coordinates.
      quads: (ne,4) int, element connectivity.
      subdom: (ne,) int, subdomain (layup) id per element.
      e3s: (ne,3) float, reference material normals.
      D_by: {subdomain: (6,6)} ABD matrices.
      G_by: {subdomain: (2,2)} transverse-shear stiffnesses.
      k22_e: (ne,) float, unused in the batched operators (signature parity).
      cross: (2,) int, cross-section coordinate indices.
      ax: int, beam-axis coordinate index.
      kg_e: optional (ne,) float, unused (signature parity).
      pen: float, drilling penalty weight; default pen_beta * max|diag(D)|.
      pen_beta: float, penalty scale relative to the ABD magnitude.
      dof_map: optional (Nn,) int, node -> dof-node index (wrapped strips).
      shear: 'full' (default) | 'mitc4_wonly' | 'mitc4_g23' | 'mitc4_both'.
      sparse: bool, return CSR for the square ndof x ndof blocks.
    Out:
      Dhh (ndof,ndof), Dhe (ndof,4), Dee (4,4), Dhl (ndof,ndof),
      Dll (ndof,ndof), Dle (ndof,4), with ndof = 6*(max(dof_map)+1);
      square blocks CSR when sparse=True.
    Note: with the independent omega_3, Dvorkin-Bathe tying aliases the algebraic
    drilling shear -- keep shear='full'; tied variants are ablation only.
    """
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
        if pen != 0.0:
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
        else:
            Ehh += w * (np.einsum('eia,eib->eab', BDh, DB) + np.einsum('eia,eib->eab', BGt, GB))
            Ehe += w * (np.einsum('eia,eib->eab', BDh, DBe) + np.einsum('eia,eib->eab', BGt, GBe))
            Dee += np.einsum('e,eab->ab', dA,
                             np.einsum('eia,eib->eab', BDe, DBe) + np.einsum('eia,eib->eab', BGe, GBe))
            Ehl += w * (np.einsum('eia,eib->eab', BDh, DBl) + np.einsum('eia,eib->eab', BGt, GBl))
            Ell += w * (np.einsum('eia,eib->eab', BDl, DBl) + np.einsum('eia,eib->eab', BGl, GBl))
            Ele += w * (np.einsum('eia,eib->eab', BDl, DBe) + np.einsum('eia,eib->eab', BGl, GBe))

    Dhe = np.zeros((ndof, 4)); Dle = np.zeros((ndof, 4))
    np.add.at(Dhe, g.reshape(-1), Ehe.reshape(-1, 4))
    np.add.at(Dle, g.reshape(-1), Ele.reshape(-1, 4))
    if sparse:
        # COO triplets from per-element 24x24 blocks (dense infeasible at ~1e6 DOF); duplicates summed by tocsr(); Dhe/Dle stay dense (thin)
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
      <DR>_e = 0 -- the inf-sup-stable choice (equal-order nodal multipliers
      over-constrain under refinement: classical LBB failure).
    lam_space='node': one multiplier per node, <N_a DR> = 0 (ablation only).
    lam_space='elem_nofold': element-constant multipliers, EXCLUDING elements
      adjacent to a FOLD line (incident-element normals disagree > 30 deg),
      where the C0-shared fields cannot satisfy both walls' symmetry rows.

    In:
      nodes: (Nn,3) float, node coordinates.
      quads: (nq,4) int, element connectivity.
      subdom: (nq,) int, subdomain ids (unused here; signature parity).
      e3s: (nq,3) float, reference material normals.
      k22_e: (nq,) float, per-element k22 (scalar 'node' path only).
      cross: (2,) int, cross-section coordinate indices.
      ax: int, beam-axis coordinate index.
      kg_e: optional (nq,) float, per-element kg (scalar 'node' path only).
      lam_space: 'elem' | 'node' | 'elem_nofold'.
      dof_map: optional (Nn,) int, node -> dof-node index (wrapped prismatic
        strip); repeated dofs require np.add.at (fancy-index += drops duplicates).
      sparse: bool, return CSR G/Gl.
    Out:
      G (P,6Nd) on w_s, Gl (P,6Nd) on w_s', Ge (P,4) on eb, where P is the
      number of multiplier rows and Nd = max(dof_map)+1.
    """
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
        # batched: element-integrated DR rows for all elements at once
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
    """Build the rigid-body kernel constraint rows C and null-space modes Psi for
    the 6-DOF segment (om3 column = 0: drilling is not a rigid mode).

    In:
      nodes: (Nn,3) float, node coordinates.
      quads: (ne,4) int, element connectivity.
      cross: (2,) int, cross-section coordinate indices.
    Out:
      C: (4, 6*Nn) area-weighted constraint rows on dofs [w1,w2,w3,om1].
      Psi: (6*Nn, 4) rigid modes (3 translations + axial rotation).
    """
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
