"""
segment_element.py     [ Windows opensg_2_0_env ]
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
"""

import numpy as np
from collections import defaultdict
from .fe_jax.msg_rm import _macro_BD


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
