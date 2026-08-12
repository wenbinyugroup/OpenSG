"""Reusable kernels for the native tri3 / MITC3 element tests.

Everything here is built ONLY from the shipped element operators
(``sg_assembly.tri_frame_batch`` / ``mitc3_shear_batch`` and
``sg_homo.solid_fluct_ops_tri_batch``) -- no re-implementation of the element,
so a defect in the element shows up as a failing test rather than being masked
by a parallel implementation.

Contents
  frames / local_xy      the element's own orthonormal surface frame (a1,a2,n)
                         and node coordinates in it
  kirchhoff_state        nodal (w, om) of a constant-membrane + constant-curvature
                         + IDENTICALLY-ZERO-transverse-shear state
  const_shear_state      nodal (w, om) of a constant transverse shear state
  rigid_state            nodal (w, om) of a rigid translation + infinitesimal rotation
  tri_Ke / quad_Ke       the element matrix, assembled exactly as
                         sg_homo.shell_sg3d assembles its triangle / quad batch:
                         plain 18x18 (24x24), or bordered 19x19 (25x25) with the
                         element's own drilling LAGRANGE multiplier row
                         g = int_e DR dA (drill='lagrange')
  patch_constrained_spectrum   zero-energy modes of a small free patch under
                         that constraint (eigenvalues of Z^T K Z, Z = null(G))
  iso_plate_law          isotropic Reissner-Mindlin wall law (ABD 6x6 + G 2x2)
  ALU_DE / ALU_G         the alu wall law of
                         examples/OpenSG_shell/4_get_solid_props_from_shell_3D_SG
  ALU_ISO_DE/ALU_ISO_G   the same layup rebuilt EXACTLY isotropic (the .out print
                         carries only 8 digits, which is a 2.3e-8 relative
                         in-plane anisotropy -- enough to swamp a machine-precision
                         isotropy gate)
  square_tri_mesh /      the self-contained triangle-meshed square plate benchmark
  plate_center_deflection

In:  every routine takes batched element data Xe (ne,3,3) and e3e (ne,3), the
     same layout the production assembly uses, with cross=[1,2], ax=0.
Out: NumPy arrays; nothing is written to disk.
"""
import os
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))          # repo root
if os.path.join(_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))

import opensg_shell                                        # noqa: E402  (sets x64)
from opensg_shell.sg_assembly import (TIE_MITC3, TRI_GPTS,  # noqa: E402
                                      TRI_NDOF, _shear_batch,
                                      mitc3_shear_batch, tri_frame_batch)
from opensg_shell.sg_homo import (solid_fluct_ops_batch,     # noqa: E402
                                  solid_fluct_ops_tri_batch,
                                  _tie_rows_solid4)

CROSS, AX = [1, 2], 0        # exactly what shell_sg3d passes

# --------------------------------------------------------------- geometry ---
# five deliberately distorted reference triangles, incl. an obtuse one and a
# sliver of aspect ratio 250
TRI_SHAPES = {
    "equilateral":  [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, 0.8660254037844386, 0.0)],
    "right-iso":    [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
    "obtuse-143deg": [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (-0.60, 0.35, 0.0)],
    "sliver-ar250": [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.40, 0.004, 0.0)],
    "skew-scaled":  [(0.0, 0.0, 0.0), (2.5, 0.0, 0.0), (2.2, 0.9, 0.0)],
}
TRI_NAMES = list(TRI_SHAPES)


def rotation(axis, angle):
    """Proper rotation matrix (Rodrigues).

    In:  axis: (3,) float, angle: float radians.  Out: (3,3) float."""
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


# a generic pose: no axis of the element is aligned with a global axis
Q_GENERIC = rotation([0.31, -0.77, 0.55], 0.9137)
SHIFT_GENERIC = np.array([3.1, -2.4, 7.7])


def tri_batch(tilt=True):
    """The five reference triangles as one batch.

    In:  tilt: bool, apply Q_GENERIC + SHIFT_GENERIC so no element is axis-aligned.
    Out: names (list), Xe (5,3,3), e3e (5,3)."""
    Xe = np.array([TRI_SHAPES[k] for k in TRI_NAMES], float)
    nrm = np.array([0.0, 0.0, 1.0])
    if tilt:
        Xe = np.einsum('ij,eaj->eai', Q_GENERIC, Xe) + SHIFT_GENERIC
        nrm = Q_GENERIC @ nrm
    return TRI_NAMES, Xe, np.tile(nrm, (len(TRI_NAMES), 1))


def frames(Xe, e3e):
    """The element's own orthonormal surface frame.

    In:  Xe (ne,3,3), e3e (ne,3).
    Out: a1, a2, n each (ne,3) in GLOBAL components (cross=[1,2], ax=0 makes the
         direction-cosine dict an identity relabelling), dA (ne,), Gm metric."""
    _, _, _, dA, c, Gm = tri_frame_batch(Xe, e3e, 1.0 / 3.0, 1.0 / 3.0, CROSS, AX)
    a1 = np.stack([c["x11"], c["x21"], c["x31"]], 1)
    a2 = np.stack([c["x12"], c["x22"], c["x32"]], 1)
    nn = np.stack([c["y1"], c["y2"], c["y3"]], 1)
    return a1, a2, nn, dA, Gm


def local_xy(Xe, a1, a2):
    """Node coordinates in the element frame, origin at node 1.  Out: x1, x2 (ne,3)."""
    d = Xe - Xe[:, 0:1, :]
    return np.einsum('eai,ei->ea', d, a1), np.einsum('eai,ei->ea', d, a2)


def _pack(w, om):
    """(ne,3,3) + (ne,3,3) -> (ne,18) in the element DOF order w1 w2 w3 om1 om2 om3."""
    return np.concatenate([w, om], axis=2).reshape(w.shape[0], TRI_NDOF)


# ------------------------------------------------------------ test states ---
def kirchhoff_state(Xe, e3e, e11, e22, g12, k11, k22, k12x2,
                    w_rigid=None, th_rigid=None):
    """Nodal (w, om) of the exact constant-membrane + constant-curvature state
    whose transverse shear vanishes IDENTICALLY.

    The operator rows fix the convention.  With phi1 := om.a2 and phi2 := -om.a1,
        B[3] : K11    =  d(om.a2)/dx1 = phi1,1
        B[4] : K22    = -d(om.a1)/dx2 = phi2,2
        B[5] : 2*K12  =  d(om.a2)/dx2 - d(om.a1)/dx1 = phi1,2 + phi2,1
        Bg[0]: 2*g13  = (w.n),1 + om.a2 = u3,1 + phi1
        Bg[1]: 2*g23  = (w.n),2 - om.a1 = u3,2 + phi2
    so gamma = 0 forces phi = -grad u3 and the deflection is QUADRATIC:
        u3   = -1/2 (k11 x1^2 + k22 x2^2 + 2*K12 x1 x2)
        phi1 =  k11 x1 + K12 x2 ,   phi2 = k22 x2 + K12 x1
    (k12x2 is the sixth strain component K12+K21 = 2*K12.)  The tri3 interpolates
    u3 LINEARLY, so the displacement-based shear of this state is NOT zero -- that
    is precisely the locking mechanism; the MITC3 assumed field must annihilate it.

    In:  Xe (ne,3,3), e3e (ne,3), six imposed constants, optional rigid add-ons.
    Out: (ne,18) nodal DOF vector."""
    a1, a2, nn, _, _ = frames(Xe, e3e)
    x1, x2 = local_xy(Xe, a1, a2)
    u1 = e11 * x1 + 0.5 * g12 * x2
    u2 = 0.5 * g12 * x1 + e22 * x2
    u3 = -0.5 * (k11 * x1**2 + k22 * x2**2 + k12x2 * x1 * x2)
    ph1 = k11 * x1 + 0.5 * k12x2 * x2
    ph2 = k22 * x2 + 0.5 * k12x2 * x1
    w = (u1[..., None] * a1[:, None, :] + u2[..., None] * a2[:, None, :]
         + u3[..., None] * nn[:, None, :])
    om = (-ph2)[..., None] * a1[:, None, :] + ph1[..., None] * a2[:, None, :]
    if w_rigid is not None:
        w = w + np.asarray(w_rigid, float)
    if th_rigid is not None:
        th = np.broadcast_to(np.asarray(th_rigid, float), Xe.shape)
        w = w + np.cross(th, Xe - Xe[:, 0:1, :])
        om = om + th
    return _pack(w, om)


def const_shear_state(Xe, e3e, g13, g23, slope=(0.7, -0.4)):
    """Nodal (w, om) of a constant transverse shear state (zero membrane, zero
    curvature).  u3 linear with slope `slope`, phi constant and chosen so that
    2*gamma_a3 = u3,a + phi_a hits the target.

    In:  Xe (ne,3,3), e3e (ne,3), g13, g23 targets for the rows [2g13, 2g23].
    Out: (ne,18)."""
    a1, a2, nn, _, _ = frames(Xe, e3e)
    a, b = slope
    x1, x2 = local_xy(Xe, a1, a2)
    u3 = a * x1 + b * x2
    ph1, ph2 = g13 - a, g23 - b
    w = u3[..., None] * nn[:, None, :]
    om = np.broadcast_to((-ph2) * a1[:, None, :] + ph1 * a2[:, None, :],
                         w.shape).copy()
    return _pack(w, om)


def rigid_state(Xe, trans, theta):
    """Nodal (w, om) of a rigid translation + infinitesimal rotation.

    In:  Xe (ne,3,3), trans (3,), theta (3,).  Out: (ne,18)."""
    th = np.broadcast_to(np.asarray(theta, float), Xe.shape)
    w = np.broadcast_to(np.asarray(trans, float), Xe.shape) \
        + np.cross(th, Xe - Xe[:, 0:1, :])
    return _pack(w, th)


def rigid_basis(X):
    """(18,6) rigid-body basis of ONE triangle with corner coordinates X (3,3)."""
    R = np.zeros((18, 6))
    for a in range(3):
        for k in range(3):
            R[6 * a + k, k] = 1.0
            e = np.zeros(3)
            e[k] = 1.0
            R[6 * a:6 * a + 3, 3 + k] = np.cross(e, X[a] - X[0])
            R[6 * a + 3:6 * a + 6, 3 + k] = e
    return R


# ---------------------------------------------------------------- operators -
def shear_rows(Xe, e3e, Gm, r, s, scheme):
    """Transverse-shear rows at (r,s) under `scheme` ('full' | 'mitc3')."""
    if scheme == "full":
        return solid_fluct_ops_tri_batch(Xe, e3e, r, s, CROSS, AX)[1]
    g = [solid_fluct_ops_tri_batch(Xe, e3e, tr, ts, CROSS, AX)[1]
         for (tr, ts) in TIE_MITC3]
    return mitc3_shear_batch(Gm, g[0], g[1], g[2], r, s)


def _border(Ke, g):
    """Border a batch of element stiffnesses with their own drilling
    multiplier: [[Ke, g], [g^T, 0]], one extra row/column per element.

    This IS the element-level object shell_sg3d assembles now -- the drilling
    omega_3 is pinned by an element-wise LAGRANGE MULTIPLIER on
    g = int_e DR dA, not by a penalty.  A penalty would have to choose a
    weight, and the weight multiplies a row that used to carry 1/C33; the
    bordered form is invariant under any nonzero scaling of g (the multiplier
    absorbs it), which is exactly why it is frame-objective.

    In:  Ke (ne,n,n), g (ne,n).   Out: (ne,n+1,n+1)."""
    ne, n, _ = Ke.shape
    Kb = np.zeros((ne, n + 1, n + 1))
    Kb[:, :n, :n] = Ke
    Kb[:, :n, n] = g
    Kb[:, n, :n] = g
    return Kb


def tri_Ke(Xe, e3e, De, Gmat, scheme="mitc3", drill=None):
    """The triangle element matrix, assembled exactly as shell_sg3d's tri batch
    (3-point degree-2 rule, hoisted MITC3 tying).

    In:  Xe (ne,3,3), e3e (ne,3), De (6,6), Gmat (2,2), scheme;
         drill: None -> the plain (ne,18,18) elastic stiffness;
                'lagrange' -> the (ne,19,19) matrix bordered with the element's
                own drilling multiplier row g = int_e DR dA.
    Out: (ne,18,18) or (ne,19,19)."""
    tie = None
    if scheme != "full":
        tie = [solid_fluct_ops_tri_batch(Xe, e3e, tr, ts, CROSS, AX)[1]
               for (tr, ts) in TIE_MITC3]
    Ke = np.zeros((Xe.shape[0], TRI_NDOF, TRI_NDOF))
    G = np.zeros((Xe.shape[0], TRI_NDOF))
    for (r, s, wq) in TRI_GPTS:
        B, Bg, Dr, dJ, Gmet = solid_fluct_ops_tri_batch(Xe, e3e, r, s, CROSS, AX)
        if tie is not None:
            Bg = mitc3_shear_batch(Gmet, tie[0], tie[1], tie[2], r, s)
        dA = wq * dJ
        w = dA[:, None, None]
        Ke += (np.einsum('eia,ij,ejb->eab', B, De, B) * w
               + np.einsum('eia,ij,ejb->eab', Bg, Gmat, Bg) * w)
        G += Dr * dA[:, None]
    if drill is None:
        return Ke
    if drill != "lagrange":
        raise ValueError("drill must be None or 'lagrange', got %r" % (drill,))
    return _border(Ke, G)


def quad_Ke(Xe, e3e, De, Gmat, scheme="mitc", drill=None):
    """The QUAD twin of tri_Ke, assembled exactly as shell_sg3d's quad batch
    (2x2 Gauss, MITC4 with BOTH covariant rows tied).

    In:  Xe (ne,4,3), e3e (ne,3), De (6,6), Gmat (2,2);
         scheme: 'mitc' (Dvorkin-Bathe) | 'full';
         drill: None -> (ne,24,24); 'lagrange' -> (ne,25,25) bordered.
    Out: (ne,24,24) or (ne,25,25)."""
    tie = None if scheme == "full" else _tie_rows_solid4(Xe, e3e, CROSS, AX)
    ne = Xe.shape[0]
    Ke = np.zeros((ne, 4 * 6, 4 * 6))
    G = np.zeros((ne, 4 * 6))
    gpv = 1.0 / np.sqrt(3.0)
    for (xi, eta) in [(-gpv, -gpv), (gpv, -gpv), (gpv, gpv), (-gpv, gpv)]:
        B, Bg, Dr, dA = solid_fluct_ops_batch(Xe, e3e, xi, eta, CROSS, AX)
        if tie is not None:
            Bg = _shear_batch(xi, eta, "mitc4_both", Bg, tie)
        w = dA[:, None, None]
        Ke += (np.einsum('eia,ij,ejb->eab', B, De, B) * w
               + np.einsum('eia,ij,ejb->eab', Bg, Gmat, Bg) * w)
        G += Dr * dA[:, None]
    if drill is None:
        return Ke
    if drill != "lagrange":
        raise ValueError("drill must be None or 'lagrange', got %r" % (drill,))
    return _border(Ke, G)


def dof_rotation(Q, ndof=TRI_NDOF, lagrange=False):
    """Block-diagonal rotation of the element DOF vector: every 3-slot (w_a and
    om_a of each node) rotates with Q.

    In:  Q (3,3); ndof int (18 tri3 / 24 quad); lagrange bool -- append the
         multiplier slot, which is a SCALAR and rotates with 1.
    Out: (ndof,ndof) or (ndof+1,ndof+1)."""
    n = ndof + (1 if lagrange else 0)
    T = np.eye(n)
    for b in range(ndof // 3):
        T[3 * b:3 * b + 3, 3 * b:3 * b + 3] = Q
    return T


def quad_batch(tilt=True):
    """Four deliberately distorted reference quads (square, trapezoid, skew,
    WARPED out of plane) as one batch -- the quad counterpart of tri_batch.

    In:  tilt: bool, apply Q_GENERIC + SHIFT_GENERIC.
    Out: names (list), Xe (4,4,3), e3e (4,3).
    The warped element is the one the MITC4 tying aliasing was measured on; its
    reference normal is the mean facet normal."""
    shapes = {
        "square": [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
        "trapezoid": [(0, 0, 0), (1.4, 0, 0), (1.0, 0.9, 0), (0.2, 0.9, 0)],
        "skew": [(0, 0, 0), (1.0, 0, 0), (1.5, 0.8, 0), (0.4, 0.8, 0)],
        "warped-0.15": [(0, 0, -0.075), (1, 0, 0.075), (1, 1, -0.075),
                        (0, 1, 0.075)],
    }
    names = list(shapes)
    Xe = np.array([shapes[k] for k in names], float)
    nrm = np.tile([0.0, 0.0, 1.0], (len(names), 1))
    if tilt:
        Xe = np.einsum('ij,eaj->eai', Q_GENERIC, Xe) + SHIFT_GENERIC
        nrm = np.einsum('ij,ej->ei', Q_GENERIC, nrm)
    return names, Xe, nrm


def rigid_basis_n(X):
    """(6n,6) rigid-body basis of ONE element with n corner coordinates X (n,3)."""
    n = len(X)
    R = np.zeros((6 * n, 6))
    for a in range(n):
        for k in range(3):
            R[6 * a + k, k] = 1.0
            e = np.zeros(3)
            e[k] = 1.0
            R[6 * a:6 * a + 3, 3 + k] = np.cross(e, X[a] - X[0])
            R[6 * a + 3:6 * a + 6, 3 + k] = e
    return R


# ------------------------------------------------------------- wall laws ----
def iso_plate_law(E, nu, t, kappa=5.0 / 6.0):
    """Isotropic Reissner-Mindlin wall law on the package's strain ordering
    [e11 e22 2e12 | K11 K22 K12+K21] and [2g13 2g23].

    Out: De (6,6), Gmat (2,2), D = E t^3 / (12 (1-nu^2))."""
    G = E / (2.0 * (1.0 + nu))
    A = E * t / (1.0 - nu**2)
    D = E * t**3 / (12.0 * (1.0 - nu**2))
    De = np.zeros((6, 6))
    De[0, 0] = De[1, 1] = A
    De[0, 1] = De[1, 0] = nu * A
    De[2, 2] = 0.5 * (1.0 - nu) * A
    De[3, 3] = De[4, 4] = D
    De[3, 4] = De[4, 3] = nu * D
    De[5, 5] = 0.5 * (1.0 - nu) * D
    return De, np.diag([kappa * G * t, kappa * G * t]), D


# the alu wall law verbatim from
# examples/OpenSG_shell/4_get_solid_props_from_shell_3D_SG/schwarz_p_3Dshell_ABDG.out
ALU_DE = np.zeros((6, 6))
ALU_DE[0, 0] = ALU_DE[1, 1] = 2.7643220E+009
ALU_DE[0, 1] = ALU_DE[1, 0] = 8.2929659E+008
ALU_DE[2, 2] = 9.6751264E+008
ALU_DE[3, 3] = ALU_DE[4, 4] = 3.0617465E+005
ALU_DE[3, 4] = ALU_DE[4, 3] = 9.1852396E+004
ALU_DE[5, 5] = 1.0716112E+005
ALU_G = np.diag([8.5193106E+008, 8.5193106E+008])

# the SAME layup (alu, t = 0.036457, E = 69 GPa, nu = 0.3) rebuilt exactly
# isotropic; the .out print above is isotropic only to 2.3e-8 relative
ALU_E, ALU_NU, ALU_T = 69.0e9, 0.30, 0.036457
ALU_ISO_DE, _tmp_G, _tmp_D = iso_plate_law(ALU_E, ALU_NU, ALU_T)
ALU_ISO_G = np.diag([8.5193106E+008, 8.5193106E+008])   # MSG's own shear block


# ------------------------------------------------- square-plate benchmark ---
# Kirchhoff centre deflection of a uniformly loaded square plate, w = alpha q L^4 / D
KIRCHHOFF_ALPHA = {"ss": 0.00406235, "clamped": 0.00126532}


def square_tri_mesh(L, n):
    """n x n cells over [0,L]^2, each split into two triangles with the diagonal
    ALTERNATING so the mesh has no preferred direction.

    Out: nodes (Nn,3) in the z = 0 plane, conn (2 n^2, 3) 0-based."""
    xs = np.linspace(0.0, L, n + 1)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    nodes = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], 1)
    nid = lambda i, j: i * (n + 1) + j                       # noqa: E731
    conn = []
    for i in range(n):
        for j in range(n):
            a, b, c, d = nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)
            conn += ([[a, b, c], [a, c, d]] if (i + j) % 2 == 0
                     else [[a, b, d], [b, c, d]])
    return nodes, np.array(conn, int)


def plate_center_deflection(L, t, n, scheme, bc="ss", E=69.0e9, nu=0.3, q=1.0):
    """Self-contained triangle-meshed square-plate bending benchmark -- the only
    test of the property MITC3 is adopted for.

    A flat plate meshed with the NATIVE tri3 element: the membrane DOFs (w1, w2)
    and the drilling DOF (om3) are exactly decoupled from bending here (B = 0 and
    the surface normal is constant), so they are removed and what is solved is the
    pure Reissner-Mindlin plate.

    In:  L: float side; t: float thickness; n: int cells per side;
         scheme: 'full' | 'mitc3'; bc: 'ss' (soft simple support, w = 0 on the
         edge) or 'clamped' (w = 0 and both rotations = 0); E, nu, q.
    Out: (w_center, w_kirchhoff) -- the FE centre deflection and the thin-plate
         analytic value alpha q L^4 / D."""
    nodes, conn = square_tri_mesh(L, n)
    De, Gmat, D = iso_plate_law(E, nu, t)
    Xe = nodes[conn]
    e3e = np.tile([0.0, 0.0, 1.0], (len(conn), 1))
    Ke = tri_Ke(Xe, e3e, De, Gmat, scheme)
    ndof = 6 * len(nodes)
    gd = (6 * conn[:, :, None] + np.arange(6)[None, None, :]).reshape(-1, TRI_NDOF)
    K = sp.csr_matrix((Ke.ravel(), (np.repeat(gd, TRI_NDOF, 1).ravel(),
                                    np.tile(gd, (1, TRI_NDOF)).ravel())),
                      shape=(ndof, ndof))
    area = 0.5 * np.linalg.norm(np.cross(Xe[:, 1] - Xe[:, 0],
                                         Xe[:, 2] - Xe[:, 0]), axis=1)
    f = np.zeros(ndof)
    np.add.at(f, 6 * conn.ravel() + 2, np.repeat(q * area / 3.0, 3))
    fixed = np.zeros(ndof, bool)
    fixed[0::6] = fixed[1::6] = fixed[5::6] = True           # membrane + drilling
    tol = 1.0e-9 * L
    onb = ((np.abs(nodes[:, 0]) < tol) | (np.abs(nodes[:, 0] - L) < tol)
           | (np.abs(nodes[:, 1]) < tol) | (np.abs(nodes[:, 1] - L) < tol))
    bn = np.where(onb)[0]
    fixed[6 * bn + 2] = True
    if bc == "clamped":
        fixed[6 * bn + 3] = True
        fixed[6 * bn + 4] = True
    free = np.where(~fixed)[0]
    u = np.zeros(ndof)
    u[free] = spla.spsolve(K[free][:, free].tocsc(), f[free])
    ic = int(np.argmin(np.linalg.norm(nodes[:, :2] - 0.5 * L, axis=1)))
    return u[6 * ic + 2], KIRCHHOFF_ALPHA[bc] * q * L**4 / D


def square_quad_mesh(L, n):
    """n x n quad cells over [0,L]^2.  Out: nodes (Nn,3) at z=0, conn (n^2,4)."""
    xs = np.linspace(0.0, L, n + 1)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    nodes = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], 1)
    nid = lambda i, j: i * (n + 1) + j                       # noqa: E731
    conn = [[nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)]
            for i in range(n) for j in range(n)]
    return nodes, np.array(conn, int)


def patch_constrained_spectrum(nodes, conn, De, Gmat, scheme=None):
    """Zero-energy analysis of a FREE-FLOATING patch with the production
    drilling treatment: one element-constant Lagrange multiplier per element.

    The constrained problem minimizes v^T K v over {G v = 0}, so the modes to
    count are the eigenvalues of Z^T K Z with Z an orthonormal basis of
    null(G) -- NOT the eigenvalues of the bordered saddle matrix (which is
    indefinite by construction and whose spectrum says nothing about zero
    energy).

    In:  nodes (Nn,3) flat or curved patch; conn (ne,3) tri or (ne,4) quad;
         De (6,6), Gmat (2,2) wall laws; scheme: element shear scheme
         (default 'mitc3' for tri, 'mitc' for quad).  The reference normals are
         taken from the element geometry, so a FOLDED/curved patch works.
    Out: dict(ev = eigenvalues of Z^T K Z sorted, nzero = count below
         1e-10*max, Z, K, G, n_drill = #multipliers, ndof).
    """
    conn = np.asarray(conn, int)
    nv = conn.shape[1]
    Xc = nodes[conn]
    e3e = np.cross(Xc[:, 1] - Xc[:, 0], Xc[:, -1] - Xc[:, 0])
    e3e /= np.linalg.norm(e3e, axis=1)[:, None]
    if nv == 3:
        Kb = tri_Ke(nodes[conn], e3e, De, Gmat, scheme or "mitc3",
                    drill="lagrange")
    else:
        Kb = quad_Ke(nodes[conn], e3e, De, Gmat, scheme or "mitc",
                     drill="lagrange")
    m = 6 * nv
    ndof = 6 * len(nodes)
    gd = (6 * conn[:, :, None] + np.arange(6)[None, None, :]).reshape(-1, m)
    K = np.zeros((ndof, ndof))
    G = np.zeros((len(conn), ndof))
    for e in range(len(conn)):
        np.add.at(K, (gd[e][:, None], gd[e][None, :]), Kb[e, :m, :m])
        np.add.at(G, (e, gd[e]), Kb[e, :m, m])
    Z = np.linalg.svd(G, full_matrices=True)[2][np.linalg.matrix_rank(
        G, tol=1e-10 * np.abs(G).max()):].T           # orthonormal null(G)
    Kc = Z.T @ K @ Z
    ev = np.linalg.eigvalsh(0.5 * (Kc + Kc.T))
    return dict(ev=ev, nzero=int((np.abs(ev) < 1e-10 * ev.max()).sum()),
                Z=Z, K=K, G=G, n_drill=len(conn), ndof=ndof)


def locking_table(ns=(4, 8, 16, 32), tls=(1e-2, 1e-3, 1e-4), bcs=("ss", "clamped")):
    """Print the normalized centre deflection w_FE / w_Kirchhoff for both schemes."""
    for bc in bcs:
        print("\n  %s   (Kirchhoff w_c = %.6f q L^4 / D)"
              % (bc.upper(), KIRCHHOFF_ALPHA[bc]))
        print("   %-8s %-5s | %-11s %-11s" % ("t/L", "n", "full", "mitc3"))
        for tl in tls:
            for n in ns:
                out = []
                for sch in ("full", "mitc3"):
                    wc, wk = plate_center_deflection(1.0, tl, n, sch, bc)
                    out.append(wc / wk)
                print("   %-8.0e %-5d | %-11.5f %-11.5f" % (tl, n, out[0], out[1]))


if __name__ == "__main__":
    print("tri3 / MITC3 locking curve -- square plate, uniform pressure")
    locking_table()
