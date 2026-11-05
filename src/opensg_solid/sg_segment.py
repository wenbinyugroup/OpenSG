"""sg_segment.py -- the solid segment route of the beam model.

A 3-D solid beam segment with aperiodic ends gives its Timoshenko 6x6,
its classical 4x4 and its mass matrix.  The segment is any piece of a
beam the user meshes: prismatic or tapered, from a CAD model or from a
generator.  sg_homo.plate_homo_2d hands a yaml with `n_model: 1` and
`aperiodic: 1` whose mesh is 3-D to segment_homo below.

THE CONTRACT
  axes      mesh z is the beam axis x1, mesh x is x2 and mesh y is x3,
            the 2-D beam SG convention (nodes [y2 y3 0]) extended to 3-D.
            The frame rows of `orientation:` use the 2-D beam SG spelling:
            each axis e1, e2, e3 lists its mesh components (x, y, z) =
            (x2, x3, x1), cycled to the solver order (x1, x2, x3) by
            sg_materials.elem_rotation_from_yaml.
  ends      the element faces that lie in the planes z = z_min and
            z = z_max: a hex8 face is a quad4, a tet4 face a tri3, a
            tet10 face a tri6, and each face keeps its parent's material
            and frame.
  elements  hex8, tet4, mixed hex8 + tet4, and tet10 (gmsh node order,
            for curved CAD geometry).  A tet10 mesh takes no linear
            elements.
  origin    kept as given.  The law is referred to the line x = y = 0 of
            the mesh (the mesh z axis): the strains of the 6x6 are the
            beam strains of that axis and the mass moments are taken
            about it.
  law       per unit length of the segment: the strain energy of the
            segment under the beam strains is divided by its length
            L = z_max - z_min.
  outputs   <base>.out in the 2-D beam route's layout (Timoshenko 6x6 for
            refined 1 or the Euler-Bernoulli 4x4 for refined 0, the
            compliance, the 6X6 mass matrix) and <base>_ends.out with
            each end section's Timoshenko 6x6, compliance and mass matrix.

THE ALGORITHM
  1. The two end sections are extracted from the 3-D mesh (the faces in
     the end planes, each with its parent's rotated stiffness).
  2. Each end section is solved as a 2-D beam SG (gamma_h of dimension
     2, the four rigid modes as a bordered KKT): its Timoshenko 6x6 and
     its fluctuations V0, V1.  The fluctuations are pinned as FEniCS
     OpenSG v1.2 and OpenSG's 2-D beam route pin them: zero area mean of
     the displacement and of the rotation (Eq. 85), the V1 load balanced
     by the same area constraints (Eq. 100).  The port pins them to the
     nodal rigid modes instead (gauge="nodal" keeps that for comparison).
     The end's own 6x6 does not depend on the gauge, a tapered segment
     does (1e-4 to 2e-2 of max|K| on the test segments).
  3. The 3-D segment (gamma_h of dimension 3) is solved with Dirichlet
     data on the end-face nodes equal to the end fluctuations: four V0
     cases and four V1 cases on one factorization.
  4. The Timoshenko reduction (B_tim, C_tim, Q_tim, the 6x6 reorder)
     with every energy term divided by L.

WHERE IT COMES FROM
  FEniCS OpenSG v1.2: opensg/core/solid.py (compute_stiffness with
  Taper=True, compute_timo_boun) and opensg/utils/solid.py
  (get_mass_solid with Taper=True).  The JAX port of that algorithm is
  opensg_jax/fe_jax/solid_taper.py in the OpenSG-TW-claude repository,
  checked there against FEniCS on identical meshes.  The element
  kernels, the end-section solve, the segment solve and the Timoshenko
  finalize below are that port's, with the math unchanged apart from the
  end gauge of step 2, which follows FEniCS.  The tet10
  and tri6 cells are added here: quadratic Lagrange simplices on a
  collapsed Gauss rule of degree 5, in the same kernels.  OpenSG's
  own pieces replace the port's where they are equivalent: the yaml
  reader and its frame handling (sg_mesh.load_sg_input,
  sg_materials.elem_rotation_from_yaml), the material stiffness and its
  rotation (sg_materials.get_heterogeneous_C_matrix, which also applies a
  material block's `angle:`), the .out writer (sg_homo.write_sc_K), the
  section mass of an end (sg_homo.mass_matrix_2d) and the mesh picture
  (sg_mesh.plot_sg_mesh).

THE MASS MATRIX
  get_mass_solid with Taper=True: mu = int rho dV, the first moments
  int rho x2 dV and int rho x3 dV, the second moments int rho x3^2,
  int rho x2^2 and int rho x2 x3 over the segment volume, all divided by
  L.  It is the mass matrix per unit length of the segment (its average
  over the length) about the line x = y = 0.  It is written in the VABS
  layout of sg_homo.mass_matrix_2d, with -i23 in the (5, 6) and (6, 5)
  entries.  FEniCS v1.2 writes +i23 there: the product of inertia
  differs in sign only, and it is zero for a section symmetric about
  either axis.  A material without a density counts as density 0, and
  a segment with no density gives the zero matrix with a note.

SOLVER
  SuperLU (scipy.sparse.linalg.splu) for the end sections and for the
  segment, as the port does.  The port records that pypardiso returned
  silently wrong digits on webbed T-junction reduced blocks of a shell
  segment, so this route never calls it.

Voigt order [11, 22, 33, 23, 13, 12] with axis 1 = the beam axis x1.
"""
import time

import numpy as np
import jax
import jax.numpy as jnp
from scipy.sparse import bmat, coo_matrix, csr_matrix
from scipy.sparse.linalg import splu

jax.config.update("jax_enable_x64", True)

# mesh (x, y, z) -> beam (x1, x2, x3) = (z, x, y)
_PERM3 = [2, 0, 1]

# 3-D element of each node count on this route, and the face it leaves
# on an end plane
SEGMENT_ELEMENTS = {8: "hex8", 4: "tet4", 10: "tet10"}
FACE_KIND = {"hex8": "quad4", "tet4": "tri3", "tet10": "tri6"}
# local faces (gmsh node order).  A face of 4 nodes is cyclic.  A tri6
# face lists its corners, then the midsides of its edges 01, 12 and 20.
# gmsh tet10 midsides: 4 on edge 01, 5 on 12, 6 on 02, 7 on 03, 8 on 23
# and 9 on 13.
_FACES = {"hex8": [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5),
                   (2, 3, 7, 6), (3, 0, 4, 7)],
          "tet4": [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)],
          "tet10": [(0, 1, 2, 4, 5, 6), (0, 1, 3, 4, 9, 7),
                    (0, 2, 3, 6, 8, 7), (1, 2, 3, 5, 8, 9)]}
_MIDS = {"tet10": [(0, 1), (1, 2), (0, 2), (0, 3), (2, 3), (1, 3)],
         "tri6": [(0, 1), (1, 2), (2, 0)]}


class SegmentInputError(ValueError):
    """A segment yaml the route cannot run (one-line reason)."""


# ============================================================ element rules
_S8 = np.array([(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)], float)
_G2 = np.array([-1.0, 1.0]) / np.sqrt(3.0)


def _rule(kind):
    """The quadrature and the shape functions of one cell type.

    In:  kind str -- hex8 | tet4 | quad4 | tri3
    Out: (xi (q, dim), w (q,), N function, dN/dxi function)."""
    if kind == "hex8":
        pts = np.array([(a, b, c) for a in _G2 for b in _G2 for c in _G2])
        w = np.ones(8)

        def N(x):
            return (0.125 * (1 + x[0] * _S8[:, 0]) * (1 + x[1] * _S8[:, 1])
                    * (1 + x[2] * _S8[:, 2]))

        def dN(x):
            return np.stack([
                0.125 * _S8[:, 0] * (1 + x[1] * _S8[:, 1])
                * (1 + x[2] * _S8[:, 2]),
                0.125 * _S8[:, 1] * (1 + x[0] * _S8[:, 0])
                * (1 + x[2] * _S8[:, 2]),
                0.125 * _S8[:, 2] * (1 + x[0] * _S8[:, 0])
                * (1 + x[1] * _S8[:, 1])], 0)
        return pts, w, N, dN
    if kind == "tet4":
        a, b = 0.5854101966249685, 0.1381966011250105
        pts = np.array([(b, b, b), (a, b, b), (b, a, b), (b, b, a)])
        w = np.full(4, 1.0 / 24.0)

        def N(x):
            return np.array([1 - x[0] - x[1] - x[2], x[0], x[1], x[2]])

        def dN(x):
            return np.array([[-1, 1, 0, 0], [-1, 0, 1, 0], [-1, 0, 0, 1]],
                            float)
        return pts, w, N, dN
    if kind == "quad4":
        pts = np.array([(a, b) for a in _G2 for b in _G2])
        w = np.ones(4)
        s = _S8[:4, :2]

        def N(x):
            return 0.25 * (1 + x[0] * s[:, 0]) * (1 + x[1] * s[:, 1])

        def dN(x):
            return np.stack([0.25 * s[:, 0] * (1 + x[1] * s[:, 1]),
                             0.25 * s[:, 1] * (1 + x[0] * s[:, 0])], 0)
        return pts, w, N, dN
    if kind == "tri3":
        pts = np.array([(1 / 6, 1 / 6), (2 / 3, 1 / 6), (1 / 6, 2 / 3)])
        w = np.full(3, 1.0 / 6.0)

        def N(x):
            return np.array([1 - x[0] - x[1], x[0], x[1]])

        def dN(x):
            return np.array([[-1, 1, 0], [-1, 0, 1]], float)
        return pts, w, N, dN
    if kind in ("tet10", "tri6"):
        dim = 3 if kind == "tet10" else 2
        pts, w = _collapsed_rule(dim)
        N, dN = _p2_simplex(dim, _MIDS[kind])
        return pts, w, N, dN
    raise KeyError(kind)


def _collapsed_rule(dim, n=3):
    """The collapsed (conical product) Gauss rule on the unit simplex:
    Gauss-Jacobi in the collapsed directions, Gauss-Legendre in the last
    one, n points each.  Exact for polynomials of degree 2 n - 1 = 5,
    which covers every integrand of a straight-sided tet10 or tri6 (degree
    4 at most).

    In:  dim 2 | 3, n int
    Out: (xi (n^dim, dim), w (n^dim,)) on {xi >= 0, sum xi <= 1}."""
    from scipy.special import roots_jacobi
    xl, wl = np.polynomial.legendre.leggauss(n)
    c, wc = (xl + 1.0) / 2.0, wl / 2.0
    if dim == 2:
        xa, wa = roots_jacobi(n, 1.0, 0.0)
        a, wa = (xa + 1.0) / 2.0, wa / 4.0
        pts = [(ai, (1.0 - ai) * ci) for ai in a for ci in c]
        w = [wai * wci for wai in wa for wci in wc]
        return np.array(pts), np.array(w)
    xa, wa = roots_jacobi(n, 2.0, 0.0)
    a, wa = (xa + 1.0) / 2.0, wa / 8.0
    xb, wb = roots_jacobi(n, 1.0, 0.0)
    b, wb = (xb + 1.0) / 2.0, wb / 4.0
    pts = [(ai, (1.0 - ai) * bi, (1.0 - ai) * (1.0 - bi) * ci)
           for ai in a for bi in b for ci in c]
    w = [wai * wbi * wci for wai in wa for wbi in wb for wci in wc]
    return np.array(pts), np.array(w)


def _p2_simplex(dim, mids):
    """The quadratic Lagrange simplex (tri6, tet10): corner functions
    L_i (2 L_i - 1), then 4 L_a L_b for each midside (a, b).

    In:  dim 2 | 3, mids list of corner pairs in the node order
    Out: (N function, dN/dxi function (dim, nn))."""
    dL = np.vstack([-np.ones(dim), np.eye(dim)])           # (dim + 1, dim)

    def bary(x):
        x = np.asarray(x, float)
        return np.concatenate([[1.0 - x.sum()], x])

    def N(x):
        L = bary(x)
        return np.array([L[i] * (2.0 * L[i] - 1.0) for i in range(dim + 1)]
                        + [4.0 * L[a] * L[b] for a, b in mids])

    def dN(x):
        L = bary(x)
        cols = ([(4.0 * L[i] - 1.0) * dL[i] for i in range(dim + 1)]
                + [4.0 * (L[b] * dL[a] + L[a] * dL[b]) for a, b in mids])
        return np.array(cols).T
    return N, dN


def _tables(kind):
    pts, w, N, dN = _rule(kind)
    return (jnp.asarray(np.stack([N(p) for p in pts])),       # (q, nn)
            jnp.asarray(np.stack([dN(p) for p in pts])),       # (q, dim, nn)
            jnp.asarray(w))


# =========================================================== element kernels
def _gamma_e(xq):
    """Gamma_e (6, 4) at the point xq = (x1, x2, x3): the FEniCS OpenSG
    v1.2 gamma_e (opensg/utils/solid.py)."""
    z = jnp.zeros(())
    o = jnp.ones(())
    return jnp.array([[o, z, xq[2], -xq[1]],
                      [z, z, z, z], [z, z, z, z], [z, z, z, z],
                      [z, xq[1], z, z],
                      [z, -xq[2], z, z]])


def _Bh(grads, nn, dim3):
    """The gamma_h B matrix (6, 3 nn) from the node gradients grads
    (3, nn) in beam axes.  dim3 True: the 3-D operator (the beam
    derivative rows on).  False: the end-section operator of dimension 2
    (row 0 zero, no beam derivative), as the FEniCS gamma_h."""
    B = jnp.zeros((6, 3 * nn))
    i = jnp.arange(nn)
    g0, g1, g2 = grads[0], grads[1], grads[2]
    if dim3:
        B = B.at[0, 3 * i + 0].set(g0)
        B = B.at[4, 3 * i + 0].add(g2).at[4, 3 * i + 2].add(g0)
        B = B.at[5, 3 * i + 0].add(g1).at[5, 3 * i + 1].add(g0)
    else:
        B = B.at[4, 3 * i + 0].set(g2)
        B = B.at[5, 3 * i + 0].set(g1)
    B = B.at[1, 3 * i + 1].set(g1)
    B = B.at[2, 3 * i + 2].set(g2)
    B = B.at[3, 3 * i + 1].add(g2).at[3, 3 * i + 2].add(g1)
    return B


def _Bl(shp, nn):
    """The gamma_l B matrix (6, 3 nn) from the shape values shp (nn,):
    rows [v0, 0, 0, 0, v2, v1]."""
    B = jnp.zeros((6, 3 * nn))
    i = jnp.arange(nn)
    B = B.at[0, 3 * i + 0].set(shp)
    B = B.at[4, 3 * i + 2].set(shp)
    B = B.at[5, 3 * i + 1].set(shp)
    return B


def _make_kernel(kind, dim3):
    """The jitted element kernel of one cell type: (xe (nn, 3) beam-axis
    coordinates, C (6, 6)) -> (Khh, Kll, Khl (3nn, 3nn), Khe, Kle (3nn, 4),
    Dee (4, 4)).  Khe carries the FEniCS minus sign."""
    Nq, dNq, Wq = _tables(kind)
    nn = Nq.shape[1]
    volume = kind in ("hex8", "tet4", "tet10")
    # the linear cells keep the port's unrolled sum over the points, the
    # quadratic ones (27 and 9 points) map over them
    unrolled = kind in ("hex8", "tet4", "quad4", "tri3")

    def kernel(xe, C):
        def qp(q):
            dN = dNq[q]                                   # (dim, nn)
            if volume:
                J = dN @ xe                               # (3, 3)
                detJ = jnp.abs(jnp.linalg.det(J))
                g = jnp.linalg.solve(J, dN)               # d/d(x1, x2, x3)
            else:
                J = dN @ xe[:, 1:3]                       # (2, 2) in-plane
                detJ = jnp.abs(jnp.linalg.det(J))
                g2 = jnp.linalg.solve(J, dN)              # d/d(x2, x3)
                g = jnp.concatenate([jnp.zeros((1, nn)), g2], 0)
            xq = Nq[q] @ xe                               # (3,)
            Bh = _Bh(g, nn, dim3)
            Bl = _Bl(Nq[q], nn)
            Ge = _gamma_e(xq)
            wd = Wq[q] * detJ
            CBh = C @ Bh
            CGe = C @ Ge
            return (wd * Bh.T @ CBh,                      # Khh
                    wd * Bl.T @ (C @ Bl),                 # Kll
                    wd * Bl.T @ CBh,                      # Khl: rows l, cols h
                    -wd * Bh.T @ CGe,                     # Khe (FEniCS minus)
                    wd * Bl.T @ CGe,                      # Kle
                    wd * Ge.T @ CGe)                      # Dee
        if unrolled:
            out = [qp(q) for q in range(Nq.shape[0])]
            return tuple(sum(o[k] for o in out) for k in range(6))
        out = jax.vmap(qp)(jnp.arange(Nq.shape[0]))
        return tuple(o.sum(0) for o in out)
    return jax.jit(jax.vmap(kernel))


_KERNELS = {}


def _kernel(kind, dim3):
    key = (kind, dim3)
    if key not in _KERNELS:
        _KERNELS[key] = _make_kernel(kind, dim3)
    return _KERNELS[key]


# ================================================================== assembly
def assemble_blocks(nodes, batches, dim3):
    """Every cell-type batch into one system of 3 dofs per node.

    In:  nodes (N, 3) beam-axis coordinates, batches {kind: (conn (n, nn)
         0-based, C (n, 6, 6) rotated stiffness, rho (n,), parent index
         (n,))}, dim3 bool
    Out: CSR Dhh, Dll, Dhl (3N, 3N), dense Dhe, Dle (3N, 4), Dee (4, 4).
         Each type is one vmapped batch and the COO triplets of all
         batches are summed under the one dof numbering."""
    N3 = 3 * len(nodes)
    rows_h, cols_h, vals_hh, vals_ll, vals_hl = [], [], [], [], []
    Dhe = np.zeros((N3, 4))
    Dle = np.zeros((N3, 4))
    Dee = np.zeros((4, 4))
    X = jnp.asarray(nodes)
    for kind, bat in batches.items():
        conn, C_e = bat[0], bat[1]
        xe = X[jnp.asarray(conn)]                          # (n, nn, 3)
        Khh, Kll, Khl, Khe, Kle, De = _kernel(kind, dim3)(
            xe, jnp.asarray(C_e))
        nn = conn.shape[1]
        edof = (3 * conn[:, :, None]
                + np.arange(3)[None, None, :]).reshape(len(conn), 3 * nn)
        R = np.repeat(edof[:, :, None], 3 * nn, axis=2).ravel()
        Cc = np.repeat(edof[:, None, :], 3 * nn, axis=1).ravel()
        rows_h.append(R)
        cols_h.append(Cc)
        vals_hh.append(np.asarray(Khh).ravel())
        vals_ll.append(np.asarray(Kll).ravel())
        vals_hl.append(np.asarray(Khl).ravel())
        np.add.at(Dhe, edof.ravel(), np.asarray(Khe).reshape(-1, 4))
        np.add.at(Dle, edof.ravel(), np.asarray(Kle).reshape(-1, 4))
        Dee += np.asarray(De).sum(0)
    R = np.concatenate(rows_h)
    Cc = np.concatenate(cols_h)

    def csr(vals):
        return coo_matrix((np.concatenate(vals), (R, Cc)),
                          shape=(N3, N3)).tocsr()
    return csr(vals_hh), csr(vals_ll), csr(vals_hl), Dhe, Dle, Dee


# ======================================================= Timoshenko finalize
def _finalize(D_eff, B_tim, C_tim):
    """The FEniCS Timoshenko finalize: Q_tim, G_tim, Y_tim, A_tim, then
    the 6x6 reorder [ext, sh2, sh3, twist, bend2, bend3]."""
    Ainv = np.linalg.inv(D_eff)
    Q = Ainv @ np.array([(0.0, 0.0), (0.0, 0.0), (0.0, -1.0), (1.0, 0.0)])
    Ginv = Q.T @ (C_tim - B_tim.T @ Ainv @ B_tim) @ Q
    G = np.linalg.inv(Ginv)
    Y = B_tim.T @ Q @ G
    A = D_eff + Y @ Ginv @ Y.T
    S = np.zeros((6, 6))
    S[0, 0] = A[0, 0]
    S[0, 1:3] = Y[0, :]
    S[0, 3:6] = A[0, 1:4]
    S[3:6, 3:6] = A[1:4, 1:4]
    S[3:6, 1:3] = Y[1:4, :]
    S[3:6, 0] = A[1:4, 0]
    S[1:3, 1:3] = G
    S[1:3, 3:6] = Y.T[:, 1:4]
    S[1:3, 0] = Y.T[:, 0]
    return S


def _factor(A, what):
    """SuperLU of A (CSC), with a one-line reason when it is singular."""
    try:
        return splu(A)
    except RuntimeError as e:
        raise SegmentInputError(
            "the %s system is singular (%s): check that the mesh is one "
            "connected solid and that every material is positive "
            "definite" % (what, e))


# ============================================================ end sections
def _end_gauge(end):
    """The rigid modes Psi (3M, 4) of an end section and its area
    constraints Dc (3M, 4): int phi dA on each translation, and
    int (dphi/dx3 v2 - dphi/dx2 v3) dA on the rotation.  These are the
    constraints of OpenSG's 2-D beam route (sg_homo.assemble_rigid_body_ops)
    and of FEniCS OpenSG v1.2 (its Eq. 85 and Eq. 100 projections)."""
    nodes = end["nodes"]
    N3 = 3 * len(nodes)
    Psi = np.zeros((N3, 4))
    Psi[0::3, 0] = 1.0
    Psi[1::3, 1] = 1.0
    Psi[2::3, 2] = 1.0
    Psi[1::3, 3] = -nodes[:, 2]
    Psi[2::3, 3] = nodes[:, 1]
    Dc = np.zeros((N3, 4))
    for kind, bat in end["batches"].items():
        conn = bat[0]
        pts, w, N, dN = _rule(kind)
        X = nodes[conn][:, :, 1:3]                          # (n, nn, 2)
        for p, wq in zip(pts, w):
            J = np.einsum("dn,enk->edk", dN(p), X)          # (n, 2, 2)
            da = wq * np.abs(np.linalg.det(J))
            g = np.linalg.solve(J, np.broadcast_to(dN(p), (len(conn),)
                                                   + dN(p).shape))
            phi = N(p)[None, :] * da[:, None]               # (n, nn)
            np.add.at(Dc[:, 0], 3 * conn, phi)
            np.add.at(Dc[:, 1], 3 * conn + 1, phi)
            np.add.at(Dc[:, 2], 3 * conn + 2, phi)
            np.add.at(Dc[:, 3], 3 * conn + 1, g[:, 1, :] * da[:, None])
            np.add.at(Dc[:, 3], 3 * conn + 2, -g[:, 0, :] * da[:, None])
    return Psi, Dc


def solve_end(end, gauge="area"):
    """One end section as a 2-D beam SG: gamma_h of dimension 2, the four
    rigid modes (three translations and the twist) as a bordered KKT on
    one SuperLU factorization.

    gauge picks the end fluctuations handed to the segment as Dirichlet
    data.  "area" (the default) is the gauge of FEniCS OpenSG v1.2 and of
    OpenSG's 2-D beam route: V0 and V1 with zero area mean of the
    displacement and of the rotation (Eq. 85), and the V1 load balanced
    by the area constraints (Eq. 100).  "nodal" is the port's: the
    fluctuations orthogonal to the nodal rigid modes.  The end's own 6x6
    does not depend on it.  A tapered segment does: the two gauges move
    its shear and bending terms by about 1e-4.

    In:  end dict from end_sections, gauge "area" | "nodal"
    Out: (Deff6 (6, 6) Timoshenko, D_eff (4, 4) classical, V0 (3M, 4),
         V1s (3M, 4))."""
    if gauge not in ("area", "nodal"):
        raise SegmentInputError("gauge must be area or nodal, got %r"
                                % (gauge,))
    nodes = end["nodes"]
    N3 = 3 * len(nodes)
    Dhh, Dll, Dhl, Dhe, Dle, Dee = assemble_blocks(nodes, end["batches"],
                                                   dim3=False)
    Rm = np.zeros((N3, 4))                                 # rigid modes
    Rm[0::3, 0] = 1.0
    Rm[1::3, 1] = 1.0
    Rm[2::3, 2] = 1.0
    Rm[1::3, 3] = -nodes[:, 2]                             # twist [0, -x3, x2]
    Rm[2::3, 3] = nodes[:, 1]
    Rm /= np.linalg.norm(Rm, axis=0, keepdims=True)
    sc = abs(Dhh.diagonal()).mean() or 1.0
    A = bmat([[Dhh, sc * csr_matrix(Rm)], [sc * csr_matrix(Rm.T), None]],
             format="csc")
    lu = _factor(A, "end-section (z = %.8g)" % end["z"])

    def proj(F):
        return F - Rm @ (Rm.T @ F)

    V0 = np.zeros((N3, 4))
    for p in range(4):
        V0[:, p] = lu.solve(np.concatenate([proj(Dhe[:, p]),
                                            np.zeros(4)]))[:N3]
    if gauge == "area":
        Psi, Dc = _end_gauge(end)

        def pin(V):                                        # Eq. 85
            return V - Psi @ np.linalg.solve(Dc.T @ Psi, Dc.T @ V)
        V0 = pin(V0)
    D1 = V0.T @ (-Dhe)
    D_eff = Dee + D1                                       # an end: no 1/L

    DhlV0 = Dhl.T @ V0
    DhlTV0Dle = Dhl @ V0 + Dle
    V0DllV0 = V0.T @ (Dll @ V0)
    b = DhlTV0Dle - DhlV0
    if gauge == "area":                                    # Eq. 100
        b = b - Dc @ np.linalg.solve(Psi.T @ Dc, Psi.T @ b)
    V1s = np.zeros((N3, 4))
    for p in range(4):
        V1s[:, p] = lu.solve(np.concatenate([proj(b[:, p]),
                                             np.zeros(4)]))[:N3]
    if gauge == "area":
        V1s = pin(V1s)
    B_tim = DhlTV0Dle.T @ V0
    C_tim = V0DllV0 + V1s.T @ (DhlV0 + DhlTV0Dle)
    C_tim = 0.5 * (C_tim + C_tim.T)
    return _finalize(D_eff, B_tim, C_tim), D_eff, V0, V1s


# ================================================================== segment
def solve_segment(seg, ends, fields):
    """The 3-D segment: gamma_h of dimension 3, Dirichlet data on the
    end-face nodes equal to the end fluctuations, one SuperLU
    factorization for the four V0 and the four V1 cases.

    In:  seg dict from segment_mesh, ends (left, right) dicts,
         fields ((V0L, V1L), (V0R, V1R))
    Out: (Deff6 (6, 6), D_eff (4, 4), dofs int, free dofs int)."""
    nodes = seg["nodes"]
    N3 = 3 * len(nodes)
    Dhh, Dll, Dhl, Dhe, Dle, Dee = assemble_blocks(nodes, seg["batches"],
                                                   dim3=True)
    L = seg["length"]
    (V0L, V1L), (V0R, V1R) = fields

    def dofs_of(node_ids):
        return (3 * node_ids[:, None] + np.arange(3)[None, :]).ravel()
    bdofL = dofs_of(ends[0]["node2seg"])
    bdofR = dofs_of(ends[1]["node2seg"])
    bdof = np.unique(np.concatenate([bdofL, bdofR]))
    free = np.setdiff1d(np.arange(N3), bdof)
    K_ff = Dhh[np.ix_(free, free)].tocsc()
    K_fb = Dhh[np.ix_(free, bdof)]
    lu = _factor(K_ff, "segment")
    posL = np.searchsorted(bdof, bdofL)
    posR = np.searchsorted(bdof, bdofR)

    def dirichlet(Frhs, WL, WR, p):
        ub = np.zeros(len(bdof))
        ub[posL] = WL[:, p]
        ub[posR] = WR[:, p]
        u = np.zeros(N3)
        u[free] = lu.solve(Frhs[free] - K_fb @ ub)
        u[bdof] = ub
        return u

    V0 = np.zeros((N3, 4))
    for p in range(4):
        V0[:, p] = dirichlet(Dhe[:, p], V0L, V0R, p)
    D1 = V0.T @ (-Dhe)
    D_eff = (Dee + D1) / L
    D_eff = 0.5 * (D_eff + D_eff.T)

    DhlV0 = Dhl.T @ V0
    DhlTV0Dle = Dhl @ V0 + Dle
    V0DllV0 = V0.T @ (Dll @ V0)
    b = DhlTV0Dle - DhlV0
    V1s = np.zeros((N3, 4))
    for p in range(4):
        V1s[:, p] = dirichlet(b[:, p], V1L, V1R, p)
    B_tim = (DhlTV0Dle.T @ V0) / L
    C_tim = V0DllV0 + V1s.T @ (DhlV0 + DhlTV0Dle)
    C_tim = 0.5 * (C_tim + C_tim.T) / L
    return _finalize(D_eff, B_tim, C_tim), D_eff, N3, len(free)


# ===================================================== the mesh and its ends
def element_kinds(cells):
    """The segment element type of every cell, or the one-line refusal.

    In:  cells list of 0-based connectivity lists
    Out: list[str] -- hex8 | tet4 | tet10 per cell."""
    counts = sorted({len(c) for c in cells})
    bad = [n for n in counts if n not in SEGMENT_ELEMENTS]
    if bad:
        raise SegmentInputError(
            "the solid segment route takes hex8, tet4, mixed hex8 + tet4 and "
            "tet10 meshes, not %s-node elements"
            % "/".join(str(n) for n in bad))
    if 10 in counts and len(counts) > 1:
        raise SegmentInputError(
            "a tet10 segment cannot mix in linear elements (their faces do "
            "not match): mesh it in tet10 only, or in hex8 and tet4")
    return [SEGMENT_ELEMENTS[len(c)] for c in cells]


def segment_mesh(sc, C_asm, rho_e):
    """The segment in beam axes, one batch per element type.

    In:  sc dict (sg_mesh.load_sg_input) with 3-D nodes in mesh axes,
         C_asm (E, 6, 6) per-element stiffness with the frames applied
         (solver order), rho_e (E,) per-element density
    Out: dict nodes (N, 3) in beam axes (x1, x2, x3), batches {kind:
         (conn, C, rho, element index)}, length L, z0, z1, nelem."""
    nodes = np.asarray(sc["nodes"], float)[:, :3][:, _PERM3]
    kinds = element_kinds(sc["cells"])
    C_asm = np.asarray(C_asm, float)
    batches = {}
    for kind in ("tet4", "hex8", "tet10"):
        ix = np.array([k for k, t in enumerate(kinds) if t == kind], int)
        if ix.size:
            conn = np.array([sc["cells"][k] for k in ix], int)
            batches[kind] = (conn, C_asm[ix], np.asarray(rho_e)[ix], ix)
    z0 = float(nodes[:, 0].min())
    z1 = float(nodes[:, 0].max())
    return dict(nodes=nodes, batches=batches, length=z1 - z0, z0=z0, z1=z1,
                nelem=len(sc["cells"]))


def end_sections(seg):
    """The two end sections: the faces of the cells whose nodes all lie
    in the plane z = z_min (left) or z = z_max (right).  A hex8 face is a
    quad4, a tet4 face a tri3 and a tet10 face a tri6, and each face
    keeps its parent's stiffness and density.

    In:  seg dict from segment_mesh
    Out: (left, right) dicts nodes (M, 3) beam axes, batches {kind: (conn,
         C, rho, parent index)}, node2seg (M,) node ids in the segment,
         z float, counts {kind: n}."""
    nodes = seg["nodes"]
    b = nodes[:, 0]
    tol = 1e-6 * max(seg["length"], 1.0)
    out = []
    for zend in (seg["z0"], seg["z1"]):
        on = np.abs(b - zend) < tol
        faces = {}
        for kind, (conn, C_e, rho, ix) in seg["batches"].items():
            fk = FACE_KIND[kind]
            for loc in _FACES[kind]:
                fn = conn[:, list(loc)]
                hit = np.where(on[fn].all(axis=1))[0]
                if hit.size:
                    f = faces.setdefault(fk, ([], [], [], []))
                    f[0].append(fn[hit])
                    f[1].append(C_e[hit])
                    f[2].append(rho[hit])
                    f[3].append(ix[hit])
        if not faces:
            raise SegmentInputError(
                "no element face lies in the end plane z = %.8g: the solid "
                "segment route needs flat end sections at z_min and z_max "
                "(mesh z is the beam axis)" % zend)
        used = np.unique(np.concatenate(
            [np.concatenate(f[0]).ravel() for f in faces.values()]))
        g2l = np.full(len(nodes), -1, int)
        g2l[used] = np.arange(len(used))
        batches = {}
        for fk in ("tri3", "quad4", "tri6"):
            if fk in faces:
                f = faces[fk]
                batches[fk] = (g2l[np.concatenate(f[0])],
                               np.concatenate(f[1]), np.concatenate(f[2]),
                               np.concatenate(f[3]))
        out.append(dict(nodes=nodes[used], batches=batches, node2seg=used,
                        z=float(zend),
                        counts={k: len(v[0]) for k, v in batches.items()}))
    shared = np.intersect1d(out[0]["node2seg"], out[1]["node2seg"])
    if shared.size:
        raise SegmentInputError(
            "the two end sections share %d nodes: the segment has no "
            "length between z = %.8g and z = %.8g"
            % (shared.size, seg["z0"], seg["z1"]))
    return out[0], out[1]


# ============================================================== mass matrix
_G3 = np.array([-np.sqrt(0.6), 0.0, np.sqrt(0.6)])
_W3 = np.array([5.0, 8.0, 5.0]) / 9.0


def _mass_rule(kind):
    """The quadrature of the mass integrals: 3 x 3 x 3 Gauss on a hex8
    (exact for the trilinear map), the 4-point rule on a tet4 (exact for
    the quadratic moments), the 27-point collapsed rule on a tet10 (exact
    for straight edges)."""
    if kind == "hex8":
        pts = np.array([(a, b, c) for a in _G3 for b in _G3 for c in _G3])
        w = np.array([wa * wb * wc for wa in _W3 for wb in _W3
                      for wc in _W3])
        _, _, N, dN = _rule("hex8")
        return pts, w, N, dN
    return _rule(kind)


def segment_mass(seg):
    """get_mass_solid with Taper=True: the volume integrals of rho, rho x2,
    rho x3, rho x3^2, rho x2^2 and rho x2 x3, divided by the segment length.

    In:  seg dict from segment_mesh
    Out: (M (6, 6) per unit length about the line x = y = 0, VABS layout
         as sg_homo.mass_matrix_2d, info dict mpus, mass_center (x2, x3),
         i22, i33, i23 per unit length, total mass of the segment)."""
    m = S2 = S3 = I22 = I33 = I23 = 0.0
    for kind, (conn, _C, rho, _ix) in seg["batches"].items():
        pts, w, N, dN = _mass_rule(kind)
        X = seg["nodes"][conn]                              # (n, nn, 3)
        for p, wq in zip(pts, w):
            Np = N(p)
            J = np.einsum("dn,enk->edk", dN(p), X)         # (n, 3, 3)
            dv = wq * np.abs(np.linalg.det(J))
            xq = np.einsum("n,enk->ek", Np, X)              # (n, 3)
            rdv = np.asarray(rho, float) * dv
            m += rdv.sum()
            S2 += (rdv * xq[:, 1]).sum()
            S3 += (rdv * xq[:, 2]).sum()
            I22 += (rdv * xq[:, 2] ** 2).sum()
            I33 += (rdv * xq[:, 1] ** 2).sum()
            I23 += (rdv * xq[:, 1] * xq[:, 2]).sum()
    L = seg["length"]
    M = np.array([[m, 0, 0, 0, S3, -S2],
                  [0, m, 0, -S3, 0, 0],
                  [0, 0, m, S2, 0, 0],
                  [0, -S3, S2, I22 + I33, 0, 0],
                  [S3, 0, 0, 0, I22, -I23],
                  [-S2, 0, 0, 0, -I23, I33]]) / L
    info = dict(mpus=m / L, mass_center=(np.array([S2 / m, S3 / m]) if m
                                         else np.zeros(2)),
                i22=I22 / L, i33=I33 / L, i23=I23 / L, mass=m)
    return M, info


def end_mass(end):
    """The section mass matrix of one end (sg_homo.mass_matrix_2d on the
    end faces, about the line x = y = 0)."""
    from .sg_homo import mass_matrix_2d
    cells, rho = [], []
    for conn, _C, r, _ix in end["batches"].values():
        cells += [list(c) for c in conn]
        rho += list(np.asarray(r, float))
    sc2 = {"nodes": end["nodes"][:, 1:3], "cells": cells,
           "mat_id": np.arange(1, len(cells) + 1)}
    M, _ = mass_matrix_2d(sc2, np.asarray(rho, float))
    return M


# ============================================================ the .out files
def write_ends_out(path, ends, laws, masses, title):
    """<base>_ends.out: each end section's 6x6, compliance and mass matrix
    in the layout of sg_homo.write_sc_K, one block per end."""
    from .sg_homo import _knum
    with open(path, "w") as f:
        f.write(" OpenSG msg-solid beam segment end sections, aperiodic "
                "ends at z %.8g and z %.8g\n" % (ends[0]["z"], ends[1]["z"]))
        for k, (end, C, M) in enumerate(zip(ends, laws, masses)):
            S = np.linalg.inv(C)
            f.write("\n End section %d at z = %.8g (%s)\n"
                    % (k + 1, end["z"], _count_text(end["counts"])))
            f.write("\n The Effective %s Stiffness Matrix\n"
                    " --------------------------------------------\n" % title)
            for i in range(C.shape[0]):
                f.write("".join(_knum(C[i, j]) for j in range(C.shape[1]))
                        + "\n")
            f.write("\n The Effective %s Compliance Matrix\n"
                    " --------------------------------------------\n" % title)
            for i in range(S.shape[0]):
                f.write("".join(_knum(S[i, j]) for j in range(S.shape[1]))
                        + "\n")
            f.write("\n The 6X6 Mass Matrix\n"
                    " ========================================================"
                    "\n\n")
            for i in range(6):
                f.write("".join(_knum(M[i, j]) for j in range(6)) + "\n")


def _count_text(counts):
    return " + ".join("%d %s" % (n, k) for k, n in sorted(counts.items()))


def _density_per_element(sc, density):
    """(E,) density per element: the `density` argument (n_mat,) in
    material-id order when given, else each material block's `density:`
    (0 when absent)."""
    mids = sorted(sc["materials"])
    if density is not None:
        d = np.asarray(density, float).ravel()
        rho_of = {mid: float(d[i]) for i, mid in enumerate(mids)}
    else:
        rho_of = {mid: float(sc["materials"][mid].get("density", 0.0) or 0.0)
                  for mid in mids}
    return np.array([rho_of[int(m)] for m in np.asarray(sc["mat_id"], int)])


# ================================================================ the route
def segment_homo(sc, base, elem_rotation=None, material_param=None,
                 angles=None, refined=1, density=None, t0=None,
                 emit_plot=None, gauge="area"):
    """The solid segment route: <base>.out and <base>_ends.out.

    In:  sc dict (sg_mesh.load_sg_input) of a 3-D mesh, base str output
         base path, elem_rotation (E, 9) solver-order frames or None,
         material_param and angles as sg_materials.build_material_C,
         refined 1 (Timoshenko 6x6 is the law) | 0 (Euler-Bernoulli
         4x4), density (n_mat,) or None (the blocks' `density:`), t0
         perf_counter start, emit_plot callable or None
    Out: the plate_homo_2d r dict: C_eff (6, 6) Timoshenko of the
         segment, C_eff_EB (4, 4), law, law_title, Mass (6, 6), mass_info,
         ends (two dicts z, counts, C_eff, C_eff_EB, Mass), length,
         omega (= length), n_model 1, n_sg 3, refined, route, solve_time,
         sc, elem_rotation."""
    from .sg_homo import write_sc_K
    from .sg_materials import get_heterogeneous_C_matrix, ply_angles_used
    t0 = time.perf_counter() if t0 is None else t0
    if int(sc["dim"]) != 3:
        raise SegmentInputError(
            "the solid segment route needs a 3-D segment mesh, got a %d-D "
            "SG" % int(sc["dim"]))
    element_kinds(sc["cells"])                     # refuse early, one line
    C_asm, _ = get_heterogeneous_C_matrix(sc, material_param, angles,
                                          elem_rotation)
    rho_e = _density_per_element(sc, density)
    seg = segment_mesh(sc, C_asm, rho_e)
    left, right = end_sections(seg)
    L = seg["length"]
    print(" route     : solid segment, length %.8g, ends z = %.8g (%s) and "
          "z = %.8g (%s)" % (L, left["z"], _count_text(left["counts"]),
                             right["z"], _count_text(right["counts"])))
    print(" reference : the law is about the line x = y = 0 of the mesh "
          "(the beam axis is mesh z), per unit length of the segment")
    print(" solver    : superlu (end sections and segment)")

    tb = time.perf_counter()
    DL, DL4, V0L, V1L = solve_end(left, gauge)
    DR, DR4, V0R, V1R = solve_end(right, gauge)
    t_end = time.perf_counter() - tb
    ts = time.perf_counter()
    D6, D4, ndof, nfree = solve_segment(seg, (left, right),
                                        ((V0L, V1L), (V0R, V1R)))
    t_seg = time.perf_counter() - ts

    M6, minfo = segment_mass(seg)
    if minfo["mass"] == 0.0:
        print(" mass      : no density in the materials: mass matrix "
              "computed with density 0 (zeros)")
    else:
        print(" mass      : %.8g per unit length (total %.8g over the "
              "segment), mass centre x2 %.8g, x3 %.8g"
              % (minfo["mpus"], minfo["mass"], minfo["mass_center"][0],
                 minfo["mass_center"][1]))
    ML = end_mass(left)
    MR = end_mass(right)

    r = {"C_eff": np.asarray(D6), "C_eff_EB": np.asarray(D4),
         "Mass": M6, "mass_info": minfo, "length": L, "omega": float(L),
         "ends": [dict(z=left["z"], counts=left["counts"], C_eff=DL,
                       C_eff_EB=DL4, Mass=ML),
                  dict(z=right["z"], counts=right["counts"], C_eff=DR,
                       C_eff_EB=DR4, Mass=MR)],
         "n_model": 1, "n_sg": 3, "refined": int(refined),
         "route": "solid segment", "boundary": "aperiodic",
         "segment": dict(dofs=ndof, free_dofs=nfree, nelem=seg["nelem"],
                         t_ends=t_end, t_segment=t_seg),
         "sc": sc,
         "elem_rotation": (None if elem_rotation is None else
                           np.asarray(elem_rotation, float).reshape(-1, 9))}
    r["ply_angles"], r["ply_angles_as_given"] = ply_angles_used(
        sc, material_param, angles)
    if int(refined) == 0:
        r["law"] = np.asarray(D4)
        r["law_title"] = ("Euler-Bernoulli Beam Stiffness Matrix  "
                          "[eps11 kappa1 kappa2 kappa3]")
        name = "Euler-Bernoulli Beam"
    else:
        r["law"] = np.asarray(D6)
        r["law_title"] = ("Timoshenko Beam Stiffness Matrix  "
                          "[eps11 gam12 gam13 kappa1 kappa2 kappa3]")
        name = "Timoshenko Beam"
    r["solve_time"] = time.perf_counter() - t0
    write_sc_K(base + ".out", r["law"], solve_time=r["solve_time"],
               model="msg-solid beam segment model, omega %.8g, aperiodic "
                     "ends at z %.8g and z %.8g" % (L, left["z"], right["z"]),
               constants=False, name=name, mass=M6)
    write_ends_out(base + "_ends.out", (left, right), (DL, DR), (ML, MR),
                   "Timoshenko Beam")
    print(" ends      : the end-section laws are stored in %s_ends.out"
          % base)
    if emit_plot is not None:
        emit_plot()
    return r
