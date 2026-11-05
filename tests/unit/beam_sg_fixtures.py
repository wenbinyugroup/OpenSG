"""Fixtures of the solid-beam (n_model 1) D-route tests: structured quad9
rectangle and annulus sections written as canonical OpenSG yamls (no
gmsh), random proper element frames, the baked type-2 twin of a framed
section, the six beam resultants of a recovered field and the Voigt
rotations -- built here from their definitions, independent of the
engine's own helpers (sg_materials.rotate_C_with_matrix,
sg_dehom._voigt_stress_T).

Frames are written as the `orientation:` block (the one name OpenSG
writes; the legacy spelling has its own tests, test_orientation_key.py).
Rows of a frame matrix M are the element axes e1 e2 e3 in the solver's
(x1, y2, y3) components; the yaml stores each axis in (y2, y3, x1)
order (sg_materials.elem_rotation_from_yaml).
"""
import os
import struct

import numpy as np

assert struct.calcsize("P") == 8

VO = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))   # xx yy zz yz xz xy
GLASS = dict(E=(41e9, 14e9, 14e9), G=(4.5e9, 4.5e9, 3.5e9),
             nu=(0.28, 0.28, 0.40))                   # glass-epoxy, Pa


# ----------------------------------------------------------- meshes (quad9)
def rect_quad9(b, h, n2, n3, center=(0.0, 0.0)):
    """Structured quad9 mesh of the b (y2) x h (y3) rectangle centred at
    `center`, n2 x n3 elements.

    Out: nodes (V, 3) rows [y2, y3, 0]; cells (E, 9) 0-based, gmsh quad9
         order (corners counter-clockwise, edge midsides 01 12 23 30,
         centre)."""
    y2 = center[0] + np.linspace(-0.5 * b, 0.5 * b, 2 * n2 + 1)
    y3 = center[1] + np.linspace(-0.5 * h, 0.5 * h, 2 * n3 + 1)
    Y2, Y3 = np.meshgrid(y2, y3)                 # (2 n3 + 1, 2 n2 + 1)
    nodes = np.column_stack([Y2.ravel(), Y3.ravel(), np.zeros(Y2.size)])
    nid = lambda i, j: j * (2 * n2 + 1) + i      # noqa: E731
    cells = []
    for c in range(n3):
        for a in range(n2):
            i, j = 2 * a, 2 * c
            cells.append([nid(i, j), nid(i + 2, j), nid(i + 2, j + 2),
                          nid(i, j + 2), nid(i + 1, j), nid(i + 2, j + 1),
                          nid(i + 1, j + 2), nid(i, j + 1),
                          nid(i + 1, j + 1)])
    return nodes, np.asarray(cells, int)


def annulus_quad9(a, b, n_theta, n_r):
    """Structured quad9 mesh of the annulus a <= r <= b (centre at the
    origin), n_theta elements around and n_r through the wall; every node
    sits on its exact polar position.

    Out: nodes (V, 3) [y2, y3, 0]; cells (E, 9) gmsh quad9 order."""
    nt, nr = 2 * n_theta, 2 * n_r + 1
    th = np.linspace(0.0, 2.0 * np.pi, nt, endpoint=False)
    rr = np.linspace(a, b, nr)
    T, R = np.meshgrid(th, rr)                   # (nr, nt)
    nodes = np.column_stack([(R * np.cos(T)).ravel(), (R * np.sin(T)).ravel(),
                             np.zeros(T.size)])
    nid = lambda i, j: j * nt + (i % nt)         # noqa: E731
    cells = []
    for c in range(n_r):
        for e in range(n_theta):
            i, j = 2 * e, 2 * c
            # (r, theta) is right-handed in (y2, y3): r out, theta CCW
            cells.append([nid(i, j), nid(i, j + 2), nid(i + 2, j + 2),
                          nid(i + 2, j), nid(i, j + 1), nid(i + 1, j + 2),
                          nid(i + 2, j + 1), nid(i + 1, j),
                          nid(i + 1, j + 1)])
    return nodes, np.asarray(cells, int)


# --------------------------------------------------------------- materials
def iso(E, nu):
    """Isotropic block (type 0)."""
    return {"type": 0, "E": float(E), "nu": float(nu)}


def ortho(E, G, nu, angle=0.0):
    """Orthotropic block (type 1, engineering E1 E2 E3 G12 G13 G23 v12 v13
    v23) with an optional block `angle` [deg]."""
    m = {"type": 1, "engineering": [float(v) for v in
                                    (*E, *G, *nu)]}
    if angle:
        m["angle"] = float(angle)
    return m


def stiffness(m):
    """(6, 6) stiffness of a material block, SwiftComp order, from its
    constants (the compliance inverse; block angle NOT applied)."""
    if m["type"] == 2:
        return np.asarray(m["C"], float)
    if m["type"] == 0:
        E, nu = m["E"], m["nu"]
        G = E / (2.0 * (1.0 + nu))
        p = [E, E, E, G, G, G, nu, nu, nu]
    else:
        p = m["engineering"]
    E1, E2, E3, G12, G13, G23, v12, v13, v23 = p
    S = np.diag([1 / E1, 1 / E2, 1 / E3, 1 / G23, 1 / G13, 1 / G12])
    S[0, 1] = S[1, 0] = -v12 / E1
    S[0, 2] = S[2, 0] = -v13 / E1
    S[1, 2] = S[2, 1] = -v23 / E2
    return np.linalg.inv(S)


# --------------------------------------------------------------- rotations
def voigt_T(M):
    """(..., 6, 6) Voigt STRESS transform of the tensor rotation
    sigma' = M sigma M^T (rows of M = new axes in old components),
    storage xx yy zz yz xz xy.  The engineering-strain transform of the
    same rotation is voigt_T(M^T)^T."""
    M = np.asarray(M, float)
    T = np.zeros(M.shape[:-2] + (6, 6))
    for I, (i, j) in enumerate(VO):
        for J, (k, l) in enumerate(VO):
            T[..., I, J] = M[..., i, k] * M[..., j, l]
            if k != l:
                T[..., I, J] += M[..., i, l] * M[..., j, k]
    return T


def strain_T(M):
    """(..., 6, 6) engineering-strain transform of the rotation M."""
    return np.swapaxes(voigt_T(np.swapaxes(np.asarray(M, float), -1, -2)),
                       -1, -2)


def ply_rotation(theta_deg):
    """(3, 3) ply axes in element components for the block angle theta
    (fiber at +theta from e1 toward e2, about e3) -- the rotation whose
    stress transform inverts sg_materials.rotate_C_matrix."""
    t = np.radians(float(theta_deg))
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def random_frames(E, seed):
    """(E, 3, 3) seeded random PROPER rotations (rows e1 e2 e3, solver
    components)."""
    rng = np.random.default_rng(seed)
    out = np.empty((E, 3, 3))
    for e in range(E):
        q, rr = np.linalg.qr(rng.standard_normal((3, 3)))
        q = q * np.sign(np.diag(rr))
        if np.linalg.det(q) < 0.0:
            q[2] *= -1.0
        out[e] = q
    return out


def frames_to_yaml(M):
    """(E, 9) yaml rows [e1 | e2 | e3], each axis in (y2, y3, x1) order,
    of solver-order frames M (E, 3, 3)."""
    return np.asarray(M, float)[:, :, [1, 2, 0]].reshape(-1, 9)


def baked_twin(materials, mat_id, M):
    """The frameless twin of a framed section: one type-2 material per
    element whose C is that element's assembly stiffness
    T(M^T) [R(a) C0 R(a)^T] T(M^T)^T (block angle a inside, then the
    frame), built here from the constants.

    Out: (materials dict {1..E: type-2 block}, mat_id (E,) 1..E)."""
    out = {}
    for e, (mid, Me) in enumerate(zip(mat_id, M)):
        m = materials[int(mid)]
        Ra = voigt_T(ply_rotation(m.get("angle", 0.0)).T)
        C = Ra @ stiffness(m) @ Ra.T
        Tm = voigt_T(np.asarray(Me).T)
        out[e + 1] = {"type": 2, "C": (Tm @ C @ Tm.T).tolist()}
    return out, np.arange(1, len(mat_id) + 1)


# ------------------------------------------------------------- yaml writer
def _mat_line(k, m):
    if m["type"] == 0:
        body = "type: 0, E: %.17g, nu: %.17g" % (m["E"], m["nu"])
    elif m["type"] == 1:
        body = "type: 1, engineering: [%s]" % ", ".join(
            "%.17g" % v for v in m["engineering"])
    else:
        body = "type: 2, C: [%s]" % ", ".join(
            "[%s]" % ", ".join("%.17g" % v for v in row) for row in m["C"])
    if m.get("angle"):
        body += ", angle: %.17g" % m["angle"]
    return "  %d: {%s}" % (k, body)


def write_yaml(path, nodes, cells, mat_id, materials, n_model=1,
               refined=1, frames=None):
    """Canonical-dialect OpenSG yaml (nodes / cells 0-based / mat_id /
    materials); frames (E, 3, 3) solver-order rotations or None, written
    as the `orientation:` block in yaml order."""
    out = ["msg: solid", "n_model: %d" % n_model, "refined: %d" % refined,
           "nodes:"]
    out += ["- [%.17g, %.17g, %.17g]" % tuple(p) for p in nodes]
    out.append("cells:")
    out += ["- [%s]" % ", ".join(str(int(v)) for v in c) for c in cells]
    out.append("mat_id: [%s]" % ", ".join(str(int(m)) for m in mat_id))
    out.append("materials:")
    out += [_mat_line(int(k), materials[k]) for k in sorted(materials)]
    if frames is not None:
        out.append("orientation:")
        out += ["- [%s]" % ", ".join("%.17g" % v for v in row)
                for row in frames_to_yaml(frames)]
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    return path


def homogenize(path, **kw):
    """plate_homo_2d with recovery, no plot, outputs next to the yaml."""
    from opensg_solid.sg_homo import plate_homo_2d
    kw.setdefault("recovery", True)
    return plate_homo_2d(path, plot=False,
                         workdir=os.path.dirname(os.path.abspath(path)), **kw)


# -------------------------------------------------------------- resultants
def gauss_xy(r):
    """(E, Q, 2) Gauss-point (y2, y3) of a single-batch 2-D beam SG."""
    return np.einsum("qn,end->eqd", np.asarray(r["phi_qn"]),
                     np.asarray(r["x_end"]))


def resultants(r, Sig):
    """The six beam resultants [F1 F2 F3 M1 M2 M3] of an SG-global stress
    cloud (E, Q, 6), integrated with the run's own quadrature dV_q at the
    Gauss points: F1 = int s11, F2 = int s12, F3 = int s13, M1 = int (y2
    s13 - y3 s12), M2 = int y3 s11, M3 = -int y2 s11."""
    S = np.asarray(Sig, float)
    dV = np.asarray(r["dehomo_data"]["dV_q"], float)
    xy = gauss_xy(r)
    y2, y3 = xy[..., 0], xy[..., 1]
    s11, s12, s13 = S[..., 0], S[..., 5], S[..., 4]
    return np.array([(s11 * dV).sum(), (s12 * dV).sum(), (s13 * dV).sum(),
                     ((y2 * s13 - y3 * s12) * dV).sum(),
                     (y3 * s11 * dV).sum(), (-y2 * s11 * dV).sum()])
