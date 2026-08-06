"""SOLID 2-D cross-section reference vs the msg_shell solid-props route: circle tube.

The same physical tube (R = 1, t = 0.03, iso and m45 walls, center reference) is
homogenized two ways and compared:

  SHELL  1-D ring SG, opensg_shell.build_solid_bundle (the author's formulation:
         Gamma_e/Gamma_h per MSG_shell_solid_properties.pdf, drilling multiplier,
         V0 solve, D_eff/(pi R^2)).
  SOLID  2-D annulus SG (nc x nt bilinear quads, 3 DOF/node), classical 3-D
         elasticity, VAMUCH zeroth order: e = Gbar + B w, <w_i> = 0 and the
         in-plane rotation pinned, C3D = (Dee + V0^T Dhe)/(pi R^2).  The m45 wall
         rotates the orthotropic ply stiffness per element: ply-1 at +45 deg from
         the axial toward the tangent about the (inward) wall normal -- the same
         convention as the shell layup angle.

Strain/stiffness order in both: [G11 G22 G33 2G23 2G13 2G12], 1 = tube axis.

# ----------------------------------------------------------------------------
# ALL VARIABLES USED IN THIS SCRIPT
# ----------------------------------------------------------------------------
#   R, t, nc, nt        geometry and meshes (nc shared by both models)
#   cases               iso / m45 material definitions (same as circle_solid_props)
#   C_sh, C_so          the two C3D matrices per case
#   circle_<case>_solidref_C3D.out, circle_<case>_compare.out   outputs
# ----------------------------------------------------------------------------
Run (from this folder):  python circle_solid_ref_compare.py
"""
import numpy as np
from scipy.linalg import lu_factor, lu_solve

from opensg_shell import build_solid_bundle, GBAR_ORDER

############### User Input #################################
R = 1.0
t = 0.03
nc = 144                # circumferential elements (shell ring AND solid annulus)
nt = 4                  # through-thickness solid elements
############################################################

A_cell = np.pi * R**2

cases = {
    "iso": {"E": [70.0e9] * 3, "G": [26.923076923e9] * 3, "nu": [0.3] * 3,
            "angle": 0.0},
    "m45": {"E": [142.0e9, 9.8e9, 9.8e9], "G": [6.0e9, 6.0e9, 4.8e9],
            "nu": [0.30, 0.30, 0.42], "angle": 45.0},
}

VI = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]     # Voigt [11 22 33 23 13 12]


def C_ply(mat):
    """Orthotropic 6x6 stiffness in ply axes from engineering constants.
    Material arrays follow the OpenSG dialect: E=[E1,E2,E3], G=[G12,G13,G23],
    nu=[nu12,nu13,nu23]."""
    E1, E2, E3 = mat["E"]; G12, G13, G23 = mat["G"]; n12, n13, n23 = mat["nu"]
    S = np.zeros((6, 6))
    S[0, 0], S[1, 1], S[2, 2] = 1/E1, 1/E2, 1/E3
    S[0, 1] = S[1, 0] = -n12/E1
    S[0, 2] = S[2, 0] = -n13/E1
    S[1, 2] = S[2, 1] = -n23/E2
    S[3, 3], S[4, 4], S[5, 5] = 1/G23, 1/G13, 1/G12
    return np.linalg.inv(S)


def rotate_C(C6, Q):
    """Rotate a 6x6 Voigt stiffness by the 3x3 matrix Q (rows = new-frame axes in
    old-frame components -- i.e. v_new = Q v_old): full 4th-order tensor rotation."""
    C4 = np.zeros((3, 3, 3, 3))
    for a, (i, j) in enumerate(VI):
        for b, (k, l) in enumerate(VI):
            C4[i, j, k, l] = C4[j, i, k, l] = C4[i, j, l, k] = C4[j, i, l, k] = C6[a, b]
    C4r = np.einsum('pi,qj,rk,sl,ijkl->pqrs', Q, Q, Q, Q, C4)
    out = np.zeros((6, 6))
    for a, (i, j) in enumerate(VI):
        for b, (k, l) in enumerate(VI):
            out[a, b] = C4r[i, j, k, l]
    return out


def solid_annulus_C3D(mat):
    """VAMUCH zeroth-order 2-D solid SG of the annulus with the per-element
    rotated wall material.  Global axes: 1 = tube axis, 2/3 = cross-section."""
    ang = np.radians(mat["angle"])
    Cp = C_ply(mat)
    thn = 2.0 * np.pi * np.arange(nc) / nc
    rr = R - t / 2 + t * np.arange(nt + 1) / nt
    xy = np.array([[r * np.cos(a), r * np.sin(a)] for r in rr for a in thn])
    nid = lambda ir, ic: ir * nc + (ic % nc)
    quads = np.array([[nid(ir, ic), nid(ir, ic + 1), nid(ir + 1, ic + 1),
                       nid(ir + 1, ic)] for ir in range(nt) for ic in range(nc)], int)
    # per-element global stiffness: ply -> wall (rotate ang about inward normal),
    # wall -> global.  Wall triad (right-handed, matches the shell yaml):
    #   e1 = axial [1,0,0], e2 = tangent [0,-sin,cos], e3 = e1 x e2 = inward normal
    Cg = np.zeros((len(quads), 6, 6))
    c, s = np.cos(ang), np.sin(ang)
    for e, q in enumerate(quads):
        am = np.arctan2(xy[q, 1].mean(), xy[q, 0].mean())
        eA = np.array([1.0, 0.0, 0.0])
        eT = np.array([0.0, -np.sin(am), np.cos(am)])
        eN = np.cross(eA, eT)                       # inward normal
        p1 = c * eA + s * eT
        p2 = -s * eA + c * eT
        p3 = eN
        Q = np.vstack([p1, p2, p3])                 # rows = ply axes in global comps
        # rotate ply->global: v_ply = Q v_glob, so C_glob = rotate by Q^T rows
        Cg[e] = rotate_C(Cp, Q.T)
    nn = len(xy); ndof = 3 * nn
    Ie = np.eye(6)
    Dhh = np.zeros((ndof, ndof)); Dhe = np.zeros((ndof, 6)); Dee = np.zeros((6, 6))
    gp = 1.0 / np.sqrt(3.0)
    for e, q in enumerate(quads):
        Xq = xy[q]; C = Cg[e]
        cols = np.concatenate([[3 * n, 3 * n + 1, 3 * n + 2] for n in q])
        Ke = np.zeros((12, 12)); Fe = np.zeros((12, 6)); Se = np.zeros((6, 6))
        for xi in (-gp, gp):
            for et in (-gp, gp):
                dN = 0.25 * np.array([[-(1 - et), -(1 - xi)], [(1 - et), -(1 + xi)],
                                      [(1 + et), (1 + xi)], [-(1 + et), (1 - xi)]])
                J = Xq.T @ dN
                dJ = abs(np.linalg.det(J))
                dNx = dN @ np.linalg.inv(J)
                B = np.zeros((6, 12))
                for a in range(4):
                    c1, c2, c3 = 3 * a, 3 * a + 1, 3 * a + 2
                    B[1, c2] = dNx[a, 0]
                    B[2, c3] = dNx[a, 1]
                    B[3, c2] = dNx[a, 1]; B[3, c3] = dNx[a, 0]
                    B[4, c1] = dNx[a, 1]
                    B[5, c1] = dNx[a, 0]
                Ke += dJ * B.T @ C @ B
                Fe += dJ * B.T @ C @ Ie
                Se += dJ * C
        Dee += Se
        for a in range(12):
            Dhe[cols[a]] += Fe[a]
        Dhh[np.ix_(cols, cols)] += Ke
    # MSG constraints <w_i> = 0 as AREA integrals: lumped nodal area weights
    # (sum of |detJ|*w_gauss shares); plus the area-weighted in-plane rotation.
    wA = np.zeros(nn)
    for q in quads:
        Xq = xy[q]
        for xi in (-gp, gp):
            for et in (-gp, gp):
                dN = 0.25 * np.array([[-(1 - et), -(1 - xi)], [(1 - et), -(1 + xi)],
                                      [(1 + et), (1 + xi)], [-(1 + et), (1 - xi)]])
                dJ = abs(np.linalg.det(Xq.T @ dN))
                Nsh = 0.25 * np.array([(1 - xi) * (1 - et), (1 + xi) * (1 - et),
                                       (1 + xi) * (1 + et), (1 - xi) * (1 + et)])
                wA[q] += Nsh * dJ
    Cc = np.zeros((4, ndof))
    for n in range(nn):
        Cc[0, 3 * n] = wA[n]; Cc[1, 3 * n + 1] = wA[n]; Cc[2, 3 * n + 2] = wA[n]
        Cc[3, 3 * n + 1] = -wA[n] * xy[n, 1]; Cc[3, 3 * n + 2] = wA[n] * xy[n, 0]
    A = np.zeros((ndof + 4, ndof + 4))
    A[:ndof, :ndof] = Dhh; A[:ndof, ndof:] = Cc.T; A[ndof:, :ndof] = Cc
    Rhs = np.zeros((ndof + 4, 6)); Rhs[:ndof] = -Dhe
    V0 = lu_solve(lu_factor(A), Rhs)[:ndof]
    Ceff = Dee + V0.T @ Dhe
    return 0.5 * (Ceff + Ceff.T) / A_cell


LBL = ["G11 ", "G22 ", "G33 ", "2G23", "2G13", "2G12"]
_dat_rows = []
for case in ("iso", "m45"):
    mat = cases[case]
    B = build_solid_bundle("circle_%s_shell.yaml" % case, cell_area=A_cell)
    C_sh = np.asarray(B["C3D"])
    C_so = solid_annulus_C3D(mat)

    print("=" * 78)
    print("CASE %s : SHELL ring SG vs SOLID 2-D annulus SG (%dx%d quads)"
          % (case, nc, nt))
    print("order %s" % GBAR_ORDER)
    print("\nSHELL C3D:")
    for i in range(6):
        print("  " + " ".join("%13.5e" % C_sh[i, j] for j in range(6)))
    print("SOLID C3D:")
    for i in range(6):
        print("  " + " ".join("%13.5e" % C_so[i, j] for j in range(6)))
    # STANDING CONVENTION: neglect near-zero (numerically unresolved) terms;
    # compare every RESOLVED term against the corresponding OpenSG solid-mesh
    # value.  Resolved = |entry| > rtol * max|C| in either model.
    rtol = 1e-8
    thr = rtol * max(np.max(np.abs(C_sh)), np.max(np.abs(C_so)))
    print("\nresolved terms (|C| > %.1e; near-zero terms neglected):" % thr)
    print("  %-5s %15s %15s %10s" % ("term", "shell", "solid-mesh", "%diff"))
    any_row = False
    for i in range(6):
        for j in range(i, 6):
            if abs(C_sh[i, j]) > thr or abs(C_so[i, j]) > thr:
                any_row = True
                ref = C_so[i, j]
                dd = (100.0 * (C_sh[i, j] - ref) / ref if abs(ref) > thr
                      else float("inf"))
                print("  C%d%d   %15.6e %15.6e %+10.3f"
                      % (i + 1, j + 1, C_sh[i, j], C_so[i, j], dd))
                _dat_rows.append((case, i + 1, j + 1, C_sh[i, j], ref, dd))
    if not any_row:
        print("  (none above threshold)")
    np.savetxt("circle_%s_solidref_C3D.out" % case, C_so, fmt="%16.8e",
               header="SOLID 2-D annulus SG reference. order " + GBAR_ORDER)
    dd = C_sh - C_so
    np.savetxt("circle_%s_compare.out" % case, dd, fmt="%16.8e",
               header="C3D_shell - C3D_solidref. order " + GBAR_ORDER)
with open("circle_stiffness_compare.dat", "w") as f:
    f.write("# circle tube R=%g t=%g nc=%d, cell area pi R^2; order %s\n"
            % (R, t, nc, GBAR_ORDER))
    f.write("# significant stiffness terms only (|C| > 1e-8*max|C|); "
            "reference = OpenSG solid-mesh (2-D annulus SG, %dx%d quads)\n"
            % (nc, nt))
    f.write("# %-6s %-5s %16s %16s %10s\n"
            % ("case", "term", "C_shell", "C_solid", "pct_diff"))
    for case, i, j, sh, so, dd in _dat_rows:
        f.write("  %-6s C%d%d   %16.8e %16.8e %+10.4f\n" % (case, i, j, sh, so, dd))
print("=" * 78)
print("wrote circle_stiffness_compare.dat  (%d significant terms)" % len(_dat_rows))
print("done.")
