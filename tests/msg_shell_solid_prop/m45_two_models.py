"""m45 circular tube, TWO OpenSG runs: shell cross-section vs solid cross-section.

Run 1 (SHELL): 1-D ring SG, center reference -- opensg_shell.build_solid_bundle.
Run 2 (SOLID): 2-D annulus SG (144x4 quads), same wall, same ply rotation.

Each run writes ONE .out file containing its 6x6 stiffness C3D, its 6x6 compliance
S = inv(C3D), and all 9 effective elastic constants:
    m45_shell.out ,  m45_solid.out
Then m45_compare.dat tabulates EVERY upper-triangle stiffness term (21 rows,
shell vs solid, %diff vs solid where the solid term is resolved, '~0' marker
otherwise) and the 9 effective-constant comparison.

Order everywhere: [G11 G22 G33 2G23 2G13 2G12], 1 = tube axis; cell area pi R^2.

# ----------------------------------------------------------------------------
# ALL VARIABLES USED IN THIS SCRIPT
# ----------------------------------------------------------------------------
#   R, t, nc, nt        geometry / meshes;  mat = the m45 ply (45 deg)
#   C_sh, S_sh, k_sh    shell stiffness, compliance, constants
#   C_so, S_so, k_so    solid counterparts
#   m45_shell.out, m45_solid.out, m45_compare.dat    outputs
# ----------------------------------------------------------------------------
Run (from this folder):  python m45_two_models.py
"""
import numpy as np
from scipy.linalg import lu_factor, lu_solve

from opensg_shell import build_solid_bundle, elastic_constants, GBAR_ORDER

############### User Input #################################
R = 1.0
t = 0.03
nc = 144
nt = 4
mat = {"E": [142.0e9, 9.8e9, 9.8e9], "G": [6.0e9, 6.0e9, 4.8e9],
       "nu": [0.30, 0.30, 0.42], "angle": 45.0}
############################################################

A_cell = np.pi * R**2
VI = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]


def C_ply(m):
    E1, E2, E3 = m["E"]; G12, G13, G23 = m["G"]; n12, n13, n23 = m["nu"]
    S = np.zeros((6, 6))
    S[0, 0], S[1, 1], S[2, 2] = 1/E1, 1/E2, 1/E3
    S[0, 1] = S[1, 0] = -n12/E1
    S[0, 2] = S[2, 0] = -n13/E1
    S[1, 2] = S[2, 1] = -n23/E2
    S[3, 3], S[4, 4], S[5, 5] = 1/G23, 1/G13, 1/G12
    return np.linalg.inv(S)


def rotate_C(C6, Q):
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


def solid_annulus_C3D(m):
    """2-D annulus SG, VAMUCH zeroth order, MSG constraints <w_i> dA = 0."""
    ang = np.radians(m["angle"]); Cp = C_ply(m)
    thn = 2.0 * np.pi * np.arange(nc) / nc
    rr = R - t / 2 + t * np.arange(nt + 1) / nt
    xy = np.array([[r * np.cos(a), r * np.sin(a)] for r in rr for a in thn])
    nid = lambda ir, ic: ir * nc + (ic % nc)
    quads = np.array([[nid(ir, ic), nid(ir, ic + 1), nid(ir + 1, ic + 1),
                       nid(ir + 1, ic)] for ir in range(nt) for ic in range(nc)], int)
    c, s = np.cos(ang), np.sin(ang)
    Cg = np.zeros((len(quads), 6, 6))
    for e, q in enumerate(quads):
        am = np.arctan2(xy[q, 1].mean(), xy[q, 0].mean())
        eA = np.array([1.0, 0.0, 0.0])
        eT = np.array([0.0, -np.sin(am), np.cos(am)])
        eN = np.cross(eA, eT)
        Q = np.vstack([c * eA + s * eT, -s * eA + c * eT, eN])
        Cg[e] = rotate_C(Cp, Q.T)
    nn = len(xy); ndof = 3 * nn
    Ie = np.eye(6); gp = 1.0 / np.sqrt(3.0)
    Dhh = np.zeros((ndof, ndof)); Dhe = np.zeros((ndof, 6)); Dee = np.zeros((6, 6))
    wA = np.zeros(nn)
    for e, q in enumerate(quads):
        Xq = xy[q]; C = Cg[e]
        cols = np.concatenate([[3 * n, 3 * n + 1, 3 * n + 2] for n in q])
        Ke = np.zeros((12, 12)); Fe = np.zeros((12, 6))
        for xi in (-gp, gp):
            for et in (-gp, gp):
                dN = 0.25 * np.array([[-(1 - et), -(1 - xi)], [(1 - et), -(1 + xi)],
                                      [(1 + et), (1 + xi)], [-(1 + et), (1 - xi)]])
                J = Xq.T @ dN; dJ = abs(np.linalg.det(J))
                dNx = dN @ np.linalg.inv(J)
                Nsh = 0.25 * np.array([(1 - xi) * (1 - et), (1 + xi) * (1 - et),
                                       (1 + xi) * (1 + et), (1 - xi) * (1 + et)])
                wA[q] += Nsh * dJ
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
                Dee += dJ * C
        Dhh[np.ix_(cols, cols)] += Ke
        for a in range(12):
            Dhe[cols[a]] += Fe[a]
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


def write_out(path, tag, C, S, k):
    with open(path, "w") as f:
        f.write("# m45 circular tube, %s cross-section, center reference\n" % tag)
        f.write("# R=%g t=%g nc=%d; ply E=[142,9.8,9.8]GPa G=[6,6,4.8]GPa "
                "nu=[0.30,0.30,0.42] at 45deg; cell area pi R^2 = %.8f\n"
                % (R, t, nc, A_cell))
        f.write("# order %s\n#\n" % GBAR_ORDER)
        f.write("# ---- C3D stiffness (6x6, Pa) ----\n")
        for i in range(6):
            f.write(" ".join("%16.8e" % C[i, j] for j in range(6)) + "\n")
        f.write("#\n# ---- compliance S = inv(C3D) (6x6, 1/Pa) ----\n")
        for i in range(6):
            f.write(" ".join("%16.8e" % S[i, j] for j in range(6)) + "\n")
        f.write("#\n# ---- 9 effective elastic constants (cond(C3D) = %.3e) ----\n"
                % k["cond"])
        for kk in ("E1", "E2", "E3", "G23", "G13", "G12", "nu12", "nu13", "nu23"):
            f.write("%-6s %16.8e\n" % (kk, k[kk]))


# ---- run 1: SHELL ----------------------------------------------------------
B = build_solid_bundle("circle_m45_shell.yaml", cell_area=A_cell)
C_sh = np.asarray(B["C3D"])
k_sh, S_sh = elastic_constants(C_sh)
write_out("m45_shell.out", "SHELL (1-D ring SG)", C_sh, S_sh, k_sh)

# ---- run 2: SOLID ----------------------------------------------------------
C_so = solid_annulus_C3D(mat)
k_so, S_so = elastic_constants(C_so)
write_out("m45_solid.out", "SOLID (2-D annulus SG, %dx%d quads)" % (nc, nt),
          C_so, S_so, k_so)

# ---- comparison .dat -------------------------------------------------------
thr = 1e-8 * max(np.max(np.abs(C_sh)), np.max(np.abs(C_so)))
with open("m45_compare.dat", "w") as f:
    f.write("# m45 tube: SHELL vs SOLID cross-section (center ref); "
            "order %s\n" % GBAR_ORDER)
    f.write("# stiffness: all 21 upper-triangle terms; resolved = |C| > %.2e "
            "in either model ('~0' = below, pct vs solid where solid resolved)\n" % thr)
    f.write("# %-5s %16s %16s %10s %9s\n"
            % ("term", "C_shell", "C_solid", "pct_diff", "resolved"))
    for i in range(6):
        for j in range(i, 6):
            sh, so = C_sh[i, j], C_so[i, j]
            res = (abs(sh) > thr or abs(so) > thr)
            pct = ("%+10.4f" % (100.0 * (sh - so) / so)
                   if res and abs(so) > thr else "       ---")
            f.write("  C%d%d   %16.8e %16.8e %s %9s\n"
                    % (i + 1, j + 1, sh, so, pct, "yes" if res else "~0"))
    f.write("#\n# 9 effective elastic constants "
            "(cond: shell %.2e, solid %.2e -- constants other than E1 rest on\n"
            "# unresolved compliance entries for this free-tube SG; "
            "physically meaningful: E1)\n" % (k_sh["cond"], k_so["cond"]))
    f.write("# %-6s %16s %16s %10s\n" % ("const", "shell", "solid", "pct_diff"))
    for kk in ("E1", "E2", "E3", "G23", "G13", "G12", "nu12", "nu13", "nu23"):
        a, b = k_sh[kk], k_so[kk]
        pct = "%+10.4f" % (100.0 * (a - b) / b) if abs(b) > 0 else "       ---"
        f.write("  %-6s %16.8e %16.8e %s\n" % (kk, a, b, pct))

print("wrote m45_shell.out, m45_solid.out, m45_compare.dat")
print("shell C11 %.8e   solid C11 %.8e   %+0.4f %%"
      % (C_sh[0, 0], C_so[0, 0], 100 * (C_sh[0, 0] - C_so[0, 0]) / C_so[0, 0]))
