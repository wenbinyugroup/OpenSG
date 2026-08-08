"""Where does the few-percent shell-vs-solid gap come from, and can it shrink?

Two independent sweeps on the square tube, unrotated ply, both routes periodic
and both normalized by the wall material area:

  A. MESH refinement at fixed geometry -- shell contour elements per side and
     solid elements through the wall.  Whatever moves here is discretization.
  B. THIN-WALL sweep t/a = 0.06 .. 0.0075 at converged meshes.  A thin-wall
     model error must shrink with t/a; a discretization error will not.

The solid here is a structured square annulus generated in numpy (mitred
corners, area exactly 4*a*t), so both routes describe the same lattice and no
PreVABS run is needed.

Run (from this folder):  python c44_convergence.py
"""
import numpy as np

from opensg_shell import build_solid_bundle
from opensg_shell.sg_periodicity import mesh_to_periodic_sparse_assembly_map

############### User Input #################################
a = 1.0
E = [142.0e9, 9.8e9, 9.8e9]
G = [6.0e9, 6.0e9, 4.8e9]
nu = [0.30, 0.30, 0.42]
MESH_SWEEP = [(10, 2), (20, 4), (40, 8)]     # (shell nseg/side, solid nt)
T_SWEEP = [0.06, 0.03, 0.015, 0.0075]
############################################################


def shell_C(t_w, nseg):
    h = a/2
    corners = [(-h, -h), (h, -h), (h, h), (-h, h)]
    pts, tans = [], []
    for k in range(4):
        p0, p1 = np.array(corners[k]), np.array(corners[(k+1) % 4])
        tv = (p1-p0)/np.linalg.norm(p1-p0)
        for i in range(nseg):
            pts.append(p0 + (p1-p0)*i/nseg)
            tans.append(tv)
    pts = np.array(pts); m = len(pts)
    o = ["nodes:"]
    for x, y in pts:
        o.append("- [%.12f %.12f 0.00000000]" % (x, y))
    o.append("elements:")
    for i in range(m):
        o.append("- [%d %d]" % (i+1, (i+1) % m + 1))
    o.append("elementOrientations:")
    for i in range(m):
        tx, ty = tans[i]
        e3 = np.cross([0, 0, 1.0], [tx, ty, 0])
        o.append("- [0.0, 0.0, 1.0, %.9f, %.9f, 0.0, %.9f, %.9f, 0.0]"
                 % (tx, ty, e3[0], e3[1]))
    o += ["sections:", "- type: shell", "  elementSet: layup_0", "  layup:",
          "  - - ply", "    - %.6e" % t_w, "    - 0.0",
          "materials:", "- name: ply", "  density: 1600.0", "  elastic:",
          "    E: [%.6e, %.6e, %.6e]" % tuple(E),
          "    G: [%.6e, %.6e, %.6e]" % tuple(G),
          "    nu: [%.6f, %.6f, %.6f]" % tuple(nu),
          "sets:", "  element:", "  - name: layup_0", "    labels:"]
    o += ["    - %d" % (e+1) for e in range(m)]
    o.append("reference: center")
    open("_sweep.yaml", "w").write("\n".join(o) + "\n")
    return np.asarray(build_solid_bundle("_sweep.yaml",
                                         cell_area=4*a*t_w)["C3D"])


def solid_C(t_w, nt, na=48):
    """structured square annulus, periodic, CST."""
    h = a/2
    lo, hi = h - t_w/2, h + t_w/2
    xa = np.concatenate([np.linspace(-hi, -lo, nt+1),
                         np.linspace(-lo, lo, na+1)[1:-1],
                         np.linspace(lo, hi, nt+1)])
    ng = len(xa)
    gid = -np.ones((ng, ng), int)
    qc = [(i, j) for i in range(ng-1) for j in range(ng-1)
          if abs(0.5*(xa[i]+xa[i+1])) > lo or abs(0.5*(xa[j]+xa[j+1])) > lo]
    nid = 0
    for (i, j) in qc:
        for di, dj in ((0, 0), (1, 0), (0, 1), (1, 1)):
            if gid[i+di, j+dj] < 0:
                gid[i+di, j+dj] = nid; nid += 1
    nd = np.zeros((nid, 2))
    for i in range(ng):
        for j in range(ng):
            if gid[i, j] >= 0:
                nd[gid[i, j]] = (xa[i], xa[j])
    tri = []
    for (i, j) in qc:
        n00, n10, n01, n11 = gid[i, j], gid[i+1, j], gid[i, j+1], gid[i+1, j+1]
        if (i + j) % 2 == 0:
            tri += [[n00, n10, n11], [n00, n11, n01]]
        else:
            tri += [[n00, n10, n01], [n10, n11, n01]]
    tri = np.array(tri, int)
    nn, ne = len(nd), len(tri)

    S = np.zeros((6, 6))
    S[0, 0], S[1, 1], S[2, 2] = 1/E[0], 1/E[1], 1/E[2]
    S[0, 1] = S[1, 0] = -nu[0]/E[0]
    S[0, 2] = S[2, 0] = -nu[1]/E[0]
    S[1, 2] = S[2, 1] = -nu[2]/E[1]
    S[3, 3], S[4, 4], S[5, 5] = 1/G[2], 1/G[1], 1/G[0]
    C0 = np.linalg.inv(S)

    p1, p2, p3 = nd[tri[:, 0]], nd[tri[:, 1]], nd[tri[:, 2]]
    det = ((p2[:, 0]-p1[:, 0])*(p3[:, 1]-p1[:, 1])
           - (p3[:, 0]-p1[:, 0])*(p2[:, 1]-p1[:, 1]))
    area = 0.5*np.abs(det)
    b = np.stack([p2[:, 1]-p3[:, 1], p3[:, 1]-p1[:, 1],
                  p1[:, 1]-p2[:, 1]], 1)/det[:, None]
    cs = np.stack([p3[:, 0]-p2[:, 0], p1[:, 0]-p3[:, 0],
                   p2[:, 0]-p1[:, 0]], 1)/det[:, None]
    B = np.zeros((ne, 6, 9))
    for k in range(3):
        B[:, 1, 3*k+1] = b[:, k]
        B[:, 2, 3*k+2] = cs[:, k]
        B[:, 3, 3*k+1] = cs[:, k]; B[:, 3, 3*k+2] = b[:, k]
        B[:, 4, 3*k] = cs[:, k]
        B[:, 5, 3*k] = b[:, k]
    Ke = np.einsum('e,eia,ij,ejb->eab', area, B, C0, B)
    Fe = np.einsum('e,eia,ij->eaj', area, B, C0)
    Dee = C0 * float(area.sum())
    ndof = 3*nn
    gd = (3*tri[:, :, None] + np.arange(3)[None, None, :]).reshape(ne, 9)
    K = np.zeros((ndof, ndof)); Dhe = np.zeros((ndof, 6))
    np.add.at(K, (gd[:, :, None].repeat(9, 2), gd[:, None, :].repeat(9, 1)), Ke)
    np.add.at(Dhe, gd.ravel(), Fe.reshape(-1, 6))

    rc, _ = mesh_to_periodic_sparse_assembly_map(nn, np.arange(nn)[:, None],
                                                 nd, 3, 3)
    master = np.asarray(rc, int).ravel()
    uniq, inv = np.unique(master, return_inverse=True)
    nred = 3*len(uniq)
    T = np.zeros((ndof, nred))
    for n in range(nn):
        for k in range(3):
            T[3*n+k, 3*inv[n]+k] = 1.0
    Kr = T.T @ K @ T; Fr = T.T @ Dhe
    wA = np.zeros(nn)
    np.add.at(wA, tri.ravel(), np.repeat(area/3.0, 3))
    wAr = np.zeros(len(uniq))
    np.add.at(wAr, inv, wA)
    Cc = np.zeros((3, nred))
    Cc[0, 0::3] = wAr; Cc[1, 1::3] = wAr; Cc[2, 2::3] = wAr
    A = np.zeros((nred+3, nred+3))
    A[:nred, :nred] = Kr; A[:nred, nred:] = Cc.T; A[nred:, :nred] = Cc
    R = np.zeros((nred+3, 6)); R[:nred] = -Fr
    V0 = T @ np.linalg.solve(A, R)[:nred]
    Cs = (Dee + V0.T @ Dhe) / (4*a*t_w)
    return 0.5*(Cs + Cs.T)


print("A. MESH refinement at t/a = 0.03 (ratios shell/solid)")
print("  %-14s %10s %10s %10s %10s" % ("nseg / nt", "C11", "C22", "C44", "C55"))
for nseg, nt in MESH_SWEEP:
    Csh, Cso = shell_C(0.03, nseg), solid_C(0.03, nt)
    print("  %-14s %10.4f %10.4f %10.4f %10.4f"
          % ("%d / %d" % (nseg, nt), Csh[0, 0]/Cso[0, 0], Csh[1, 1]/Cso[1, 1],
             Csh[3, 3]/Cso[3, 3], Csh[4, 4]/Cso[4, 4]))

print("\nB. THIN-WALL sweep at nseg=40, nt=8 (ratios shell/solid)")
print("  %-8s %10s %10s %10s %10s" % ("t/a", "C11", "C22", "C44", "C55"))
for t_w in T_SWEEP:
    Csh, Cso = shell_C(t_w, 40), solid_C(t_w, 8)
    print("  %-8.4f %10.4f %10.4f %10.4f %10.4f"
          % (t_w/a, Csh[0, 0]/Cso[0, 0], Csh[1, 1]/Cso[1, 1],
             Csh[3, 3]/Cso[3, 3], Csh[4, 4]/Cso[4, 4]))
