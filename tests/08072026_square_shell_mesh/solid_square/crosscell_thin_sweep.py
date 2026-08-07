"""Is msg_shell's solid-prop route shear-locking as the wall thins?

The RING cell cannot answer this: its walls sit ON the cell boundary, so the
solid periodic tie bonds neighbouring walls into one 2t wall while the shell
keeps two independent t walls -- a t-independent bending factor 4 that hits
C44 alone.  The CROSS cell (one wall of each family running THROUGH the cell)
has no such mismatch: arms meet end-to-end across the boundary, walls are t
thick in both routes, material 2Lt - t^2 in both.  There every term must tend
to 1.000 as t/L -> 0, and C44 must approach the exact guided-guided value
    C44 = 0.5 * E' * (t/L)^3,   E' = E/(1-nu^2)   (per cell area).

Any degradation HERE would be genuine locking.  Isotropic wall, so nothing
material-specific can enter.

Run (from this folder):  python crosscell_thin_sweep.py
"""
import numpy as np

from opensg_shell import build_solid_bundle
from opensg_shell.periodic_multiscale import mesh_to_periodic_sparse_assembly_map

############### User Input #################################
L = 1.0
E_iso, nu_iso = 70.0e9, 0.30
T_SWEEP = [0.10, 0.05, 0.025, 0.0125]
nseg, nt, na = 40, 6, 30        # shell elems/wall; solid elems across t / along arm
############################################################

G_iso = E_iso/(2*(1+nu_iso))
Ep = E_iso/(1-nu_iso**2)
c = L/2


def shell_C(t_w):
    xs = np.linspace(0.0, L, nseg+1)
    h_n = np.stack([xs, np.full(nseg+1, c)], 1)
    v_n = np.stack([np.full(nseg+1, c), xs], 1)
    ic = nseg//2
    keep = [j for j in range(nseg+1) if j != ic]
    pts = np.vstack([h_n, v_n[keep]])
    vid, k = {}, len(h_n)
    for j in range(nseg+1):
        if j == ic:
            vid[j] = ic
        else:
            vid[j] = k; k += 1
    cells = ([[i, i+1] for i in range(nseg)]
             + [[vid[j], vid[j+1]] for j in range(nseg)])
    ori = ([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]*nseg
           + [[0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0]]*nseg)
    o = ["nodes:"]
    for x, y in pts:
        o.append("- [%.12f %.12f 0.00000000]" % (x, y))
    o.append("elements:")
    for a_, b_ in cells:
        o.append("- [%d %d]" % (a_+1, b_+1))
    o.append("elementOrientations:")
    for r in ori:
        o.append("- [" + ", ".join("%.1f" % v for v in r) + "]")
    o += ["sections:", "- type: shell", "  elementSet: layup_0", "  layup:",
          "  - - iso", "    - %.6e" % t_w, "    - 0.0",
          "materials:", "- name: iso", "  density: 2700.0", "  elastic:",
          "    E: [%.6e, %.6e, %.6e]" % (E_iso, E_iso, E_iso),
          "    G: [%.6e, %.6e, %.6e]" % (G_iso, G_iso, G_iso),
          "    nu: [%.6f, %.6f, %.6f]" % (nu_iso, nu_iso, nu_iso),
          "sets:", "  element:", "  - name: layup_0", "    labels:"]
    o += ["    - %d" % (e+1) for e in range(len(cells))]
    o.append("reference: center")
    open("_cross.yaml", "w").write("\n".join(o) + "\n")
    return np.asarray(build_solid_bundle("_cross.yaml", cell_area=L*L)["C3D"])


def solid_C(t_w):
    lo, hi = c - t_w/2, c + t_w/2
    xa = np.unique(np.concatenate([np.linspace(0, lo, na+1),
                                   np.linspace(lo, hi, nt+1),
                                   np.linspace(hi, L, na+1)]))
    ng = len(xa)
    gid = -np.ones((ng, ng), int)
    qc = [(i, j) for i in range(ng-1) for j in range(ng-1)
          if abs(0.5*(xa[i]+xa[i+1]) - c) < t_w/2
          or abs(0.5*(xa[j]+xa[j+1]) - c) < t_w/2]
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

    lam = E_iso*nu_iso/((1+nu_iso)*(1-2*nu_iso))
    C0 = np.zeros((6, 6)); C0[:3, :3] = lam
    C0[np.arange(3), np.arange(3)] = lam + 2*G_iso
    C0[3, 3] = C0[4, 4] = C0[5, 5] = G_iso

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
    Dee = C0*float(area.sum())
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
    wA = np.zeros(nn)
    np.add.at(wA, tri.ravel(), np.repeat(area/3.0, 3))
    wAr = np.zeros(len(uniq))
    np.add.at(wAr, inv, wA)
    Cc = np.zeros((3, nred))
    Cc[0, 0::3] = wAr; Cc[1, 1::3] = wAr; Cc[2, 2::3] = wAr
    A = np.zeros((nred+3, nred+3))
    A[:nred, :nred] = (T.T @ K @ T); A[:nred, nred:] = Cc.T; A[nred:, :nred] = Cc
    R = np.zeros((nred+3, 6)); R[:nred] = -(T.T @ Dhe)
    V0 = T @ np.linalg.solve(A, R)[:nred]
    Cs = (Dee + V0.T @ Dhe)/(L*L)
    return 0.5*(Cs + Cs.T)


print("CROSS cell, isotropic, ratios shell/solid   (all must -> 1.000)")
print("  %-8s %8s %8s %8s %8s %8s   %14s %14s" %
      ("t/L", "C11", "C22", "C44", "C55", "C66", "C44 shell", "C44 exact"))
for t_w in T_SWEEP:
    Csh, Cso = shell_C(t_w), solid_C(t_w)
    exact = 0.5*Ep*(t_w/L)**3
    print("  %-8.4f %8.4f %8.4f %8.4f %8.4f %8.4f   %14.6e %14.6e" %
          (t_w/L, Csh[0, 0]/Cso[0, 0], Csh[1, 1]/Cso[1, 1], Csh[3, 3]/Cso[3, 3],
           Csh[4, 4]/Cso[4, 4], Csh[5, 5]/Cso[5, 5], Csh[3, 3], exact))
