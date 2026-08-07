"""Closed SQUARE tube cross-section: shell 1-D line SG (center ref) vs solid
2-D SG -- equivalent 3-D solid properties.

Centerline square, side a = 1, wall t = 0.03, iso E = 70 GPa nu = 0.3 (same
wall as the circle tests).  SHELL = closed 4-wall ring yaml, center reference,
opensg_shell.build_solid_bundle.  SOLID = structured square-annulus CST mesh,
free-SG constraints (<w_i> = 0, rotation pin).  Both normalized by the
centerline-enclosed area a^2.  Voigt [e11 e22 e33 2e23 2e13 2e12].

Run (from this folder):  python square_tube_shell_vs_solid.py
"""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from opensg_shell import build_solid_bundle, GBAR_ORDER
from opensg_shell.solid_props import elastic_constants

############### User Input #################################
a_sq = 1.0               # centerline square side
t_w = 0.03               # wall thickness
E, nu = 70.0e9, 0.30
nseg = 36                # shell elements per side
nt, na = 4, 48           # solid: through-thickness / along-side divisions
############################################################

G = E/(2*(1+nu))
A_cell = a_sq**2
t0 = time.perf_counter()

# ---------------- SHELL: closed square ring yaml (center ref) ---------------
hs = a_sq/2
corners = [(-hs, -hs), (hs, -hs), (hs, hs), (-hs, hs)]
pts, tans = [], []
for k in range(4):
    p0, p1 = np.array(corners[k]), np.array(corners[(k+1) % 4])
    for i in range(nseg):
        pts.append(p0 + (p1-p0)*i/nseg)
        tans.append((p1-p0)/np.linalg.norm(p1-p0))
pts = np.array(pts)
m = len(pts)
out = ["nodes:"]
for x, y in pts:
    out.append("- [%.12f %.12f 0.00000000]" % (x, y))
out.append("elements:")
for i in range(m):
    out.append("- [%d %d]" % (i+1, (i+1) % m + 1))
out.append("elementOrientations:")
for i in range(m):
    tx, ty = tans[i]
    e3 = np.cross([0, 0, 1.0], [tx, ty, 0])
    out.append("- [0.0, 0.0, 1.0, %.9f, %.9f, 0.0, %.9f, %.9f, 0.0]"
               % (tx, ty, e3[0], e3[1]))
out += ["sections:",
        "- type: shell",
        "  elementSet: layup_0",
        "  layup:",
        "  - - ply",
        "    - %.6e" % t_w,
        "    - 0.0",
        "materials:",
        "- name: ply",
        "  density: 1600.0",
        "  elastic:",
        "    E: [%.6e, %.6e, %.6e]" % (E, E, E),
        "    G: [%.6e, %.6e, %.6e]" % (G, G, G),
        "    nu: [%.6f, %.6f, %.6f]" % (nu, nu, nu),
        "sets:",
        "  element:",
        "  - name: layup_0",
        "    labels:"]
out += ["    - %d" % (e+1) for e in range(m)]
out.append("reference: center")
with open("square_tube_shell.yaml", "w") as f:
    f.write("\n".join(out) + "\n")

Bs = build_solid_bundle("square_tube_shell.yaml", cell_area=A_cell)
C_sh = np.asarray(Bs["C3D"])
t_shell = time.perf_counter() - t0
print("shell ring done: %d nodes/elems  [%.1f s]" % (m, t_shell))

# ---------------- SOLID: structured square annulus --------------------------
t1 = time.perf_counter()
lo, hi = hs - t_w/2, hs + t_w/2
g1 = np.linspace(-hi, -lo, nt+1)
g2 = np.linspace(-lo, lo, na+1)[1:-1]
g3 = np.linspace(lo, hi, nt+1)
xa = np.concatenate([g1, g2, g3])
ng = len(xa)
gid = -np.ones((ng, ng), int)
qc = []
for i in range(ng-1):
    for j in range(ng-1):
        xc, yc = 0.5*(xa[i]+xa[i+1]), 0.5*(xa[j]+xa[j+1])
        if abs(xc) > lo or abs(yc) > lo:
            qc.append((i, j))
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
    tri += [[n00, n10, n11], [n00, n11, n01]]
tri = np.array(tri, int)
nn, ne = len(nd), len(tri)

lam = E*nu/((1+nu)*(1-2*nu)); mu = G
C0 = np.zeros((6, 6)); C0[:3, :3] = lam
C0[np.arange(3), np.arange(3)] = lam + 2*mu
C0[3, 3] = C0[4, 4] = C0[5, 5] = mu

p1, p2, p3 = nd[tri[:, 0]], nd[tri[:, 1]], nd[tri[:, 2]]
det = ((p2[:, 0]-p1[:, 0])*(p3[:, 1]-p1[:, 1])
       - (p3[:, 0]-p1[:, 0])*(p2[:, 1]-p1[:, 1]))
area = 0.5*np.abs(det)
A_mesh = float(area.sum())
b = np.stack([p2[:, 1]-p3[:, 1], p3[:, 1]-p1[:, 1], p1[:, 1]-p2[:, 1]], 1)/det[:, None]
cs = np.stack([p3[:, 0]-p2[:, 0], p1[:, 0]-p3[:, 0], p2[:, 0]-p1[:, 0]], 1)/det[:, None]
B = np.zeros((ne, 6, 9))
for k in range(3):
    B[:, 1, 3*k+1] = b[:, k]
    B[:, 2, 3*k+2] = cs[:, k]
    B[:, 3, 3*k+1] = cs[:, k]; B[:, 3, 3*k+2] = b[:, k]
    B[:, 4, 3*k] = cs[:, k]
    B[:, 5, 3*k] = b[:, k]
Ke = np.einsum('e,eia,ij,ejb->eab', area, B, C0, B)
Fe = np.einsum('e,eia,ij->eaj', area, B, C0)
Dee = C0 * A_mesh

ndof = 3*nn
gdof = (3*tri[:, :, None] + np.arange(3)[None, None, :]).reshape(ne, 9)
K = np.zeros((ndof, ndof)); Dhe = np.zeros((ndof, 6))
np.add.at(K, (gdof[:, :, None].repeat(9, 2), gdof[:, None, :].repeat(9, 1)), Ke)
np.add.at(Dhe, gdof.ravel(), Fe.reshape(-1, 6))

wA = np.zeros(nn)
np.add.at(wA, tri.ravel(), np.repeat(area/3.0, 3))
Cc = np.zeros((4, ndof))
Cc[0, 0::3] = wA; Cc[1, 1::3] = wA; Cc[2, 2::3] = wA
Cc[3, 1::3] = -wA*nd[:, 1]; Cc[3, 2::3] = wA*nd[:, 0]
A_k = np.zeros((ndof+4, ndof+4))
A_k[:ndof, :ndof] = K; A_k[:ndof, ndof:] = Cc.T; A_k[ndof:, :ndof] = Cc
Rhs = np.zeros((ndof+4, 6)); Rhs[:ndof] = -Dhe
V0 = np.linalg.solve(A_k, Rhs)[:ndof]
C_so = (Dee + V0.T @ Dhe) / A_cell
C_so = 0.5*(C_so + C_so.T)
t_solid = time.perf_counter() - t1
# mitred square annulus: (a+t)^2 - (a-t)^2 = 4at exactly (corners counted once)
print("solid annulus done: %d nodes, %d tris, A_mesh=%.6f (exact 4at=%.6f)"
      "  [%.1f s]" % (nn, ne, A_mesh, 4*a_sq*t_w, t_solid))

# ---------------- outputs ---------------------------------------------------
def write_out(path, tag, Cm, dt):
    cons, S = elastic_constants(Cm)
    with open(path, "w") as f:
        f.write("# closed SQUARE tube a=%.2f t=%.3f iso E=%.3e nu=%.2f, %s\n"
                % (a_sq, t_w, E, nu, tag))
        f.write("# %s; A_cell=%.6f; solve %.1f s\n" % (GBAR_ORDER, A_cell, dt))
        f.write("# ---- C3D (6x6, Pa) ----\n")
        for i in range(6):
            f.write(" ".join("%16.8e" % Cm[i, j] for j in range(6)) + "\n")
        f.write("# ---- 9 constants ----\n")
        for kx in ("E1", "E2", "E3", "G23", "G13", "G12",
                   "nu12", "nu13", "nu23"):
            f.write("%-6s %16.8e\n" % (kx, cons[kx]))


write_out("square_tube_shell.out", "SHELL 1-D ring, center ref", C_sh, t_shell)
write_out("square_tube_solid.out", "SOLID 2-D annulus, free SG", C_so, t_solid)

thr = 1e-8 * max(np.max(np.abs(C_sh)), np.max(np.abs(C_so)))
print("\nresolved C_ij, shell vs solid (free closed section):")
print("  %-5s %16s %16s %10s" % ("term", "C_shell", "C_solid", "pct"))
with open("square_tube_compare.dat", "w") as f:
    f.write("# closed square tube: shell (center ref) vs solid; %s\n"
            % GBAR_ORDER)
    f.write("# analytical anchor (both routes): C11 = E*4at/a^2 = %.6e Pa;"
            " mitred annulus area = 4at exactly\n" % (E*4*a_sq*t_w/A_cell))
    f.write("# %-5s %16s %16s %10s\n" % ("term", "C_shell", "C_solid", "pct"))
    for i in range(6):
        for j in range(i, 6):
            sh, so = C_sh[i, j], C_so[i, j]
            if abs(sh) > thr or abs(so) > thr:
                pct = 100*(sh-so)/so if abs(so) > thr else float("inf")
                print("  C%d%d   %16.8e %16.8e %+10.3f" % (i+1, j+1, sh, so, pct))
                f.write("  C%d%d   %16.8e %16.8e %+10.4f\n"
                        % (i+1, j+1, sh, so, pct))

fig, axs = plt.subplots(1, 2, figsize=(11, 5.5))
axs[0].tripcolor(nd[:, 0], nd[:, 1], tri, facecolors=np.zeros(ne),
                 cmap="Blues", vmin=-1, vmax=1, edgecolors="k", linewidth=0.2)
axs[0].set_aspect("equal"); axs[0].set_xlabel("y2"); axs[0].set_ylabel("y3")
ring = np.vstack([pts, pts[:1]])
axs[1].plot(ring[:, 0], ring[:, 1], "b-", lw=1.2)
axs[1].plot(pts[:, 0], pts[:, 1], "k.", ms=3)
axs[1].set_aspect("equal"); axs[1].set_xlabel("y2"); axs[1].set_ylabel("y3")
lim = hs + 3*t_w
axs[1].set_xlim(-lim, lim); axs[1].set_ylim(-lim, lim)
fig.tight_layout(); fig.savefig("square_tube_meshes.png", dpi=200)
print("\nwrote square_tube_shell.out, square_tube_solid.out,"
      " square_tube_compare.dat, square_tube_meshes.png")
