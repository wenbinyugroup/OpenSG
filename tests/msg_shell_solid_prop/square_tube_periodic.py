"""PERIODIC square tube: shell (1-D line SG, center ref) vs solid (2-D SG),
effective 3-D elastic properties from homogenization.

Periodicity comes straight from the core JAX map,
src/fe_jax/periodic_multiscale.py -> periodic_map(points, n_model, atol):
opposite bounding-box faces are paired (right -> left by Lx, top -> bottom by
Ly for n_model = 3), corner chains resolved by repeating dof_map[dof_map], and
the element connectivity is re-pointed at the master nodes -- so periodicity
rides in the local->global assembly map, with no constraint rows.  Same call on
both sides: 3 DOF/node for the solid mesh, 6 DOF/node for the shell contour.

Geometry: centerline square side a, wall t, isotropic wall, cell area a^2.
Voigt order [e11 e22 e33 2e23 2e13 2e12], 1 = tube axis.

Run (from this folder):  python square_tube_periodic.py
"""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from opensg_shell import build_solid_bundle, GBAR_ORDER
from opensg_shell.solid_props import elastic_constants
from opensg_shell.periodic_map import periodic_node_map

############### User Input #################################
a_sq = 1.0               # centerline square side
t_w = 0.03               # wall thickness
E, nu = 70.0e9, 0.30
nseg = 36                # shell elements per side
nt, na = 4, 48           # solid: through-thickness / along-side divisions
############################################################

G = E/(2*(1+nu))
A_cell = a_sq**2
hs = a_sq/2

# ---------------- SHELL: closed square ring (center ref) --------------------
t0 = time.perf_counter()
corners = [(-hs, -hs), (hs, -hs), (hs, hs), (-hs, hs)]
pts, tans = [], []
for k in range(4):
    p0, p1 = np.array(corners[k]), np.array(corners[(k+1) % 4])
    for i in range(nseg):
        pts.append(p0 + (p1-p0)*i/nseg)
        tans.append((p1-p0)/np.linalg.norm(p1-p0))
pts = np.array(pts); m = len(pts)
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
out += ["sections:", "- type: shell", "  elementSet: layup_0", "  layup:",
        "  - - ply", "    - %.6e" % t_w, "    - 0.0",
        "materials:", "- name: ply", "  density: 1600.0", "  elastic:",
        "    E: [%.6e, %.6e, %.6e]" % (E, E, E),
        "    G: [%.6e, %.6e, %.6e]" % (G, G, G),
        "    nu: [%.6f, %.6f, %.6f]" % (nu, nu, nu),
        "sets:", "  element:", "  - name: layup_0", "    labels:"]
out += ["    - %d" % (e+1) for e in range(m)]
out.append("reference: center")
with open("square_tube_shell.yaml", "w") as f:
    f.write("\n".join(out) + "\n")

nm_sh, nu_sh = periodic_node_map(pts, n_model=3)
print("shell periodic map (core): %d -> %d independent nodes" % (m, nu_sh))
C_sh = np.asarray(build_solid_bundle("square_tube_shell.yaml",
                                     cell_area=A_cell, periodic=True)["C3D"])
t_shell = time.perf_counter() - t0

# ---------------- SOLID: square annulus, same core map ----------------------
t1 = time.perf_counter()
lo, hi = hs - t_w/2, hs + t_w/2
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

# same core map, 3 DOF/node
master, nu_so = periodic_node_map(nd, n_model=3)
print("solid periodic map (core): %d -> %d independent nodes" % (nn, nu_so))
uniq, inv = np.unique(master, return_inverse=True)
nred = 3*len(uniq)
T = np.zeros((ndof, nred))
for n in range(nn):
    for k in range(3):
        T[3*n+k, 3*inv[n]+k] = 1.0
Kr = T.T @ K @ T
Fr = T.T @ Dhe
wA = np.zeros(nn)
np.add.at(wA, tri.ravel(), np.repeat(area/3.0, 3))
wAr = np.zeros(len(uniq))
np.add.at(wAr, inv, wA)
Cc = np.zeros((3, nred))
Cc[0, 0::3] = wAr; Cc[1, 1::3] = wAr; Cc[2, 2::3] = wAr
A = np.zeros((nred+3, nred+3))
A[:nred, :nred] = Kr; A[:nred, nred:] = Cc.T; A[nred:, :nred] = Cc
Rhs = np.zeros((nred+3, 6)); Rhs[:nred] = -Fr
V0 = T @ np.linalg.solve(A, Rhs)[:nred]
C_so = (Dee + V0.T @ Dhe) / A_cell
C_so = 0.5*(C_so + C_so.T)
t_solid = time.perf_counter() - t1
print("solid mesh: %d nodes, %d tris, A_mesh=%.6f (4at=%.6f)"
      % (nn, ne, A_mesh, 4*a_sq*t_w))

# ---------------- comparison ------------------------------------------------
def dump(path, tag, Cm, dt):
    cons, S = elastic_constants(Cm)
    ev = np.linalg.eigvalsh(Cm)
    with open(path, "w") as f:
        f.write("# PERIODIC square tube a=%.2f t=%.3f iso E=%.3e nu=%.2f, %s\n"
                % (a_sq, t_w, E, nu, tag))
        f.write("# %s; cell area=%.6f; solve %.1f s\n" % (GBAR_ORDER, A_cell, dt))
        f.write("# eigenvalues: %s\n" % " ".join("%.4e" % v for v in ev))
        f.write("# ---- C3D (6x6, Pa) ----\n")
        for i in range(6):
            f.write(" ".join("%16.8e" % Cm[i, j] for j in range(6)) + "\n")
        f.write("# ---- 9 constants ----\n")
        for kx in ("E1", "E2", "E3", "G23", "G13", "G12",
                   "nu12", "nu13", "nu23"):
            f.write("%-6s %16.8e\n" % (kx, cons[kx]))
    return cons, ev


cons_sh, ev_sh = dump("square_tube_per_shell.out", "SHELL 1-D ring", C_sh, t_shell)
cons_so, ev_so = dump("square_tube_per_solid.out", "SOLID 2-D annulus", C_so, t_solid)

print("\nC3D shell (Pa):")
for i in range(6):
    print("  " + " ".join("%12.4e" % C_sh[i, j] for j in range(6)))
print("C3D solid (Pa):")
for i in range(6):
    print("  " + " ".join("%12.4e" % C_so[i, j] for j in range(6)))
print("\neigenvalues shell: %s" % " ".join("%.3e" % v for v in ev_sh))
print("eigenvalues solid: %s" % " ".join("%.3e" % v for v in ev_so))

thr = 1e-8*max(np.max(np.abs(C_sh)), np.max(np.abs(C_so)))
print("\nresolved terms, shell vs solid:")
print("  %-5s %16s %16s %10s" % ("term", "C_shell", "C_solid", "pct"))
lines = []
for i in range(6):
    for j in range(i, 6):
        sh, so = C_sh[i, j], C_so[i, j]
        if abs(sh) > thr or abs(so) > thr:
            pct = 100*(sh-so)/so if abs(so) > thr else float("inf")
            row = "  C%d%d   %16.8e %16.8e %+10.3f" % (i+1, j+1, sh, so, pct)
            print(row); lines.append(row)

names = ["E1", "E2", "E3", "G23", "G13", "G12", "nu12", "nu13", "nu23"]
print("\n9 effective constants (E,G in GPa):")
print("  %-6s %14s %14s %10s" % ("const", "shell", "solid", "pct"))
crows = []
for k in names:
    sc = 1e-9 if k[0] in "EG" else 1.0
    a, b2 = cons_sh[k]*sc, cons_so[k]*sc
    pct = 100*(a-b2)/b2 if abs(b2) > 1e-30 else float("inf")
    row = "  %-6s %14.6f %14.6f %+10.3f" % (k, a, b2, pct)
    print(row); crows.append(row)

with open("square_tube_per_compare.dat", "w") as f:
    f.write("# PERIODIC square tube, shell vs solid; periodicity from the core\n")
    f.write("# fe_jax/periodic_multiscale.py map (n_model=3: x and y paired)\n")
    f.write("# a=%.3f t=%.3f E=%.3e nu=%.2f cell=%.4f; %s\n"
            % (a_sq, t_w, E, nu, A_cell, GBAR_ORDER))
    f.write("# eigenvalues shell: %s\n" % " ".join("%.4e" % v for v in ev_sh))
    f.write("# eigenvalues solid: %s\n" % " ".join("%.4e" % v for v in ev_so))
    f.write("# ---- resolved stiffness terms (Pa) ----\n")
    f.write("# %-5s %16s %16s %10s\n" % ("term", "C_shell", "C_solid", "pct"))
    f.write("\n".join(lines) + "\n")
    f.write("# ---- 9 constants (E,G in GPa) ----\n")
    f.write("# %-6s %14s %14s %10s\n" % ("const", "shell", "solid", "pct"))
    f.write("\n".join(crows) + "\n")

fig, axs = plt.subplots(1, 2, figsize=(11, 5.5))
axs[0].tripcolor(nd[:, 0], nd[:, 1], tri, facecolors=np.zeros(ne),
                 cmap="Blues", vmin=-1, vmax=1, edgecolors="k", linewidth=0.2)
axs[0].set_aspect("equal"); axs[0].set_xlabel("y2"); axs[0].set_ylabel("y3")
ring = np.vstack([pts, pts[:1]])
axs[1].plot(ring[:, 0], ring[:, 1], "b-", lw=1.2)
axs[1].plot(pts[:, 0], pts[:, 1], "k.", ms=3)
tied = np.where(nm_sh != np.arange(m))[0]
axs[1].plot(pts[tied, 0], pts[tied, 1], "rs", ms=5, mfc="none")
axs[1].set_aspect("equal"); axs[1].set_xlabel("y2"); axs[1].set_ylabel("y3")
lim = hs + 3*t_w
axs[1].set_xlim(-lim, lim); axs[1].set_ylim(-lim, lim)
fig.tight_layout(); fig.savefig("square_tube_per_meshes.png", dpi=200)
print("\nwrote square_tube_per_shell.out, square_tube_per_solid.out,"
      " square_tube_per_compare.dat, square_tube_per_meshes.png")
