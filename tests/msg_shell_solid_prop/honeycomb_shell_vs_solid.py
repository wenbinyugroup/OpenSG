"""Square-honeycomb periodic unit cell: SHELL-route vs SOLID-route equivalent
3-D properties (the Deo-Yu cellular-solid benchmark).

Unit cell: L x L square in the 2-3 plane, cross-shaped wall pattern (one wall
along each in-plane axis, thickness t, crossing at the center), extruded along
axis 1.  Both routes are PERIODIC in-plane -- the ingredient the free
cross-section SGs (circle, airfoil) lack, which is why they are rank-1:

  SOLID  2-D CST mesh of the wall material, generalized plane strain, periodic
         master-slave pairing of the wall-cut faces on the cell boundary,
         3 translation pins, D_eff = (Dee + V0^T Dhe)/L^2  (validated on the
         UD-composite cell vs FEniCS/ASC-23 to 4 decimals).
  SHELL  1-D SG of the wall contour (opensg_shell ring machinery, the author's
         Gamma_e/Gamma_h Cab formulation), with in-plane periodicity imposed by
         MERGING the paired boundary endpoints in the strip dof_map (the paired
         wall ends carry identical local frames, so identifying all 6 DOFs is
         exact), translation-only 3-mode kernel (periodicity kills the rigid
         rotation), C3D = D_eff / L^2.

Voigt order [e11 e22 e33 2e23 2e13 2e12], 1 = extrusion axis (global z).

Run (from this folder):  python honeycomb_shell_vs_solid.py
"""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml as _yaml
from scipy.linalg import lu_factor, lu_solve

from opensg_shell.segment_indep import (assemble_segment_indep,
                                        assemble_constraint, NDOF6)
from opensg_shell.solid_props import (assemble_solid_macro, elastic_constants,
                                      GBAR_ORDER)
from opensg_shell.fe_jax.msg_rm_timo import build_C_Psi
from opensg_shell.oml_ring import load_ring_ref

############### User Input #################################
L = 1.0                  # cell size
t_w = 0.1                # wall thickness
E, nu = 70.0e9, 0.30     # isotropic wall material
nseg = 40                # shell contour elements per wall (even)
nt, na = 6, 40           # solid: elements through t / along each half-wall arm
PAPER = None             # {"E1": ..., ...} Deo-Yu published 9 constants (GPa)
############################################################

G = E / (2*(1+nu))
A_cell = L * L
c = L / 2.0
rho_bar = (2*L*t_w - t_w**2) / A_cell
t0 = time.perf_counter()

# ============================================================================
# SHELL route
# ============================================================================
# --- contour: horizontal wall y=c (nseg elems) + vertical wall x=c, shared
# center node; endpoints at the 4 cell faces are the periodic pairs.
xs = np.linspace(0.0, L, nseg + 1)
h_nodes = np.stack([xs, np.full(nseg+1, c), np.zeros(nseg+1)], 1)
v_nodes = np.stack([np.full(nseg+1, c), xs, np.zeros(nseg+1)], 1)
ic = nseg // 2                                   # center index on each wall
v_keep = [j for j in range(nseg+1) if j != ic]   # drop duplicate center node
nodes3 = np.vstack([h_nodes, v_nodes[v_keep]])
vid = {}                                         # vertical-wall j -> global id
k = len(h_nodes)
for j in range(nseg+1):
    if j == ic:
        vid[j] = ic                              # shared center = horizontal ic
    else:
        vid[j] = k; k += 1
cells = ([[i, i+1] for i in range(nseg)]
         + [[vid[j], vid[j+1]] for j in range(nseg)])
m = len(nodes3)
# orientations: e1 = axial z; horizontal e2 = +x -> e3 = +y; vertical e2 = +y
# -> e3 = -x  (right-handed)
ori = ([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]] * nseg
       + [[0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0]] * nseg)

out = ["nodes:"]
for x, y, z in nodes3:
    out.append("- [%.12f %.12f %.12f]" % (x, y, z))
out.append("elements:")
for a, b in cells:
    out.append("- [%d %d]" % (a+1, b+1))
out.append("elementOrientations:")
for r in ori:
    out.append("- [" + ", ".join("%.1f" % v for v in r) + "]")
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
out += ["    - %d" % (e+1) for e in range(len(cells))]
out.append("reference: center")
with open("honeycomb_shell.yaml", "w") as f:
    f.write("\n".join(out) + "\n")

R = load_ring_ref("honeycomb_shell.yaml", "center")
rx, rcells = R["rx"], R["cells"]
ax, cross = R["ax"], R["cross"]
G_by = list(R["G_by"])
# MSG wall transverse-shear upgrade (same as build_solid_bundle)
from opensg_solid.rm_plate_1D.msg_rm_plate import rm_plate_msg
from opensg_shell.emit_abd import material_db_from_yaml
d_sh = _yaml.safe_load(open("honeycomb_shell.yaml"))
_mdb = material_db_from_yaml(d_sh["materials"])
for si, sec in enumerate(d_sh["sections"]):
    _pl = [[str(p[0]), float(p[1]), float(p[2])] for p in sec["layup"]]
    _rr = rm_plate_msg([p[1] for p in _pl], [p[2] for p in _pl],
                       [p[0] for p in _pl], _mdb, fraction=0.5)
    if _rr["G_msg"] is not None:
        G_by[si] = np.asarray(_rr["G_msg"])

# --- periodic node merge through the CORE model-aware map (n_model = 3 ties
# both in-plane directions), plus the shell frame-consistency check
from opensg_shell.periodic_map import (periodic_node_map,
                                       check_frame_consistency, rigid_modes)
mloc = len(rx)
node_map, n_unique_sh = periodic_node_map(rx[:, cross], n_model=3)
check_frame_consistency(rx[:, cross], rcells, node_map)
print("shell periodic map: %d -> %d independent nodes, kernel %d rigid modes"
      % (mloc, n_unique_sh, rigid_modes(True)))

# --- strip assembly (ring_solid with node_map merging + 3-mode kernel)
h = float(np.mean(np.linalg.norm(rx[rcells[:, 1]] - rx[rcells[:, 0]], axis=1)))
ez = np.zeros(3); ez[ax] = 1.0
nodes_st = np.vstack([rx, rx + h*ez])
dof_map = np.concatenate([node_map, node_map])
quads = np.array([[a, b, mloc+b, mloc+a] for a, b in rcells], int)
e3q = np.asarray(R["re3"])

Dhh, _, _, _, _, _ = assemble_segment_indep(
    nodes_st, quads, R["rsub"], e3q, R["D_by"], G_by, np.asarray(R["k22"]),
    cross, ax, kg_e=None, pen=0.0, dof_map=dof_map, shear="mitc4_g23")
Gc, _, _ = assemble_constraint(nodes_st, quads, R["rsub"], e3q,
                               np.asarray(R["k22"]), cross, ax,
                               dof_map=dof_map, lam_space="elem")
Dhe6, Dee6 = assemble_solid_macro(nodes_st, quads, R["rsub"], e3q, R["D_by"],
                                  G_by, cross, ax, dof_map=dof_map,
                                  shear="mitc4_g23")
Dhh = np.asarray(Dhh) / h; Gc = Gc / h; Dhe6 = Dhe6 / h; Dee6 = Dee6 / h

M = Dhh.shape[0]; P = Gc.shape[0]
C5, _ = build_C_Psi(rx[:, cross], rcells, p=1)
C3 = np.zeros((3, M))
for n in range(mloc):
    s = NDOF6 * node_map[n]
    C3[:, s:s+5] += C5[:3, 5*n:5*n+5]        # translations only: periodicity
                                             # kills the rigid rotation mode
naug = M + P
A = np.zeros((naug + 3, naug + 3))
A[:M, :M] = Dhh; A[:M, M:naug] = Gc.T; A[M:naug, :M] = Gc
A[:M, naug:] = C3.T; A[naug:, :M] = C3
R0 = np.zeros((naug + 3, 6)); R0[:M] = -Dhe6
V0s = lu_solve(lu_factor(A), R0)[:naug]
Deff_sh = Dee6 + V0s[:M].T @ Dhe6
C_sh = 0.5*(Deff_sh + Deff_sh.T) / A_cell
t_shell = time.perf_counter() - t0
print("shell route done: %d contour nodes (%d merged), %d elems  [%.1f s]"
      % (mloc, mloc - len(np.unique(node_map)), len(rcells), t_shell))

# ============================================================================
# SOLID route
# ============================================================================
t1 = time.perf_counter()
lo, hi = c - t_w/2, c + t_w/2
xa = np.unique(np.concatenate([np.linspace(0, lo, na+1),
                               np.linspace(lo, hi, nt+1),
                               np.linspace(hi, L, na+1)]))
ng = len(xa)
XX, YY = np.meshgrid(xa, xa, indexing="ij")
gid = -np.ones((ng, ng), int)
qc = []
for i in range(ng-1):
    for j in range(ng-1):
        xc, yc = 0.5*(xa[i]+xa[i+1]), 0.5*(xa[j]+xa[j+1])
        if abs(yc-c) < t_w/2 or abs(xc-c) < t_w/2:
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
    n00, n10 = gid[i, j], gid[i+1, j]
    n01, n11 = gid[i, j+1], gid[i+1, j+1]
    tri += [[n00, n10, n11], [n00, n11, n01]]
tri = np.array(tri, int)
nn, ne = len(nd), len(tri)

lam = E*nu/((1+nu)*(1-2*nu)); mu = E/(2*(1+nu))
C0 = np.zeros((6, 6)); C0[:3, :3] = lam
C0[np.arange(3), np.arange(3)] = lam + 2*mu
C0[3, 3] = C0[4, 4] = C0[5, 5] = mu

p1, p2, p3 = nd[tri[:, 0]], nd[tri[:, 1]], nd[tri[:, 2]]
det = (p2[:, 0]-p1[:, 0])*(p3[:, 1]-p1[:, 1]) - (p3[:, 0]-p1[:, 0])*(p2[:, 1]-p1[:, 1])
area = 0.5*np.abs(det)
A_mesh = float(area.sum())
b = np.stack([p2[:, 1]-p3[:, 1], p3[:, 1]-p1[:, 1], p1[:, 1]-p2[:, 1]], 1)/det[:, None]
cs = np.stack([p3[:, 0]-p2[:, 0], p1[:, 0]-p3[:, 0], p2[:, 0]-p1[:, 0]], 1)/det[:, None]
B = np.zeros((ne, 6, 9))
for a in range(3):
    B[:, 1, 3*a+1] = b[:, a]
    B[:, 2, 3*a+2] = cs[:, a]
    B[:, 3, 3*a+1] = cs[:, a]; B[:, 3, 3*a+2] = b[:, a]
    B[:, 4, 3*a] = cs[:, a]
    B[:, 5, 3*a] = b[:, a]
Ke = np.einsum('e,eia,ij,ejb->eab', area, B, C0, B)
Fe = np.einsum('e,eia,ij->eaj', area, B, C0)
Dee = C0 * A_mesh

ndof = 3*nn
gdof = (3*tri[:, :, None] + np.arange(3)[None, None, :]).reshape(ne, 9)
K = np.zeros((ndof, ndof)); Dhe = np.zeros((ndof, 6))
np.add.at(K, (gdof[:, :, None].repeat(9, 2), gdof[:, None, :].repeat(9, 1)), Ke)
np.add.at(Dhe, gdof.ravel(), Fe.reshape(-1, 6))

# the SAME core map, 3 DOF/node (this is what opensg_solid.sg_homo feeds every
# macro model: periodicity rides in the local->global assembly map)
master, n_unique_so = periodic_node_map(nd, n_model=3)
print("solid periodic map: %d -> %d independent nodes" % (nn, n_unique_so))
uniq, inv = np.unique(master, return_inverse=True)
nred = 3*len(uniq)
T = np.zeros((ndof, nred))
for n in range(nn):
    for kk in range(3):
        T[3*n+kk, 3*inv[n]+kk] = 1.0
Kr = T.T @ K @ T
Fr = T.T @ Dhe
wA = np.zeros(nn)
np.add.at(wA, tri.ravel(), np.repeat(area/3.0, 3))
wAr = np.zeros(len(uniq))
np.add.at(wAr, inv, wA)
Cc = np.zeros((3, nred))
Cc[0, 0::3] = wAr; Cc[1, 1::3] = wAr; Cc[2, 2::3] = wAr
Akkt = np.zeros((nred+3, nred+3))
Akkt[:nred, :nred] = Kr; Akkt[:nred, nred:] = Cc.T; Akkt[nred:, :nred] = Cc
Rhs = np.zeros((nred+3, 6)); Rhs[:nred] = -Fr
V0 = T @ np.linalg.solve(Akkt, Rhs)[:nred]
C_so = (Dee + V0.T @ Dhe) / A_cell
C_so = 0.5*(C_so + C_so.T)
t_solid = time.perf_counter() - t1
print("solid route done: %d nodes, %d tris, A_mesh=%.6f (rho=%.4f, exact %.4f)"
      "  [%.1f s]" % (nn, ne, A_mesh, A_mesh/A_cell, rho_bar, t_solid))

# ============================================================================
# outputs + comparison
# ============================================================================
def write_out(path, tag, Cm, dt):
    cons, S = elastic_constants(Cm)
    with open(path, "w") as f:
        f.write("# square honeycomb UC L=%.3f t=%.3f, iso E=%.3e nu=%.2f, %s\n"
                % (L, t_w, E, nu, tag))
        f.write("# %s;  cell area = %.6f;  solve %.1f s\n" % (GBAR_ORDER, A_cell, dt))
        f.write("# ---- C3D stiffness (6x6, Pa) ----\n")
        for i in range(6):
            f.write(" ".join("%16.8e" % Cm[i, j] for j in range(6)) + "\n")
        f.write("# ---- compliance inv(C3D) ----\n")
        for i in range(6):
            f.write(" ".join("%16.8e" % S[i, j] for j in range(6)) + "\n")
        f.write("# ---- 9 engineering constants ----\n")
        for kx in ("E1", "E2", "E3", "G23", "G13", "G12", "nu12", "nu13", "nu23"):
            f.write("%-6s %16.8e\n" % (kx, cons[kx]))
    return cons


cons_sh = write_out("honeycomb_shell.out", "SHELL 1-D SG periodic", C_sh, t_shell)
cons_so = write_out("honeycomb_solid.out", "SOLID 2-D SG periodic", C_so, t_solid)

thr = 1e-8 * max(np.max(np.abs(C_sh)), np.max(np.abs(C_so)))
print("\nresolved C_ij, shell vs solid:")
print("  %-5s %16s %16s %10s" % ("term", "C_shell", "C_solid", "pct_diff"))
lines = []
for i in range(6):
    for j in range(i, 6):
        sh, so = C_sh[i, j], C_so[i, j]
        if abs(sh) > thr or abs(so) > thr:
            pct = 100.0*(sh-so)/so if abs(so) > thr else float("inf")
            print("  C%d%d   %16.8e %16.8e %+10.3f" % (i+1, j+1, sh, so, pct))
            lines.append("  C%d%d   %16.8e %16.8e %+10.4f" % (i+1, j+1, sh, so, pct))

names = ["E1", "E2", "E3", "G23", "G13", "G12", "nu12", "nu13", "nu23"]
print("\n9 effective constants (GPa / -):")
hdr = "  %-6s %14s %14s" % ("const", "shell", "solid")
if PAPER:
    hdr += " %14s %10s %10s" % ("Deo", "sh_vs_Deo", "so_vs_Deo")
print(hdr)
crows = []
for nm in names:
    a, b2 = cons_sh[nm], cons_so[nm]
    sc = 1e-9 if nm[0] in "EG" else 1.0
    rowtxt = "  %-6s %14.6f %14.6f" % (nm, a*sc, b2*sc)
    if PAPER and nm in PAPER:
        p = PAPER[nm]
        rowtxt += " %14.6f %+10.3f %+10.3f" % (p, 100*(a*sc-p)/p, 100*(b2*sc-p)/p)
    print(rowtxt)
    crows.append(rowtxt)

with open("honeycomb_compare.dat", "w") as f:
    f.write("# square honeycomb UC: shell vs solid (vs Deo paper if given)\n")
    f.write("# L=%.4f t=%.4f E=%.4e nu=%.2f rho_bar=%.5f; %s\n"
            % (L, t_w, E, nu, rho_bar, GBAR_ORDER))
    f.write("# ---- resolved stiffness terms (Pa) ----\n")
    f.write("# %-5s %16s %16s %10s\n" % ("term", "C_shell", "C_solid", "pct"))
    f.write("\n".join(lines) + "\n")
    f.write("# ---- 9 constants (E,G in GPa) ----\n")
    f.write("#" + hdr[1:] + "\n")
    f.write("\n".join(crows) + "\n")

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].tripcolor(nd[:, 0], nd[:, 1], tri, facecolors=np.zeros(ne),
                 cmap="Blues", vmin=-1, vmax=1, edgecolors="k", linewidth=0.2)
axs[0].set_aspect("equal"); axs[0].set_xlabel("y2"); axs[0].set_ylabel("y3")
for a2, b2 in rcells:
    axs[1].plot(rx[[a2, b2], cross[0]], rx[[a2, b2], cross[1]], "b-", lw=1)
axs[1].plot(rx[:, cross[0]], rx[:, cross[1]], "k.", ms=2)
for pt in ([0, c], [L, c], [c, 0], [c, L]):
    axs[1].plot(pt[0], pt[1], "rs", ms=6)
axs[1].set_aspect("equal"); axs[1].set_xlabel("y2"); axs[1].set_ylabel("y3")
axs[1].set_xlim(-0.05*L, 1.05*L); axs[1].set_ylim(-0.05*L, 1.05*L)
fig.tight_layout(); fig.savefig("honeycomb_meshes.png", dpi=200)
print("\nwrote honeycomb_shell.out, honeycomb_solid.out, honeycomb_compare.dat,"
      " honeycomb_meshes.png")
