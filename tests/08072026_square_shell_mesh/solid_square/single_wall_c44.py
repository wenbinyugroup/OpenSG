"""THE decisive C44 test: ONE wall, periodic ends, under 2G23 alone.

  shell:  a single straight wall along y2 (length L, thickness t, iso), its two
          end nodes tied periodically, homogenized by ring_solid.  C44 is read
          from D_eff[3,3]/(L*t)  (normalized by the wall material area).
  solid:  the same L x t strip meshed 2-D (CST, aspect ~1), left/right edges
          tied periodically by the same core map, 3 translation pins.

No junctions, no second wall family, no cell-construction question: whatever
C44 the solid strip gives IS the reference for the shell's drive+relaxation of
the 2G23 mode.  Everything else in the 6x6 rides along for free.

Run (from this folder):  python single_wall_c44.py
"""
import numpy as np

from opensg_shell.solid_props import ring_solid
from opensg_shell.oml_ring import load_ring_ref
from opensg_shell.emit_abd import material_db_from_yaml
from opensg_solid.rm_plate_1D.msg_rm_plate import rm_plate_msg
from opensg_shell.periodic_multiscale import mesh_to_periodic_sparse_assembly_map
import yaml as _yaml

############### User Input #################################
L_w, t_w = 0.10, 0.03
E, nu = 70.0e9, 0.30
nseg = 20                     # shell elements along the wall
nt = 6                        # solid elements through t (along-wall size ~ t/nt)
############################################################

G_iso = E/(2*(1+nu))
A_mat = L_w*t_w

# ---- shell: 1-D wall along y2 at y3 = 0, periodic ends ---------------------
pts = np.stack([np.linspace(0, L_w, nseg+1), np.zeros(nseg+1)], 1)
o = ["nodes:"]
for x, y in pts:
    o.append("- [%.12f %.12f 0.00000000]" % (x, y))
o.append("elements:")
for i in range(nseg):
    o.append("- [%d %d]" % (i+1, i+2))
o.append("elementOrientations:")
for i in range(nseg):
    o.append("- [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]")
o += ["sections:", "- type: shell", "  elementSet: layup_0", "  layup:",
      "  - - iso", "    - %.6e" % t_w, "    - 0.0",
      "materials:", "- name: iso", "  density: 2700.0", "  elastic:",
      "    E: [%.6e, %.6e, %.6e]" % (E, E, E),
      "    G: [%.6e, %.6e, %.6e]" % (G_iso, G_iso, G_iso),
      "    nu: [%.6f, %.6f, %.6f]" % (nu, nu, nu),
      "sets:", "  element:", "  - name: layup_0", "    labels:"]
o += ["    - %d" % (e+1) for e in range(nseg)]
o.append("reference: center")
open("_wall.yaml", "w").write("\n".join(o) + "\n")

R = load_ring_ref("_wall.yaml", "center")
d_sh = _yaml.safe_load(open("_wall.yaml"))
mdb = material_db_from_yaml(d_sh["materials"])
G_by = list(R["G_by"])
r = rm_plate_msg([t_w], [0.0], ["iso"], mdb, fraction=0.5)
if r["G_msg"] is not None:
    G_by[0] = np.asarray(r["G_msg"])

Deff = ring_solid(R["rx"], R["cells"], R["rsub"], R["re3"], R["D_by"], G_by,
                  R["k22"], R["ax"], R["cross"], shear="mitc4_g23",
                  lam_space="elem", periodic=True)
C_sh = np.asarray(Deff)/A_mat

# ---- solid: L x t strip, periodic left/right -------------------------------
na = max(8, int(round(L_w/(t_w/nt))))
xg = np.linspace(0, L_w, na+1)
yg = np.linspace(-t_w/2, t_w/2, nt+1)
nx, ny = len(xg), len(yg)
nd = np.array([[xg[i], yg[j]] for i in range(nx) for j in range(ny)])
nid = lambda i, j: i*ny + j
tri = []
for i in range(nx-1):
    for j in range(ny-1):
        n00, n10 = nid(i, j), nid(i+1, j)
        n01, n11 = nid(i, j+1), nid(i+1, j+1)
        if (i + j) % 2 == 0:
            tri += [[n00, n10, n11], [n00, n11, n01]]
        else:
            tri += [[n00, n10, n01], [n10, n11, n01]]
tri = np.array(tri, int)
nn, ne = len(nd), len(tri)

lam = E*nu/((1+nu)*(1-2*nu)); mu = G_iso
C0 = np.zeros((6, 6)); C0[:3, :3] = lam
C0[np.arange(3), np.arange(3)] = lam + 2*mu
C0[3, 3] = C0[4, 4] = C0[5, 5] = mu

p1, p2, p3 = nd[tri[:, 0]], nd[tri[:, 1]], nd[tri[:, 2]]
det = ((p2[:, 0]-p1[:, 0])*(p3[:, 1]-p1[:, 1])
       - (p3[:, 0]-p1[:, 0])*(p2[:, 1]-p1[:, 1]))
area = 0.5*np.abs(det)
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
Dee = C0*float(area.sum())
ndof = 3*nn
gd = (3*tri[:, :, None] + np.arange(3)[None, None, :]).reshape(ne, 9)
K = np.zeros((ndof, ndof)); Dhe = np.zeros((ndof, 6))
np.add.at(K, (gd[:, :, None].repeat(9, 2), gd[:, None, :].repeat(9, 1)), Ke)
np.add.at(Dhe, gd.ravel(), Fe.reshape(-1, 6))

rc, _ = mesh_to_periodic_sparse_assembly_map(nn, np.arange(nn)[:, None],
                                             nd, 2, 3)   # n_model=2: tie x only
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
A[:nred, :nred] = T.T @ K @ T; A[:nred, nred:] = Cc.T; A[nred:, :nred] = Cc
Rhs = np.zeros((nred+3, 6)); Rhs[:nred] = -(T.T @ Dhe)
V0 = T @ np.linalg.solve(A, Rhs)[:nred]
C_so = (Dee + V0.T @ Dhe)/A_mat
C_so = 0.5*(C_so + C_so.T)

print("single wall along y2, periodic ends, iso, L=%.2f t=%.2f,"
      " normalized by L*t" % (L_w, t_w))
print("solid mesh: %d x %d grid (%d tris)" % (na, nt, ne))
print("\n  %-5s %14s %14s %9s" % ("term", "shell", "solid", "ratio"))
thr = 1e-6*max(np.max(np.abs(C_sh)), np.max(np.abs(C_so)))
for i in range(6):
    for j in range(i, 6):
        a, b2 = C_sh[i, j], C_so[i, j]
        if abs(a) > thr or abs(b2) > thr:
            rr = a/b2 if abs(b2) > thr else np.inf
            print("  C%d%d   %14.5e %14.5e %9.4f" % (i+1, j+1, a, b2, rr))
