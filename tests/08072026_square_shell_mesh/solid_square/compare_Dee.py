"""Compare Dee (pure macro energy, NO fluctuation) between the two routes.

    shell:  Dee = int Gamma_e^T K Gamma_e dA / h / w     (K = wall ABD + G law)
    solid:  Dee = int C dA / w = (A_mat/w) * C           (C = 3-D material law)

This isolates Gamma_e and the wall law -- no V0, no periodicity, no KKT enters.
Cross cell, L = 1, t = 0.1, isotropic E = 70 GPa nu = 0.3, w = cell area = 1,
A_mat = 2Lt - t^2.

Run (from this folder):  python compare_Dee.py
"""
import numpy as np
import yaml as _yaml

from opensg_shell.sg_mesh import load_ring_ref
from opensg_shell.sg_homo import assemble_solid_macro, NDOF6
from opensg_shell.sg_periodicity import mesh_to_periodic_sparse_assembly_map
from opensg_solid.rm_plate_1D.msg_rm_plate import rm_plate_msg
from opensg_shell.sg_materials import material_db_from_yaml

############### User Input #################################
L, t_w = 1.0, 0.10
E, nu = 70.0e9, 0.30
nseg = 40
############################################################

G_iso = E/(2*(1+nu))
c = L/2

# ---- cross-cell 1-D shell yaml ---------------------------------------------
xs = np.linspace(0.0, L, nseg+1)
h_n = np.stack([xs, np.full(nseg+1, c)], 1)
v_n = np.stack([np.full(nseg+1, c), xs], 1)
ic = nseg//2
pts = np.vstack([h_n, v_n[[j for j in range(nseg+1) if j != ic]]])
vid, kk = {}, len(h_n)
for j in range(nseg+1):
    vid[j] = ic if j == ic else kk
    if j != ic:
        kk += 1
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
      "    E: [%.6e, %.6e, %.6e]" % (E, E, E),
      "    G: [%.6e, %.6e, %.6e]" % (G_iso, G_iso, G_iso),
      "    nu: [%.6f, %.6f, %.6f]" % (nu, nu, nu),
      "sets:", "  element:", "  - name: layup_0", "    labels:"]
o += ["    - %d" % (e+1) for e in range(len(cells))]
o.append("reference: center")
open("_dee.yaml", "w").write("\n".join(o) + "\n")

# ---- shell Dee --------------------------------------------------------------
R = load_ring_ref("_dee.yaml", "center")
d_sh = _yaml.safe_load(open("_dee.yaml"))
_mdb = material_db_from_yaml(d_sh["materials"])
G_by = list(R["G_by"])
for si, sec in enumerate(d_sh["sections"]):
    _pl = [[str(p[0]), float(p[1]), float(p[2])] for p in sec["layup"]]
    _rr = rm_plate_msg([p[1] for p in _pl], [p[2] for p in _pl],
                       [p[0] for p in _pl], _mdb, fraction=0.5)
    if _rr["G_msg"] is not None:
        G_by[si] = np.asarray(_rr["G_msg"])
rx, rcells = R["rx"], R["cells"]
m = len(rx)
h_st = float(np.mean(np.linalg.norm(rx[rcells[:, 1]] - rx[rcells[:, 0]], axis=1)))
ez = np.zeros(3); ez[R["ax"]] = 1.0
nodes_st = np.vstack([rx, rx + h_st*ez])
rc, _ = mesh_to_periodic_sparse_assembly_map(m, np.arange(m)[:, None],
                                             rx[:, R["cross"]], 3, NDOF6)
nm = np.asarray(rc, int).ravel()
dof_map = np.concatenate([nm, nm])
quads = np.array([[a, b, m+b, m+a] for a, b in rcells], int)
_, Dee_sh = assemble_solid_macro(nodes_st, quads, R["rsub"], np.asarray(R["re3"]),
                                 R["D_by"], G_by, R["cross"], R["ax"],
                                 dof_map=dof_map, shear="mitc4_g23")
Dee_sh = Dee_sh / h_st / (L*L)

# ---- solid Dee (exact integral: material C times area fraction) -------------
lam = E*nu/((1+nu)*(1-2*nu)); mu = G_iso
C0 = np.zeros((6, 6)); C0[:3, :3] = lam
C0[np.arange(3), np.arange(3)] = lam + 2*mu
C0[3, 3] = C0[4, 4] = C0[5, 5] = mu
A_mat = 2*L*t_w - t_w**2
Dee_so = C0 * A_mat / (L*L)

np.set_printoptions(precision=4, suppress=False)
print("Dee msg_shell (Pa), order [e11 e22 e33 2e23 2e13 2e12]:")
for i in range(6):
    print("  " + " ".join("%12.4e" % Dee_sh[i, j] for j in range(6)))
print("Dee msg_solid = (A_mat/w) * C  (Pa):")
for i in range(6):
    print("  " + " ".join("%12.4e" % Dee_so[i, j] for j in range(6)))

thr = 1e-6*np.max(np.abs(Dee_so))
print("\n  %-5s %15s %15s %9s" % ("term", "shell", "solid", "ratio"))
for i in range(6):
    for j in range(i, 6):
        if abs(Dee_sh[i, j]) > thr or abs(Dee_so[i, j]) > thr:
            r = Dee_sh[i, j]/Dee_so[i, j] if abs(Dee_so[i, j]) > thr else np.inf
            print("  D%d%d   %15.5e %15.5e %9.4f" % (i+1, j+1, Dee_sh[i, j],
                                                     Dee_so[i, j], r))
