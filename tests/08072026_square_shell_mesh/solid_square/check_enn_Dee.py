"""Check the e_nn-completed Gamma_e: analytical Dee vs numerical vs solid.

Analytical (iso square, side a, thickness t):
  normal block   = A_mat * C0[0:3,0:3]   (exact -- Bond rotation invariance)
  D44            = 4a Gm                  (transverse shear, computed wall G)
  D55 = D66      = 2a (A66 + Gm)
Solid: Dee = A_mat * C0.

Run (from this folder):  python check_enn_Dee.py
"""
import numpy as np
import yaml as _yaml

from opensg_shell.oml_ring import load_ring_ref
from opensg_shell.solid_props import (assemble_solid_macro, wall_solid_law,
                                      NDOF6)
from opensg_shell.emit_abd import material_db_from_yaml
from opensg_solid.rm_plate_1D.msg_rm_plate import rm_plate_msg
from opensg_shell.periodic_multiscale import mesh_to_periodic_sparse_assembly_map

############### User Input #################################
a, t = 1.0, 0.03
E, nu = 70.0e9, 0.30
SHELL_YAML = "square_tube_1Dshell_iso.yaml"
############################################################

G = E/(2*(1+nu))
Ep = E/(1-nu**2)
A66 = G*t
mat = {"name": "iso", "density": 2700.0,
       "elastic": {"E": [E]*3, "G": [G]*3, "nu": [nu]*3}}
Gm = float(np.asarray(rm_plate_msg([t], [0.0], ["iso"],
                                   material_db_from_yaml([mat]),
                                   fraction=0.5)["G_msg"])[0, 0])
lam = E*nu/((1+nu)*(1-2*nu)); mu = G
C0 = np.zeros((6, 6)); C0[:3, :3] = lam
C0[np.arange(3), np.arange(3)] = lam + 2*mu
C0[3, 3] = C0[4, 4] = C0[5, 5] = mu
A_mat = 4*a*t

Dee_an = np.zeros((6, 6))
Dee_an[:3, :3] = A_mat*C0[:3, :3]
Dee_an[3, 3] = 4*a*Gm
Dee_an[4, 4] = Dee_an[5, 5] = 2*a*(A66 + Gm)

Dee_so = A_mat*C0

# ---- numerical: updated Gamma_e with the e_nn completion --------------------
R = load_ring_ref(SHELL_YAML, "center")
d_sh = _yaml.safe_load(open(SHELL_YAML))
G_by = list(R["G_by"])
G_by[0] = np.asarray(rm_plate_msg([t], [0.0], ["iso"],
                                  material_db_from_yaml(d_sh["materials"]),
                                  fraction=0.5)["G_msg"])
Cw_by = wall_solid_law(d_sh["sections"], d_sh["materials"])
rx, rcells = R["rx"], R["cells"]
m = len(rx)
h_st = float(np.mean(np.linalg.norm(rx[rcells[:, 1]] - rx[rcells[:, 0]], axis=1)))
ez = np.zeros(3); ez[R["ax"]] = 1.0
nodes_st = np.vstack([rx, rx + h_st*ez])
rc, _ = mesh_to_periodic_sparse_assembly_map(m, np.arange(m)[:, None],
                                             rx[:, R["cross"]], 3, NDOF6)
nm = np.asarray(rc, int).ravel()
dof_map = np.concatenate([nm, nm])
quads = np.array([[q1, q2, m+q2, m+q1] for q1, q2 in rcells], int)
_, Dee_sh = assemble_solid_macro(nodes_st, quads, R["rsub"],
                                 np.asarray(R["re3"]), R["D_by"], G_by,
                                 R["cross"], R["ax"], dof_map=dof_map,
                                 shear="mitc4_g23", Cw_by=Cw_by)
Dee_sh = Dee_sh / h_st

print("iso square a=%.2f t=%.3f;  A_mat=%.4f  lam=%.5e  Gm=%.5e\n"
      % (a, t, A_mat, lam, Gm))
print("  %-5s %14s %14s %9s %14s %9s"
      % ("term", "analytical", "shell num", "an/sh", "solid", "sh/so"))
thr = 1e-6*np.max(np.abs(Dee_so))
for i in range(6):
    for j in range(i, 6):
        an, sh, so = Dee_an[i, j], Dee_sh[i, j], Dee_so[i, j]
        if abs(an) > thr or abs(sh) > thr or abs(so) > thr:
            rs = an/sh if abs(sh) > thr else np.inf
            ro = sh/so if abs(so) > thr else np.inf
            print("  D%d%d   %14.6e %14.6e %9.6f %14.6e %9.4f"
                  % (i+1, j+1, an, sh, rs, so, ro))
