"""ANALYTICAL Dee for the isotropic square tube -- line integration over the
four wall edges -- vs numerical msg_shell and msg_solid.

Geometry: midline square, side a, wall t, iso (E, nu).  Dee is the pure macro
energy int Gamma_e^T K Gamma_e ds and involves NO fluctuation, so it is the
same with or without periodic BCs (periodicity enters only the V0 relaxation);
the numerical pipeline below nevertheless runs with the periodic map in place.

Per unit wall length (from the wall triads):
  HORIZONTAL wall (t=y2, n=+-y3):  Gamma_e rows land in
      eps11->G11, eps22->G22, 2eps12->2G12, 2g13->2G13, 2g23->2G23
  VERTICAL wall (t=y3, n=-+y2):
      eps11->G11, eps22->G33, 2eps12->2G13, 2g13->2G12, 2g23->2G23

With A11 = E' t, A12 = nu E' t, A66 = G t, Gm = G_msg (the Yu-LS wall shear),
E' = E/(1-nu^2), summing 2 walls of length a per family:

  D11 = 4a A11                D12 = 2a A12 (horiz)   D13 = 2a A12 (vert)
  D22 = 2a A11 (horiz)        D33 = 2a A11 (vert)    D23 = 0
  D44 = 4a Gm                 D55 = D66 = 2a (A66 + Gm)

Run (from this folder):  python analytical_Dee_square.py
"""
import numpy as np
import yaml as _yaml

from opensg_shell.oml_ring import load_ring_ref
from opensg_shell.solid_props import assemble_solid_macro, NDOF6
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
A11 = Ep*t
A12 = nu*Ep*t
A66 = G*t

mat = {"name": "iso", "density": 2700.0,
       "elastic": {"E": [E]*3, "G": [G]*3, "nu": [nu]*3}}
mdb = material_db_from_yaml([mat])
Gm = float(np.asarray(rm_plate_msg([t], [0.0], ["iso"], mdb,
                                   fraction=0.5)["G_msg"])[0, 0])

Dee_an = np.zeros((6, 6))
Dee_an[0, 0] = 4*a*A11
Dee_an[1, 1] = 2*a*A11
Dee_an[2, 2] = 2*a*A11
Dee_an[0, 1] = Dee_an[1, 0] = 2*a*A12
Dee_an[0, 2] = Dee_an[2, 0] = 2*a*A12
Dee_an[3, 3] = 4*a*Gm
Dee_an[4, 4] = 2*a*(A66 + Gm)
Dee_an[5, 5] = 2*a*(A66 + Gm)

# ---- numerical msg_shell Dee (periodic map in place) -----------------------
R = load_ring_ref(SHELL_YAML, "center")
d_sh = _yaml.safe_load(open(SHELL_YAML))
G_by = list(R["G_by"])
r = rm_plate_msg([t], [0.0], ["iso"],
                 material_db_from_yaml(d_sh["materials"]), fraction=0.5)
if r["G_msg"] is not None:
    G_by[0] = np.asarray(r["G_msg"])
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
                                 shear="mitc4_g23")
Dee_sh = Dee_sh / h_st

# ---- numerical msg_solid Dee: int C dA = A_mat * C0 ------------------------
lam = E*nu/((1+nu)*(1-2*nu)); mu = G
C0 = np.zeros((6, 6)); C0[:3, :3] = lam
C0[np.arange(3), np.arange(3)] = lam + 2*mu
C0[3, 3] = C0[4, 4] = C0[5, 5] = mu
A_mat = (a+t)**2 - (a-t)**2                       # mitred square annulus = 4at
Dee_so = A_mat * C0

print("iso square tube, a=%.2f t=%.3f;  A11=%.5e A12=%.5e A66=%.5e"
      " Gm=%.5e (=%.6f*Gt)" % (a, t, A11, A12, A66, Gm, Gm/(G*t)))
print("Dee independent of BCs (no fluctuation); periodic map active in the"
      " numerical pipeline.\n")
print("  %-5s %14s %14s %9s %14s %9s"
      % ("term", "analytical", "msg_shell", "an/sh", "msg_solid", "an/so"))
thr = 1e-6*np.max(np.abs(Dee_so))
for i in range(6):
    for j in range(i, 6):
        an, sh, so = Dee_an[i, j], Dee_sh[i, j], Dee_so[i, j]
        if abs(an) > thr or abs(sh) > thr or abs(so) > thr:
            rs = an/sh if abs(sh) > thr else np.inf
            ro = an/so if abs(so) > thr else np.inf
            print("  D%d%d   %14.6e %14.6e %9.6f %14.6e %9.4f"
                  % (i+1, j+1, an, sh, rs, so, ro))
