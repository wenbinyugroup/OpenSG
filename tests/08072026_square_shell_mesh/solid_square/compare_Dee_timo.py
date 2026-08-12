"""Does Dee match between msg_shell and msg_solid in the BEAM (Timo) case?

The square tube whose final Timoshenko 6x6 matched FEniCSx within 1%:
m45 ply at -45 deg, midline a = 1, t = 0.03.

  shell:  Dee (4x4 EB block) from assemble_segment_indep on the 1-D shell SG
          -- int Gamma_e^T (ABD+G) Gamma_e ds, columns [eps11 kap1 kap2 kap3]
  solid:  int Ge^T C Ge dA on the 2-D solid SG with the SAME per-element
          material frames + -45 deg ply; Ge = the Beam_solid EB masks
          (e11 = eps + y3 kap2 - y2 kap3 ; 2e12 = -y3 kap1 ; 2e13 = y2 kap1)

Run (from this folder):  python compare_Dee_timo.py
"""
import numpy as np
import jax.numpy as jnp
import yaml as _yaml

from opensg_shell.sg_mesh import load_ring_ref
from opensg_shell.sg_assembly import assemble_segment_indep
from opensg_shell.sg_materials import material_db_from_yaml
from opensg_solid.rm_plate_1D.msg_rm_plate import rm_plate_msg
from opensg_solid.sg_materials import (build_material_C, rotate_C_with_matrix,
                                       elem_rotation_from_yaml)
import jax

############### User Input #################################
SHELL_YAML = "../square_tube_1Dshell.yaml"
SOLID_YAML = "../square_tube_2Dsolid_t1only.yaml"
material_param = jnp.array([
    (142.0e9, 9.8e9, 9.8e9, 6.0e9, 6.0e9, 4.8e9, 0.30, 0.30, 0.42)])
angles = jnp.array([-45.0])
############################################################

# ---- shell beam Dee (4x4) ---------------------------------------------------
R = load_ring_ref(SHELL_YAML, "center")
d_sh = _yaml.safe_load(open(SHELL_YAML))
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
dof_map = np.concatenate([np.arange(m), np.arange(m)])
quads = np.array([[a, b, m+b, m+a] for a, b in rcells], int)
_, _, Dee_sh, _, _, _ = assemble_segment_indep(
    nodes_st, quads, R["rsub"], np.asarray(R["re3"]), R["D_by"], G_by,
    np.asarray(R["k22"]), R["cross"], R["ax"], kg_e=None,
    dof_map=dof_map, shear="mitc4_g23")
Dee_sh = np.asarray(Dee_sh)/h_st

# ---- solid beam Dee (4x4) ---------------------------------------------------
d = _yaml.safe_load(open(SOLID_YAML))
row = lambda r: [float(v) for v in
                 " ".join(str(x) for x in (r if isinstance(r, list) else [r])).split()]
nd = np.array([row(r) for r in d["nodes"]], float)
tri = np.array([[int(v) for v in row(r)] for r in d["elements"]], int) - 1
ori = np.array(d["elementOrientations"], float)
xy = nd[:, :2]                                   # (y2, y3)
ne = len(tri)

C_m = build_material_C({"materials": {}}, material_param, angles)
er = jnp.asarray(elem_rotation_from_yaml(ori), float)
C_e = np.asarray(jax.vmap(rotate_C_with_matrix, in_axes=(None, 0))(C_m[0], er))

p1, p2, p3 = xy[tri[:, 0]], xy[tri[:, 1]], xy[tri[:, 2]]
det = ((p2[:, 0]-p1[:, 0])*(p3[:, 1]-p1[:, 1])
       - (p3[:, 0]-p1[:, 0])*(p2[:, 1]-p1[:, 1]))
area = 0.5*np.abs(det)
cen = (p1 + p2 + p3)/3.0                          # (ne, 2) = (y2, y3)

# Ge masks (Beam_solid): strain order [e11 e22 e33 2e23 2e13 2e12],
# columns [eps11 kap1 kap2 kap3]
Dee_so = np.zeros((4, 4))
for e in range(ne):
    y2, y3 = cen[e]
    Ge = np.zeros((6, 4))
    Ge[0, 0] = 1.0; Ge[0, 2] = y3; Ge[0, 3] = -y2   # e11 = eps + y3 k2 - y2 k3
    Ge[4, 1] = y2                                    # 2e13 =  y2 k1
    Ge[5, 1] = -y3                                   # 2e12 = -y3 k1
    Dee_so += area[e] * (Ge.T @ C_e[e] @ Ge)

LBL = ["eps11", "kap1", "kap2", "kap3"]
np.set_printoptions(precision=5)
print("Dee msg_shell beam (4x4, [eps11 kap1 kap2 kap3]):")
for i in range(4):
    print("  " + " ".join("%13.5e" % Dee_sh[i, j] for j in range(4)))
print("Dee msg_solid beam (4x4):")
for i in range(4):
    print("  " + " ".join("%13.5e" % Dee_so[i, j] for j in range(4)))
print("\n  %-12s %14s %14s %9s" % ("term", "shell", "solid", "ratio"))
thr = 1e-6*np.max(np.abs(Dee_so))
for i in range(4):
    for j in range(i, 4):
        if abs(Dee_sh[i, j]) > thr or abs(Dee_so[i, j]) > thr:
            r = Dee_sh[i, j]/Dee_so[i, j] if abs(Dee_so[i, j]) > thr else np.inf
            print("  D_%s,%s %14.5e %14.5e %9.4f"
                  % (LBL[i], LBL[j], Dee_sh[i, j], Dee_so[i, j], r))
