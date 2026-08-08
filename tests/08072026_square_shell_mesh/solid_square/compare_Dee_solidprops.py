"""Per-term Dee comparison for the SOLID-PROPS route, msg_shell vs msg_solid,
on the same square tube sections as the Timo Dee check.

  shell:  Dee6 = int Gamma_e^T (ABD+G) Gamma_e ds   (assemble_solid_macro / h)
  solid:  Dee  = int Ge^T C Ge dA with Ge = I6 (n_model = 3)  = sum area_e C_e

Cases: m45 (-45 ply, per-element frames) and ISO.  Both in Pa*m^2, same
measure, so terms compare directly.  Order [e11 e22 e33 2e23 2e13 2e12].

Run (from this folder):  python compare_Dee_solidprops.py
"""
import numpy as np
import jax
import jax.numpy as jnp
import yaml as _yaml

from opensg_shell.sg_mesh import load_ring_ref
from opensg_shell.sg_homo import assemble_solid_macro, NDOF6
from opensg_shell.sg_materials import material_db_from_yaml
from opensg_solid.rm_plate_1D.msg_rm_plate import rm_plate_msg
from opensg_solid.sg_materials import (build_material_C, rotate_C_with_matrix,
                                       elem_rotation_from_yaml)
from opensg_shell.sg_periodicity import mesh_to_periodic_sparse_assembly_map

############### User Input #################################
SOLID_YAML = "../square_tube_2Dsolid_t1only.yaml"
CASES = [
    ("m45", "../square_tube_1Dshell.yaml",
     jnp.array([(142.0e9, 9.8e9, 9.8e9, 6.0e9, 6.0e9, 4.8e9,
                 0.30, 0.30, 0.42)]), jnp.array([-45.0]), True),
    ("ISO", "square_tube_1Dshell_iso.yaml",
     jnp.array([(70.0e9, 70.0e9, 70.0e9, 26.9230769e9, 26.9230769e9,
                 26.9230769e9, 0.30, 0.30, 0.30)]), jnp.array([0.0]), False),
]
############################################################

d = _yaml.safe_load(open(SOLID_YAML))
row = lambda r: [float(v) for v in
                 " ".join(str(x) for x in (r if isinstance(r, list) else [r])).split()]
nd = np.array([row(r) for r in d["nodes"]], float)
tri = np.array([[int(v) for v in row(r)] for r in d["elements"]], int) - 1
ori = np.array(d["elementOrientations"], float)
xy = nd[:, :2]
p1, p2, p3 = xy[tri[:, 0]], xy[tri[:, 1]], xy[tri[:, 2]]
det = ((p2[:, 0]-p1[:, 0])*(p3[:, 1]-p1[:, 1])
       - (p3[:, 0]-p1[:, 0])*(p2[:, 1]-p1[:, 1]))
area = 0.5*np.abs(det)
er = jnp.asarray(elem_rotation_from_yaml(ori), float)

TERMS = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2),
         (3, 3), (4, 4), (5, 5), (0, 5), (1, 5)]

for label, shell_yaml, mp, ang, use_frames in CASES:
    # ---- solid Dee: Ge = I6  ->  sum area_e * C_e ---------------------------
    C_m = build_material_C({"materials": {}}, mp, ang)
    if use_frames:
        C_e = np.asarray(jax.vmap(rotate_C_with_matrix,
                                  in_axes=(None, 0))(C_m[0], er))
    else:
        C_e = np.broadcast_to(np.asarray(C_m[0]), (len(tri), 6, 6))
    Dee_so = np.einsum('e,eij->ij', area, C_e)

    # ---- shell Dee: assemble_solid_macro / h --------------------------------
    R = load_ring_ref(shell_yaml, "center")
    d_sh = _yaml.safe_load(open(shell_yaml))
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
    h_st = float(np.mean(np.linalg.norm(rx[rcells[:, 1]] - rx[rcells[:, 0]],
                                        axis=1)))
    ez = np.zeros(3); ez[R["ax"]] = 1.0
    nodes_st = np.vstack([rx, rx + h_st*ez])
    rc, _ = mesh_to_periodic_sparse_assembly_map(m, np.arange(m)[:, None],
                                                 rx[:, R["cross"]], 3, NDOF6)
    nm = np.asarray(rc, int).ravel()
    dof_map = np.concatenate([nm, nm])
    quads = np.array([[a, b, m+b, m+a] for a, b in rcells], int)
    _, Dee_sh = assemble_solid_macro(nodes_st, quads, R["rsub"],
                                     np.asarray(R["re3"]), R["D_by"], G_by,
                                     R["cross"], R["ax"], dof_map=dof_map,
                                     shear="mitc4_g23")
    Dee_sh = Dee_sh / h_st

    print("\n==== %s ====  (Pa*m^2, order [e11 e22 e33 2e23 2e13 2e12])" % label)
    print("  %-5s %14s %14s %9s" % ("term", "Dee shell", "Dee solid", "ratio"))
    thr = 1e-6*np.max(np.abs(Dee_so))
    for (i, j) in TERMS:
        a, b = Dee_sh[i, j], Dee_so[i, j]
        if abs(a) > thr or abs(b) > thr:
            r = a/b if abs(b) > thr else np.inf
            print("  D%d%d   %14.5e %14.5e %9.4f" % (i+1, j+1, a, b, r))
