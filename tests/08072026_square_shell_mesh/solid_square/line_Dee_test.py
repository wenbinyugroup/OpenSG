"""Smallest possible Dee test: ONE straight wall segment.

  shell:  a single 1-D line element (wall along y2, normal y3, length L,
          thickness t) -> Dee = L * Gamma_e^T (ABDG) Gamma_e
          with ABDG the MSG wall law (rm_plate_msg, same as the pipeline).
  solid:  the same line of material as a solid strip, L x t
          -> Dee = (L*t) * C_3D   (Ge = I6).

No junctions, no fluctuation, no periodicity -- pure operator + law, per wall.
Order [e11 e22 e33 2e23 2e13 2e12].

Run (from this folder):  python line_Dee_test.py
"""
import numpy as np
import jax.numpy as jnp

from opensg_shell.solid_props import solid_macro_ops_batch
from opensg_shell.emit_abd import material_db_from_yaml
from opensg_solid.rm_plate_1D.msg_rm_plate import rm_plate_msg
from opensg_solid.sg_materials import build_material_C

############### User Input #################################
L_w, t_w = 0.10, 0.03
CASES = [
    ("iso", {"name": "iso", "density": 2700.0,
             "elastic": {"E": [70.0e9]*3, "G": [26.9230769e9]*3,
                         "nu": [0.30]*3}}, 0.0,
     jnp.array([(70.0e9, 70.0e9, 70.0e9, 26.9230769e9, 26.9230769e9,
                 26.9230769e9, 0.30, 0.30, 0.30)])),
    ("m45", {"name": "m45", "density": 1600.0,
             "elastic": {"E": [142.0e9, 9.8e9, 9.8e9],
                         "G": [6.0e9, 6.0e9, 4.8e9],
                         "nu": [0.30, 0.30, 0.42]}}, -45.0,
     jnp.array([(142.0e9, 9.8e9, 9.8e9, 6.0e9, 6.0e9, 4.8e9,
                 0.30, 0.30, 0.42)])),
]
############################################################

############### wall direction ############################
WALL = "vertical"            # "horizontal": tangent y2, normal +y3
                             # "vertical":   tangent y3, normal -y2
###########################################################
if WALL == "horizontal":
    Xe = np.array([[[0.0, 0.0, 0.5], [0.0, L_w, 0.5],
                    [0.1, L_w, 0.5], [0.1, 0.0, 0.5]]])   # (x1, y2, y3)
    e3e = np.array([[0.0, 0.0, 1.0]])
else:
    Xe = np.array([[[0.0, 0.5, 0.0], [0.0, 0.5, L_w],
                    [0.1, 0.5, L_w], [0.1, 0.5, 0.0]]])   # (x1, y2, y3)
    e3e = np.array([[0.0, -1.0, 0.0]])
BDe6, BGe6, _ = solid_macro_ops_batch(Xe, e3e, 0.0, 0.0, [1, 2], 0)
Ge_m, Ge_g = BDe6[0], BGe6[0]                     # (6,6), (2,6)
print("WALL = %s  (tangent %s, normal %s)"
      % (WALL, "y2" if WALL == "horizontal" else "y3",
         "+y3" if WALL == "horizontal" else "-y2"))

for label, mat, ang, mp in CASES:
    mdb = material_db_from_yaml([mat])
    r = rm_plate_msg([t_w], [ang], [mat["name"]], mdb, fraction=0.5)
    ABDG = np.asarray(r["ABDG"])
    ABD, G = ABDG[0:6, 0:6], ABDG[6:8, 6:8]

    Dee_sh = L_w * (Ge_m.T @ ABD @ Ge_m + Ge_g.T @ G @ Ge_g)

    C3 = np.asarray(build_material_C({"materials": {}}, mp,
                                     jnp.array([ang]))[0])
    Dee_so = (L_w * t_w) * C3

    print("\n==== %s ====  single wall, L=%.2f t=%.2f  (Pa*m^2)" % (label, L_w, t_w))
    print("Dee shell:")
    for i in range(6):
        print("  " + " ".join("%12.4e" % Dee_sh[i, j] for j in range(6)))
    print("Dee solid = L*t*C:")
    for i in range(6):
        print("  " + " ".join("%12.4e" % Dee_so[i, j] for j in range(6)))
    thr = 1e-6*np.max(np.abs(Dee_so))
    print("  %-5s %13s %13s %9s" % ("term", "shell", "solid", "ratio"))
    for i in range(6):
        for j in range(i, 6):
            if abs(Dee_sh[i, j]) > thr or abs(Dee_so[i, j]) > thr:
                rr = Dee_sh[i, j]/Dee_so[i, j] if abs(Dee_so[i, j]) > thr else np.inf
                print("  D%d%d   %13.4e %13.4e %9.4f"
                      % (i+1, j+1, Dee_sh[i, j], Dee_so[i, j], rr))
