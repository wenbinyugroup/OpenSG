"""Dump the wall's 8x8 ABDG law used by the solid-props (and Timo) routes.

    [ A (3x3)  B (3x3) ]   membrane / coupling / bending  (from load_ring_ref)
    [ B^T      D (3x3) ]
    [ G (2x2) ]            transverse shear (rm_plate_msg, the OpenSG-RM
                            Yu-LS first-order plate G -- NOT an assumed kappa)

Writes wall_ABDG_iso.out and wall_ABDG_m45.out, with the ratios of every
diagonal to its naive plate value (A11 vs E't, G vs G*t) so any emergent
"shear correction" is visible as a number, not an assumption.

Run (from this folder):  python abdg_out.py
"""
import numpy as np
import yaml as _yaml

from opensg_shell.sg_mesh import load_ring_ref
from opensg_shell.sg_materials import material_db_from_yaml
from opensg_solid.rm_plate_1D.msg_rm_plate import rm_plate_msg

############### User Input #################################
CASES = [("iso", "square_tube_1Dshell_iso.yaml"),
         ("m45", "../square_tube_1Dshell.yaml")]
############################################################

for label, yml in CASES:
    R = load_ring_ref(yml, "center")
    d_sh = _yaml.safe_load(open(yml))
    mdb = material_db_from_yaml(d_sh["materials"])
    sec = d_sh["sections"][0]
    pl = [[str(p[0]), float(p[1]), float(p[2])] for p in sec["layup"]]
    r = rm_plate_msg([p[1] for p in pl], [p[2] for p in pl],
                     [p[0] for p in pl], mdb, fraction=0.5)
    ABD = np.asarray(R["D_by"][0])
    Gm = np.asarray(r["G_msg"])
    t = sum(p[1] for p in pl)
    mat = d_sh["materials"][0]["elastic"]
    E1 = float(mat["E"][0]); nu12 = float(mat["nu"][0])
    G12, G13, G23 = [float(v) for v in mat["G"]]

    print("\n==== %s ====  (layup %s, t = %g)" % (label, pl, t))
    print("rm_plate_msg keys: %s" % sorted(r.keys()))
    print("A (3x3):");  print(ABD[0:3, 0:3])
    print("B (3x3):");  print(ABD[0:3, 3:6])
    print("D (3x3):");  print(ABD[3:6, 3:6])
    print("G_msg (2x2):"); print(Gm)
    print("naive plate values: E1'*t = %.6e   G13*t = %.6e   G23*t = %.6e"
          % (E1/(1-nu12**2)*t, G13*t, G23*t))
    print("G_msg[0,0]/(G13*t) = %.6f    G_msg[1,1]/(G23*t) = %.6f"
          % (Gm[0, 0]/(G13*t), Gm[1, 1]/(G23*t)))

    with open("wall_ABDG_%s.out" % label, "w") as f:
        f.write("# wall 8x8 ABDG law, %s square tube (t = %g, center ref)\n"
                % (label, t))
        f.write("# ABD (6x6) from load_ring_ref; G (2x2) from rm_plate_msg"
                " (OpenSG-RM Yu-LS first-order plate shear)\n")
        f.write("# ---- ABD (6x6) [eps11 eps22 2eps12 | K11 K22 K12+K21] ----\n")
        for i in range(6):
            f.write(" ".join("%16.8e" % ABD[i, j] for j in range(6)) + "\n")
        f.write("# ---- G (2x2) [2g13 2g23] ----\n")
        for i in range(2):
            f.write(" ".join("%16.8e" % Gm[i, j] for j in range(2)) + "\n")
        f.write("# G_msg[0,0]/(G13*t) = %.8f ; G_msg[1,1]/(G23*t) = %.8f\n"
                % (Gm[0, 0]/(G13*t), Gm[1, 1]/(G23*t)))
    print("wrote wall_ABDG_%s.out" % label)
