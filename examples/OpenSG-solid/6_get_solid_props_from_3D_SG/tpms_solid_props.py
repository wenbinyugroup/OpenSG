"""msg-solid equivalent 3-D properties of the TPMS 3-D SGs (linear-tet solid
meshes): periodic in ALL THREE directions (the default for a 3-D SG),
aluminum E = 70 GPa, nu = 0.3.  Writes <sample>_msg_solid.out (with solve
time) and the mesh png (<sample>_mesh.png, emitted by plate_homo_2d).

Run (from this folder):  python tpms_solid_props.py [1|2]"""
import sys
import time

import numpy as np
import jax.numpy as jnp

from opensg_solid.sg_homo import plate_homo_2d

SCDIR = "../../../tests/08072026_square_shell_mesh/TPMS_SC_files"
E, nu = 70.0e9, 0.30
G = E/(2*(1+nu))
material_param = jnp.array([(E, E, E, G, G, G, nu, nu, nu)])

which = sys.argv[1:] or ["2", "1"]
for s in which:
    sc = "%s/Sample_%s.sc" % (SCDIR, s)
    t0 = time.perf_counter()
    r = plate_homo_2d(sc, material_param=material_param,
                      angles=jnp.array([0.0]), n_model=3, workdir=".",
                      plot=True)
    dt = time.perf_counter() - t0
    C = np.asarray(r["C_eff"])
    C = 0.5*(C + C.T)
    print("Sample_%s: omega=%.6g  [%.1f s]" % (s, r["omega"], dt))
    for i in range(6):
        print("  " + " ".join("%13.5e" % C[i, j] for j in range(6)))
    with open("Sample_%s_msg_solid.out" % s, "w") as f:
        f.write("# TPMS Sample_%s -- equivalent 3-D solid stiffness, OpenSG"
                " msg-solid (3-D SG, tets)\n" % s)
        f.write("# periodic in all 3 directions (default); alu E=70 GPa"
                " nu=0.3; order [G11 G22 G33 2G23 2G13 2G12]; Pa\n")
        f.write("# omega=%.8g; solve time: %.2f s\n" % (r["omega"], dt))
        f.write("# ---- C_eff (6x6, Pa) ----\n")
        for i in range(6):
            f.write(" ".join("%16.8e" % C[i, j] for j in range(6)) + "\n")
    print("Homogenization stored in Sample_%s_msg_solid.out" % s)
    print("Time taken: %.2f sec" % dt)
