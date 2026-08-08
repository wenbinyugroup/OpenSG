"""TPMS solid 3-D SG: aperiodic (boundary-solution Dirichlet) vs periodic.

Runs plate_homo_2d(n_model=3) on the SAME .sc for both samples with both
boundary treatments.  The periodic column is digit-identical to the
SwiftComp .K (the sample runners), so the % column measures the aperiodic
(kinematic upper-bound) boundary layer of a single cell directly against
the exact periodic answer; the timing columns answer whether aperiodic
costs more.

Variables
---------
MP      : aluminum (E = 69 GPa, nu = 0.3) material row
r       : plate_homo_2d result; boundary="aperiodic"/"periodic";
          each writes Sample_<S>.out, so per-mode copies are saved as
          Sample_<S>_<mode>.out
Cc      : per-unit-cell stiffness = C_eff x omega  (SwiftComp norm)

Run (from this folder):  python compare_boundary_modes.py
"""
import shutil

import numpy as np
import jax.numpy as jnp

from opensg_solid.sg_homo import plate_homo_2d

E, nu = 69.0e9, 0.30
G = E/(2*(1+nu))
MP = jnp.array([(E, E, E, G, G, G, nu, nu, nu)])
SCROOT = "../../../tests/08072026_square_shell_mesh/TPMS_SC_files/"
IJ = [("C11", 0, 0), ("C22", 1, 1), ("C33", 2, 2), ("C12", 0, 1),
      ("C13", 0, 2), ("C23", 1, 2), ("C44", 3, 3), ("C55", 4, 4),
      ("C66", 5, 5)]

rows = ["# TPMS solid 3-D SG (tets): aperiodic (w=0 Dirichlet on boundary"
        " nodes) vs periodic (== SwiftComp .K)",
        "# per-unit-cell C in MPa; alu E=69 GPa nu=0.3"]
for S in ("1", "2"):
    res = {}
    for mode in ("aperiodic", "periodic"):
        r = plate_homo_2d(SCROOT + "Sample_%s.sc" % S, material_param=MP,
                          angles=jnp.array([0.0]), n_model=3, workdir=".",
                          plot=False, boundary=mode)
        C = np.asarray(r["C_eff"])
        res[mode] = (0.5*(C + C.T)*float(r["omega"]), r["solve_time"],
                     r["n_boundary_nodes"])
        shutil.copy("Sample_%s.out" % S, "Sample_%s_%s.out" % (S, mode))
    Ca, ta, nb = res["aperiodic"]
    Cp, tp, _ = res["periodic"]
    rows.append("")
    rows.append("## Sample_%s   aperiodic: %d boundary nodes, %.1f s"
                "   periodic: %.1f s" % (S, nb, ta, tp))
    rows.append("#  %-5s %13s %13s %9s" % ("term", "aperiodic",
                                           "periodic", "%diff"))
    for nm, i, j in IJ:
        a, p = Ca[i, j]*1e-6, Cp[i, j]*1e-6
        rows.append("   %-5s %13.3f %13.3f %+9.3f"
                    % (nm, a, p, 100*(a - p)/p))
open("boundary_compare.dat", "w").write("\n".join(rows) + "\n")
print("\n".join(rows))
