"""Schwarz-P TPMS shell: aperiodic (boundary-solution Dirichlet) vs periodic.

Runs shell_sg3d on the CURRENT schwarz_p_3Dshell.yaml with both boundary
treatments and appends a per-unit-cell comparison block to
boundary_compare.dat.  On a genuine unit cell the aperiodic (kinematic
Dirichlet, upper-bound) result must agree closely with the periodic one --
that agreement verifies the boundary-solution path; the timing columns
answer whether aperiodic costs more.

Variables
---------
T       : layup thickness read from the yaml (labels the block)
ra, rp  : shell_sg3d results, boundary="aperiodic" / "periodic";
          each writes/overwrites <yaml>_C3D.out, so per-mode copies are
          saved as schwarz_t<T>_<mode>.out
Ca, Cp  : per-unit-cell stiffness  D_eff / V_cell  (MPa in the table)
"""
import shutil

import numpy as np
import yaml as _yaml

from opensg_shell.shell_sg3d import shell_sg3d

YAML = "schwarz_p_3Dshell.yaml"
T = float(_yaml.safe_load(open(YAML))["sections"][0]["layup"][0][1])

res, out = {}, {}
for mode in ("aperiodic", "periodic"):
    r = shell_sg3d(YAML, boundary=mode)
    V_cell = 1.0                      # unit cube cell
    res[mode] = (r["D_eff"]/V_cell, r["solve_time"], r["ndof"],
                 r["n_boundary_nodes"])
    out[mode] = "schwarz_t%.6f_%s.out" % (T, mode)
    shutil.copy("schwarz_p_3Dshell_C3D.out", out[mode])

Ca, ta, na, nb = res["aperiodic"]
Cp, tp, np_, _ = res["periodic"]
IJ = [("C11", 0, 0), ("C22", 1, 1), ("C33", 2, 2), ("C12", 0, 1),
      ("C13", 0, 2), ("C23", 1, 2), ("C44", 3, 3), ("C55", 4, 4),
      ("C66", 5, 5)]
rows = ["## t = %.6f   aperiodic: %d free-face nodes clamped, ndof %d,"
        " %.1f s   periodic: ndof %d, %.1f s" % (T, nb, na, ta, np_, tp),
        "#  %-5s %13s %13s %9s" % ("term", "aperiodic", "periodic", "%diff")]
for nm, i, j in IJ:
    a, p = Ca[i, j]*1e-6, Cp[i, j]*1e-6
    rows.append("   %-5s %13.2f %13.2f %+9.3f" % (nm, a, p, 100*(a - p)/p))
with open("boundary_compare.dat", "a") as f:
    f.write("\n".join(rows) + "\n\n")
print("\n".join(rows))
