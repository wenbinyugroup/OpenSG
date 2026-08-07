"""Solid-vs-shell TPMS comparison: msg-solid Sample_1/Sample_2 (per-cell,
E = 69 GPa) vs the msg_shell 3-D shell SG Schwarz-P run.  Different surfaces
and densities, so raw C AND density-normalized C/(E*rho) are tabulated.
Run (after the three case runs):  python solid_vs_shell_compare.py"""
import numpy as np

E = 69.0e9
# (name, path, rho, per-cell scale: shell .out is per SG surface area
#  A = 2.3533, x A gives the per-unit-cell value; solids already per cell)
CASES = [("Sample_1 solid (Schwarz-P, t~0.128)",
          "sample_1/Sample_1_msg_solid.out", 0.300, 1.0),
         ("Sample_2 solid (Schwarz-P, t~0.0365)",
          "sample_2/Sample_2_msg_solid.out", 0.0857, 1.0),
         ("Schwarz-P shell t=0.0365 (msg_shell)",
          "../../OpenSG_shell/4_get_solid_props_from_shell_3D_SG/"
          "schwarz_p_3Dshell_C3D.out", 0.0860, 1.0)]


def read_C(path):
    """First 6x6 numeric block (the effective stiffness) of a plain or
    SwiftComp-format OpenSG .out."""
    C = []
    for ln in open(path):
        p = ln.split()
        try:
            row = [float(v) for v in p]
        except ValueError:
            continue
        if len(row) == 6:
            C.append(row)
        if len(C) == 6:
            break
    return np.array(C)


lines = ["# TPMS equivalent 3-D solid properties: msg-solid vs msg_shell"
         " (all periodic in 3 directions, alu E=69 GPa nu=0.3)",
         "# per-cell C in MPa; normalized = C/(E*rho); Zener ="
         " 2*C44/(C11-C12)",
         "# %-34s %6s %9s %9s %9s %9s %9s %7s"
         % ("case", "rho", "C11", "C12", "C44", "C11n", "C44n", "Zener")]
for name, path, rho, scl in CASES:
    C = read_C(path)*1e-6*scl
    C11, C12, C44 = C[0, 0], C[0, 1], C[3, 3]
    lines.append("  %-34s %6.4f %9.1f %9.1f %9.1f %9.4f %9.4f %7.3f"
                 % (name, rho, C11, C12, C44, C11*1e6/(E*rho),
                    C44*1e6/(E*rho), 2*C44/(C11-C12)))
open("solid_vs_shell_compare.dat", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
