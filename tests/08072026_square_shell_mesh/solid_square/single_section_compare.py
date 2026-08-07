"""ONE section: square-lattice cross cell, isotropic -- shell vs solid.

L = 1, t = 0.1, E = 70 GPa, nu = 0.3.  Both routes periodic, both normalized by
the CELL area L^2.  Analytical anchors (isotropic, slender):
    C11 = rho_bar * E,        rho_bar = (2Lt - t^2)/L^2
    C44 = 0.5 * E' * (t/L)^3, E' = E/(1-nu^2)

Run (from this folder):  python single_section_compare.py
"""
import numpy as np

import crosscell_thin_sweep as cc     # reuse its mesh/route builders

############### User Input #################################
t_w = 0.10
############################################################

L, E, nu = cc.L, cc.E_iso, cc.nu_iso
Csh = cc.shell_C(t_w)
Cso = cc.solid_C(t_w)
rho = (2*L*t_w - t_w**2)/L**2
anchor = {0: rho*E, 3: 0.5*cc.Ep*(t_w/L)**3}

print("square cross cell, L=%.2f t=%.2f (t/L=%.2f), E=%.3e nu=%.2f, cell area %.2f"
      % (L, t_w, t_w/L, E, nu, L*L))
print("\nC3D msg_shell (Pa), order [e11 e22 e33 2e23 2e13 2e12]:")
for i in range(6):
    print("  " + " ".join("%13.5e" % Csh[i, j] for j in range(6)))
print("C3D msg_solid (Pa):")
for i in range(6):
    print("  " + " ".join("%13.5e" % Cso[i, j] for j in range(6)))

thr = 1e-6*np.max(np.abs(Cso))
print("\n  %-5s %15s %15s %9s %15s" % ("term", "msg_shell", "msg_solid",
                                       "shell/sol", "analytical"))
for i in range(6):
    for j in range(i, 6):
        if abs(Csh[i, j]) > thr or abs(Cso[i, j]) > thr:
            a = "%15.5e" % anchor[i] if (i == j and i in anchor) else " " * 15
            print("  C%d%d   %15.5e %15.5e %9.4f %s"
                  % (i+1, j+1, Csh[i, j], Cso[i, j], Csh[i, j]/Cso[i, j], a))
