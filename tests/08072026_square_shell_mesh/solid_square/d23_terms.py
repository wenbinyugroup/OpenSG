"""Which Gamma_e term is responsible for D23 = 0?

D23 = int Gamma_e[:, G22]^T K Gamma_e[:, G33] dA.  Print the two columns row by
row for each wall family of the square cross cell, and the row-by-row product
with the wall law -- whatever row carries BOTH columns on the SAME wall is the
D23 channel.

Run (from this folder):  python d23_terms.py
"""
import numpy as np

from opensg_shell.sg_homo import solid_macro_ops_batch

ROWS = ["eps11", "eps22", "2eps12", "K11", "K22", "K12+K21", "2g13", "2g23"]

# one horizontal-wall element (tangent y2, normal y3) and one vertical
# (tangent y3, normal -y2); strip quads in (ax, y2, y3) with ax = z
t_w = 0.10
Xh = np.array([[[0.0, 0.0, 0.5], [0.0, 0.1, 0.5],
                [0.1, 0.1, 0.5], [0.1, 0.0, 0.5]]])      # z, y2 vary; y3 = c
e3h = np.array([[0.0, 0.0, 1.0]])
Xv = np.array([[[0.0, 0.5, 0.0], [0.0, 0.5, 0.1],
                [0.1, 0.5, 0.1], [0.1, 0.5, 0.0]]])      # z, y3 vary; y2 = c
e3v = np.array([[0.0, -1.0, 0.0]])

for name, Xe, e3e in (("HORIZONTAL wall (t = y2, n = y3)", Xh, e3h),
                      ("VERTICAL   wall (t = y3, n = -y2)", Xv, e3v)):
    BDe6, BGe6, _ = solid_macro_ops_batch(Xe, e3e, 0.0, 0.0, [1, 2], 0)
    Ge = np.vstack([BDe6[0], BGe6[0]])                   # (8, 6)
    print("\n%s" % name)
    print("  %-8s %10s %10s   %s" % ("row", "col G22", "col G33", "product"))
    for r in range(8):
        a, b = Ge[r, 1], Ge[r, 2]
        print("  %-8s %10.4f %10.4f   %10.4f" % (ROWS[r], a, b, a*b))
    print("  eps_nn = C_3i G_ij C_3j (through-thickness normal strain):"
          " coeff(G22) = %.4f, coeff(G33) = %.4f  -- NOT in the 8-strain set"
          % (e3e[0, 1]**2, e3e[0, 2]**2))
