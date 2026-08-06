"""debug_g23.py -- why does the shell C3D keep stiffness in the 2G23 row?

Continuum fact: under pure Gbar = [0,0,0, 2G23=1, 0,0] the fluctuation
    w2 = -G23*y3,  w3 = -G23*y2,  omega = 0     (G23 = 1/2)
makes the TOTAL displacement identically zero -- an undeformed state.  A consistent
operator pair must give (Gamma_e[:,4] + Gamma_h * w_relax) = 0 in every row, hence
zero stiffness in that column (which the 2-D solid reference confirms).

This script evaluates that residual row by row on the circle ring for TWO readings
of the two ambiguous 2G23 buckets of manuscript page 7:

  reading A (as confirmed):  row 2g13: (X31*C32 + X11*C33)
                             row 2g23: (X32*C32 + X12*C33)
  reading B (index-pattern,
  = the page-6 expansions):  row 2g13: (X31*C32 + X21*C33)
                             row 2g23: (X32*C32 + X22*C33)

Page-6 check (the manuscript's own expansions): in 2g23 the coefficient of
u3,2 is C33*X22 (the C33(u3'2 X22) term) and of u2,3 is C32*X32, so the bucket on
(2G23) is (X32*C32 + X22*C33) -> reading B.  Same for 2g13: C33*X21 -> reading B.

Then it re-runs the iso circle C3D with reading B to show the 2G23 row.
Run (from this folder):  python debug_g23.py
"""
import numpy as np

from opensg_shell.solid_props import (solid_macro_ops_batch, ring_solid,
                                      build_solid_bundle, GBAR_ORDER)
from opensg_shell.segment_indep import quad_ops_indep_batch, _surf_frame_batch
import opensg_shell.solid_props as SP
from opensg_shell.oml_ring import load_ring_ref

R, t, nc = 1.0, 0.03, 144
A_cell = np.pi * R**2

Rd = load_ring_ref("circle_iso_shell.yaml", "center")
rx, cells, rsub, re3 = Rd["rx"], Rd["cells"], Rd["rsub"], Rd["re3"]
ax, cross = Rd["ax"], Rd["cross"]
m = len(rx)
h = float(np.mean(np.linalg.norm(rx[cells[:, 1]] - rx[cells[:, 0]], axis=1)))
ez = np.zeros(3); ez[ax] = 1.0
nodes = np.vstack([rx, rx + h * ez])
quads = np.array([[a, b, m + b, m + a] for a, b in cells], dtype=int)
Xe = nodes[quads]; e3e = np.asarray(re3)

# exact relaxation field for unit 2G23 (G23 = 1/2), on the strip nodes, 6 dof/node.
# nodal dof order is [w_ax, w_cross0, w_cross1, om_ax, om_cross0, om_cross1]:
# slot 1 = global cross[0] component, slot 2 = global cross[1] component.
G23 = 0.5
w_nodal = np.zeros((len(nodes), 6))
w_nodal[:, 1] = -G23 * nodes[:, cross[1]]             # w2 = -G23*y3
w_nodal[:, 2] = -G23 * nodes[:, cross[0]]             # w3 = -G23*y2
we = w_nodal[quads].reshape(len(quads), 24)


def macro_ops_reading(reading, Xq, e3q, xi, eta):
    """Gamma_e columns for bucket reading 'A' (confirmed) or 'B' (page-6)."""
    N, D1, D2, dA, c = _surf_frame_batch(Xq, e3q, xi, eta, cross, ax)
    X11, X21, X31 = c["x11"], c["x21"], c["x31"]
    X12, X22, X32 = c["x12"], c["x22"], c["x32"]
    C31, C32, C33 = c["y1"], c["y2"], c["y3"]
    BDe6, BGe6, _ = solid_macro_ops_batch(Xq, e3q, xi, eta, cross, ax)
    if reading == "B":
        BGe6 = BGe6.copy()
        BGe6[:, 0, 3] = X31 * C32 + X21 * C33          # 2g13 row, 2G23 column
        BGe6[:, 1, 3] = X32 * C32 + X22 * C33          # 2g23 row, 2G23 column
    return BDe6, BGe6


names = ["e11", "e22", "2e12", "K11", "K22", "K12+21", "2g13", "2g23"]
print("residual strain rows under the exact 2G23 relaxation field")
print("(macro column 4 + Gamma_h * w_relax at the element Gauss point; max |.| over ring)")
print("%-8s %14s %14s" % ("row", "reading A", "reading B"))
gp = 1.0 / np.sqrt(3.0)
resA = np.zeros(8); resB = np.zeros(8)
for (xi, eta) in [(-gp, -gp), (gp, -gp), (gp, gp), (-gp, gp)]:
    BDe, BDh, _, BGe, BGh, _, _, _, _, dA = quad_ops_indep_batch(
        Xe, e3e, xi, eta, cross, ax)
    fluct6 = np.einsum('erb,eb->er', BDh, we)          # 6 membrane+curv rows
    fluct2 = np.einsum('erb,eb->er', BGh, we)          # 2 shear rows
    for tag, res in (("A", resA), ("B", resB)):
        BDe6, BGe6 = macro_ops_reading(tag, Xe, e3e, xi, eta)
        r6 = BDe6[:, :, 3] + fluct6                    # unit 2G23 drive
        r2 = BGe6[:, :, 3] + fluct2
        res[:6] = np.maximum(res[:6], np.max(np.abs(r6), axis=0))
        res[6:] = np.maximum(res[6:], np.max(np.abs(r2), axis=0))
for i, nm in enumerate(names):
    print("%-8s %14.6e %14.6e" % (nm, resA[i], resB[i]))

# ---- re-run the iso circle with reading B patched in ------------------------
_orig = SP.solid_macro_ops_batch


def _patched(Xq, e3q, xi, eta, cr, axx):
    BDe6, BGe6, dA = _orig(Xq, e3q, xi, eta, cr, axx)
    N, D1, D2, dA2, c = _surf_frame_batch(Xq, e3q, xi, eta, cr, axx)
    BGe6 = BGe6.copy()
    BGe6[:, 0, 3] = c["x31"] * c["y2"] + c["x21"] * c["y3"]
    BGe6[:, 1, 3] = c["x32"] * c["y2"] + c["x22"] * c["y3"]
    return BDe6, BGe6, dA


from opensg_shell import elastic_constants

for case in ("iso", "m45"):
    for tag in ("A", "B"):
        SP.solid_macro_ops_batch = _orig if tag == "A" else _patched
        B = build_solid_bundle("circle_%s_shell.yaml" % case, cell_area=A_cell)
        C = np.asarray(B["C3D"])
        k, S = elastic_constants(C)
        print("\n%s circle, reading %s:  C3D diagonal:" % (case, tag))
        print("  " + "  ".join("%12.4e" % C[i, i] for i in range(6)))
        print("  9 constants (cond(C3D) = %.3e):" % k["cond"])
        print("    E1  = %13.5e   E2  = %13.5e   E3  = %13.5e"
              % (k["E1"], k["E2"], k["E3"]))
        print("    G23 = %13.5e   G13 = %13.5e   G12 = %13.5e"
              % (k["G23"], k["G13"], k["G12"]))
        print("    nu12= %13.5f   nu13= %13.5f   nu23= %13.5f"
              % (k["nu12"], k["nu13"], k["nu23"]))
SP.solid_macro_ops_batch = _orig
