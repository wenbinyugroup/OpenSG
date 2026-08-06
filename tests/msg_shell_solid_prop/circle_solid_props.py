"""Equivalent 3-D solid properties of a circular tube cross-section: ISO and M45.

The msg_shell solid-props route (opensg_shell.build_solid_bundle: the author's
MSG-shell-to-solid formulation, Gamma_e/Gamma_h per MSG_shell_solid_properties.pdf,
drilling omega_3 by element-constant Lagrange multiplier, single V0 KKT solve,
D_eff = Dee + V0^T Dhe, C3D = D_eff / (pi R^2)) on two walls:

  iso   isotropic aluminium-like        E = 70 GPa, nu = 0.3
  m45   single orthotropic ply at +45deg (AS4/3501-6-like:
        E1 = 142, E2 = E3 = 9.8, G12 = G13 = 6.0, G23 = 4.8 GPa,
        nu12 = nu13 = 0.30, nu23 = 0.42)

Prints, per case: the 6x6 stiffness C3D (order [G11 G22 G33 2G23 2G13 2G12]),
the compliance S = inv(C3D), and the 9 effective engineering constants
E1 E2 E3 G23 G13 G12 nu12 nu13 nu23.  Outputs also written to
circle_<case>_C3D.out / _S.out / _constants.out.

# ----------------------------------------------------------------------------
# ALL VARIABLES USED IN THIS SCRIPT
# ----------------------------------------------------------------------------
#   R, t, nc            mid-surface radius, wall thickness, contour elements
#   cases               {"iso": ..., "m45": ...} material/layup definitions
#   circle_<case>_shell.yaml   generated 1-D shell SG (center ref, axis = z)
#   B                   build_solid_bundle result per case
# ----------------------------------------------------------------------------
Run (from this folder):  python circle_solid_props.py
"""
import numpy as np

from opensg_shell import build_solid_bundle, elastic_constants, GBAR_ORDER

############### User Input #################################
R = 1.0                 # mid-surface radius
t = 0.03                # wall thickness (t/R = 3%)
nc = 144                # circumferential line elements
############################################################

A_cell = np.pi * R**2

cases = {
    "iso": {
        "E": [70.0e9] * 3, "G": [26.923076923e9] * 3, "nu": [0.3] * 3,
        "angle": 0.0,
    },
    "m45": {
        "E": [142.0e9, 9.8e9, 9.8e9], "G": [6.0e9, 6.0e9, 4.8e9],
        "nu": [0.30, 0.30, 0.42],
        "angle": 45.0,
    },
}
# material G order in the OpenSG yaml dialect: [G12, G13, G23]; nu: [nu12, nu13, nu23]


def write_circle_yaml(path, mat):
    th = 2.0 * np.pi * np.arange(nc) / nc
    lines = ["nodes:"]
    for a in th:
        lines.append("- [%.10f %.10f 0.00000000]" % (R * np.cos(a), R * np.sin(a)))
    lines.append("elements:")
    for i in range(nc):
        lines.append("- [%d %d]" % (i + 1, (i + 1) % nc + 1))
    lines.append("elementOrientations:")
    for i in range(nc):
        am = th[i] + np.pi / nc
        e2 = (-np.sin(am), np.cos(am))
        e3 = (-np.cos(am), -np.sin(am))          # e3 = e1 x e2, e1 = +z
        lines.append("- [0.0, 0.0, 1.0, %.12f, %.12f, 0.0, %.12f, %.12f, 0.0]"
                     % (e2[0], e2[1], e3[0], e3[1]))
    lines += ["sections:",
              "- type: shell",
              "  elementSet: layup_0",
              "  layup:",
              "  - - ply",
              "    - %.6e" % t,
              "    - %.1f" % mat["angle"],
              "materials:",
              "- name: ply",
              "  density: 1600.0",
              "  elastic:",
              "    E: [%.6e, %.6e, %.6e]" % tuple(mat["E"]),
              "    G: [%.6e, %.6e, %.6e]" % tuple(mat["G"]),
              "    nu: [%.6f, %.6f, %.6f]" % tuple(mat["nu"]),
              "sets:",
              "  element:",
              "  - name: layup_0",
              "    labels:"]
    lines += ["    - %d" % (i + 1) for i in range(nc)]
    lines.append("reference: center")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


LBL = ["C11", "C22", "C33", "C44", "C55", "C66"]
for case, mat in cases.items():
    yml = "circle_%s_shell.yaml" % case
    write_circle_yaml(yml, mat)
    B = build_solid_bundle(yml, cell_area=A_cell)
    C = np.asarray(B["C3D"])
    k, S = elastic_constants(C)

    print("=" * 78)
    print("CASE %s   (R=%g, t=%g, nc=%d, cell area pi R^2 = %.6f)"
          % (case, R, t, nc, A_cell))
    print("C3D stiffness, order %s:" % GBAR_ORDER)
    for i in range(6):
        print("  " + " ".join("%13.5e" % C[i, j] for j in range(6)))
    print("compliance S = inv(C3D):")
    for i in range(6):
        print("  " + " ".join("%13.5e" % S[i, j] for j in range(6)))
    print("effective engineering constants  (cond(C3D) = %.3e):" % k["cond"])
    print("  E1  = %13.5e   E2  = %13.5e   E3  = %13.5e" % (k["E1"], k["E2"], k["E3"]))
    print("  G23 = %13.5e   G13 = %13.5e   G12 = %13.5e" % (k["G23"], k["G13"], k["G12"]))
    print("  nu12= %13.5f   nu13= %13.5f   nu23= %13.5f" % (k["nu12"], k["nu13"], k["nu23"]))

    np.savetxt("circle_%s_C3D.out" % case, C, fmt="%16.8e",
               header="C3D, order " + GBAR_ORDER)
    np.savetxt("circle_%s_S.out" % case, S, fmt="%16.8e",
               header="compliance inv(C3D)")
    with open("circle_%s_constants.out" % case, "w") as f:
        for kk in ("E1", "E2", "E3", "G23", "G13", "G12", "nu12", "nu13", "nu23",
                   "cond"):
            f.write("%-6s %16.8e\n" % (kk, k[kk]))
print("=" * 78)
print("done.")
