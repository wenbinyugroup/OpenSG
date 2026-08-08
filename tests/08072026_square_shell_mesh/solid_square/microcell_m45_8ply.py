"""Microcell validation: (1) m45 ring D23 with Lfac convergence up to the
full cell (where the mini IS the benchmark by construction); (2) a GENERAL
8-layer unbalanced laminate [0/45/-30/90/15/-45/60/0] on the cross cell,
shell+microcell vs the ply-resolved periodic solid of the full cell.

Run (from this folder):  python microcell_m45_8ply.py
"""
import numpy as np
import yaml as _yaml

from opensg_shell import build_solid_bundle
from opensg_shell.sg_mesh import load_ring_ref
from opensg_shell.sg_junction import microcell_law
from opensg_shell.sg_materials import material_db_from_yaml
from opensg_solid.rm_plate_1D.msg_rm_plate import rm_plate_msg

############### User Input #################################
a = 1.0
M45_YAML = "../square_tube_1Dshell.yaml"
M45_SOLID = "square_solid_newarch.out"      # msg_solid benchmark (0.12 norm)
PLY8 = [0.0, 45.0, -30.0, 90.0, 15.0, -45.0, 60.0, 0.0]
TP8 = 0.00375                                # 8 x 0.00375 = 0.03
E1, E2, G12, nu12, nu23 = 142.0e9, 9.8e9, 6.0e9, 0.30, 0.42
############################################################


def msg_G(sections, materials, si):
    pl = [[str(p[0]), float(p[1]), float(p[2])]
          for p in sections[si]["layup"]]
    rr = rm_plate_msg([p[1] for p in pl], [p[2] for p in pl],
                      [p[0] for p in pl], material_db_from_yaml(materials),
                      fraction=0.5)
    return np.asarray(rr["G_msg"])


def read_out_C(path):
    C, mode = [], None
    for ln in open(path):
        if ln.startswith("# ---- C"):
            mode = "C"; continue
        if ln.startswith("#"):
            continue
        p = ln.split()
        if mode == "C" and len(p) == 6:
            C.append([float(v) for v in p])
        if len(C) == 6:
            break
    return np.array(C)


TERMS = [("D11", 0, 0), ("D22", 1, 1), ("D33", 2, 2), ("D12", 0, 1),
         ("D13", 0, 2), ("D23", 1, 2)]

# ==== 1. m45 ring: microcell vs msg_solid, with Lfac convergence ===========
d = _yaml.safe_load(open(M45_YAML))
R = load_ring_ref(M45_YAML, "center")
G_by = [msg_G(d["sections"], d["materials"], si)
        for si in range(len(d["sections"]))]
stackA = [(0, False), (0, True)]             # merged antisymmetric pair
Cso = read_out_C(M45_SOLID)*0.12

print("==== m45 ring: Lfac convergence of the merged-X microcell dC ====")
print("  %-6s %6s %12s %12s %12s %12s" %
      ("Lfac", "tris", "dC22", "dC33", "dC23", "dC12"))
for Lfac in (6.0, 10.0, 16.67):
    dC, info = microcell_law(stackA, stackA, d["sections"], d["materials"],
                             R["D_by"], G_by, fill="mitre4", Lfac=Lfac)
    print("  %-6.2f %6d %12.4e %12.4e %12.4e %12.4e"
          % (Lfac, info["n_tris"], dC[1, 1], dC[2, 2], dC[1, 2], dC[0, 1]))

b0 = build_solid_bundle(M45_YAML)
b3 = build_solid_bundle(M45_YAML, junction="microcell")
print("\n==== m45 ring: shell / shell+microcell vs msg_solid ====")
print("  %-5s %13s %13s %13s %9s %9s"
      % ("term", "microcell", "solid", "off", "off/so", "cell/so"))
for nm, i, j in TERMS:
    so = Cso[i, j]
    thr = 1e-3*abs(Cso[0, 0])
    r0 = b0["D_eff"][i, j]/so if abs(so) > thr else np.inf
    r3 = b3["D_eff"][i, j]/so if abs(so) > thr else np.inf
    print("  %-5s %13.5e %13.5e %13.5e %9.4f %9.4f"
          % (nm, b3["D_eff"][i, j], so, b0["D_eff"][i, j], r0, r3))

# ==== 2. general 8-ply unbalanced laminate, cross cell =====================
G23 = E2/(2*(1 + nu23))
o = ["nodes:"]
nseg = 40
xs = np.linspace(0.0, a, nseg+1)
c = a/2
h_n = [(x, c) for x in xs]
v_n = [(c, y) for y in xs if abs(y - c) > 1e-12]
pts = h_n + v_n
for x, y in pts:
    o.append("- [%.12f %.12f 0.00000000]" % (x, y))
vid = {}
k = len(h_n)
for j, y in enumerate(xs):
    if abs(y - c) < 1e-12:
        vid[j] = nseg//2
    else:
        vid[j] = k; k += 1
cells = ([[i, i+1] for i in range(nseg)]
         + [[vid[j], vid[j+1]] for j in range(nseg)])
o.append("elements:")
for a_, b_ in cells:
    o.append("- [%d %d]" % (a_+1, b_+1))
o.append("elementOrientations:")
o += ["- [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]"]*nseg
o += ["- [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0]"]*nseg
o += ["sections:", "- type: shell", "  elementSet: layup_0", "  layup:"]
for ang in PLY8:
    o += ["  - - pl", "    - %.6e" % TP8, "    - %.1f" % ang]
o += ["materials:", "- name: pl", "  density: 1800.0", "  elastic:",
      "    E: [%.6e, %.6e, %.6e]" % (E1, E2, E2),
      "    G: [%.6e, %.6e, %.6e]" % (G12, G12, G23),
      "    nu: [%.6f, %.6f, %.6f]" % (nu12, nu12, nu23),
      "sets:", "  element:", "  - name: layup_0", "    labels:"]
o += ["    - %d" % (e+1) for e in range(len(cells))]
o.append("reference: center")
open("_ply8_cross.yaml", "w").write("\n".join(o) + "\n")

d8 = _yaml.safe_load(open("_ply8_cross.yaml"))
R8 = load_ring_ref("_ply8_cross.yaml", "center")
G8 = [msg_G(d8["sections"], d8["materials"], 0)]
s1 = [(0, False)]

b0 = build_solid_bundle("_ply8_cross.yaml", cell_area=a*a)
b3 = build_solid_bundle("_ply8_cross.yaml", cell_area=a*a,
                        junction="microcell")
# benchmark: the ply-resolved periodic solid of the FULL cell (Lm = a)
t8 = 8*TP8
_, info = microcell_law(s1, s1, d8["sections"], d8["materials"],
                        R8["D_by"], G8, fill="A", Lfac=a/t8, nply_el=2,
                        along_fac=1.0)
Dbench = info["D_solid_mini"]

print("\n==== 8-ply [0/45/-30/90/15/-45/60/0] cross cell (a=1, t=0.03) ====")
print("  benchmark: ply-resolved periodic CST of the full cell"
      " (%d tris)" % info["n_tris"])
print("  %-5s %13s %13s %13s %9s %9s"
      % ("term", "shell", "shell+cell", "solid bench", "off/so", "cell/so"))
for nm, i, j in TERMS:
    so = Dbench[i, j]
    thr = 1e-3*abs(Dbench[0, 0])
    r0 = b0["D_eff"][i, j]/so if abs(so) > thr else np.inf
    r3 = b3["D_eff"][i, j]/so if abs(so) > thr else np.inf
    print("  %-5s %13.5e %13.5e %13.5e %9.4f %9.4f"
          % (nm, b0["D_eff"][i, j], b3["D_eff"][i, j], so, r0, r3))
