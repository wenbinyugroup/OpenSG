"""Generate the 1-D shell SG yaml for the Deo composite square (MAMS 2023
Fig. 13): cell 25.4 x 25.4 mm, horizontal segments at y3 = +-5.84 mm,
vertical segment through the center, [15]_8 laminate t = 1.016 mm.
Run:  python make_deo_composite_yaml.py"""
import numpy as np

L2, y3w = 25.4e-3, 5.84e-3
tp, ang, npl = 0.127e-3, 15.0, 8
E1, E2, G12, nu12 = 141.96e9, 9.79e9, 6.136e9, 0.42
nseg = 16
h = L2/2

pts, cells, tans, idx = [], [], [], {}


def add(p):
    key = (round(p[0], 12), round(p[1], 12))
    if key not in idx:
        idx[key] = len(pts)
        pts.append(np.array(p))
    return idx[key]


def wall(p0, p1):
    p0, p1 = np.array(p0, float), np.array(p1, float)
    tv = (p1-p0)/np.linalg.norm(p1-p0)
    ids = [add(p0 + (p1-p0)*k/nseg) for k in range(nseg+1)]
    for k in range(nseg):
        cells.append([ids[k], ids[k+1]])
        tans.append(tv)


wall([-h,  y3w], [0,  y3w]); wall([0,  y3w], [h,  y3w])
wall([-h, -y3w], [0, -y3w]); wall([0, -y3w], [h, -y3w])
wall([0, -h], [0, -y3w]); wall([0, -y3w], [0, y3w]); wall([0, y3w], [0, h])

o = ["n_model: 3      # 1 = beam, 2 = plate (msg_solid), 3 = solid"
     " -- the macro model this SG homogenizes to",
     "refined: 0      # 0 = classical (beam EB 4x4, KL wall);"
     " 1 = shear-refined (beam Timoshenko 6x6, RM wall)",
     "msg: shell      # the ENGINE this SG belongs to (opensg_shell);"
     " `opensg <yaml>` dispatches on it",
     "nodes:"]
for x, y in np.array(pts):
    o.append("- [%.12f %.12f 0.00000000]" % (x, y))
o.append("elements:")
for a, b in cells:
    o.append("- [%d %d]" % (a+1, b+1))
o.append("elementOrientations:")
for tv in tans:
    e3 = np.cross([0, 0, 1.0], [tv[0], tv[1], 0])
    o.append("- [0.0, 0.0, 1.0, %.9f, %.9f, 0.0, %.9f, %.9f, 0.0]"
             % (tv[0], tv[1], e3[0], e3[1]))
o += ["sections:", "- type: shell", "  elementSet: layup_0", "  layup:"]
for _ in range(npl):
    o += ["  - - pl", "    - %.6e" % tp, "    - %.1f" % ang]
o += ["materials:", "- name: pl", "  density: 1800.0", "  elastic:",
      "    E: [%.6e, %.6e, %.6e]" % (E1, E2, E2),
      "    G: [%.6e, %.6e, %.6e]" % (G12, G12, G12),
      "    nu: [%.6f, %.6f, %.6f]" % (nu12, nu12, nu12),
      "sets:", "  element:", "  - name: layup_0", "    labels:"]
o += ["    - %d" % (e+1) for e in range(len(cells))]
o.append("reference: center")
open("deo_composite_square_1Dshell.yaml", "w").write("\n".join(o) + "\n")
