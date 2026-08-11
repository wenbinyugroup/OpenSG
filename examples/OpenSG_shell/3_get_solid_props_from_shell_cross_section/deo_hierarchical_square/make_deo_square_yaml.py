"""Generate the 1-D shell SG yaml for the Deo hierarchical square (MAMS 2023
Fig. 9 / diss. Fig. 3.20): cell 2R, interior square 2r, four ligaments,
t = 5 mm, aluminum.  Run:  python make_deo_square_yaml.py"""
import numpy as np

R, r, t = 0.10, 0.05, 0.005
E, nu = 68.9e9, 0.33
G = E/(2*(1+nu))
nseg = 10

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


wall([-r,  r], [0,  r]); wall([0,  r], [r,  r])
wall([-r, -r], [0, -r]); wall([0, -r], [r, -r])
wall([-r, -r], [-r, 0]); wall([-r, 0], [-r,  r])
wall([r, -r], [r, 0]); wall([r, 0], [r,  r])
wall([r, 0], [R, 0]); wall([-r, 0], [-R, 0])
wall([0, r], [0, R]); wall([0, -r], [0, -R])

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
o += ["sections:", "- type: shell", "  elementSet: layup_0", "  layup:",
      "  - - alu", "    - %.6e" % t, "    - 0.0",
      "materials:", "- name: alu", "  density: 2700.0", "  elastic:",
      "    E: [%.6e, %.6e, %.6e]" % (E, E, E),
      "    G: [%.6e, %.6e, %.6e]" % (G, G, G),
      "    nu: [%.6f, %.6f, %.6f]" % (nu, nu, nu),
      "sets:", "  element:", "  - name: layup_0", "    labels:"]
o += ["    - %d" % (e+1) for e in range(len(cells))]
o.append("reference: center")
open("deo_square_1Dshell.yaml", "w").write("\n".join(o) + "\n")
