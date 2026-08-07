"""Deo hierarchical square (dissertation Fig. 3.20 / Table 3.12): msg_shell
solid props vs Deo's published MSG-TW and MSG-solid columns.

Cell 2R x 2R, interior square side 2r, four ligaments from the square-wall
midpoints to the cell boundary; t = 5 mm, aluminum E = 68.9 GPa, nu = 0.33;
C normalized by the cell area (2R)^2, order [G11 G22 G33 2G23 2G13 2G12].

Run (from this folder):  python deo_hier_square.py
"""
import numpy as np

from opensg_shell import build_solid_bundle

############### User Input #################################
R, r, t = 0.10, 0.05, 0.005
E, nu = 68.9e9, 0.33
nseg = 10                       # elements per segment (half-wall / ligament)
############################################################

G = E/(2*(1+nu))
pts, cells, tans, idx = [], [], [], {}


def add_node(p):
    key = (round(p[0], 12), round(p[1], 12))
    if key not in idx:
        idx[key] = len(pts)
        pts.append(np.array(p))
    return idx[key]


def wall(p0, p1, n=nseg):
    p0, p1 = np.array(p0, float), np.array(p1, float)
    tv = (p1-p0)/np.linalg.norm(p1-p0)
    ids = [add_node(p0 + (p1-p0)*k/n) for k in range(n+1)]
    for k in range(n):
        cells.append([ids[k], ids[k+1]])
        tans.append(tv)


# square frame, each side split at its midpoint (T-junction node)
wall([-r,  r], [0,  r]); wall([0,  r], [r,  r])          # top
wall([-r, -r], [0, -r]); wall([0, -r], [r, -r])          # bottom
wall([-r, -r], [-r, 0]); wall([-r, 0], [-r,  r])         # left
wall([r, -r], [r, 0]); wall([r, 0], [r,  r])             # right
# ligaments from wall midpoints to the cell boundary
wall([r, 0], [R, 0]); wall([-r, 0], [-R, 0])
wall([0, r], [0, R]); wall([0, -r], [0, -R])

o = ["nodes:"]
for x, y in np.array(pts):
    o.append("- [%.12f %.12f 0.00000000]" % (x, y))
o.append("elements:")
for c1, c2 in cells:
    o.append("- [%d %d]" % (c1+1, c2+1))
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
open("_deo_hier.yaml", "w").write("\n".join(o) + "\n")

DEO_TW = {"C11": 5186.39, "C12": 24.83, "C13": 24.83, "C22": 47.13,
          "C23": 27.94, "C33": 47.13, "C44": 3.22, "C55": 650.00,
          "C66": 650.00}
DEO_SO = {"C11": 5099.03, "C12": 26.75, "C13": 26.75, "C22": 50.46,
          "C23": 30.59, "C33": 50.45, "C44": 3.39, "C55": 670.92,
          "C66": 670.93}
TERMS = [("C11", 0, 0), ("C12", 0, 1), ("C13", 0, 2), ("C22", 1, 1),
         ("C23", 1, 2), ("C33", 2, 2), ("C44", 3, 3), ("C55", 4, 4),
         ("C66", 5, 5)]

area = (2*R)**2
b0 = build_solid_bundle("_deo_hier.yaml", cell_area=area)
b1 = build_solid_bundle("_deo_hier.yaml", cell_area=area,
                        junction="microcell")
print("junctions detected: %d, types %s"
      % (b1["junction"]["n_junctions"], b1["junction"]["types"]))
print("\nDeo hierarchical square (R=%.2f r=%.2f t=%.3f, alu), C in MPa:"
      % (R, r, t))
print("  %-5s %10s %10s %10s %10s %9s %9s %9s"
      % ("term", "ours", "ours+junc", "Deo TW", "Deo solid",
         "ours%", "junc%", "DeoTW%"))
for nm, i, j in TERMS:
    v0 = b0["C3D"][i, j]*1e-6
    v1 = b1["C3D"][i, j]*1e-6
    so = DEO_SO[nm]
    print("  %-5s %10.2f %10.2f %10.2f %10.2f %+9.2f %+9.2f %+9.2f"
          % (nm, v0, v1, DEO_TW[nm], so, 100*(v0-so)/so, 100*(v1-so)/so,
             100*(DEO_TW[nm]-so)/so))
