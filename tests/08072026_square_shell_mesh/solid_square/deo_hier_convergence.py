"""Deo hierarchical square: (1) mesh convergence of our linear-RM elements
(nseg 10/20/40 per segment); (2) shear-rigid (Kirchhoff) limit via G x1000 --
if that reproduces Deo's CLPT column, our RM offset is physics, not error.

Run (from this folder):  python deo_hier_convergence.py
"""
import numpy as np
import yaml as _yaml

from opensg_shell import build_solid_bundle
from opensg_shell.sg_mesh import load_ring_ref
from opensg_shell.sg_homo import ring_solid

############### User Input #################################
R, r, t = 0.10, 0.05, 0.005
E, nu = 68.9e9, 0.33
DEO_TW = [5186.39, 24.83, 47.13, 27.94, 3.22, 650.00]
DEO_SO = [5099.03, 26.75, 50.46, 30.59, 3.39, 670.92]
LBL = ["C11", "C12", "C22", "C23", "C44", "C55"]
IJ = [(0, 0), (0, 1), (1, 1), (1, 2), (3, 3), (4, 4)]
############################################################

G = E/(2*(1+nu))


def make_yaml(nseg, path):
    pts, cells, tans, idx = [], [], [], {}

    def add_node(p):
        key = (round(p[0], 12), round(p[1], 12))
        if key not in idx:
            idx[key] = len(pts)
            pts.append(np.array(p))
        return idx[key]

    def wall(p0, p1):
        p0, p1 = np.array(p0, float), np.array(p1, float)
        tv = (p1-p0)/np.linalg.norm(p1-p0)
        ids = [add_node(p0 + (p1-p0)*k/nseg) for k in range(nseg+1)]
        for k in range(nseg):
            cells.append([ids[k], ids[k+1]])
            tans.append(tv)

    wall([-r,  r], [0,  r]); wall([0,  r], [r,  r])
    wall([-r, -r], [0, -r]); wall([0, -r], [r, -r])
    wall([-r, -r], [-r, 0]); wall([-r, 0], [-r,  r])
    wall([r, -r], [r, 0]); wall([r, 0], [r,  r])
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
    open(path, "w").write("\n".join(o) + "\n")


area = (2*R)**2
print("---- mesh convergence (linear RM + MITC), C in MPa ----")
print("  %-6s" % "nseg" + "".join("%10s" % l for l in LBL))
for nseg in (10, 20, 40):
    make_yaml(nseg, "_deo_c.yaml")
    b = build_solid_bundle("_deo_c.yaml", cell_area=area)
    C = np.asarray(b["C3D"])*1e-6
    print("  %-6d" % nseg + "".join("%10.2f" % C[i, j] for i, j in IJ))
print("  %-6s" % "DeoTW" + "".join("%10.2f" % v for v in DEO_TW))
print("  %-6s" % "Deoso" + "".join("%10.2f" % v for v in DEO_SO))

# ---- shear-rigid (Kirchhoff) limit at nseg=20 -----------------------------
make_yaml(20, "_deo_c.yaml")
Rr = load_ring_ref("_deo_c.yaml", "center")
from opensg_solid.rm_plate_1D.msg_rm_plate import rm_plate_msg
from opensg_shell.sg_materials import material_db_from_yaml
d_c = _yaml.safe_load(open("_deo_c.yaml"))
Gmsg = np.asarray(rm_plate_msg([t], [0.0], ["alu"],
                               material_db_from_yaml(d_c["materials"]),
                               fraction=0.5)["G_msg"])
G_by = [Gmsg*1.0e3]                                  # shear-rigid walls
D = ring_solid(Rr["rx"], Rr["cells"], Rr["rsub"], Rr["re3"], Rr["D_by"],
               G_by, Rr["k22"], Rr["ax"], Rr["cross"], periodic=True)
C = np.asarray(D)/area*1e-6
print("\n---- shear-rigid limit (G x1000 ~ Kirchhoff), nseg=20 ----")
print("  %-6s" % "KLlim" + "".join("%10.2f" % C[i, j] for i, j in IJ))
print("  %-6s" % "DeoTW" + "".join("%10.2f" % v for v in DEO_TW))
