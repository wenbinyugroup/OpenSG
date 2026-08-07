"""TASK D follow-up: break ALL reflection symmetries of the square ring --
45-deg ply on the TOP wall only, 0-deg ply on the other three (same m45
orthotropic material).  If shell D23 (= C23) is nonzero here, the exact zeros
of the symmetric m45 / mixed rings were symmetry cancellations, not a
structural decoupling of the 0/90 wall model.

Run (from this folder):  python wf_d23_asym.py
"""
import time

import numpy as np

from opensg_shell import build_solid_bundle


def write_ring_yaml(fname, walls, sections_txt, mats_txt, set_lists):
    pts, cells, tans, idx = [], [], [], {}

    def add_node(p):
        key = (round(p[0], 12), round(p[1], 12))
        if key not in idx:
            idx[key] = len(pts)
            pts.append(np.array(p, float))
        return idx[key]

    eids_per_wall = []
    for p0, p1, n in walls:
        p0, p1 = np.array(p0, float), np.array(p1, float)
        tv = (p1 - p0)/np.linalg.norm(p1 - p0)
        ids = [add_node(p0 + (p1 - p0)*k/n) for k in range(n + 1)]
        wall_eids = []
        for k in range(n):
            wall_eids.append(len(cells))
            cells.append([ids[k], ids[k + 1]])
            tans.append(tv)
        eids_per_wall.append(wall_eids)

    o = ["nodes:"]
    for x, y in np.array(pts):
        o.append("- [%.12f %.12f 0.00000000]" % (x, y))
    o.append("elements:")
    for c1, c2 in cells:
        o.append("- [%d %d]" % (c1 + 1, c2 + 1))
    o.append("elementOrientations:")
    for tv in tans:
        e3 = np.cross([0, 0, 1.0], [tv[0], tv[1], 0])
        o.append("- [0.0, 0.0, 1.0, %.9f, %.9f, 0.0, %.9f, %.9f, 0.0]"
                 % (tv[0], tv[1], e3[0], e3[1]))
    o += sections_txt + mats_txt
    o += ["sets:", "  element:"]
    for sname, wlist in set_lists:
        o.append("  - name: %s" % sname)
        o.append("    labels:")
        for w in wlist:
            o += ["    - %d" % (e + 1) for e in eids_per_wall[w]]
    o.append("reference: center")
    open(fname, "w").write("\n".join(o) + "\n")


m45_mat = ["materials:", "- name: m45", "  density: 1600.0", "  elastic:",
           "    E: [1.420000e+11, 9.800000e+09, 9.800000e+09]",
           "    G: [6.000000e+09, 6.000000e+09, 4.800000e+09]",
           "    nu: [0.300000, 0.300000, 0.420000]"]


def sec_txt(entries):
    o = ["sections:"]
    for sname, mname, t, ang in entries:
        o += ["- type: shell", "  elementSet: %s" % sname, "  layup:",
              "  - - %s" % mname, "    - %.6e" % t, "    - %.1f" % ang]
    return o


a_sq, t_sq, nsg = 1.0, 0.03, 12
t0 = time.perf_counter()
write_ring_yaml("wf_asym45.yaml",
                [((-a_sq/2, -a_sq/2), (a_sq/2, -a_sq/2), nsg),   # bottom 0
                 ((a_sq/2, -a_sq/2), (a_sq/2, a_sq/2), nsg),     # right  0
                 ((a_sq/2, a_sq/2), (-a_sq/2, a_sq/2), nsg),     # TOP    45
                 ((-a_sq/2, a_sq/2), (-a_sq/2, -a_sq/2), nsg)],  # left   0
                sec_txt([("layup_top", "m45", t_sq, 45.0),
                         ("layup_rest", "m45", t_sq, 0.0)]), m45_mat,
                [("layup_top", [2]), ("layup_rest", [0, 1, 3])])
B = build_solid_bundle("wf_asym45.yaml", cell_area=4*a_sq*t_sq)
C = np.asarray(B["C3D"]); C = 0.5*(C + C.T)
print("ASYMMETRIC square ring: 45-ply TOP wall only, 0-ply elsewhere  [%.1f s]"
      % (time.perf_counter() - t0))
print("  C22 = %.5e  C33 = %.5e" % (C[1, 1], C[2, 2]))
print("  C23 = %.5e   -> |C23|/sqrt(C22*C33) = %.2e"
      % (C[1, 2], abs(C[1, 2])/np.sqrt(C[1, 1]*C[2, 2])))
print("  full row 2 (couplings of e22): " +
      " ".join("%10.3e" % v for v in C[1]))
print("  full row 3 (couplings of e33): " +
      " ".join("%10.3e" % v for v in C[2]))
