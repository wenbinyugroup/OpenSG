"""Generate the 1-D shell SG yamls for the Deo cellular solid (MAMS 2023
Fig. 4), theta = +15 / -15 deg.  Periodic double cell, columns inside the
box; the figure's stubs are the box-clipped halves of the odd-row walls
(h/2, tiling-constrained).  Run:  python make_deo_cellular_yamls.py"""
import numpy as np

l, h, t = 0.10, 0.10, 0.005
E, nu = 68.9e9, 0.33
G = E/(2*(1+nu))
nseg = 10

for th_deg in (15.0, -15.0):
    th = np.radians(th_deg)
    c2, ls = l*np.cos(th)/2.0, l*np.sin(th)
    P3 = h + ls
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
        n = max(4, int(round(np.linalg.norm(p1-p0)/(l/nseg))))
        ids = [add(p0 + (p1-p0)*k/n) for k in range(n+1)]
        for k in range(n):
            cells.append([ids[k], ids[k+1]])
            tans.append(tv)

    wall([-c2, -h/2], [-c2, h/2])
    wall([c2, P3-h/2], [c2, P3])
    wall([c2, -P3], [c2, -P3+h/2])
    wall([-c2, h/2], [c2, h/2+ls])
    wall([-c2, -h/2], [c2, -h/2-ls])
    wall([-c2, h/2], [-2*c2, h/2+ls/2])
    wall([2*c2, h/2+ls/2], [c2, h/2+ls])
    wall([-c2, -h/2], [-2*c2, -h/2-ls/2])
    wall([2*c2, -h/2-ls/2], [c2, -h/2-ls])

    o = ["nodes:"]
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
    open("deo_cellular_%+03d_1Dshell.yaml" % int(th_deg),
         "w").write("\n".join(o) + "\n")
