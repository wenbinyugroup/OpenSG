"""Thick I-beam sweep: msg_shell vs msg_solid equivalent 3-D solid D_eff
(n_model = 3 physics), both periodic, t/h from 0.03 to 0.20.

Geometry per case: flanges width b at y3 = +-h/2, web at y2 = 0, thickness t
(center ref).  Solid route: standard 2-D solid yaml -> plate_homo_2d
(opensg_solid, n_model=3).  Shell route: 1-D shell yaml -> build_solid_bundle.
Analytical thin-wall row for reference (E' membrane, frame D44 harmonic mean,
shear-flow D55/D66) -- expected to degrade as t/h grows.

Run (from this folder):  python ibeam_thick_sweep.py
"""
import time

import numpy as np
import yaml as _yaml
import jax.numpy as jnp

from opensg_shell import build_solid_bundle
from opensg_solid.sg_homo import plate_homo_2d
from opensg_solid.sg_materials import elem_rotation_from_yaml

############### User Input #################################
b, h = 0.5, 1.0
E, nu = 70.0e9, 0.30
T_SWEEP = [0.03, 0.05, 0.10, 0.15, 0.20]
nseg = 16                          # shell elems per half-flange / half-web
ntw, ntf = 8, 16                   # solid elems through web t / through 2t
OUT = "ibeam_thick_sweep.dat"
############################################################

G = E/(2*(1+nu))
Ep = E/(1-nu**2)
TERMS = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2), (3, 3), (4, 4),
         (5, 5)]


def frame_D44(EIh, EIv, span_h, span_v):
    kv, kh = 6.0*EIv/span_v, 6.0*EIh/span_h
    return 2.0*kv*kh/(kv + kh)


def analytical(t):
    Lf, Lw = 2*b, h
    D = np.zeros((6, 6))
    D[0, 0] = Ep*t*(Lf+Lw)
    D[1, 1] = Ep*t*Lf
    D[2, 2] = Ep*t*Lw
    D[0, 1] = D[1, 0] = nu*Ep*t*Lf
    D[0, 2] = D[2, 0] = nu*Ep*t*Lw
    D[3, 3] = frame_D44(Ep*(2*t)**3/12, Ep*t**3/12, b, h + t)
    D[4, 4] = G*t*Lw
    D[5, 5] = G*t*Lf
    return D


def shell_Deff(t):
    pts, cells, tans, idx = [], [], [], {}

    def add_node(p):
        key = (round(p[0], 12), round(p[1], 12))
        if key not in idx:
            idx[key] = len(pts)
            pts.append(np.array(p))
        return idx[key]

    def add_wall(p0, p1, n):
        p0, p1 = np.array(p0, float), np.array(p1, float)
        tv = (p1-p0)/np.linalg.norm(p1-p0)
        ids = [add_node(p0 + (p1-p0)*k/n) for k in range(n+1)]
        for k in range(n):
            cells.append([ids[k], ids[k+1]])
            tans.append(tv)

    add_wall([-b/2,  h/2], [0.0,  h/2], nseg)
    add_wall([0.0,  h/2], [b/2,  h/2], nseg)
    add_wall([-b/2, -h/2], [0.0, -h/2], nseg)
    add_wall([0.0, -h/2], [b/2, -h/2], nseg)
    add_wall([0.0, -h/2], [0.0,  h/2], 2*nseg)

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
          "  - - iso", "    - %.6e" % t, "    - 0.0",
          "materials:", "- name: iso", "  density: 2700.0", "  elastic:",
          "    E: [%.6e, %.6e, %.6e]" % (E, E, E),
          "    G: [%.6e, %.6e, %.6e]" % (G, G, G),
          "    nu: [%.6f, %.6f, %.6f]" % (nu, nu, nu),
          "sets:", "  element:", "  - name: layup_0", "    labels:"]
    o += ["    - %d" % (e+1) for e in range(len(cells))]
    o.append("reference: center")
    open("_ibthk.yaml", "w").write("\n".join(o) + "\n")
    bun = build_solid_bundle("_ibthk.yaml", cell_area=b*h)
    D = np.asarray(bun["D_eff"])
    return 0.5*(D + D.T)


def solid_Deff(t):
    sz = max(t/6.0, 0.0075)
    nfa = max(2, int(round((b/2 - t/2)/sz)))
    nwa = max(2, int(round((h - t)/sz)))
    xa = np.unique(np.round(np.concatenate([
        np.linspace(-b/2, -t/2, nfa+1), np.linspace(-t/2, t/2, ntw+1),
        np.linspace(t/2, b/2, nfa+1)]), 12))
    ya = np.unique(np.round(np.concatenate([
        np.linspace(-h/2-t/2, -h/2+t/2, ntf//2+1),
        np.linspace(-h/2+t/2, h/2-t/2, nwa+1),
        np.linspace(h/2-t/2, h/2+t/2, ntf//2+1)]), 12))
    nx, ny = len(xa), len(ya)
    gid = -np.ones((nx, ny), int)
    qc = []
    for i in range(nx-1):
        for j in range(ny-1):
            cx, cy = 0.5*(xa[i]+xa[i+1]), 0.5*(ya[j]+ya[j+1])
            if abs(abs(cy) - h/2) < t/2 or (abs(cx) < t/2
                                            and abs(cy) < h/2 - t/2):
                qc.append((i, j))
    nid = 0
    for (i, j) in qc:
        for di, dj in ((0, 0), (1, 0), (0, 1), (1, 1)):
            if gid[i+di, j+dj] < 0:
                gid[i+di, j+dj] = nid; nid += 1
    nd = np.zeros((nid, 2))
    for i in range(nx):
        for j in range(ny):
            if gid[i, j] >= 0:
                nd[gid[i, j]] = (xa[i], ya[j])
    tri = []
    for (i, j) in qc:
        n00, n10, n01, n11 = gid[i, j], gid[i+1, j], gid[i, j+1], gid[i+1, j+1]
        if (i + j) % 2 == 0:
            tri += [[n00, n10, n11], [n00, n11, n01]]
        else:
            tri += [[n00, n10, n01], [n10, n11, n01]]
    tri = np.array(tri, int)

    o = ["nodes:"]
    for x, y in nd:
        o.append("- [%.12f %.12f 0.00000000]" % (x, y))
    o.append("elements:")
    for n1, n2, n3 in tri:
        o.append("- [%d %d %d]" % (n1+1, n2+1, n3+1))
    o.append("elementOrientations:")
    o += ["- [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]"]*len(tri)
    o += ["materials:", "- name: iso", "  density: 2700.0", "  elastic:",
          "    E: [%.6e, %.6e, %.6e]" % (E, E, E),
          "    G: [%.6e, %.6e, %.6e]" % (G, G, G),
          "    nu: [%.6f, %.6f, %.6f]" % (nu, nu, nu),
          "sets:", "  element:", "  - name: iso_m0", "    labels:"]
    o += ["    - %d" % (e+1) for e in range(len(tri))]
    open("_ibthk_2Dsolid.yaml", "w").write("\n".join(o) + "\n")

    d = _yaml.safe_load(open("_ibthk_2Dsolid.yaml"))
    row = lambda r: [float(v) for v in
                     " ".join(str(x) for x in
                              (r if isinstance(r, list) else [r])).split()]
    nd_y = np.array([row(r) for r in d["nodes"]], float)
    tri_y = np.array([[int(v) for v in row(r)] for r in d["elements"]], int) - 1
    ori = np.array(d["elementOrientations"], float)
    mat_id = np.ones(len(tri_y), int)
    for k, s in enumerate(d["sets"]["element"]):
        mat_id[np.array(s["labels"], int) - 1] = k + 1
    sc = {"dim": 2, "nodes": nd_y[:, :2], "cells": [list(c) for c in tri_y],
          "mat_id": mat_id,
          "materials": {0: {"name": d["materials"][0]["name"]}}, "scale": 1.0}
    r = plate_homo_2d(sc, material_param=jnp.array([(E, E, E, G, G, G,
                                                     nu, nu, nu)]),
                      angles=jnp.array([0.0]), n_model=3,
                      elem_rotation=elem_rotation_from_yaml(ori), plot=False)
    C = np.asarray(r["C_eff"]); C = 0.5*(C + C.T)
    return C*float(r["omega"]), float(r["omega"]), len(tri)


lines = []


def emit(s):
    print(s)
    lines.append(s)


emit("I-beam thickness sweep b=%.2f h=%.2f iso E=%.3g nu=%.2f;"
     " D_eff (per unit axial length per cell), all periodic" % (b, h, E, nu))
summ = []
for t in T_SWEEP:
    t0 = time.perf_counter()
    Dan, Dsh = analytical(t), shell_Deff(t)
    Dso, omega, ntri = solid_Deff(t)
    dt = time.perf_counter() - t0
    emit("\n==== t=%.3f (t/h=%.3f)  shell census %.4f, solid omega %.6f,"
         " %d tris  [%.1f s] ====" % (t, t/h, (2*b+h)*t, omega, ntri, dt))
    emit("  %-5s %13s %13s %13s %9s %9s"
         % ("term", "analytical", "shell num", "msg_solid", "sh/so", "an/so"))
    thr = 1e-5*max(np.max(np.abs(Dsh)), np.max(np.abs(Dso)))
    for i, j in TERMS:
        an, sh, so = Dan[i, j], Dsh[i, j], Dso[i, j]
        if abs(an) > thr or abs(sh) > thr or abs(so) > thr:
            ro = sh/so if abs(so) > thr else np.inf
            ra = an/so if abs(so) > thr else np.inf
            emit("  D%d%d   %13.5e %13.5e %13.5e %9.4f %9.4f"
                 % (i+1, j+1, an, sh, so, ro, ra))
    summ.append((t, Dsh[0, 0]/Dso[0, 0], Dsh[1, 1]/Dso[1, 1],
                 Dsh[2, 2]/Dso[2, 2], Dso[1, 2]/Dso[1, 1],
                 Dsh[3, 3]/Dso[3, 3], Dsh[4, 4]/Dso[4, 4],
                 Dsh[5, 5]/Dso[5, 5]))

emit("\n==== summary: shell/solid ratios and the shell-invisible D23 ====")
emit("  %-6s %8s %8s %8s %8s %8s %8s %11s"
     % ("t/h", "D11", "D22", "D33", "D44", "D55", "D66", "D23so/D22so"))
for t, r11, r22, r33, f23, r44, r55, r66 in summ:
    emit("  %-6.3f %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %11.4f"
         % (t/h, r11, r22, r33, r44, r55, r66, f23))

with open(OUT, "w") as f:
    f.write("# thick I-beam sweep: msg_shell vs msg_solid equivalent 3-D"
            " solid D_eff, periodic\n")
    f.write("# order [e11 e22 e33 2e23 2e13 2e12]; analytical = thin-wall"
            " closed forms (frame D44 with solid params)\n")
    f.write("\n".join(lines) + "\n")
print("\nwrote %s" % OUT)
