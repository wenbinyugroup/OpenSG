"""I-beam FINAL D_eff (relaxed, Dee + V0^T Dhe): analytical vs numerical
msg_shell vs msg_solid (opensg_solid n_model=3), all periodic.

Center-ref I, iso: flanges (width b, y3 = +-h/2), web (height h, y2 = 0),
thickness t.  Under the periodic tiling the two flanges coincide with the
neighbours' -> the lattice has ONE horizontal wall line per cell (spacing h)
carrying BOTH flanges: membrane 2t in both routes, bending 2*t^3/12 in the
shell (two independent t-walls sharing tied nodes) vs (2t)^3/12 in the solid
(merged 2t material).  Webs: t walls, spacing b.

Analytical relaxed D_eff (thin-wall lattice, per unit axial length per cell):
  membrane channels do NOT relax further (continuous periodic straight walls
  => average in-wall strain = macro => in-plane plane strain + sigma_nn = 0):
    D11 = E' t (Lf+Lw)   D12 = nu E' t Lf   D13 = nu E' t Lw
    D22 = E' t Lf        D33 = E' t Lw      D23 = 0        (Lf = 2b, Lw = h)
  in-plane shear: rigid-joint frame, both wall families bend guided-guided
  with a free joint rotation th and free lattice rotation rho (sympy below):
    D44 = closed form in (EI_h, EI_v, b, h_sp)
  out-of-plane shear: anti-plane shear flow -- only walls along the shear
  direction carry it (the other family sheds through-thickness shear):
    D55 = G t Lw         D66 = G t Lf
Solid-route frame parameters: EI_h = E'(2t)^3/12, spacing h+t (merged-wall
centres); junction rigidity is NOT in the frame model (known to stiffen the
solid at finite t/h).  Shell-route: EI_h = 2 E' t^3/12, spacing h.

Run (from this folder):  python ibeam_Deff.py
"""
import time

import numpy as np
import sympy as sp
import yaml as _yaml
import jax.numpy as jnp

from opensg_shell import build_solid_bundle
from opensg_solid.sg_homo import plate_homo_2d
from opensg_solid.sg_materials import elem_rotation_from_yaml

############### User Input #################################
b, h, t = 0.5, 1.0, 0.03          # flange width, web height, thickness
E, nu = 70.0e9, 0.30
nseg = 16                          # shell elems per half-flange / half-web
sz = 0.015                         # solid along-wall element size (~t/2)
ntw = 6                            # solid elems through web t
ntf = 12                           # solid elems through the merged 2t flange
OUT = "ibeam_Deff.dat"
############################################################

G = E/(2*(1+nu))
Ep = E/(1-nu**2)
Lf, Lw = 2*b, h

# ---- analytical: membrane + out-of-plane shear ------------------------------
Dan = np.zeros((6, 6))
Dan[0, 0] = Ep*t*(Lf+Lw)
Dan[1, 1] = Ep*t*Lf
Dan[2, 2] = Ep*t*Lw
Dan[0, 1] = Dan[1, 0] = nu*Ep*t*Lf
Dan[0, 2] = Dan[2, 0] = nu*Ep*t*Lw
Dan[4, 4] = G*t*Lw
Dan[5, 5] = G*t*Lf


# ---- analytical D44: rigid-joint frame, minimized over (th, rho) ------------
def frame_D44(EIh, EIv, span_h, span_v):
    """One junction per cell; one vertical wall (span_v, EIv) and one
    horizontal wall line (span_h, EIh), joints periodic images of each other.
    Joint CCW rotation thJ enters the two families with OPPOSITE beam-frame
    signs (vertical wall: axial y3, transverse y2 -> theta_beam = -thJ), so the
    energy depends only on x = rho - thJ:
        U = 6 EIv/lv (g/2 - x)^2 + 6 EIh/lh (g/2 + x)^2,  min over x
        -> D44 = 2 kv kh/(kv + kh),  k = 6 EI/span   (harmonic mean).
    Returns D44 = C44*omega (energy per unit axial length per cell)."""
    g, x = sp.symbols("g x")
    kv = 6*sp.Float(EIv)/sp.Float(span_v)
    kh = 6*sp.Float(EIh)/sp.Float(span_h)
    U = kv*(g/2 - x)**2 + kh*(g/2 + x)**2
    sol = sp.solve(sp.diff(U, x), x)[0]
    Umin = sp.simplify(U.subs(x, sol))
    D44 = float(sp.simplify(2*Umin/g**2))
    assert abs(D44 - 2*float(kv*kh/(kv+kh))) < 1e-6*D44
    return D44


D44_sh_an = frame_D44(2*Ep*t**3/12, Ep*t**3/12, b, h)
D44_so_an = frame_D44(Ep*(2*t)**3/12, Ep*t**3/12, b, h + t)
# square sanity: equal families must give the exact 6 EI / L
sq = frame_D44(Ep*t**3/12, Ep*t**3/12, 1.0, 1.0)
assert abs(sq/(6*(Ep*t**3/12)/1.0) - 1) < 1e-9, sq
Dan[3, 3] = D44_sh_an

# ---- numerical msg_shell: 1-D yaml (junction nodes shared), periodic --------
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
open("_ibeam.yaml", "w").write("\n".join(o) + "\n")

t0 = time.perf_counter()
bun = build_solid_bundle("_ibeam.yaml", cell_area=b*h)
Dsh = np.asarray(bun["D_eff"])
Dsh = 0.5*(Dsh + Dsh.T)
t_sh = time.perf_counter() - t0

# ---- numerical msg_solid: 2-D tri mesh of the actual I, periodic ------------
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
        in_fl = abs(abs(cy) - h/2) < t/2
        in_web = abs(cx) < t/2 and abs(cy) < h/2 - t/2
        if in_fl or in_web:
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

# write the standard OpenSG 2-D solid yaml, then run it exactly as
# solid_props_newarch.py does (yaml -> sc -> plate_homo_2d)
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
open("_ibeam_2Dsolid.yaml", "w").write("\n".join(o) + "\n")

d = _yaml.safe_load(open("_ibeam_2Dsolid.yaml"))
row = lambda r: [float(v) for v in
                 " ".join(str(x) for x in (r if isinstance(r, list) else [r])).split()]
nd_y = np.array([row(r) for r in d["nodes"]], float)
tri_y = np.array([[int(v) for v in row(r)] for r in d["elements"]], int) - 1
ori = np.array(d["elementOrientations"], float)
mat_id = np.ones(len(tri_y), int)
for k, s in enumerate(d["sets"]["element"]):
    mat_id[np.array(s["labels"], int) - 1] = k + 1
sc = {"dim": 2, "nodes": nd_y[:, :2], "cells": [list(c) for c in tri_y],
      "mat_id": mat_id, "materials": {0: {"name": d["materials"][0]["name"]}},
      "scale": 1.0}
material_param = jnp.array([(E, E, E, G, G, G, nu, nu, nu)])

t0 = time.perf_counter()
r = plate_homo_2d(sc, material_param=material_param, angles=jnp.array([0.0]),
                  n_model=3, elem_rotation=elem_rotation_from_yaml(ori),
                  plot=False)
Cso = np.asarray(r["C_eff"]); Cso = 0.5*(Cso + Cso.T)
omega = float(r["omega"])
Dso = Cso*omega
t_so = time.perf_counter() - t0

# ---- table ------------------------------------------------------------------
A_sh = (Lf+Lw)*t
hdr = ("I-beam b=%.2f h=%.2f t=%.3f iso E=%.3g nu=%.2f;  Lf=%.2f Lw=%.2f\n"
       "shell census %.4f, solid omega %.6f;  D44 frame: shell-an %.6e"
       " (EIh=2t^3), solid-an %.6e (EIh=(2t)^3, spacing h+t)\n"
       "shell %d elems %.1f s;  solid %d tris %.1f s"
       % (b, h, t, E, nu, Lf, Lw, A_sh, omega, D44_sh_an, D44_so_an,
          len(cells), t_sh, len(tri), t_so))
print(hdr + "\n")
rows = []
thr = 1e-5*max(np.max(np.abs(Dsh)), np.max(np.abs(Dso)))
print("  %-5s %13s %13s %9s %13s %9s" %
      ("term", "analytical", "shell num", "an/sh", "msg_solid", "sh/so"))
for i in range(6):
    for j in range(i, 6):
        an, sh, so = Dan[i, j], Dsh[i, j], Dso[i, j]
        if abs(an) > thr or abs(sh) > thr or abs(so) > thr:
            rs = an/sh if abs(sh) > thr else np.inf
            ro = sh/so if abs(so) > thr else np.inf
            ln = ("  D%d%d   %13.5e %13.5e %9.4f %13.5e %9.4f"
                  % (i+1, j+1, an, sh, rs, so, ro))
            rows.append(ln); print(ln)

with open(OUT, "w") as f:
    f.write("# I-beam FINAL D_eff (relaxed): analytical vs msg_shell vs"
            " msg_solid, all periodic\n# " + hdr.replace("\n", "\n# ") + "\n")
    f.write("# analytical D44 row uses the SHELL frame parameters; the solid"
            " frame value is in the header\n")
    f.write("# %-5s %13s %13s %9s %13s %9s\n" %
            ("term", "analytical", "shell num", "an/sh", "msg_solid", "sh/so"))
    f.write("\n".join(rows) + "\n")
print("\nwrote %s" % OUT)
