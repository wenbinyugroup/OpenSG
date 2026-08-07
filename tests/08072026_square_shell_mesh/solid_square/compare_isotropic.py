"""ISOTROPIC square tube: msg_shell vs the old-architecture solid BENCHMARK.

Same geometry as everywhere else (wall midline square a = 1.0, t = 0.03), one
ISOTROPIC material E = 70 GPa, nu = 0.3 -- no ply angle, no per-element frames,
so nothing about material orientation can enter the comparison.

  solid : square_solid_bicg_iso.out  (JAX_BICGoptimize, n_model = 3, the benchmark)
  shell : computed here from a 1-D shell SG, periodic, same wall material

Both normalized by the wall material area 4*a*t = 0.12.
Writes square_solid_iso_compare.dat.

Run (from this folder, AFTER solid_props_bicg_iso.py):
    python compare_isotropic.py
"""
import numpy as np

from opensg_shell import build_solid_bundle
from opensg_shell.solid_props import elastic_constants

############### User Input #################################
a, t_w, nseg = 1.0, 0.03, 40
E_iso, nu_iso = 70.0e9, 0.30
SOLID = "square_solid_bicg_iso.out"
YAML = "square_tube_1Dshell_iso.yaml"
OUT = "square_solid_iso_compare.dat"
############################################################

G_iso = E_iso/(2*(1+nu_iso))
CELL_AREA = 4*a*t_w

h = a/2
corners = [(-h, -h), (h, -h), (h, h), (-h, h)]
pts, tans = [], []
for k in range(4):
    p0, p1 = np.array(corners[k]), np.array(corners[(k+1) % 4])
    tv = (p1-p0)/np.linalg.norm(p1-p0)
    for i in range(nseg):
        pts.append(p0 + (p1-p0)*i/nseg)
        tans.append(tv)
pts = np.array(pts); m = len(pts)
o = ["nodes:"]
for x, y in pts:
    o.append("- [%.12f %.12f 0.00000000]" % (x, y))
o.append("elements:")
for i in range(m):
    o.append("- [%d %d]" % (i+1, (i+1) % m + 1))
o.append("elementOrientations:")
for i in range(m):
    tx, ty = tans[i]
    e3 = np.cross([0, 0, 1.0], [tx, ty, 0])
    o.append("- [0.0, 0.0, 1.0, %.9f, %.9f, 0.0, %.9f, %.9f, 0.0]"
             % (tx, ty, e3[0], e3[1]))
o += ["sections:", "- type: shell", "  elementSet: layup_0", "  layup:",
      "  - - iso", "    - %.6e" % t_w, "    - 0.0",
      "materials:", "- name: iso", "  density: 2700.0", "  elastic:",
      "    E: [%.6e, %.6e, %.6e]" % (E_iso, E_iso, E_iso),
      "    G: [%.6e, %.6e, %.6e]" % (G_iso, G_iso, G_iso),
      "    nu: [%.6f, %.6f, %.6f]" % (nu_iso, nu_iso, nu_iso),
      "sets:", "  element:", "  - name: layup_0", "    labels:"]
o += ["    - %d" % (e+1) for e in range(m)]
o.append("reference: center")
open(YAML, "w").write("\n".join(o) + "\n")

Cso, kso, mode = [], {}, None
for ln in open(SOLID):
    if ln.startswith("# ---- C"):
        mode = "C"; continue
    if ln.startswith("# ---- 9"):
        mode = "k"; continue
    if ln.startswith("#"):
        continue
    p = ln.split()
    if mode == "C" and len(p) == 6:
        Cso.append([float(v) for v in p])
    elif mode == "k" and len(p) == 2:
        kso[p[0]] = float(p[1])
Cso = np.array(Cso)

Csh = np.asarray(build_solid_bundle(YAML, cell_area=CELL_AREA)["C3D"])
ksh, _ = elastic_constants(Csh)

thr = 1e-6*np.max(np.abs(Cso))
hdr = "  %-6s %16s %16s %9s %9s" % ("term", "msg_shell", "solid (bench)",
                                    "ratio", "pct")
lines = []
for i in range(6):
    for j in range(i, 6):
        if abs(Cso[i, j]) > thr or abs(Csh[i, j]) > thr:
            b = Cso[i, j]
            lines.append("  C%d%d    %16.6e %16.6e %9.4f %+9.2f"
                         % (i+1, j+1, Csh[i, j], b, Csh[i, j]/b,
                            100*(Csh[i, j]-b)/b))
krows = []
for k in ("E1", "E2", "E3", "G23", "G13", "G12", "nu12", "nu13", "nu23"):
    b = kso[k]
    krows.append("  %-6s %16.6e %16.6e %9.4f %+9.2f"
                 % (k, ksh[k], b, ksh[k]/b, 100*(ksh[k]-b)/b))

with open(OUT, "w") as f:
    f.write("# ISOTROPIC square tube (a=%.2f, t=%.3f, E=%.3e, nu=%.2f)\n"
            % (a, t_w, E_iso, nu_iso))
    f.write("# msg_shell (1-D shell SG, periodic) vs the OLD-ARCHITECTURE\n")
    f.write("# solid benchmark (%s), both normalized by the wall\n" % SOLID)
    f.write("# material area %.4f; order [e11 e22 e33 2e23 2e13 2e12]\n"
            % CELL_AREA)
    f.write("# ---- stiffness terms (Pa) ----\n")
    f.write("#" + hdr[1:] + "\n")
    f.write("\n".join(lines) + "\n")
    f.write("# ---- 9 effective constants ----\n")
    f.write("#" + hdr[1:].replace("term  ", "const ") + "\n")
    f.write("\n".join(krows) + "\n")

print(hdr)
print("\n".join(lines))
print(hdr.replace("term  ", "const "))
print("\n".join(krows))
print("\nwrote %s" % OUT)
