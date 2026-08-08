"""Three-way comparison of equivalent 3-D solid properties for the square tube.

  msg_shell   1-D shell SG  (opensg_shell.build_solid_bundle, periodic)
  new solid   2-D solid SG  (opensg_solid.sg_homo, n_model = 3)
  old solid   2-D solid SG  (JAX_BICGoptimize single script, n_model = 3)
              -- the BENCHMARK for solid results

All three run the UNROTATED ply (0 deg): that is the only material state the
.sc-based old driver can express, since a .sc carries one material id per
element and one fibre angle per material.  Same wall material, same geometry,
all normalized by the wall material area 0.12.

Writes square_tube_1Dshell_0deg.yaml (the 0 deg shell SG) and
square_solid_three_way.dat.  Percentages are against the old-solid benchmark.

Run (from this folder):  python compare_three_routes.py
"""
import numpy as np

from opensg_shell import build_solid_bundle
from opensg_shell.sg_homo import elastic_constants

############### User Input #################################
a, t_w, nseg = 1.0, 0.03, 10       # midline side, wall thickness, elems/side
E = [142.0e9, 9.8e9, 9.8e9]
G = [6.0e9, 6.0e9, 4.8e9]
nu = [0.30, 0.30, 0.42]
CELL_AREA = 4*a*t_w                # wall material area = 0.12
NEW = "square_solid_newarch_plain.out"
OLD = "square_solid_bicg.out"
SHELL_YAML = "square_tube_1Dshell_0deg.yaml"
OUT = "square_solid_three_way.dat"
############################################################

# ---- the 0 deg shell SG on the wall midline -------------------------------
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
out = ["nodes:"]
for x, y in pts:
    out.append("- [%.12f %.12f 0.00000000]" % (x, y))
out.append("elements:")
for i in range(m):
    out.append("- [%d %d]" % (i+1, (i+1) % m + 1))
out.append("elementOrientations:")
for i in range(m):
    tx, ty = tans[i]
    e3 = np.cross([0, 0, 1.0], [tx, ty, 0])
    out.append("- [0.0, 0.0, 1.0, %.9f, %.9f, 0.0, %.9f, %.9f, 0.0]"
               % (tx, ty, e3[0], e3[1]))
out += ["sections:", "- type: shell", "  elementSet: layup_0", "  layup:",
        "  - - ply", "    - %.6e" % t_w, "    - 0.0",
        "materials:", "- name: ply", "  density: 1600.0", "  elastic:",
        "    E: [%.6e, %.6e, %.6e]" % tuple(E),
        "    G: [%.6e, %.6e, %.6e]" % tuple(G),
        "    nu: [%.6f, %.6f, %.6f]" % tuple(nu),
        "sets:", "  element:", "  - name: layup_0", "    labels:"]
out += ["    - %d" % (e+1) for e in range(m)]
out.append("reference: center")
open(SHELL_YAML, "w").write("\n".join(out) + "\n")


def read_out(path):
    C, cons, mode = [], {}, None
    for ln in open(path):
        if ln.startswith("# ---- C"):
            mode = "C"; continue
        if ln.startswith("# ---- 9"):
            mode = "k"; continue
        if ln.startswith("#"):
            continue
        p = ln.split()
        if mode == "C" and len(p) == 6:
            C.append([float(v) for v in p])
        elif mode == "k" and len(p) == 2:
            cons[p[0]] = float(p[1])
    return np.array(C), cons


Cnew, knew = read_out(NEW)
Cold, kold = read_out(OLD)
Csh = np.asarray(build_solid_bundle(SHELL_YAML, cell_area=CELL_AREA)["C3D"])
ksh, _ = elastic_constants(Csh)

with open("square_solid_msg_shell_0deg.out", "w") as f:
    f.write("# equivalent 3-D solid properties from the 1-D SHELL SG,"
            " UNROTATED ply (0 deg)\n")
    f.write("# msg_shell: opensg_shell.build_solid_bundle, periodic\n")
    f.write("# source: %s ; normalized by the wall material area %.4f\n"
            % (SHELL_YAML, CELL_AREA))
    f.write("# order [e11 e22 e33 2e23 2e13 2e12]\n")
    f.write("# ---- C3D (6x6, Pa) ----\n")
    for i in range(6):
        f.write(" ".join("%16.8e" % Csh[i, j] for j in range(6)) + "\n")
    f.write("# ---- 9 effective constants ----\n")
    for k in ("E1", "E2", "E3", "G23", "G13", "G12", "nu12", "nu13", "nu23"):
        f.write("%-5s %16.8e\n" % (k, ksh[k]))

thr = 1e-6*np.max(np.abs(Cold))
lines, hdr = [], ("  %-6s %16s %16s %16s %9s %9s"
                  % ("term", "msg_shell", "new solid", "old solid",
                     "shell/old", "new/old"))
for i in range(6):
    for j in range(i, 6):
        if abs(Cold[i, j]) > thr or abs(Cnew[i, j]) > thr:
            b = Cold[i, j]
            lines.append("  C%d%d    %16.6e %16.6e %16.6e %9.4f %9.4f"
                         % (i+1, j+1, Csh[i, j], Cnew[i, j], b,
                            Csh[i, j]/b, Cnew[i, j]/b))
krows = []
for k in ("E1", "E2", "E3", "G23", "G13", "G12", "nu12", "nu13", "nu23"):
    b = kold[k]
    krows.append("  %-6s %16.6e %16.6e %16.6e %9.4f %9.4f"
                 % (k, ksh[k], knew[k], b, ksh[k]/b, knew[k]/b))

with open(OUT, "w") as f:
    f.write("# square tube -- equivalent 3-D solid properties, three routes\n")
    f.write("# msg_shell = 1-D shell SG (opensg_shell, periodic)\n")
    f.write("# new solid = 2-D solid SG (opensg_solid n_model=3)\n")
    f.write("# old solid = 2-D solid SG (JAX_BICGoptimize) -- the BENCHMARK\n")
    f.write("# UNROTATED ply (0 deg): the only state the .sc route can express\n")
    f.write("# all normalized by the wall material area %.4f;"
            " order [e11 e22 e33 2e23 2e13 2e12]\n" % CELL_AREA)
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
