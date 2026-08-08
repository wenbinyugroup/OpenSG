"""msg-solid on the earlier TPMS attempt (`preovios_try_solid.msh`, gmsh
4-node tets) — a DIFFERENT surface from Sample_1/Sample_2: midsurface area
3.186 (Neovius-class) vs Schwarz-P's 2.345, relative density 0.100,
sheet thickness t = 2V/S_free = 0.0314.

The mesh is converted to the SwiftComp `.sc` layout the solver reads, then
homogenized with aluminium E = 69 GPa, nu = 0.3, periodic in all three
directions.  Writes preovios_try.sc, the timed .out and the mesh png.

Run (from this folder):  python run_previous_try.py"""
import numpy as np
import jax.numpy as jnp

from opensg_solid.sg_homo import plate_homo_2d

MSH = ("../../../../tests/08072026_square_shell_mesh/TPMS_SC_files/"
       "preovios_try_solid.msh")
SC = "preovios_try.sc"
E, nu = 69.0e9, 0.30
G = E/(2*(1+nu))

ln = open(MSH).read().split("\n")
i = ln.index("$Nodes")
nn = int(ln[i+1])
nd = np.array([[float(v) for v in ln[i+2+k].split()[1:4]] for k in range(nn)])
j = ln.index("$Elements")
ne = int(ln[j+1])
el = []
for k in range(ne):
    p = ln[j+2+k].split()
    if int(p[1]) == 4:                       # 4-node tet
        ntag = int(p[2])
        el.append([int(v) for v in p[3+ntag:3+ntag+4]])
el = np.array(el, int)

out = ["0", "0 0 0 0", "", "3 %d %d 1 0 0" % (len(nd), len(el)), ""]
out += ["%d %.12f %.12f %.12f" % (k+1, x, y, z)
        for k, (x, y, z) in enumerate(nd)]
out.append("")
zeros = " " + " ".join(["0"]*16)
out += ["%d 1 %d %d %d %d%s" % (k+1, a, b, c, d, zeros)
        for k, (a, b, c, d) in enumerate(el)]
out += ["1 0 1", "0 0", "69000 0.3", "", "1.0"]
open(SC, "w").write("\n".join(out) + "\n")
print("%d nodes, %d tets -> %s" % (len(nd), len(el), SC))

r = plate_homo_2d(SC, material_param=jnp.array([(E, E, E, G, G, G,
                                                 nu, nu, nu)]),
                  angles=jnp.array([0.0]), n_model=3, workdir=".", plot=True)
C = np.asarray(r["C_eff"])
C = 0.5*(C + C.T)
om = float(r["omega"])
Cc = C*om
print("omega (relative density) = %.6f   solve %.1f s" % (om, r["solve_time"]))
for i in range(6):
    print("  " + " ".join("%13.5e" % Cc[i, j] for j in range(6)))
with open("preovios_try_msg_solid.out", "w") as f:
    f.write("# TPMS previous try (preovios_try_solid.msh) -- msg-solid 3-D SG,"
            " periodic all 3 dirs, alu E=69 GPa nu=0.3\n"
            "# per-UNIT-CELL C (Pa) = C_eff x omega; omega %.6g;"
            " solve time: %.2f s\n# ---- C (6x6, Pa) ----\n"
            % (om, r["solve_time"]))
    for i in range(6):
        f.write(" ".join("%16.8e" % Cc[i, j] for j in range(6)) + "\n")
print("wrote preovios_try_msg_solid.out")
