"""msg_shell (RM) equivalent 3-D solid properties: Deo hierarchical square,
vs Deo MSG-TW and SwiftComp (MSG solid; = OpenSG-solid equivalent).
Writes deo_square_msg_shell.out (with solve time), .dat, mesh .png.
Run (after make_deo_square_yaml.py):  python deo_square_solid_props.py"""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml as _yaml

from opensg_shell import build_solid_bundle

R = 0.10
DEO = {"C11": (5186.39, 5099.03), "C12": (24.83, 26.75),
       "C13": (24.83, 26.75), "C22": (47.13, 50.46),
       "C23": (27.94, 30.59), "C33": (47.13, 50.45),
       "C44": (3.22, 3.39), "C55": (650.00, 670.92),
       "C66": (650.00, 670.93)}
IJ = {"C11": (0, 0), "C12": (0, 1), "C13": (0, 2), "C22": (1, 1),
      "C23": (1, 2), "C33": (2, 2), "C44": (3, 3), "C55": (4, 4),
      "C66": (5, 5)}

t0 = time.perf_counter()
b = build_solid_bundle("deo_square_1Dshell.yaml", cell_area=(2*R)**2)
dt = time.perf_counter() - t0
C = np.asarray(b["C3D"])*1e-6

with open("deo_square_msg_shell.out", "w") as f:
    f.write("# Deo hierarchical square -- OpenSG msg_shell (RM)\n"
            "# order [G11 G22 G33 2G23 2G13 2G12]; MPa; cell area %.4f\n"
            "# solve time: %.2f s\n# ---- C_eff (6x6, MPa) ----\n"
            % ((2*R)**2, dt))
    for i in range(6):
        f.write(" ".join("%16.8e" % C[i, j] for j in range(6)) + "\n")

lines = ["# Deo hierarchical square (MAMS 2023 Table 3), all MPa;"
         "  solve %.1f s" % dt,
         "# ours = OpenSG msg_shell (RM); SwiftComp = the paper's MSG solid"
         " reference (= OpenSG-solid equivalent)",
         "# %-5s %11s %11s %11s %9s %9s"
         % ("term", "msg_shell", "DeoTW", "SwiftComp", "ours%", "DeoTW%")]
for nm, (tw, so) in DEO.items():
    i, j = IJ[nm]
    lines.append("  %-5s %11.2f %11.2f %11.2f %+9.2f %+9.2f"
                 % (nm, C[i, j], tw, so, 100*(C[i, j]-so)/so,
                    100*(tw-so)/so))
open("deo_square_compare.dat", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

d = _yaml.safe_load(open("deo_square_1Dshell.yaml"))
row = lambda r: " ".join(str(x) for x in
                         (r if isinstance(r, list) else [r])).split()
nd = np.array([[float(v) for v in row(r)][:2] for r in d["nodes"]])
el = np.array([[int(v) for v in row(r)] for r in d["elements"]]) - 1
fig, ax = plt.subplots(figsize=(5.5, 5.5))
for a, c in el:
    ax.plot([nd[a, 0], nd[c, 0]], [nd[a, 1], nd[c, 1]], "-",
            color="0.25", lw=1.4)
ax.plot(nd[:, 0], nd[:, 1], ".", ms=3, color="crimson")
ax.set_aspect("equal")
ax.set_xlabel("y2 (m)")
ax.set_ylabel("y3 (m)")
plt.tight_layout()
plt.savefig("deo_square_mesh.png", dpi=150)
