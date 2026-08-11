"""msg_shell (RM) equivalent 3-D solid properties: Deo composite square,
vs Deo MSG-TW and SwiftComp (MSG solid; = OpenSG-solid equivalent).
Writes .out (with solve time), .dat, mesh .png.
Run (after make_deo_composite_yaml.py):  python deo_composite_square_props.py"""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml as _yaml

from opensg_shell import build_solid_bundle
from opensg_shell.cli import sg_cell_area          # the yaml IS the input:
                                                   # omega header, else bbox
YAML = "deo_composite_square_1Dshell.yaml"
L2SQ = sg_cell_area(YAML)                          # cell area (25.4 mm)^2
DEO ={"C11": (16133.90, 15518.11), "C12": (1016.82, 1008.76),
       "C13": (468.38, 459.01), "C15": (-1192.81, -1135.27),
       "C16": (2589.74, 2535.23), "C22": (983.56, 988.88),
       "C23": (3.04e-6, 20.44), "C26": (317.09, 311.80),
       "C33": (453.06, 456.25), "C35": (-146.06, -140.04),
       "C44": (1.08, 1.16), "C55": (547.32, 545.43),
       "C66": (1188.20, 1188.30)}
IJ = {"C11": (0, 0), "C12": (0, 1), "C13": (0, 2), "C15": (0, 4),
      "C16": (0, 5), "C22": (1, 1), "C23": (1, 2), "C26": (1, 5),
      "C33": (2, 2), "C35": (2, 4), "C44": (3, 3), "C55": (4, 4),
      "C66": (5, 5)}

t0 = time.perf_counter()
b = build_solid_bundle(YAML, cell_area=L2SQ)
dt = time.perf_counter() - t0
C = np.asarray(b["C3D"])*1e-6

with open("deo_composite_square_msg_shell.out", "w") as f:
    f.write("# Deo composite square -- OpenSG msg_shell (RM), [15]_8 walls\n"
            "# order [G11 G22 G33 2G23 2G13 2G12]; MPa; cell area %.8f\n"
            "# solve time: %.2f s\n# ---- C_eff (6x6, MPa) ----\n"
            % (L2SQ, dt))
    for i in range(6):
        f.write(" ".join("%16.8e" % C[i, j] for j in range(6)) + "\n")

lines = ["# Deo composite square (MAMS 2023 Fig 13 / Table 4), all MPa;"
         "  solve %.1f s" % dt,
         "# ours = OpenSG msg_shell (RM); SwiftComp = the paper's MSG solid"
         " reference (= OpenSG-solid equivalent)",
         "# %-5s %11s %11s %11s %9s %9s"
         % ("term", "msg_shell", "DeoTW", "SwiftComp", "ours%", "DeoTW%")]
for nm, (tw, so) in DEO.items():
    i, j = IJ[nm]
    pe = 100*(C[i, j]-so)/so if abs(so) > 1e-9 else float("inf")
    pt = 100*(tw-so)/so if abs(so) > 1e-9 else float("inf")
    lines.append("  %-5s %11.2f %11.2f %11.2f %+9.2f %+9.2f"
                 % (nm, C[i, j], tw, so, pe, pt))
open("deo_composite_square_compare.dat", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

d = _yaml.safe_load(open(YAML))
row = lambda r: " ".join(str(x) for x in
                         (r if isinstance(r, list) else [r])).split()
nd = np.array([[float(v) for v in row(r)][:2] for r in d["nodes"]])
el = np.array([[int(v) for v in row(r)] for r in d["elements"]]) - 1
fig, ax = plt.subplots(figsize=(5.5, 5.5))
for a, c in el:
    ax.plot([nd[a, 0]*1e3, nd[c, 0]*1e3], [nd[a, 1]*1e3, nd[c, 1]*1e3], "-",
            color="0.25", lw=1.4)
ax.plot(nd[:, 0]*1e3, nd[:, 1]*1e3, ".", ms=3, color="crimson")
ax.set_aspect("equal")
ax.set_xlabel("y2 (mm)")
ax.set_ylabel("y3 (mm)")
plt.tight_layout()
plt.savefig("deo_composite_square_mesh.png", dpi=150)
