"""msg_shell (RM) equivalent 3-D solid properties: Deo cellular solid,
theta = +15 / -15 deg, vs Deo MSG-TW and SwiftComp (MSG solid; equivalent to
OpenSG-solid).  Writes .out (with solve time), .dat, mesh .png per angle.
Run (after make_deo_cellular_yamls.py):  python deo_cellular_solid_props.py"""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml as _yaml

from opensg_shell import build_solid_bundle

l, h = 0.10, 0.10
DEO = {15: {"C11": (4736.9, 4678.9), "C12": (1089.4, 1105.5),
            "C13": (381.81, 386.88), "C22": (2446.39, 2488.9),
            "C23": (847.44, 860.89), "C33": (306.99, 311.48),
            "C44": (4.3215, 4.1919), "C55": (564.15, 573.11),
            "C66": (997.52, 1000.1)},
       -15: {"C11": (7507.4, 7352.9), "C12": (1094.1, 1075.2),
             "C13": (-220.53, -213.94), "C22": (4154.9, 4080.0),
             "C23": (-847.45, -821.77), "C33": (180.75, 173.48),
             "C44": (2.6775, 2.4705), "C55": (332.17, 346.01),
             "C66": (1694.2, 1706.4)}}
IJ = {"C11": (0, 0), "C12": (0, 1), "C13": (0, 2), "C22": (1, 1),
      "C23": (1, 2), "C33": (2, 2), "C44": (3, 3), "C55": (4, 4),
      "C66": (5, 5)}

lines = ["# Deo cellular solid (MAMS 2023 Fig 4, Tables 1-2), all MPa",
         "# ours = OpenSG msg_shell (RM segments + MSG G); SwiftComp = the"
         " paper's MSG solid reference (= OpenSG-solid equivalent)"]
for thd in (15, -15):
    th = np.radians(thd)
    area = (2*l*np.cos(th))*(2*(h + l*np.sin(th)))
    yml = "deo_cellular_%+03d_1Dshell.yaml" % thd
    t0 = time.perf_counter()
    b = build_solid_bundle(yml, cell_area=area)
    dt = time.perf_counter() - t0
    C = np.asarray(b["C3D"])*1e-6
    with open("deo_cellular_%+03d_msg_shell.out" % thd, "w") as f:
        f.write("# Deo cellular solid theta=%+d deg -- OpenSG msg_shell (RM)\n"
                "# order [G11 G22 G33 2G23 2G13 2G12]; MPa; cell area %.6f\n"
                "# solve time: %.2f s\n# ---- C_eff (6x6, MPa) ----\n"
                % (thd, area, dt))
        for i in range(6):
            f.write(" ".join("%16.8e" % C[i, j] for j in range(6)) + "\n")
    d = _yaml.safe_load(open(yml))
    row = lambda r: " ".join(str(x) for x in
                             (r if isinstance(r, list) else [r])).split()
    nd = np.array([[float(v) for v in row(r)][:2] for r in d["nodes"]])
    el = np.array([[int(v) for v in row(r)] for r in d["elements"]]) - 1
    fig, ax = plt.subplots(figsize=(5, 6))
    for a, c in el:
        ax.plot([nd[a, 0], nd[c, 0]], [nd[a, 1], nd[c, 1]], "-",
                color="0.25", lw=1.4)
    ax.plot(nd[:, 0], nd[:, 1], ".", ms=3, color="crimson")
    ax.set_aspect("equal")
    ax.set_xlabel("y2 (m)")
    ax.set_ylabel("y3 (m)")
    plt.tight_layout()
    plt.savefig("deo_cellular_%+03d_mesh.png" % thd, dpi=150)
    plt.close(fig)
    lines.append("# ---- theta = %+d deg (Table %d)  solve %.1f s ----"
                 % (thd, 1 if thd > 0 else 2, dt))
    lines.append("# %-5s %11s %11s %11s %9s %9s"
                 % ("term", "msg_shell", "DeoTW", "SwiftComp", "ours%",
                    "DeoTW%"))
    for nm, (tw, so) in DEO[thd].items():
        i, j = IJ[nm]
        lines.append("  %-5s %11.2f %11.2f %11.2f %+9.2f %+9.2f"
                     % (nm, C[i, j], tw, so, 100*(C[i, j]-so)/so,
                        100*(tw-so)/so))
open("deo_cellular_compare.dat", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
