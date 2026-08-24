"""make_path_plots_4way.py -- the 6-component stress comparison along
both paths with FOUR curves, every datapoint marked:

    Abaqus 3-D FEA          solid red, circles
    MSG-RM (OpenSG)         dashed black, squares
    classical (OpenSG)      dash-dot blue, triangles
    classical (SwiftComp)   dotted green, STARS

All material frame, per-element values, row-aligned by construction.

  path_1/  S11..S23 vs x1        top surface of the top face sheet
  path_2/  S11..S23 vs s/s_tot   through-thickness zig-zag (0 = top,
                                 1 = bottom); dotted vertical lines mark
                                 material changes

In:  ../../opensg/path_N/path_N_opensg.dat,
     ../../abaqus/station_path_stress/path_N_abaqus.dat,
     ../../opensg/classical/path_N_classical.dat,
     ./path_N_swiftcomp.dat
Out: ./path_N_S11.png .. _S23.png (12 total) + ./rms_4way.txt
"""
import datetime
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                            # noqa: E402
import numpy as np                                         # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OPG = os.path.join(HERE, "..", "..", "opensg")
COMP = ["S11", "S22", "S33", "S12", "S13", "S23"]
NICE = {"S11": "11", "S22": "22", "S33": "33", "S12": "12",
        "S13": "13", "S23": "23"}
XLAB = {1: r"$x_1$  (mm)",
        2: r"$s/s_{\mathrm{total}}$  (0 = top surface, 1 = bottom"
           r" surface)"}
CURVES = [  # (tag, style dict, label) -- plotted in this z-order
    ("abq", dict(ls="-", color="#c0392b", marker="o", ms=3.2, lw=1.7),
     "Reference 3-D FEA (Abaqus C3D20R)"),
    ("rm", dict(ls="--", color="k", marker="s", ms=2.8, lw=1.6),
     "Shear-refined (RM) Plate (OpenSG)"),
    ("cl", dict(ls="-.", color="#2471a3", marker="s", ms=7.0, lw=1.4,
                markerfacecolor="none"),
     "Classical Plate (OpenSG)"),
]
# the SwiftComp classical curve was dropped from the FIGURES on request
# (it coincides with the OpenSG classical to plotting precision --
# path_N_swiftcomp.dat and the rms column keep the record)

print("start : %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
lines = ["# normalised RMS of (model - 3-D) along each path, material"
         " frame\n# %-8s %-6s %10s %14s %16s\n"
         % ("path", "comp", "MSG-RM %", "OpenSG-cl %", "SwiftComp-cl %")]
for n in (1, 2):
    F = {"rm": os.path.join(OPG, "path_%d" % n, "path_%d_opensg.dat" % n),
         "abq": os.path.join(HERE, "..", "..", "abaqus",
                             "station_path_stress",
                             "path_%d_abaqus.dat" % n),
         "cl": os.path.join(OPG, "classical", "path_%d_classical.dat" % n),
         "sw": os.path.join(HERE, "path_%d_swiftcomp.dat" % n)}
    if not all(os.path.exists(p) for p in F.values()):
        print("path %d: a .dat is missing -- skipped" % n)
        continue
    D = {k: np.loadtxt(p) for k, p in F.items()}
    shp = {k: v.shape for k, v in D.items()}
    if len(set(shp.values())) != 1:
        raise SystemExit("path %d: shapes differ: %s" % (n, shp))
    cerr = max(np.abs(D[k][:, :3] - D["abq"][:, :3]).max() for k in D)
    print("path %d: coords agreement %.2e (tol 1e-4) %s"
          % (n, cerr, "OK" if cerr < 1e-4 else "FAIL"))
    if cerr >= 1e-4:
        raise SystemExit("path %d: rows are not aligned" % n)
    s, mt = D["abq"][:, 0], D["abq"][:, 3].astype(int)
    mchg = np.where(np.diff(mt) != 0)[0]
    for j, c in enumerate(COMP):
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        for tag, st, lab in CURVES:
            ax.plot(s, D[tag][:, 4 + j], label=lab, **st)
        for i in mchg:
            ax.axvline(0.5 * (s[i] + s[i + 1]), color="k", lw=0.7,
                       ls=":", alpha=0.6)
        ax.set_xlabel(XLAB[n])
        ax.set_ylabel(r"$\sigma_{%s}$  (MPa)" % NICE[c])
        ax.grid(alpha=0.3)
        if n == 2:
            ax.set_xlim(0, 1)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        png = os.path.join(HERE, "path_%d_%s.png" % (n, c))
        fig.savefig(png, dpi=170, bbox_inches="tight")
        plt.close(fig)
        den = max(np.abs(D["abq"][:, 4 + j]).max(), 1e-30)
        rms = {k: 100 * np.sqrt(np.mean(
            (D[k][:, 4 + j] - D["abq"][:, 4 + j]) ** 2)) / den
            for k in ("rm", "cl", "sw")}
        lines.append("%-8s %-6s %10.2f %14.2f %16.2f\n"
                     % ("path_%d" % n, c, rms["rm"], rms["cl"],
                        rms["sw"]))
        print("  %s  RMS: RM %.2f %%  OpenSG-cl %.2f %%  SwiftComp-cl"
              " %.2f %%" % (os.path.basename(png), rms["rm"],
                            rms["cl"], rms["sw"]))
open(os.path.join(HERE, "rms_4way.txt"), "w").writelines(lines)
print("wrote rms_4way.txt")
print("end   : %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
