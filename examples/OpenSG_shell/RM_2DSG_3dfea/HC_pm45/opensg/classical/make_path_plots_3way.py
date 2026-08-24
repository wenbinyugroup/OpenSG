"""make_path_plots_3way.py -- the 6-component stress comparison along
both paths with THREE curves: Abaqus 3-D FEA, the MSG-RM refined
recovery, and the MSG CLASSICAL-plate recovery (this folder's run).
All material frame, row-aligned by construction.

  path_1/  S11..S23 vs x1        top surface of the top face sheet
  path_2/  S11..S23 vs s/s_tot   through-thickness zig-zag (0 = top
                                 surface, 1 = bottom); dotted vertical
                                 lines mark material changes

In:  ../path_N/path_N_opensg.dat,
     ../../abaqus/station_path_stress/path_N_abaqus.dat,
     ./path_N_classical.dat
Out: ./path_N_S11.png .. _S23.png (12 total) + ./rms_3way.txt
"""
import datetime
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                            # noqa: E402
import numpy as np                                         # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
COMP = ["S11", "S22", "S33", "S12", "S13", "S23"]
NICE = {"S11": "11", "S22": "22", "S33": "33", "S12": "12",
        "S13": "13", "S23": "23"}
XLAB = {1: r"$x_1$  (mm)",
        2: r"$s/s_{\mathrm{total}}$  (0 = top surface, 1 = bottom"
           r" surface)"}

print("start : %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
lines = ["# normalised RMS of (model - 3-D) along each path, material"
         " frame\n# %-8s %-6s %10s %14s\n" % ("path", "comp",
                                              "MSG-RM %", "classical %")]
for n in (1, 2):
    fo = os.path.join(HERE, "..", "path_%d" % n,
                      "path_%d_opensg.dat" % n)
    fa = os.path.join(HERE, "..", "..", "abaqus", "station_path_stress",
                      "path_%d_abaqus.dat" % n)
    fc = os.path.join(HERE, "path_%d_classical.dat" % n)
    if not all(os.path.exists(p) for p in (fo, fa, fc)):
        print("path %d: a .dat is missing -- skipped" % n)
        continue
    O, A, K = np.loadtxt(fo), np.loadtxt(fa), np.loadtxt(fc)
    if not (O.shape == A.shape == K.shape):
        raise SystemExit("path %d: the .dat files differ in shape" % n)
    cerr = max(np.abs(O[:, :3] - A[:, :3]).max(),
               np.abs(O[:, :3] - K[:, :3]).max())
    print("path %d: coords agreement %.2e (tol 1e-4) %s"
          % (n, cerr, "OK" if cerr < 1e-4 else "FAIL"))
    if cerr >= 1e-4:
        raise SystemExit("path %d: rows are not aligned -- regenerate"
                         " the .dat files from the same .coords" % n)
    s, mt = O[:, 0], O[:, 3].astype(int)
    mchg = np.where(np.diff(mt) != 0)[0]
    for j, c in enumerate(COMP):
        a, b, k = O[:, 4 + j], A[:, 4 + j], K[:, 4 + j]
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        ax.plot(s, b, "-", lw=1.8, color="#c0392b",
                label="Abaqus 3-D FEA (C3D20R)")
        ax.plot(s, a, "--", lw=1.8, color="k",
                label="MSG-RM 2-D SG recovery")
        ax.plot(s, k, "-.", lw=1.6, color="#2471a3",
                label="MSG classical-plate recovery")
        for i in mchg:
            ax.axvline(0.5 * (s[i] + s[i + 1]), color="k", lw=0.7,
                       ls=":", alpha=0.6)
        ax.set_xlabel(XLAB[n])
        ax.set_ylabel(r"$\sigma_{%s}$  (MPa)" % NICE[c])
        ax.grid(alpha=0.3)
        ax.axhline(0.0, color="k", lw=0.6, alpha=0.4)
        if n == 2:
            ax.set_xlim(0, 1)
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        png = os.path.join(HERE, "path_%d_%s.png" % (n, c))
        fig.savefig(png, dpi=170, bbox_inches="tight")
        plt.close(fig)
        den = max(np.abs(b).max(), 1e-30)
        r_rm = 100 * np.sqrt(np.mean((a - b) ** 2)) / den
        r_cl = 100 * np.sqrt(np.mean((k - b) ** 2)) / den
        lines.append("%-8s %-6s %10.2f %14.2f\n"
                     % ("path_%d" % n, c, r_rm, r_cl))
        print("  %s  RMS: RM %.2f %%, classical %.2f %%"
              % (os.path.basename(png), r_rm, r_cl))
open(os.path.join(HERE, "rms_3way.txt"), "w").writelines(lines)
print("wrote rms_3way.txt")
print("end   : %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
