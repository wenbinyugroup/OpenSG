"""abaqus_cell_scan.py -- is the path-1 mismatch a FREE-EDGE effect?

Test: extract the SAME top-surface-row profile (the path-1 quantity,
material frame, Richardson onto midspan) from EVERY width-offset copy
of the station window, y0 = YCELL + k P for k = -3..+3 (k = +-3 sits
one cell from the free width edges, k = 0 is the station itself).  The
honeycomb is periodic in the width with pitch P, so if the free edges
do not reach the panel centre, every interior window must show the
SAME profile; a systematic drift towards the |k| = 3 windows is the
free-edge signature.  Whatever the k = 0 window shares with its
neighbours CANNOT be a free-edge effect.

In:  ../../abaqus/3dfea_output/45pm_S_material.csv
Out: ./abaqus_cell_scan.dat (k y0 x1 S11..S23 per row),
     ./cellscan_S11.png / _S22.png / _S12.png, the printed RMS table
"""
import datetime
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                            # noqa: E402
import numpy as np                                         # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
COMP = ["S11", "S22", "S33", "S12", "S13", "S23"]
YCELL, P = 30.10244, 7.5256081
HALF = P / 2.0
ZTOP = 4.782412                      # path-1 element row (SG x2)

print("start : %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
S = np.genfromtxt(os.path.join(HERE, "..", "..", "abaqus",
                               "3dfea_output", "45pm_S_material.csv"),
                  delimiter=",", names=True)
XC = 0.5 * (S["x"].min() + S["x"].max())
lay = np.unique(np.round(S["x"] - XC, 4))
d1 = np.sort(np.abs(lay[np.abs(lay) > 1e-6]))[0]
print("3-D dump  : %d rows; midspan x = %.6f; layers +/-%.5f, +/-%.5f"
      % (len(S), XC, d1, 3 * d1))


def layer(o):
    """one spanwise centroid layer, TOP row only, sorted by width y"""
    k = np.abs(np.round(S["x"] - XC, 4) - round(o, 4)) < 1e-4
    k &= np.abs(S["z"] - ZTOP) < 1e-3
    r = S[k]
    return r[np.argsort(np.round(r["y"], 5))]


avg = {}
for tag, dd in (("in", d1), ("out", 3.0 * d1)):
    m, p = layer(-dd), layer(+dd)
    if len(m) != len(p) or not np.allclose(m["y"], p["y"], atol=1e-5):
        raise SystemExit("layer pair %s does not align" % tag)
    avg[tag] = {c: 0.5 * (m[c] + p[c]) for c in COMP}
    avg[tag]["y"] = m["y"]
F = np.stack([(9.0 * avg["in"][c] - avg["out"][c]) / 8.0
              for c in COMP], axis=-1)
Y = avg["in"]["y"]
print("top row   : %d elements across the width (y %.3f..%.3f)"
      % (len(Y), Y.min(), Y.max()))

# ---- windows
wins = {}
for k in range(-3, 4):
    y0 = YCELL + k * P
    m = (Y > y0 - HALF) & (Y < y0 + HALF)
    wins[k] = (Y[m] - y0, F[m])
n0 = len(wins[0][0])
print("windows   : " + ", ".join("k=%+d:%d" % (k, len(wins[k][0]))
                                 for k in sorted(wins)))

x0 = wins[0][0]
print("\n---- RMS deviation from the k = 0 (station) window, %% of the"
      " station max|.|")
print("%-4s %9s " % ("k", "y0")
      + " ".join("%7s" % c for c in COMP) + "   grid_off")
rows = []
for k in sorted(wins):
    xk, Fk = wins[k]
    goff = (np.abs(xk - x0).max() if len(xk) == n0 else np.inf)
    if len(xk) != n0 or goff > 1e-3:
        # non-matching grid: interpolate onto the station grid
        Fk = np.stack([np.interp(x0, xk, Fk[:, j])
                       for j in range(6)], axis=-1)
        xk = x0
    rows.append((k, Fk))
    dev = [100 * np.sqrt(np.mean((Fk[:, j] - wins[0][1][:, j]) ** 2))
           / max(np.abs(wins[0][1][:, j]).max(), 1e-30)
           for j in range(6)]
    print("%-4d %9.4f " % (k, YCELL + k * P)
          + " ".join("%7.2f" % v for v in dev)
          + "   %.1e" % goff)

out = os.path.join(HERE, "abaqus_cell_scan.dat")
with open(out, "w") as f:
    f.write("# top-surface-row stress profile of every width-offset"
            " window copy, MATERIAL frame,\n# Richardson onto midspan"
            " x = %.6f; x1 is the offset from that window's centre\n"
            % XC)
    f.write("# %3s %9s %12s" % ("k", "y0", "x1")
            + "".join(" %13s" % c for c in COMP) + "\n")
    for k, Fk in rows:
        for i in range(n0):
            f.write("%5d %9.4f %12.6f" % (k, YCELL + k * P, x0[i])
                    + "".join(" %13.6e" % Fk[i, j] for j in range(6))
                    + "\n")
print("\nwrote %s" % os.path.basename(out))

for c, j in (("11", 0), ("22", 1), ("12", 3)):
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for k, Fk in rows:
        if k == 0:
            ax.plot(x0, Fk[:, j], "-", lw=2.2, color="#c0392b",
                    label="station window (k = 0)", zorder=5)
        else:
            ax.plot(x0, Fk[:, j], "-", lw=0.9,
                    color=plt.cm.viridis((k + 3) / 6.0),
                    label="k = %+d" % k, alpha=0.85)
    ax.set_xlabel(r"$x_1$  (mm, window-local)")
    ax.set_ylabel(r"$\sigma_{%s}$  (MPa)" % c)
    ax.grid(alpha=0.3)
    ax.axhline(0.0, color="k", lw=0.6, alpha=0.4)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    png = os.path.join(HERE, "cellscan_S%s.png" % c)
    fig.savefig(png, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote %s" % os.path.basename(png))
print("end   : %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
