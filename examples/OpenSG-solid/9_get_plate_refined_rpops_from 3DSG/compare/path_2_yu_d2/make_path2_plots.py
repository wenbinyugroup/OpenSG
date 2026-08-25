"""make_path2_plots.py -- the path-2 figures of the PURE YU-2003
composition run (uniform reaction + Eq. 64-66 d2eps chains), saved
HERE: Reference 3-D FEA against the shear-refined plate, exact element
pairing, s = y normalized 0..1 across the contact patch.

    python make_path2_plots.py

In:  ../AR5_yu_d2/path2_{abaqus,rm}.dat (sample_paths_paired_yu_d2.py)
Out: ./path2_<comp>.png x 6 + ./path2_all.png + ./rms.txt
"""
import datetime
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = "AR5_yu_d2"
COMP = ["S11", "S22", "S33", "S12", "S13", "S23"]
CURVES = [("abaqus", "Reference 3-D FEA (Abaqus C3D4)", dict(
    ls="-", marker="o", ms=3.0, color="#c0392b", lw=1.1,
    markeredgewidth=0)),
    ("rm", "Shear-refined (RM) Plate (OpenSG)", dict(
        ls="--", marker="s", ms=2.8, color="black", lw=1.0,
        markeredgewidth=0))]

print("start : %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
data = {}
for tag, _, _ in CURVES:
    data[tag] = np.loadtxt(os.path.join(HERE, "..", CASE,
                                        "path2_%s.dat" % tag))


def binned(d, w=0.025):
    edges = np.arange(0.0, 1.0 + w / 2, w)
    k = np.digitize(d[:, 0], edges) - 1
    out = []
    for b in range(len(edges) - 1):
        m = k == b
        if m.any():
            out.append([d[m, 0].mean()] + list(d[m][:, 5:11].mean(0)))
    return np.array(out)


B = {tag: binned(d) for tag, d in data.items()}
lines = []
ref, rm = B["abaqus"], B["rm"]
for j, c in enumerate(COMP):
    v = np.interp(ref[:, 0], rm[:, 0], rm[:, 1 + j])
    e = np.sqrt(np.mean((ref[:, 1 + j] - v) ** 2))
    s = np.sqrt(np.mean(ref[:, 1 + j] ** 2))
    lines.append("%s  rms %.4e  ref %.4e  rel %5.1f%%"
                 % (c, e, s, 100 * e / max(1e-30, s)))
open(os.path.join(HERE, "rms.txt"), "w").write("\n".join(lines) + "\n")

fig6, ax6 = plt.subplots(2, 3, figsize=(12.5, 6.2), sharex=True)
for j, c in enumerate(COMP):
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    for a in (ax, ax6[j // 3, j % 3]):
        for tag, lbl, st in CURVES:
            b = B[tag]
            a.plot(b[:, 0], b[:, 1 + j],
                   label=lbl if (a is ax or j == 0) else None, **st)
        a.set_ylabel("%s [MPa]" % c)
        a.grid(alpha=0.25, lw=0.4)
    ax.set_xlabel("normalized y (0 to 1 across the contact patch)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "path2_%s.png" % c), dpi=200)
    plt.close(fig)
for j in range(3, 6):
    ax6[1, j - 3].set_xlabel("normalized y (0 to 1)")
fig6.legend(loc="upper center", ncol=2, frameon=False,
            bbox_to_anchor=(0.5, 1.00))
fig6.tight_layout(rect=(0, 0, 1, 0.95))
fig6.savefig(os.path.join(HERE, "path2_all.png"), dpi=200)
print("wrote 6 individual + 1 grid figure -> %s" % HERE)
print("end   : %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
