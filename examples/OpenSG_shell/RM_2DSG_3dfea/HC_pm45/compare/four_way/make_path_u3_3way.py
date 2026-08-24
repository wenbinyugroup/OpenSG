"""make_path_u3_3way.py -- the U3 (deflection) comparison along both
paths, in the four_way conventions: three curves, marked datapoints,
LINEAR 3-D reference.

Composition (the validated Eq.-65 rule of make_path_disp.py; U3 needs
no detilt): U3 = w_plate(macro, at the path point's width position) +
w3_fluct (element mean of the dehom .U).  The macro w is the SAME plate
solution for both recoveries -- classical differs from RM only in the
FLUCTUATION field (no pressure ladder, no gradient chains), so its U3
misses the through-thickness core-compression content.

In:  ../../opensg/path_N/path_N.coords,
     ../../opensg/sg_2d/45pm_singleextrude_bd.yaml,
     ../../opensg/dehomo/45pm_singleextrude_bd_dehom.U      (corrected RM)
     ../../opensg/classical/45pm_singleextrude_bd_classical_dehom.U,
     ../../opensg/dehomo/45pm_singleextrude_bd_plate_clamped{.inp,_U.rpt},
     ../../abaqus/3dfea_linear/45pm_U_linear.csv            (LINEAR)
Out: ./path_1_U3.png, ./path_2_U3.png, ./path_N_u3.dat, ./u3_rms.txt
"""
import datetime
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                            # noqa: E402
import numpy as np                                         # noqa: E402
import yaml                                                # noqa: E402
from scipy.interpolate import griddata                     # noqa: E402
from scipy.spatial import cKDTree                          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
HC = os.path.normpath(os.path.join(HERE, "..", ".."))
DEH = os.path.join(HC, "opensg", "dehomo")
PL = "45pm_singleextrude_bd_plate_clamped"
YCELL, XC = 30.10244, 37.628040
XLAB = {1: r"$x_1$  (mm)",
        2: r"$s/s_{\mathrm{total}}$  (0 = top surface, 1 = bottom"
           r" surface)"}

print("start : %s" % datetime.datetime.now().strftime("%H:%M:%S"))
d = yaml.safe_load(open(os.path.join(HC, "opensg", "sg_2d",
                                     "45pm_singleextrude_bd.yaml")))
nd = np.asarray(d["nodes"], float)[:, :2]
cells = np.asarray(d["cells"], int)
E = len(cells)
cen = nd[cells].mean(axis=1)


def w3_fluct(path):
    """element-mean w3 fluctuation of a dehom .U, order-gated"""
    Uw = np.loadtxt(path)
    ngp = len(Uw) // E
    off = np.abs(Uw[:, 1:3].reshape(E, ngp, 2).mean(axis=1) - cen).max()
    if off >= 1e-6:
        raise SystemExit("%s not in yaml element order" % path)
    return Uw[:, 5].reshape(E, ngp).mean(axis=1)


F_rm = w3_fluct(os.path.join(DEH, "45pm_singleextrude_bd_dehom.U"))
F_cl = w3_fluct(os.path.join(HC, "opensg", "classical",
                             "45pm_singleextrude_bd_classical_dehom.U"))
print("fluct w3 : RM ptp %.3e mm, classical ptp %.3e mm"
      % (np.ptp(F_rm), np.ptp(F_cl)))

# ---- plate macro w at the midspan node column
nodes = {}
on = None
for ln in open(os.path.join(DEH, PL + ".inp")):
    s = ln.strip()
    if s.startswith("**"):
        continue
    if s.startswith("*"):
        on = "n" if (s.split(",")[0].lower() == "*node"
                     and "output" not in s.lower()) else None
        continue
    if on == "n" and s:
        v = s.split(",")
        nodes[int(v[0])] = (float(v[1]), float(v[2]))
hdr, rows = None, []
for ln in open(os.path.join(DEH, PL + "_U.rpt")):
    s = ln.rstrip("\n")
    if not s.strip() or s.lstrip().startswith(("---", "@Loc", "Pt")):
        continue
    if hdr is None:
        if "Label" in s and re.search(r"\bU\.", s):
            hdr = s.replace("Node Label", "N").split()
        continue
    if re.match(r"^\s*-?\d", s):
        v = s.split()
        if len(v) == len(hdr):
            try:
                rows.append([float(x) for x in v])
            except ValueError:
                pass
A = np.array(rows)
iU3 = hdr.index("U.U3")
kcol = np.array([abs(nodes[int(r[0])][0] - XC) < 1e-4 for r in A])
ys = np.array([nodes[int(r[0])][1] for r in A[kcol]])
w0 = A[kcol, iU3]
o = np.argsort(ys)
ys, w0 = ys[o], w0[o]
print("plate    : midspan node column x = %.5f, %d nodes, w at cell"
      " centre %.4e mm" % (XC, len(ys), np.interp(YCELL, ys, w0)))

# ---- 3-D nodal U3 on the midspan plane, LINEAR run
T = np.genfromtxt(os.path.join(HC, "abaqus", "3dfea_linear",
                               "45pm_U_linear.csv"),
                  delimiter=",", names=True)
k = np.abs(T["x"] - XC) < 1e-4
P3 = np.column_stack([T["y"][k], T["z"][k]])
V3 = T["U3"][k]
print("3-D      : %d nodes on the LINEAR midspan plane, U3 %.4e .."
      " %.4e mm" % (int(k.sum()), V3.min(), V3.max()))
tree3 = cKDTree(P3)

STY = [
    ("Reference 3-D FEA (Abaqus C3D20R)",
     dict(ls="-", color="#c0392b", marker="o", ms=3.2, lw=1.7)),
    ("Shear-refined (RM) Plate (OpenSG)",
     dict(ls="--", color="k", marker="s", ms=2.8, lw=1.6)),
    ("Classical Plate (OpenSG)",
     dict(ls="-.", color="#2471a3", marker="s", ms=7.0, lw=1.4,
          markerfacecolor="none")),
]
lines = ["# normalised RMS of (model - 3-D) U3 along each path,"
         " LINEAR reference\n# %-8s %10s %14s\n"
         % ("path", "MSG-RM %", "classical %")]
for n in (1, 2):
    C = np.loadtxt(os.path.join(HC, "opensg", "path_%d" % n,
                                "path_%d.coords" % n))
    s, mt = C[:, 0], C[:, 3].astype(int)
    dist, idx = cKDTree(cen).query(C[:, 1:3])
    if dist.max() >= 1e-4:
        raise SystemExit("path %d coords pairing failed" % n)
    yz = np.column_stack([YCELL + C[:, 1], C[:, 2]])
    wmac = np.interp(yz[:, 0], ys, w0)
    u3 = {"rm": wmac + F_rm[idx], "cl": wmac + F_cl[idx]}
    a3 = griddata(P3, V3, yz, method="linear")
    bad = np.isnan(a3)
    if bad.any():
        _, j = tree3.query(yz[bad])
        a3[bad] = V3[j]
    u3["abq"] = a3

    dat = os.path.join(HERE, "path_%d_u3.dat" % n)
    with open(dat, "w") as f:
        f.write("# U3 along path_%d -- LINEAR 3-D nodal (midspan"
                " plane) vs plate-w + dehom w3 fluctuation, mm\n" % n)
        f.write("# %10s %12s %12s %4s %13s %13s %13s\n"
                % ("s", "x1", "x2", "mat", "U3_3d", "U3_rm", "U3_cl"))
        for i in range(len(C)):
            f.write("%12.6f %12.6f %12.6f %4d %13.6e %13.6e %13.6e\n"
                    % (s[i], C[i, 1], C[i, 2], mt[i], u3["abq"][i],
                       u3["rm"][i], u3["cl"][i]))
    print("wrote %s" % os.path.basename(dat))

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for (lab, st), key in zip(STY, ("abq", "rm", "cl")):
        ax.plot(s, u3[key], label=lab, **st)
    for i in np.where(np.diff(mt) != 0)[0]:
        ax.axvline(0.5 * (s[i] + s[i + 1]), color="k", lw=0.7, ls=":",
                   alpha=0.6)
    ax.set_xlabel(XLAB[n])
    ax.set_ylabel(r"$U_3$  (mm)")
    ax.grid(alpha=0.3)
    if n == 2:
        ax.set_xlim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    png = os.path.join(HERE, "path_%d_U3.png" % n)
    fig.savefig(png, dpi=170, bbox_inches="tight")
    plt.close(fig)
    den = max(np.abs(u3["abq"]).max(), 1e-30)
    r_rm = 100 * np.sqrt(np.mean((u3["rm"] - u3["abq"]) ** 2)) / den
    r_cl = 100 * np.sqrt(np.mean((u3["cl"] - u3["abq"]) ** 2)) / den
    lines.append("%-10s %10.2f %14.2f\n" % ("path_%d" % n, r_rm, r_cl))
    print("wrote path_%d_U3.png  RMS: RM %.2f %%  classical %.2f %%"
          % (n, r_rm, r_cl))
open(os.path.join(HERE, "u3_rms.txt"), "w").writelines(lines)
print("end   : %s" % datetime.datetime.now().strftime("%H:%M:%S"))
