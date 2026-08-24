"""check_3d_moment.py -- does the 3-D panel actually CARRY the plate's
station resultants at midspan?  Integrates the GLOBAL-frame 3-D stress
over the station cell cross-section:

    N11 = sum sig11 A_e / W      M11 = sum sig11 z A_e / W
    N22 = sum sig22 A_e / W      M22 = sum sig22 z A_e / W

(Richardson onto midspan, same stencil as the path extraction) and
compares them with the .ff drivers.  A mismatch here explains a MEAN
offset of every recovered in-plane component -- a macro (plate vs 3-D
boundary modelling) difference, not a dehom defect.

In:  ../../abaqus/3dfea_output/45pm_S_global.csv,
     ../sg_2d/45pm_singleextrude_bd.yaml, ../dehomo/45pm_singleextrude_bd.ff
Out: the printed comparison
"""
import datetime
import os

import numpy as np
import yaml
from scipy.spatial import cKDTree

from opensg_solid.cli import read_ff_state

HERE = os.path.dirname(os.path.abspath(__file__))
YCELL = 30.10244

print("start : %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
S = np.genfromtxt(os.path.join(HERE, "..", "..", "abaqus",
                               "3dfea_output", "45pm_S_global.csv"),
                  delimiter=",", names=True)
XC = 0.5 * (S["x"].min() + S["x"].max())
lay = np.unique(np.round(S["x"] - XC, 4))
d1 = np.sort(np.abs(lay[np.abs(lay) > 1e-6]))[0]

d = yaml.safe_load(open(os.path.join(
    HERE, "..", "sg_2d", "45pm_singleextrude_bd.yaml")))
nd = np.asarray(d["nodes"], float)[:, :2]
cl = np.asarray(d["cells"], int)
mat = np.asarray(d["mat_id"], int)
cen = nd[cl].mean(axis=1)
x, y = nd[cl][:, :, 0], nd[cl][:, :, 1]
area = 0.5 * np.abs(
    (x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y).sum(1))
W = nd[:, 0].max() - nd[:, 0].min()
print("SG mesh   : %d elements, width W = %.5f, material area %.4f"
      " (walls %.4f, faces %.4f)"
      % (len(cl), W, area.sum(), area[mat == 1].sum(),
         area[mat != 1].sum()))
print("uniform-reaction bay-load fraction 1 - A_topface/A_mat = %.3f"
      % (1.0 - (W * 0.5) / area.sum()))


def layer(o):
    k = np.abs(np.round(S["x"] - XC, 4) - round(o, 4)) < 1e-4
    k &= (S["y"] >= YCELL - W / 2) & (S["y"] <= YCELL + W / 2)
    r = S[k]
    return r[np.lexsort((np.round(r["z"], 5), np.round(r["y"], 5)))]


COMP = ["S11", "S22", "S33", "S12", "S13", "S23"]
avg = {}
for tag, dd in (("in", d1), ("out", 3.0 * d1)):
    m, p = layer(-dd), layer(+dd)
    if len(m) != len(p) or not (np.allclose(m["y"], p["y"], atol=1e-5)
                                and np.allclose(m["z"], p["z"],
                                                atol=1e-5)):
        raise SystemExit("layer pair %s does not align" % tag)
    avg[tag] = {c: 0.5 * (m[c] + p[c]) for c in COMP}
    avg[tag]["y"], avg[tag]["z"] = m["y"], m["z"]
F = np.stack([(9.0 * avg["in"][c] - avg["out"][c]) / 8.0
              for c in COMP], axis=-1)
f_xy = np.column_stack([avg["in"]["y"] - YCELL, avg["in"]["z"]])
dist, idx = cKDTree(f_xy).query(cen)
print("pairing   : %d cell elements, worst SG-centroid -> 3-D match"
      " %.2e %s" % (len(cen), dist.max(),
                    "OK" if dist.max() < 1e-4 else "FAIL"))
if dist.max() >= 1e-4:
    raise SystemExit("the 3-D midspan elements do not match the SG")

z = cen[:, 1]
sig = F[idx]
res = {
    "N11": (sig[:, 0] * area).sum() / W,
    "N22": (sig[:, 1] * area).sum() / W,
    "N12": (sig[:, 3] * area).sum() / W,
    "M11": (sig[:, 0] * z * area).sum() / W,
    "M22": (sig[:, 1] * z * area).sum() / W,
    "M12": (sig[:, 3] * z * area).sum() / W,
}
state = read_ff_state(os.path.join(HERE, "..", "dehomo",
                                   "45pm_singleextrude_bd.ff"))
FFp = dict(zip(["N11", "N22", "N12", "M11", "M22", "M12"],
               state["FF"]))
print("\n---- station resultants: 3-D integral vs the plate .ff")
print("%-4s %14s %14s %10s" % ("", "3-D integral", "plate .ff",
                               "ratio"))
for k in res:
    rat = res[k] / FFp[k] if abs(FFp[k]) > 1e-12 else np.nan
    print("%-4s %14.6e %14.6e %10.4f" % (k, res[k], FFp[k], rat))
print("\n(q L^2 / 24 = %.4f for the clamped-clamped strip check)"
      % (0.1 * 75.256081 ** 2 / 24.0))
print("end   : %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
