"""reference_floor.py -- the honest lower bound on ANY periodic-SG
model error: the 3-D reference's own cell-to-cell variability.

The panel is 8 identical cells across the width; a periodic SG predicts
ONE profile for all of them.  So the RMS difference between the station
cell's profile and its neighbours' (same path, offset +-1 cell pitch)
is the floor below which "model vs 3-D" error is meaningless -- the
reference itself does not agree with the reference that well.  Both
paths, all six components, LINEAR reference; floor = max over the two
neighbours, same normalization as every RMS table (path max |station|).

In:  ../../../abaqus/3dfea_linear/45pm_S_material_linear.csv,
     ../../../opensg/path_N/{path_N.coords, path_N_opensg.dat},
     ../../../abaqus/station_path_stress/path_N_abaqus.dat
Out: ./reference_floor.txt, the printed table
"""
import datetime
import os

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
HC = os.path.join(HERE, "..", "..", "..")
COMP = ["S11", "S22", "S33", "S12", "S13", "S23"]
YCELL, P = 30.10244, 7.5256081
HALF = P / 2.0

print("start : %s" % datetime.datetime.now().strftime("%H:%M:%S"))
S = np.genfromtxt(os.path.join(HC, "abaqus", "3dfea_linear",
                               "45pm_S_material_linear.csv"),
                  delimiter=",", names=True)
XC = 0.5 * (S["x"].min() + S["x"].max())
lay = np.unique(np.round(S["x"] - XC, 4))
d1 = np.sort(np.abs(lay[np.abs(lay) > 1e-6]))[0]


def layer(o):
    k = np.abs(np.round(S["x"] - XC, 4) - round(o, 4)) < 1e-4
    r = S[k]
    return r[np.lexsort((np.round(r["z"], 5), np.round(r["y"], 5)))]


avg = {}
for tag, dd in (("in", d1), ("out", 3.0 * d1)):
    m, p = layer(-dd), layer(+dd)
    assert len(m) == len(p)
    avg[tag] = {c: 0.5 * (m[c] + p[c]) for c in COMP}
    avg[tag]["y"], avg[tag]["z"] = m["y"], m["z"]
F = np.stack([(9.0 * avg["in"][c] - avg["out"][c]) / 8.0
              for c in COMP], axis=-1)
tree = cKDTree(np.column_stack([avg["in"]["y"], avg["in"]["z"]]))
print("3-D midspan plane: %d elements across the full width"
      % len(avg["in"]["y"]))

lines = ["# the reference's own cell-to-cell RMS %% (of path max"
         " |station|), LINEAR run --\n# the floor below which any"
         " periodic-SG-vs-3-D error is meaningless\n"
         "# %-8s %-4s %9s %9s %9s %12s\n"
         % ("path", "comp", "floor+1", "floor-1", "floorMAX",
            "RM err now")]
print(lines[-1].strip("\n"))
for n in (1, 2):
    C = np.loadtxt(os.path.join(HC, "opensg", "path_%d" % n,
                                "path_%d.coords" % n))
    rm = np.loadtxt(os.path.join(HC, "opensg", "path_%d" % n,
                                 "path_%d_opensg.dat" % n))[:, 4:10]
    ab = np.loadtxt(os.path.join(HC, "abaqus", "station_path_stress",
                                 "path_%d_abaqus.dat" % n))[:, 4:10]
    prof = {}
    for k in (-1, 0, 1):
        pts = np.column_stack([C[:, 1] + YCELL + k * P, C[:, 2]])
        dist, idx = tree.query(pts)
        if dist.max() >= 5e-3:
            print("path %d k=%+d: pairing %.1e -- mesh not periodic"
                  " here, skipped" % (n, k, dist.max()))
            continue
        prof[k] = F[idx]
    for j, c in enumerate(COMP):
        den = max(np.abs(prof[0][:, j]).max(), 1e-30)
        fl = {k: 100 * np.sqrt(np.mean(
            (prof[k][:, j] - prof[0][:, j]) ** 2)) / den
            for k in prof if k != 0}
        err = 100 * np.sqrt(np.mean((rm[:, j] - ab[:, j]) ** 2)) / den
        row = ("%-10s %-4s %9.2f %9.2f %9.2f %12.2f   %s"
               % ("path_%d" % n, c, fl.get(1, np.nan),
                  fl.get(-1, np.nan), max(fl.values()), err,
                  "AT FLOOR" if err <= 1.2 * max(fl.values())
                  else "above floor"))
        print(row)
        lines.append(row + "\n")
open(os.path.join(HERE, "reference_floor.txt"), "w").writelines(lines)
print("wrote reference_floor.txt")
print("end   : %s" % datetime.datetime.now().strftime("%H:%M:%S"))
