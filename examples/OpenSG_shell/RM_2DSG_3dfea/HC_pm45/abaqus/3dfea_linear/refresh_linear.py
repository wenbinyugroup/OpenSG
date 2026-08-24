"""refresh_linear.py -- AFTER the linear Abaqus run: extract the
path_1/path_2 references from 45pm_S_material_linear.csv (the same
Richardson-onto-midspan + element pairing as csv_to_dat.py) and print
what the nlgeom=NO rerun changed against the nonlinear reference.

Writes ../station_path_stress/path_N_abaqus_linear.dat -- the
nonlinear path_N_abaqus.dat is NOT touched; promote by hand once the
numbers are reviewed.
"""
import datetime
import os

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
STP = os.path.join(HERE, "..", "station_path_stress")
COMP = ["S11", "S22", "S33", "S12", "S13", "S23"]
YCELL, HALF = 30.10244, 3.762805

print("start : %s" % datetime.datetime.now().strftime("%H:%M:%S"))
S = np.genfromtxt(os.path.join(HERE, "45pm_S_material_linear.csv"),
                  delimiter=",", names=True)
XC = 0.5 * (S["x"].min() + S["x"].max())
lay = np.unique(np.round(S["x"] - XC, 4))
d1 = np.sort(np.abs(lay[np.abs(lay) > 1e-6]))[0]
print("linear dump: %d rows; midspan x = %.6f" % (len(S), XC))


def layer(o):
    k = np.abs(np.round(S["x"] - XC, 4) - round(o, 4)) < 1e-4
    k &= (S["y"] >= YCELL - HALF) & (S["y"] <= YCELL + HALF)
    r = S[k]
    return r[np.lexsort((np.round(r["z"], 5), np.round(r["y"], 5)))]


avg = {}
for tag, dd in (("in", d1), ("out", 3.0 * d1)):
    m, p = layer(-dd), layer(+dd)
    if len(m) != len(p):
        raise SystemExit("layer pair %s does not align" % tag)
    avg[tag] = {c: 0.5 * (m[c] + p[c]) for c in COMP}
    avg[tag]["y"], avg[tag]["z"] = m["y"], m["z"]
F = np.stack([(9.0 * avg["in"][c] - avg["out"][c]) / 8.0
              for c in COMP], axis=-1)
tree = cKDTree(np.column_stack([avg["in"]["y"] - YCELL,
                                avg["in"]["z"]]))

for n in (1, 2):
    cf = os.path.join(HERE, "..", "..", "opensg", "path_%d" % n,
                      "path_%d.coords" % n)
    C = np.loadtxt(cf)
    dist, idx = tree.query(C[:, 1:3])
    print("path %d: pairing %.2e %s" % (n, dist.max(),
                                        "OK" if dist.max() < 1e-4
                                        else "FAIL"))
    if dist.max() >= 1e-4:
        raise SystemExit("pairing failed")
    out = os.path.join(STP, "path_%d_abaqus_linear.dat" % n)
    with open(out, "w") as f:
        f.write("# Abaqus 3-D FEA stress along path_%d -- MATERIAL"
                " frame, LINEAR (nlgeom=NO) rerun, Richardson onto"
                " midspan x = %.6f\n" % (n, XC))
        f.write("# %10s %12s %12s %4s" % ("s", "x1", "x2", "mat"))
        for c in COMP:
            f.write(" %13s" % c)
        f.write("\n")
        for i in range(len(C)):
            f.write("%12.6f %12.6f %12.6f %4d"
                    % (C[i, 0], C[i, 1], C[i, 2], int(C[i, 3])))
            for j in range(6):
                f.write(" %13.6e" % F[idx[i], j])
            f.write("\n")
    print("wrote %s" % os.path.basename(out))
    old = np.loadtxt(os.path.join(STP, "path_%d_abaqus.dat" % n))
    new = np.loadtxt(out)
    print("  linear vs NONLINEAR reference (RMS %% of nonlinear max):")
    for j, c in enumerate(COMP):
        den = max(np.abs(old[:, 4 + j]).max(), 1e-30)
        print("    %-4s %6.2f %%" % (c, 100 * np.sqrt(np.mean(
            (new[:, 4 + j] - old[:, 4 + j]) ** 2)) / den))
print("end   : %s" % datetime.datetime.now().strftime("%H:%M:%S"))
