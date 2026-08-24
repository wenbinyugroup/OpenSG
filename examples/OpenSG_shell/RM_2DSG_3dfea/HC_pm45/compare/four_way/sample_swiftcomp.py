"""sample_swiftcomp.py -- sample the SwiftComp CLASSICAL dehom at both
path coordinate sets, deciding BY GATE (not by assumption) which output
file carries per-element values and which frame it is in.

The SwiftComp run (../../opensg/classical/SwiftComp/, driven by the
yaml_to_sc .sc + the ff_to_glb .glb, same FF as every other curve)
wrote two 5488-row field files -- exactly 4 rows per element:

    .sc.sn   nodal values, problem (SG) frame        } which is which is
    .sc.sg   (undocumented in SCManual 2.1)          } MEASURED below

Gates, in order:
  1. the .sc.k effective stiffness must equal the OpenSG classical .out
     law (the homogenization cross-check, expected ~1e-7);
  2. per-row coordinates matched against the mesh: a file whose rows sit
     on the element GAUSS points pairs (element, gp) uniquely -- that
     file is used; a file on element corner NODES is identified and
     reported;
  3. the frame: element means, rotated (or not) into the ply material
     frame with OpenSG's own gated rotation (material_frame_fields),
     compared against the OpenSG classical dehom .SM -- same theory,
     same law, same FF, so ONE hypothesis must land within a few
     percent and the other far away.  The winner is printed and used.

In:  ../../opensg/classical/SwiftComp/45pm_singleextrude_bd_classical.sc.{k,sn,sg},
     ../../opensg/classical/45pm_singleextrude_bd_classical.yaml + _dehom.SM,
     ../../opensg/sg_2d/45pm_singleextrude_bd_classical.out,
     ../../opensg/path_N/path_N.coords
Out: ./path_1_swiftcomp.dat, ./path_2_swiftcomp.dat -- rows
     `s x1 x2 mat S11 S22 S33 S12 S13 S23` (MATERIAL frame, per-element
     means, row-aligned with the other .dat files)
"""
import datetime
import os
import re

import numpy as np
import yaml
from scipy.spatial import cKDTree

from opensg_solid.sg_dehom import (SGIDX, ORDER, gauss_coords,
                                   material_frame_fields)
from opensg_solid.sg_homo import plate_homo_2d

HERE = os.path.dirname(os.path.abspath(__file__))
OPG = os.path.join(HERE, "..", "..", "opensg")
SWC = os.path.join(OPG, "classical", "SwiftComp",
                   "45pm_singleextrude_bd_classical.sc")
YAML = os.path.join(OPG, "classical",
                    "45pm_singleextrude_bd_classical.yaml")
OSM = os.path.join(OPG, "classical",
                   "45pm_singleextrude_bd_classical_dehom.SM")
OUT = os.path.join(OPG, "sg_2d", "45pm_singleextrude_bd_classical.out")
COMP = ["S11", "S22", "S33", "S12", "S13", "S23"]
PIDX = [SGIDX[c] for c in ORDER]      # storage -> printed 11 22 33 12 13 23

print("start : %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# ---- gate 1: the homogenized law, SwiftComp vs OpenSG
def read_6x6(path, pat):
    rows, on = [], False
    for ln in open(path):
        if pat in ln:
            on = True
            continue
        v = ln.split()
        if on and len(v) == 6 and re.match(r"^-?\d", v[0]):
            rows.append([float(x) for x in v])
            if len(rows) == 6:
                break
    return np.array(rows)


K_sw = read_6x6(SWC + ".k", "Effective Stiffness Matrix")
K_og = read_6x6(OUT, "Classical Plate Stiffness Matrix")
rel = np.abs(K_sw - K_og).max() / np.abs(K_og).max()
print("gate 1: SwiftComp .sc.k vs OpenSG classical .out  rel %.2e %s"
      % (rel, "OK" if rel < 1e-5 else "FAIL"))
if rel >= 1e-5:
    raise SystemExit("the two homogenizations disagree -- wrong pair of"
                     " files?")

# ---- mesh objects (element order = yaml order everywhere)
r = plate_homo_2d(YAML, refined=0, plot=False)
d = yaml.safe_load(open(YAML))
nd = np.asarray(d["nodes"], float)[:, :2]
cells = np.asarray(d["cells"], int)
E = len(cells)
cen = nd[cells].mean(axis=1)
gxy = np.asarray(gauss_coords(r)).reshape(E * 4, 2)

# ---- gate 2: which file sits on the Gauss points?
def classify(path):
    a = np.loadtxt(path)
    if a.shape != (E * 4, 14):
        return a, None, np.inf
    dist, idx = cKDTree(gxy).query(a[:, :2])
    return a, idx, dist.max()


cand = {}
for ext in (".sg", ".sn"):
    a, idx, dmax = classify(SWC + ext)
    cand[ext] = (a, idx, dmax)
    print("gate 2: %s rows -> Gauss-point match  max %.2e" % (ext, dmax))
ext = min(cand, key=lambda k: cand[k][2])
a, idx, dmax = cand[ext]
if dmax >= 1e-5:
    raise SystemExit("neither file sits on the Gauss points -- the"
                     " layout assumption is wrong, inspect by hand")
print("        -> using %s (the Gauss-point file; the other is nodal)"
      % ext)
# scatter rows into (E, 4, 6): idx maps row -> flat (element*4 + gp)
Sig = np.zeros((E * 4, 6))
Gam = np.zeros((E * 4, 6))
Sig[idx] = a[:, 8:14]                # sig11 22 33 23 13 12 = storage order
Gam[idx] = a[:, 2:8]
Sig = Sig.reshape(E, 4, 6)
Gam = Gam.reshape(E, 4, 6)

# ---- gate 3: frame, decided against the OpenSG classical .SM
sm = np.loadtxt(OSM)
og_el = sm[:, 3:9].reshape(E, 4, 6).mean(axis=1)   # printed order, material
_, Sig_rot = material_frame_fields(Gam, Sig, r)
hyp = {"already-material": Sig.mean(axis=1)[:, PIDX],
       "global-rotated-to-material": np.asarray(Sig_rot).mean(axis=1)[:, PIDX]}
score = {}
for k, v in hyp.items():
    score[k] = (np.linalg.norm(v - og_el)
                / max(np.linalg.norm(og_el), 1e-30))
    print("gate 3: frame hypothesis %-28s rel dev vs OpenSG classical"
          " .SM  %.3f" % (k, score[k]))
frame = min(score, key=score.get)
if score[frame] > 0.15:
    raise SystemExit("no frame hypothesis matches the OpenSG classical"
                     " field -- something else differs, inspect")
print("        -> %s wins" % frame)
S_el = hyp[frame]                                   # (E, 6) printed order

# ---- sample both paths, row-aligned with the other .dat files
tree = cKDTree(cen)
for n in (1, 2):
    cf = os.path.join(OPG, "path_%d" % n, "path_%d.coords" % n)
    C = np.loadtxt(cf)
    dist, pidx = tree.query(C[:, 1:3])
    print("path %d    : %d points, coords -> centroid %.2e %s"
          % (n, len(C), dist.max(),
             "OK" if dist.max() < 1e-4 else "FAIL"))
    if dist.max() >= 1e-4:
        raise SystemExit("path %d coords do not sit on this mesh" % n)
    out = os.path.join(HERE, "path_%d_swiftcomp.dat" % n)
    with open(out, "w") as f:
        f.write("# SwiftComp classical dehom stress along path_%d --"
                " MATERIAL frame (ply axes), per-element mean of 4"
                " Gauss points (%s file, frame: %s)\n" % (n, ext, frame))
        f.write("# %10s %12s %12s %4s" % ("s", "x1", "x2", "mat"))
        for c in COMP:
            f.write(" %13s" % c)
        f.write("\n")
        for i in range(len(C)):
            f.write("%12.6f %12.6f %12.6f %4d"
                    % (C[i, 0], C[i, 1], C[i, 2], int(C[i, 3])))
            for j in range(6):
                f.write(" %13.6e" % S_el[pidx[i], j])
            f.write("\n")
    print("wrote %s (%d rows)" % (os.path.basename(out), len(C)))
print("end   : %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
