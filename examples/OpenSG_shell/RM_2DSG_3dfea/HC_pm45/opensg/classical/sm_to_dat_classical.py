"""sm_to_dat_classical.py -- sample the CLASSICAL dehom .SM (MATERIAL
frame) at BOTH path coordinate sets.

Same pairing and gates as ../path_1/sm_to_dat.py, pointed at this
folder's classical run: coords row -> SG element by centroid (exact,
gated), element 4-Gauss-point mean from the .SM, output row-aligned
with the .coords so the Abaqus and RM .dat files line up row-for-row.

Gates: .SM element-major ordering vs the yaml centroids (1e-6); the
.SM header must carry the `material` frame tag; coords -> centroid
match 1e-4.

In:  ./45pm_singleextrude_bd_classical_dehom.SM,
     ./45pm_singleextrude_bd_classical.yaml,
     ../path_1/path_1.coords, ../path_2/path_2.coords
Out: ./path_1_classical.dat, ./path_2_classical.dat --
     rows `s x1 x2 mat S11 S22 S33 S12 S13 S23`
"""
import datetime
import os

import numpy as np
import yaml
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
COMP = ["S11", "S22", "S33", "S12", "S13", "S23"]

print("start : %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
d = yaml.safe_load(open(os.path.join(
    HERE, "45pm_singleextrude_bd_classical.yaml")))
nd = np.asarray(d["nodes"], float)[:, :2]
cells = np.asarray(d["cells"], int)
E = len(cells)
cen = nd[cells].mean(axis=1)

smf = os.path.join(HERE, "45pm_singleextrude_bd_classical_dehom.SM")
hdr = open(smf).readline() + open(smf).readlines()[1]
if "material" not in hdr.lower():
    raise SystemExit("the .SM is not material-frame (header: %r...) --"
                     " rerun `opensg <yaml> D` (material is the"
                     " default)" % hdr[:120])
sm = np.loadtxt(smf)
ngp = len(sm) // E
g_xy = sm[:, 1:3].reshape(E, ngp, 2).mean(axis=1)
off = np.abs(g_xy - cen).max()
print("dehom .SM : %d Gauss points = %d per element; ordering gate"
      " %.2e %s" % (len(sm), ngp, off, "OK" if off < 1e-6 else "FAIL"))
if off >= 1e-6:
    raise SystemExit("the .SM is not in yaml element order")
S_el = sm[:, 3:9].reshape(E, ngp, 6).mean(axis=1)
tree = cKDTree(cen)

for n in (1, 2):
    cf = os.path.join(HERE, "..", "path_%d" % n, "path_%d.coords" % n)
    if not os.path.exists(cf):
        print("path_%d.coords missing -- skipped" % n)
        continue
    C = np.loadtxt(cf)
    dist, idx = tree.query(C[:, 1:3])
    print("path %d    : %d points, worst coords -> centroid distance"
          " %.2e (tol 1e-4) %s" % (n, len(C), dist.max(),
                                   "OK" if dist.max() < 1e-4 else
                                   "FAIL"))
    if dist.max() >= 1e-4:
        raise SystemExit("path %d coords do not sit on this yaml's"
                         " centroids" % n)
    out = os.path.join(HERE, "path_%d_classical.dat" % n)
    with open(out, "w") as f:
        f.write("# MSG classical-plate dehom stress along path_%d --"
                " MATERIAL frame (ply axes), per-element mean of %d"
                " Gauss points\n" % (n, ngp))
        f.write("# %10s %12s %12s %4s" % ("s", "x1", "x2", "mat"))
        for c in COMP:
            f.write(" %13s" % c)
        f.write("\n")
        for i in range(len(C)):
            f.write("%12.6f %12.6f %12.6f %4d"
                    % (C[i, 0], C[i, 1], C[i, 2], int(C[i, 3])))
            for j in range(6):
                f.write(" %13.6e" % S_el[idx[i], j])
            f.write("\n")
    print("wrote %s (%d rows)" % (os.path.basename(out), len(C)))
print("end   : %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
