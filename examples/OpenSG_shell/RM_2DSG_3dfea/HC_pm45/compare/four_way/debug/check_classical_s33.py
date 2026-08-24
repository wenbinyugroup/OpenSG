"""check_classical_s33.py -- is the classical model's sigma33 INVERTED
by a code bug, or is it the honest (equilibrium-free) strain part?

Three independent checks:

  A. QUANTIFY the inversion on the benchmark: correlation of classical
     vs 3-D sigma33 along both paths, and classical-vs-RM.  (RM shares
     the SAME base kernel -- classical IS that base with the pressure
     and gradient chains skipped -- so a base sigma33 sign bug would
     poison RM too, and RM sits at/below the reference floor.)
  B. KERNEL SIGN TEST on a homogeneous isotropic plate SG, where the
     answers are exact: classical recovery under pure kappa11 must give
     sigma33 ~ 0 (plane-stress relief; a z-coupling sign bug would show
     sigma33 ~ nu sigma11 ~ 0.3 sigma11), and the RM pressure chain
     under qt6 = [1,0..] must give sigma33 = -q on the top face, 0 on
     the bottom, monotone between.
  C. PARITY of the classical sigma33 field on the honeycomb: pair each
     element with its top-bottom mirror image; a strain-driven sigma33
     under bending is ODD in the thickness (sigma33(x1,-x2) ~
     -sigma33(x1,x2)) -- which LOOKS inverted in the upper half against
     the true pressure field (-q at top decaying to 0).

In:  the four_way .dat files, ../../opensg/sg_2d/45pm_singleextrude_bd.yaml
Out: printed report (no files)
"""
import datetime
import os

import numpy as np
import yaml
from scipy.spatial import cKDTree

from opensg_solid.sg_dehom import SGIDX, dehom_fields, material_frame_fields
from opensg_solid.sg_homo import plate_homo_2d

HERE = os.path.dirname(os.path.abspath(__file__))
HC = os.path.join(HERE, "..", "..", "..")
J33 = 2                       # printed order 11 22 33 12 13 23 -> S33

print("start : %s" % datetime.datetime.now().strftime("%H:%M:%S"))
print("==== A. how inverted is it, exactly ====")
for n in (1, 2):
    ab = np.loadtxt(os.path.join(HC, "abaqus", "station_path_stress",
                                 "path_%d_abaqus.dat" % n))[:, 4 + J33]
    cl = np.loadtxt(os.path.join(HC, "opensg", "classical",
                                 "path_%d_classical.dat" % n))[:, 4 + J33]
    rm = np.loadtxt(os.path.join(HC, "opensg", "path_%d" % n,
                                 "path_%d_opensg.dat" % n))[:, 4 + J33]
    print("path %d: corr(classical, 3-D) = %+.3f   corr(RM, 3-D) ="
          " %+.3f" % (n, np.corrcoef(cl, ab)[0, 1],
                      np.corrcoef(rm, ab)[0, 1]))
    print("         mean sigma33: 3-D %+.4f  classical %+.4f  RM %+.4f"
          % (ab.mean(), cl.mean(), rm.mean()))

print("\n==== B. kernel sign test: homogeneous isotropic plate SG ====")
W, H, NX, NZ = 1.0, 1.0, 8, 16
xs = np.linspace(-W / 2, W / 2, NX + 1)
zs = np.linspace(-H / 2, H / 2, NZ + 1)
nid = {}
nodes = []
for i, x in enumerate(xs):
    for k, z in enumerate(zs):
        nid[(i, k)] = len(nodes)
        nodes.append([float(x), float(z), 0.0])
cells = []
for i in range(NX):
    for k in range(NZ):
        cells.append([nid[(i, k)], nid[(i + 1, k)],
                      nid[(i + 1, k + 1)], nid[(i, k + 1)]])
iso = os.path.join(HERE, "_iso_plate_sg.yaml")
with open(iso, "w") as f:
    f.write("n_model: 2\nrefined: 1\nmsg: solid\n")
    f.write("nodes:\n")
    for p in nodes:
        f.write("- [%.9f, %.9f, 0.0]\n" % (p[0], p[1]))
    f.write("cells:\n")
    for c in cells:
        f.write("- [%d, %d, %d, %d]\n" % tuple(c))
    f.write("mat_id: [%s]\n" % ", ".join(["1"] * len(cells)))
    f.write("materials:\n  1:\n    type: 0\n    E: 70000.0\n"
            "    nu: 0.3\n    density: 0.0\n")
r = plate_homo_2d(iso, refined=1, plot=False)
gz = None
# pure bending kappa11: classical recovery -> sigma33 must vanish
G_, S_, _ = dehom_fields(r, np.array([0.0, 0, 0, 1e-3, 0, 0]))
S_ = np.asarray(S_)
s11 = np.abs(S_[..., 0]).max()
s33 = np.abs(S_[..., 2]).max()
print("pure kappa11 : max|sigma33| / max|sigma11| = %.2e"
      " (bug would be ~0.3)  %s" % (s33 / s11,
                                    "OK" if s33 / s11 < 1e-6 else
                                    "SUSPICIOUS"))
# pure membrane e11
G_, S_, _ = dehom_fields(r, np.array([1e-3, 0, 0, 0, 0, 0]))
S_ = np.asarray(S_)
print("pure e11     : max|sigma33| / max|sigma11| = %.2e  %s"
      % (np.abs(S_[..., 2]).max() / np.abs(S_[..., 0]).max(),
         "OK" if np.abs(S_[..., 2]).max()
         / np.abs(S_[..., 0]).max() < 1e-6 else "SUSPICIOUS"))
# the RM pressure chain: sigma33 must run -q (top face) -> 0 (bottom)
G_, S_, _ = dehom_fields(r, np.zeros(6), qt6=[1.0, 0, 0, 0, 0, 0])
S_ = np.asarray(S_)
from opensg_solid.sg_dehom import gauss_coords
gz = np.asarray(gauss_coords(r)).reshape(-1, 2)[:, 1]
s33f = S_.reshape(-1, 6)[:, 2]
top = gz > H / 2 - H / NZ / 2
bot = gz < -H / 2 + H / NZ / 2
print("qt6 = 1      : sigma33 top row %+.4f (target ~ -1), bottom row"
      " %+.4f (target ~ 0), monotone %s"
      % (s33f[top].mean(), s33f[bot].mean(),
         "yes" if np.all(np.diff(np.array(
             [s33f[np.abs(gz - z) < 1e-9].mean()
              for z in np.unique(np.round(gz, 9))])) < 1e-6) else
         "check"))

print("\n==== C. parity of the classical sigma33 on the honeycomb ====")
YAML = os.path.join(HC, "opensg", "sg_2d", "45pm_singleextrude_bd.yaml")
d = yaml.safe_load(open(YAML))
nd = np.asarray(d["nodes"], float)[:, :2]
cl_ = np.asarray(d["cells"], int)
cen = nd[cl_].mean(axis=1)
r2 = plate_homo_2d(YAML, refined=1, plot=False)
ff = os.path.join(HC, "opensg", "dehomo", "45pm_singleextrude_bd.ff")
FF = [float(v) for v in open(ff).read().split("FF: [")[1]
      .split("]")[0].split(",")]
eps = np.linalg.solve(np.asarray(r2["C_eff"], float), FF)
G_, S_, _ = dehom_fields(r2, eps)
_, Sm = material_frame_fields(G_, S_, r2)
s33e = np.asarray(Sm).mean(axis=1)[:, SGIDX["33"]]
mirror = np.column_stack([cen[:, 0], -cen[:, 1]])
dist, idx = cKDTree(cen).query(mirror)
ok = dist < 1e-3
print("mirror pairing: %d of %d elements have a top-bottom image"
      % (int(ok.sum()), len(cen)))
c = np.corrcoef(s33e[ok], -s33e[idx[ok]])[0, 1]
print("corr( sigma33(x1, x2), -sigma33(x1, -x2) ) = %+.3f"
      "  (1.0 = perfectly ODD in the thickness)" % c)
print("end   : %s" % datetime.datetime.now().strftime("%H:%M:%S"))
