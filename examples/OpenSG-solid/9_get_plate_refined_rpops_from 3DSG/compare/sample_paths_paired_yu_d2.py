"""sample_paths_paired_yu_d2.py -- the exact-pairing path dats of the
PURE YU-2003 composition run (opensg/AR5_yu_d2: uniform reaction +
Eq. 64-66 d2eps chains), the separate-folder companion of
sample_paths_paired.py.  Same paths, same pairing doctrine (one C3D4
IP vs the same tet's Gauss mean); the classical curve is
mode-independent (FF-only) and is read from opensg/AR5.

Run me, then compare/path_2_yu_d2/make_path2_plots.py.

In:  ../preovios_try.yaml, opensg/AR5_yu_d2/preovios_try_dehom.SM,
     opensg/AR5/preovios_try_classical_dehom.SM, abaqus/AR5 csvs,
     ./AR5_path{3,4}_*.coordinate
Out: ./AR5_yu_d2/path{1,2,3,4}_{rm,classical,abaqus}.dat
"""
import datetime
import os
import re

import numpy as np
from scipy.spatial import cKDTree

from opensg_solid.helper import (elem_mean_sm, load_3d_csv, pair_cell,
                                 ply_region, sg_centroids,
                                 write_path_dat)
from opensg_solid.sg_mesh import load_sg_input

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
CASE = "AR5"
NC = {"AR5": 5, "AR10": 10}[CASE]
RIN = 0.08
PLY0 = (0.50, 0.52)
CC = NC // 2

print("start : %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
OD = os.path.join(HERE, CASE + "_yu_d2")
if not os.path.isdir(OD):
    os.makedirs(OD)

cen, _, _ = sg_centroids(os.path.join(ROOT, "preovios_try.yaml"))

S_rm = elem_mean_sm(os.path.join(ROOT, "opensg", CASE + "_yu_d2",
                                 "preovios_try_dehom.SM"), cen)
S_cl = elem_mean_sm(os.path.join(ROOT, "opensg", CASE,
                                 "preovios_try_classical_dehom.SM"),
                    cen)
band, tre_band = load_3d_csv(os.path.join(
    ROOT, "abaqus", CASE, "%s_3D_S_material.csv" % CASE))
iface, tre_ifac = load_3d_csv(os.path.join(
    ROOT, "abaqus", CASE, "%s_3D_S_material_iface.csv" % CASE))


def emit(n, elems, svals, tree, data, ix=CC, iy=CC):
    j3, ok = pair_cell(cen, elems, ix, iy, tree)
    if not ok.all():
        print("path %d: %d/%d elements failed the centroid gate"
              % (n, int((~ok).sum()), len(ok)))
    el, sv, j3 = elems[ok], svals[ok], j3[ok]
    o = np.argsort(sv)
    el, sv, j3 = el[o], sv[o], j3[o]
    x3 = cen[el] + np.array([ix + 0.5, iy + 0.5, 0.0])
    regs = [ply_region(z) for z in cen[el][:, 2]]
    write_path_dat(os.path.join(OD, "path%d_abaqus.dat" % n),
                   [[sv[k], x3[k, 0], x3[k, 1], x3[k, 2], regs[k]]
                    + list(data[j3[k], 4:10]) for k in range(len(el))],
                   "3-D FEA along path %d" % n)
    for tag, S in (("rm", S_rm), ("classical", S_cl)):
        write_path_dat(os.path.join(OD, "path%d_%s.dat" % (n, tag)),
                       [[sv[k], x3[k, 0], x3[k, 1], x3[k, 2], regs[k]]
                        + list(S[el[k]]) for k in range(len(el))],
                       "%s dehom along path %d" % (tag, n))


m1 = np.hypot(cen[:, 0], cen[:, 1]) <= RIN
e1 = np.nonzero(m1)[0]
emit(1, e1, cen[e1][:, 2], tre_band, band)

for n, nm in ((3, "pipe"), (4, "corevert")):
    pf = os.path.join(HERE, "%s_path%d_%s.coordinate" % (CASE, n, nm))
    L = np.loadtxt(pf)
    P = L[:, 1:4].copy()
    P[:, 0] -= NC / 2.0
    P[:, 1] -= NC / 2.0
    sarc = np.r_[0.0, np.cumsum(np.hypot(np.hypot(
        *np.diff(P[:, :2], axis=0).T), np.diff(P[:, 2])))]
    dist, jn = cKDTree(P).query(cen)
    mp = dist <= 0.05
    ep = np.nonzero(mp)[0]
    emit(n, ep, sarc[jn[mp]], tre_band, band)

# PATH 2: the horseshoe contact-patch MID-LINE (one side), s = y
# normalized 0..1 -- identical construction to sample_paths_paired.py
_mats = np.asarray(load_sg_input(os.path.join(
    ROOT, "preovios_try.yaml"))["mat_id"], int)
ct = (_mats == 1) & (cen[:, 2] > 0.46) & (cen[:, 2] <= 0.50)
ct_xy = cen[ct][:, :2]
patch = ct_xy[(ct_xy[:, 0] > 0.15) & (np.abs(ct_xy[:, 1]) <= 0.12)]
y0, y1 = float(patch[:, 1].min()), float(patch[:, 1].max())
nb = 24
ybins = np.linspace(y0, y1, nb + 1)
mid = []
for b in range(nb):
    m = (patch[:, 1] >= ybins[b]) & (patch[:, 1] < ybins[b + 1])
    if m.any():
        mid.append((float(patch[m, 0].mean()),
                    0.5 * (ybins[b] + ybins[b + 1])))
mid = np.array(mid)
tre_ct = cKDTree(patch)
xmid = lambda y: np.interp(y, mid[:, 1], mid[:, 0])   # noqa: E731
m2 = ((cen[:, 2] > PLY0[0]) & (cen[:, 2] <= PLY0[1]) &
      (cen[:, 1] >= y0) & (cen[:, 1] <= y1))
e2 = np.nonzero(m2)[0]
keep = (np.abs(cen[e2][:, 0] - xmid(cen[e2][:, 1])) <= 0.035) & \
    (tre_ct.query(cen[e2][:, :2])[0] <= 0.03)
e2 = e2[keep]
yv = cen[e2][:, 1]
print("path 2: horseshoe-patch MID-LINE (x %.3f..%.3f along the U),"
      " y %.3f..%.3f, %d ply elements (s = y normalized 0..1)"
      % (mid[:, 0].min(), mid[:, 0].max(), y0, y1, len(e2)))
emit(2, e2, (yv - yv.min()) / (yv.max() - yv.min()),
     tre_band, band)
print("end   : %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
