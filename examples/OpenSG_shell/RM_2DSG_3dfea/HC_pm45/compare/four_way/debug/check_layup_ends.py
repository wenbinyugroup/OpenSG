"""check_layup_ends.py -- is a LAYUP/ANGLE mismatch between the [45/-45]s
given to the homogenization and what Abaqus actually used the cause of
the path-1 END deviations in the in-plane stresses?

No assumptions: the angle Abaqus REALLY applied per element is inferred
from its own two dumps.  The material dump is the global dump rotated
by that element's *Orientation angle, so for the in-plane deviatoric
pair (A, B) = ((s11-s22)/2, s12):

    2 theta = atan2(B_g, A_g) - atan2(B_m, A_m)

and independently, the transverse-shear pair (s13, s23) rotates by
theta itself.  Comparing the inferred theta with the yaml's map
(mat 2: angle -45 -> rotation theta = +45; mat 3: -45; core: 0) tests
the layup element by element -- including the path ENDS.

In:  ../../../abaqus/3dfea_linear/45pm_S_{global,material}_linear.csv,
     ../../../opensg/sg_2d/45pm_singleextrude_bd.yaml
Out: printed verdict (per-row summary + the path-1 row element list)
"""
import datetime
import os

import numpy as np
import yaml
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
HC = os.path.join(HERE, "..", "..", "..")
YCELL, HALF = 30.10244, 3.7628040

print("start : %s" % datetime.datetime.now().strftime("%H:%M:%S"))
G = np.genfromtxt(os.path.join(HC, "abaqus", "3dfea_linear",
                               "45pm_S_global_linear.csv"),
                  delimiter=",", names=True)
M = np.genfromtxt(os.path.join(HC, "abaqus", "3dfea_linear",
                               "45pm_S_material_linear.csv"),
                  delimiter=",", names=True)
gi = {int(e): j for j, e in enumerate(G["elem"])}
mi = {int(e): j for j, e in enumerate(M["elem"])}

d = yaml.safe_load(open(os.path.join(
    HC, "opensg", "sg_2d", "45pm_singleextrude_bd.yaml")))
nd = np.asarray(d["nodes"], float)[:, :2]
cells = np.asarray(d["cells"], int)
mat = np.asarray(d["mat_id"], int)
ang = {int(k): float(v.get("angle", 0.0) or 0.0)
       for k, v in d["materials"].items()}
cen = nd[cells].mean(axis=1)
# theta used by material_frame_fields = -angle (the flat_pm45 gate)
th_yaml = np.array([-ang[m_] for m_ in mat])

# one spanwise layer of the station cell (angle inference is
# per-element -- no extrapolation needed)
XC = 0.5 * (G["x"].min() + G["x"].max())
lay = np.unique(np.round(G["x"] - XC, 4))
d1 = np.sort(np.abs(lay[np.abs(lay) > 1e-6]))[0]
k = np.abs(np.round(G["x"] - XC, 4) - round(-d1, 4)) < 1e-4
k &= (G["y"] >= YCELL - HALF) & (G["y"] <= YCELL + HALF)
els = G["elem"][k].astype(int)
yz = np.column_stack([G["y"][k] - YCELL, G["z"][k]])
dist, idx = cKDTree(cen).query(yz)
assert dist.max() < 1e-4
print("layer     : %d elements paired to the SG (gate %.1e)"
      % (len(els), dist.max()))


def theta_of(e):
    g, m = gi[e], mi[e]
    Ag = 0.5 * (G["S11"][g] - G["S22"][g])
    Bg = G["S12"][g]
    Am = 0.5 * (M["S11"][m] - M["S22"][m])
    Bm = M["S12"][m]
    t_ip = 0.5 * np.degrees(np.arctan2(Bg, Ag) - np.arctan2(Bm, Am))
    t_sh = np.degrees(np.arctan2(G["S23"][g], G["S13"][g])
                      - np.arctan2(M["S23"][m], M["S13"][m]))
    for t in (t_ip, t_sh):
        pass
    wrap = lambda t: (t + 180.0) % 360.0 - 180.0        # noqa: E731
    mag_ip = np.hypot(Ag, Bg)
    mag_sh = np.hypot(G["S13"][g], G["S23"][g])
    return wrap(t_ip), wrap(t_sh), mag_ip, mag_sh


print("\n---- per-SG-row summary: inferred Abaqus angle vs the yaml map")
rows = np.unique(np.round(cen[idx][:, 1], 4))
for z0 in rows[np.abs(rows) > 4.3]:        # the two face sheets' rows
    kk = np.abs(cen[idx][:, 1] - z0) < 1e-3
    tt = np.array([theta_of(int(e)) for e in els[kk]])
    strong = tt[:, 2] > 0.02               # deviatoric big enough to trust
    ty = th_yaml[idx[kk]][0]
    if strong.sum() == 0:
        print("row z=%+.4f: (deviatoric too small to infer)" % z0)
        continue
    med = np.median(tt[strong, 0])
    print("row z=%+.4f  yaml theta %+5.1f  inferred median %+6.1f"
          " (n=%d)  worst dev %5.1f deg %s"
          % (z0, ty, med, int(strong.sum()),
             np.abs(tt[strong, 0] - ty).max(),
             "OK" if abs(med - ty) < 2 else "MISMATCH"))

print("\n---- path-1 row, element by element (ends included)")
z_top = rows[-1]
kk = np.where(np.abs(cen[idx][:, 1] - z_top) < 1e-3)[0]
order = np.argsort(cen[idx[kk]][:, 0])
print("%8s %6s %9s %9s %10s %10s" % ("x1", "yaml", "th_inpl",
                                     "th_shear", "|dev_ip|", "|dev_sh|"))
bad = 0
for j in kk[order]:
    e = int(els[j])
    t_ip, t_sh, m_ip, m_sh = theta_of(e)
    ty = th_yaml[idx[j]]
    d_ip = abs((t_ip - ty + 90) % 180 - 90)
    d_sh = abs((t_sh - ty + 180) % 360 - 180)
    flag = ""
    if m_ip > 0.02 and d_ip > 2.0:
        flag = "  <-- MISMATCH"
        bad += 1
    x1 = cen[idx[j]][0]
    if abs(x1) > 2.9 or flag:              # print the END zones + flags
        print("%8.3f %6.1f %9.2f %9.2f %10.2f %10.2f%s"
              % (x1, ty, t_ip, t_sh, d_ip, d_sh, flag))
print("\nverdict: %d of %d path-1 row elements disagree with the yaml"
      " angle map (in-plane inference, >2 deg, trustworthy magnitude)"
      % (bad, len(kk)))
print("end   : %s" % datetime.datetime.now().strftime("%H:%M:%S"))
