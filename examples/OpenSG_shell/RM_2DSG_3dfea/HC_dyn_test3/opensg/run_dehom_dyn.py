"""run_dehom_dyn.py -- dehomogenize EVERY time frame of ff_dyn.dat and
write the stress history at the TOP-CENTRE point of the midspan
recovery station.

    cd <this folder>
    PYTHONPATH=$HOME/OpenSG-2.0/src python -s run_dehom_dyn.py

Dehomogenization is LINEAR in its 37 drivers (6 eps + 6 dE1 + 6 dE2 +
18 d2eps + the constant qt6), so instead of 251 dehom calls this runs
ONE homogenization + 37 UNIT dehoms and contracts:

    sigma(t) = sum_k  w_k(t) * unit_field_k

-- exactly what a per-frame loop would produce, at 1/7 of the work.
GATE: the last frame's contraction is checked against a full one-shot
dehom_fields call before anything is written.

THE POINT: top surface of the top face sheet, at the cell centre --
the outermost element row of the top face (x2 = 4.7824) and, on that
row, the element nearest x1 = 0.  That is the same physical point as
path_1's centre sample in the static study.

Output frame: MATERIAL (ply axes), the project default.

In:  ff_dyn.dat, ../../HC_pm45/opensg/sg_2d/45pm_singleextrude_bd.yaml
Out: topcenter_stress_t.dat -- t + S11 S22 S33 S12 S13 S23 per frame
     dyn_unit_bank.npz      -- the 37 unit fields (whole SG), so any
                               other point/frame is a contraction away
"""
import os
import sys
import time

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
# output frame, same contract as `opensg <yaml> D --material|--global`:
# material (ply axes) is the project default; global (SG axes) is what
# the DYNAMIC 3-D csv carries, because abq_dump_dyn3d.py transformed it
# with getTransformedField(GLOBAL_CSYS).  Comparing against that dump is
# cleanest in global -- neither side is rotated.
FRAME = "material"
# TAG = the step duration; inputs and outputs live in that subfolder so
# the 500 us and 2500 us studies stay separate:
#     python run_dehom_dyn.py --global           -> 500us/
#     python run_dehom_dyn.py 2500us --global    -> 2500us/
TAG = "500us"
for a in sys.argv[1:]:
    s = a.strip().lower()
    if s in ("--material", "--global"):
        FRAME = s.lstrip("-")
    elif s.endswith("us") and s[:-2].isdigit():
        TAG = s
SUF = "" if FRAME == "material" else "_global"
DIR = os.path.join(HERE, TAG)
if not os.path.isdir(DIR):
    os.makedirs(DIR)
YAML = os.path.normpath(os.path.join(
    HERE, "..", "..", "HC_pm45", "opensg", "sg_2d",
    "45pm_singleextrude_bd.yaml"))
FACE_Z = 4.34491                 # top face sheet spans 4.34491..4.84491
COMP = ["S11", "S22", "S33", "S12", "S13", "S23"]
SGIDX = [0, 1, 2, 5, 4, 3]       # storage xx yy zz yz xz xy -> printed

print("start :", time.strftime("%Y-%m-%d %H:%M:%S"))
from opensg_solid.sg_homo import plate_homo_2d              # noqa: E402
from opensg_solid.sg_dehom import (dehom_fields,            # noqa: E402
                                   material_frame_fields)

print("tag   : %s, frame %s" % (TAG, FRAME.upper()))
T = np.loadtxt(os.path.join(DIR, "ff_dyn.dat"))
t = T[:, 0]
FF = T[:, 1:7]
eps = T[:, 7:13]
dE1, dE2 = T[:, 13:19], T[:, 19:25]
d11, d12, d22 = T[:, 25:31], T[:, 31:37], T[:, 37:43]
qt6 = T[0, 43:49]
# STATION CONSISTENCY (the static path-1/path-2 lesson, applied per
# frame): the 3-D reference history is Richardson-extrapolated onto the
# TRUE midspan, where the response is mirror-symmetric at every instant
# and Q1(t) = Q2(t) = 0 -- but the plate station (ESTA) sits half an
# element off, so its central-difference dE1/dE2 carry the Q(t) content
# the reference plane cannot contain.  Zero the odd first derivatives;
# the even d2eps stay.  (Statically this + the tau column took the
# classical-beats-RM count to 0/24.)
dE1[:] = 0.0
dE2[:] = 0.0
print("driver: dE1 = dE2 = 0 per frame (midspan-symmetric reference);"
      " d2eps kept")
if not np.allclose(T[:, 43:49], qt6):
    raise SystemExit("qt6 varies between frames -- this bank assumes a"
                     " constant face pressure")
print("frames: %d, t = %.6e .. %.6e; qt6 = %s"
      % (len(t), t.min(), t.max(), qt6))

# ---- the point: outermost top-face row, element nearest x1 = 0
d = yaml.safe_load(open(YAML))
nd = np.asarray(d["nodes"], float)[:, :2]
cells = np.asarray(d["cells"], int)
cen = nd[cells].mean(axis=1)
rows = np.unique(np.round(cen[:, 1][cen[:, 1] > FACE_Z], 4))
top = rows[-1]
k = np.where(np.abs(cen[:, 1] - top) < 1e-3)[0]
E_PT = int(k[np.argmin(np.abs(cen[k, 0]))])
print("point : element %d at (x1, x2) = (%.5f, %.5f); top-face rows %s"
      % (E_PT, cen[E_PT, 0], cen[E_PT, 1], np.round(rows, 4)))

# tau reaction: the pressure load column re-reacted along the cell's
# own shear path (the static fix; uniform under-loads the face-sheet
# bays of this in-plane-heterogeneous cell by ~20 %)
r = plate_homo_2d(YAML, refined=1, q_reaction="tau")
print("homo  : done, omega %.6f" % r["omega"])

Z = np.zeros(6)


def unit(**kw):
    """One unit recovery -> (E, Q, 6) at the GAUSS POINTS.

    NOT element-averaged: the through-thickness variation inside an
    element is exactly what a face-sheet ply needs, and averaging the
    4 Gauss points throws it away.  material_frame_fields rotates by a
    fixed per-element angle, so it commutes with the superposition and
    can be baked into the bank."""
    Gam, Sig, _U = dehom_fields(r, kw.pop("eps", Z), **kw)
    if FRAME == "material":
        Gam, Sig = material_frame_fields(Gam, Sig, r)
    return np.asarray(Sig)                       # (E, Q, 6)


bank = []
for kk in range(6):
    e = np.eye(6)[kk]
    for kw in (dict(eps=e), dict(dE1=e), dict(dE2=e), dict(dE11=e),
               dict(dE12=e), dict(dE22=e)):
        bank.append(unit(**kw))
bank.append(unit(qt6=qt6))
bank = np.stack(bank)                            # (37, E, Q, 6)
NQ = bank.shape[2]
print("bank  : %d unit fields, %d elements x %d Gauss points"
      % (len(bank), bank.shape[1], NQ))

# the Gauss point of the point element nearest the TRUE top surface
gx = np.asarray(r["phi_qn"]) @ np.asarray(r["x_end"])[E_PT]   # (Q, 2)
Q_PT = int(np.argmax(gx[:, 1]))
print("gauss : element %d has %d gps, x2 = %s -> using gp %d at"
      " (x1, x2) = (%.5f, %.5f)"
      % (E_PT, NQ, np.round(gx[:, 1], 4), Q_PT, gx[Q_PT, 0],
         gx[Q_PT, 1]))

W = np.zeros((len(t), 37))
for kk in range(6):
    W[:, 6 * kk + 0] = eps[:, kk]
    W[:, 6 * kk + 1] = dE1[:, kk]
    W[:, 6 * kk + 2] = dE2[:, kk]
    W[:, 6 * kk + 3] = d11[:, kk]
    W[:, 6 * kk + 4] = d12[:, kk]
    W[:, 6 * kk + 5] = d22[:, kk]
W[:, 36] = 1.0

# ---- gate on the LAST frame (largest drivers)
i = len(t) - 1
g = W[i] @ bank.reshape(37, -1)
Gam0, Sig0, _ = dehom_fields(r, eps[i], dE1=dE1[i], dE2=dE2[i],
                             qt6=qt6, dE11=d11[i], dE12=d12[i],
                             dE22=d22[i])
if FRAME == "material":
    Gam0, Sig0 = material_frame_fields(Gam0, Sig0, r)
one = np.asarray(Sig0).ravel()
err = np.abs(g - one).max() / max(np.abs(one).max(), 1e-30)
print("gate  : bank vs one-shot, last frame, max rel err %.2e  %s"
      % (err, "OK" if err < 1e-10 else "FAIL"))
if err >= 1e-10:
    raise SystemExit("the unit bank does not reproduce the one-shot"
                     " recovery -- do not trust the history")

# The COMPARISON series is the ELEMENT MEAN of the 4 Gauss points.
# The 3-D deck writes `*Element Output, position=CENTROIDAL`, i.e. ONE
# value per C3D20R -- not its integration points -- so an element mean
# is the matching granularity.  A single Gauss point would be finer
# than anything the reference can support (it differs by ~5 % here).
# The full Gauss bank is still saved below for through-thickness work.
S_pt = W @ bank[:, E_PT, :, :].mean(axis=1)      # (nt, 6) storage order
S_gp = W @ bank[:, E_PT, Q_PT, :]                # the top gp, for ref
out = os.path.join(DIR, "topcenter_stress_t%s.dat" % SUF)
with open(out, "w") as f:
    f.write("# MSG-RM recovered stress history at the TOP-CENTRE point"
            " of the midspan recovery station\n")
    f.write("# SG element %d at (x1, x2) = (%.5f, %.5f); ELEMENT MEAN"
            " of its %d Gauss points -- matching the 3-D deck's\n"
            % (E_PT, cen[E_PT, 0], cen[E_PT, 1], NQ))
    f.write("# `position=CENTROIDAL` output (one value per element).\n")
    f.write("# %s frame; stress in the SG input system (MPa)."
            "  Gauss-point detail: dyn_unit_bank%s.npz\n"
            % (FRAME.upper(), SUF))
    f.write("# %14s" % "t[s]")
    for c in COMP:
        f.write(" %15s" % c)
    f.write("\n")
    for i in range(len(t)):
        f.write("%16.8e" % t[i])
        for j in SGIDX:
            f.write(" %15.7e" % S_pt[i, j])
        f.write("\n")
print("wrote %s (%d rows)" % (os.path.basename(out), len(t)))
gauss_xy = np.einsum("qn,end->eqd", np.asarray(r["phi_qn"]),
                     np.asarray(r["x_end"]))     # (E, Q, 2)
np.savez_compressed(os.path.join(DIR, "dyn_unit_bank%s.npz" % SUF),
                    bank=bank.astype(np.float32), W=W, t=t,
                    gauss_xy=gauss_xy.astype(np.float32),
                    elem_point=E_PT, gp_point=Q_PT,
                    xy_point=gx[Q_PT],
                    frame=np.array(FRAME),
                    storage=np.array("xx yy zz yz xz xy"))
print("wrote dyn_unit_bank%s.npz  (bank %s, Gauss coords included)"
      % (SUF, bank.shape))
for j, c in zip(SGIDX, COMP):
    print("  %-4s  %+.4f .. %+.4f MPa" % (c, S_pt[:, j].min(),
                                          S_pt[:, j].max()))

# ---- the CLASSICAL history at the same point: strain-only recovery
# (the classical route = the same base kernel, no gradient chains, no
# load ladder), driven by eps_cl(t) = C_classical^-1 FF(t).  The
# classical 6x6 IS r["C_eff"] (the ABD the refined run computes first),
# and the first 6 bank entries (columns 6*k+0) are exactly the eps-unit
# fields of that base kernel.
eps_cl = np.linalg.solve(np.asarray(r["C_eff"], float),
                         FF.T).T                          # (nt, 6)
bank_eps = bank[[6 * kk for kk in range(6)]]              # (6, E, Q, 6)
S_cl = eps_cl @ bank_eps[:, E_PT, :, :].mean(axis=1)      # (nt, 6)
i = len(t) - 1
Gc, Sc, _ = dehom_fields(r, eps_cl[i])
if FRAME == "material":
    Gc, Sc = material_frame_fields(Gc, Sc, r)
one_c = np.asarray(Sc)[E_PT].mean(axis=0)
err_c = (np.abs(eps_cl[i] @ bank_eps[:, E_PT, :, :].mean(axis=1)
               - one_c).max() / max(np.abs(one_c).max(), 1e-30))
print("gate  : classical bank vs one-shot, last frame, %.2e %s"
      % (err_c, "OK" if err_c < 1e-10 else "FAIL"))
if err_c >= 1e-10:
    raise SystemExit("classical contraction failed the gate")
out_c = os.path.join(DIR, "topcenter_stress_t_classical%s.dat" % SUF)
with open(out_c, "w") as f:
    f.write("# CLASSICAL plate recovered stress history at the same"
            " top-centre point -- strain-only\n")
    f.write("# recovery (no gradient chains, no pressure ladder):"
            " eps_cl(t) = C_classical^-1 FF(t)\n")
    f.write("# SG element %d, element mean of %d Gauss points, %s"
            " frame (MPa)\n" % (E_PT, NQ, FRAME.upper()))
    f.write("# %14s" % "t[s]")
    for c in COMP:
        f.write(" %15s" % c)
    f.write("\n")
    for i in range(len(t)):
        f.write("%16.8e" % t[i])
        for j in SGIDX:
            f.write(" %15.7e" % S_cl[i, j])
        f.write("\n")
print("wrote %s (%d rows)" % (os.path.basename(out_c), len(t)))
print("end   :", time.strftime("%Y-%m-%d %H:%M:%S"))
