"""bay_fix_and_decompose.py -- the two follow-ups of rm_fix_analysis:

A. the CORRECTED bay-load figure: M_loc(x1) of the top face sheet for
   the 3-D reference (bay_d2_3d.dat, per-row-interpolated) against the
   classical / shipped-RM / tau-reacted-RM recoveries, plus per-bay
   parabola fits |2a|/q for every curve -- the one-figure explanation
   of the path-1 oscillation gap AND the Abaqus correctness check.

B. WHICH RM chain pollutes the path-2 core S13/S12 where classical
   wins?  The recovery is linear, so along path 2 the field splits into
   base (= classical) + grad (Eq. 63/64-66) + q (pressure).  Per core
   component this prints the RMS of every composition and each chain's
   own core contribution -- the polluter is named by data.

In:  ../../opensg/sg_2d/45pm_singleextrude_bd.yaml, ../../opensg/dehomo/
     45pm_singleextrude_bd.ff, ../../opensg/path_2/path_2.coords,
     ../../abaqus/station_path_stress/path_2_abaqus.dat, ./bay_d2_3d.dat
Out: ./bay_load_check.png, ./core_decompose.txt, the printed report
"""
import datetime
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                            # noqa: E402
import numpy as np                                         # noqa: E402
import yaml                                                # noqa: E402
from scipy.spatial import cKDTree                          # noqa: E402

from opensg_solid.cli import read_ff_state                 # noqa: E402
from opensg_solid.sg_dehom import (SGIDX, ORDER,           # noqa: E402
                                   dehom_fields, material_frame_fields)
from opensg_solid.sg_homo import (plate_homo_2d,           # noqa: E402
                                  plate_shear_ladder)
from opensg_solid.sg_mixed import fe_tables                # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OPG = os.path.join(HERE, "..", "..", "opensg")
YAML = os.path.join(OPG, "sg_2d", "45pm_singleextrude_bd.yaml")
FF = os.path.join(OPG, "dehomo", "45pm_singleextrude_bd.ff")
PIDX = [SGIDX[c] for c in ORDER]
COMP = ["S11", "S22", "S33", "S12", "S13", "S23"]
ZLO, ZHI, ZMID = 4.34491, 4.84491, 4.59491

print("start : %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
r = plate_homo_2d(YAML, refined=1, plot=False)
state = read_ff_state(FF)
C6 = np.asarray(r["C_eff"], float)
eps = np.linalg.solve(C6, state["FF"])

# ---- ladder rebuild + tau variant (gated, as in rm_fix_analysis)
sc = r["sc"]
points = np.asarray(sc["nodes"], float)[:, :2]
cells0 = np.asarray(sc["cells"], int)
cells_bx = cells0[:, [0, 1, 3, 2]]
x_end = np.asarray(r["x_end"])
assert np.abs(x_end - points[cells_bx]).max() < 1e-12
pce = np.asarray(r["periodic_cells_en"], int)
n_unique = np.asarray(r["V0"]).shape[0]
red_of = np.full(points.shape[0], -1, dtype=np.int64)
red_of[cells_bx.ravel()] = pce.ravel()
omega = float(r["omega"])
xx, yy = points[cells0][:, :, 0], points[cells0][:, :, 1]
area_e = 0.5 * np.abs((xx * np.roll(yy, -1, 1)
                       - np.roll(xx, -1, 1) * yy).sum(1))
node_y2 = np.zeros(n_unique // 3)
node_y2[red_of] = points[:, 1]
ftol = 1e-6 * float(max(np.ptp(points[:, 0]), np.ptp(points[:, 1])))
f_faces = np.zeros((n_unique, 2))
for col, (yf, sgn) in enumerate(
        ((points[:, 1].max(), 1.0), (points[:, 1].min(), -1.0))):
    nid = np.where(np.abs(points[:, 1] - yf) < ftol)[0]
    nid = nid[np.argsort(points[nid, 0])]
    seg = np.diff(points[nid, 0])
    wgt = np.zeros(len(nid))
    wgt[:-1] += 0.5 * seg
    wgt[1:] += 0.5 * seg
    np.add.at(f_faces[:, col], 3 * red_of[nid] + 2, sgn * wgt / omega)
C_ess = np.asarray(r["C_ess"])
phi_hi, dphi_hi, W_hi = fe_tables(2, 4, hi=True)


def qcol(ff):
    lad = plate_shear_ladder(x_end, dphi_hi, phi_hi, W_hi, C_ess, pce,
                             n_unique, 2, omega, f_faces=ff,
                             node_y2=node_y2)
    return lad["V1Lt"], lad["V2Lt"]


V1Lt0, V2Lt0 = qcol(f_faces)
assert (np.abs(np.asarray(V1Lt0) - np.asarray(r["V1Lt"])).max()
        / np.abs(np.asarray(r["V1Lt"])).max()) < 1e-8
z6 = np.zeros(6)
dE1u = np.linalg.solve(C6, np.array([0.0, 0, 0, 1.0, 0, 0]))
_, Su, _ = dehom_fields(r, z6, dE1=dE1u, dE2=z6)
tau_e = np.asarray(Su).mean(axis=1)[:, 4]
tau_w = np.zeros(n_unique // 3)
np.add.at(tau_w, red_of[cells0.ravel()],
          np.repeat(tau_e * area_e / 4.0, 4))
fv = f_faces.copy()
net = f_faces[2::3, 0].sum()
fv[2::3, 0] -= net * tau_w / tau_w.sum()
V1Lt_t, V2Lt_t = qcol(fv)

# ---- chain fields (global storage + material printed, element means)
kw_grad = dict(dE1=state["dE1"], dE2=state["dE2"],
               dE11=state["dE11"], dE12=state["dE12"],
               dE22=state["dE22"])


def fld(*a, **k):
    G_, S_, _ = dehom_fields(*a, **k)
    _, Sm = material_frame_fields(G_, S_, a[0])
    return (np.asarray(S_).mean(axis=1),
            np.asarray(Sm).mean(axis=1)[:, PIDX])


base_g, base_m = fld(r, eps)
gr_g, gr_m = fld(r, eps, **kw_grad)
grad_g, grad_m = gr_g - base_g, gr_m - base_m
qo_g, qo_m = fld(r, z6, qt6=state["qt6"])
r2 = dict(r)
r2["V1Lt"], r2["V2Lt"] = V1Lt_t, V2Lt_t
qt_g, qt_m = fld(r2, z6, qt6=state["qt6"])

# ---- A. corrected bay figure: M_loc(x1), per-row interpolation
d = yaml.safe_load(open(YAML))
nd = np.asarray(d["nodes"], float)[:, :2]
cl = np.asarray(d["cells"], int)
cen = nd[cl].mean(axis=1)
b3 = np.loadtxt(os.path.join(HERE, "bay_d2_3d.dat"))
jx = [float(v) for v in
      open(os.path.join(HERE, "bay_d2_3d.dat")).readlines()[1]
      .split(":")[1].replace("[", "").replace("]", "").split(",")]
face = (cen[:, 1] > ZLO) & (cen[:, 1] < ZHI)
zlv = np.unique(np.round(cen[face, 1], 4))
ktop = face & (np.abs(cen[:, 1] - zlv[-1]) < 1e-3)
o = np.argsort(cen[ktop, 0])
xg = cen[ktop, 0][o]


def mloc(Sg):
    M = np.zeros(len(xg))
    for z0 in zlv:
        k = face & (np.abs(cen[:, 1] - z0) < 1e-3)
        oo = np.argsort(cen[k, 0])
        M += np.interp(xg, cen[k, 0][oo],
                       Sg[k, 0][oo]) * (z0 - ZMID) * 0.125
    return M


def bayfit(x, M):
    edges = sorted([x.min() - 0.01] + jx + [x.max() + 0.01])
    out = []
    for b in range(len(edges) - 1):
        k = (x > edges[b] + 0.35) & (x < edges[b + 1] - 0.35)
        if k.sum() >= 5:
            out.append(abs(2.0 * np.polyfit(x[k], M[k], 2)[0]) / 0.1)
    return out


curves = [
    ("Reference 3-D FEA (Abaqus C3D20R)", b3[:, 0], b3[:, 1],
     dict(ls="-", color="#c0392b", marker="o", ms=3.0, lw=1.7)),
    ("Classical Plate (OpenSG)", xg, mloc(base_g),
     dict(ls="-.", color="#2471a3", marker="^", ms=3.0, lw=1.4)),
    ("Shear-refined (RM) Plate (OpenSG)", xg,
     mloc(base_g + grad_g + qo_g),
     dict(ls="--", color="k", marker="s", ms=2.6, lw=1.6)),
    (r"RM, $\tau$-reacted q column", xg, mloc(base_g + grad_g + qt_g),
     dict(ls=":", color="#1e8449", marker="*", ms=6.0, lw=1.5)),
]
fig, ax = plt.subplots(figsize=(7.0, 4.2))
print("\n---- per-bay |d2M/dx1^2| / q (parabola fits, open-bay"
      " interiors)")
for lab, x, M, st in curves:
    ax.plot(x, M, label=lab, **st)
    print("  %-36s %s" % (lab, " ".join("%.3f" % v
                                        for v in bayfit(x, M))))
for x0 in jx:
    ax.axvline(x0, color="k", lw=0.7, ls=":", alpha=0.6)
ax.set_xlabel(r"$x_1$  (mm)")
ax.set_ylabel(r"$M_{\mathrm{loc}}$ of the top face sheet  (N)")
ax.grid(alpha=0.3)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "bay_load_check.png"), dpi=170,
            bbox_inches="tight")
plt.close(fig)
print("wrote bay_load_check.png")

# ---- B. path-2 core decomposition
C = np.loadtxt(os.path.join(OPG, "path_2", "path_2.coords"))
dist, idx = cKDTree(cen).query(C[:, 1:3])
assert dist.max() < 1e-4
abq = np.loadtxt(os.path.join(HERE, "..", "..", "abaqus",
                              "station_path_stress",
                              "path_2_abaqus.dat"))[:, 4:10]
core = C[:, 3].astype(int) == 1
P = {"base": base_m[idx], "grad": grad_m[idx], "qo": qo_m[idx],
     "qt": qt_m[idx]}
lines = ["# path-2 AL-CORE decomposition: RMS %% (of path max|3-D|)"
         " per composition, and each chain's own core std\n"
         "# %-4s | %8s %9s %9s %9s %9s | %9s %9s %9s\n"
         % ("comp", "base=cl", "base+gr", "base+qo", "full", "full_tau",
            "std(gr)", "std(qo)", "corr(qo)")]
print("\n---- path-2 Al-core decomposition (%d of %d points)"
      % (int(core.sum()), len(core)))
print(lines[-1].strip("\n"))
for j, c in enumerate(COMP):
    den = max(np.abs(abq[:, j]).max(), 1e-30)
    a = abq[core, j]

    def rms(v):
        return 100 * np.sqrt(np.mean((v - a) ** 2)) / den

    b = P["base"][core, j]
    g = P["grad"][core, j]
    qo = P["qo"][core, j]
    qt = P["qt"][core, j]
    res = a - b
    cq = (np.corrcoef(res, qo)[0, 1]
          if qo.std() > 1e-12 and res.std() > 1e-12 else 0.0)
    row = ("%-6s | %8.2f %9.2f %9.2f %9.2f %9.2f | %9.4f %9.4f %9.2f"
           % (c, rms(b), rms(b + g), rms(b + qo), rms(b + g + qo),
              rms(b + g + qt), g.std(), qo.std(), cq))
    print(row)
    lines.append(row + "\n")
open(os.path.join(HERE, "core_decompose.txt"), "w").writelines(lines)
print("wrote core_decompose.txt")
print("end   : %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
