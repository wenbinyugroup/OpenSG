"""final_composition_test.py -- how much of the path-1 gap does each
identified cause carry?  Combinations of:

  q-column reaction  orig    uniform (the shipped <w> = 0 KKT reaction)
                     walls   net face force reacted on the core (mat 1)
                             nodes, area-weighted
                     tau     reacted with the cell's OWN sigma_xz
                             distribution under unit Q1 (the Eq. 63
                             chain at dE1 = S [0 0 0 1 0 0]) -- the
                             asymptotically consistent weight
  macro state        FFp     the plate .ff resultants (shipped)
                     FF3     the resultants the 3-D really carries at
                             the station (check_3d_moment.py integral:
                             membrane N11/N22 included)

Every combination is recovered along path 1 (material frame, element
means) and scored against the 3-D; means and RMS per component.

In:  ../sg_2d/45pm_singleextrude_bd.yaml, ../dehomo/45pm_singleextrude_bd.ff,
     ../path_1/path_1.coords, ../../abaqus/station_path_stress/path_1_abaqus.dat
Out: ./final_composition_path1.dat, ./path_1_final_S11.png,
     the printed matrix
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
YAML = os.path.join(HERE, "..", "sg_2d", "45pm_singleextrude_bd.yaml")
FF = os.path.join(HERE, "..", "dehomo", "45pm_singleextrude_bd.ff")
COORDS = os.path.join(HERE, "..", "path_1", "path_1.coords")
ABQ = os.path.join(HERE, "..", "..", "abaqus", "station_path_stress",
                   "path_1_abaqus.dat")
PIDX = [SGIDX[c] for c in ORDER]

# the station resultants the 3-D really carries (check_3d_moment.py)
FF3 = np.array([-2.621642e-01, -1.715065e-01, -2.827218e-03,
                -2.348258e+01, -7.803040e+00, -1.719703e-02])

print("start : %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
r = plate_homo_2d(YAML, refined=1, plot=False)
state = read_ff_state(FF)
C6 = np.asarray(r["C_eff"], float)

# ---- mesh-side reconstruction (gated in test_qcolumn_redistribution)
sc = r["sc"]
points = np.asarray(sc["nodes"], float)[:, :2]
cells0 = np.asarray(sc["cells"], int)
cells_bx = cells0[:, [0, 1, 3, 2]]
x_end = np.asarray(r["x_end"])
if np.abs(x_end - points[cells_bx]).max() > 1e-12:
    raise SystemExit("cell-order reconstruction failed")
pce = np.asarray(r["periodic_cells_en"], int)
n_unique = np.asarray(r["V0"]).shape[0]
red_of = np.full(points.shape[0], -1, dtype=np.int64)
red_of[cells_bx.ravel()] = pce.ravel()
omega = float(r["omega"])
x, y = points[cells0][:, :, 0], points[cells0][:, :, 1]
area_e = 0.5 * np.abs(
    (x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y).sum(1))
mat_e = np.asarray(sc["mat_id"], int)

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
    np.add.at(f_faces[:, col], 3 * red_of[nid] + 2,
              sgn * wgt / omega)
C_ess = np.asarray(r["C_ess"])
phi_hi, dphi_hi, W_hi = fe_tables(2, 4, hi=True)


def qcol(ff):
    lad = plate_shear_ladder(x_end, dphi_hi, phi_hi, W_hi, C_ess, pce,
                             n_unique, 2, omega, f_faces=ff,
                             node_y2=node_y2)
    return lad["V1Lt"], lad["V2Lt"]


V1Lt0, V2Lt0 = qcol(f_faces)
if (np.abs(np.asarray(V1Lt0) - np.asarray(r["V1Lt"])).max()
        / np.abs(np.asarray(r["V1Lt"])).max()) > 1e-8:
    raise SystemExit("f_faces reconstruction does not match")

# ---- reaction weights
z6 = np.zeros(6)
net = f_faces[2::3, 0].sum()
wall_A = np.zeros(n_unique // 3)
np.add.at(wall_A, red_of[cells0[mat_e == 1].ravel()],
          np.repeat(area_e[mat_e == 1] / 4.0, 4))
# tau-hat: the cell's sigma_xz under unit Q1 (dM11/dx1 = 1), Eq. 63
dE1u = np.linalg.solve(C6, np.array([0.0, 0, 0, 1.0, 0, 0]))
_, Su, _ = dehom_fields(r, z6, dE1=dE1u, dE2=z6)
tau_e = np.asarray(Su).mean(axis=1)[:, 4]          # xz, SG-global
Ish = (tau_e * area_e).sum() / omega
print("gate: integral of sigma_xz under unit Q1 = %.6f (target 1)"
      % Ish)
tau_w = np.zeros(n_unique // 3)
np.add.at(tau_w, red_of[cells0.ravel()],
          np.repeat(tau_e * area_e / 4.0, 4))

cols = {"orig": (np.asarray(V1Lt0), np.asarray(V2Lt0))}
for tag, w in (("walls", wall_A), ("tau", tau_w)):
    fv = f_faces.copy()
    fv[2::3, 0] -= net * w / w.sum()
    cols[tag] = qcol(fv)

# ---- path recovery per combination
d = yaml.safe_load(open(YAML))
nd = np.asarray(d["nodes"], float)[:, :2]
cen = nd[np.asarray(d["cells"], int)].mean(axis=1)
C = np.loadtxt(COORDS)
dist, idx = cKDTree(cen).query(C[:, 1:3])
if dist.max() >= 1e-4:
    raise SystemExit("coords pairing failed")
A = np.loadtxt(ABQ)
abq = A[:, 4:10]
s = C[:, 0]

kw_grad = dict(dE1=state["dE1"], dE2=state["dE2"],
               dE11=state["dE11"], dE12=state["dE12"],
               dE22=state["dE22"])


def path_piece(rr, *a, **k):
    G_, S_, _ = dehom_fields(rr, *a, **k)
    _, Sm = material_frame_fields(G_, S_, rr)
    return np.asarray(Sm)[:, :, PIDX].mean(axis=1)[idx]


beps = {"FFp": path_piece(r, np.linalg.solve(C6, state["FF"]),
                          **kw_grad),
        "FF3": path_piece(r, np.linalg.solve(C6, FF3), **kw_grad)}
qp = {}
for tag, (v1, v2) in cols.items():
    r2 = dict(r)
    r2["V1Lt"], r2["V2Lt"] = v1, v2
    qp[tag] = path_piece(r2, z6, qt6=state["qt6"])

print("\n---- path 1 RMS %% (of max|3-D|) per combination")
hdr = "%-12s" % "combo"
for c in ORDER:
    hdr += " %6s" % ("S" + c)
print(hdr + "   meanS11(3D %.4f)" % abq[:, 0].mean())
best = None
for ftag in ("FFp", "FF3"):
    for qtag in ("orig", "walls", "tau"):
        tot = beps[ftag] + qp[qtag]
        row = "%-12s" % (ftag + "+" + qtag)
        for j in range(6):
            rms = (100 * np.sqrt(np.mean((tot[:, j] - abq[:, j]) ** 2))
                   / np.abs(abq[:, j]).max())
            row += " %6.2f" % rms
        print(row + "   %8.4f" % tot[:, 0].mean())
        if ftag == "FF3" and qtag == "tau":
            best = tot

out = os.path.join(HERE, "final_composition_path1.dat")
with open(out, "w") as f:
    f.write("# path 1 -- 3-D vs shipped RM vs the corrected"
            " composition (FF3 + tau-reacted q column)\n")
    f.write("# %10s %4s" % ("s", "mat"))
    for c in ORDER:
        f.write(" %13s %13s %13s" % ("S%s_abq" % c, "S%s_rm" % c,
                                     "S%s_fix" % c))
    f.write("\n")
    ship = beps["FFp"] + qp["orig"]
    for i in range(len(s)):
        f.write("%12.6f %4d" % (s[i], int(C[i, 3])))
        for j in range(6):
            f.write(" %13.6e %13.6e %13.6e"
                    % (abq[i, j], ship[i, j], best[i, j]))
        f.write("\n")
print("\nwrote %s" % os.path.basename(out))

fig, ax = plt.subplots(figsize=(7.0, 4.2))
ax.plot(s, abq[:, 0], "-", lw=2.0, color="#c0392b",
        label="Abaqus 3-D FEA (C3D20R)")
ax.plot(s, ship[:, 0], "--", lw=1.6, color="k",
        label="MSG-RM recovery (shipped)")
ax.plot(s, best[:, 0], "-.", lw=1.6, color="#2471a3",
        label=r"MSG-RM, $\tau$-reacted q column + 3-D resultants")
ax.set_xlabel(r"$x_1$  (mm)")
ax.set_ylabel(r"$\sigma_{11}$  (MPa)")
ax.grid(alpha=0.3)
ax.axhline(0.0, color="k", lw=0.6, alpha=0.4)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "path_1_final_S11.png"), dpi=170,
            bbox_inches="tight")
plt.close(fig)
print("wrote path_1_final_S11.png")
print("end   : %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
