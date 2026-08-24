"""test_qcolumn_redistribution.py -- MECHANISM test for the path-1
in-plane amplitude deficit.

The pressure load column V1Lt is a PURE-NEUMANN cell solve: the net
face force is absorbed by the <w> = 0 KKT rows, which is mechanically
a UNIFORM body force over the whole material area.  In the real
sandwich the balancing spanwise shear flows through the CORE WALLS, so
the uniform reaction under-loads the top-face bay bending.  Test: keep
everything else identical and re-solve the top load column with the
net face force reacted explicitly (net-zero column -> the KKT rows
stay silent):

    orig  the shipped column (gate: rebuild must match r["V1Lt"])
    A     reaction spread over the CORE (mat 1) nodes, area-weighted
    B     reaction spread over ALL nodes, C_xz-shear-stiffness-weighted

then compare the path-1 in-plane fluctuation of each q-chain against
the 3-D residual (3-D minus the eps/gradient chains).

In:  ../sg_2d/45pm_singleextrude_bd.yaml, ../dehomo/45pm_singleextrude_bd.ff,
     ../path_1/path_1.coords, ../../abaqus/station_path_stress/path_1_abaqus.dat
Out: ./qcolumn_redistribution_path1.dat, ./path_1_qredist_S11.png,
     the printed table
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
JP = {"11": 0, "22": 1, "12": 3}

print("start : %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
r = plate_homo_2d(YAML, refined=1, plot=False)
state = read_ff_state(FF)
eps = np.linalg.solve(np.asarray(r["C_eff"], float), state["FF"])

# ---- rebuild the mesh-side objects of the f_faces block (sg_homo)
sc = r["sc"]
points = np.asarray(sc["nodes"], float)[:, :2]
cells0 = np.asarray(sc["cells"], int)
cells_bx = cells0[:, [0, 1, 3, 2]]           # quad4 basix order
x_end = np.asarray(r["x_end"])
gate = np.abs(x_end - points[cells_bx]).max()
print("gate: basix cell order vs x_end  %.1e %s"
      % (gate, "OK" if gate < 1e-12 else "FAIL"))
if gate > 1e-12:
    raise SystemExit("cell-order reconstruction failed")

pce = np.asarray(r["periodic_cells_en"], int)
n_unique = np.asarray(r["V0"]).shape[0]
red_of = np.full(points.shape[0], -1, dtype=np.int64)
red_of[cells_bx.ravel()] = pce.ravel()
omega = float(r["omega"])

# lumped nodal areas (shoelace element area / 4, scattered to reduced)
x, y = points[cells0][:, :, 0], points[cells0][:, :, 1]
area_e = 0.5 * np.abs(
    (x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y).sum(1))
mat_e = np.asarray(sc["mat_id"], int)
wA = np.zeros(n_unique // 3)
np.add.at(wA, red_of[cells0.ravel()],
          np.repeat(area_e / 4.0, 4))
wall_A = np.zeros(n_unique // 3)
np.add.at(wall_A, red_of[cells0[mat_e == 1].ravel()],
          np.repeat(area_e[mat_e == 1] / 4.0, 4))
C_ess = np.asarray(r["C_ess"])
cxz_A = np.zeros(n_unique // 3)
np.add.at(cxz_A, red_of[cells0.ravel()],
          np.repeat(C_ess[:, 4, 4] * area_e / 4.0, 4))
print("areas: total %.4f, walls (mat 1) %.4f, faces %.4f"
      % (area_e.sum(), area_e[mat_e == 1].sum(),
         area_e[mat_e != 1].sum()))

# ---- the original f_faces block, verbatim (sg_homo lines 1029-1055)
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

phi_hi, dphi_hi, W_hi = fe_tables(2, 4, hi=True)


def qcol(ff):
    lad = plate_shear_ladder(x_end, dphi_hi, phi_hi, W_hi,
                             C_ess, pce, n_unique, 2, omega,
                             f_faces=ff, node_y2=node_y2)
    return lad["V1Lt"], lad["V2Lt"]


V1Lt0, V2Lt0 = qcol(f_faces)
dev = (np.abs(np.asarray(V1Lt0) - np.asarray(r["V1Lt"])).max()
       / (np.abs(np.asarray(r["V1Lt"])).max() + 1e-30))
print("gate: rebuilt top load column vs shipped V1Lt  rel %.1e %s"
      % (dev, "OK" if dev < 1e-8 else "FAIL"))
if dev > 1e-8:
    raise SystemExit("f_faces reconstruction does not match the"
                     " package -- fix before trusting the variants")

# ---- variants: add the explicit reaction so the column is net-zero
net = f_faces[2::3, 0].sum()          # the net the KKT would absorb
variants = {}
for tag, w in (("A_walls", wall_A), ("B_Cxz", cxz_A)):
    fv = f_faces.copy()
    fv[2::3, 0] -= net * w / w.sum()
    print("variant %s: column net %.2e (was %.4f)"
          % (tag, fv[2::3, 0].sum(), net))
    variants[tag] = qcol(fv)

# ---- q-chains along path 1
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
fl = lambda v: v - v.mean()                            # noqa: E731

kw_grad = dict(dE1=state["dE1"], dE2=state["dE2"],
               dE11=state["dE11"], dE12=state["dE12"],
               dE22=state["dE22"])
z6 = np.zeros(6)


def path_piece(rr, *a, **k):
    G_, S_, _ = dehom_fields(rr, *a, **k)
    _, Sm = material_frame_fields(G_, S_, rr)
    return np.asarray(Sm)[:, :, PIDX].mean(axis=1)[idx]

beps = path_piece(r, eps, **kw_grad)
runs = {"orig": path_piece(r, z6, qt6=state["qt6"])}
for tag, (v1, v2) in variants.items():
    r2 = dict(r)
    r2["V1Lt"], r2["V2Lt"] = v1, v2
    runs[tag] = path_piece(r2, z6, qt6=state["qt6"])

print("\n---- path 1: q-chain variants vs the 3-D residual"
      " (3-D minus eps/gradient chains)")
print("%-8s | %s" % ("column", " | ".join(
    "S%s: fl  ratio3D  rms%%full" % c for c in JP)))
tgt = abq - beps
for tag in runs:
    q = runs[tag]
    cols = []
    for c, j in JP.items():
        fq, f3 = np.std(fl(q[:, j])), np.std(fl(tgt[:, j]))
        rms = (100 * np.sqrt(np.mean((beps[:, j] + q[:, j]
                                      - abq[:, j]) ** 2))
               / np.abs(abq[:, j]).max())
        cols.append("%7.4f %7.3f %8.2f" % (fq, fq / f3, rms))
    print("%-8s | %s" % (tag, " | ".join(cols)))
cols = []
for c, j in JP.items():
    cols.append("%7.4f %7s %8s" % (np.std(fl(tgt[:, j])), "1.000", "-"))
print("%-8s | %s" % ("3-D tgt", " | ".join(cols)))

out = os.path.join(HERE, "qcolumn_redistribution_path1.dat")
with open(out, "w") as f:
    f.write("# q-column reaction-redistribution test along path_1 --"
            " MATERIAL frame, element means\n")
    f.write("# tgt = 3-D minus the eps/gradient chains (what the"
            " q-chain must supply)\n")
    f.write("# %10s %4s" % ("s", "mat"))
    for c in JP:
        f.write(" %13s" % ("S%s_tgt" % c))
        for tag in runs:
            f.write(" %13s" % ("S%s_%s" % (c, tag.split("_")[0])))
    f.write("\n")
    for i in range(len(s)):
        f.write("%12.6f %4d" % (s[i], int(C[i, 3])))
        for c, j in JP.items():
            f.write(" %13.6e" % tgt[i, j])
            for tag in runs:
                f.write(" %13.6e" % runs[tag][i, j])
        f.write("\n")
print("\nwrote %s" % os.path.basename(out))

fig, ax = plt.subplots(figsize=(7.0, 4.2))
ax.plot(s, tgt[:, 0], "-", lw=2.0, color="#c0392b",
        label="3-D minus eps/gradient chains")
ax.plot(s, runs["orig"][:, 0], "--", lw=1.6, color="k",
        label="q-chain, shipped column")
ax.plot(s, runs["A_walls"][:, 0], "-.", lw=1.6, color="#2471a3",
        label="q-chain, reaction on core walls")
ax.plot(s, runs["B_Cxz"][:, 0], ":", lw=1.8, color="#27ae60",
        label="q-chain, reaction $C_{xz}$-weighted")
ax.set_xlabel(r"$x_1$  (mm)")
ax.set_ylabel(r"$\sigma_{11}$  (MPa)")
ax.grid(alpha=0.3)
ax.axhline(0.0, color="k", lw=0.6, alpha=0.4)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "path_1_qredist_S11.png"), dpi=170,
            bbox_inches="tight")
plt.close(fig)
print("wrote path_1_qredist_S11.png")
print("end   : %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
