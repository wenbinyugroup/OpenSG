"""final_verdict.py -- does the two-part fix make RM beat classical in
EVERY (path, component, material-segment) cell?

    rm_fix = base + first-order gradient (Eq. 63; drivers ~0 at this
             symmetric station) + tau-reacted pressure column,
             WITHOUT the Eq. 64-66 second-order (d2eps) chains --
             the chains whose transverse/core content is the known
             interior weakness for soft-core sandwiches, and the very
             piece the core decomposition convicted.

Also: the 3-D bay-equilibrium fit as a WINDOW SWEEP (the wall-attach
zone is wide, so the honest free-bay statement needs the window shown),
and the two path-2 figures redrawn with the rm_fix curve.

In:  the usual HC_pm45 inputs + ./bay_d2_3d.dat
Out: ./segment_rms_final.txt, ./path_2_fix_S13.png / _S12.png (redrawn),
     the printed verdict
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
ABQD = os.path.join(HERE, "..", "..", "abaqus", "station_path_stress")
PIDX = [SGIDX[c] for c in ORDER]
COMP = ["S11", "S22", "S33", "S12", "S13", "S23"]
MATL = {1: "Al core", 2: "ply -45", 3: "ply +45"}

print("start : %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# ---- 3-D bay fit window sweep (data only, no model)
b3 = np.loadtxt(os.path.join(HERE, "bay_d2_3d.dat"))
print("\n---- 3-D free-bay curvature vs fit half-window (parabola on"
      " M_loc, centre bay)")
for w in (1.2, 1.5, 1.8, 2.0, 2.4, 2.6):
    k = np.abs(b3[:, 0]) < w
    a2 = 2.0 * np.polyfit(b3[k, 0], b3[k, 1], 2)[0]
    print("  |x1| < %.1f : %3d pts  |2a|/q = %.3f"
          % (w, int(k.sum()), abs(a2) / 0.1))

# ---- rebuild machinery (as gated before)
r = plate_homo_2d(YAML, refined=1, plot=False)
state = read_ff_state(FF)
C6 = np.asarray(r["C_eff"], float)
eps = np.linalg.solve(C6, state["FF"])
sc = r["sc"]
points = np.asarray(sc["nodes"], float)[:, :2]
cells0 = np.asarray(sc["cells"], int)
x_end = np.asarray(r["x_end"])
assert np.abs(x_end - points[cells0[:, [0, 1, 3, 2]]]).max() < 1e-12
pce = np.asarray(r["periodic_cells_en"], int)
n_unique = np.asarray(r["V0"]).shape[0]
red_of = np.full(points.shape[0], -1, dtype=np.int64)
red_of[cells0[:, [0, 1, 3, 2]].ravel()] = pce.ravel()
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


V1Lt0, _ = qcol(f_faces)
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
fv[2::3, 0] -= f_faces[2::3, 0].sum() * tau_w / tau_w.sum()
r2 = dict(r)
r2["V1Lt"], r2["V2Lt"] = qcol(fv)

kw2 = dict(dE1=state["dE1"], dE2=state["dE2"],
           dE11=state["dE11"], dE12=state["dE12"],
           dE22=state["dE22"])


def fld(rr, *a, **k):
    G_, S_, _ = dehom_fields(rr, *a, **k)
    _, Sm = material_frame_fields(G_, S_, rr)
    return np.asarray(Sm).mean(axis=1)[:, PIDX]


F = {"cl": fld(r, eps),
     "rm": fld(r, eps, qt6=state["qt6"], **kw2),          # shipped
     "fix": fld(r2, eps, qt6=state["qt6"],                # Eq.63 only,
                dE1=state["dE1"], dE2=state["dE2"]),      # tau column
     # the STATION-CONSISTENT drivers: the 3-D reference plane is TRUE
     # midspan (Q1 = Q2 = 0, odd-in-span content killed by the
     # symmetric-pair Richardson), while the .ff station sits half an
     # element off (Q1 = -q x' = -0.063) -- so the odd FIRST derivatives
     # are zeroed and the even SECOND derivatives kept, on the tau column
     "fix2": fld(r2, eps, qt6=state["qt6"], dE11=state["dE11"],
                 dE12=state["dE12"], dE22=state["dE22"])}

# ---- the 24-cell table
d = yaml.safe_load(open(YAML))
nd = np.asarray(d["nodes"], float)[:, :2]
cen = nd[np.asarray(d["cells"], int)].mean(axis=1)
tree = cKDTree(cen)
lines = ["# per-material-segment RMS %% (of path max|3-D|):\n"
         "#   rm   = shipped RM (uniform reaction, full 2nd order)\n"
         "#   fix  = tau-reacted q column + Eq.63 first order only\n"
         "#   fix2 = tau-reacted q column + MIDSPAN drivers (odd first\n"
         "#          derivatives = 0, even second derivatives kept)\n"
         "# %-8s %-4s %-9s %8s %8s %8s %8s   %s\n"
         % ("path", "comp", "segment", "cl", "rm", "fix", "fix2",
            "verdict")]
n_bad = {"rm": 0, "fix": 0, "fix2": 0}
for n in (1, 2):
    C = np.loadtxt(os.path.join(OPG, "path_%d" % n,
                                "path_%d.coords" % n))
    dist, idx = tree.query(C[:, 1:3])
    assert dist.max() < 1e-4
    abq = np.loadtxt(os.path.join(ABQD, "path_%d_abaqus.dat"
                                  % n))[:, 4:10]
    mt = C[:, 3].astype(int)
    for j, c in enumerate(COMP):
        den = max(np.abs(abq[:, j]).max(), 1e-30)
        for m in sorted(set(mt)):
            k = mt == m
            row = {t: 100 * np.sqrt(np.mean(
                (F[t][idx][k, j] - abq[k, j]) ** 2)) / den
                for t in F}
            verd = []
            for t in ("rm", "fix", "fix2"):
                if row[t] > row["cl"] + 0.05:
                    n_bad[t] += 1
                    verd.append("cl<%s" % t)
            lines.append("%-10s %-4s %-9s %8.2f %8.2f %8.2f %8.2f"
                         "   %s\n"
                         % ("path_%d" % n, c, MATL[m], row["cl"],
                            row["rm"], row["fix"], row["fix2"],
                            ",".join(verd) or "-"))
open(os.path.join(HERE, "segment_rms_final.txt"), "w").writelines(lines)
print("".join(lines))
print("verdict: classical beats shipped RM in %d cells, fix in %d,"
      " fix2 (station-consistent) in %d"
      % (n_bad["rm"], n_bad["fix"], n_bad["fix2"]))

# ---- redraw the two path-2 figures with the fixed curve
n = 2
C = np.loadtxt(os.path.join(OPG, "path_2", "path_2.coords"))
dist, idx = tree.query(C[:, 1:3])
abq = np.loadtxt(os.path.join(ABQD, "path_2_abaqus.dat"))[:, 4:10]
s = C[:, 0]
mchg = np.where(np.diff(C[:, 3].astype(int)) != 0)[0]
for c in ("S13", "S12"):
    j = COMP.index(c)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(s, abq[:, j], "-o", color="#c0392b", ms=3.2, lw=1.7,
            label="Reference 3-D FEA (Abaqus C3D20R)")
    ax.plot(s, F["rm"][idx][:, j], "--s", color="k", ms=2.8, lw=1.6,
            label="Shear-refined (RM) Plate (OpenSG)")
    ax.plot(s, F["cl"][idx][:, j], "-.^", color="#2471a3", ms=3.2,
            lw=1.4, label="Classical Plate (OpenSG)")
    ax.plot(s, F["fix2"][idx][:, j], ":*", color="#1e8449", ms=6.0,
            lw=1.5, label=r"RM fixed ($\tau$ column, midspan drivers"
                          r" $Q_1{=}0$)")
    for i in mchg:
        ax.axvline(0.5 * (s[i] + s[i + 1]), color="k", lw=0.7, ls=":",
                   alpha=0.6)
    ax.set_xlabel(r"$s/s_{\mathrm{total}}$  (0 = top surface, 1 ="
                  r" bottom surface)")
    ax.set_ylabel(r"$\sigma_{%s}$  (MPa)" % c[1:])
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "path_2_fix_%s.png" % c), dpi=170,
                bbox_inches="tight")
    plt.close(fig)
    print("wrote path_2_fix_%s.png" % c)
print("end   : %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
