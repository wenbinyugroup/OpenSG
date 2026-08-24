"""rm_fix_analysis.py -- WHERE does classical beat the shipped RM
recovery, and does re-reacting the pressure column through the cell's
shear path fix it EVERYWHERE?

Builds four recoveries on the same law + FF (material frame, element
means): classical base (V0 eps -- what the classical route computes),
RM as shipped (base + gradient + q with the uniform KKT reaction), and
RM with the q-column net force re-reacted on (a) the core walls,
(b) the cell's own sigma_xz distribution under unit Q1 (tau).  Scores
every (path, component, MATERIAL segment) cell, prints the verdict
counts, and draws:

  bay_load_check.png     d2M_loc/dx1^2 of the top face sheet: 3-D
                         (from bay_d2_3d.dat) vs classical / RM / RM-tau
                         -- the one-figure explanation of path 1
  path_2_fix_*.png       the components where classical beats shipped
                         RM in the core, with the tau curve added

Gates: rebuilt V1Lt == shipped; RM-orig path samples == the shipped
path_N_opensg.dat rows; sigma_xz-under-unit-Q1 integral == 1.

In:  ../../opensg/sg_2d/45pm_singleextrude_bd.yaml, ../../opensg/dehomo/
     45pm_singleextrude_bd.ff, ../../opensg/path_N/{path_N.coords,
     path_N_opensg.dat}, ../../opensg/classical/path_N_classical.dat,
     ../../abaqus/station_path_stress/path_N_abaqus.dat, ./bay_d2_3d.dat
Out: ./path_N_rmtau.dat, ./segment_rms.txt, the PNGs above
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
ZLO, ZHI, ZMID = 4.34491, 4.84491, 4.59491

print("start : %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
r = plate_homo_2d(YAML, refined=1, plot=False)
state = read_ff_state(FF)
C6 = np.asarray(r["C_eff"], float)
eps = np.linalg.solve(C6, state["FF"])

# ---- rebuild f_faces + the reaction variants (gated)
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
    np.add.at(f_faces[:, col], 3 * red_of[nid] + 2, sgn * wgt / omega)
C_ess = np.asarray(r["C_ess"])
phi_hi, dphi_hi, W_hi = fe_tables(2, 4, hi=True)


def qcol(ff):
    lad = plate_shear_ladder(x_end, dphi_hi, phi_hi, W_hi, C_ess, pce,
                             n_unique, 2, omega, f_faces=ff,
                             node_y2=node_y2)
    return lad["V1Lt"], lad["V2Lt"]


V1Lt0, V2Lt0 = qcol(f_faces)
dev = (np.abs(np.asarray(V1Lt0) - np.asarray(r["V1Lt"])).max()
       / np.abs(np.asarray(r["V1Lt"])).max())
print("gate: rebuilt V1Lt vs shipped  rel %.1e %s"
      % (dev, "OK" if dev < 1e-8 else "FAIL"))
assert dev < 1e-8

z6 = np.zeros(6)
net = f_faces[2::3, 0].sum()
wall_A = np.zeros(n_unique // 3)
np.add.at(wall_A, red_of[cells0[mat_e == 1].ravel()],
          np.repeat(area_e[mat_e == 1] / 4.0, 4))
dE1u = np.linalg.solve(C6, np.array([0.0, 0, 0, 1.0, 0, 0]))
_, Su, _ = dehom_fields(r, z6, dE1=dE1u, dE2=z6)
tau_e = np.asarray(Su).mean(axis=1)[:, 4]
print("gate: integral sigma_xz under unit Q1 = %.6f (target 1)"
      % ((tau_e * area_e).sum() / omega))
tau_w = np.zeros(n_unique // 3)
np.add.at(tau_w, red_of[cells0.ravel()],
          np.repeat(tau_e * area_e / 4.0, 4))
cols = {"rm": (np.asarray(V1Lt0), np.asarray(V2Lt0))}
for tag, w in (("walls", wall_A), ("tau", tau_w)):
    fv = f_faces.copy()
    fv[2::3, 0] -= net * w / w.sum()
    cols[tag] = qcol(fv)

# ---- the four fields, GLOBAL (E,6) element means + MATERIAL copies
kw_grad = dict(dE1=state["dE1"], dE2=state["dE2"],
               dE11=state["dE11"], dE12=state["dE12"],
               dE22=state["dE22"])


def field(qtag=None):
    if qtag is None:
        G_, S_, _ = dehom_fields(r, eps)                 # classical base
    else:
        r2 = dict(r)
        r2["V1Lt"], r2["V2Lt"] = cols[qtag]
        G_, S_, _ = dehom_fields(r2, eps, qt6=state["qt6"], **kw_grad)
    _, Sm = material_frame_fields(G_, S_, r)
    return (np.asarray(S_).mean(axis=1),                 # global, storage
            np.asarray(Sm).mean(axis=1)[:, PIDX])        # material, printed


F = {"cl": field(None), "rm": field("rm"), "walls": field("walls"),
     "tau": field("tau")}

# ---- path sampling + gates against the shipped curves
d = yaml.safe_load(open(YAML))
nd = np.asarray(d["nodes"], float)[:, :2]
cen = nd[np.asarray(d["cells"], int)].mean(axis=1)
tree = cKDTree(cen)
paths = {}
for n in (1, 2):
    C = np.loadtxt(os.path.join(OPG, "path_%d" % n,
                                "path_%d.coords" % n))
    dist, idx = tree.query(C[:, 1:3])
    assert dist.max() < 1e-4
    P = {k: F[k][1][idx] for k in F}
    P["abq"] = np.loadtxt(os.path.join(
        ABQD, "path_%d_abaqus.dat" % n))[:, 4:10]
    ship = np.loadtxt(os.path.join(OPG, "path_%d" % n,
                                   "path_%d_opensg.dat" % n))[:, 4:10]
    g = np.abs(P["rm"] - ship).max()
    print("gate: path %d RM-orig vs shipped path_%d_opensg.dat  %.1e %s"
          % (n, n, g, "OK" if g < 1e-6 else "FAIL"))
    shipc = np.loadtxt(os.path.join(OPG, "classical",
                                    "path_%d_classical.dat" % n))[:, 4:10]
    gc = np.abs(P["cl"] - shipc).max() / np.abs(shipc).max()
    print("gate: path %d classical base vs classical route  rel %.1e"
          " (small law-vs-ladder diff expected)" % (n, gc))
    paths[n] = (C, P)
    out = os.path.join(HERE, "path_%d_rmtau.dat" % n)
    with open(out, "w") as f:
        f.write("# MSG-RM recovery with the tau-reacted q column along"
                " path_%d -- MATERIAL frame, element means\n" % n)
        f.write("# %10s %12s %12s %4s" % ("s", "x1", "x2", "mat"))
        for c in COMP:
            f.write(" %13s" % c)
        f.write("\n")
        for i in range(len(C)):
            f.write("%12.6f %12.6f %12.6f %4d" % (C[i, 0], C[i, 1],
                                                  C[i, 2], int(C[i, 3])))
            for j in range(6):
                f.write(" %13.6e" % P["tau"][i, j])
            f.write("\n")
    print("wrote %s" % os.path.basename(out))

# ---- segment scores
lines = ["# per-material-segment RMS %% (of path max|3-D|), material"
         " frame\n# %-6s %-4s %-9s %8s %8s %8s %8s   %s\n"
         % ("path", "comp", "segment", "cl", "rm", "walls", "tau",
            "verdict")]
n_bad = {"rm": 0, "walls": 0, "tau": 0}
n_cells = 0
fix_targets = []
for n in (1, 2):
    C, P = paths[n]
    mt = C[:, 3].astype(int)
    for j, c in enumerate(COMP):
        den = max(np.abs(P["abq"][:, j]).max(), 1e-30)
        for m in sorted(set(mt)):
            k = mt == m
            row = {t: 100 * np.sqrt(np.mean(
                (P[t][k, j] - P["abq"][k, j]) ** 2)) / den
                for t in ("cl", "rm", "walls", "tau")}
            n_cells += 1
            verd = []
            for t in ("rm", "walls", "tau"):
                if row[t] > row["cl"] + 0.05:
                    n_bad[t] += 1
                    verd.append("cl<%s" % t)
            if "cl<rm" in verd:
                fix_targets.append((n, c, m, row))
            lines.append("%-8s %-4s %-9s %8.2f %8.2f %8.2f %8.2f   %s\n"
                         % ("path_%d" % n, c, MATL[m], row["cl"],
                            row["rm"], row["walls"], row["tau"],
                            ",".join(verd) or "-"))
open(os.path.join(HERE, "segment_rms.txt"), "w").writelines(lines)
print("".join(lines))
print("verdict: of %d (path, comp, segment) cells, classical beats"
      " -> shipped RM in %d, walls-reacted in %d, tau-reacted in %d"
      % (n_cells, n_bad["rm"], n_bad["walls"], n_bad["tau"]))

# (the bay-load figure moved to bay_fix_and_decompose.py: it needs the
#  per-row interpolation + per-bay parabola treatment, not raw columns)
np.save(os.path.join(HERE, "_fields_global.npy"),
        np.stack([F[t][0] for t in ("cl", "rm", "walls", "tau")]))
np.save(os.path.join(HERE, "_fields_material.npy"),
        np.stack([F[t][1] for t in ("cl", "rm", "walls", "tau")]))
print("wrote _fields_global/_material.npy (cl, rm, walls, tau)")

# ---- path-2 figures for every component where classical beat RM
done = set()
for n, c, m, row in fix_targets:
    if (n, c) in done:
        continue
    done.add((n, c))
    C, P = paths[n]
    s = C[:, 0]
    j = COMP.index(c)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(s, P["abq"][:, j], "-o", color="#c0392b", ms=3.2, lw=1.7,
            label="Reference 3-D FEA (Abaqus C3D20R)")
    ax.plot(s, P["rm"][:, j], "--s", color="k", ms=2.8, lw=1.6,
            label="Shear-refined (RM) Plate (OpenSG)")
    ax.plot(s, P["cl"][:, j], "-.^", color="#2471a3", ms=3.2, lw=1.4,
            label="Classical Plate (OpenSG)")
    ax.plot(s, P["tau"][:, j], ":*", color="#1e8449", ms=6.0, lw=1.5,
            label=r"RM, $\tau$-reacted q column")
    mchg = np.where(np.diff(C[:, 3].astype(int)) != 0)[0]
    for i in mchg:
        ax.axvline(0.5 * (s[i] + s[i + 1]), color="k", lw=0.7, ls=":",
                   alpha=0.6)
    ax.set_xlabel(r"$x_1$  (mm)" if n == 1 else
                  r"$s/s_{\mathrm{total}}$  (0 = top surface, 1 ="
                  r" bottom surface)")
    ax.set_ylabel(r"$\sigma_{%s}$  (MPa)" % c[1:])
    ax.grid(alpha=0.3)
    if n == 2:
        ax.set_xlim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    png = os.path.join(HERE, "path_%d_fix_%s.png" % (n, c))
    fig.savefig(png, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote %s" % os.path.basename(png))
print("end   : %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
