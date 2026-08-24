"""debug_decompose_path1.py -- WHY does the RM recovery under-predict
the in-plane oscillation along path 1?  The dehom is linear in its
drivers, so the recovered field splits EXACTLY into its chains:

    full = base + grad + q
    base = (Gamma_h V0 + Ge) eps          the classical-recovery term
    grad = the Eq. 63/64-66 strain-derivative addition (deps, d2eps)
    q    = the qt6 pressure load-ladder addition

Gates: (1) additivity of the split (linearity, 1e-9 of max|full|);
(2) the rebuilt full field must reproduce the stored
../dehomo/45pm_singleextrude_bd_dehom.SM digit-tight (same code, same
drivers -- a mismatch means the stored .SM is stale); (3) the usual
element-ordering / coords pairing gates of sm_to_dat.py.

The report then answers, per component along path 1: does the macro
mean match the 3-D (macro-state check); how damped is the fluctuation
(std ratio); WHICH chain's shape the residual (3-D - full) follows
(correlation), and what scale alpha on that chain would close it.

In:  ../sg_2d/45pm_singleextrude_bd.yaml,
     ../dehomo/45pm_singleextrude_bd.ff,
     ../dehomo/45pm_singleextrude_bd_dehom.SM,
     ../path_1/path_1.coords,
     ../../abaqus/station_path_stress/path_1_abaqus.dat
Out: ./rm_decomposition_path1.dat  (s x1 x2 mat + 6 x [abq full base
     grad q], printed order 11 22 33 12 13 23),
     ./path_1_decomp_S11/S22/S12.png, the printed report
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
                                   dehom_fields, gauss_coords,
                                   material_frame_fields)
from opensg_solid.sg_homo import plate_homo_2d             # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
YAML = os.path.join(HERE, "..", "sg_2d", "45pm_singleextrude_bd.yaml")
FF = os.path.join(HERE, "..", "dehomo", "45pm_singleextrude_bd.ff")
SM = os.path.join(HERE, "..", "dehomo",
                  "45pm_singleextrude_bd_dehom.SM")
COORDS = os.path.join(HERE, "..", "path_1", "path_1.coords")
ABQ = os.path.join(HERE, "..", "..", "abaqus", "station_path_stress",
                   "path_1_abaqus.dat")
PIDX = [SGIDX[c] for c in ORDER]           # storage -> printed order

print("start : %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# ---- homogenize (refined: the full RM ladder + load columns)
r = plate_homo_2d(YAML, refined=1, plot=False)
state = read_ff_state(FF)
eps = np.linalg.solve(np.asarray(r["C_eff"], float), state["FF"])
# eps sensitivity: the .out's own 6x6 (= the ladder A6) vs r["C_eff"]
eps_lad = np.linalg.solve(np.asarray(r["A6_ladder"], float),
                          state["FF"])
print("eps  (C_eff)    :", np.array2string(eps, precision=6))
print("eps  (A6_ladder):", np.array2string(eps_lad, precision=6))
print("eps rel diff    : %.2e"
      % (np.abs(eps - eps_lad).max() / np.abs(eps).max()))

# ---- the linear split
z6 = np.zeros(6)
kw_grad = dict(dE1=state["dE1"], dE2=state["dE2"],
               dE11=state["dE11"], dE12=state["dE12"],
               dE22=state["dE22"])
Gf, Sf, _ = dehom_fields(r, eps, Q=state["Q"], qt6=state["qt6"],
                         qb6=state["qb6"], **kw_grad)
Gb, Sb, _ = dehom_fields(r, eps)
Gg0, Sg0, _ = dehom_fields(r, eps, **kw_grad)
Gq, Sq, _ = dehom_fields(r, z6, qt6=state["qt6"], qb6=state["qb6"])
Gg, Sg = Gg0 - Gb, Sg0 - Sb

add = np.abs(Sf - (Sb + Sg + Sq)).max() / np.abs(Sf).max()
print("gate: additivity of the split  %.2e %s"
      % (add, "OK" if add < 1e-9 else "FAIL"))

# ---- material frame + printed order, per piece
pieces = {}
for tag, (G_, S_) in (("full", (Gf, Sf)), ("base", (Gb, Sb)),
                      ("grad", (Gg, Sg)), ("q", (Gq, Sq))):
    _, Sm = material_frame_fields(G_, S_, r)
    pieces[tag] = np.asarray(Sm)[:, :, PIDX]           # (E, Q, 6)

# ---- gate: the stored .SM (per-Gauss-point rows, printed order)
sm = np.loadtxt(SM)
full_rows = pieces["full"].reshape(-1, 6)
if sm.shape[0] != full_rows.shape[0]:
    raise SystemExit("stored .SM has %d rows, rebuilt field %d"
                     % (sm.shape[0], full_rows.shape[0]))
dev = np.abs(sm[:, 3:9] - full_rows).max()
rel = dev / np.abs(sm[:, 3:9]).max()
print("gate: rebuilt full vs stored .SM  max dev %.3e (rel %.1e) %s"
      % (dev, rel, "OK" if rel < 1e-6 else
         "FAIL -- the stored .SM is NOT this code + these drivers"))

# ---- element pairing along path 1 (the sm_to_dat gates)
d = yaml.safe_load(open(YAML))
nd = np.asarray(d["nodes"], float)[:, :2]
cells = np.asarray(d["cells"], int)
E = len(cells)
cen = nd[cells].mean(axis=1)
gxy = np.asarray(gauss_coords(r)).reshape(E, -1, 2).mean(axis=1)
off = np.abs(gxy - cen).max()
print("gate: element ordering (gauss vs yaml centroids)  %.2e %s"
      % (off, "OK" if off < 1e-6 else "FAIL"))

C = np.loadtxt(COORDS)
dist, idx = cKDTree(cen).query(C[:, 1:3])
print("gate: coords -> centroid pairing  %.2e %s"
      % (dist.max(), "OK" if dist.max() < 1e-4 else "FAIL"))
if max(off, add) > 1e-6 or dist.max() >= 1e-4:
    raise SystemExit("a gate failed -- do not trust the table below")

P = {t: pieces[t].mean(axis=1)[idx] for t in pieces}   # (n_path, 6)
A = np.loadtxt(ABQ)
if np.abs(A[:, :3] - C[:, :3]).max() >= 1e-4:
    raise SystemExit("path_1_abaqus.dat is not row-aligned with the"
                     " coords")
abq = A[:, 4:10]
s = C[:, 0]

# ---- the report
fl = lambda v: v - v.mean()                            # noqa: E731
print("\n---- path 1, material frame: chain decomposition per component")
print("%-4s %9s %9s | %8s %8s %6s | %6s %7s | %6s %7s | %7s %7s"
      % ("cmp", "mean3D", "meanRM", "fl3D", "flRM", "ratio",
         "c(r,q)", "alpha_q", "c(r,b)", "alpha_b", "rms%", "rms%aq"))
for j, c in enumerate(ORDER):
    a, f = abq[:, j], P["full"][:, j]
    b, g, q = P["base"][:, j], P["grad"][:, j], P["q"][:, j]
    res = a - f
    fa, ff = fl(a), fl(f)
    rq = fl(res); qf = fl(q); bf = fl(b + g)
    den = np.abs(a).max() + 1e-30
    cq = (np.corrcoef(rq, qf)[0, 1]
          if qf.std() > 1e-12 and rq.std() > 1e-12 else 0.0)
    cb = (np.corrcoef(rq, bf)[0, 1]
          if bf.std() > 1e-12 and rq.std() > 1e-12 else 0.0)
    aq = 1.0 + (rq @ qf) / max(qf @ qf, 1e-30)
    ab = 1.0 + (rq @ bf) / max(bf @ bf, 1e-30)
    rms0 = 100 * np.sqrt(np.mean(res ** 2)) / den
    res_aq = a - (b + g + aq * q)
    rmsq = 100 * np.sqrt(np.mean(res_aq ** 2)) / den
    print("S%-3s %9.4f %9.4f | %8.4f %8.4f %6.3f | %6.2f %7.3f |"
          " %6.2f %7.3f | %7.2f %7.2f"
          % (c, a.mean(), f.mean(), fa.std(), ff.std(),
             ff.std() / max(fa.std(), 1e-30), cq, aq, cb, ab,
             rms0, rmsq))

# joint in-plane alpha on the q chain (S11, S22, S12 stacked)
JP = [0, 1, 3]                       # printed order: 11 22 12
rq = np.concatenate([fl(abq[:, j] - P["full"][:, j]) for j in JP])
qf = np.concatenate([fl(P["q"][:, j]) for j in JP])
print("\njoint in-plane (S11,S22,S12): corr(residual, q-fluct) = %.3f,"
      " alpha_q = %.3f" % (np.corrcoef(rq, qf)[0, 1],
                           1.0 + (rq @ qf) / (qf @ qf)))
bf = np.concatenate([fl(P["base"][:, j] + P["grad"][:, j])
                     for j in JP])
print("joint in-plane (S11,S22,S12): corr(residual, base-fluct) = %.3f,"
      " alpha_base = %.3f" % (np.corrcoef(rq, bf)[0, 1],
                              1.0 + (rq @ bf) / (bf @ bf)))
print("fluctuation share along path 1 (std, S11): base+grad %.4f,"
      " q %.4f" % (np.std(fl(P["base"][:, 0] + P["grad"][:, 0])),
                   np.std(fl(P["q"][:, 0]))))

# ---- the table + debug figures
out = os.path.join(HERE, "rm_decomposition_path1.dat")
with open(out, "w") as f:
    f.write("# RM dehom chain decomposition along path_1 -- MATERIAL"
            " frame, element Gauss means\n")
    f.write("# full = base + grad + q (additivity %.1e); abq = the 3-D"
            " reference\n" % add)
    f.write("# %10s %12s %12s %4s" % ("s", "x1", "x2", "mat"))
    for c in ORDER:
        for t in ("abq", "full", "base", "grad", "q"):
            f.write(" %13s" % ("S%s_%s" % (c, t)))
    f.write("\n")
    for i in range(len(s)):
        f.write("%12.6f %12.6f %12.6f %4d"
                % (C[i, 0], C[i, 1], C[i, 2], int(C[i, 3])))
        for j in range(6):
            for arr in (abq, P["full"], P["base"], P["grad"], P["q"]):
                f.write(" %13.6e" % arr[i, j])
        f.write("\n")
print("\nwrote %s (%d rows)" % (os.path.basename(out), len(s)))

NICE = {"11": 0, "22": 1, "12": 3}
for c, j in NICE.items():
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(s, abq[:, j], "-", lw=1.8, color="#c0392b",
            label="Abaqus 3-D FEA")
    ax.plot(s, P["full"][:, j], "--", lw=1.8, color="k",
            label="MSG-RM full recovery")
    ax.plot(s, P["base"][:, j] + P["grad"][:, j], "-.", lw=1.4,
            color="#7f8c8d", label="base + gradient chains")
    ax.plot(s, P["q"][:, j], ":", lw=1.4, color="#2471a3",
            label="pressure (qt6) chain alone")
    ax.plot(s, abq[:, j] - P["full"][:, j], "-", lw=1.0,
            color="#27ae60", label="residual (3-D $-$ RM)")
    ax.set_xlabel(r"$x_1$  (mm)")
    ax.set_ylabel(r"$\sigma_{%s}$  (MPa)" % c)
    ax.grid(alpha=0.3)
    ax.axhline(0.0, color="k", lw=0.6, alpha=0.4)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    png = os.path.join(HERE, "path_1_decomp_S%s.png" % c)
    fig.savefig(png, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote %s" % os.path.basename(png))

print("end   : %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
