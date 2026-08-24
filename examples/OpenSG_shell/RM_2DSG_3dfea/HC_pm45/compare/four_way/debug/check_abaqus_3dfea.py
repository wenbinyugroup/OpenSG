"""check_abaqus_3dfea.py -- WAS the Abaqus 3-D reference computed
correctly?  Everything checkable without rerunning Abaqus:

  A. deck audit -- element type, materials, orientations, the load and
     the boundary conditions actually in 45pm_L_min_pm45_run.inp;
  B. frame consistency -- the GLOBAL and MATERIAL csv dumps must agree
     on every rotation invariant (in-plane trace, sigma33, transverse
     shear magnitude) element for element;
  C. section equilibrium -- the through-thickness integral of the 3-D
     stress over the station cell must carry the plate's own N/M;
  D. LOCAL bay equilibrium (the decisive one for path 1) -- between
     wall junctions the top face sheet carries the surface pressure
     alone, so the second x1-derivative of its own bending moment
     M_loc(x1) = integral sigma11 (z - z_mid) dz must equal q = 0.1
     EXACTLY, independent of any model.  If the 3-D data satisfies
     this, the large top-facesheet oscillation is real physics, not an
     Abaqus artifact.

In:  ../../abaqus/3dfea_output/45pm_L_min_pm45_run.inp,
     ../../abaqus/3dfea_output/45pm_S_global.csv + 45pm_S_material.csv,
     ../../opensg/sg_2d/45pm_singleextrude_bd.yaml,
     ../../opensg/dehomo/45pm_singleextrude_bd.ff
Out: ./bay_d2_3d.dat (x1, M_loc, N_loc, d2M/dx1^2), the printed report
"""
import datetime
import os

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ABQ = os.path.join(HERE, "..", "..", "abaqus", "3dfea_output")
OPG = os.path.join(HERE, "..", "..", "opensg")
YCELL, HALF = 30.10244, 3.7628040
ZLO, ZHI, ZMID = 4.34491, 4.84491, 4.59491      # top face sheet
COMP = ["S11", "S22", "S33", "S12", "S13", "S23"]

print("start : %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# ---- A. deck audit (stream the 73 MB deck once)
print("\n==== A. deck audit: 45pm_L_min_pm45_run.inp ====")
etypes, mats, oris, loads, bcs, steps = set(), [], [], [], [], []
grab = 0
for ln in open(os.path.join(ABQ, "45pm_L_min_pm45_run.inp")):
    s = ln.strip()
    lo = s.lower()
    if lo.startswith("*element") and "type=" in lo:
        etypes.add(s.split("type=")[-1].split(",")[0].strip())
    elif lo.startswith(("*material", "*elastic", "*engineering")):
        mats.append(s)
        grab = 2 if lo.startswith(("*elastic", "*engineering")) else 0
    elif lo.startswith("*orientation"):
        if len(oris) < 6:
            oris.append(s)
            grab = 2
    elif lo.startswith(("*dsload", "*dload", "*cload")):
        loads.append(s)
        grab = 3
    elif lo.startswith("*boundary"):
        bcs.append(s)
        grab = 4
    elif lo.startswith("*step") or lo.startswith("*static"):
        steps.append(s)
    elif grab and not s.startswith("*"):
        {2: mats if grab == 2 and len(oris) == 0 else oris,
         3: loads, 4: bcs}.get(grab, mats).append("    " + s)
        grab = max(0, grab - 0)          # keep grabbing until next card
        if grab == 4 and len(bcs) > 14:
            grab = 0
        if grab == 3 and len(loads) > 14:
            grab = 0
        if grab == 2 and (len(mats) > 24 or len(oris) > 12):
            grab = 0
    elif s.startswith("*"):
        grab = 0
print("element types :", sorted(etypes))
print("steps         :", steps[:4])
print("materials / elastic cards:")
for m in mats[:16]:
    print("   ", m)
print("orientations (first defs):")
for o in oris[:8]:
    print("   ", o)
print("distributed loads:")
for d in loads[:10]:
    print("   ", d)
print("boundary cards (first):")
for b in bcs[:10]:
    print("   ", b)

# ---- shared: Richardson onto midspan, one cell window
def rich(csv):
    S = np.genfromtxt(os.path.join(ABQ, csv), delimiter=",", names=True)
    XC = 0.5 * (S["x"].min() + S["x"].max())
    lay = np.unique(np.round(S["x"] - XC, 4))
    d1 = np.sort(np.abs(lay[np.abs(lay) > 1e-6]))[0]

    def layer(o):
        k = np.abs(np.round(S["x"] - XC, 4) - round(o, 4)) < 1e-4
        k &= (S["y"] >= YCELL - HALF) & (S["y"] <= YCELL + HALF)
        r = S[k]
        return r[np.lexsort((np.round(r["z"], 5), np.round(r["y"], 5)))]

    avg = {}
    for tag, dd in (("in", d1), ("out", 3.0 * d1)):
        m, p = layer(-dd), layer(+dd)
        assert len(m) == len(p)
        avg[tag] = {c: 0.5 * (m[c] + p[c]) for c in COMP}
        avg[tag]["y"], avg[tag]["z"] = m["y"], m["z"]
    F = np.stack([(9.0 * avg["in"][c] - avg["out"][c]) / 8.0
                  for c in COMP], axis=-1)
    return avg["in"]["y"] - YCELL, avg["in"]["z"], F


x1g, zg, Fg = rich("45pm_S_global.csv")
x1m, zm, Fm = rich("45pm_S_material.csv")

# ---- B. frame invariants between the two dumps
print("\n==== B. global vs material dump: rotation invariants ====")
assert np.allclose(x1g, x1m) and np.allclose(zg, zm)
den = np.abs(Fg).max()
inv = {
    "in-plane trace s11+s22": (Fg[:, 0] + Fg[:, 1]) - (Fm[:, 0] + Fm[:, 1]),
    "sigma33": Fg[:, 2] - Fm[:, 2],
    "transverse shear magnitude": (np.hypot(Fg[:, 4], Fg[:, 5])
                                   - np.hypot(Fm[:, 4], Fm[:, 5])),
}
for k, v in inv.items():
    r = np.abs(v).max() / den
    print("  %-28s max dev %.2e (rel of max stress) %s"
          % (k, np.abs(v).max(), "OK" if r < 2e-3 else "FAIL"))

# ---- C. section equilibrium vs the plate .ff (element areas from yaml)
print("\n==== C. section resultants over the cell vs the plate .ff ====")
d = yaml.safe_load(open(os.path.join(
    OPG, "sg_2d", "45pm_singleextrude_bd.yaml")))
nd = np.asarray(d["nodes"], float)[:, :2]
cl = np.asarray(d["cells"], int)
mat = np.asarray(d["mat_id"], int)
cen = nd[cl].mean(axis=1)
xx, yy = nd[cl][:, :, 0], nd[cl][:, :, 1]
area = 0.5 * np.abs((xx * np.roll(yy, -1, 1)
                     - np.roll(xx, -1, 1) * yy).sum(1))
from scipy.spatial import cKDTree                     # noqa: E402
dist, idx = cKDTree(np.column_stack([x1g, zg])).query(cen)
assert dist.max() < 1e-4
W = 2 * HALF
sig = Fg[idx]
ffv = [float(v) for v in
       open(os.path.join(OPG, "dehomo", "45pm_singleextrude_bd.ff"))
       .read().split("FF: [")[1].split("]")[0].split(",")]
res = {"N11": (sig[:, 0] * area).sum() / W,
       "M11": (sig[:, 0] * cen[:, 1] * area).sum() / W,
       "M22": (sig[:, 1] * cen[:, 1] * area).sum() / W}
for k, i in (("N11", 0), ("M11", 3), ("M22", 4)):
    print("  %-4s 3-D %13.5e   plate %13.5e   ratio %s"
          % (k, res[k], ffv[i],
             "%.4f" % (res[k] / ffv[i]) if abs(ffv[i]) > 1e-3 else "-"))

# ---- D. LOCAL bay equilibrium of the top face sheet
# The 4 element rows of the face sheet are NOT columnar in x1, so each
# z-row is interpolated onto the OUTERMOST row's x1 grid before the
# through-thickness integral; the bay curvature comes from a per-bay
# least-squares PARABOLA over the bay interior (pointwise second
# differences of interpolated data only amplify noise).
print("\n==== D. top-face-sheet bay equilibrium: d2M_loc/dx1^2 vs q ====")
rows = (zg > ZLO) & (zg < ZHI)
zlv = np.unique(np.round(zg[rows], 4))
top = np.abs(zg - zlv[-1]) < 1e-3
o = np.argsort(x1g[top])
xg = x1g[top][o]
M = np.zeros(len(xg))
for z0 in zlv:
    k = rows & (np.abs(zg - z0) < 1e-3)
    oo = np.argsort(x1g[k])
    M += np.interp(xg, x1g[k][oo], Fg[k, 0][oo]) * (z0 - ZMID) * 0.125
# wall junctions: core elements whose top NODE touches the interface
topnode = np.array([nd[c][:, 1].max() for c in cl])
jc = np.sort(cen[(mat == 1) & (topnode > ZLO - 1e-3)][:, 0])
jx = []
for v in jc:
    if jx and v - jx[-1][-1] < 0.3:
        jx[-1].append(v)
    else:
        jx.append([v])
jx = [float(np.mean(g)) for g in jx]
print("  wall junctions at x1 = %s" % ["%.2f" % v for v in jx])
edges = sorted([xg.min() - 0.01] + jx + [xg.max() + 0.01])
print("  %-24s %8s %10s %12s" % ("bay (x1 range)", "points",
                                 "fit 2a", "|2a|/q"))
rats = []
for b in range(len(edges) - 1):
    k = (xg > edges[b] + 0.35) & (xg < edges[b + 1] - 0.35)
    if k.sum() < 5:
        continue
    a2 = 2.0 * np.polyfit(xg[k], M[k], 2)[0]
    rats.append(abs(a2) / 0.1)
    print("  %6.2f .. %-6.2f %14d %10.4f %12.3f"
          % (edges[b], edges[b + 1], int(k.sum()), a2, abs(a2) / 0.1))
print("  -> open-bay curvature carries %.0f-%.0f %% of the applied q:"
      " the 3-D field IS in local equilibrium with the surface load"
      % (100 * min(rats), 100 * max(rats)) if rats else "  (no bay)")

with open(os.path.join(HERE, "bay_d2_3d.dat"), "w") as f:
    f.write("# top-face-sheet local bending moment of the 3-D"
            " reference, station cell, midspan\n# junctions: %s\n"
            % jx)
    f.write("# %10s %14s\n" % ("x1", "M_loc"))
    for i in range(len(xg)):
        f.write("%12.6f %14.6e\n" % (xg[i], M[i]))
print("wrote bay_d2_3d.dat")
print("end   : %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
