"""Two-step SHELL dehomogenization: beam FF -> RM shell section strains -> MSG-RM
through-thickness stress recovery (the production OpenSG-TW pipeline, msgrm_dehom route).

step 1 (section, opensg_shell.dehom_rm): st = C6^-1 FF and the RM warping recombination
  (_macro_fields), then per element the 6 shell strains [e11 e22 2e12 k11 k22 2k12]
  (_rm_shell_strain) plus the span/arc strain gradients dE1/dE2 (quad_ops_indep;
  arc gradient from LAYUP-BOUNDARY-AWARE nodal averaging -- no differencing across
  section changes).
step 2 (wall through-thickness): MSG-RM first-order plate recovery at depth z --
  msgrm_strain_at_depth imported from opensg_solid.rm_plate_1D.msg_rm_plate
  (shared with the plate/solid examples, NOT duplicated here).

# ----------------------------------------------------------------------------
# ALL VARIABLES USED IN THIS SCRIPT
# ----------------------------------------------------------------------------
#   name, ff_file       <name>_shell.yaml + the beam FF table (eta F1 F2 F3 M1 M2 M3,
#   station_row         VABS order, RM CENTER-ref); station_row 10 -> eta = 0.20
#   n_depth             through-thickness recovery points per element
#   stress_frame        "material" (ply axes, failure criteria) | "plate" (wall axes)
#   B, C6, FF           RM bundle, Timoshenko 6x6, beam force/moment vector
#   st                  (6,) macro section strains  C6^-1 FF
#   s6mid, dE1e, dE2e   (n_el, 6) shell strains + span/arc gradients at element mid-arc
#   warpM               rm_plate_msg through-thickness SG per layup (MSG G + warping)
#   pts, sig            recovery-point coords (y2, y3) and 3-D stress (MPa)
#   <name>_dehom_shell.txt, <name>_dehom_stress.png, <name>_cap_tt.png   the outputs
# ----------------------------------------------------------------------------
"""
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from opensg_shell import build_rm_bundle, _macro_fields, _rm_shell_strain
from opensg_shell.segment_indep import quad_ops_indep
from opensg_shell.fe_jax.msg_dehom import _macro_recovery
from opensg_solid.rm_plate_1D.msg_rm_plate import rm_plate_msg, msgrm_strain_at_depth

############### User Input #################################
name = "iea_s10"                  # reads <name>_shell.yaml
ff_file = "ff51_rmc_reform.dat"   # beam FF per station (# eta F1 F2 F3 M1 M2 M3)
station_row = 10                  # row 10 -> eta = r/R = 0.20 (iea_s10)
n_depth = 9                       # through-thickness recovery points per element
stress_frame = "material"         # "material" (ply axes) | "plate" (wall axes)
############################################################

t0 = time.perf_counter()
B = build_rm_bundle(name + "_shell.yaml")
C6 = np.asarray(B["Timo"])
FF = np.loadtxt(ff_file)[station_row, 1:]
eta = np.loadtxt(ff_file)[station_row, 0]
print("bundle built %.1fs (reference=%s, wall G=%s)" % (time.perf_counter() - t0,
                                                        B["ref"], B["g_source"]))
print("beam FF at eta=%.2f  [F1 F2 F3 M1 M2 M3] = %s" % (eta, np.array2string(FF, precision=4)))

# ================= step 1: RM shell section strains =================
st, st_m, aA, aB = _macro_fields(B, beam_force_vabs=FF)
_, _sm, st_cl1, st_cl2 = _macro_recovery(C6, np.linalg.inv(C6) @ FF)
print("macro section strains st = C6^-1 FF = %s" % np.array2string(np.asarray(st), precision=5))

corners = np.asarray(B["corners"]); rc = np.asarray(B["red_cells"]); cen = corners.mean(0)
nodes_s, quads_s, _hs = B["strip"]
n_el = rc.shape[0]; n_nd = int(rc.max()) + 1
layups = B["layup_per_elem"]; ldb = B["layup_db"]; mdb = B["material_db"]
frac = float(B.get("frac", 0.0))
hth = {ln: float(sum(i["thick"])) for ln, i in ldb.items()}

s6mid = np.zeros((n_el, 6)); dE1e = np.zeros((n_el, 6))
for e in range(n_el):
    s6, _ = _rm_shell_strain(B, e, 0.5, st_m, aA, aB)
    s6mid[e] = np.asarray(s6, float)
    Xe = nodes_s[quads_s[e]]; e3e = B["re3"][e]
    BDe, BDh, BDl, *_ = quad_ops_indep(Xe, e3e, 0.0, 0.0, float(B["k22"][e]),
                                       B["cross"], B["ax"])
    c0, c1 = int(rc[e, 0]), int(rc[e, 1])
    g = np.r_[c0 * 6:c0 * 6 + 6, c1 * 6:c1 * 6 + 6, c1 * 6:c1 * 6 + 6, c0 * 6:c0 * 6 + 6]
    dE1e[e] = np.asarray(BDe @ st_cl1 + BDh @ aB[g], float)

# layup-boundary-aware nodal average of s6 (only where two SAME-layup elements meet)
deg = np.zeros(n_nd, int)
nd_el = [[] for _ in range(n_nd)]
for e in range(n_el):
    for nd in (int(rc[e, 0]), int(rc[e, 1])):
        deg[nd] += 1; nd_el[nd].append(e)
s6n = np.full((n_nd, 6), np.nan)
for nd in range(n_nd):
    if deg[nd] == 2 and layups[nd_el[nd][0]] == layups[nd_el[nd][1]]:
        s6n[nd] = 0.5 * (s6mid[nd_el[nd][0]] + s6mid[nd_el[nd][1]])
T = corners[rc[:, 1]] - corners[rc[:, 0]]
Lel = np.sqrt((T ** 2).sum(1))
dE2e = np.zeros((n_el, 6))
for e in range(n_el):
    c0, c1 = int(rc[e, 0]), int(rc[e, 1])
    v0 = s6n[c0] if np.isfinite(s6n[c0, 0]) else s6mid[e]
    v1 = s6n[c1] if np.isfinite(s6n[c1, 0]) else s6mid[e]
    dE2e[e] = (v1 - v0) / Lel[e]

# inward wall normal per element (plate +z = OML -> IML)
tun = T / Lel[:, None]
nvec = np.column_stack([tun[:, 1], -tun[:, 0]])
emid = 0.5 * (corners[rc[:, 0]] + corners[rc[:, 1]])
flip = ((cen - emid) * nvec).sum(1) < 0
nvec[flip] *= -1.0

# ================= step 2: MSG-RM through-thickness recovery =================
t1 = time.perf_counter()
warpM = {ln: rm_plate_msg(i["thick"], i["angles"], i["mat_names"], mdb, fraction=frac)
         for ln, i in ldb.items()}
print("MSG-RM through-thickness SG per layup: %d layups, %.1fs" % (len(warpM),
                                                                   time.perf_counter() - t1))

t1 = time.perf_counter()
zeta = (np.arange(n_depth) + 0.5) / n_depth            # ply-interior stations, 0=OML..1=IML
pts = np.zeros((n_el * n_depth, 2)); sig = np.zeros((n_el * n_depth, 6))
tab = np.zeros((n_el * n_depth, 12))
for e in range(n_el):
    ln = layups[e]; h = hth[ln]
    c0, c1 = int(rc[e, 0]), int(rc[e, 1])
    s6 = s6mid[e].copy()
    for row in (2, 5):                                  # contour-derivative rows: nodal interp
        v0 = s6n[c0, row] if np.isfinite(s6n[c0, row]) else s6mid[e, row]
        v1 = s6n[c1, row] if np.isfinite(s6n[c1, row]) else s6mid[e, row]
        s6[row] = 0.5 * (v0 + v1)
    for k in range(n_depth):
        z = (zeta[k] - frac) * h                        # depth from the reference surface
        p = emid[e] + z * nvec[e]
        Gam, Sig, ply = msgrm_strain_at_depth(warpM[ln], z, s6, dE1e[e], dE2e[e],
                                              frame=stress_frame)
        i = e * n_depth + k
        pts[i] = p; sig[i] = np.asarray(Sig, float) / 1e6
        tab[i] = [e, p[0], p[1], z, zeta[k], ply, *sig[i]]
print("recovery at %d points (%d elements x %d depths): %.1fs"
      % (len(pts), n_el, n_depth, time.perf_counter() - t1))
imax = int(np.argmax(np.abs(sig[:, 0])))
print("peak |sigma11| = %.2f MPa at (y2, y3) = (%.3f, %.3f), zeta = %.2f  [%s frame]"
      % (abs(sig[imax, 0]), pts[imax, 0], pts[imax, 1], tab[imax, 4], stress_frame))

np.savetxt(name + "_dehom_shell.txt", tab,
           fmt="%6d %12.5e %12.5e %11.4e %6.3f %7.2f " + " ".join(["%12.5e"] * 6),
           header="two-step RM shell dehom, %s frame, eta=%.2f, FF=[%s]\n"
                  "elem  y2(m)  y3(m)  z(m,from %s ref)  zeta(0=OML,1=IML)  ply(deg)  "
                  "S11 S22 S33 S23 S13 S12 (MPa)"
                  % (stress_frame, eta, " ".join("%.4e" % v for v in FF), B["ref"]))

# ---- section stress cloud: sigma11 | sigma12 (MPa) ----
fig, ax = plt.subplots(2, 1, figsize=(9.5, 8.6))
for a, (ci, lab) in zip(ax, [(0, r"$\sigma_{11}$"), (5, r"$\sigma_{12}$")]):
    mlim = np.nanpercentile(np.abs(sig[:, ci]), 99) or 1e-9
    cs = a.scatter(pts[:, 0], pts[:, 1], c=np.clip(sig[:, ci], -mlim, mlim),
                   s=4, cmap="rainbow", vmin=-mlim, vmax=mlim)
    a.set_aspect("equal"); a.set_xlabel("y2 (m)"); a.set_ylabel("y3 (m)")
    cb = fig.colorbar(cs, ax=a, shrink=0.9, pad=0.02)
    cb.set_label(lab + " (MPa)")
fig.tight_layout()
fig.savefig(name + "_dehom_stress.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---- through-thickness line at the (top) spar-cap CENTRE element ----
# cap layup = the layup of the peak-|sigma11| element; centre = arc-middle of that
# set on the top surface (clean single-wall path, away from the web junctions)
lncap = layups[int(imax // n_depth)]
cap_els = [e for e in range(n_el) if layups[e] == lncap and emid[e, 1] > cen[1]]
ecap = cap_els[len(cap_els) // 2]
m = slice(ecap * n_depth, (ecap + 1) * n_depth)
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6))
for a, (ci, lab) in zip(ax, [(0, r"$\sigma_{11}$ (MPa)"), (5, r"$\sigma_{12}$ (MPa)")]):
    a.plot(zeta, sig[m, ci], "o-", ms=5.5, lw=1.4)
    a.set_xlabel(r"$\zeta$ (0=OML, 1=IML), element %d" % ecap)
    a.set_ylabel(lab); a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(name + "_cap_tt.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote %s_dehom_shell.txt (+ %s_dehom_stress.png, %s_cap_tt.png)" % (name, name, name))
