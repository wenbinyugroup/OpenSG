"""IEA-22 blade, r/R = 0.2 -- STEP 2: TWO-STEP dehomogenization, second order
(V2), benchmarked against VABS.

    beam FF  --(RM shell beam model, Timo 6x6)-->  per-element PLATE reaction
    forces + strains  --(opensg_solid RM plate SG)-->  3-D stress at depth

step 2a (section, opensg_shell): st = C6^-1 FF, the RM warping recombination
  (_macro_fields), then per element the 6 shell strains
  s6 = [e11 e22 2e12 k11 k22 2k12] (_rm_shell_strain) and the corresponding
  PLATE REACTION FORCES  F6 = A6 s6 = [N11 N22 N12 M11 M22 M12]  from that
  wall's own MSG ABD -- the interface quantity handed to the plate SG.
step 2b (wall, opensg_solid.rm_plate_1D): MSG-RM recovery at depth z with the
  SECOND-ORDER (V2, Yu Eq. 66) driver set.  The strain gradients come from the
  NEIGHBOURING ELEMENTS of the ring:
      dE1  span      = BDe st_cl1 + BDh aB      (beam macro recovery ladder)
      dE2  arc       = layup-boundary-aware nodal differencing of s6 between
                       the two elements sharing each node (never across a
                       section/layup change -- that jump is physical)
      dE11 span-span = BDe st_cl2               (the second macro rung)
      dE12 span-arc  = arc-derivative of dE1
      dE22 arc-arc   = arc-derivative of dE2
  V2 carries the transverse pair sigma_33 / sigma_13 that first order cannot.

# ----------------------------------------------------------------------------
# ALL VARIABLES USED IN THIS SCRIPT
# ----------------------------------------------------------------------------
#   name, ff_file, station_row   <name>_shell.yaml, the BeamDyn FF table
#                       (# eta F1 F2 F3 M1 M2 M3, RM CENTER-ref, VABS order),
#                       row 10 -> eta = r/R = 0.20
#   vabs_sm, jpath      the VABS .SM reference cloud and the spar-cap junction
#                       thickness path (both r=0.2)
#   B, C6, FF, st       RM bundle, Timo 6x6, beam forces, macro section strains
#   st_cl1, st_cl2      the 1st/2nd classical macro-strain rungs (_macro_recovery)
#   s6mid (n_el,6)      per-element shell strains at mid-arc
#   F6mid (n_el,6)      per-element PLATE REACTION FORCES A6 s6  [N/m, N]
#   dE1e/dE2e           first strain gradients (span / arc)
#   dE11e/dE12e/dE22e   second strain gradients (V2 drivers)
#   s6n (n_nd,6)        nodal-averaged s6 (NaN where layups differ)
#   warpM               per-layup rm_plate_msg through-thickness SG (MSG G,
#                       warping ladder V0/V1/V2 + the V2L load columns)
#   elg/xig/zdg         VABS gauss points projected onto the ring
#   is_web, zoff        web elements and their ply-registration z offset
#   sM1/sM2/sVg         first-order / V2 / VABS stress at the .SM cloud [MPa]
#   OUT   <name>_dehom_v2.txt, <name>_elem_plate_forces.dat,
#         <name>_report.txt, <name>_v2_section.png, <name>_junction_v2.png,
#         <name>_parity.png, <name>_dehom_v2.npz
# ----------------------------------------------------------------------------
"""
import os
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from opensg_shell import build_rm_bundle, _macro_fields, _rm_shell_strain
from opensg_shell.sg_assembly import quad_ops_indep
from opensg_shell.fe_jax.msg_dehom import _macro_recovery
from opensg_solid.rm_plate_1D.msg_rm_plate import (
    rm_plate_msg, msgrm_strain_at_depth, msgrm_strain_at_depth_batch)

############### User Input #################################
name = "iea_s10"                     # IEA-22 station at r/R = 0.20
ff_file = "ff51_rmc_reform.dat"      # BeamDyn FF (RM center-ref, VABS order)
station_row = 10                     # row 10 -> eta = 0.20
vabs_sm = "iea_s10.sg.SM"            # the VABS 2-D solid reference cloud
# Through-thickness paths: two are GENERATED from the ring itself (clean
# single walls -- spar-cap centre and a web), so they are frame-consistent
# with the recovery and the VABS cloud by construction.  The station's own
# reference path is added when present; its printed nearest-VABS distance is
# the frame check (mm-level = same frame, 100s of mm = a foreign mesh frame).
jpath_file = "iea_s10.lp_sparcap_left_thickness.coords"
############################################################

t0 = time.perf_counter()
B = build_rm_bundle(name + "_shell.yaml")
C6 = np.asarray(B["Timo"])
ffall = np.loadtxt(ff_file)
FF = ffall[station_row, 1:]
eta = ffall[station_row, 0]
print("bundle %.1f s (ref=%s, wall G=%s); FF at eta=%.2f = %s"
      % (time.perf_counter() - t0, B["ref"], B["g_source"], eta,
         np.array2string(FF, precision=4)))

# ================= step 2a: beam -> section strains + PLATE FORCES ==========
st, st_m, aA, aB = _macro_fields(B, beam_force_vabs=FF)
_, _sm, st_cl1, st_cl2 = _macro_recovery(C6, np.linalg.inv(C6) @ FF)
print("macro strains st = C6^-1 FF = %s" % np.array2string(np.asarray(st), precision=5))

corners = np.asarray(B["corners"]); rc = np.asarray(B["red_cells"])
cen = corners.mean(0)
nodes_s, quads_s, _hs = B["strip"]
n_el = rc.shape[0]; n_nd = int(rc.max()) + 1
layups = B["layup_per_elem"]; ldb = B["layup_db"]; mdb = B["material_db"]
frac = float(B.get("frac", 0.0))
hth = {ln: float(sum(i["thick"])) for ln, i in ldb.items()}

A0 = corners[rc[:, 0]]; T = corners[rc[:, 1]] - corners[rc[:, 0]]
L2 = (T ** 2).sum(1); Lel = np.sqrt(L2); tun = T / Lel[:, None]
nvec = np.column_stack([tun[:, 1], -tun[:, 0]])
emid = 0.5 * (corners[rc[:, 0]] + corners[rc[:, 1]])
flip = ((cen - emid) * nvec).sum(1) < 0
nvec[flip] *= -1.0

s6mid = np.zeros((n_el, 6)); dE1e = np.zeros((n_el, 6)); dE11e = np.zeros((n_el, 6))
for e in range(n_el):
    s6, _ = _rm_shell_strain(B, e, 0.5, st_m, aA, aB)
    s6mid[e] = np.asarray(s6, float)
    Xe = nodes_s[quads_s[e]]; e3e = B["re3"][e]
    BDe, BDh, BDl, *_ = quad_ops_indep(Xe, e3e, 0.0, 0.0, float(B["k22"][e]),
                                       B["cross"], B["ax"])
    c0, c1 = int(rc[e, 0]), int(rc[e, 1])
    g = np.r_[c0 * 6:c0 * 6 + 6, c1 * 6:c1 * 6 + 6, c1 * 6:c1 * 6 + 6, c0 * 6:c0 * 6 + 6]
    dE1e[e] = np.asarray(BDe @ st_cl1 + BDh @ aB[g], float)
    # span-span rung: the macro second recovery dominates for the constant-force
    # blade state (the warping-gradient counterpart needs the next warping order)
    dE11e[e] = np.asarray(BDe @ st_cl2, float)

# ---- NEIGHBOURING-ELEMENT arc derivative (layup-boundary aware) ----
deg = np.zeros(n_nd, int)
nd_el = [[] for _ in range(n_nd)]
for e in range(n_el):
    for nd in (int(rc[e, 0]), int(rc[e, 1])):
        deg[nd] += 1; nd_el[nd].append(e)


def arc_derivative(field_e):
    """d/ds of an elementwise 6-field from its NEIGHBOURING elements: nodal
    average only where exactly two SAME-layup elements meet (a layup change
    is a physical jump, never differenced across), then the element slope.
    Out: (n_el,6) derivative, (n_nd,6) nodal values (NaN where undefined)."""
    fn = np.full((n_nd, 6), np.nan)
    for nd in range(n_nd):
        if deg[nd] == 2 and layups[nd_el[nd][0]] == layups[nd_el[nd][1]]:
            fn[nd] = 0.5 * (field_e[nd_el[nd][0]] + field_e[nd_el[nd][1]])
    out = np.zeros((n_el, 6))
    for e in range(n_el):
        c0, c1 = int(rc[e, 0]), int(rc[e, 1])
        v0 = fn[c0] if np.isfinite(fn[c0, 0]) else field_e[e]
        v1 = fn[c1] if np.isfinite(fn[c1, 0]) else field_e[e]
        out[e] = (v1 - v0) / Lel[e]
    return out, fn


dE2e, s6n = arc_derivative(s6mid)
dE12e, _ = arc_derivative(dE1e)
dE22e, _ = arc_derivative(dE2e)
print("gradient rms: |E| %.3e |E,1| %.3e |E,2| %.3e |E,11| %.3e |E,12| %.3e |E,22| %.3e"
      % tuple(float(np.sqrt(np.mean(x ** 2)))
              for x in (s6mid, dE1e, dE2e, dE11e, dE12e, dE22e)))

# ---- per-layup through-thickness SG (MSG G + the V0/V1/V2 warping ladder) ----
t1 = time.perf_counter()
warpM = {ln: rm_plate_msg(i["thick"], i["angles"], i["mat_names"], mdb, fraction=frac)
         for ln, i in ldb.items()}
print("per-layup plate SG: %d layups, %.1f s" % (len(warpM), time.perf_counter() - t1))

# ---- the INTERFACE quantity: per-element plate reaction forces F6 = A6 s6 ----
F6mid = np.zeros((n_el, 6))
for e in range(n_el):
    F6mid[e] = np.asarray(warpM[layups[e]]["A6"], float) @ s6mid[e]
np.savetxt(name + "_elem_plate_forces.dat",
           np.column_stack([np.arange(n_el), emid, s6mid, F6mid]),
           fmt="%6d " + " ".join(["%13.6e"] * 14),
           header="per-element PLATE REACTION FORCES from the RM shell beam model"
                  " (Timo), eta=%.2f\nelem  y2(m) y3(m)  e11 e22 2e12 k11 k22 2k12"
                  "  N11 N22 N12(N/m) M11 M22 M12(N)" % eta)

# ================= VABS reference + projection =================
dsm = np.loadtxt(vabs_sm, skiprows=2)
sm_xy = dsm[:, :2]
sVg = dsm[:, 2:8][:, [0, 3, 5, 4, 2, 1]] / 1e6      # -> (11,22,33,23,13,12) MPa


def fast_project(P, chunk=4000):
    """Nearest ring element + local xi + signed wall depth for each point."""
    P = np.atleast_2d(np.asarray(P, float))
    el = np.zeros(len(P), int); xi = np.zeros(len(P)); zd = np.zeros(len(P))
    for i0 in range(0, len(P), chunk):
        Pc = P[i0:i0 + chunk]
        dp = Pc[:, None, :] - A0[None, :, :]
        tpar = np.clip((dp * T[None]).sum(2) / L2[None], 0.0, 1.0)
        foot = A0[None] + tpar[..., None] * T[None]
        d2 = ((Pc[:, None, :] - foot) ** 2).sum(2)
        e = np.argmin(d2, 1); r_ = np.arange(len(Pc))
        el[i0:i0 + chunk] = e
        xi[i0:i0 + chunk] = tpar[r_, e]
        zd[i0:i0 + chunk] = ((Pc - foot[r_, e]) * nvec[e]).sum(1)
    return el, xi, zd


elg, xig, zdg = fast_project(sm_xy)

# web elements (straight chords between junctions) + their ply registration
adjm = {}
for e, (a, b) in enumerate(rc):
    adjm.setdefault(a, []).append((b, e)); adjm.setdefault(b, []).append((a, e))
junc = set(np.where(deg >= 3)[0])
is_web = np.zeros(n_el, bool); seen = set()
for j in junc:
    for (nxt, e0) in adjm[j]:
        if e0 in seen:
            continue
        chain, prev, cur = [e0], j, nxt
        seen.add(e0)
        while cur not in junc and deg[cur] == 2:
            (n1, e1), (n2_, e2) = adjm[cur][0], adjm[cur][1]
            nn2, ee = (n1, e1) if n1 != prev else (n2_, e2)
            if ee in seen:
                break
            chain.append(ee); seen.add(ee)
            prev, cur = cur, nn2
        if cur in junc:
            arc = sum(Lel[c] for c in chain)
            cv = corners[cur] - corners[j]; ch = float(np.linalg.norm(cv))
            if ch / max(arc, 1e-30) > 0.99 and abs(cv[1]) / max(ch, 1e-30) > 0.6:
                is_web[chain] = True
zoff = np.zeros(n_el)
for e in np.where(is_web)[0]:
    zz = zdg[elg == e]
    if len(zz) >= 8:
        zoff[e] = 0.5 * (np.percentile(zz, 2) + np.percentile(zz, 98))


def recover(ip_el, ip_xi, ip_z, second):
    """MSG-RM stress (Pa, PLY MATERIAL frame -- the core default) at one point."""
    e = ip_el; ln = layups[e]
    c0, c1 = int(rc[e, 0]), int(rc[e, 1])
    s6 = s6mid[e].copy()
    for row in (2, 5):                       # contour-derivative rows: nodal interp
        v0 = s6n[c0, row] if np.isfinite(s6n[c0, row]) else s6mid[e, row]
        v1 = s6n[c1, row] if np.isfinite(s6n[c1, row]) else s6mid[e, row]
        s6[row] = (1.0 - ip_xi) * v0 + ip_xi * v1
    kw = dict(dE11=dE11e[e], dE12=dE12e[e], dE22=dE22e[e]) if second else {}
    Gam, Sig, ply = msgrm_strain_at_depth(warpM[ln], ip_z, s6, dE1e[e], dE2e[e], **kw)
    return np.asarray(Sig, float)


# ================= full section: first order vs V2 vs VABS =================
# BATCHED: the recovery is grouped by layup and evaluated with
# msgrm_strain_at_depth_batch (identical algebra to the scalar call, gated
# to ~5e-15).  The per-point Python loop over ~1.7e5 points was the whole
# runtime of this script; grouping collapses it to a handful of einsums.
t1 = time.perf_counter()
P = len(sm_xy)
zP = zdg - zoff[elg]
# per-point drivers: element strains, with the two contour-derivative rows
# interpolated along the arc exactly as the scalar `recover` does
s6P = s6mid[elg].copy()
for row in (2, 5):
    v0 = s6n[rc[elg, 0], row]; v1 = s6n[rc[elg, 1], row]
    v0 = np.where(np.isfinite(v0), v0, s6mid[elg, row])
    v1 = np.where(np.isfinite(v1), v1, s6mid[elg, row])
    s6P[:, row] = (1.0 - xig) * v0 + xig * v1
sM1 = np.zeros((P, 6)); sM2 = np.zeros((P, 6))
lay_of_pt = np.array([layups[e] for e in elg])
for ln in set(layups):
    m = np.where(lay_of_pt == ln)[0]
    if not len(m):
        continue
    _, S1, _ = msgrm_strain_at_depth_batch(warpM[ln], zP[m], s6P[m],
                                           dE1e[elg[m]], dE2e[elg[m]])
    _, S2, _ = msgrm_strain_at_depth_batch(warpM[ln], zP[m], s6P[m],
                                           dE1e[elg[m]], dE2e[elg[m]],
                                           dE11e[elg[m]], dE12e[elg[m]],
                                           dE22e[elg[m]])
    sM1[m] = S1; sM2[m] = S2
sM1 /= 1e6; sM2 /= 1e6
print("recovery at %d VABS gauss points, both orders: %.1f s (batched)"
      % (P, time.perf_counter() - t1))

webp = is_web[elg]
rms = lambda x: float(np.sqrt(np.mean(np.asarray(x) ** 2)))
names = ["s11", "s22", "s33", "s23", "s13", "s12"]
rep = ["=== IEA-22 r/R = 0.20 : two-step RM shell->plate dehom vs VABS ===",
       "beam FF (VABS order) = %s" % np.array2string(FF, precision=4),
       "Timo diag [EA GA2 GA3 GJ EI2 EI3] = %s"
       % np.array2string(np.diag(C6), precision=4),
       "", "rms difference vs VABS [MPa]   WEB / SKIN:"]
for ci in (0, 1, 5, 2, 4, 3):
    rep.append("  %-4s  first %8.3f / %8.3f   V2 %8.3f / %8.3f   (VABS rms %8.3f / %8.3f)"
               % (names[ci],
                  rms(sVg[webp, ci] - sM1[webp, ci]), rms(sVg[~webp, ci] - sM1[~webp, ci]),
                  rms(sVg[webp, ci] - sM2[webp, ci]), rms(sVg[~webp, ci] - sM2[~webp, ci]),
                  rms(sVg[webp, ci]), rms(sVg[~webp, ci])))

# what the SECOND-ORDER rung actually adds (its own magnitude, not a rounded
# rms difference): if these are ~0 the V2 drivers are inert at this state
rep += ["", "V2 payload -- what the second order changes [MPa]:"]
for ci in (0, 1, 5, 2, 4, 3):
    d = sM2[:, ci] - sM1[:, ci]
    rep.append("  %-4s  rms %10.3e   max %10.3e   (first-order rms %9.3e,"
               " VABS rms %9.3e)"
               % (names[ci], rms(d), float(np.abs(d).max()),
                  rms(sM1[:, ci]), rms(sVg[:, ci])))
print("\n".join(rep), flush=True)

# ================= through-thickness paths, generated FROM THE RING =========
# The path is built from the ring geometry itself (element mid-arc + the wall
# normal), so it is frame-consistent with both the recovery and the VABS cloud
# by construction -- an external coords file from another mesh is not.
# Two walls: the top spar-cap (thick, in-plane dominated) and a web element
# (where the transverse pair sigma_33 / sigma_13 lives).
icap = int(np.argmax(np.abs(sM2[:, 0])))
ecap_layup = layups[elg[icap]]
cap_els = [e for e in range(n_el)
           if layups[e] == ecap_layup and emid[e, 1] > cen[1] and not is_web[e]]
web_els = list(np.where(is_web)[0])
paths = [("spar cap (top, elem %d)" % (cap_els[len(cap_els) // 2]),
          cap_els[len(cap_els) // 2])]
if web_els:
    paths.append(("shear web (elem %d)" % web_els[len(web_els) // 2],
                  web_els[len(web_els) // 2]))

# plus the reference thickness path shipped with the station, when present
# (its points are already in the section frame -- verify with the printed
# nearest-VABS distance; a large value means a foreign mesh frame)
if os.path.isfile(jpath_file):
    paths.append(("cap-left junction (reference path)", jpath_file))

jrep = []
pathdata = {}
fig, ax = plt.subplots(len(paths), 3, figsize=(15.0, 4.4 * len(paths)),
                       squeeze=False)
for irow, (ptag, spec) in enumerate(paths):
    if isinstance(spec, str):                       # file-based path
        jc = np.loadtxt(spec)
        pxy = jc[:, :2]
        pel, pxi, pzd = fast_project(pxy)
        pz = pzd - zoff[pel]
        zeta = jc[:, 2] if jc.shape[1] > 2 else np.arange(len(pxy), float)
        ylab = "thickness position [mm]"
    else:                                           # ring-generated path
        epath = spec
        h = hth[layups[epath]]
        zeta = np.linspace(0.02, 0.98, 40)          # 0 = OML, 1 = IML
        zline = (zeta - frac) * h
        pxy = emid[epath] + (zline + zoff[epath])[:, None] * nvec[epath]
        pel = np.full(len(pxy), epath, int)
        pxi = np.full(len(pxy), 0.5)
        pz = zline
        ylab = r"$\zeta$  (0 = OML, 1 = IML)"
    d2near = ((sm_xy[None, :, :] - pxy[:, None, :]) ** 2).sum(2)
    near = np.argmin(d2near, 1)
    dist = np.sqrt(d2near[np.arange(len(pxy)), near])
    sJ1 = np.zeros((len(pxy), 6)); sJ2 = np.zeros((len(pxy), 6))
    for k in range(len(pxy)):
        sJ1[k] = recover(int(pel[k]), float(pxi[k]), pz[k], False) / 1e6
        sJ2[k] = recover(int(pel[k]), float(pxi[k]), pz[k], True) / 1e6
    sJV = sVg[near]
    pathdata[ptag] = (zeta, sJ1, sJ2, sJV, dist)
    jrep += ["", "=== through-thickness path: %s (%d pts, nearest VABS"
             " gauss %.2f mm) ===" % (ptag, len(pxy), 1e3 * dist.max()),
             "rms difference vs VABS along the path [MPa]:"]
    for ci in (0, 1, 5, 2, 4, 3):
        jrep.append("  %-4s  first %8.3f   V2 %8.3f   (VABS rms %8.3f)"
                    % (names[ci], rms(sJV[:, ci] - sJ1[:, ci]),
                       rms(sJV[:, ci] - sJ2[:, ci]), rms(sJV[:, ci])))
    for a_, ci in zip(ax[irow], (0, 4, 2)):
        a_.plot(sJV[:, ci], zeta, "k-", lw=2, label="VABS 2-D solid")
        a_.plot(sJ1[:, ci], zeta, "s--", color="#1f77b4", ms=4, label="RM first order")
        a_.plot(sJ2[:, ci], zeta, "o-", color="#ff7f0e", ms=4, mfc="none",
                label="RM V2 (Eq. 66)")
        a_.set_xlabel(r"$\sigma_{%s}$ [MPa]   %s"
                      % ({0: "11", 4: "13", 2: "33"}[ci], ptag))
        a_.set_ylabel(ylab)
        a_.grid(alpha=0.3)
    ax[irow][0].legend(frameon=False, fontsize=10)
print("\n".join(jrep), flush=True)
fig.tight_layout()
fig.savefig(name + "_thickness_paths.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ================= section contours: V2 vs VABS =================
fig, ax = plt.subplots(4, 2, figsize=(15.5, 12.5))
for row, ci in enumerate((0, 5, 4, 2)):
    mlim = np.nanpercentile(np.abs(sVg[:, ci]), 99) or 1e-9
    for col, (S, tag) in enumerate([(sVg, "VABS 2-D solid"),
                                    (sM2, "RM V2 (shell->plate)")]):
        a_ = ax[row, col]
        cs = a_.scatter(sm_xy[:, 0], sm_xy[:, 1], c=np.clip(S[:, ci], -mlim, mlim),
                        s=2.5, cmap="rainbow", vmin=-mlim, vmax=mlim)
        a_.set_aspect("equal"); a_.set_xlabel("$y_2$ [m]"); a_.set_ylabel("$y_3$ [m]")
        cb = fig.colorbar(cs, ax=a_, shrink=0.85, pad=0.02)
        cb.set_label(r"$\sigma_{%s}$ [MPa]  %s"
                     % (names[ci][1:], tag))
fig.tight_layout()
fig.savefig(name + "_v2_section.png", dpi=140, bbox_inches="tight")
plt.close(fig)

# ================= parity: RM vs VABS, first order and V2 =================
fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.6))
for a_, ci in zip(ax, (0, 5, 4)):
    lim = np.nanpercentile(np.abs(sVg[:, ci]), 99.5)
    a_.plot([-lim, lim], [-lim, lim], "k-", lw=1)
    a_.plot(sVg[:, ci], sM1[:, ci], ".", ms=1.6, color="#1f77b4", label="first order")
    a_.plot(sVg[:, ci], sM2[:, ci], ".", ms=1.6, color="#ff7f0e", label="V2")
    a_.set_xlim(-lim, lim); a_.set_ylim(-lim, lim)
    a_.set_xlabel(r"VABS $\sigma_{%s}$ [MPa]" % names[ci][1:])
    a_.set_ylabel(r"RM $\sigma_{%s}$ [MPa]" % names[ci][1:])
    a_.grid(alpha=0.3)
ax[0].legend(frameon=False, markerscale=8)
fig.tight_layout()
fig.savefig(name + "_parity.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ================= outputs =================
np.savetxt(name + "_dehom_v2.txt",
           np.column_stack([sm_xy, elg, zdg, sM2, sVg]),
           fmt="%13.6e %13.6e %6d %12.5e " + " ".join(["%12.5e"] * 12),
           header="two-step RM shell->plate dehom (V2) vs VABS, eta=%.2f, ply frame\n"
                  "y2(m) y3(m) elem z(m)  S11 S22 S33 S23 S13 S12 (RM V2, MPa)"
                  "  S11 S22 S33 S23 S13 S12 (VABS, MPa)" % eta)
pathsave = {}
for i, (ptag, (zeta_p, p1, p2, pV, pd)) in enumerate(pathdata.items()):
    pathsave["path%d_tag" % i] = ptag
    pathsave["path%d_zeta" % i] = zeta_p
    pathsave["path%d_first" % i] = p1
    pathsave["path%d_v2" % i] = p2
    pathsave["path%d_vabs" % i] = pV
np.savez(name + "_dehom_v2.npz", sm_xy=sm_xy, s_first=sM1, s_v2=sM2, s_vabs=sVg,
         is_web_gauss=webp, C6=C6, FF=FF, s6mid=s6mid, F6mid=F6mid,
         dE1e=dE1e, dE2e=dE2e, dE11e=dE11e, dE12e=dE12e, dE22e=dE22e,
         **pathsave)
open(name + "_report.txt", "w").write("\n".join(rep + jrep) + "\n")
print("\nwrote %s_report.txt, %s_dehom_v2.txt/.npz, %s_elem_plate_forces.dat,"
      " %s_v2_section.png, %s_parity.png, %s_thickness_paths.png"
      % ((name,) * 6))
