"""IEA-22 blade -- the FULL 51-STATION SPANWISE SWEEP.

For every BeamDyn station of the blade: homogenize the RM shell ring to the
Timoshenko 6x6, then dehomogenize with that station's own beam FF through the
two-step chain (beam -> per-element plate reaction forces -> MSG-RM
through-thickness stress), all on the batched recovery.

Outputs the spanwise beam-property table a 1-D solver consumes, the peak
recovered wall stress per station, and the timing.

# ----------------------------------------------------------------------------
# ALL VARIABLES USED IN THIS SCRIPT
# ----------------------------------------------------------------------------
#   yaml_dir        where iea_s00_shell.yaml .. iea_s50_shell.yaml live
#   ff_file         BeamDyn FF table, one row per station (VABS order)
#   n_depth         through-thickness recovery points per wall element
#   stations        the station indices actually processed
#   eta             r/R of each station (from the FF table)
#   C6s (n,6,6)     per-station Timoshenko 6x6
#   props (n,6)     its diagonal [EA GA2 GA3 GJ EI2 EI3]
#   peak (n,6)      peak |sigma| per component over the whole section [MPa]
#   t_homo/t_dehom  per-station wall time [s]
#   OUT  iea51_beam_props.dat, iea51_peak_stress.dat, iea51_sweep.png,
#        iea51_sweep.npz
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
    rm_plate_msg, msgrm_strain_at_depth_batch)

############### User Input #################################
yaml_dir = os.path.expanduser(
    "~/OpenSG-TW-claude/examples/data/iea_all_stations/shell51/1d_yaml")
ff_file = "ff51_rmc_reform.dat"
n_depth = 9                  # through-thickness recovery points per element
stations = range(51)         # s00 .. s50
############################################################

LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
NAMES = ["s11", "s22", "s33", "s23", "s13", "s12"]
ffall = np.loadtxt(ff_file)

rows_p = []; rows_s = []; t_h = []; t_d = []; done = []
t_all = time.perf_counter()
for i in stations:
    y = os.path.join(yaml_dir, "iea_s%02d_shell.yaml" % i)
    if not os.path.isfile(y):
        print("s%02d: yaml missing -- skipped" % i)
        continue
    try:
        t0 = time.perf_counter()
        B = build_rm_bundle(y)
        C6 = np.asarray(B["Timo"])
        th = time.perf_counter() - t0

        t0 = time.perf_counter()
        FF = ffall[i, 1:]
        eta = ffall[i, 0]
        st, st_m, aA, aB = _macro_fields(B, beam_force_vabs=FF)
        _, _sm, st_cl1, st_cl2 = _macro_recovery(C6, np.linalg.inv(C6) @ FF)

        rc = np.asarray(B["red_cells"]); corners = np.asarray(B["corners"])
        n_el = rc.shape[0]; n_nd = int(rc.max()) + 1
        layups = B["layup_per_elem"]; ldb = B["layup_db"]; mdb = B["material_db"]
        frac = float(B.get("frac", 0.0))
        hth = {ln: float(sum(v["thick"])) for ln, v in ldb.items()}
        nodes_s, quads_s, _hs = B["strip"]
        Lel = np.linalg.norm(corners[rc[:, 1]] - corners[rc[:, 0]], axis=1)

        s6mid = np.zeros((n_el, 6)); dE1e = np.zeros((n_el, 6))
        for e in range(n_el):
            s6, _ = _rm_shell_strain(B, e, 0.5, st_m, aA, aB)
            s6mid[e] = np.asarray(s6, float)
            Xe = nodes_s[quads_s[e]]
            BDe, BDh, *_ = quad_ops_indep(Xe, B["re3"][e], 0.0, 0.0,
                                          float(B["k22"][e]), B["cross"],
                                          B["ax"])
            c0, c1 = int(rc[e, 0]), int(rc[e, 1])
            g = np.r_[c0 * 6:c0 * 6 + 6, c1 * 6:c1 * 6 + 6,
                      c1 * 6:c1 * 6 + 6, c0 * 6:c0 * 6 + 6]
            dE1e[e] = np.asarray(BDe @ st_cl1 + BDh @ aB[g], float)

        deg = np.zeros(n_nd, int); nd_el = [[] for _ in range(n_nd)]
        for e in range(n_el):
            for nd in (int(rc[e, 0]), int(rc[e, 1])):
                deg[nd] += 1; nd_el[nd].append(e)
        fn = np.full((n_nd, 6), np.nan)
        for nd in range(n_nd):
            if deg[nd] == 2 and layups[nd_el[nd][0]] == layups[nd_el[nd][1]]:
                fn[nd] = 0.5 * (s6mid[nd_el[nd][0]] + s6mid[nd_el[nd][1]])
        dE2e = np.zeros((n_el, 6))
        for e in range(n_el):
            c0, c1 = int(rc[e, 0]), int(rc[e, 1])
            v0 = fn[c0] if np.isfinite(fn[c0, 0]) else s6mid[e]
            v1 = fn[c1] if np.isfinite(fn[c1, 0]) else s6mid[e]
            dE2e[e] = (v1 - v0) / Lel[e]

        warpM = {ln: rm_plate_msg(v["thick"], v["angles"], v["mat_names"],
                                  mdb, fraction=frac) for ln, v in ldb.items()}
        # batched recovery: every element x every depth, grouped by layup
        zeta = (np.arange(n_depth) + 0.5) / n_depth
        el_of = np.repeat(np.arange(n_el), n_depth)
        h_of = np.array([hth[layups[e]] for e in el_of])
        z_of = (np.tile(zeta, n_el) - frac) * h_of
        lay_of = np.array([layups[e] for e in el_of])
        sig = np.zeros((len(el_of), 6))
        for ln in set(layups):
            m = np.where(lay_of == ln)[0]
            if not len(m):
                continue
            _, S, _ = msgrm_strain_at_depth_batch(
                warpM[ln], z_of[m], s6mid[el_of[m]], dE1e[el_of[m]],
                dE2e[el_of[m]])
            sig[m] = S
        sig /= 1e6
        td = time.perf_counter() - t0

        rows_p.append([eta] + [C6[k, k] for k in range(6)])
        rows_s.append([eta] + [float(np.abs(sig[:, c]).max()) for c in range(6)])
        t_h.append(th); t_d.append(td); done.append(i)
        print("s%02d eta=%.3f  EA %.4e  EI2 %.4e  EI3 %.4e | peak s11 %8.2f MPa"
              "  | homo %5.1fs dehom %5.1fs" % (i, eta, C6[0, 0], C6[4, 4],
                                                C6[5, 5], rows_s[-1][1], th, td),
              flush=True)
    except Exception as exc:                      # a bad station must not kill the sweep
        print("s%02d: FAILED -- %s: %s" % (i, type(exc).__name__, exc), flush=True)

P = np.array(rows_p); S = np.array(rows_s)
print("\n%d/%d stations completed in %.1f s (homo %.1f s, dehom %.1f s)"
      % (len(done), len(list(stations)), time.perf_counter() - t_all,
         sum(t_h), sum(t_d)))

np.savetxt("iea51_beam_props.dat", P, fmt="%12.6f " + " ".join(["%16.8e"] * 6),
           header="IEA-22 spanwise Timoshenko beam properties (RM shell ring)\n"
                  "eta(r/R)  " + "  ".join("%16s" % l for l in LBL))
np.savetxt("iea51_peak_stress.dat", S, fmt="%12.6f " + " ".join(["%16.8e"] * 6),
           header="peak |sigma| per station over the whole section, ply frame"
                  " [MPa]\neta(r/R)  " + "  ".join("%16s" % n for n in NAMES))
np.savez("iea51_sweep.npz", eta=P[:, 0], props=P[:, 1:], peak=S[:, 1:],
         t_homo=np.array(t_h), t_dehom=np.array(t_d), stations=np.array(done))

fig, ax = plt.subplots(2, 3, figsize=(15.0, 8.0))
for k in range(6):
    a_ = ax[k // 3, k % 3]
    a_.semilogy(P[:, 0], np.abs(P[:, 1 + k]), "o-", ms=4, color="#1f77b4")
    a_.set_xlabel("$r/R$"); a_.set_ylabel(LBL[k])
    a_.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig("iea51_beam_props.png", dpi=150, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
for k, c in enumerate((0, 5)):
    ax[0].plot(S[:, 0], S[:, 1 + c], "o-", ms=4,
               label=r"$\sigma_{%s}$" % NAMES[c][1:])
ax[0].set_xlabel("$r/R$"); ax[0].set_ylabel("peak $|\\sigma|$ [MPa]")
ax[0].legend(frameon=False); ax[0].grid(alpha=0.3)
ax[1].plot(P[:, 0], t_h, "o-", ms=4, label="homogenization")
ax[1].plot(P[:, 0], t_d, "s-", ms=4, label="dehomogenization")
ax[1].set_xlabel("$r/R$"); ax[1].set_ylabel("wall time [s]")
ax[1].legend(frameon=False); ax[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig("iea51_sweep.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote iea51_beam_props.dat/.png, iea51_peak_stress.dat,"
      " iea51_sweep.png/.npz")
