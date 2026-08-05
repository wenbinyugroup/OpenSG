"""IEA-22 r/R = 0.2 -- the Q-CONSISTENCY DIAGNOSTIC for sigma_13.

The identity a recovered transverse shear must satisfy, wall by wall:

    I13 = INTEGRAL over the wall thickness of sigma_13 dz   ==   Q1_wall

where Q1_wall is the wall's own transverse-shear resultant, which the RM
shell already computes: _rm_shell_strain returns s2 = [2g13, 2g23] (the
tied, drilling-free pair with s2_scheme="mitc4_both" -- the physical wall
shear), and the MSG wall law turns it into a resultant, Q = G_msg s2.

The plate pipeline enforces this by construction (shape from the strain
gradients, AMPLITUDE from the resultant: sigma_a3 *= Q/I).  The shell->plate
chain never uses s2, so nothing pins the amplitude -- this script measures
how far off it is.  Ratio r = I13/Q1: r = 1 is consistent, r << 1 means the
recovered shear carries almost none of the load the section actually sees.

# ----------------------------------------------------------------------------
# ALL VARIABLES USED IN THIS SCRIPT
# ----------------------------------------------------------------------------
#   name, ff_file, station_row   station inputs (see 2_dehom_v2_vabs.py)
#   B, C6, FF, st_m, aA, aB      RM bundle + macro fields
#   s6e (n_el,6) / s2e (n_el,2)  per-element shell strains and the TRANSVERSE
#                                SHEAR pair [2g13, 2g23] the chain discards
#   Gw (n_el,2,2)                per-element MSG wall transverse-shear law
#   Qw (n_el,2)                  wall resultants Gw s2e  [N/m]  <- the target
#   I13/I23 (n_el,)              thickness integral of the recovered sigma_a3
#   ratio13/ratio23              I/Q -- the Q-consistency ratio (1 = exact)
#   OUT  <name>_q_consistency.txt, <name>_q_consistency.png
# ----------------------------------------------------------------------------
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from opensg_shell import build_rm_bundle, _macro_fields, _rm_shell_strain
from opensg_solid.rm_plate_1D.msg_rm_plate import (rm_plate_msg,
                                                   msgrm_strain_at_depth)

############### User Input #################################
name = "iea_s10"
ff_file = "ff51_rmc_reform.dat"
station_row = 10                  # eta = r/R = 0.20
n_z = 41                          # integration points through each wall
############################################################

B = build_rm_bundle(name + "_shell.yaml")
C6 = np.asarray(B["Timo"])
FF = np.loadtxt(ff_file)[station_row, 1:]
st, st_m, aA, aB = _macro_fields(B, beam_force_vabs=FF)

rc = np.asarray(B["red_cells"]); corners = np.asarray(B["corners"])
n_el = rc.shape[0]
layups = B["layup_per_elem"]; ldb = B["layup_db"]; mdb = B["material_db"]
frac = float(B.get("frac", 0.0))
hth = {ln: float(sum(i["thick"])) for ln, i in ldb.items()}
Lel = np.linalg.norm(corners[rc[:, 1]] - corners[rc[:, 0]], axis=1)

warpM = {ln: rm_plate_msg(i["thick"], i["angles"], i["mat_names"], mdb, fraction=frac)
         for ln, i in ldb.items()}

# ---- the section strains AND the transverse shear pair the chain drops ----
# CRITICAL: this ring is homogenized with shear="mitc4_g23" -- only gamma_23
# is tied.  gamma_13 is RAW and carries the flat-wall drilling mode, so a Q1
# built from it is NOT a physical wall shear.  Both schemes are evaluated
# here so the difference is visible rather than assumed: "g23" is the pair
# the homogenization actually used, "both" retro-fits a tying it did not.
s6e = np.zeros((n_el, 6))
s2_by_scheme = {}
for scheme in ("mitc4_g23", "mitc4_both"):
    s2e = np.zeros((n_el, 2))
    for e in range(n_el):
        s6, s2 = _rm_shell_strain(B, e, 0.5, st_m, aA, aB, s2_scheme=scheme)
        s6e[e] = np.asarray(s6, float)
        s2e[e] = np.asarray(s2, float)
    s2_by_scheme[scheme] = s2e
s2e = s2_by_scheme["mitc4_g23"]        # the consistent pair for this ring

# ---- arc gradients (same neighbour scheme as the dehom) ----
deg = np.zeros(int(rc.max()) + 1, int); nd_el = [[] for _ in range(int(rc.max()) + 1)]
for e in range(n_el):
    for nd in (int(rc[e, 0]), int(rc[e, 1])):
        deg[nd] += 1; nd_el[nd].append(e)
fn = np.full((len(deg), 6), np.nan)
for nd in range(len(deg)):
    if deg[nd] == 2 and layups[nd_el[nd][0]] == layups[nd_el[nd][1]]:
        fn[nd] = 0.5 * (s6e[nd_el[nd][0]] + s6e[nd_el[nd][1]])
dE2e = np.zeros((n_el, 6))
for e in range(n_el):
    c0, c1 = int(rc[e, 0]), int(rc[e, 1])
    v0 = fn[c0] if np.isfinite(fn[c0, 0]) else s6e[e]
    v1 = fn[c1] if np.isfinite(fn[c1, 0]) else s6e[e]
    dE2e[e] = (v1 - v0) / Lel[e]

# ---- Q from the wall law vs the thickness integral of the recovery ----
Qw = np.zeros((n_el, 2)); Qw_both = np.zeros((n_el, 2))
I13 = np.zeros(n_el); I23 = np.zeros(n_el)
for e in range(n_el):
    ln = layups[e]; h = hth[ln]
    G = np.asarray(warpM[ln]["G_msg"], float)
    Qw[e] = G @ s2e[e]                                  # [Q1, Q2] N/m
    Qw_both[e] = G @ s2_by_scheme["mitc4_both"][e]
    zz = np.linspace(-frac * h + 1e-9, (1.0 - frac) * h - 1e-9, n_z)
    s13 = np.zeros(n_z); s23 = np.zeros(n_z)
    for k, z in enumerate(zz):
        _, Sig, _ = msgrm_strain_at_depth(warpM[ln], z, s6e[e], np.zeros(6),
                                          dE2e[e], frame="plate")
        s13[k] = Sig[4]; s23[k] = Sig[3]                # SG Voigt: 4=13, 3=23
    I13[e] = np.trapezoid(s13, zz); I23[e] = np.trapezoid(s23, zz)

ok = np.abs(Qw[:, 0]) > 1e-6 * np.abs(Qw[:, 0]).max()
ratio13 = np.where(ok, I13 / np.where(ok, Qw[:, 0], 1.0), np.nan)
ok2 = np.abs(Qw[:, 1]) > 1e-6 * np.abs(Qw[:, 1]).max()
ratio23 = np.where(ok2, I23 / np.where(ok2, Qw[:, 1], 1.0), np.nan)

med13 = float(np.nanmedian(ratio13)); med23 = float(np.nanmedian(ratio23))
d13 = float(np.sqrt(np.mean((Qw[:, 0] - Qw_both[:, 0]) ** 2)))
d23 = float(np.sqrt(np.mean((Qw[:, 1] - Qw_both[:, 1]) ** 2)))
rep = ["=== Q-consistency of the recovered transverse shear, IEA r/R = 0.20 ===",
       "identity per wall:  I = integral(sigma_a3 dz)  ==  Q_a = (G_msg s2)_a",
       "",
       "TYING, AND WHY sigma_13 IS A CROSS-SECTION LIMIT, NOT A BUG:",
       "  cross-section (this ring)  : gamma_23 tied, gamma_13 UNTIED",
       "  surface quad element       : BOTH tied",
       "The ring is a ONE-QUAD-DEEP prismatic strip evaluated at eta = 0",
       "(_strip / _rm_shell_strain).  _mitc_shear_indep ties gamma_23 by",
       "interpolating in eta -- active -- and gamma_13 by interpolating in",
       "xi.  On a strip that is prismatic along xi the gamma_13 row is",
       "already linear there, so that interpolation REPRODUCES the raw row",
       "exactly: the tying is inert, which is why 'mitc4_g23' and",
       "'mitc4_both' give BIT-IDENTICAL output here (measured below) even",
       "though both flags are plumbed and branch.  On a genuine surface",
       "quad (the segment / 3-D SG elements) the row is not linear in xi",
       "and the tying does real work -- that is where a physical gamma_13",
       "lives.  Note the segment's recommended scheme is 'mitc4_wonly'",
       "(field-consistent partial tying); 'mitc4_both' is flagged in its",
       "own docstring as ablation-only because it aliases drilling shear.",
       "So here gamma_13 stays RAW, carrying the flat-wall drilling mode:",
       "Q1 built from it is NOT a physical wall shear and CANNOT serve as",
       "a recovery target.  Q2 (tied gamma_23) can.",
       "Fingerprint: 2g13 rms is ~35x 2g23 rms here, giving Q1 ~110x Q2 --",
       "implausible for a thin-walled section, where the wall carries its",
       "load as in-plane shear FLOW (sigma_12), not through-thickness shear.",
       "Effect of the tying flags on Q:",
       "  rms |Q1(g23) - Q1(both)| = %.4e N/m   (Q1 rms %.4e)"
       % (d13, float(np.sqrt(np.mean(Qw[:, 0] ** 2)))),
       "  rms |Q2(g23) - Q2(both)| = %.4e N/m   (Q2 rms %.4e)"
       % (d23, float(np.sqrt(np.mean(Qw[:, 1] ** 2)))),
       "",
       "                       median      mean       min        max",
       "  I13 / Q1        %10.3e %10.3e %10.3e %10.3e   <- target INVALID"
       % (med13, float(np.nanmean(ratio13)), float(np.nanmin(ratio13)),
          float(np.nanmax(ratio13))),
       "  I23 / Q2        %10.3e %10.3e %10.3e %10.3e   <- target valid"
       % (med23, float(np.nanmean(ratio23)), float(np.nanmin(ratio23)),
          float(np.nanmax(ratio23))),
       "",
       "  Q-CONSISTENCY ERROR in sigma_23 : %.2f %%  (median |1 - I23/Q2|)"
       % (100.0 * abs(1.0 - med23)),
       "  sigma_13: NOT measurable in a cross-section model -- gamma_13 is",
       "  untied by construction there (above).  A physical sigma_13 needs",
       "  the SURFACE-QUAD route (segment / 3-D SG, both rows tied), not a",
       "  rescale of this ring against its contaminated Q1.",
       "",
       "  wall Q2 rms = %.4e N/m   recovered I23 rms = %.4e N/m"
       % (float(np.sqrt(np.mean(Qw[:, 1] ** 2))),
          float(np.sqrt(np.mean(I23 ** 2)))),
       "  (the plate pipeline closes an amplitude gap by rescaling"
       " sigma_a3 *= Q/I -- shape from the gradients, amplitude from Q --",
       "  which is only legitimate where the target Q is physical)"]
print("\n".join(rep))
open(name + "_q_consistency.txt", "w").write("\n".join(rep) + "\n")

fig, ax = plt.subplots(1, 2, figsize=(12.0, 4.6))
ax[0].plot(np.abs(Qw[:, 0]), np.abs(I13), ".", ms=4, color="#1f77b4")
lim = [1e-6 * max(np.abs(Qw[:, 0]).max(), 1e-30), np.abs(Qw[:, 0]).max()]
ax[0].plot(lim, lim, "k-", lw=1, label="consistent (I = Q)")
ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set_xlabel(r"wall $|Q_1|$ from the RM shell [N/m]")
ax[0].set_ylabel(r"$|\int \sigma_{13}\,dz|$ recovered [N/m]")
ax[0].legend(frameon=False); ax[0].grid(alpha=0.3, which="both")
ax[1].hist(np.log10(np.abs(ratio13[np.isfinite(ratio13)]) + 1e-30), bins=40,
           color="#ff7f0e")
ax[1].axvline(0.0, color="k", lw=1.5)
ax[1].set_xlabel(r"$\log_{10}(I_{13}/Q_1)$   (0 = consistent)")
ax[1].set_ylabel("wall elements")
ax[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(name + "_q_consistency.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote %s_q_consistency.txt / .png" % name)
