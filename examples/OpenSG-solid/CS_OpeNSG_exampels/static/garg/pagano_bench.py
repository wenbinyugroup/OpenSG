"""pagano_bench.py -- the orchestrator for the RM-vs-Pagano benchmarks (driven by
examples/garg/case*/7_helper_RM_Pagano_benchmark{1,2,3}.py).

Exactly three standalone chains, one module each:

  pagano_exact.py   the exact 3-D reference (Pagano 1969 / Garg Eqs. 18-24) --
                    REFERENCE CURVES ONLY, never an input to the other chains
  statics_fsdt.py   the statics of Q1/M11 (full derivation in its docstring) and the
                    standalone FSDT chain with the Whitney-1973 Eq.-(7) k1^2
  the MSG core      rm_plate_msg (8x8 at the mid-surface) + msgrm_strain_at_depth
                    (Eq.-63 recovery), driven by the SAME statics Q1

Stations (the harmonic families): sigma_13 at x = 0 (cos peak, Q1 = q0/p there);
sigma_33 at x = a/2 (sin peak; from thickness equilibrium of the recovered sigma_13
amplitude).  FSDT has no sigma_33 (plane-stress plies).  No Abaqus content anywhere:
a shell FE cannot predict a through-thickness stress standalone; where statics cannot
supply the resultants, an Abaqus run may provide FF -- carrying the SAME section law
as the recovery chain it feeds.
"""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
# repo root = the first ancestor holding opensg_jax/ (location-robust: the
# benchmark also runs from the examples/RM_OpenSG_pagano copy)
CC = HERE
while not os.path.isdir(os.path.join(CC, "src", "opensg_solid", "rm_plate_1D")):
    _up = os.path.dirname(CC)
    if _up == CC:
        raise RuntimeError("rm_plate repo root not found above " + __file__)
    CC = _up
sys.path.insert(0, os.path.join(CC, "src"))
sys.path.insert(0, HERE)

from opensg_solid.rm_plate_1D.msg_rm_plate import rm_plate_msg, msgrm_strain_at_depth  # noqa: E402
from opensg_solid.rm_plate_1D.msg_transverse_shear import transverse_shear_stiffness   # noqa: E402
from pagano_exact import pagano_profiles                              # noqa: E402
from statics_fsdt import statics_resultants, fsdt_s13                 # noqa: E402
from garg_layups import MATERIAL_DB, LAYUPS, H                        # noqa: E402

q0 = 1.0e4
a = 1.0
p = np.pi / a


def run_case(case, S, tag):
    """One (layup, S) benchmark: all three chains, the .dat, and the two-panel plot.

    Variables
    ---------
    case, S, tag   LAYUPS key ("caseA"/"caseB"/"caseC"), aspect ratio a/h, and the
                   benchmark number used only in the .dat header
    outdir         the case subfolder examples/garg/<case>/ everything is written to
    h              laminate thickness for this S: h = a/S [m]
    lay, fr, thk   the layup dict; its ply thickness FRACTIONS (thick_i / H); the
                   ply thicknesses re-scaled to this h (fr_i * h) [m]
    ang, mats      ply angles [deg] and material names
    zc, sig, uvw   chain 1 (reference ONLY): exact through-thickness grid from the
                   mid-surface [m], stress amplitudes (n, 6) Voigt
                   [11,22,33,23,13,12], displacement amplitudes (n, 3)
    w_ex           exact mid-surface deflection amplitude w(a/2) [m]
    Q1             the statics shear resultant at x = 0: q0/p [N/m] -- the ONLY
                   load input the two plate chains receive
    FF_end         (8,) plate force-resultant vector at x = 0 in the RM order
                   [N11, N22, N12, M11, M22, M12, Q1, Q2]: pure shear there
    r              chain 2: rm_plate_msg result dict (mid-surface, fraction = 0.5);
                   r["ABDG"] the 8x8, r["A6"] its 6x6 in-plane/bending block,
                   r["G_msg"] the Yu-2003 Eq.-61 least-squares 2x2 shear
    D11, G11       bending / transverse-shear diagonals of the 8x8 (rows k11, 2g13)
    w_msg          closed-form plate deflection q0/(p^4 D11) + q0/(p^2 G11) [m]
    S6             inverse of the 6x6 block: strains from resultants
    E6_end         (6,) mid-surface strain state at x = 0: S6 @ FF_end[:6]
    dE1_end        (6,) x-gradient of that strain at x = 0: the ANALYTICAL
                   derivative of the 1-D harmonic Pagano cylindrical-bending
                   mode (M11,1 = Q1 exactly for this single-harmonic mode,
                   all other resultant gradients vanish), NOT a general
                   equilibrium closure, so dE1 = S6 @ [0, 0, 0, Q1, 0, 0].
                   General 2-D fields must FD the FF field itself instead.
    z6             (6,) zeros: the y-gradient (cylindrical bending, d/dy = 0)
    s13_m          (n,) MSG-RM sigma_13 amplitude: Eq.-63 recovery at each zc
                   (msgrm_strain_at_depth returns (strain, STRESS, angle); [1][4]
                   is the sigma_13 slot of the Voigt stress)
    s33_m          (n,) MSG-RM sigma_33 amplitude by through-thickness equilibrium:
                   families s13 ~ cos(px), s33 ~ sin(px) turn
                   sigma_33,3 = -sigma_13,1 into d(s33_hat)/dz = p * s13_hat,
                   trapezoid-integrated from the free bottom face
    s13_f          (n,) chain 3: the FSDT staircase C55(z) Q1 / (k1^2 A55)
    k1sq, A55      the Whitney-1973 Eq.-(7) correction and int C55 dz [N/m]
    G_w            (2, 2) complementary-energy transverse-shear stiffness (printed
                   next to r["G_msg"] in the .dat for the two-construction compare)
    relerr         inner helper: 100 ||m - e|| / ||e||, the relative L2 error [%]
    e13, e13f, e33 the three error numbers: MSG s13, FSDT s13, MSG s33 [%]
    hdr            the .dat header block (8x8, statics input, both G's, k1^2,
                   the chain description, errors, column legend)
    fig, ax1, ax2  the two panels: sigma_13 at x = 0, sigma_33 at x = a/2, with
                   ONE shared legend outside the right edge

    Writes pagano_S<S>.dat + pagano_S<S>.png into the case folder and returns
    dict(case, S, e13, e13f, e33, ABDG).
    """
    outdir = os.path.join(HERE, case)
    h = a / S
    lay = LAYUPS[case]
    fr = [t / H for t in lay["thick"]]
    thk = [f * h for f in fr]; ang = lay["angles"]; mats = lay["mat_names"]

    # chain 1: the exact reference (curves only)
    zc, sig, uvw = pagano_profiles(thk, ang, mats, MATERIAL_DB, a=a, q0=q0)
    w_ex = float(uvw[np.argmin(np.abs(zc)), 2])

    # the shared input: statics of the problem statement (statics_fsdt sec. 1)
    Q1, _ = statics_resultants(q0, a, 0.0)            # x = 0: Q1 = q0/p, M11 = 0
    FF_end = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, Q1, 0.0])

    # chain 2: MSG-RM (8x8 inversion -> gradient closure -> Eq.-63 recovery; sigma33
    # by through-thickness equilibrium integration of the recovered sigma13 amplitude)
    r = rm_plate_msg(thk, ang, mats, MATERIAL_DB, fraction=0.5)
    D11 = float(r["ABDG"][3, 3]); G11 = float(r["ABDG"][6, 6])
    w_msg = q0 / (p ** 4 * D11) + q0 / (p ** 2 * G11)
    S6 = np.linalg.inv(np.asarray(r["A6"]))
    E6_end = S6 @ FF_end[:6]
    # analytical x-gradient of the strain for THIS 1-D harmonic Pagano mode only:
    # dM11/dx = Q1 exactly (cylindrical bending, M11,1 = Q1 for sin(px)/cos(px)
    # fields), and every other resultant gradient vanishes by the harmonics.
    # This is a mode-specific closed form, NOT a general equilibrium closure --
    # general 2-D fields must get dE1 from finite-differencing the FF field.
    dE1_end = S6 @ np.array([0, 0, 0, Q1, 0, 0.0])
    z6 = np.zeros(6)
    s13_m = np.empty_like(zc)
    for i, z in enumerate(zc):
        s13_m[i] = msgrm_strain_at_depth(r, z, E6_end, dE1_end, z6, frame="plate")[1][4]
    # sigma33 amplitude by 3-D equilibrium (families s13 ~ cos -> s33 ~ sin):
    #   d(s33_hat)/dz = p s13_hat  ->  top face lands on q0 by resultant closure
    s33_m = np.concatenate([[0.0], np.cumsum(0.5 * p * (s13_m[1:] + s13_m[:-1])
                                             * np.diff(zc))])

    # chain 3: standalone FSDT with the Whitney-1973 k (statics_fsdt secs. 2-3)
    s13_f, k1sq, A55 = fsdt_s13(zc, thk, ang, mats, MATERIAL_DB, Q1)
    G_w = np.asarray(transverse_shear_stiffness(thk, ang, mats, MATERIAL_DB)[0])

    def relerr(m, e):
        """Relative L2 error [%]: m = model profile, e = exact profile (both (n,))."""
        return 100 * np.linalg.norm(m - e) / np.linalg.norm(e)

    e13 = relerr(s13_m, sig[:, 4]); e13f = relerr(s13_f, sig[:, 4])
    e33 = relerr(s33_m, sig[:, 2])

    # ------------------------------------------------------------------- .dat
    hdr = ["%s -- MSG-RM vs EXACT (Pagano) benchmark %s,  S = a/h = %g" % (case, tag, S),
           "reference surface: MID-SURFACE (fraction = 0.5) everywhere",
           "stations: x = 0 (s13, Q1 = q0/p), x = a/2 (s33)",
           "plies: " + ", ".join("%s(%.4gmm/%g)" % (m, 1e3 * t, x)
                                 for m, t, x in zip(mats, thk, ang)),
           "",
           "RM 8x8 ABDG (rows e11,e22,g12,k11,k22,k12,2g13,2g23):"]
    hdr += ["  " + " ".join("%14.6e" % v for v in row) for row in np.asarray(r["ABDG"])]
    hdr += ["",
            "INPUT FROM STATICS ONLY (statics_fsdt.py sec. 1): Q1(0) = q0/p = %.6g" % Q1,
            "  FF_end (x=0) = [%s]" % ", ".join("%.6g" % v for v in FF_end),
            "  plate w(a/2) = %.6e ; exact w = %.6e" % (w_msg, w_ex),
            "",
            "transverse-shear stiffness, the two constructions:",
            "  G_whitney (complementary energy, Whitney JAM 40 (1973) Eq. (7)):",
            "    [[%13.6e, %13.6e], [%13.6e, %13.6e]]"
            % (G_w[0, 0], G_w[0, 1], G_w[1, 0], G_w[1, 1]),
            "  G_msg (MSG-RM least-squares projection, Yu 2003 Eq. 61):",
            "    [[%13.6e, %13.6e], [%13.6e, %13.6e]]"
            % (r["G_msg"][0, 0], r["G_msg"][0, 1], r["G_msg"][1, 0], r["G_msg"][1, 1]),
            "  k1^2 (statics_fsdt.whitney_k1sq, used in the FSDT curve): %.6f" % k1sq,
            "  (uniform-k reference 5/6 = 0.8333;  A55 = %.6e)" % A55,
            "",
            "THREE STANDALONE CHAINS, out-of-plane stresses only:",
            "  exact   Pagano cyl. bending (JCM 3 (1969) 398-411; Garg Eqs. 18-24)",
            "  MSG-RM  statics Q1 -> 8x8 inversion -> Eq.-63 recovery -> s33 by",
            "          through-thickness equilibrium of the recovered s13 amplitude",
            "  FSDT    statics Q1 -> staircase C55(z) Q1/(k1^2 A55); NO s33 in FSDT",
            "rel L2 errors vs exact:  s13 %7.3f%%  (FSDT-Whitney %7.2f%%)   s33 %7.3f%%"
            % (e13, e13f, e33),
            "s33 top-face closure: %.4f q0" % (s33_m[-1] / q0),
            "",
            "columns: z[m]  s13_msg  s13_exact  s13_fsdt  s33_msg  s33_exact  [Pa]"]
    np.savetxt(os.path.join(outdir, "pagano_S%g.dat" % S),
               np.column_stack([zc, s13_m, sig[:, 4], s13_f, s33_m, sig[:, 2]]),
               header="\n".join(hdr), fmt="%15.6e")

    # ------------------------------------------------------------------- plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 5.2))
    ax1.plot(sig[:, 4], zc / h, "-", color="k", lw=2.0, label="exact 3-D (Pagano)")
    ax1.plot(s13_m, zc / h, ":s", color="#ff7f0e", ms=4, mfc="none", mew=1.2, lw=1.6,
             markevery=4, label="MSG-RM")
    ax1.plot(s13_f, zc / h, "--", color="#1f77b4", lw=1.4,
             label="FSDT constitutive\n(Whitney-1973 $k_1^2$)")
    ax1.set_xlabel(r"$\sigma_{13}$ [Pa]  at  $x=0$", fontsize=11)
    ax1.set_ylabel("$z/h$", fontsize=11)
    ax2.plot(sig[:, 2], zc / h, "-", color="k", lw=2.0)
    ax2.plot(s33_m, zc / h, ":s", color="#ff7f0e", ms=4, mfc="none", mew=1.2, lw=1.6,
             markevery=4)
    ax2.set_xlabel(r"$\sigma_{33}$ [Pa]  at  $x=a/2$", fontsize=11)
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
    # vertical legend OUTSIDE the axes (right of the second panel): it can never
    # block a curve, whatever shape the profiles take across the nine cases
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.995, 0.5),
               frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "pagano_S%g.png" % S), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return dict(case=case, S=S, e13=e13, e13f=e13f, e33=e33,
                ABDG=np.asarray(r["ABDG"]))


def run_benchmark(case, S_list, tag):
    """Run one laminate family over its aspect-ratio sweep and write rm_8x8.out.

    Variables: case/tag as in run_case; S_list = the aspect ratios a/h to sweep
    (the benchmark set is (10, 50): moderate + thin, both inside the plate-model
    regime); m = the per-S summary dict from run_case; out8 = the accumulated
    rm_8x8.out lines (one labelled 8x8 ABDG block per S, since the 8x8 depends on
    h).  Prints the one-line error summary per S.
    """
    print("%s (benchmark %s): stations x=0 (s13) and x=a/2 (s33), mid-surface reference"
          % (case, tag))
    out8 = ["RM 8x8 ABDG for %s (mid-surface reference; rows e11,e22,g12,k11,k22,"
            "k12,2g13,2g23)" % case]
    for S in S_list:
        m = run_case(case, S, tag)
        print("  S = %-4g  s13 %7.3f%% (FSDT %7.2f%%)   s33 %7.3f%%"
              % (S, m["e13"], m["e13f"], m["e33"]))
        out8.append("")
        out8.append("S = a/h = %g  (h = %g m):" % (S, a / S))
        out8 += ["  " + " ".join("%14.6e" % v for v in row) for row in m["ABDG"]]
    with open(os.path.join(HERE, case, "rm_8x8.out"), "w") as f:
        f.write("\n".join(out8) + "\n")
