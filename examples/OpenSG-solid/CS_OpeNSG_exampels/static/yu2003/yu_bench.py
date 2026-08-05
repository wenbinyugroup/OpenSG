"""yu_bench.py -- the orchestrator for the Yu-2003 section-6.1 benchmarks (driven by
examples/yu2003/case<N>/7_helper_RM_Pagano_yu2003_case<N>.py).

Reproduces the cylindrical-bending validation of Yu-Hodges-Volovoi, C&S 81 (2003)
439-454, sec. 6.1 (their Figs. 3-20) with the same three standalone chains as the
Garg benchmark (examples/garg/pagano_bench.py), at THEIR configuration: L/h = 4,
Pagano-ratio material in psi, the split face load s3 = b3 = (p0/2) sin(px), and
normalization sigma_bar = sigma/p0 (p0 = 1 -> raw values are already normalized):

  chain 1  the exact 3-D reference: ExactCyl (state-space Pagano; the angle plies
           make this the SHEAR-COUPLED problem of Pagano 1970) with q0 = +p0/2 on
           the top face and q_bot = -p0/2 on the bottom -- REFERENCE CURVES ONLY
  chain 2  MSG-RM: rm_plate_msg 8x8 -> the HARMONIC RM CYLINDRICAL-BENDING SOLVE
           (rm_cyl_bend below) -> Eq.-63 recovery -> sigma_33 by thickness
           equilibrium from the loaded bottom face
  chain 3  FSDT with the Whitney-1973 Eq.-(7) k1^2 staircase (sigma_13 only; the
           Whitney construction is orthotropic -- for these ANGLE plies it is a
           baseline, exactly the role FOSDT plays in Yu's paper)

WHY a plate solve here when the Garg benchmark fed statics directly: statics alone
determines only Q1 = (q0/p) cos(px) and M11 = (q0/p^2) sin(px).  For Yu's ANGLE-PLY
laminates the section is shear-coupled (A16, D16, B16 != 0): the twist moment M12
and the shear Q2 are NONZERO and are set by the constitutive coupling, not by
statics.  The 5-DOF harmonic solve supplies them from the MSG 8x8 itself -- the
exact analog of Yu's own sec. 6.2, where DYMORE (a 2-D solver fed with the VAPAS
stiffness) supplies the 2-D solution that VAPAS then recovers from.  No Pagano
content enters chains 2-3: the solve uses only the 8x8 and the load.

Stations (harmonic families, as in Yu's paper): sigma_13 and sigma_23 ~ cos(px)
at x = 0; sigma_33 ~ sin(px) at x = L/2.  Out-of-plane components only.
"""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
# repo root = the first ancestor holding opensg_jax/; the garg pipeline is the
# SIBLING folder (both hold in examples/ and in the RM_OpenSG_pagano copy)
CC = HERE
while not os.path.isdir(os.path.join(CC, "src", "opensg_solid", "rm_plate_1D")):
    _up = os.path.dirname(CC)
    if _up == CC:
        raise RuntimeError("rm_plate repo root not found above " + __file__)
    CC = _up
sys.path.insert(0, os.path.join(CC, "src"))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "garg"))

from opensg_solid.rm_plate_1D.msg_rm_plate import rm_plate_msg, msgrm_strain_at_depth  # noqa: E402
from pagano_exact import ExactCyl                                     # noqa: E402
from statics_fsdt import fsdt_s13                                     # noqa: E402
from yu_layups import MATERIAL_DB, LAYUPS, H, L_SPAN, P0              # noqa: E402


def rm_cyl_bend(A6, G2, p, q0):
    """The harmonic RM cylindrical-bending plate solution from the MSG 8x8 blocks.

    Kinematics (single harmonic, d/dy = 0, simply supported at x = 0 and x = L):
        u0 = U cos(px), v0 = V cos(px), w = W sin(px),
        phi1 = F1 cos(px), phi2 = F2 cos(px)
    which gives the sin-family plate strains E6 = Es sin(px) and the cos-family
    transverse shears gam = gs cos(px), both LINEAR in the DOF amplitudes.  The
    principle of virtual work over the half-wave (harmonic orthogonality removes
    the x-integrals) yields the symmetric 5x5 system K y = f -- equilibrium and
    the constitutive law solved together, so N11 = N12 = 0, M11 = q0/p^2 and
    Q1 = q0/p come OUT of the solve (asserted below), while the coupling-driven
    M12, Q2, N22, M22 are supplied by the 8x8.

    Variables
    ---------
    A6, G2   the MSG blocks: 6x6 in-plane/bending (rows e11, e22, g12, k11, k22,
             k12) and 2x2 transverse shear (rows 2g13, 2g23); the 8x8 is their
             block diagonal (Yu 2003 Eq. 61: R'AR + gam'G gam, no cross term)
    p, q0    wavenumber pi/L [1/in] and TOTAL transverse load amplitude [psi]
             (the top/bottom SPLIT of the load never enters a plate model --
             only the resultant q(x) = q0 sin(px) does)
    y        (5,) DOF amplitudes [U, V, W, F1, F2] ([in], rotations [-])
    B_E      (6, 5) dEs/dy: e11 = -pU, g12 = -pV, k11 = -pF1, k12 = -pF2
             (e22 = k22 = 0 identically: cylindrical bending)
    B_g      (2, 5) dgs/dy: 2g13 = pW + F1, 2g23 = F2 (engineering shears)
    K, f     the 5x5 stiffness B_E'A6 B_E + B_g'G2 B_g and load [0,0,q0,0,0]
    Es, gs   the solved strain amplitudes (sin- and cos-family)
    R6       (6,) resultant sin-amplitudes [N11, N22, N12, M11, M22, M12]:
             N22/M22 are the infinite-width reactions, M12 the coupling twist
    Q        (2,) shear cos-amplitudes [Q1, Q2]; Q2 = p M12 (twist equilibrium)

    Returns (y, Es, gs, R6, Q).
    """
    B_E = np.zeros((6, 5))
    B_E[0, 0] = -p
    B_E[2, 1] = -p
    B_E[3, 3] = -p
    B_E[5, 4] = -p
    B_g = np.zeros((2, 5))
    B_g[0, 2] = p
    B_g[0, 3] = 1.0
    B_g[1, 4] = 1.0
    K = B_E.T @ A6 @ B_E + B_g.T @ G2 @ B_g
    f = np.zeros(5)
    f[2] = q0
    y = np.linalg.solve(K, f)
    Es = B_E @ y
    gs = B_g @ y
    R6 = A6 @ Es
    Q = G2 @ gs
    # the statics identities MUST come out of the solve (equilibrium check)
    assert abs(R6[0]) < 1e-8 * abs(R6[3]) * p          # N11 = 0
    assert abs(R6[2]) < 1e-8 * abs(R6[3]) * p          # N12 = 0
    assert abs(R6[3] - q0 / p ** 2) < 1e-8 * q0 / p ** 2   # M11 = q0/p^2
    assert abs(Q[0] - q0 / p) < 1e-8 * q0 / p              # Q1 = q0/p
    assert abs(p * R6[5] - Q[1]) < 1e-8 * max(abs(Q[0]), abs(Q[1]))  # Q2 = p M12
    return y, Es, gs, R6, Q


def run_case(case):
    """One Yu-2003 sec.-6.1 case: all three chains, the .dat, and the 3-panel plot.

    Variables
    ---------
    case           LAYUPS key ("case1"/"case2"/"case3")
    outdir         the case subfolder examples/yu2003/<case>/
    lay            the layup dict; thk/ang/mats its plies (bottom -> top)
    a, h, p, q0    span L = 4 [in], thickness 1 [in], pi/L [1/in], p0 [psi]
    ex             chain 1: the exact solver with the SPLIT load q0 = +p0/2 top,
                   q_bot = -p0/2 bottom; bcres = its traction residual (~1e-12)
    zc, sig, uvw   exact grid from the mid-surface [in], stress amplitudes (n, 6)
                   Voigt [11,22,33,23,13,12], displacement amplitudes (n, 3)
    w_ex           exact mid-surface deflection amplitude [in]
    r              chain 2: rm_plate_msg result dict (mid-surface, fraction 0.5)
    A6, G2         its 6x6 and 2x2 blocks fed to the harmonic solve
    y, Es, gs,     the rm_cyl_bend solution (see its docstring); w_msg = y[2] the
    R6, Q          plate deflection amplitude [in]
    E6_0, dE1_0    recovery inputs at x = 0: the sin-family strain is ZERO there
                   and its x-gradient is p Es (E6(x) = Es sin(px)); dE2 = 0
                   (d/dy = 0); the second gradients are 0 at x = 0 (~ sin)
    s13_m, s23_m   (n,) MSG-RM Eq.-63 recovered amplitudes at each zc
                   (msgrm_strain_at_depth returns (strain, STRESS, angle);
                   stress slots [4] = sigma_13, [3] = sigma_23)
    s33_m          (n,) MSG-RM sigma_33 by thickness equilibrium: families
                   s13 ~ cos, s33 ~ sin turn sigma_33,3 = -sigma_13,1 (the
                   sigma_23,2 term dies with d/dy = 0) into
                   d(s33hat)/dz = p s13hat, trapezoid-integrated from the LOADED
                   bottom face s33hat(-h/2) = -p0/2; top closure -> +p0/2
    s13_f          (n,) chain 3: the Whitney-k1^2 FSDT staircase from Q1 = q0/p
    k1sq, A55      its shear correction and int C55 dz
    relerr         inner helper: 100 ||m - e|| / ||e|| [%]
    e13, e23, e33, the error numbers vs exact (s23 also reports the exact norm
    e13f           scale: for case3 the nearly-cross-ply s23 is SMALL, a relative
                   error on it is fragile by construction)
    hdr            the .dat header (8x8, solve amplitudes, resultants, checks,
                   k1^2, errors, column legend)
    fig, axes      the 3 panels: s13 (x=0), s23 (x=0), s33 (x=L/2), one legend
                   outside right; z/h vertical axis; stresses already sigma/p0

    Writes yu_<case>.dat + yu_<case>.png + rm_8x8.out into the case folder and
    returns the error summary dict.
    """
    outdir = os.path.join(HERE, case)
    lay = LAYUPS[case]
    thk = lay["thick"]; ang = lay["angles"]; mats = lay["mat_names"]
    a = L_SPAN; h = H; p = np.pi / a; q0 = P0

    # chain 1: exact, with Yu's split face load (reference only)
    ex = ExactCyl(thk, ang, mats, MATERIAL_DB, a, q0=0.5 * q0, q_bot=-0.5 * q0)
    bcres = ex.bc_residual()
    zc, sig, _, uvw = ex.profile(n_per_layer=81)
    w_ex = float(uvw[np.argmin(np.abs(zc)), 2])

    # chain 2: MSG-RM -- 8x8, harmonic plate solve, Eq.-63 recovery
    r = rm_plate_msg(thk, ang, mats, MATERIAL_DB, fraction=0.5)
    ABDG = np.asarray(r["ABDG"])
    A6 = np.asarray(r["A6"]); G2 = ABDG[6:8, 6:8]
    y, Es, gs, R6, Q = rm_cyl_bend(A6, G2, p, q0)
    w_msg = float(y[2])
    E6_0 = np.zeros(6)
    dE1_0 = p * Es
    z6 = np.zeros(6)
    s13_m = np.empty_like(zc); s23_m = np.empty_like(zc)
    for i, z in enumerate(zc):
        Sig = msgrm_strain_at_depth(r, z, E6_0, dE1_0, z6, frame="plate")[1]
        s13_m[i] = Sig[4]; s23_m[i] = Sig[3]
    s33_m = -0.5 * q0 + np.concatenate(
        [[0.0], np.cumsum(0.5 * p * (s13_m[1:] + s13_m[:-1]) * np.diff(zc))])

    # chain 3: Whitney-k1^2 FSDT staircase (orthotropic construction -> baseline)
    s13_f, k1sq, A55 = fsdt_s13(zc, thk, ang, mats, MATERIAL_DB, Q[0])

    def relerr(m, e):
        """Relative L2 error [%]: m = model profile, e = exact profile (both (n,))."""
        return 100 * np.linalg.norm(m - e) / np.linalg.norm(e)

    e13 = relerr(s13_m, sig[:, 4]); e13f = relerr(s13_f, sig[:, 4])
    e23 = relerr(s23_m, sig[:, 3]); e33 = relerr(s33_m, sig[:, 2])

    # ------------------------------------------------------------------- .dat
    hdr = ["%s -- Yu-2003 sec. 6.1 cylindrical bending, L/h = %g" % (case, a / h),
           "material EL/ET = 25 (psi set), load s3 = b3 = (p0/2) sin(px), p0 = %g" % q0,
           "  -> exact faces sigma33 = -p0/2 (bottom) / +p0/2 (top); traction "
           "residual %.2e" % bcres,
           "reference surface: MID-SURFACE (fraction = 0.5) everywhere",
           "stations: x = 0 (s13, s23), x = L/2 (s33); all values already sigma/p0",
           "plies (bottom->top): " + ", ".join(
               "%s(%.4gin/%g)" % (m, t, x) for m, t, x in zip(mats, thk, ang)),
           "",
           "RM 8x8 ABDG (rows e11,e22,g12,k11,k22,k12,2g13,2g23):"]
    hdr += ["  " + " ".join("%14.6e" % v for v in row) for row in ABDG]
    hdr += ["",
            "HARMONIC RM PLATE SOLVE (rm_cyl_bend; the DYMORE role of Yu sec. 6.2):",
            "  DOF amplitudes [U, V, W, F1, F2] = [%s]" % ", ".join("%.6g" % v for v in y),
            "  resultant sin-amplitudes [N11,N22,N12,M11,M22,M12] = [%s]"
            % ", ".join("%.6g" % v for v in R6),
            "  shear cos-amplitudes [Q1, Q2] = [%.6g, %.6g]" % (Q[0], Q[1]),
            "  statics checks: M11 = q0/p^2 = %.6g, Q1 = q0/p = %.6g (asserted)"
            % (q0 / p ** 2, q0 / p),
            "  coupling supplied by the 8x8: M12 = %.6g, Q2 = p M12 = %.6g"
            % (R6[5], Q[1]),
            "  plate w(L/2) = %.6e ; exact w = %.6e  (%.2f%%)"
            % (w_msg, w_ex, 100 * (w_msg - w_ex) / w_ex),
            "",
            "FSDT baseline: Whitney-1973 Eq.-(7) k1^2 = %.6f (A55 = %.6e);"
            % (k1sq, A55),
            "  the Whitney construction is ORTHOTROPIC -- for these angle plies "
            "it is a baseline only (the role FOSDT plays in Yu's paper)",
            "",
            "rel L2 errors vs exact:  s13 %7.3f%% (FSDT %7.2f%%)   s23 %7.3f%%"
            "   s33 %7.3f%%" % (e13, e13f, e23, e33),
            "  (s23 exact scale ||s23||_L2 = %.4g x ||s13||_L2)"
            % (np.linalg.norm(sig[:, 3]) / np.linalg.norm(sig[:, 4])),
            "s33 face closure: bottom %.4f, top %.4f (x p0; exact -0.5 / +0.5)"
            % (s33_m[0] / q0, s33_m[-1] / q0),
            "",
            "columns: z[in]  s13_msg  s13_exact  s13_fsdt  s23_msg  s23_exact"
            "  s33_msg  s33_exact  [/p0]"]
    np.savetxt(os.path.join(outdir, "yu_%s.dat" % case),
               np.column_stack([zc, s13_m, sig[:, 4], s13_f,
                                s23_m, sig[:, 3], s33_m, sig[:, 2]]),
               header="\n".join(hdr), fmt="%15.6e")

    # ------------------------------------------------------------------- plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.2, 5.0))
    ax1.plot(sig[:, 4], zc / h, "-", color="k", lw=2.0, label="exact 3-D (Pagano)")
    ax1.plot(s13_m, zc / h, ":s", color="#ff7f0e", ms=4, mfc="none", mew=1.2, lw=1.6,
             markevery=4, label="MSG-RM")
    ax1.plot(s13_f, zc / h, "--", color="#1f77b4", lw=1.4,
             label="FSDT constitutive\n(Whitney-1973 $k_1^2$)")
    ax1.set_xlabel(r"$\sigma_{13}/p_0$  at  $x=0$", fontsize=11)
    ax1.set_ylabel("$z/h$", fontsize=11)
    ax2.plot(sig[:, 3], zc / h, "-", color="k", lw=2.0)
    ax2.plot(s23_m, zc / h, ":s", color="#ff7f0e", ms=4, mfc="none", mew=1.2, lw=1.6,
             markevery=4)
    ax2.set_xlabel(r"$\sigma_{23}/p_0$  at  $x=0$", fontsize=11)
    ax3.plot(sig[:, 2], zc / h, "-", color="k", lw=2.0)
    ax3.plot(s33_m, zc / h, ":s", color="#ff7f0e", ms=4, mfc="none", mew=1.2, lw=1.6,
             markevery=4)
    ax3.set_xlabel(r"$\sigma_{33}/p_0$  at  $x=L/2$", fontsize=11)
    for ax in (ax1, ax2, ax3):
        ax.grid(alpha=0.3)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.995, 0.5),
               frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "yu_%s.png" % case), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------------- rm_8x8.out
    out8 = ["RM 8x8 ABDG for %s (mid-surface reference; rows e11,e22,g12,k11,k22,"
            "k12,2g13,2g23); L/h = 4, h = %g in" % (case, h)]
    out8 += ["  " + " ".join("%14.6e" % v for v in row) for row in ABDG]
    with open(os.path.join(outdir, "rm_8x8.out"), "w") as f8:
        f8.write("\n".join(out8) + "\n")

    print("  %s: s13 %7.3f%% (FSDT %7.2f%%)  s23 %7.3f%%  s33 %7.3f%%  "
          "w %+.2f%%  bc %.1e"
          % (case, e13, e13f, e23, e33, 100 * (w_msg - w_ex) / w_ex, bcres))
    return dict(case=case, e13=e13, e13f=e13f, e23=e23, e33=e33,
                w_msg=w_msg, w_ex=w_ex, ABDG=ABDG)
