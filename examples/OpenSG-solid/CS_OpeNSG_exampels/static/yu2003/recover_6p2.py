"""recover_6p2.py -- the post-processor of the Yu-2003 sec.-6.2 analog: OpenSG-RM
3-D recovery DRIVEN BY ABAQUS section forces, benchmarked against the Abaqus 3-D
solid model, with the inbuilt-FSDT Abaqus shell as the baseline.

Chain (Abaqus plays DYMORE, exactly Yu's sec.-6.2 architecture):
    yu_<case>_RM.inp    -> job .dat -> SF/SM at the two stations -> FF amplitudes
                        -> Es = S6 FF6, dE1 = p Es -> Eq.-63 recovery (s13, s23)
                        -> s33 by thickness equilibrium from the loaded bottom face
    yu_<case>_FSDT.inp  -> job .dat -> TSHR13/TSHR23 (Abaqus's own equilibrium-
                        based composite-shell estimate) = the inbuilt FOSDT curves
    yu_<case>_SOLID.inp -> job .dat -> centroidal S along the COL0/COLM element
                        columns = the BENCHMARK profiles
The exact Pagano profiles are drawn as a thin reference (they also validate the
solid model itself).  NO Pagano content enters the recovery chain: the FF comes
from Abaqus, the recovery from the MSG 8x8.

Usage: put the three job .dat files into examples/yu2003/<case>/Abaqus_Plate/
(names yu_<case>_RM.dat / _FSDT.dat / _SOLID.dat), then
    python examples/yu2003/recover_6p2.py [--case case1]
Writes yu_<case>_6p2.dat + yu_<case>_6p2.png into the case folder.
"""
import argparse
import os
import re
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
from yu_layups import MATERIAL_DB, LAYUPS, H, L_SPAN, P0              # noqa: E402


def read_elprint_tables(dat_path):
    """Parse every *EL PRINT table of an Abaqus job .dat into labeled arrays.

    Variables
    ---------
    dat_path   the job .dat text file
    lines      its content split into lines
    tables     the result: {(elset, tuple(labels)): rows}, rows = (n, 1+nval)
               arrays [element id, values...]; multiple tables for the same
               elset (e.g. SF and TSHR requests) stay separate entries; a table
               Abaqus prints as "ALL VALUES IN THIS TABLE ARE ZERO" has no data
               rows and is dropped
    elset      the current "ELEMENT SET <NAME>" context.  Abaqus WRAPS long
               preambles: "... FOR ELEMENT TYPE C3D8I AND ELEMENT" / "SET COL0"
               on the next line -- both the one-line and the wrapped form are
               matched (pend flags the dangling "...ELEMENT" line)
    labels     the column names taken from the header line that lists the
               requested identifiers (tokens like S11/SF1/TSHR13); Abaqus column
               ORDER is version-dependent -- everything downstream selects
               columns BY LABEL, never by position
    rows       data lines "<eid> [pt] [sec] [OR] <values...>": the leading token
               must be an integer id; every remaining FLOAT-parsable token is
               kept and non-numeric footnote tokens (the "OR" orientation flag
               of oriented solids) are dropped, so a row is [eid, numbers...]
               with the labeled values LAST; summary lines (MAXIMUM/MINIMUM/
               ELEMENT) start with a word and are skipped

    Returns the tables dict.
    """
    with open(dat_path, errors="replace") as f:
        lines = f.read().splitlines()
    tables = {}
    elset, labels, key, pend = None, None, None, False
    for ln in lines:
        m = re.search(r"ELEMENT SET\s+(\S+)", ln)
        if m:
            elset, labels, key, pend = m.group(1), None, None, False
            continue
        if pend:
            m = re.match(r"\s*SET\s+(\S+)\s*$", ln) or \
                re.match(r"\s*([A-Z][A-Z0-9_]*)\s*$", ln)
            pend = False
            if m:
                elset, labels, key = m.group(1), None, None
                continue
        if re.search(r"AND ELEMENT\s*$", ln) or \
                re.search(r"ELEMENT SET\s*$", ln):
            # Abaqus wraps the preamble at either point: "...AND ELEMENT" /
            # "SET COL0" (solids) or "...AND ELEMENT SET" / "PATCHC" (shells)
            pend = True
            continue
        if "TABLE IS PRINTED" in ln:
            # any OTHER table (node prints etc.) ends the element-table context;
            # without this the NMID/NW node rows (different width) would be
            # appended to the previous element table and make it ragged
            elset, labels, key = None, None, None
            continue
        toks = ln.split()
        if elset and not labels and toks and any(
                re.fullmatch(r"(S|SF|SM|TSHR|E|SE|COOR|COORD)\d+", t)
                for t in toks):
            labels = [t for t in toks if re.fullmatch(r"[A-Z]+\d+", t)]
            key = (elset, tuple(labels))
            tables.setdefault(key, [])
            continue
        if key and toks and re.fullmatch(r"\d+", toks[0]):
            vals = [float(toks[0])]
            for t in toks[1:]:
                try:
                    vals.append(float(t))
                except ValueError:
                    pass
            if len(vals) > 1:
                tables[key].append(vals)
    return {k: np.array(v) for k, v in tables.items() if len(v)}


def pick(tables, elset, want):
    """Select columns by LABEL from the parsed tables.

    Variables: tables = read_elprint_tables output; elset = the element-set name;
    want = the label list to extract (e.g. ["S13", "S23", "S33"]).  Finds the
    table of that elset containing ALL wanted labels; leading non-data columns
    (element id, section point) are skipped by aligning the label list to the
    LAST len(labels) columns of the rows.  Returns (ids, cols) with cols one
    array per wanted label.
    """
    for (es, labels), rows in tables.items():
        if es == elset and all(w in labels for w in want):
            ofs = rows.shape[1] - len(labels)
            ids = rows[:, 0].astype(int)
            return ids, [rows[:, ofs + labels.index(w)] for w in want]
    raise KeyError("no table for elset %s with %s" % (elset, want))


def ff_from_shell_dat(dat_path, a, nel=100):
    """FF amplitudes from the strip-shell job .dat (the DYMORE-output step).

    Variables
    ---------
    a, nel      span and element count of the strip deck (centroid corrections)
    p           wavenumber pi/a
    xc_e, xc_m  the EEND / EMID element centroids dx/2 and a/2 - dx/2
    sfm, smm    EMID section forces/moments (mean over the section-point rows --
                SF/SM are section quantities, printed once per element here)
    sfe         EEND section forces
    R6amp       (6,) sin-family resultant AMPLITUDES [N11,N22,N12,M11,M22,M12]:
                the EMID values divided by sin(p xc_m)
    Qamp        (2,) cos-family shear amplitudes [Q1, Q2]: EEND SF4/SF5 divided
                by cos(p xc_e)

    Returns (R6amp, Qamp).
    """
    p = np.pi / a
    dx = a / nel
    xc_e, xc_m = 0.5 * dx, a / 2 - 0.5 * dx
    t = read_elprint_tables(dat_path)
    _, sfm = pick(t, "EMID", ["SF1", "SF2", "SF3", "SF4", "SF5"])
    _, smm = pick(t, "EMID", ["SM1", "SM2", "SM3"])
    _, sfe = pick(t, "EEND", ["SF4", "SF5"])
    R6amp = np.array([np.mean(sfm[0]), np.mean(sfm[1]), np.mean(sfm[2]),
                      np.mean(smm[0]), np.mean(smm[1]), np.mean(smm[2])])
    R6amp /= np.sin(p * xc_m)
    Qamp = np.array([np.mean(sfe[0]), np.mean(sfe[1])]) / np.cos(p * xc_e)
    return R6amp, Qamp


def solid_profiles(dat_path, thick, angles_deg, a, nx=160, nz_per_ply=10):
    """Benchmark through-thickness profiles from the solid job .dat.

    Variables
    ---------
    thick, angles_deg, the deck parameters (element -> z/ply mapping and the
    a, nx, nz_per_ply  harmonic centroid corrections must match make_abaqus_6p2)
    p, dx           wavenumber and span element size
    zpl, zk         ply interfaces and node planes from the bottom face
    zc              (nzt,) element-centroid heights from the MID-surface [in]
    t               the parsed tables; COL0 rows ascending element id = ascending
                    k (eid(0, k) = 1 + k nx), same for COLM
    th, cth, sth    the PLY ANGLE of each thickness element and its cos/sin:
                    with a *SOLID SECTION ... ORIENTATION Abaqus prints S in the
                    PLY LOCAL frame (the "OR" footnote) -- e.g. a 90.5-deg ply
                    reports local S13 ~ global sigma_23 and local S23 ~
                    -sigma_13.  The transverse-shear pair rotates back with
                    sigma_13 = c S13loc - s S23loc, sigma_23 = s S13loc +
                    c S23loc (sigma_33 is rotation-invariant about the normal);
                    verified against exact to 4 digits on all three Yu cases
    s13, s23        COL0 GLOBAL-frame shears divided by cos(p dx/2)
    s33             COLM centroidal stresses divided by sin(p (a/2 + dx/2))

    Returns (zc, s13, s23, s33).
    """
    p = np.pi / a
    dx = a / nx
    zpl = np.concatenate([[0.0], np.cumsum(np.asarray(thick, float))])
    zk = np.concatenate([np.linspace(zpl[m], zpl[m + 1], nz_per_ply + 1)[:-1]
                         for m in range(len(thick))] + [[zpl[-1]]])
    zc = 0.5 * (zk[:-1] + zk[1:]) - zpl[-1] / 2
    lay = np.searchsorted(zpl[1:-1], zc + zpl[-1] / 2, side="right")
    th = np.deg2rad(np.asarray(angles_deg, float))[lay]
    cth, sth = np.cos(th), np.sin(th)
    t = read_elprint_tables(dat_path)
    _, c0 = pick(t, "COL0", ["S13", "S23"])
    _, cm = pick(t, "COLM", ["S33"])
    s13 = (cth * c0[0] - sth * c0[1]) / np.cos(p * 0.5 * dx)
    s23 = (sth * c0[0] + cth * c0[1]) / np.cos(p * 0.5 * dx)
    s33 = cm[0] / np.sin(p * (a / 2 + 0.5 * dx))
    return zc, s13, s23, s33


def tshr_profile(dat_path, thick, a, nel=100):
    """Abaqus inbuilt-FSDT transverse-shear estimate TSHR13/TSHR23 at x = 0.

    Variables: thick/a/nel as the FSDT deck; rows = the EEND TSHR table, columns
    [eid, PT, SEC, TSHR13, TSHR23] (PT = the S4 in-plane integration point 1-4,
    SEC = the composite section point, 3 per ply numbered from the BOTTOM ply
    up); sec/v13/v23 = the SEC index and the values AVERAGED over the 4 in-plane
    points of the element, divided by cos(p dx/2) (the element sits at x =
    dx/2); z_pt = the section-point heights from the mid-surface (ply bottom /
    mid / top).  Returns (z_pt, tshr13, tshr23) or None when the table is absent
    or Abaqus printed it as all-zero (TSHR is not populated for every shell
    element type -- plain S4 gives zeros; the caller then simply omits the
    inbuilt curve).
    """
    p = np.pi / a
    dx = a / nel
    try:
        t = read_elprint_tables(dat_path)
        for (es, labels), rows in t.items():
            if es == "EEND" and "TSHR13" in labels and rows.shape[1] >= 5:
                break
        else:
            return None
    except OSError:
        return None
    zpl = np.concatenate([[0.0], np.cumsum(np.asarray(thick, float))])
    z_all = np.array([zpl[m] + f * (zpl[m + 1] - zpl[m])
                      for m in range(len(thick)) for f in (0.0, 0.5, 1.0)])
    z_all -= zpl[-1] / 2
    secs = sorted(set(rows[:, 2].astype(int)))
    z_pt, v13, v23 = [], [], []
    for s in secs:
        sel = rows[rows[:, 2].astype(int) == s]
        if s - 1 < len(z_all):
            z_pt.append(z_all[s - 1])
            v13.append(np.mean(sel[:, 3]))
            v23.append(np.mean(sel[:, 4]))
    if not z_pt or np.allclose(v13, 0.0):
        return None
    c = np.cos(p * 0.5 * dx)
    return np.array(z_pt), np.array(v13) / c, np.array(v23) / c


def run_case(case):
    """One 6.2 comparison: Abaqus-FF-driven OpenSG-RM recovery vs the Abaqus 3-D
    solid benchmark and the inbuilt-FSDT shell, with exact Pagano as reference.

    Variables
    ---------
    adir            <case>/Abaqus_Plate/ holding yu_<case>_{RM,FSDT,SOLID}.dat
    lay, thk, ang,  the laminate; a/h/p/q0 the geometry and load (P0 = 1 ->
    mats            everything is already sigma/p0)
    R6amp, Qamp     the Abaqus FF amplitudes (ff_from_shell_dat) -- the FULL
                    resultant set DRIVING the recovery, coupling terms included
    r, S6           rm_plate_msg result and inv(A6): strains from resultants
    ze, sige        the exact profile grid and stress amplitudes (n, 6)
    Q1_ex, M11_ex   printed checks only: section resultants integrated from the
                    exact 3-D stresses (both land on statics q0/p and q0/p^2)
    dE1_0           the recovery input at x = 0: p S6 R6amp (every sin-family
                    resultant gradient at x = 0 is p times its amplitude;
                    E6(0) = 0)
    zc              recovery grid from the mid-surface (81 points per ply)
    s13_r, s23_r    the Eq.-63 recovered amplitudes at each zc
    s33_r           thickness-equilibrium integral d(s33)/dz = p s13 from the
                    loaded bottom face -p0/2
    zs, s13_s, ...  the solid-benchmark profiles (solid_profiles)
    tsh             the inbuilt-FSDT TSHR profile or None
    ex, ze, sige    the exact reference (thin gray line; also validates the
                    solid model itself)
    e13, e23, e33   rel L2 errors of the recovery vs the SOLID benchmark
                    (interpolated onto the solid zc grid)

    Writes yu_<case>_6p2.dat + yu_<case>_6p2.png into the case folder.
    """
    adir = os.path.join(HERE, case, "Abaqus_Plate")
    lay = LAYUPS[case]
    thk = lay["thick"]; ang = lay["angles"]; mats = lay["mat_names"]
    a = L_SPAN; h = H; p = np.pi / a; q0 = P0

    R6amp, Qamp = ff_from_shell_dat(
        os.path.join(adir, "yu_%s_RM.dat" % case), a)
    r = rm_plate_msg(thk, ang, mats, MATERIAL_DB, fraction=0.5)
    S6 = np.linalg.inv(np.asarray(r["A6"]))
    ex = ExactCyl(thk, ang, mats, MATERIAL_DB, a, q0=0.5 * q0, q_bot=-0.5 * q0)
    ze, sige, _, _ = ex.profile(n_per_layer=81)
    # FF for the recovery = the FULL resultant set from the ABAQUS RM shell
    # (R6amp): dE1 = p S6 R6amp since every sin-family resultant gradient at
    # x = 0 is p times its amplitude.  This carries the coupling resultants
    # (N22, M22, M12) that the angle plies need for sigma_23 -- the Q1-only
    # statics input loses them (case2 s23 92% vs 16%).  The exact-section
    # integrals Q1 = int s13 dz and M11 = int s11 z dz are kept as printed
    # checks of the statics anchors q0/p and q0/p^2.
    Q1_ex = float(np.trapezoid(sige[:, 4], ze))
    M11_ex = float(np.trapezoid(sige[:, 0] * ze, ze))
    dE1_0 = p * (S6 @ R6amp)
    z6 = np.zeros(6)
    zpl = np.concatenate([[0.0], np.cumsum(np.asarray(thk, float))]) - h / 2
    zc = np.concatenate([np.linspace(zpl[m] + 1e-9, zpl[m + 1] - 1e-9, 81)
                         for m in range(len(thk))])
    s13_r = np.empty_like(zc); s23_r = np.empty_like(zc)
    for i, z in enumerate(zc):
        Sig = msgrm_strain_at_depth(r, z, np.zeros(6), dE1_0, z6)[1]
        s13_r[i] = Sig[4]; s23_r[i] = Sig[3]
    s33_r = -0.5 * q0 + np.concatenate(
        [[0.0], np.cumsum(0.5 * p * (s13_r[1:] + s13_r[:-1]) * np.diff(zc))])

    zs, s13_s, s23_s, s33_s = solid_profiles(
        os.path.join(adir, "yu_%s_SOLID.dat" % case), thk, ang, a)
    tsh = tshr_profile(os.path.join(adir, "yu_%s_FSDT.dat" % case), thk, a)

    def relerr(zm, m, zb, b):
        """Rel L2 error [%] of model (zm, m) vs benchmark (zb, b), interpolated
        onto the benchmark grid."""
        mi = np.interp(zb, zm, m)
        return 100 * np.linalg.norm(mi - b) / np.linalg.norm(b)

    e13 = relerr(zc, s13_r, zs, s13_s)
    e23 = relerr(zc, s23_r, zs, s23_s)
    e33 = relerr(zc, s33_r, zs, s33_s)
    # the benchmark's own credentials: solid vs the exact Pagano solution
    es13 = relerr(zs, s13_s, ze, sige[:, 4])
    es23 = relerr(zs, s23_s, ze, sige[:, 3])
    es33 = relerr(zs, s33_s, ze, sige[:, 2])

    hdr = ["%s -- Yu-2003 sec.-6.2 analog: Abaqus as the 2-D solver (DYMORE role)"
           % case,
           "chain: Abaqus RM-shell SF/SM -> FF -> dE1 = p S6 FF6 -> OpenSG-RM",
           "Eq.-63 recovery; benchmark = Abaqus 3-D solid strip (cross-check",
           "numbers); the full FF carries the coupling N22/M22/M12 the angle",
           "plies need for sigma_23 (the Q1-only statics input loses them)",
           "FF amplitudes from Abaqus [N11,N22,N12,M11,M22,M12] = [%s]"
           % ", ".join("%.6g" % v for v in R6amp),
           "  with Abaqus Q = [%.6g, %.6g]" % (Qamp[0], Qamp[1]),
           "exact-section anchors: Q1 = int s13 dz = %.6g (q0/p = %.6g);"
           % (Q1_ex, q0 / p),
           "  M11 = int s11 z dz = %.6g (q0/p^2 = %.6g)" % (M11_ex, q0 / p ** 2),
           "rel L2 errors of the recovery vs the SOLID benchmark:  "
           "s13 %7.3f%%  s23 %7.3f%%  s33 %7.3f%%" % (e13, e23, e33),
           "benchmark credentials, solid vs exact Pagano:  "
           "s13 %7.3f%%  s23 %7.3f%%  s33 %7.3f%%" % (es13, es23, es33),
           "TSHR13/23 present: %s" % ("yes" if tsh else "no"),
           "",
           "columns: z[in]  s13_rec  s23_rec  s33_rec  [/p0]  "
           "(solid/TSHR profiles carry their own grids -- see the .png)"]
    np.savetxt(os.path.join(HERE, case, "yu_%s_6p2.dat" % case),
               np.column_stack([zc, s13_r, s23_r, s33_r]),
               header="\n".join(hdr), fmt="%15.6e")

    # the plot compares against the EXACT Pagano curves only -- the Abaqus solid
    # is kept as the parsed/validated cross-check in the .dat numbers, not drawn
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.2, 5.0))
    for ax, (zm, mth), exact, lbl in (
            (ax1, (zc, s13_r), sige[:, 4], r"$\sigma_{13}/p_0$  at  $x=0$"),
            (ax2, (zc, s23_r), sige[:, 3], r"$\sigma_{23}/p_0$  at  $x=0$"),
            (ax3, (zc, s33_r), sige[:, 2], r"$\sigma_{33}/p_0$  at  $x=L/2$")):
        ax.plot(exact, ze / h, "-", color="k", lw=2.0,
                label="exact 3-D (Pagano)" if ax is ax1 else None)
        ax.plot(mth, zm / h, ":s", color="#ff7f0e", ms=4, mfc="none", mew=1.2,
                lw=1.6, markevery=6,
                label="OpenSG-RM recovery\n(FF from Abaqus RM shell)"
                if ax is ax1 else None)
        ax.set_xlabel(lbl, fontsize=11)
        ax.grid(alpha=0.3)
    if tsh:
        ax1.plot(tsh[1], tsh[0] / h, "--", color="#1f77b4", lw=1.4,
                 label="Abaqus inbuilt FSDT (TSHR)")
        ax2.plot(tsh[2], tsh[0] / h, "--", color="#1f77b4", lw=1.4)
    ax1.set_ylabel("$z/h$", fontsize=11)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.995, 0.5),
               frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, case, "yu_%s_6p2.png" % case), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("  %s (vs solid benchmark): s13 %7.3f%%  s23 %7.3f%%  s33 %7.3f%%"
          "   [solid vs exact: %5.2f / %5.2f / %5.2f%%]"
          % (case, e13, e23, e33, es13, es23, es33))
    return dict(case=case, e13=e13, e23=e23, e33=e33)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default=None, help="case1/case2/case3 (default: all "
                    "with an Abaqus_Plate folder present)")
    a_ = ap.parse_args()
    for name in LAYUPS:
        if a_.case and name != a_.case:
            continue
        if not os.path.isdir(os.path.join(HERE, name, "Abaqus_Plate")):
            print("  %s: no Abaqus_Plate folder yet -- run the decks first" % name)
            continue
        run_case(name)
