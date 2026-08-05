"""make_curves9.py -- ALL NINE component HISTORY curves for Example 5
(step + blast): U1, U2, U3 and S11, S22, S33, S12, S13, S23 versus time,
each at its natural station, in TWO sets:
    curves/with_FSDT/     RM + 3-D solid + Abaqus-FSDT where it exists
    curves/without_FSDT/  RM + 3-D solid only

Stations and depths (matched between the models):
  U1, U2, U3    center node (a/2, b/2): direct nodal prints of all three
                jobs, every 50 us.  U1 = U2 = 0 there by symmetry -- the
                plots show the numerical zeros with their magnitude
                annotated (a symmetry check, not an omission).
  S11, S22, S12 center, top surface (axis labels say z = h/2; the exact
                sample depth is the top-ply centroid h/2 - tply/2, the
                solid's outermost integration point -- both models are
                read at that same depth).  S12 = 0 at the center by
                symmetry -- annotated like U1/U2.
  S13           (0, b/2), mid-core;   S23  (a/2, 0), mid-core.
  S33           center, mid-core (z = 0), CONSTITUTIVE: C(z) Gamma(z)
                with the applied-load ladder qt6 -- the same route as
                the whole pipeline; the momentum-integral (rho w_dd +
                shear divergence) route is REMOVED, no equilibrium
                closure anywhere.  Plotted at the SOLID DUMP INSTANTS
                (RM interpolated there) so the two models share the
                same time locations.

Per-increment RM values come from the patch fits + the LINEAR-OPERATOR
form of the recovery (through-thickness operators built once from 42 unit
drivers, then one matrix product per increment/station) with the same
Q-rescale of sigma_a3 as recover_dyn.  Solid values are the COLC/X/Y
centroid prints at every 10th increment.  FSDT contributes only the
displacement histories: a conventional shell run has NO recoverable 3-D
stress state -- the stress panels of the with_FSDT set carry that note.

Run:  python examples/opensg-rm_dynamic/ex5/make_curves9.py
"""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from recover_dyn import (step_times, node_history, patch_series, fit_patch,
                         load_ladder, solid_columns, r, S6, thick, H, XSTA,
                         Q0, CBLAST, read_elprint_tables, P)
from opensg_solid.rm_plate_1D.msg_rm_plate import msgrm_strain_at_depth

TPLY = thick[0]
Z_TOP = H / 2 - TPLY / 2               # solid top-ply centroid
STY_SOL = dict(ls="--", marker="o", color="k", lw=1.6, ms=5, mfc="none",
               mew=1.2)
STY_RM = dict(ls="-", marker="s", color="#ff7f0e", lw=1.4, ms=4,
              mfc="none", mew=1.1)
STY_FS = dict(ls="-.", marker="^", color="#2ca02c", lw=1.3, ms=5,
              mfc="none", mew=1.1)


def one_plot(path, curves, ylabel, note=None):
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for x, y, sty, lab in curves:
        ax.plot(x, y, label=lab, **sty)
    ax.set_xlabel("time [ms]", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlim(0.0, 8.0)              # display window (data files: 20 ms)
    ax.grid(alpha=0.3)
    if note:
        ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=8,
                color="0.35")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_ops(zgrid):
    """Through-thickness stress operators (nz, 6, 42) -- recovery is
    linear in the 42 drivers."""
    z6 = np.zeros(6)
    T = np.zeros((len(zgrid), 6, 42))
    for iz, z in enumerate(zgrid):
        for j in range(42):
            d = np.zeros(42)
            d[j] = 1.0
            T[iz, :, j] = msgrm_strain_at_depth(
                r, z, d[0:6], d[6:12], d[12:18], d[18:24], d[24:30],
                d[30:36], qt6=d[36:42], frame="plate")[1]
    return T


def run(kind):
    # the ex5-standard pairing (S8R shells vs C3D20 solid); the legacy
    # S4/C3D8I runs are documented fallbacks with a printed warning
    def prefer(preferred, legacy, what):
        if os.path.isfile(preferred):
            return preferred
        if os.path.isfile(legacy):
            print("  %s: %s -- using the LEGACY run %s"
                  % (kind, what, os.path.basename(legacy)))
        return legacy

    AR = os.path.join(HERE, "Abaqus_results")
    frm = prefer(os.path.join(AR, "sandwich_RM_S8R_%s.dat" % kind),
                 os.path.join(AR, "sandwich_RM_%s.dat" % kind),
                 "S8R shell history not found")
    fso = prefer(os.path.join(AR, "sandwich_SOLID_C3D20_%s.dat" % kind),
                 os.path.join(AR, "sandwich_SOLID_%s.dat" % kind),
                 "C3D20 solid history not found")
    ffs = prefer(os.path.join(AR, "sandwich_FSDT_S8R_%s.dat" % kind),
                 os.path.join(AR, "sandwich_FSDT_%s.dat" % kind),
                 "S8R FSDT history not found")
    # ---- nodal displacement histories ---------------------------------
    t_rm = step_times(frm)
    uc = node_history(frm, "NCEN")
    t_rm = t_rm[len(t_rm) - len(uc):]
    t_so_all = step_times(fso)
    uc3 = node_history(fso, "NCEN3D")
    t_so = t_so_all[len(t_so_all) - len(uc3):]
    ufs = None
    if os.path.isfile(ffs):
        t_fs = step_times(ffs)
        uf = node_history(ffs, "NCEN")
        t_fs = t_fs[len(t_fs) - len(uf):]
        ufs = uf
    # ---- per-increment RM stress recovery (operator form) -------------
    tables = read_elprint_tables(frm)
    fits = {st: fit_patch(*patch_series(tables, "PATCH" + st), XSTA[st])
            for st in ("C", "X", "Y")}
    n = min(len(t_rm), min(f.shape[0] for f in fits.values()))
    # target depths (the sigma33 integral grid is gone with the momentum
    # route; sigma33 is read constitutively at the center, z = 0)
    zint = np.linspace(-H / 2 + 1e-9, 0.0, 61)
    TC = build_ops([Z_TOP])                     # center top-ply depth
    TC0 = build_ops([0.0])                      # center mid-core (S33)
    TX = build_ops(zint)                        # X-station profile to mid
    TY = build_ops(zint)
    TXf = build_ops(np.linspace(-H / 2 + 1e-9, H / 2 - 1e-9, 61))  # rescale
    TYf = TXf

    def ft(t):
        return 1.0 if kind == "step" else np.exp(-CBLAST * t)

    def drivers(st, k):
        F = fits[st][k]
        E6 = S6 @ F[[0, 1, 2, 5, 6, 7], 0]
        dE1 = S6 @ F[[0, 1, 2, 5, 6, 7], 1]
        dE2 = S6 @ F[[0, 1, 2, 5, 6, 7], 2]
        qt6 = load_ladder(*XSTA[st], Q0 * ft(t_rm[k]))
        return np.concatenate([E6, dE1, dE2, -P * P * E6, np.zeros(6),
                               -P * P * E6, qt6])

    zfull = np.linspace(-H / 2 + 1e-9, H / 2 - 1e-9, 61)
    s11 = np.empty(n); s22 = np.empty(n); s12 = np.empty(n)
    s13 = np.empty(n); s23 = np.empty(n); s33 = np.empty(n)
    for k in range(n):
        dC, dX, dY = drivers("C", k), drivers("X", k), drivers("Y", k)
        sC = TC[0] @ dC
        s11[k], s22[k], s12[k] = sC[0], sC[1], sC[5]
        # X/Y full profiles for the rescale + the mid-core values
        pX = TXf @ dX
        pY = TYf @ dY
        Q1, Q2 = fits["X"][k][3, 0], fits["Y"][k][4, 0]
        I13 = np.trapezoid(pX[:, 4], zfull)
        I23 = np.trapezoid(pY[:, 3], zfull)
        c13 = Q1 / I13 if abs(I13) > 1e-6 * max(abs(Q1), 1) else 1.0
        c23 = Q2 / I23 if abs(I23) > 1e-6 * max(abs(Q2), 1) else 1.0
        s13[k] = (TX[-1] @ dX)[4] * c13         # z = 0 mid-core
        s23[k] = (TY[-1] @ dY)[3] * c23
        # sigma33(center, z=0) CONSTITUTIVE (qt6 ladder in dC; the
        # momentum-integral route is removed -- no equilibrium closure)
        s33[k] = (TC0[0] @ dC)[2]
    # ---- solid histories at the dumps ---------------------------------
    sol = solid_columns(fso)
    td = list(t_so_all[9::10])
    zcC, gC = sol["COLC"]
    if gC.shape[0] == len(td) + 1:
        td.append(t_so_all[-1])
    td = np.array(td[:gC.shape[0]])
    kmid = int(np.argmin(np.abs(zcC)))          # mid-core layer
    zcX, gX = sol["COLX"]
    zcY, gY = sol["COLY"]
    SOLID = {"S11": gC[:, -1, 0], "S22": gC[:, -1, 1],
             "S12": gC[:, -1, 5], "S33": gC[:, kmid, 2],
             "S13": gX[:, kmid, 4], "S23": gY[:, kmid, 3]}
    # ---- emit the two sets --------------------------------------------
    for setname, with_fs in (("with_FSDT", True), ("without_FSDT", False)):
        out = os.path.join(HERE, "curves", setname)
        os.makedirs(out, exist_ok=True)
        # displacements
        for c, lbl in ((0, "U1"), (1, "U2"), (2, "U3")):
            curves = [(1e3 * t_so, -uc3[:, c],
                       dict(STY_SOL, markevery=(0, 32)),
                       "Abaqus 3-D solid"),
                      (1e3 * t_rm, uc[:, c],
                       dict(STY_RM, markevery=(10, 30)), "OpenSG-RM")]
            if with_fs and ufs is not None:
                curves.append((1e3 * t_fs, ufs[:, c],
                               dict(STY_FS, markevery=(20, 30)),
                               "Abaqus FSDT"))
            note = None
            if c < 2:
                note = ("= 0 at the center by symmetry: max |RM| = %.1e m,"
                        " max |solid| = %.1e m (numerical zeros)"
                        % (np.abs(uc[:, c]).max(), np.abs(uc3[:, c]).max()))
            one_plot(os.path.join(out, "%s_%s.png" % (lbl, kind)), curves,
                     "%s at (a/2, b/2) [m]" % lbl, note=note)
        # stresses
        SPEC = [("S11", s11, "S11",
                 r"$\sigma_{11}(a/2,b/2,h/2)$ [MPa]", None),
                ("S22", s22, "S22",
                 r"$\sigma_{22}(a/2,b/2,h/2)$ [MPa]", None),
                ("S12", s12, "S12",
                 r"$\sigma_{12}(a/2,b/2,h/2)$ [MPa]",
                 "= 0 at the center by symmetry: max |RM| = %.1e MPa"
                 % (1e-6 * np.abs(s12).max())),
                ("S33", s33, "S33", r"$\sigma_{33}(a/2,b/2,0)$ [MPa]",
                 None),
                ("S13", s13, "S13", r"$\sigma_{13}(0,b/2,0)$ [MPa]", None),
                ("S23", s23, "S23", r"$\sigma_{23}(a/2,0,0)$ [MPa]", None)]
        for lbl, rmv, solkey, ylab, note in SPEC:
            if lbl == "S33":
                # same time locations for both models: RM interpolated
                # onto the solid dump instants, markers on every point
                t_rm_p, y_rm_p = td, np.interp(td, t_rm[:n], rmv)
                sty_rm = STY_RM
                lab_rm = "OpenSG-RM dehom\n(at the solid dump times)"
            else:
                t_rm_p, y_rm_p = t_rm[:n], rmv
                sty_rm = dict(STY_RM, markevery=(10, 30))
                lab_rm = "OpenSG-RM dehom\n(every 50 $\\mu$s)"
            curves = [(1e3 * td, 1e-6 * SOLID[solkey],
                       STY_SOL, "Abaqus 3-D solid"),
                      (1e3 * t_rm_p, 1e-6 * y_rm_p, sty_rm, lab_rm)]
            nt = note
            if with_fs:
                nt = ((note + "; " if note else "")
                      + "Abaqus FSDT: no 3-D stress output exists")
            one_plot(os.path.join(out, "%s_%s.png" % (lbl, kind)), curves,
                     ylab, note=nt)
        # the data file
        np.savetxt(os.path.join(out, "histories_%s.dat" % kind),
                   np.column_stack([1e3 * t_rm[:n], uc[:n, 0], uc[:n, 1],
                                    uc[:n, 2], s11, s22, s33, s12, s13,
                                    s23]),
                   header="t[ms] U1 U2 U3[m] S11 S22 S33 S12 S13 S23[Pa]"
                          " -- OpenSG-RM at the stations of the plots;"
                          " solid dump values in solid_%s.dat" % kind)
        np.savetxt(os.path.join(out, "solid_%s.dat" % kind),
                   np.column_stack([1e3 * td] +
                                   [SOLID[k] for k in
                                    ("S11", "S22", "S33", "S12", "S13",
                                     "S23")]),
                   header="t[ms] S11 S22 S33 S12 S13 S23 [Pa] -- Abaqus"
                          " 3-D solid at the same stations")
    print("%s: 9 curves x 2 sets written" % kind)


if __name__ == "__main__":
    for kind in ("step", "blast"):
        run(kind)
