"""recover_dyn.py -- the dynamic recovery post-processor of the Nayak Ex.5
sandwich benchmark (README.md): OpenSG-RM per-time-step 3-D recovery from the
Abaqus RM-shell job, judged against the Abaqus 3-D solid of the same problem.

Inputs (Abaqus_results/, copied back from the Abaqus machine, per pulse kind):
    sandwich_RM_<kind>.dat      400 fixed 50-us increments; per increment the
                                center-node U (NSET NCEN) + SF/SM and COORD at
                                the three 2x2-element patches PATCHC/X/Y
    sandwich_SOLID_<kind>.dat   automatic increments (~430); center U every
                                increment + centroidal S at the element
                                columns COLC/COLX/COLY every 10th increment

Outputs per kind (this folder):
    dyn_<kind>_w.png            center deflection history, RM vs solid
    dyn_<kind>_profiles.png     through-thickness paths at the RM peak-|w|
                                instant: s13 (x-edge), s23 (y-edge), s33 and
                                s11 (center) -- RM recovery vs solid column
    dyn_<kind>_iface.png        face-core interface s13 history at the x-edge
                                station, RM (every 50 us) vs solid (dumps)
    dyn_<kind>.dat              the numbers (peaks, %err, closure, cost)

Chain per increment: patch-fit the printed SF/SM over the patch COORDs
(quadratic LSQ -> resultant values + first gradients at the station), convert
with the section 8x8 (in-plane 6x6 inverse), take the mode-consistent second
gradients (double-sine load: E,11 = E,22 = -p^2 E, E,12 = 0 at the stations),
add the DOUBLE-SINE load ladder of the applied face pressure, and recover
with msgrm_strain_at_depth; sigma_33 at the center is CONSTITUTIVE --
C(z) Gamma(z) with the applied-load ladder qt6 carrying the face
tractions, the same route as the whole pipeline (the momentum-integral
route is REMOVED: no equilibrium closure anywhere).  Inertia enters the
constitutive route only through the FF field itself; the interior
rho*u3_dd contribution of the old momentum integral is not represented,
which is an accepted limitation of the method, not a defect of the run.

Conventions reconciled here (do not "simplify" these away):
  * Abaqus solid loads face P2 of the top layer -> load acts -z, while the
    RM shell P acts +z: the LINEAR solid response is sign-flipped wholesale.
  * Solid S is printed in each ply's *ORIENTATION frame ("OR" footnote): the
    90-deg plies need s11<->s22, s12->-s12, s13=-S23', s23=+S13'.
  * The solid job ran AUTOMATIC time increments (~430, non-uniform): times
    are parsed from the "STEP TIME COMPLETED" lines, never assumed.
  * Sig component order of the recovery is 3-D Voigt (11,22,33,23,13,12);
    the solid tables are reordered to match at parse time.

Run:  python examples/opensg-rm_dynamic/recover_dyn.py [--kind step|blast]
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
try:
    import opensg_solid                      # pip install -e . -- nothing to do
except ImportError:                          # fall back to the in-repo source tree
    ROOT = HERE
    while not os.path.isdir(os.path.join(ROOT, "src", "opensg_solid")):
        parent = os.path.dirname(ROOT)
        if parent == ROOT:                   # hit the filesystem root
            raise ImportError(
                "opensg_solid not installed and no src/ found above " + HERE)
        ROOT = parent
    sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "examples", "OpenSG-solid",
                                "CS_OpeNSG_exampels", "static", "yu2003"))

from opensg_solid.rm_plate_1D.segment_plate import read_plate_sg_yaml
from opensg_solid.rm_plate_1D.msg_rm_plate import rm_plate_msg, msgrm_strain_at_depth
from recover_6p2 import read_elprint_tables                     # noqa: E402

A = 1.524                      # plate side a = b = 10 h [m]
NX = 20                        # elements per side -> dx = 0.0762 m
DXE = A / NX
Q0 = 68.9476e6                 # load intensity [Pa]
CBLAST = 330.0                 # blast decay e^(-c t)
NZC = 8                        # solid core sub-layers
P = np.pi / A                  # both wavenumbers (square plate)

inp = read_plate_sg_yaml(os.path.join(HERE, "sandwich_sg.yaml"))
r = rm_plate_msg(inp["thick"], inp["angles"], inp["mat_names"],
                 inp["material_db"], fraction=inp["fraction"])
S6 = np.linalg.inv(np.asarray(r["A6"]))
thick = inp["thick"]
H = float(sum(thick))
# stations = where each solid column actually sits (COLX/COLY are the first
# element off the edge: x = dx/2, cos(p dx/2) = 0.997 of the edge value)
XSTA = {"C": (A / 2, A / 2), "X": (DXE / 2, A / 2), "Y": (A / 2, DXE / 2)}


def step_times(dat_path):
    """The end-of-increment times, parsed (the solid runs are automatic)."""
    ts = []
    with open(dat_path, errors="replace") as f:
        for ln in f:
            m = re.match(r"\s*STEP TIME COMPLETED\s+([0-9.E+-]+)", ln)
            if m:
                ts.append(float(m.group(1)))
    return np.array(ts)


def node_history(dat_path, nset):
    """All printed rows of the single-node *NODE PRINT nset, in increment
    order: (n_rows, nval).  A "ALL VALUES IN THIS TABLE ARE ZERO" table
    contributes NO row (align times from the END)."""
    rows = []
    with open(dat_path, errors="replace") as f:
        lines = f.read().splitlines()
    active, labels_seen = False, False
    for ln in lines:
        if "NODE SET %s" % nset in ln and "TABLE IS PRINTED" in ln:
            active, labels_seen = True, False
            continue
        toks = ln.split()
        if not toks:
            continue
        if active and not labels_seen:
            if toks[0] == "NODE":
                labels_seen = True
            continue
        if active and labels_seen and re.fullmatch(r"\d+", toks[0]):
            vals = []
            for t in toks[1:]:
                try:
                    vals.append(float(t))
                except ValueError:
                    pass
            if vals:
                rows.append(vals)
            active = False
    return np.array(rows)


def patch_series(tables, name):
    """One patch elset: SF/SM series (n_inc, 16, 8) in the resultant order
    (N11,N22,N12,Q1,Q2,M11,M22,M12) + the 16 integration-point coordinates
    (constant in time -> taken from the first COORD table occurrence)."""
    sf = None; xy = None
    for (es, labels), rows in tables.items():
        if es != name:
            continue
        ofs = rows.shape[1] - len(labels)
        if "SF1" in labels:
            idx = [labels.index(k) + ofs for k in
                   ("SF1", "SF2", "SF3", "SF4", "SF5", "SM1", "SM2", "SM3")]
            sf = rows[:, idx].reshape(-1, 16, 8)
        if "COORD1" in labels:
            idx = [labels.index(k) + ofs for k in ("COORD1", "COORD2")]
            xy = rows[:len(rows) - len(rows) % 16][:16, idx]
    if sf is None or xy is None:
        raise RuntimeError("patch %s: SF/SM or COORD table missing" % name)
    return sf, xy


def fit_patch(sf, xy, x0):
    """Quadratic LSQ of the 8 resultants over the 16 patch points, expanded
    about the station x0: per-increment (val, d/dx, d/dy) -> (n_inc, 8, 3)."""
    xi = xy[:, 0] - x0[0]
    eta = xy[:, 1] - x0[1]
    B = np.column_stack([np.ones(16), xi, eta, xi * eta, xi ** 2, eta ** 2])
    n_inc = sf.shape[0]
    out = np.empty((n_inc, 8, 3))
    for k in range(n_inc):
        C, *_ = np.linalg.lstsq(B, sf[k], rcond=None)
        out[k, :, 0] = C[0]
        out[k, :, 1] = C[1]
        out[k, :, 2] = C[2]
    return out


def load_ladder(x, y, ampl):
    """[q, q,1, q,2, q,11, q,12, q,22] of q = ampl sin(px)sin(py) at (x,y)."""
    s1, c1 = np.sin(P * x), np.cos(P * x)
    s2, c2 = np.sin(P * y), np.cos(P * y)
    return ampl * np.array([s1 * s2, P * c1 * s2, P * s1 * c2,
                            -P * P * s1 * s2, P * P * c1 * c2,
                            -P * P * s1 * s2])


def solid_columns(dat_path):
    """{COLC/COLX/COLY: (zc, g)} with g (n_dump, 16, 6) element-centroid
    stresses in the RECOVERY frame: 3-D Voigt order (11,22,33,23,13,12),
    rotated from the printed ply-local frames to global, and sign-flipped
    (the solid deck loads -z, the shell +z).  zc = layer centroids about the
    mid-plane, ascending bottom -> top (= ascending element id, k-major)."""
    t = read_elprint_tables(dat_path)
    zk = [0.0]
    for tt in thick[:4]:
        zk.append(zk[-1] + tt)
    for _ in range(NZC):
        zk.append(zk[-1] + thick[4] / NZC)
    for tt in thick[5:]:
        zk.append(zk[-1] + tt)
    zk = np.array(zk)
    zc = 0.5 * (zk[:-1] + zk[1:]) - H / 2
    lay_of = [0, 1, 2, 3] + [4] * NZC + [5, 6, 7, 8]
    ang = np.array([inp["angles"][m] for m in lay_of])
    out = {}
    for name, ncell in (("COLC", 4), ("COLX", 2), ("COLY", 2)):
        for (es, labels), rows in t.items():
            if es != name or "S11" not in labels:
                continue
            ofs = rows.shape[1] - len(labels)
            idx = [labels.index(k) + ofs for k in
                   ("S11", "S22", "S33", "S12", "S13", "S23")]
            v = rows[:, idx].reshape(-1, len(zc), ncell, 6).mean(axis=2)
            g = np.empty_like(v)          # -> Voigt (11,22,33,23,13,12)
            for kz, a_ in enumerate(ang):
                loc = abs(abs(a_) - 90.0) < 1e-6
                s11 = v[:, kz, 1] if loc else v[:, kz, 0]
                s22 = v[:, kz, 0] if loc else v[:, kz, 1]
                s12 = -v[:, kz, 3] if loc else v[:, kz, 3]
                s13 = -v[:, kz, 5] if loc else v[:, kz, 4]
                s23 = v[:, kz, 4] if loc else v[:, kz, 5]
                g[:, kz, 0], g[:, kz, 1], g[:, kz, 2] = s11, s22, v[:, kz, 2]
                g[:, kz, 3], g[:, kz, 4], g[:, kz, 5] = s23, s13, s12
            out[name] = (zc, -g)          # load-direction flip
    return out


def wallclock(dat_path):
    """The job's total wallclock seconds from the .dat closing summary."""
    w = None
    with open(dat_path, errors="replace") as f:
        for ln in f:
            m = re.search(r"WALLCLOCK TIME \(SEC\)\s*=\s*([0-9.]+)", ln)
            if m:
                w = float(m.group(1))
    return w


def model_label(dat_path, nelem):
    """'400 S8R (7686 dof)'-style cost label, read from the .dat itself."""
    name = os.path.basename(dat_path)
    for selt in ("S8R", "C3D20", "C3D8I", "S4"):
        if selt in name:
            break
    else:
        selt = "C3D8I" if "SOLID" in name else "S4"   # the legacy runs
    ndof = "?"
    with open(dat_path, errors="replace") as f:
        for ln in f:
            m = re.search(r"TOTAL NUMBER OF VARIABLES IN THE MODEL\s+(\d+)",
                          ln)
            if m:
                ndof = m.group(1)
                break
    return "%d %s (%s dof)" % (nelem, selt, ndof)


def prefer(preferred, legacy, what, kind):
    """The ex5-standard .dat when present, else the legacy run + warning."""
    if os.path.isfile(preferred):
        return preferred
    if os.path.isfile(legacy):
        print("  %s: %s -- using the LEGACY run %s (generate the deck with"
              " make_abaqus_dyn.py and run it in Abaqus)"
              % (kind, what, os.path.basename(legacy)))
    return legacy


def run(kind):
    # the ex5-standard pairing: S8R shell histories vs the C3D20 full-
    # integration solid (the legacy S4/C3D8I runs are ~20 % too stiff --
    # S4 via its shear treatment, C3D8I via the 80:1-aspect face plies)
    AR = os.path.join(HERE, "Abaqus_results")
    frm = prefer(os.path.join(AR, "sandwich_RM_S8R_%s.dat" % kind),
                 os.path.join(AR, "sandwich_RM_%s.dat" % kind),
                 "S8R shell history not found", kind)
    fso = prefer(os.path.join(AR, "sandwich_SOLID_C3D20_%s.dat" % kind),
                 os.path.join(AR, "sandwich_SOLID_%s.dat" % kind),
                 "C3D20 solid history not found", kind)
    if not (os.path.isfile(frm) and os.path.isfile(fso)):
        print("  %s: job .dat files not present yet -- skipped" % kind)
        return

    # ---- histories: times parsed, node rows tail-aligned ------------------
    t_rm_all = step_times(frm)
    t_so_all = step_times(fso)
    uc = node_history(frm, "NCEN")
    uc3 = node_history(fso, "NCEN3D")
    t_rm = t_rm_all[len(t_rm_all) - len(uc):]
    t_so = t_so_all[len(t_so_all) - len(uc3):]
    w_rm = uc[:, 2]
    w_so = -uc3[:, 2]                       # load-direction flip

    tables = read_elprint_tables(frm)
    fits = {}
    for st in ("C", "X", "Y"):
        sf, xy = patch_series(tables, "PATCH" + st)
        fits[st] = fit_patch(sf, xy, XSTA[st])
    n_inc = min(len(t_rm), len(w_rm), min(f.shape[0] for f in fits.values()))

    def ampl(t):
        return Q0 * (1.0 if kind == "step" else np.exp(-CBLAST * t))

    def recover(st, k, zpts):
        """RM recovery at station st, increment k, depths zpts: (nz, 6) Sig
        in Voigt (11,22,33,23,13,12) + the FITTED shear resultants (Q1, Q2).

        DYNAMIC sigma_a3: the E,alpha-driven chain reproduces the shear a
        plate in STATIC equilibrium carries (Q = M,x); dynamically the
        thickness-shear ringing lives in the plate's ACTUAL resultant, so
        the through-thickness DISTRIBUTION from the SG is rescaled to carry
        the MEASURED Q1/Q2 (SF4/SF5) -- in the static limit the scale -> 1.
        Face-zero and interface continuity survive a scalar rescale."""
        F = fits[st][k]
        E6 = S6 @ F[[0, 1, 2, 5, 6, 7], 0]
        dE1 = S6 @ F[[0, 1, 2, 5, 6, 7], 1]
        dE2 = S6 @ F[[0, 1, 2, 5, 6, 7], 2]
        x0 = XSTA[st]
        qt6 = load_ladder(x0[0], x0[1], ampl(t_rm[k]))
        d11, d22 = -P * P * E6, -P * P * E6   # mode-consistent 2nd gradients
        z66 = np.zeros(6)
        out = np.empty((len(zpts), 6))
        for i, z in enumerate(zpts):
            out[i] = msgrm_strain_at_depth(r, z, E6, dE1, dE2, d11, z66, d22,
                                           qt6=qt6, frame="plate")[1]
        Q1, Q2 = F[3, 0], F[4, 0]
        I13 = np.trapz(out[:, 4], zpts)
        I23 = np.trapz(out[:, 3], zpts)
        if abs(I13) > 1e-6 * max(abs(Q1), 1.0):
            out[:, 4] *= Q1 / I13
        if abs(I23) > 1e-6 * max(abs(Q2), 1.0):
            out[:, 3] *= Q2 / I23
        return out, Q1 / I13, Q2 / I23

    # ---- profiles at the RM peak-|w| instant ------------------------------
    kpk = int(np.argmax(np.abs(w_rm[:n_inc])))
    tpk = t_rm[kpk]
    zg = np.concatenate([np.linspace(sum(thick[:m]) + 1e-9,
                                     sum(thick[:m + 1]) - 1e-9, 15)
                         for m in range(len(thick))]) - H / 2
    prof, scq = {}, {}
    for st in ("C", "X", "Y"):
        prof[st], sc13, sc23 = recover(st, kpk, zg)
        scq[st] = (sc13, sc23)
    # sigma_33(center) CONSTITUTIVE: the recovery at the C station already
    # carries it -- C(z) Gamma(z) with the qt6 face-load ladder and the
    # mode-consistent second gradients (row-split transverse rows).  The
    # momentum-integral route (cumsum of p*(s13|X + s23|Y) + rho*w_dd) is
    # REMOVED; its interior rho*u3_dd content (~40 % of q0 at peak) is the
    # one contribution the constitutive route does not see.
    s33 = prof["C"][:, 2]

    sol = solid_columns(fso)
    td = list(t_so_all[9::10])                # element dumps every 10th inc
    ndump = sol["COLC"][1].shape[0]
    if ndump == len(td) + 1:                  # + the final increment's dump
        td.append(t_so_all[-1])
    t_dump = np.array(td[:ndump])
    kd = int(np.argmin(np.abs(t_dump - tpk)))

    # ------------------------------------------------------- plot styling
    # every model gets a DISTINCT staggered open marker: solid = black o
    # (dashed), OpenSG-RM = orange s, Abaqus-FSDT = green ^ (dash-dot);
    # EVERY figure individual with its own legend; benchmark = 3-D solid
    STY_SOL = dict(ls="--", marker="o", color="k", lw=1.6, ms=5,
                   mfc="none", mew=1.2)
    STY_RM = dict(ls="-", marker="s", color="#ff7f0e", lw=1.4, ms=4,
                  mfc="none", mew=1.1)
    STY_FS = dict(ls="-.", marker="^", color="#2ca02c", lw=1.3, ms=5,
                  mfc="none", mew=1.1)

    def one_plot(fname, curves, xlabel, ylabel, note=None, xlim=None):
        fig, ax = plt.subplots(figsize=(7.4, 4.6))
        for x, y, sty, lab in curves:
            ax.plot(x, y, label=lab, **sty)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(alpha=0.3)
        if xlim is not None:
            ax.set_xlim(*xlim)
        if note:
            ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=8,
                    color="0.35")
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                  frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, fname), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------- w(t)
    # THREE deflection figures per pulse kind (6 total across step+blast):
    # solid-vs-RM, solid-vs-FSDT, and the three-way -- the pairwise pair
    # stays readable where the shell curves overlap
    c_sol = (1e3 * np.concatenate([[0], t_so]),
             1e3 * np.concatenate([[0], w_so]),
             dict(STY_SOL, markevery=(0, 32)), "Abaqus 3-D solid\n"
             "(benchmark)")
    c_rm = (1e3 * np.concatenate([[0], t_rm[:n_inc]]),
            1e3 * np.concatenate([[0], w_rm[:n_inc]]),
            dict(STY_RM, markevery=(10, 30)), "OpenSG-RM shell")
    ffs = prefer(os.path.join(HERE, "Abaqus_results",
                              "sandwich_FSDT_S8R_%s.dat" % kind),
                 os.path.join(HERE, "Abaqus_results",
                              "sandwich_FSDT_%s.dat" % kind),
                 "S8R FSDT history not found", kind)
    fs_elt = "S8R" if "S8R" in os.path.basename(ffs) else "S4"
    w_fs_pk = None
    c_fs = None
    if os.path.isfile(ffs):
        t_fs_all = step_times(ffs)
        uf = node_history(ffs, "NCEN")
        t_fs = t_fs_all[len(t_fs_all) - len(uf):]
        w_fs = uf[:, 2]
        w_fs_pk = (t_fs, w_fs)
        c_fs = (1e3 * np.concatenate([[0], t_fs]),
                1e3 * np.concatenate([[0], w_fs]),
                dict(STY_FS, markevery=(20, 30)),
                "Abaqus FSDT\n(composite %s)" % fs_elt)
    one_plot("w_history_%s_RM.png" % kind, [c_sol, c_rm], "time [ms]",
             "center deflection $w$ [mm]")
    if c_fs is not None:
        one_plot("w_history_%s_FSDT.png" % kind, [c_sol, c_fs], "time [ms]",
                 "center deflection $w$ [mm]")
    one_plot("w_history_%s.png" % kind,
             [c_sol, c_rm] + ([c_fs] if c_fs is not None else []),
             "time [ms]", "center deflection $w$ [mm]")

    # ------------------------------------- through-thickness path profiles
    panels = [("s13", prof["X"][:, 4], "COLX", 4,
               r"$\sigma_{13}$ [MPa]  at  $(0,\,b/2)$"),
              ("s23", prof["Y"][:, 3], "COLY", 3,
               r"$\sigma_{23}$ [MPa]  at  $(a/2,\,0)$"),
              ("s33", s33, "COLC", 2,
               r"$\sigma_{33}$ [MPa]  at  $(a/2,\,b/2)$"),
              ("s11", prof["C"][:, 0], "COLC", 0,
               r"$\sigma_{11}$ [MPa]  at  $(a/2,\,b/2)$")]
    for kkey, rmv, col, jc, lbl in panels:
        zc_s, gs = sol[col]
        curves = [(1e-6 * gs[kd, :, jc], zc_s / H, STY_SOL,
                   "Abaqus 3-D solid"),
                  (1e-6 * rmv, zg / H, dict(STY_RM, markevery=5),
                   "OpenSG-RM recovery")]
        one_plot("profile_%s_%s.png" % (kkey, kind), curves, lbl, "$z/h$")

    # ------------------------------------------------ interface s13 history
    # distribution shape frozen at the peak instant (its integral = Q1),
    # amplitude = the MEASURED Q1(t) series -- the resultant x SG-shape
    # factorization that IS the dynamic recovery
    z_if = sum(thick[:4]) - H / 2 - 1e-6      # bottom face-core interface
    iz_if = int(np.argmin(np.abs(zg - z_if)))
    shp_if = prof["X"][iz_if, 4] / fits["X"][kpk, 3, 0]
    s13_rm_t = shp_if * fits["X"][:n_inc, 3, 0]
    zc_s, gs = sol["COLX"]
    kz_if = int(np.argmin(np.abs(zc_s - z_if)))
    curves = [(1e3 * t_dump, 1e-6 * gs[:, kz_if, 4], STY_SOL,
               "Abaqus 3-D solid"),
              (1e3 * t_rm[:n_inc], 1e-6 * s13_rm_t,
               dict(STY_RM, markevery=12), "OpenSG-RM recovery")]
    one_plot("iface_s13_%s.png" % kind, curves, "time [ms]",
             r"face--core interface $\sigma_{13}$ [MPa]  at  $(0,\,b/2)$")

    # ------------------------------------------------------------- numbers
    ipk_so = int(np.argmax(np.abs(w_so)))
    wpk_rm = w_rm[kpk]; wpk_so = w_so[ipk_so]
    fs_line = "  (Abaqus FSDT job not present)"
    if w_fs_pk is not None:
        ifs = int(np.argmax(np.abs(w_fs_pk[1])))
        fs_line = ("Abaqus FSDT (composite %s) peak w: %.6e m at t=%.3f ms"
                   " (%+.2f %% vs solid)"
                   % (fs_elt, w_fs_pk[1][ifs], 1e3 * w_fs_pk[0][ifs],
                      100 * (w_fs_pk[1][ifs] - wpk_so) / wpk_so))
    clos = s33[-1] / (ampl(tpk) * np.sin(P * A / 2) ** 2)   # top-face value
    wc_rm, wc_so = wallclock(frm), wallclock(fso)
    lines = [
        "Nayak Ex.5 sandwich, %s pulse -- OpenSG-RM vs Abaqus 3-D solid" % kind,
        "peak center w: RM %.6e m at t=%.3f ms | solid %.6e m at t=%.3f ms"
        % (wpk_rm, 1e3 * tpk, wpk_so, 1e3 * t_so[ipk_so]),
        "  peak-w difference RM vs solid: %+.2f %%"
        % (100 * (wpk_rm - wpk_so) / wpk_so),
        fs_line,
        "profiles at RM increment %d (t=%.3f ms), solid dump at t=%.3f ms"
        % (kpk + 1, 1e3 * tpk, 1e3 * t_dump[kd]),
        "sigma_33 top-face closure (CONSTITUTIVE recovery / applied q):"
        " %.4f (target 1; V2L face enforcement, fitted drivers)" % clos,
        "dynamic shear amplification Q_meas/Q_quasistatic: sc13(X)=%.3f"
        " sc23(Y)=%.3f" % (scq["X"][0], scq["Y"][1]),
        "cost: RM %s s vs solid %s s wallclock; RM %s vs solid %s"
        % (wc_rm, wc_so, model_label(frm, NX * NX),
           model_label(fso, 6400)),
    ]
    with open(os.path.join(HERE, "dyn_%s.dat" % kind), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join("  " + ln for ln in lines))


if __name__ == "__main__":
    import time as _t
    print("start: " + _t.strftime("%Y-%m-%d %H:%M:%S"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default=None, choices=("step", "blast"))
    args = ap.parse_args()
    for kind in ([args.kind] if args.kind else ("step", "blast")):
        run(kind)
    print("end:   " + _t.strftime("%Y-%m-%d %H:%M:%S"))
