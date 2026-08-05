"""compare_stress_top.py -- sigma_xx, sigma_yy and the deflection at the CENTRE
OF THE TOP SURFACE (a/2, b/2, +h/2): OpenSG-RM against the 3-D FEA benchmark.

    Abaqus_results/ex5_shell_S8R.dat      SF/SM on the centre patch
    Abaqus_results/ex5_solid_C3D20.dat    S and U at the top-surface centre
                 ->  top_sxx.png  top_syy.png  top_w.png  +  top_centre.out

WHAT IS BEING TESTED.  The shell deck carries no plies -- only the homogenized
8x8.  Every quantity here therefore comes out of the SG recovery driven by the
patch resultants and their in-plane gradients, so this tests the
DEHOMOGENIZATION, not just the homogenization.  In particular the shell CANNOT
read the top surface off a node: an RM shell has one w per point (eps33 = 0)
and it belongs to the reference surface, so w(top) = w_node + w3(+h/2) with the
warping supplied by the recovery.

WHY THESE THREE.  At the centre of a doubly symmetric plate under a symmetric
load, sigma_xy, sigma_xz and sigma_yz vanish by symmetry and sigma_zz is pinned
to -q0 by the traction condition on the loaded face.  sigma_xx, sigma_yy and
the deflection are what carry accuracy information at this station.

THE SHELL ELEMENT is S8R: 8-node quadratic, SIX dof per node, and it prints
SF4/SF5 -- the transverse shear forces -- so the MSG 2x2 block is actually
used.  S9R5 prints no SF4/SF5 at all: it is a 5-dof Kirchhoff shell carrying no
transverse shear, so *TRANSVERSE SHEAR STIFFNESS is accepted and then has
nothing to apply it to.  With a core of 0.95 h of soft foam that is the
dominant compliance, so S9R5 would not be testing the same model.

THE SOLID SIDE reads *EL PRINT, POSITION=NODES over the four top-layer elements
meeting at the centre node, so its stress is extrapolated to z = +h/2 rather
than reported at the outermost integration point, which sits inside the ply.

THE SIGN.  The solid is loaded with P2 on its top face (acts -z), the shell
with P on a +z-normal element.  The solid history is negated here.

Run:  python compare_stress_top.py
"""
import os
import re
import sys

import numpy as np
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "src", "opensg_solid", "rm_plate_1D")):
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "examples", "rm_plate", "static", "yu2003"))

from opensg_solid.rm_plate_1D.segment_plate import read_plate_sg_yaml
from opensg_solid.rm_plate_1D.msg_rm_plate import (rm_plate_msg,
                                            msgrm_strain_at_depth,
                                            msgrm_warping_at_depth)
from recover_6p2 import read_elprint_tables

# ----------------------------------------------------------------------------
# ALL VARIABLES USED IN THIS SCRIPT
# ----------------------------------------------------------------------------
#   RES, SHELL, SOLID          Abaqus_results/ and the two history .dat paths
#   SHELL_ELT, SOLID_ELT       S8R / C3D20, used in names and labels
#   LAB_SOL, LAB_RM, STY_SOL, STY_RM   plot labels and line styles
#   db, p, A, B, Q0, fraction, P       case data from layup_db.yaml
#   inp, r, S6, h, ZTOP        SG input, homogenization, inv(A6), thickness,
#                              and the top-surface depth in the SG frame
#   step_times(dat)            the increment instants of a .dat
#   node_history(dat, nset)    one nodal value per increment
#   patch_fit(dat, name, x0)   (n_inc, 6, 3) resultants + gradients at x0,
#                              least-squares over the patch's ip; nip inferred
#   solid_node_stress(dat, n)  (n_inc, nel, 6) POSITION=NODES stresses at n
#   fit, NIP                   the PATCHC fit and its ip-per-increment count
#   w_ref                      centre-node reference-surface deflection
#   qt6                        the load ladder at the plate centre
#   E6, dE1, dE2, d11, d22, s, warp    per-increment recovery working set
#   sxx_rm, syy_rm, w_rm       the OpenSG-RM histories at (a/2, b/2, +h/2)
#   NTOP                       the solid's top-centre node id (read from the
#                              deck, "next" is the read-state sentinel)
#   S_all, sxx_so, syy_so, w_so, spread   the solid histories (negated) and
#                              the 4-element nodal-extrapolation spread
#   t_rm, t_so, t, n, nrm, nso  time axes and common increment count
#   CURVES                     (key, ylabel, rm, solid, unit) per quantity
#   k, k0, k1, pk, y_rm, y_so  peak/RMS working variables
#   lines, tab                 the .out report and the .dat summary table
#   fig, ax, key, ylab, unit   the per-quantity figure loop
# ----------------------------------------------------------------------------

RES = os.path.join(HERE, "Abaqus_results")
SHELL_ELT = "S8R"
SOLID_ELT = "C3D20"
SHELL = os.path.join(RES, "ex5_shell_%s.dat" % SHELL_ELT)
SOLID = os.path.join(RES, "ex5_solid_%s.dat" % SOLID_ELT)
LAB_SOL = "Abaqus 3-D FEA (%s)" % SOLID_ELT
LAB_RM = "OpenSG-RM (%s)" % SHELL_ELT
STY_SOL = dict(ls="--", marker="o", color="k", lw=1.6, ms=5, mfc="none",
               mew=1.2, markevery=(0, 32))
STY_RM = dict(ls="-", marker="s", color="#ff7f0e", lw=1.4, ms=4, mfc="none",
              mew=1.1, markevery=(16, 32))

# ---- case data --------------------------------------------------------------
db = yaml.safe_load(open(os.path.join(HERE, "layup_db.yaml")))
p = db["plate"]
A, B, Q0 = float(p["a"]), float(p["b"]), float(p["q0"])
fraction = float(db["fraction"])
P = np.pi / A

inp = read_plate_sg_yaml(os.path.join(HERE, "1dsg.yaml"))
r = rm_plate_msg(inp["thick"], inp["angles"], inp["mat_names"],
                 inp["material_db"], n_per_layer=inp["n_per_layer"],
                 elem_order=inp["elem_order"], fraction=fraction)
S6 = np.linalg.inv(np.asarray(r["A6"]))
h = float(sum(inp["thick"]))
ZTOP = (1.0 - fraction) * h - 1e-9


def step_times(dat):
    return np.array([float(ln.split()[-1]) for ln in
                     open(dat, errors="replace").read().splitlines()
                     if "STEP TIME COMPLETED" in ln])


def node_history(dat, nset, comp=2):
    """One value per increment from a *NODE PRINT table."""
    w, active, seen = [], False, False
    for ln in open(dat, errors="replace").read().splitlines():
        if ("NODE SET %s" % nset) in ln and "TABLE IS PRINTED" in ln:
            active, seen = True, False
            continue
        tok = ln.split()
        if not tok:
            continue
        if active and not seen:
            if tok[0] == "NODE":
                seen = True
            continue
        if active and seen and re.fullmatch(r"\d+", tok[0]):
            v = []
            for t in tok[1:]:
                try:
                    v.append(float(t))
                except ValueError:
                    pass
            if len(v) > comp:
                w.append(v[comp])
            active = False
    return np.array(w)


def patch_fit(dat, name, x0):
    """(n_inc, 6, 3): N11 N22 N12 M11 M22 M12 and their two in-plane gradients
    at x0, least-squares fitted over the 2x2 patch's integration points.

    The six are collected BY NAME.  A 6-dof shell also prints SF4/SF5 and a
    5-dof one does not, so indexing the table by position would work for S4/S8R
    and break for S9R5."""
    tables = read_elprint_tables(dat)
    sf = xy = None
    for (es, labels), rows in tables.items():
        if es != name:
            continue
        ofs = rows.shape[1] - len(labels)
        if "SF1" in labels:
            keys = ("SF1", "SF2", "SF3", "SM1", "SM2", "SM3")
            missing = [k for k in keys if k not in labels]
            if missing:
                raise RuntimeError("%s: %s missing from %s"
                                   % (name, missing, dat))
            sf = rows[:, [labels.index(k) + ofs for k in keys]]
        if "COORD1" in labels:
            xy = rows[:, [labels.index(k) + ofs for k in
                          ("COORD1", "COORD2")]]
    if sf is None or xy is None:
        raise RuntimeError("%s: SF/SM or COORD table missing from %s"
                           % (name, dat))
    # integration points per increment is INFERRED, not assumed: the reduced
    # quadratic shells do not carry the same count as S4
    same = np.all(np.isclose(xy, xy[0]), axis=1)
    nip = int(np.flatnonzero(same[1:])[0] + 1)
    if sf.shape[0] % nip:
        raise RuntimeError("%s: %d rows is not a multiple of %d integration "
                           "points" % (name, sf.shape[0], nip))
    sf = sf.reshape(-1, nip, 6)
    xi, eta = xy[:nip, 0] - x0[0], xy[:nip, 1] - x0[1]
    Bm = np.column_stack([np.ones(nip), xi, eta, xi * eta, xi ** 2, eta ** 2])
    out = np.empty((sf.shape[0], 6, 3))
    for k in range(sf.shape[0]):
        C, *_ = np.linalg.lstsq(Bm, sf[k], rcond=None)
        out[k, :, 0], out[k, :, 1], out[k, :, 2] = C[0], C[1], C[2]
    return out, nip


def solid_node_stress(dat, node):
    """(n_inc, nel, 6) stresses at `node`.  POSITION=NODES repeats the element
    id only on its first row, so it is carried forward."""
    rows, cur, block = [], None, False
    for ln in open(dat, errors="replace").read().splitlines():
        if "THE FOLLOWING TABLE IS PRINTED" in ln:
            block = "NODES" in ln.upper()
            cur = None
            continue
        if not block:
            continue
        tok = ln.split()
        if not tok or not re.fullmatch(r"\d+", tok[0]):
            continue
        vals = []
        for t in tok:
            try:
                vals.append(float(t))
            except ValueError:
                pass
        if len(vals) >= 8:
            cur = int(vals[0])
            nd, s = int(vals[1]), vals[2:8]
        elif len(vals) >= 7:
            nd, s = int(vals[0]), vals[1:7]
        else:
            continue
        if nd == node:
            rows.append((cur, s))
    if not rows:
        raise RuntimeError("no POSITION=NODES rows for node %d in %s"
                           % (node, dat))
    nel = len(set(e for e, _ in rows[:8]))
    return np.array([s for _, s in rows]).reshape(-1, nel, 6)


# ---- the OpenSG-RM side: recover the top surface from the patch -------------
fit, NIP = patch_fit(SHELL, "PATCHC", (A / 2, B / 2))
w_ref = node_history(SHELL, "NCEN_REF")
qt6 = Q0 * np.array([1.0, 0.0, 0.0, -P * P, 0.0, -P * P])
nrm = min(fit.shape[0], len(w_ref))
sxx_rm = np.empty(nrm)
syy_rm = np.empty(nrm)
w_rm = np.empty(nrm)
z6 = np.zeros(6)
for k in range(nrm):
    E6 = S6 @ fit[k][:, 0]
    dE1 = S6 @ fit[k][:, 1]
    dE2 = S6 @ fit[k][:, 2]
    # Eq.-63 CONSISTENT first-order recovery: NO second gradients.  They
    # belong to Yu's Eq. 66 together with the Gh V2 column we do not carry;
    # taking the E,ab brackets alone was measured to overshoot the face
    # sigma_xx peak by +8.4 points (+8.67% -> +0.32% when dropped).
    s = np.asarray(msgrm_strain_at_depth(r, ZTOP, E6, dE1, dE2, z6,
                                         z6, z6, qt6=qt6, frame="plate")[1])
    sxx_rm[k], syy_rm[k] = s[0], s[1]   # SG Voigt order (11,22,33,23,13,12)
    warp = msgrm_warping_at_depth(r, ZTOP, E6, dE1, dE2, z6,
                                  z6, z6, qt6=qt6)
    w_rm[k] = w_ref[k] + float(np.asarray(warp)[2])
t_rm = step_times(SHELL)[-nrm:]

# ---- the 3-D benchmark ------------------------------------------------------
NTOP = None
for ln in open(os.path.join(HERE, "layup_db_3dfea.inp")):
    if NTOP == "next":
        NTOP = int(ln.split(",")[0])
        break
    if ln.strip().upper() == "*NSET, NSET=NTOP3D":
        NTOP = "next"

S_all = solid_node_stress(SOLID, NTOP)
sxx_so = -S_all[:, :, 0].mean(axis=1)       # P2 acts -z: negate throughout
syy_so = -S_all[:, :, 1].mean(axis=1)
spread = np.ptp(S_all[:, :, :2], axis=1).max()
w_so = -node_history(SOLID, "NTOP3D")
nso = min(len(sxx_so), len(w_so))
t_so = step_times(SOLID)[-nso:]

n = min(nrm, nso)
assert np.allclose(t_rm[:n], t_so[:n]), (
    "the decks did not share increments -- both need *DYNAMIC, DIRECT")
t = t_rm[:n]
CURVES = [("sxx", r"$\sigma_{xx}(a/2,\,b/2,\,h/2)$ [MPa]",
           1e-6 * sxx_rm[:n], 1e-6 * sxx_so[:n], "MPa"),
          ("syy", r"$\sigma_{yy}(a/2,\,b/2,\,h/2)$ [MPa]",
           1e-6 * syy_rm[:n], 1e-6 * syy_so[:n], "MPa"),
          ("w", r"$U_3(a/2,\,b/2,\,h/2)$ [m]",
           w_rm[:n], w_so[:n], "m")]

# ---- report -----------------------------------------------------------------
lines = ["OpenSG-RM vs the 3-D FEA benchmark, at the CENTRE OF THE TOP SURFACE",
         "(a/2, b/2, +h/2) = (%.4f, %.4f, %+.4f) m" % (A / 2, B / 2, h / 2),
         "",
         "  shell : 400 %s (8-node, 6 dof/node) + the homogenized 8x8 only,"
         % SHELL_ELT,
         "          no plies; dehomogenized to z = +h/2.  %d ip/element."
         % (NIP // 4),
         "  solid : 6400 %s ply by ply, full 3x3x3." % SOLID_ELT,
         "          Stress POSITION=NODES, 4 elements averaged (spread"
         " %.3g MPa)." % (1e-6 * spread),
         "",
         "%-10s %16s %16s %10s %12s"
         % ("", "OpenSG-RM", "3-D FEA", "diff", "at t [ms]")]
for key, _, y_rm, y_so, unit in CURVES:
    k1 = int(np.argmax(np.abs(y_rm)))
    k0 = int(np.argmax(np.abs(y_so)))
    lines.append("%-10s %16.4f %16.4f %9.2f%% %12.2f"
                 % ("peak %s [%s]" % (key, unit), y_rm[k1], y_so[k0],
                    100 * (y_rm[k1] - y_so[k0]) / y_so[k0], 1e3 * t[k0]))
lines.append("")
lines.append("RMS difference over the whole history, as %% of that peak:"
             .replace("%%", "%"))
for key, _, y_rm, y_so, unit in CURVES:
    pk = np.abs(y_so).max()
    lines.append("  %-6s %6.2f%%" % (key, 100 * np.sqrt(
        np.mean((y_rm - y_so) ** 2)) / pk))
lines += ["",
          "the recovery's contribution to the deflection (what a bare shell",
          "node cannot give):",
          "%-30s %14.6f m" % ("  peak w at the reference surface",
                              w_ref[:n].max()),
          "%-30s %14.6f m" % ("  peak warping w3(+h/2)",
                              (w_rm[:n] - w_ref[:n]).max()),
          "%-30s %13.2f%%" % ("  i.e. of the top deflection",
                              100 * (w_rm[:n] - w_ref[:n]).max()
                              / w_rm[:n].max()),
          "",
          "%-30s %14d" % ("increments compared", n)]
open(os.path.join(HERE, "top_centre.out"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

# ---- the machine-readable summary table ------------------------------------
# one row per quantity: peak RM, peak 3-D FEA, peak % error, RMS/peak %
tab = ["# OpenSG-RM (S8R) vs Abaqus 3-D FEA (C3D20), centre of the top"
       " surface (a/2, b/2, +h/2)",
       "# %-10s %16s %16s %12s %12s"
       % ("quantity", "OpenSG-RM", "3D-FEA", "peak_err_%", "RMS/peak_%")]
for key, _, y_rm, y_so, unit in CURVES:
    k1 = int(np.argmax(np.abs(y_rm)))
    k0 = int(np.argmax(np.abs(y_so)))
    pk = np.abs(y_so).max()
    tab.append("%-12s %16.6e %16.6e %12.2f %12.2f"
               % ("%s[%s]" % (key, unit), y_rm[k1], y_so[k0],
                  100 * (y_rm[k1] - y_so[k0]) / y_so[k0],
                  100 * np.sqrt(np.mean((y_rm - y_so) ** 2)) / pk))
open(os.path.join(HERE, "top_centre_table.dat"), "w").write(
    "\n".join(tab) + "\n")

# ---- figures ----------------------------------------------------------------
for key, ylab, y_rm, y_so, unit in CURVES:
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.plot(1e3 * t, y_so, label=LAB_SOL, **STY_SOL)
    ax.plot(1e3 * t, y_rm, label=LAB_RM, **STY_RM)
    ax.set_xlabel("time [ms]", fontsize=11)
    ax.set_ylabel(ylab, fontsize=11)
    ax.set_xlim(0, 1e3 * t[-1])
    ax.grid(alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "top_%s.png" % key), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
print("\nwrote top_sxx.png, top_syy.png, top_w.png + top_centre.out")
