"""build_ff.py -- the midspan-station .ff of the CMC sandwich plate in
the PURE YU-2003 ASYMPTOTIC COMPOSITION (the companion of the HC-mode
folder ../AR5): the second-order recovery relies on the d2eps drivers
(eps,alphabeta -> the Eq. 64-66 V2 chains) and the pressure enters
CONSTITUTIVELY through the load columns with Yu's uniform (<w> = 0
KKT) reaction -- no plate-equilibrium shear-path (tau) reaction
anywhere.  `q_reaction: uniform` is written EXPLICITLY so the cli runs
this composition instead of the heterogeneous-cell auto mode.

Same cell-matched doctrine as ../AR5: one shell element per 3-D SG
cell, FF cell-averaged from the finest ABDG run, d2eps by local
quadratic LS on that grid.

    python build_ff.py

In:  ../../abaqus/<CASE>/<CASE>_ABDG*_{SF,SM}.rpt (ABDCL excluded),
     ../../preovios_try.out
Out: ./preovios_try.ff, ./preovios_try_classical.ff (FF-only)
"""
import glob
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
CASE = "AR5"
A = {"AR5": 5.0, "AR10": 10.0}[CASE]
NC = int(A)
Q_TOP = 1.0
OUT = os.path.join(ROOT, "preovios_try.out")


def read_rpt(path):
    """Abaqus element report -> (labels, {column name: values})."""
    hdr, lab, rows = None, [], []
    for ln in open(path):
        s = ln.rstrip("\n")
        if not s.strip() or s.lstrip().startswith(("---", "@Loc", "Pt")):
            continue
        if hdr is None:
            if "Element Label" in s and re.search(r"\b(SF|SM)\.", s):
                hdr = s.replace("Element Label", "El").split()
            continue
        if re.match(r"^\s*-?\d", s):
            v = s.split()
            if len(v) == len(hdr):
                lab.append(int(float(v[0])))
                rows.append([float(x) for x in v])
    a = np.array(rows)
    return np.array(lab), {h: a[:, i] for i, h in enumerate(hdr)}


def ff_grid(rptbase):
    """One structured plate run -> (NE, FF (NE, NE, 6)), element
    numbering e = j*NE + i + 1 (the deck generators' rule)."""
    lf, SF = read_rpt(rptbase + "_SF.rpt")
    lm, SM = read_rpt(rptbase + "_SM.rpt")
    assert (lf == lm).all(), "SF and SM report different element sets"
    NE = int(round(np.sqrt(len(lf))))
    assert NE * NE == len(lf), "%s is not a structured square run" \
        % os.path.basename(rptbase)
    F = np.zeros((NE, NE, 6))
    FFrows = np.column_stack([SF["SF.SF1"], SF["SF.SF2"], SF["SF.SF3"],
                              SM["SM.SM1"], SM["SM.SM2"], SM["SM.SM3"]])
    for k, e in enumerate(lf):
        F[(int(e) - 1) % NE, (int(e) - 1) // NE] = FFrows[k]
    return NE, F


ODD = np.array([0, 0, 1, 0, 0, 1], bool)          # N12, M12 mirror-odd


def station_sym(F):
    """The station cell's FF, symmetrized over the 4-cell mirror orbit
    with component parity (cell-matched grid: one element per cell)."""
    c = NC // 2
    orb = []
    for sx in (0, 1):
        for sy in (0, 1):
            cx_ = c if sx == 0 else NC - 1 - c
            cy_ = c if sy == 0 else NC - 1 - c
            sgn = np.where(ODD, (-1.0) ** sx * (-1.0) ** sy, 1.0)
            orb.append(sgn * F[cx_, cy_])
    return np.mean(orb, axis=0)


def d2_ls(eps_g, NE, px):
    """Second (and first, as a symmetry report) derivatives of the eps
    grid at the plate centre: local quadratic LS over a half-cell
    window, parity-cleaned."""
    xs = (np.arange(NE) + 0.5) * px
    cx = A / 2.0
    win = max(2.5 * px, 1.2)
    sel = [(i, j) for i in range(NE) for j in range(NE)
           if abs(xs[i] - cx) <= win and abs(xs[j] - cx) <= win]
    X = np.array([[1.0, xs[i] - cx, xs[j] - cx, (xs[i] - cx) ** 2,
                   (xs[i] - cx) * (xs[j] - cx), (xs[j] - cx) ** 2]
                  for i, j in sel])
    Y = np.array([eps_g[i, j] for i, j in sel])
    cf, *_ = np.linalg.lstsq(X, Y, rcond=None)
    d11, d12, d22 = 2 * cf[3], cf[4], 2 * cf[5]
    d11[ODD] = 0.0
    d22[ODD] = 0.0
    d12[~ODD] = 0.0
    return d11, d22, d12, max(np.abs(cf[1]).max(), np.abs(cf[2]).max())


def vec(v):
    return "[%s]" % ", ".join("%.8e" % x for x in v)


# ---- the SG's law
rows = []
for ln in open(OUT):
    v = ln.split()
    if len(v) == 8 and re.match(r"^-?\d", v[0]):
        try:
            rows.append([float(x) for x in v])
        except ValueError:
            pass
    if len(rows) == 8:
        break
C = 0.5 * (np.array(rows)[:6, :6] + np.array(rows)[:6, :6].T)

# ---- the CELL-MATCHED run (NE == NC), name-agnostic
cand = [p[:-7] for p in glob.glob(os.path.join(
    ROOT, "abaqus", CASE, "%s_ABDG*_SF.rpt" % CASE))
    if "ABDCL" not in os.path.basename(p)]
if not cand:
    raise SystemExit("no %s_ABDG*_SF.rpt found -- run the ABDG plate"
                     " job first" % CASE)
sizes = {b: ff_grid(b)[0] for b in cand}
RPT = max(sizes, key=sizes.get)
NEf, Ffine = ff_grid(RPT)
nsub = NEf // NC
F = Ffine.reshape(NC, nsub, NC, nsub, 6).mean(axis=(1, 3))
print("state source: %s (%d x %d), cell-averaged onto the %d x %d"
      " SG-cell grid (positions cell-matched)"
      % (os.path.basename(RPT), NEf, NEf, NC, NC))

ff_sym = station_sym(F)
print("station FF (mirror-orbit symmetrized): %s" % np.array2string(
    ff_sym, formatter={"float_kind": lambda v: "%.6e" % v}))
eps = np.linalg.solve(C, ff_sym)
print("eps = C_eff^-1 FF: %s" % np.array2string(
    eps, formatter={"float_kind": lambda v: "%.6e" % v}))

eps_g = np.einsum("st,ijt->ijs", np.linalg.inv(C), Ffine)
d11, d22, d12, dmax1 = d2_ls(eps_g, NEf, A / float(NEf))
eqsum = (C @ d11)[3] + 2 * (C @ d12)[5] + (C @ d22)[4]
print("stencil diagnostic: d2M11/dx2 + 2 d2M12/dxdy + d2M22/dy2 ="
      " %.4f (identity: +q = +%.1f for the downward load); first-"
      "derivative residual %.2e" % (eqsum, Q_TOP, dmax1))

rpt_cl = os.path.join(ROOT, "abaqus", CASE, "%s_ABDCL" % CASE)
if os.path.exists(rpt_cl + "_SF.rpt"):
    _, Fc = ff_grid(rpt_cl)
    ff_cl, cl_src = station_sym(Fc), os.path.basename(rpt_cl)
else:
    ff_cl, cl_src = ff_sym, os.path.basename(RPT)

for name, ffv, src, full in (
        ("preovios_try.ff", ff_sym, os.path.basename(RPT), True),
        ("preovios_try_classical.ff", ff_cl, cl_src, False)):
    hdr = [
        "# dehomogenization load for preovios_try.yaml (%s sandwich"
        " plate)" % CASE,
        "# PURE YU-2003 ASYMPTOTIC COMPOSITION: d2eps (eps,alphabeta)"
        " drive the",
        "# Eq. 64-66 V2 chains; the pressure acts CONSTITUTIVELY via"
        " the V1L/V2L",
        "# load columns with Yu's uniform <w>=0 reaction.  No plate-"
        "equilibrium",
        "# (tau) shortcut anywhere.",
        "# FF = station element of plate run %s, mirror-orbit"
        " symmetrized." % src,
    ]
    body = "FF: %s\n" % vec(ffv)
    if full:
        hdr += [
            "# d2eps: quadratic LS on the cell grid, parity-cleaned;"
            " stencil",
            "# diagnostic %.4f vs +q = +%.1f." % (eqsum, Q_TOP),
        ]
        body += "deps_dx1: %s\n" % vec(np.zeros(6))
        body += "deps_dx2: %s\n" % vec(np.zeros(6))
        body += "d2eps_dx1dx1: %s\n" % vec(d11)
        body += "d2eps_dx2dx2: %s\n" % vec(d22)
        body += "d2eps_dx1dx2: %s\n" % vec(d12)
        body += "qt6: [%g, 0.0, 0.0, 0.0, 0.0, 0.0]\n" % Q_TOP
        body += "q_reaction: uniform\n"
    else:
        hdr += ["# FF-only: the classical model carries no derivative"
                " or pressure",
                "# machinery in Yu's construction."]
    with open(os.path.join(HERE, name), "w") as g:
        g.write("\n".join(hdr) + "\n" + body)
    print("wrote %s (FF from %s)" % (name, src))
