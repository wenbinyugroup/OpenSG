"""build_ff.py -- the midspan-station .ff of the CMC sandwich plate in
the PURE YU-2003 ASYMPTOTIC COMPOSITION (the companion of the HC-mode
folder ../AR5): the second-order recovery relies on the d2eps drivers
(eps,alphabeta -> the Eq. 64-66 V2 chains) and the pressure enters
CONSTITUTIVELY through the load columns with Yu's uniform (<w> = 0
KKT) reaction -- no plate-equilibrium shear-path (tau) reaction
anywhere.  `q_reaction: uniform` is written EXPLICITLY so the cli runs
this composition instead of the heterogeneous-cell auto mode.

Same cell-matched doctrine as ../AR5: one shell element per 3-D SG
cell, FF cell-averaged from the finest ABDG run.  d2eps drivers, best
source first:
  (1) SPECTRAL ANALYTIC (./macro_drivers.npz from macro_ritz.py, the
      hp-graded Ritz of the clamped plate under this study's own 8x8
      ABDG law): d2eps = analytic eps_R,ab of the converged solution,
      equilibrium identity to ~5e-7 -- PREFERRED when the file exists;
  (2) fallback: local quadratic LS on the grid, mirror parity BY
      CONSTRUCTION and the plate equilibrium identity
      d2M11/dx2 + 2 d2M12/dxdy + d2M22/dy2 = +q as one KKT equality.
Both LS fits (unconstrained/constrained) are always printed for the
record; the .ff records which source it carries.

Eq. 50 NOTE (audited against the pinned engine, do not "fix"): the
opensg_pin kernels take the RM measures R for EVERY driver -- eps =
C_eff^-1 FF, dE_a = R,a, d2eps = R,ab.  The classical conversion
eps_cl = R - D_a gamma,a (Yu Eq. 50) happens INTERNALLY via the
tilt/detilt row split (sg_homo _detilt_cols_2d -> V11barD/V12barD in
the D-chain value slots, raw V11bar/V12bar in the T-chain rows
33/23/13; msg_rm_plate_README.md rule 1 + sec. 3: pass R, never
pre-converted eps -- it would double-correct).  The would-be external
corrections are 0.1x-13.7x the drivers here (sqrt(D11/G11) = 3.68 vs
a = 5), so NO Eq. 50 term is applied caller-side.

    python build_ff.py

In:  ../../abaqus/<CASE>/<CASE>_ABDG*_{SF,SM}.rpt (ABDCL excluded),
     ../../preovios_try.out, ./macro_drivers.npz (optional, preferred)
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


def d2_ls(eps_g, NE, px, C, q):
    """Second (and first, as a symmetry report) derivatives of the eps
    grid at the plate centre: local quadratic LS over a half-cell
    window.  Two fits come back: the UNCONSTRAINED one (full quadratic
    basis, parity zeroed after the fact -- the old code path, kept for
    the printed comparison) and the CONSTRAINED one, where (a) the
    mirror parity is imposed BY CONSTRUCTION (even components on the
    basis {1, x^2, y^2}; the mirror-odd N12/M12 slots on {x y} alone,
    so d11/d22 of the odd slots and d12 of the even slots are zero as
    unknowns, not by post-hoc surgery) and (b) the plate equilibrium
    identity (C d11)[3] + 2 (C d12)[5] + (C d22)[4] = +q is one linear
    equality tying the otherwise component-independent quadratic fits
    together -- solved as the joint block-diagonal normal equations
    with a single KKT row."""
    xs = (np.arange(NE) + 0.5) * px
    cx = A / 2.0
    win = max(2.5 * px, 1.2)
    sel = [(i, j) for i in range(NE) for j in range(NE)
           if abs(xs[i] - cx) <= win and abs(xs[j] - cx) <= win]
    dx = np.array([xs[i] - cx for i, j in sel])
    dy = np.array([xs[j] - cx for i, j in sel])
    Y = np.array([eps_g[i, j] for i, j in sel])

    # -- unconstrained (the old fit): full basis, parity-cleaned after
    X = np.column_stack([np.ones_like(dx), dx, dy, dx ** 2, dx * dy,
                         dy ** 2])
    cf, *_ = np.linalg.lstsq(X, Y, rcond=None)
    d11u, d12u, d22u = 2 * cf[3], cf[4].copy(), 2 * cf[5]
    d11u[ODD] = 0.0
    d22u[ODD] = 0.0
    d12u[~ODD] = 0.0
    dmax1 = max(np.abs(cf[1]).max(), np.abs(cf[2]).max())

    # -- constrained: parity by construction + equilibrium KKT row
    even = np.nonzero(~ODD)[0]
    odd = np.nonzero(ODD)[0]
    Xe = np.column_stack([np.ones_like(dx), dx ** 2, dy ** 2])
    xo = dx * dy
    n = 3 * len(even) + len(odd)
    H = np.zeros((n, n))
    b = np.zeros(n)
    g = np.zeros(n)                 # constraint row: g . c = q
    HXe = Xe.T @ Xe
    for k, s in enumerate(even):
        sl = slice(3 * k, 3 * k + 3)
        H[sl, sl] = HXe
        b[3 * k:3 * k + 3] = Xe.T @ Y[:, s]
        g[3 * k + 1] = 2.0 * C[3, s]        # d11[s] = 2 c_x2
        g[3 * k + 2] = 2.0 * C[4, s]        # d22[s] = 2 c_y2
    off = 3 * len(even)
    for k, s in enumerate(odd):
        H[off + k, off + k] = xo @ xo
        b[off + k] = xo @ Y[:, s]
        g[off + k] = 2.0 * C[5, s]          # 2 (C d12)[5], d12[s] = c_xy
    K = np.zeros((n + 1, n + 1))
    K[:n, :n] = H
    K[:n, n] = g
    K[n, :n] = g
    sol = np.linalg.solve(K, np.r_[b, q])
    d11c, d22c, d12c = np.zeros(6), np.zeros(6), np.zeros(6)
    for k, s in enumerate(even):
        d11c[s] = 2.0 * sol[3 * k + 1]
        d22c[s] = 2.0 * sol[3 * k + 2]
    for k, s in enumerate(odd):
        d12c[s] = sol[off + k]
    return (d11u, d22u, d12u), (d11c, d22c, d12c), dmax1


def vec(v):
    return "[%s]" % ", ".join("%.8e" % x for x in v)


# ---- the SG's law (6x6 plate block + RM transverse-shear G 2x2)
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
C8 = np.array(rows)
C = 0.5 * (C8[:6, :6] + C8[:6, :6].T)
G2 = 0.5 * (C8[6:8, 6:8] + C8[6:8, 6:8].T)

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
(d11u, d22u, d12u), (d11, d22, d12), dmax1 = \
    d2_ls(eps_g, NEf, A / float(NEf), C, Q_TOP)
eqs_u = (C @ d11u)[3] + 2 * (C @ d12u)[5] + (C @ d22u)[4]
eqsum = (C @ d11)[3] + 2 * (C @ d12)[5] + (C @ d22)[4]
print("unconstrained d2 (parity-cleaned LS, the old fit):")
print("  d11 %s" % vec(d11u))
print("  d22 %s" % vec(d22u))
print("  d12 %s" % vec(d12u))
print("  stencil diagnostic %.7f vs +q = +%.1f; first-derivative"
      " residual %.2e" % (eqs_u, Q_TOP, dmax1))
print("constrained d2 (parity by construction + equilibrium KKT):")
print("  d11 %s" % vec(d11))
print("  d22 %s" % vec(d22))
print("  d12 %s" % vec(d12))
print("  stencil diagnostic: d2M11/dx2 + 2 d2M12/dxdy + d2M22/dy2 ="
      " %.7f (identity: +q = +%.1f)" % (eqsum, Q_TOP))
assert abs(eqsum - Q_TOP) <= 1e-10, \
    "equilibrium constraint not machine-tight: %.3e" % (eqsum - Q_TOP)

# ---- driver source selection: spectral analytic when delivered
NPZ = os.path.join(HERE, "macro_drivers.npz")
if os.path.exists(NPZ):
    mz = np.load(NPZ)
    d11f = mz["d2eps_dx1dx1"]
    d22f = mz["d2eps_dx2dx2"]
    d12f = mz["d2eps_dx1dx2"]
    dg1, dg2 = mz["dgamma_dx1"], mz["dgamma_dx2"]
    eq_final = (C @ d11f)[3] + 2 * (C @ d12f)[5] + (C @ d22f)[4]
    DRV = ("spectral analytic eps_R,ab (macro_drivers.npz, hp-Ritz"
           " pe=%d, %d DOF)" % (int(mz["basis_pe"]), int(mz["ndof"])))
    print("spectral analytic d2 (macro_drivers.npz) -- SELECTED:")
    print("  d11 %s" % vec(d11f))
    print("  d22 %s" % vec(d22f))
    print("  d12 %s" % vec(d12f))
    print("  stencil diagnostic: d2M11/dx2 + 2 d2M12/dxdy +"
          " d2M22/dy2 = %.10f (identity: +q = +%.1f, residual"
          " %.2e)" % (eq_final, Q_TOP, eq_final - Q_TOP))
    assert abs(eq_final - Q_TOP) <= 1e-5, \
        "spectral drivers violate plate equilibrium: %.3e" \
        % (eq_final - Q_TOP)
else:
    d11f, d22f, d12f = d11, d22, d12
    dg1 = dg2 = None
    eq_final = eqsum
    DRV = ("constrained quadratic LS (parity by construction +"
           " equilibrium KKT); macro_drivers.npz absent")
    print("macro_drivers.npz NOT found -- falling back to the"
          " constrained LS drivers")

# consistency gate: the transverse-shear gradients the engine will
# synthesize from these d2eps (gamma,a = X [Q1,a; Q2,a] with Q_a from
# moment equilibrium, X = G^-1) vs the macro plate's actual values --
# the G-anisotropy (G22/G11 = 8.3) makes the split, not just the sum
# Q1,1 + Q2,2 = q, the S13-critical content.
D1 = np.zeros((6, 2))
D1[3, 0] = 1.0
D1[5, 1] = 1.0
D2 = np.zeros((6, 2))
D2[4, 1] = 1.0
D2[5, 0] = 1.0
X = np.linalg.inv(G2)
g1 = X @ (D1.T @ C @ d11f + D2.T @ C @ d12f)      # gamma,1
g2 = X @ (D1.T @ C @ d12f + D2.T @ C @ d22f)      # gamma,2
print("driver-implied gamma1,1 = %.6e, gamma2,2 = %.6e"
      % (g1[0], g2[1]))
if dg1 is not None:
    r1 = g1[0] / dg1[0] - 1.0
    r2 = g2[1] / dg2[1] - 1.0
    print("  spectral reference   %.6e, %.6e -> dev %+.2f%% /"
          " %+.2f%%" % (dg1[0], dg2[1], 100 * r1, 100 * r2))
    if max(abs(r1), abs(r2)) > 0.02:
        print("  WARNING: driver gamma-gradient split deviates from"
              " the plate's actual by > 2%")

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
            "# d2eps source: %s;" % DRV,
            "# stencil diagnostic %.10f vs +q = +%.1f (residual"
            " %.2e)." % (eq_final, Q_TOP, eq_final - Q_TOP),
            "# Eq. 50: all drivers are the RM measures R (eps ="
            " C_eff^-1 FF,",
            "# d2eps = R,ab); the engine detilts internally"
            " (V11barD/V12barD) --",
            "# no caller-side classical conversion (it would"
            " double-correct).",
        ]
        body += "deps_dx1: %s\n" % vec(np.zeros(6))
        body += "deps_dx2: %s\n" % vec(np.zeros(6))
        body += "d2eps_dx1dx1: %s\n" % vec(d11f)
        body += "d2eps_dx2dx2: %s\n" % vec(d22f)
        body += "d2eps_dx1dx2: %s\n" % vec(d12f)
        body += "qt6: [%g, 0.0, 0.0, 0.0, 0.0, 0.0]\n" % Q_TOP
        body += "q_reaction: uniform\n"
    else:
        hdr += ["# FF-only: the classical model carries no derivative"
                " or pressure",
                "# machinery in Yu's construction."]
    with open(os.path.join(HERE, name), "w") as g:
        g.write("\n".join(hdr) + "\n" + body)
    print("wrote %s (FF from %s)" % (name, src))
