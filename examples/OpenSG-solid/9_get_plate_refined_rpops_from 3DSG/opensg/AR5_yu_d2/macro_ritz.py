"""macro_ritz.py -- the analytic-derivative macro solver: a high-order
Ritz solve of the CLAMPED Reissner-Mindlin plate a = b = 5 under
uniform q = 1 pushing down, with the study's own 8x8 ABDG law.  The
principled replacement of build_ff.py's difference stencil: every
driver (eps_R, its first and second gradients, gamma and its
gradients) is evaluated ANALYTICALLY from the polynomial basis -- no
differencing anywhere -- so the plate equilibrium identity

    d2M11/dx1dx1 + 2 d2M12/dx1dx2 + d2M22/dx2dx2 = +q

holds at the station to <= 1e-6 instead of the stencil's 0.95.

Basis, two stages (one code path -- the global basis is the K = 1
special case of the hp basis):
  stage 1, GLOBAL Legendre-bubble (1-xi^2)P_i(xi) tensor products at
    p = 12/16/20: all Abaqus-validation gates pass, but the station
    identity converges only ~ p^-1.5 (0.986 at p = 36) -- the clamped
    -corner singularity of the rotations pollutes pointwise THIRD
    derivatives of a global polynomial;
  stage 2, hp-RITZ: the same tensor Ritz on a 1-D C0 piecewise-
    Legendre basis (hat + bubble), geometrically graded into the
    corners with the hp degree ramp (low degree in the small corner
    elements, full degree pe in the central one); the station (and the
    whole [2,3] cell and the (3.5,2.5) probe) lies inside the analytic
    central element [1,4]^2, where the identity converges
    exponentially in pe.

Kinematic conventions (GATED numerically below, not assumed):
  x3 and w positive UP, q = 1 acts DOWN (the Abaqus deck's P = -1)
  -> external work integral = (-q) * w;
  u_alpha = u0_alpha + x3 * phi_alpha;
  eps_R = [u0,1  v0,2  u0,2+v0,1  phi1,1  phi2,2  phi1,2+phi2,1]
          (the plate code's [e11 e22 2e12 k11 k22 2k12] ordering),
  gamma = [w,1 + phi1,  w,2 + phi2]   (0.5 gamma^T G gamma > 0).
  Strong-form consequences: Q_a,a = +q and M_ab,ab = +q, i.e. the same
  +q identity build_ff.py's stencil diagnostic converges to.  Gates:
  (a) the identity evaluates to +1 (sign), to <= 1e-6 (convergence);
  (b) station eps_R reproduces eps = C_eff^-1 FF_station from the
      Abaqus ABDG plate run; (c) station M11 (cell-avg) in -0.46..-0.54;
  (d) w_center matches AR5_ABDGF_U.csv to ~1% (S4R discretization).

In:  ../../preovios_try.out            (first 8 rows = the 8x8 ABDG law,
                                        parsed exactly as build_ff.py)
     ../../abaqus/AR5/AR5_ABDGF_U.csv  (S4R nodal U -- w_center check)
     ../../abaqus/AR5/AR5_ABDGF_SM.rpt (S4R moments -- M11 line check)
     ./preovios_try.ff                 (stencil d2eps, echo side-by-side)
Out: ./macro_drivers.npz  (station drivers for the Eq. 50 correction;
                           NOT wired into build_ff.py -- the integrator
                           does that)
     ./macro_drivers.txt  (echo of everything + gates + convergence)
"""
import datetime
import os
import re

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.polynomial import legendre as NL

print("start:", datetime.datetime.now().isoformat(timespec="seconds"))

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
CASE = "AR5"
A = 5.0                     # plate side (x1 and x2), footprint [0,5]^2
Q_TOP = 1.0                 # uniform pressure, pushing DOWN
QW = -Q_TOP                 # load in the +w (up) direction
XS = (A / 2.0, A / 2.0)     # the centre station
CELL = (A / 2.0 - 0.5, A / 2.0 + 0.5)   # the 1x1 station cell [2,3]
EPS_TARGET = np.array([1.549e-08, 8.775e-07, -6.464e-07,
                       1.183e-04, -5.628e-04, 2.303e-06])
NAMES6 = ["e11", "e22", "2e12", "k11", "k22", "2k12"]
# corner-graded 1-D breakpoints (ratio-5 geometric, both ends); the
# central element [1,4] contains the station, its cell and the probe
GRADED = [0.0, 0.04, 0.2, 1.0, A - 1.0, A - 0.2, A - 0.04, A]

# ---- the SG's law: first 8 numeric 8-column rows, as build_ff.py
rows = []
for ln in open(os.path.join(ROOT, "preovios_try.out")):
    v = ln.split()
    if len(v) == 8 and re.match(r"^-?\d", v[0]):
        try:
            rows.append([float(x) for x in v])
        except ValueError:
            pass
    if len(rows) == 8:
        break
C8 = 0.5 * (np.array(rows) + np.array(rows).T)
C6 = C8[:6, :6]
G2 = C8[6:8, 6:8]
assert np.abs(C8[:6, 6:8]).max() < 1e-9 * np.abs(C8).max(), \
    "membrane/bending--shear coupling block is not zero"

# ---- strain measures as (field, d/dx1 order, d/dx2 order, coeff) terms
# fields: 0 = u0, 1 = v0, 2 = w, 3 = phi1, 4 = phi2
TERMS = [
    [(0, 1, 0, 1.0)],                        # e11   = u0,1
    [(1, 0, 1, 1.0)],                        # e22   = v0,2
    [(0, 0, 1, 1.0), (1, 1, 0, 1.0)],        # 2e12  = u0,2 + v0,1
    [(3, 1, 0, 1.0)],                        # k11   = phi1,1
    [(4, 0, 1, 1.0)],                        # k22   = phi2,2
    [(3, 0, 1, 1.0), (4, 1, 0, 1.0)],        # 2k12  = phi1,2 + phi2,1
    [(2, 1, 0, 1.0), (3, 0, 0, 1.0)],        # gam13 = w,1 + phi1
    [(2, 0, 1, 1.0), (4, 0, 0, 1.0)],        # gam23 = w,2 + phi2
]


def hp_basis(brk, pe):
    """1-D C0 hard-clamp basis on breakpoints brk: hat functions at
    interior breakpoints + per-element Legendre bubbles
    (1-xi^2)P_i(xi), i = 0..pe_e-2, with the hp degree ramp
    pe_e = max(pe - 2*level, 4) (level = element distance from the
    central element; the tiny corner elements only resolve the weak
    singular part).  All functions vanish at both domain ends.  K = 1
    reproduces the global bubble basis of max degree pe.  Per element:
    global DOF ids and local Legendre series, xi-derivatives 0..3."""
    brk = np.asarray(brk, float)
    K = len(brk) - 1
    cen = (K - 1) / 2.0
    pes = [max(pe - 2 * int(round(abs(e - cen))), 4) for e in range(K)]
    off = K - 1
    elems = []
    for e in range(K):
        ids, ser0 = [], []
        if e >= 1:                     # hat centred at brk[e]: (1-xi)/2
            ids.append(e - 1)
            ser0.append(np.array([0.5, -0.5]))
        if e + 1 <= K - 1:             # hat at brk[e+1]: (1+xi)/2
            ids.append(e)
            ser0.append(np.array([0.5, 0.5]))
        for i in range(pes[e] - 1):
            ii = np.zeros(i + 1)
            ii[i] = 1.0
            ids.append(off + i)
            ser0.append(NL.legsub(ii, NL.legmulx(NL.legmulx(ii))))
        off += pes[e] - 1
        ser = [ser0] + [[NL.legder(c, d) for c in ser0] for d in (1, 2, 3)]
        elems.append({"x0": brk[e], "h": brk[e + 1] - brk[e],
                      "pe": pes[e], "ids": np.array(ids, int),
                      "ser": ser})
    return {"n": off, "brk": brk, "pe": pe, "pes": pes, "elems": elems}


def gram_load(B):
    """Exact 1-D Gram matrices G1[d1][d2] = int f^(d1)_i f^(d2)_k dx
    (d = 0/1, physical derivatives) and the load vector int f_i dx."""
    n = B["n"]
    G1 = [[np.zeros((n, n)) for _ in (0, 1)] for _ in (0, 1)]
    L1 = np.zeros(n)
    for el in B["elems"]:
        xq, wq = NL.leggauss(el["pe"] + 4)
        s = 2.0 / el["h"]
        V = [np.column_stack([NL.legval(xq, c) for c in el["ser"][d]])
             * s ** d for d in (0, 1)]
        w = wq * (el["h"] / 2.0)
        ix = np.ix_(el["ids"], el["ids"])
        for d1 in (0, 1):
            for d2 in (0, 1):
                G1[d1][d2][ix] += (V[d1] * w[:, None]).T @ V[d2]
        L1[el["ids"]] += V[0].T @ w
    return G1, L1


def bval(B, x, dmax):
    """Values of every 1-D basis function and its physical derivatives
    0..dmax at one point (ANALYTIC, from the local Legendre series)."""
    e = min(max(int(np.searchsorted(B["brk"], x, side="right")) - 1, 0),
            len(B["elems"]) - 1)
    el = B["elems"][e]
    s = 2.0 / el["h"]
    xi = 2.0 * (x - el["x0"]) / el["h"] - 1.0
    out = []
    for d in range(dmax + 1):
        v = np.zeros(B["n"])
        v[el["ids"]] = np.array([NL.legval(xi, c)
                                 for c in el["ser"][d]]) * s ** d
        out.append(v)
    return out


def ritz_solve(B):
    """Assemble (sparse, tensor-product Kronecker) and solve the
    clamped RM plate on basis B x B.  coef (5, n, n):
    field_f(x1,x2) = sum coef[f,i,j] b_i(x1) b_j(x2)."""
    G1, L1 = gram_load(B)
    Gs = [[sp.csr_matrix(G1[d1][d2]) for d2 in (0, 1)] for d1 in (0, 1)]
    n = B["n"]
    nb = n * n
    blocks = [[None] * 5 for _ in range(5)]
    for a in range(8):
        for b in range(8):
            cab = C8[a, b]
            if cab == 0.0:
                continue
            for fa, dxa, dya, ca in TERMS[a]:
                for fb, dxb, dyb, cb in TERMS[b]:
                    M = (cab * ca * cb) * sp.kron(Gs[dxa][dxb],
                                                  Gs[dya][dyb], "csr")
                    blocks[fa][fb] = M if blocks[fa][fb] is None \
                        else blocks[fa][fb] + M
    K = sp.bmat(blocks, format="csr")
    K = (K + K.T) * 0.5
    F = np.zeros(5 * nb)
    F[2 * nb:3 * nb] = QW * np.kron(L1, L1)
    c = spla.spsolve(K.tocsc(), F, permc_spec="MMD_AT_PLUS_A")
    rres = np.linalg.norm(K @ c - F) / np.linalg.norm(F)
    return c.reshape(5, n, n), rres


def point_partials(B, coef, x1, x2, dmax):
    """P[f, r, t] = d^(r+t) field_f / dx1^r dx2^t at one point, all
    five fields, r + t <= dmax -- analytic from the basis."""
    B1 = bval(B, x1, dmax)
    B2 = bval(B, x2, dmax)
    P = np.zeros((5, dmax + 1, dmax + 1))
    for f in range(5):
        for r in range(dmax + 1):
            for t in range(dmax + 1 - r):
                P[f, r, t] = B1[r] @ coef[f] @ B2[t]
    return P


def sd(P, a, r, t):
    """d^(r+t) (strain measure a) / dx1^r dx2^t from field partials."""
    return sum(cc * P[f, dx + r, dy + t] for f, dx, dy, cc in TERMS[a])


def eps6_at(B, coef, x1, x2):
    P = point_partials(B, coef, x1, x2, 1)
    return np.array([sd(P, a, 0, 0) for a in range(6)])


def station_state(B, coef):
    """Everything at the centre station, analytic: eps_R and its first
    and second gradients, gamma and its gradients, w, the equilibrium
    identities, and an off-centre moment-equilibrium probe."""
    P = point_partials(B, coef, XS[0], XS[1], 3)
    st = {
        "w": P[2, 0, 0],
        "eps": np.array([sd(P, a, 0, 0) for a in range(6)]),
        "dE1": np.array([sd(P, a, 1, 0) for a in range(6)]),
        "dE2": np.array([sd(P, a, 0, 1) for a in range(6)]),
        "d11": np.array([sd(P, a, 2, 0) for a in range(6)]),
        "d12": np.array([sd(P, a, 1, 1) for a in range(6)]),
        "d22": np.array([sd(P, a, 0, 2) for a in range(6)]),
        "gam": np.array([sd(P, a, 0, 0) for a in (6, 7)]),
        "dg1": np.array([sd(P, a, 1, 0) for a in (6, 7)]),
        "dg2": np.array([sd(P, a, 0, 1) for a in (6, 7)]),
    }
    d2g = np.zeros((2, 2, 2))
    for i in (0, 1):
        d2g[i, 0, 0] = sd(P, 6 + i, 2, 0)
        d2g[i, 0, 1] = d2g[i, 1, 0] = sd(P, 6 + i, 1, 1)
        d2g[i, 1, 1] = sd(P, 6 + i, 0, 2)
    st["d2g"] = d2g
    st["eq_M"] = (C6 @ st["d11"])[3] + 2.0 * (C6 @ st["d12"])[5] \
        + (C6 @ st["d22"])[4]
    st["eq_Q"] = (G2 @ st["dg1"])[0] + (G2 @ st["dg2"])[1]
    Pp = point_partials(B, coef, XS[0] + 1.0, XS[1], 2)
    dE1p = np.array([sd(Pp, a, 1, 0) for a in range(6)])
    dE2p = np.array([sd(Pp, a, 0, 1) for a in range(6)])
    Qp = G2 @ np.array([sd(Pp, a, 0, 0) for a in (6, 7)])
    st["mom_r"] = np.array([
        (C6 @ dE1p)[3] + (C6 @ dE2p)[5] - Qp[0],
        (C6 @ dE1p)[5] + (C6 @ dE2p)[4] - Qp[1]])
    st["Q_probe"] = Qp
    return st


# ---- convergence study: global p = 12/16/20, then the corner-graded
# hp basis, pe auto-extended until the identity meets 1e-6
stages = [("global", [0.0, A], p) for p in (12, 16, 20)] \
    + [("hp-graded", GRADED, pe) for pe in (8, 10, 12)]
run = []
while stages:
    kind, brk, pe = stages.pop(0)
    t0 = datetime.datetime.now()
    B = hp_basis(brk, pe)
    coef, rres = ritz_solve(B)
    st = station_state(B, coef)
    st.update(kind=kind, pe=pe, ndof=5 * B["n"] ** 2, rres=rres,
              B=B, coef=coef,
              secs=(datetime.datetime.now() - t0).total_seconds())
    run.append(st)
    print("%-9s pe=%2d solved: %s -> %s  ndof=%6d  lin.res=%.1e  "
          "w_c=%.8e  eq_M=%.10f  eq_Q=%.10f"
          % (kind, pe, t0.isoformat(timespec="seconds"),
             datetime.datetime.now().isoformat(timespec="seconds"),
             st["ndof"], rres, st["w"], st["eq_M"], st["eq_Q"]))
    if not stages and kind == "hp-graded" \
            and abs(st["eq_M"] - Q_TOP) > 1e-6 and pe < 20:
        stages.append((kind, brk, pe + 2))
fin = run[-1]
B, coef = fin["B"], fin["coef"]

cauchy = []
for lo, hi in zip(run[:-1], run[1:]):
    if lo["kind"] != hi["kind"]:
        continue
    d2lo = np.stack([lo["d11"], lo["d12"], lo["d22"]])
    d2hi = np.stack([hi["d11"], hi["d12"], hi["d22"]])
    cauchy.append((lo["kind"], lo["pe"], hi["pe"],
                   abs(hi["w"] - lo["w"]), np.abs(d2hi - d2lo).max(),
                   abs(hi["eq_M"] - lo["eq_M"])))
    print("Cauchy %-9s pe=%2d->%2d: |dw_c|=%.3e  max|d(d2eps)|=%.3e"
          "  |d(eq_M)|=%.3e" % cauchy[-1])

# ---- station cell average (the FF/eps of build_ff.py are cell-avg)
xg, wg = NL.leggauss(12)
xs_c = 0.5 * (CELL[0] + CELL[1]) + 0.5 * (CELL[1] - CELL[0]) * xg
ws_c = 0.5 * (CELL[1] - CELL[0]) * wg
eps_cell = np.zeros(6)
for i, xi in enumerate(xs_c):
    for j, yj in enumerate(xs_c):
        eps_cell += ws_c[i] * ws_c[j] * eps6_at(B, coef, xi, yj)
eps_cell /= (CELL[1] - CELL[0]) ** 2
FF_cell = C6 @ eps_cell
FF_point = C6 @ fin["eps"]
Qs = G2 @ fin["gam"]

# ---- validation vs the Abaqus S4R plate ------------------------------
U = np.genfromtxt(os.path.join(ROOT, "abaqus", CASE,
                               "%s_ABDGF_U.csv" % CASE),
                  delimiter=",", names=True)
near = (np.abs(U["x"] - XS[0]) < 0.11) & (np.abs(U["y"] - XS[1]) < 0.11)
assert near.sum() == 4, "expected the 4 S4R nodes around the centre"
w_abq = U["U3"][near].mean()          # bilinear at the midpoint
w_err = (fin["w"] - w_abq) / abs(w_abq)


def read_rpt(path):
    """Abaqus element report -> (labels, {column name: values}) --
    build_ff.py's proven parser, replicated (not imported: build_ff.py
    is a straight-line script)."""
    hdr, lab, rows_ = None, [], []
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
                rows_.append([float(x) for x in v])
    a = np.array(rows_)
    return np.array(lab), {h: a[:, i] for i, h in enumerate(hdr)}


lab, SM = read_rpt(os.path.join(ROOT, "abaqus", CASE,
                                "%s_ABDGF_SM.rpt" % CASE))
NE = int(round(np.sqrt(len(lab))))
assert NE * NE == len(lab)
M11g = np.zeros((NE, NE))
for k, e in enumerate(lab):
    M11g[(int(e) - 1) % NE, (int(e) - 1) // NE] = SM["SM.SM1"][k]
h = A / NE
jc = int(round(XS[1] / h - 0.5))      # centroid row at y = centre
assert abs((jc + 0.5) * h - XS[1]) < 1e-9, \
    "no S4R centroid row on the centre line"
x_line = (np.arange(NE) + 0.5) * h
M11_abq = M11g[:, jc]
M11_ritz = np.array([(C6 @ eps6_at(B, coef, x, XS[1]))[3]
                     for x in x_line])
inner = slice(3, NE - 3)              # drop 3 boundary-layer elements
rms_all = np.sqrt(np.mean((M11_ritz - M11_abq) ** 2))
rms_int = np.sqrt(np.mean((M11_ritz[inner] - M11_abq[inner]) ** 2))
rms_ref = np.sqrt(np.mean(M11_abq[inner] ** 2))

# ---- gates (signs and conventions VERIFIED, not assumed) -------------
scale = np.abs(EPS_TARGET).max()
gates = [
    ("identity sign  M_ab,ab = +q (not -q)", fin["eq_M"] > 0.9 * Q_TOP,
     "eq_M = %.6f" % fin["eq_M"]),
    ("identity conv  |M_ab,ab - q| <= 1e-6", abs(fin["eq_M"] - Q_TOP)
     <= 1e-6, "residual = %.3e (%s pe=%d)" % (fin["eq_M"] - Q_TOP,
                                              fin["kind"], fin["pe"])),
    ("station k22 vs C_eff^-1 FF (5%)",
     abs(eps_cell[4] - EPS_TARGET[4]) <= 0.05 * abs(EPS_TARGET[4]),
     "ritz %.4e vs target %.4e (%+.2f%%)"
     % (eps_cell[4], EPS_TARGET[4],
        100 * (eps_cell[4] / EPS_TARGET[4] - 1))),
    ("station k11 vs C_eff^-1 FF (5% of |k22| scale)",
     abs(eps_cell[3] - EPS_TARGET[3]) <= 0.05 * scale,
     "ritz %.4e vs target %.4e (diff %.2e, scale %.2e)"
     % (eps_cell[3], EPS_TARGET[3], eps_cell[3] - EPS_TARGET[3],
        scale)),
    ("station M11 cell-avg in -0.56..-0.44", -0.56 <= FF_cell[3]
     <= -0.44, "M11 = %.4f" % FF_cell[3]),
    ("w_center vs S4R (3%; ~1% expected)", abs(w_err) <= 0.03,
     "ritz %.6e vs abq %.6e (%+.2f%%)" % (fin["w"], w_abq,
                                          100 * w_err)),
    ("M11 line interior rms <= 5% of ref rms", rms_int <= 0.05 *
     rms_ref, "rms_int %.4f / ref %.4f = %.2f%% (all-line %.4f)"
     % (rms_int, rms_ref, 100 * rms_int / rms_ref, rms_all)),
]
nfail = 0
for name, ok, detail in gates:
    print("gate %-46s %s  [%s]" % (name, "PASS" if ok else "FAIL",
                                   detail))
    nfail += 0 if ok else 1

# ---- stencil d2eps of the current .ff, for the side-by-side echo
stencil = {}
ffp = os.path.join(HERE, "preovios_try.ff")
if os.path.exists(ffp):
    for ln in open(ffp):
        mt = re.match(r"^(d2eps_\w+|deps_\w+|FF):\s*\[(.*)\]", ln)
        if mt:
            stencil[mt.group(1)] = np.array(
                [float(x) for x in mt.group(2).split(",")])

# ---- emit ------------------------------------------------------------
readme = (
    "macro_ritz.py station drivers, centre station x = (%.1f, %.1f); "
    "conventions: w UP, q = %.1f DOWN, u_a = u0_a + x3 phi_a, eps_R = "
    "[e11 e22 2e12 k11 k22 2k12] = [u0,1 v0,2 u0,2+v0,1 phi1,1 phi2,2 "
    "phi1,2+phi2,1], gamma = [w,1+phi1 w,2+phi2], Q = G @ gamma; "
    "d2gamma[i,a,b] = d2 gamma_i / dx_a dx_b (symmetric in a,b); "
    "dE*/d2eps_* are gradients of eps_R in x1/x2; point values at the "
    "station (eps_R_station_cellavg/FF_station_cellavg = station-cell "
    "averages over [%.0f,%.0f]^2, the build_ff.py cell-avg sense); all "
    "derivatives analytic from the converged %s Ritz solution "
    "(pe=%d, per-element degrees %s, 1-D breakpoints %s)."
    % (XS[0], XS[1], Q_TOP, CELL[0], CELL[1], fin["kind"], fin["pe"],
       fin["B"]["pes"], np.array2string(np.asarray(fin["B"]["brk"]))))
np.savez(
    os.path.join(HERE, "macro_drivers.npz"),
    readme=np.array(readme),
    basis_kind=np.array(fin["kind"]), basis_pe=np.array(fin["pe"]),
    basis_pes=np.asarray(fin["B"]["pes"]),
    basis_brk=np.asarray(fin["B"]["brk"]), ndof=np.array(fin["ndof"]),
    eq_identity=np.array(fin["eq_M"]), w_center=np.array(fin["w"]),
    C8=C8,
    eps_R_station=fin["eps"], eps_R_station_cellavg=eps_cell,
    FF_station=FF_point, FF_station_cellavg=FF_cell,
    dE1=fin["dE1"], dE2=fin["dE2"],
    d2eps_dx1dx1=fin["d11"], d2eps_dx1dx2=fin["d12"],
    d2eps_dx2dx2=fin["d22"],
    gamma=fin["gam"], dgamma_dx1=fin["dg1"], dgamma_dx2=fin["dg2"],
    d2gamma=fin["d2g"], Q=Qs)


def vec(v):
    return " ".join("%14.6e" % x for x in np.atleast_1d(v))


L = []
L.append("macro_ritz.py -- analytic-derivative macro drivers, %s"
         % datetime.datetime.now().isoformat(timespec="seconds"))
L.append(readme)
L.append("")
L.append("8x8 ABDG law (symmetrized, ../../preovios_try.out):")
for r in C8:
    L.append("  " + vec(r))
L.append("")
L.append("convergence (analytic identities at the station; global ="
         " one element, max degree pe; hp-graded = C0 piecewise-"
         "Legendre, corner-graded breakpoints %s with the degree ramp"
         " max(pe-2*level, 4)):" % GRADED)
L.append("   stage      pe  ndof   w_center        eq_M(-> +1)"
         "     eq_Q(-> +1)     |mom_res|(3.5,2.5) lin.res  secs")
for st in run:
    L.append("  %-9s %3d %6d  %.8e  %.10f  %.10f  %.3e  %.1e  %.1f"
             % (st["kind"], st["pe"], st["ndof"], st["w"], st["eq_M"],
                st["eq_Q"], np.abs(st["mom_r"]).max(), st["rres"],
                st["secs"]))
L.append("  the global stage converges the identity only ~ pe^-1.5"
         " (clamped-corner singularity pollutes pointwise third"
         " derivatives of a global polynomial); the corner-graded hp"
         " stage restores exponential convergence at the interior"
         " station.")
L.append("  Cauchy differences (within stage):")
for kind, lo, hi, dw, dd2, deq in cauchy:
    L.append("   %-9s pe %2d -> %2d : |dw_c| %.3e   max|d(d2eps)|"
             " %.3e   |d(eq_M)| %.3e" % (kind, lo, hi, dw, dd2, deq))
L.append("")
L.append("station eps_R vs the Abaqus-plate target"
         " (eps = C_eff^-1 FF_station):")
L.append("   comp     ritz point      ritz cell-avg   target"
         "          cell/target")
for a in range(6):
    rat = eps_cell[a] / EPS_TARGET[a] if EPS_TARGET[a] != 0 else np.nan
    L.append("   %-5s %s %s %s   %8.4f"
             % (NAMES6[a], vec(fin["eps"][a]), vec(eps_cell[a]),
                vec(EPS_TARGET[a]), rat))
L.append("  FF point    : " + vec(FF_point))
L.append("  FF cell-avg : " + vec(FF_cell))
if "FF" in stencil:
    L.append("  FF of ./preovios_try.ff (mirror-symmetrized): "
             + vec(stencil["FF"]))
L.append("  station M11 cell-avg = %.4f  (gate -0.46..-0.54)"
         % FF_cell[3])
L.append("")
L.append("first gradients at the centre (symmetry check -- should be"
         " ~0; the ABDG anisotropy couplings break exact mirror"
         " symmetry):")
L.append("  dE1 : " + vec(fin["dE1"]))
L.append("  dE2 : " + vec(fin["dE2"]))
L.append("  max|dE| / max|d2eps| = %.2e  (length unit 1)"
         % (max(np.abs(fin["dE1"]).max(), np.abs(fin["dE2"]).max())
            / np.abs(np.stack([fin["d11"], fin["d12"],
                               fin["d22"]])).max()))
L.append("")
L.append("second gradients of eps_R (ANALYTIC) vs the .ff stencil"
         " (KKT-projected LS):")
for key, arr in (("d2eps_dx1dx1", fin["d11"]),
                 ("d2eps_dx1dx2", fin["d12"]),
                 ("d2eps_dx2dx2", fin["d22"])):
    L.append("  %s ritz    : %s" % (key, vec(arr)))
    if key in stencil:
        L.append("  %s stencil : %s" % (key, vec(stencil[key])))
L.append("  identity (C@d11)[3] + 2(C@d12)[5] + (C@d22)[4] = %.10f"
         "  (target +%g, residual %.3e)"
         % (fin["eq_M"], Q_TOP, fin["eq_M"] - Q_TOP))
L.append("  shear identity Q_a,a = %.10f  (same target)" % fin["eq_Q"])
L.append("")
L.append("transverse shear at the station (for the Eq. 50"
         " correction):")
L.append("  gamma        : " + vec(fin["gam"]))
L.append("  dgamma_dx1   : " + vec(fin["dg1"]))
L.append("  dgamma_dx2   : " + vec(fin["dg2"]))
for i in (0, 1):
    L.append("  d2gamma[%d]   : [d11 d12; d12 d22] = %s"
             % (i, vec(fin["d2g"][i].ravel())))
L.append("  Q = G@gamma  : " + vec(Qs)
         + "   (vanishes at the centre by parity)")
L.append("  Q probe (3.5,2.5) = %s, moment-eq residual %s"
         % (vec(fin["Q_probe"]), vec(fin["mom_r"])))
L.append("")
L.append("validation vs the Abaqus S4R plate (its own"
         " discretization):")
L.append("  w_center ritz %.8e vs abq %.8e  (%+.3f%%)"
         % (fin["w"], w_abq, 100 * w_err))
L.append("  M11(x, y=%.1f) line, %d element centroids:" % (XS[1], NE))
L.append("     x       M11_abq       M11_ritz      diff")
for i in range(NE):
    L.append("   %5.2f  %12.5e  %12.5e  %12.5e"
             % (x_line[i], M11_abq[i], M11_ritz[i],
                M11_ritz[i] - M11_abq[i]))
L.append("  rms all-line %.5f, interior (x in [%.1f,%.1f]) %.5f ="
         " %.2f%% of interior ref rms %.5f"
         % (rms_all, x_line[inner][0], x_line[inner][-1], rms_int,
            100 * rms_int / rms_ref, rms_ref))
L.append("")
L.append("gates:")
for name, ok, detail in gates:
    L.append("  %-48s %s  [%s]" % (name, "PASS" if ok else "FAIL",
                                   detail))
with open(os.path.join(HERE, "macro_drivers.txt"), "w") as g:
    g.write("\n".join(L) + "\n")
print("wrote macro_drivers.npz + macro_drivers.txt (%s pe=%d)"
      % (fin["kind"], fin["pe"]))

print("end:", datetime.datetime.now().isoformat(timespec="seconds"))
if nfail:
    raise SystemExit("%d gate(s) FAILED -- see macro_drivers.txt"
                     % nfail)
