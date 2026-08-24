"""build_ff.py -- the dehomogenization load, taken from a plate run's
OWN internal resultants at the plate centre.

    python build_ff.py <plate job stem> <SG stem>

The RM plate already solved the macro problem, so its section forces at
the centre element ARE the generalized forces the SG has to be driven
with -- no hand equilibrium formula, no assumed load path.

TWO SEPARATE MODELS FEED THIS, and it matters which supplies what:

  FF  comes from the PLATE JOB.  Section forces are equilibrium
      quantities; at the centre of a uniformly loaded panel they are
      fixed by the load and the span, so they belong to the panel.
  C_eff comes from the SG.  It is the constitutive law, and it is what
      turns those forces into the macro strain the SG is driven with.
      deps_dx1/deps_dx2 are central differences of eps = C_eff^-1 FF, so
      they inherit the SG's law too -- which is why the SG stem, not the
      plate job, decides them.

Q1/Q2 are deliberately NOT written.  The transverse shear resultant is
not an independent input a user supplies: in the Eq. 63 refined
recovery, sigma_13/sigma_23 follow from the strain DERIVATIVES, which is
what deps_dx1/deps_dx2 carry.  Writing Q as well would switch on the
extra Q-consistency rescale, a separate correction rather than part of
"here is my load".

In:  <plate>_SF.rpt, <plate>_SM.rpt, <plate>.inp, <SG>.out
Out: <SG>.ff  + the station coordinates printed, so the same point can
     be cut out of the 3-D model
"""
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PL = sys.argv[1] if len(sys.argv) > 1 else "L_min_2UC_3D_plate"
SG = sys.argv[2] if len(sys.argv) > 2 else "Core_2UC_SC"

# build_ff reads its inputs from ITS OWN folder (HERE), not the shell's
# working directory -- run it where the plate job files live.  Name the
# missing pieces up front instead of tracebacking on the first open.
_need = [PL + ".inp", PL + "_SF.rpt", PL + "_SM.rpt", SG + ".out"]
_missing = [f for f in _need if not os.path.exists(os.path.join(HERE, f))]
if _missing:
    raise SystemExit(
        "build_ff: missing next to the script (%s):\n  %s\n"
        "The plate job files must sit in the SAME folder as build_ff.py."
        "  For HC_pm45 that is opensg/dehomo/ or the HC_pm45 root -- cd"
        " there and rerun:\n"
        "  python build_ff.py %s %s" % (HERE, "\n  ".join(_missing),
                                        PL, SG))

# THE SYMMETRY RULE.  The panel centre falls on an ELEMENT BOUNDARY of
# the plate lattice, so the nearest-centroid station sits half a pitch
# off the symmetry plane -- where the shear force Q1 = -q x' is
# genuinely nonzero and the central differences deliver the matching
# nonzero deps_dx*.  A 3-D reference extracted AT the symmetry plane
# (the Richardson symmetric-pair average kills the odd-in-span content)
# carries Q = 0 there, so those drivers must not be fed to the
# recovery.  With SYMMETRIZE on, the station state is rebuilt as the
# panel centre's own by symmetry: FF and the SECOND derivatives are
# averaged over the station's mirror images about the centre (their odd
# content cancels), and the FIRST derivatives are written as exactly
# zero.  Verified on HC_pm45: with this + q_reaction: tau the RM
# recovery beats the classical one in all 24 (path, component,
# material-segment) cells (compare/four_way/segment_rms_final.txt).
SYMMETRIZE = True


def read_rpt(path):
    """Abaqus element report -> (labels, {column name: values}).

    The columns come out in the odb's internal order, NOT numeric order
    (SF1, SF2, SF6, SF3, SF4, SF5), so they are taken BY NAME.
    """
    hdr, lab, rows = None, [], []
    for ln in open(path):
        s = ln.rstrip("\n")
        if not s.strip() or s.lstrip().startswith(("---", "@Loc", "Pt")):
            continue
        if hdr is None:
            if "Element Label" in s and re.search(r"\b(SF|SM|S)\.", s):
                hdr = s.replace("Element Label", "El").replace(
                    "Int", "Ip").split()
            continue
        if re.match(r"^\s*-?\d", s):
            v = s.split()
            if len(v) == len(hdr):
                try:
                    f = [float(x) for x in v]
                except ValueError:
                    continue
                lab.append(int(f[0]))
                rows.append(f)
    a = np.array(rows)
    return np.array(lab), {h: a[:, i] for i, h in enumerate(hdr)}


# ---- plate mesh: element centroids
nodes, elems, on = {}, {}, None
for ln in open(os.path.join(HERE, PL + ".inp")):
    s = ln.strip()
    if s.startswith("**"):
        continue
    if s.startswith("*"):
        k = s.split(",")[0].lower()
        on = ("n" if k == "*node" and "output" not in s.lower()
              else ("e" if k == "*element" else None))
        continue
    if not s or on is None:
        continue
    v = [x.strip() for x in s.split(",")]
    if on == "n" and len(v) >= 4:
        nodes[int(v[0])] = (float(v[1]), float(v[2]))
    elif on == "e" and v[0].isdigit():
        elems[int(v[0])] = [int(x) for x in v[1:]]
cen = {e: np.mean([nodes[n] for n in ns], axis=0)
       for e, ns in elems.items()}

lf, SF = read_rpt(os.path.join(HERE, PL + "_SF.rpt"))
lm, SM = read_rpt(os.path.join(HERE, PL + "_SM.rpt"))
assert (lf == lm).all(), "SF and SM report different element sets"
XY = np.array([cen[e] for e in lf])

# .ff force order: N11 N22 N12 M11 M22 M12
FF = np.column_stack([SF["SF.SF1"], SF["SF.SF2"], SF["SF.SF3"],
                      SM["SM.SM1"], SM["SM.SM2"], SM["SM.SM3"]])
QQ = np.column_stack([SF["SF.SF4"], SF["SF.SF5"]])   # reported, not used

# ---- the SG's law
rows = []
for ln in open(os.path.join(HERE, SG + ".out")):
    v = ln.split()
    if len(v) == 8 and re.match(r"^-?\d", v[0]):
        try:
            rows.append([float(x) for x in v])
        except ValueError:
            pass
    if len(rows) == 8:
        break
K = np.array(rows)
C6 = 0.5 * (K[:6, :6] + K[:6, :6].T)

ctr = np.array([XY[:, 0].mean(), XY[:, 1].mean()])
i0 = int(np.argmin(((XY - ctr) ** 2).sum(axis=1)))
x0, y0 = XY[i0]
dx = np.unique(np.round(np.diff(np.unique(XY[:, 0])), 9))[0]
dy = np.unique(np.round(np.diff(np.unique(XY[:, 1])), 9))[0]
print("plate     : %s, %d elements, spacing %.6f x %.6f"
      % (PL, len(lf), dx, dy))
print("  centre (%.5f, %.5f) -> station element %d at (%.5f, %.5f)"
      % (ctr[0], ctr[1], lf[i0], x0, y0))
print("SG law    : %s.out   D11 %.5e  D22 %.5e"
      % (SG, C6[3, 3], C6[4, 4]))


def at(x, y):
    k = int(np.argmin((XY[:, 0] - x) ** 2 + (XY[:, 1] - y) ** 2))
    if abs(XY[k, 0] - x) > 1e-6 or abs(XY[k, 1] - y) > 1e-6:
        return None
    return k


# the mirror images of the station about the panel centre; averaging
# over them extracts the centre's own (even) state -- see SYMMETRIZE
mset = [i0]
if SYMMETRIZE:
    mm = []
    for xm, ym in ((2 * ctr[0] - x0, y0), (x0, 2 * ctr[1] - y0),
                   (2 * ctr[0] - x0, 2 * ctr[1] - y0)):
        k = at(xm, ym)
        if k is not None and k not in mm and k != i0:
            mm.append(k)
    mset = [i0] + mm
    if len(mset) > 1:
        FF[i0] = np.mean([FF[k] for k in mset], axis=0)
        print("\nSYMMETRIZED: FF = mean over the %d mirror stations %s"
              % (len(mset), [lf[k] for k in mset]))
    else:
        print("\nSYMMETRIZE requested but no mirror stations found --"
              " keeping the raw station")

eps = np.linalg.solve(C6, FF[i0])
print("\nstation resultants (the plate's own SF/SM, not a formula)")
for j, t in enumerate(["N11", "N22", "N12", "M11", "M22", "M12"]):
    print("  %-4s %14.6e" % (t, FF[i0, j]))
print("  %-4s %14.6e   %-4s %14.6e   (reported, NOT written)"
      % ("Q1", QQ[i0, 0], "Q2", QQ[i0, 1]))
print("\n  eps = C_eff^-1 FF  [e11 e22 2e12 k11 k22 2k12]")
print("   ", np.array2string(eps, precision=6))

# ---- finite differences on the plate's own Gauss-point lattice.
# S4R carries ONE integration point, at the element centre, so the 600
# reported (6,1) resultant vectors sit on a regular 60 x 10 grid of
# spacing (dx, dy).  That regularity is what makes plain central
# differences valid -- and second order accurate -- with no fitting.
#
#   E,1   = (e[i+1,j] - e[i-1,j]) / (2 h1)
#   E,2   = (e[i,j+1] - e[i,j-1]) / (2 h2)
#   E,11  = (e[i+1,j] - 2 e[i,j] + e[i-1,j]) / h1^2
#   E,22  = (e[i,j+1] - 2 e[i,j] + e[i,j-1]) / h2^2
#   E,12  = (e[i+1,j+1] - e[i+1,j-1] - e[i-1,j+1] + e[i-1,j-1])
#           / (4 h1 h2)
#
# all evaluated on eps = C_eff^-1 FF, so every one of them inherits the
# SG's law, not the plate mesh's.
def eps_at(ix, iy):
    """eps at the station offset by (ix, iy) ELEMENTS, or None."""
    k = at(x0 + ix * dx, y0 + iy * dy)
    return None if k is None else np.linalg.solve(C6, FF[k])


st = {}
for a in (-1, 0, 1):
    for b in (-1, 0, 1):
        st[(a, b)] = eps_at(a, b)
missing = [k for k, v in st.items() if v is None]
if missing:
    raise SystemExit("station is too close to an edge: the 3x3 stencil "
                     "is missing %s -- move it inward" % missing)

der = {
    "deps_dx1": (st[(1, 0)] - st[(-1, 0)]) / (2.0 * dx),
    "deps_dx2": (st[(0, 1)] - st[(0, -1)]) / (2.0 * dy),
}
der2 = {
    "d2eps_dx1dx1": (st[(1, 0)] - 2.0 * st[(0, 0)] + st[(-1, 0)])
    / dx ** 2,
    "d2eps_dx2dx2": (st[(0, 1)] - 2.0 * st[(0, 0)] + st[(0, -1)])
    / dy ** 2,
    "d2eps_dx1dx2": (st[(1, 1)] - st[(1, -1)] - st[(-1, 1)]
                     + st[(-1, -1)]) / (4.0 * dx * dy),
}
print("\nfinite differences on the Gauss-point lattice "
      "(h1 = %.6f, h2 = %.6f)" % (dx, dy))
for tag in ("deps_dx1", "deps_dx2"):
    print("  %-13s" % tag, np.array2string(der[tag], precision=6))
for tag in ("d2eps_dx1dx1", "d2eps_dx2dx2", "d2eps_dx1dx2"):
    print("  %-13s" % tag, np.array2string(der2[tag], precision=6))

if SYMMETRIZE and len(mset) > 1:
    # the centre's own drivers: first derivatives vanish there (Q = 0
    # at the symmetry plane); the second derivatives are even, so the
    # mirror average cancels their odd half-pitch error
    def der2_at(xs, ys):
        stm = {}
        for a in (-1, 0, 1):
            for b in (-1, 0, 1):
                k = at(xs + a * dx, ys + b * dy)
                if k is None:
                    return None
                stm[(a, b)] = np.linalg.solve(C6, FF[k])
        return {
            "d2eps_dx1dx1": (stm[(1, 0)] - 2 * stm[(0, 0)]
                             + stm[(-1, 0)]) / dx ** 2,
            "d2eps_dx2dx2": (stm[(0, 1)] - 2 * stm[(0, 0)]
                             + stm[(0, -1)]) / dy ** 2,
            "d2eps_dx1dx2": (stm[(1, 1)] - stm[(1, -1)] - stm[(-1, 1)]
                             + stm[(-1, -1)]) / (4 * dx * dy),
        }

    packs = [p for p in (der2_at(XY[k, 0], XY[k, 1]) for k in mset)
             if p is not None]
    der = {t: np.zeros(6) for t in der}
    der2 = {t: np.mean([p[t] for p in packs], axis=0) for t in der2}
    print("\nSYMMETRIZED drivers: deps_dx1 = deps_dx2 = 0 (Q = 0 at the"
          " symmetry plane); d2eps = mean over %d mirror stencils"
          % len(packs))
    for tag in ("d2eps_dx1dx1", "d2eps_dx2dx2", "d2eps_dx1dx2"):
        print("  %-13s" % tag, np.array2string(der2[tag], precision=6))

# ---- is the second difference signal, or amplified round-off?
#
# The test has to be made on FF, not on eps.  For a clamped-clamped
# strip under uniform q the span moment runs from +q L^2/12 at the ends
# to -q L^2/24 at midspan, so the parabola opens UPWARD and
#
#     d2 M11 / dx1^2 = +q      exactly, independent of the law.
#
# eps mixes all six resultants through the compliance, so checking the
# k11 row of E,11 against q S[3,3] alone would be comparing against an
# incomplete prediction -- the M22 curvature contributes too.  Checking
# FF isolates the equilibrium statement.
def ff_at(ix, iy):
    k = at(x0 + ix * dx, y0 + iy * dy)
    return None if k is None else FF[k]


d2FF = (ff_at(1, 0) - 2.0 * ff_at(0, 0) + ff_at(-1, 0)) / dx ** 2
q_mag = 0.1
print("\n  check : d2 FF / dx1^2 =", np.array2string(d2FF, precision=6))
print("          the M11 component is %.6e, and equilibrium of a"
      % d2FF[3])
print("          clamped-clamped strip forces it to +q = %.6e"
      "   -> off by %.2f %%"
      % (q_mag, 100 * (d2FF[3] - q_mag) / q_mag))
print("          (so E,11 = S d2FF/dx1^2 is signal, not round-off;"
      " its k11 row\n           differs from q S[3,3] only because"
      " d2 M22/dx1^2 = %.3e\n           contributes through the"
      " compliance)" % d2FF[4])

ff = os.path.join(HERE, SG + ".ff")
# carry the face-pressure keys of a PRE-EXISTING .ff forward -- they
# are the analyst's load declaration, not something this script can
# derive from the plate reports (kills the "rebuild dropped qt6" trap)
carry, carry_s = {}, {}
if os.path.exists(ff):
    import yaml as _y
    try:
        old = _y.safe_load(open(ff))
        for k in ("qt6", "qb6"):
            if isinstance(old, dict) and old.get(k) is not None:
                carry[k] = [float(v) for v in old[k]]
        # scalar declarations ride the same way (q_reaction: uniform|tau
        # -- how the pressure column reacts its net force, see
        # opensg_solid.cli.read_ff_state)
        for k in ("q_reaction",):
            if isinstance(old, dict) and old.get(k) is not None:
                carry_s[k] = str(old[k]).strip()
    except Exception:                                   # noqa: BLE001
        pass
with open(ff, "w") as g:
    g.write("# dehomogenization load for %s.yaml\n" % SG)
    g.write("# FF  = section forces of plate run %s at its centre\n" % PL)
    g.write("#       element %d, x1 = %.6f, x2 = %.6f\n"
            % (lf[i0], x0, y0))
    if SYMMETRIZE and len(mset) > 1:
        g.write("# SYMMETRIZED about the panel centre (elements %s):\n"
                "#       FF and d2eps mirror-averaged, deps_dx* = 0\n"
                "#       (Q = 0 at the symmetry plane)\n"
                % [int(lf[k]) for k in mset])
    g.write("# eps = C_eff^-1 FF with C_eff from %s.out; deps_dx* are\n"
            "#       central differences of that eps over the"
            " neighbouring elements\n" % SG)
    g.write("# Q is intentionally absent: transverse shear is not a user"
            " input.\n")
    g.write("FF: [%s]\n" % ", ".join("%.8e" % v for v in FF[i0]))
    for k in ("deps_dx1", "deps_dx2"):
        g.write("%s: [%s]\n" % (k, ", ".join("%.8e" % v
                                             for v in der[k])))
    # SECOND derivatives, LIVE: the Eq. 64-66 two-chain recovery now
    # exists in the 2-D route and read_ff_state consumes these keys.
    g.write("# SECOND derivatives of eps, same lattice, second-order"
            " central stencils --\n# drive the Eq. 64-66 two-chain"
            " recovery.\n")
    for k in ("d2eps_dx1dx1", "d2eps_dx2dx2", "d2eps_dx1dx2"):
        g.write("%s: [%s]\n" % (k, ", ".join("%.8e" % v
                                             for v in der2[k])))
    for k, v in carry.items():
        g.write("%s: [%s]   # carried from the previous .ff\n"
                % (k, ", ".join("%r" % x for x in v)))
    for k, v in carry_s.items():
        g.write("%s: %s   # carried from the previous .ff\n" % (k, v))
np.savez(os.path.join(HERE, "_ff_station.npz"), x=x0, y=y0,
         elem=lf[i0], FF=FF[i0], Q=QQ[i0], eps=eps)
print("\nwrote %s\n" % os.path.basename(ff))
print(open(ff).read())
