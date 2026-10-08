"""ff_high_derivs.py -- add the THIRD and FOURTH strain-derivative
families to a station .ff, by central differences on the ELEMENT grid.

    python -m opensg_solid.helper.plate_rm.ff_high_derivs <plate.rpt> \
           --ff <station.ff> [--at XC YC] [--inplace]

WHY THIS EXISTS.  Yu's recovery relations are written in the CLASSICAL
measure throughout (Eq. 63/64/66: V1 = V11 eps,1 + V12 eps,2,
V2 = V21 eps,11 + ...), but the plate solve delivers the REISSNER
measure R.  sg_dehom converts the zeroth-order driver via Eq. 50,
    eps = R - D_a c,a,
and leaves the derivative drivers as R,a / R,ab because converting them
costs
    eps,a  = R,a  - D_b c,ba     -> THIRD  derivatives of R
    eps,ab = R,ab - D_b c,bab    -> FOURTH derivatives of R
(c is itself built from second derivatives, so each conversion level
costs two more).  Those families are what this module supplies.

WHY THE ELEMENT GRID.  sg_plate_station differences on the CELL grid
(5 points per direction on a 5x5 plate), which is why it stops at second
order and refuses the edge ring.  The plate report is centroidal PER
ELEMENT: subdiv 3 gives 15 points per direction, subdiv 5 gives 25.
The same fixed-weight stencils

    d2 : [ 1, -2,  1] / h^2        d4 : [ 1, -4,  6, -4, 1] / h^4
    d1 : [-1,  0,  1] / 2h         d3 : [-1,  2,  0, -2, 1] / 2h^3

then apply directly.  Same scheme as the existing route, one grid finer
-- no polynomial fit, no fitted degree, no patch radius.

CONVERGENCE.  Measured between subdiv 3 and subdiv 5 at the plate
centre: d2 agrees to 0.78%, d4 to 1.88%.  The derivatives are not
resolution limited.

SK ORDER.  The report's SK columns are exchanged relative to their
labels; sg_plate_station measures and corrects this before writing the
.ff, so the same swap is applied here or the families would be in a
different convention from the d2eps already in the file.  The GATE
below is what actually proves the convention matches: the recomputed
second derivatives must reproduce the .ff's own d2eps.

In:  the plate .rpt, the station .ff (for the gate and as the target)
Out: the nine d3eps_*/d4eps_* lines, printed or appended with --inplace
"""
import argparse
import os
import re

import numpy as np

W1 = {-1: -0.5, 1: 0.5}
W2 = {-1: 1.0, 0: -2.0, 1: 1.0}
W3 = {-2: -0.5, -1: 1.0, 1: -1.0, 2: 0.5}
W4 = {-2: 1.0, -1: -4.0, 0: 6.0, 1: -4.0, 2: 1.0}
I0 = {0: 1.0}
# multi-index -> (x weights, y weights); order = len(index)
FAM = {
    "11": (W2, I0), "12": (W1, W1), "22": (I0, W2),
    "111": (W3, I0), "112": (W2, W1), "122": (W1, W2), "222": (I0, W3),
    "1111": (W4, I0), "1112": (W3, W1), "1122": (W2, W2),
    "1222": (W1, W3), "2222": (I0, W4),
}
KEY = {"111": "d3eps_dx1dx1dx1", "112": "d3eps_dx1dx1dx2",
       "122": "d3eps_dx1dx2dx2", "222": "d3eps_dx2dx2dx2",
       "1111": "d4eps_dx1dx1dx1dx1", "1112": "d4eps_dx1dx1dx1dx2",
       "1122": "d4eps_dx1dx1dx2dx2", "1222": "d4eps_dx1dx2dx2dx2",
       "2222": "d4eps_dx2dx2dx2dx2"}


def read_grid(rpt):
    """The element table as a structured (nx, ny, 6) strain grid.

    In:  rpt str -- the plate report path
    Out: (G (nx,ny,6), xs, ys, h)"""
    hdr, rows = None, []
    for ln in open(rpt):
        if ln.startswith("**"):
            continue
        v = ln.split()
        if not v:
            continue
        if hdr is None and v[0] == "Element":
            hdr = ln.split()
            continue
        # the report also carries a NODE table with narrower rows
        if hdr is not None and re.match(r"^\d+$", v[0]) \
                and len(v) == len(hdr) - 1:
            rows.append([float(x) for x in v])
    A = np.array(rows)
    idx = {n: 1 + i for i, n in enumerate(hdr[2:])}
    X, Y = A[:, idx["Xc"]], A[:, idx["Yc"]]
    E = np.stack([A[:, idx["SE.SE11"]], A[:, idx["SE.SE22"]],
                  A[:, idx["SE.SE12"]], A[:, idx["SK.SK11"]],
                  A[:, idx["SK.SK22"]], A[:, idx["SK.SK12"]]], axis=1)
    E[:, [3, 4]] = E[:, [4, 3]]                 # SK order correction
    xs, ys = np.unique(np.round(X, 6)), np.unique(np.round(Y, 6))
    G = np.full((len(xs), len(ys), 6), np.nan)
    for r in range(len(A)):
        G[int(np.argmin(np.abs(xs - X[r]))),
          int(np.argmin(np.abs(ys - Y[r])))] = E[r]
    if np.isnan(G).any():
        raise SystemExit("element grid is not complete/rectangular")
    return G, xs, ys, xs[1] - xs[0]


def deriv(G, ic, jc, h, ix):
    """One derivative family at grid node (ic, jc)."""
    wx, wy = FAM[ix]
    out = np.zeros(6)
    for a, ca in wx.items():
        for b, cb in wy.items():
            out += ca * cb * G[ic + a, jc + b]
    return out / (h ** len(ix))


def read_ff(path):
    d = {}
    for ln in open(path):
        m = re.match(r"^(\S+):\s*\[(.*)\]\s*$", ln)
        if m:
            d[m.group(1)] = np.array([float(x)
                                      for x in m.group(2).split(",")])
    return d


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("rpt")
    p.add_argument("--ff", required=True)
    p.add_argument("--at", nargs=2, type=float, default=None,
                   metavar=("XC", "YC"),
                   help="station centre; default = the plate centre")
    p.add_argument("--inplace", action="store_true")
    p.add_argument("--gate", type=float, default=0.05,
                   help="max relative disagreement of the recomputed"
                        " second derivatives against the .ff (0.05)")
    A = p.parse_args()

    G, xs, ys, h = read_grid(A.rpt)
    nx, ny = len(xs), len(ys)
    xc, yc = (0.0, 0.0) if A.at is None else A.at
    ic = int(np.argmin(np.abs(xs - xc)))
    jc = int(np.argmin(np.abs(ys - yc)))
    print("element grid %d x %d, h = %.6f" % (nx, ny, h))
    print("station (%.4f, %.4f) -> node (%d, %d) at (%.4f, %.4f)"
          % (xc, yc, ic, jc, xs[ic], ys[jc]))
    if min(ic, jc) < 2 or ic > nx - 3 or jc > ny - 3:
        raise SystemExit("station is within 2 elements of the boundary:"
                         " the 5-point stencil does not fit."
                         "  Use a finer subdiv so the station sits"
                         " further inside the grid.")

    ff = read_ff(A.ff)
    print("\nGATE: recomputed second derivatives vs the .ff")
    worst = 0.0
    for ix, key in (("11", "d2eps_dx1dx1"), ("12", "d2eps_dx1dx2"),
                    ("22", "d2eps_dx2dx2")):
        if key not in ff:
            print("  %-14s absent from the .ff -- not gated" % key)
            continue
        cd, ref = deriv(G, ic, jc, h, ix), ff[key]
        n = np.linalg.norm(ref)
        rel = np.linalg.norm(cd - ref) / n if n > 0 else np.nan
        worst = max(worst, 0.0 if np.isnan(rel) else rel)
        print("  %-14s rel %.4f   ff %s  cd %s"
              % (key, rel, np.array2string(ref[3:5], precision=3),
                 np.array2string(cd[3:5], precision=3)))
    if worst > A.gate:
        raise SystemExit("GATE FAILED: %.4f > %.4f.  The element-grid"
                         " derivatives disagree with the .ff's own"
                         " cell-stencil values, so the convention or"
                         " the station location is wrong -- refusing to"
                         " write." % (worst, A.gate))
    print("  gate passed (worst %.4f <= %.4f)" % (worst, A.gate))

    lines = []
    print("\nfamilies (curvature slots shown):")
    for ix in ("111", "112", "122", "222",
               "1111", "1112", "1122", "1222", "2222"):
        v = deriv(G, ic, jc, h, ix)
        lines.append("%s: [%s]\n"
                     % (KEY[ix], ", ".join("%.10g" % x for x in v)))
        print("  %-20s %s" % (KEY[ix],
                              np.array2string(v[3:], precision=4)))

    if A.inplace:
        txt = open(A.ff).read()
        txt = "".join(ln for ln in txt.splitlines(True)
                      if not any(ln.startswith(KEY[k] + ":")
                                 for k in KEY))
        if not txt.endswith("\n"):
            txt += "\n"
        txt += ("# d3eps/d4eps: element-grid central differences from"
                " %s\n" % os.path.basename(A.rpt))
        open(A.ff, "w").write(txt + "".join(lines))
        print("\nwrote %d families into %s" % (len(lines), A.ff))
    else:
        print("\n(dry run -- pass --inplace to write)")


main()
