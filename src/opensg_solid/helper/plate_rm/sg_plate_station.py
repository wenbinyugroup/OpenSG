"""sg_plate_station.py -- the LEVEL-1 (cell) macro state of one plate
station, from an Abaqus plate report + the two-level cellmap.

The dehomogenization of a 3-D plate SG is driven by CELL quantities:
one unit cell of the tiling = one SG = one station.  When the plate is
meshed with subdiv sub-elements per cell edge (the two-level scheme),
every cell carries subdiv^2 sub-elements whose element table rows and
whose nodes all belong to it.  This module reduces that sub-cell data
to the cell's macro state:

    strain  [e11 e22 2e12 k11 k22 2k12]   mean of the cell's
                                          sub-element states (SE + SK)
    force   [N11 N22 N12 M11 M22 M12]     mean of the sub-element
                                          resultants -- written to the
                                          .ff as the 1: fallback only;
                                          the 0: strain drives the dehom
    u, theta                              the cell-average of the nodal
                                          U / UR over the nodes inside
                                          the cell footprint (level-1
                                          value; the centre-node value
                                          is also returned for
                                          reference -- they differ by
                                          O(h^2) curvature)
    dE1, dE2, dE11, dE12, dE22            5-point central differences
                                          on the CELL grid: fixed
                                          weights, O(h^4), no fitting

CONVENTIONS MEASURED, NOT TRUSTED.  Abaqus's componentLabels for these
fields are unreliable (SE repeats a name; SM is moment-about-axis).
The packing is found by requiring {N, M} = [ABD] {e, k} at every
element, and the search is refused if no packing satisfies it -- that
also catches pairing a report with the wrong law.

That gate has ONE blind spot, and it bit: kappa and M are work
conjugate, so a free SM permutation can absorb an exchange of the SK
columns and still satisfy {N, M} = [ABD] {e, k} exactly.  The SK order
is therefore measured separately, against the plate's own nodal
rotations (measure_sk_order), and the result is fed INTO the packing
search so the SM permutation is measured against correctly ordered
curvatures.

names
  read_plate_rpt(path)         the element/node tables of the combined
                               report dump
  station_state(rpt, cellmap, law, x=0, y=0)
                               -> the dict of everything above
  write_station_ff(out, state, q0=None, ...)
                               -> build_ff on the state (0:/1:, u,
                               theta, derivs, qt6)
"""
import csv
import itertools
import math
import os
import re

import numpy as np


# ----------------------------------------------------------- report IO
def read_plate_rpt(path):
    """The combined plate report (dump_plate_rpt layout).

    In:  path str
    Out: {"elem": (cols, {label: row}), "node": (cols, {label: row})}"""
    tabs, cols, rows, kind = {}, None, None, None
    for ln in open(path):
        if ln.startswith("**"):
            continue
        v = ln.split()
        if not v:
            continue
        if not v[0].lstrip("-").isdigit():
            if kind and kind not in tabs:
                tabs[kind] = (cols, rows)
            kind = ("elem" if "Element" in ln
                    else "node" if "Node" in ln else None)
            cols = ln.replace("Element Label", "").replace(
                "Node Label", "").split()
            rows = {}
            continue
        if kind and len(v) - 1 == len(cols):
            rows[int(v[0])] = [float(x) for x in v[1:]]
    if kind and kind not in tabs:
        tabs[kind] = (cols, rows)
    return tabs


def read_cellmap(path):
    """The two-level element map the plate generator writes.

    In:  path str -- ..._cellmap.csv
    Out: {(Ic, Jc): {"xc", "yc", "subs": [element labels]}}"""
    cells = {}
    with open(path) as f:
        for v in csv.reader(f):
            if not v or v[0].startswith("#") or v[0] == "cell":
                continue
            cells[(int(v[1]), int(v[2]))] = {
                "xc": float(v[3]), "yc": float(v[4]),
                "subs": [int(s) for s in v[5:]]}
    return cells


def read_abd(path):
    """The 6x6 ABD block of an 8x8 .out.  In: path.  Out: (6,6) array."""
    R = []
    for ln in open(path):
        v = ln.split()
        if len(v) == 8 and re.match(r"^[-+]?\d", v[0]):
            try:
                R.append([float(x) for x in v])
            except ValueError:
                pass
        if len(R) == 8:
            break
    K = np.asarray(R)
    return 0.5 * (K[:6, :6] + K[:6, :6].T)


# ------------------------------------------------- packing measurement
def measure_sk_order(T, cells):
    """Which SK column holds kappa11 -- measured, not read off a label.

    The {N, M} = [ABD] {e, k} gate CANNOT order the SK columns: kappa
    and M are work-conjugate, so measure_packing's free SM permutation
    absorbs any SK exchange and the residual still passes.  It did:
    until 2026-08-31 this route ran with slots 4/5 exchanged at a gate
    residual of 1.4e-06, which left the classical recovery intact at
    symmetric stations (kappa11 = kappa22 there) while feeding every
    derivative family the wrong component.

    The plate's own NODAL ROTATIONS decide it with no component label
    involved -- for a Reissner-Mindlin shell

        kappa11 = -d(UR2)/dx1        kappa22 = +d(UR1)/dx2

    exactly (unlike -w,ab, which carries a shear-deformation offset).
    The single global sign/scale between these and SK is a convention;
    it is fitted on the order-INVARIANT sum (kappa11 + kappa22) and
    divided out, so it cannot bias the verdict.  The order is then read
    off the cells where kappa11 != kappa22, where the two candidates
    are tens of percent apart.

    In:  T the read_plate_rpt tables; cells the cellmap dict
    Out: {"order": (i, j) -- SK column indices holding (k11, k22),
         "err": (as_is, swapped) relative errors, "sign": float,
         "n": int} or None when no structured rotation grid is
         available (caller then keeps file order)."""
    node = T.get("node")
    if not node:
        return None
    nc, nr = node
    ec, er = T["elem"]
    i_sk = [i for i, c in enumerate(ec) if c.startswith("SK.")]
    if len(i_sk) < 2:
        return None
    try:
        ix, iy = nc.index("X"), nc.index("Y")
        ir1 = next(i for i, c in enumerate(nc) if c.endswith("UR1"))
        ir2 = next(i for i, c in enumerate(nc) if c.endswith("UR2"))
    except (ValueError, StopIteration):
        return None
    P = np.array([[v[ix], v[iy], v[ir1], v[ir2]] for v in nr.values()])
    xs = np.unique(np.round(P[:, 0], 9))
    ys = np.unique(np.round(P[:, 1], 9))
    if len(xs) < 3 or len(ys) < 3:
        return None
    xi = {v: i for i, v in enumerate(xs)}
    yi = {v: i for i, v in enumerate(ys)}
    R1 = np.full((len(xs), len(ys)), np.nan)
    R2 = np.full((len(xs), len(ys)), np.nan)
    for px, py, r1, r2 in P:
        R1[xi[round(px, 9)], yi[round(py, 9)]] = r1
        R2[xi[round(px, 9)], yi[round(py, 9)]] = r2
    hx, hy = np.diff(xs).min(), np.diff(ys).min()
    hc = (max(c["xc"] for c in cells.values())
          - min(c["xc"] for c in cells.values()))
    hc = hc / max(1, (1 + max(k[0] for k in cells)) - 1)

    rows = []
    for key, c in cells.items():
        k11, k22 = [], []
        for i, px in enumerate(xs):
            if abs(px - c["xc"]) > hc / 2 or i in (0, len(xs) - 1):
                continue
            for j, py in enumerate(ys):
                if abs(py - c["yc"]) > hc / 2 or j in (0, len(ys) - 1):
                    continue
                a = -(R2[i + 1, j] - R2[i - 1, j]) / (2 * hx)
                b = (R1[i, j + 1] - R1[i, j - 1]) / (2 * hy)
                if not np.isnan(a):
                    k11.append(a)
                if not np.isnan(b):
                    k22.append(b)
        if not k11 or not k22:
            continue
        sk = np.mean([[er[e][i_sk[0]], er[e][i_sk[1]]]
                      for e in c["subs"]], axis=0)
        rows.append((np.mean(k11), np.mean(k22), sk[0], sk[1]))
    if not rows:
        return None
    A = np.array(rows)
    # the SUM is invariant under the exchange -> an unbiased sign/scale
    tot_n, tot_s = A[:, 0] + A[:, 1], A[:, 2] + A[:, 3]
    if not np.any(np.abs(tot_n) > 0):
        return None
    sign = float((tot_s @ tot_n) / (tot_n @ tot_n))
    p11, p22 = sign * A[:, 0], sign * A[:, 1]
    asym = np.abs(A[:, 0] - A[:, 1]) > 0.02 * np.maximum(
        np.abs(A[:, 0]), np.abs(A[:, 1]))
    if not asym.any():
        return None                       # nothing can discriminate
    sc = np.maximum(np.abs(p11), np.abs(p22))[asym]
    e_as = float(np.mean((np.abs(p11[asym] - A[asym, 2])
                          + np.abs(p22[asym] - A[asym, 3])) / sc))
    e_sw = float(np.mean((np.abs(p22[asym] - A[asym, 2])
                          + np.abs(p11[asym] - A[asym, 3])) / sc))
    order = (0, 1) if e_as <= e_sw else (1, 0)
    return {"order": order, "err": (e_as, e_sw), "sign": sign,
            "n": int(asym.sum())}


def measure_packing(ec, er, ABD, sk_order=None):
    """Find the (SM permutation, SE membrane slots, shear/twist factor)
    that satisfies {N,M} = ABD {e,k} at every element.

    The SK columns are NOT searched here -- this gate is provably blind
    to their order (see measure_sk_order).  Pass sk_order from that
    measurement; without it the file order is used.

    THE MEMBRANE ROWS ARE SCORED ON THEIR OWN SCALE, and that is the
    whole point of this version.  Until 2026-09-03 the residual was
    normalised by ONE scale, max(|N|, |M|) over the report.  On a
    bending-dominated station |M| ~ 1e2 while |N| ~ 1e-6, so a membrane
    mis-packing worth 3x the N-scale still scored ~1e-7 of that global
    scale -- below the moment noise floor -- and the SE triple was
    chosen by rounding noise.  It bit exactly that way on the HC 5x5
    plate: the winning triple was (e11, e33, 2 e22), i.e. Abaqus's
    THICKNESS strain in the e22 slot and twice the real e22 in the
    shear slot, at a "residual" of 1.75e-07.  Scoring N and M against
    their own scales separates the two candidates by seven orders
    (4.2e-07 vs 3.3e+00) and the right one wins on its own merit.

    Two further diagnostics come back for the caller to print, because
    a silently-chosen convention is what caused the bug:
      res_N/res_M  the two block residuals of the winner
      margin       best score / runner-up score.  ~1 means the data
                   CANNOT tell the packings apart (e.g. a pure-bending
                   station with no membrane action at all); the caller
                   should say so rather than imply a measurement.
      trace_free   the SE column triple whose sum vanishes elementwise,
                   if any: Abaqus's (e11, e22, e33) for a shell section
                   with the default nu_eff = 0.5.  A chosen membrane
                   pair that straddles this triple's third member is
                   the exact failure above, so it is worth seeing.

    In:  ec [str] element-table columns; er {label: row}; ABD (6,6);
         sk_order (i, j) | None -- SK columns holding (k11, k22)
    Out: dict {perm, tri, fs, fk, residual, res_N, res_M, margin,
         trace_free}; raises SystemExit if no packing reaches 1e-4
         relative on EITHER block."""
    i_se = [i for i, c in enumerate(ec) if c.startswith("SE.")]
    i_sk = [i for i, c in enumerate(ec) if c.startswith("SK.")]
    if sk_order is not None:
        i_sk = [i_sk[sk_order[0]], i_sk[sk_order[1]]] + i_sk[2:]
    jN = [ec.index("SF.SF%d" % k) for k in (1, 2, 3)]
    jM = [ec.index("SM.SM%d" % k) for k in (1, 2, 3)]

    V = np.array([er[k] for k in sorted(er)], float)     # (n_el, n_col)
    ABD = np.asarray(ABD, float)
    N_ac = V[:, jN]
    M_all = V[:, jM]
    # each block against ITS OWN scale -- the fix
    sN = float(np.abs(N_ac).max()) or 1.0
    sM = float(np.abs(M_all).max()) or 1.0

    # the trace-free SE triple, if the report carries one
    trace_free = None
    for t in itertools.combinations(range(len(i_se)), 3):
        s3 = V[:, [i_se[t[0]], i_se[t[1]], i_se[t[2]]]].sum(axis=1)
        col = np.abs(V[:, [i_se[q] for q in t]]).max() or 1.0
        # 1e-6 of the columns' own magnitude, not 1e-9: these are
        # %.8e-printed report numbers, so the identity can only hold to
        # the print precision (~1e-8 relative), and a 1e-9 gate never
        # fires -- the trap the SK docstring calls out, one level down
        if np.abs(s3).max() <= 1e-6 * col:
            trace_free = t
            break

    scored = []
    for tri in itertools.permutations(range(len(i_se)), 3):
        for fs, fk in itertools.product((1.0, 2.0), repeat=2):
            x = np.column_stack([
                V[:, i_se[tri[0]]], V[:, i_se[tri[1]]],
                fs * V[:, i_se[tri[2]]],
                V[:, i_sk[0]], V[:, i_sk[1]], fk * V[:, i_sk[2]]])
            pr = x @ ABD.T
            rN = float(np.abs(pr[:, :3] - N_ac).max()) / sN
            for perm in itertools.permutations(range(3)):
                M_ac = M_all[:, list(perm)]
                rM = float(np.abs(pr[:, 3:] - M_ac).max()) / sM
                scored.append((max(rN, rM), rN, rM, perm, tri, fs, fk))
    scored.sort(key=lambda r: r[0])
    score, rN, rM, perm, tri, fs, fk = scored[0]
    runner = next((r[0] for r in scored[1:] if r[4] != tri), None)
    degenerate_N = False
    if score > 1e-4:
        # THE DEGENERATE-MEMBRANE CASE.  A plate deck with NO in-plane
        # forcing at all (the classical / F-less deck under a transverse
        # pressure) returns SF ~ 1e-3 N/m and SE ~ 1e-13 against
        # M ~ 1e3 N: the N block is noise on both sides of the identity
        # and can never reach 1e-4 of its own scale, however the SE
        # columns are packed.  That is not "report and law do not
        # belong together" -- the M rows still decide perm/fk exactly.
        # So: if SOME packing satisfies the M rows, take the M-row
        # winner, assign the SE slots BY COLUMN NAME (SE11, SE22, SE12,
        # the order the SK measurement already fixed for curvatures),
        # mirror the shear factor, and hand back a flag the caller must
        # print.  eps is ~0 on such a station, so the convention can
        # not bias anything downstream -- but it is a convention, and
        # the station print says so.
        byM = sorted(scored, key=lambda r: r[2])
        if byM[0][2] <= 1e-4:
            _, rN, rM, perm, _, _, fk = byM[0]
            names = [ec[i].replace("SE.", "") for i in i_se]
            want = ("SE11", "SE22", "SE12")
            if not all(w in names for w in want):
                raise SystemExit("membrane block is degenerate (max|N|"
                                 " = %.2e) and the SE columns %s do not"
                                 " carry the names needed to assign"
                                 " slots by name" % (sN, names))
            tri = tuple(names.index(w) for w in want)
            fs, score, degenerate_N = fk, rM, True
            runner = None
        else:
            raise SystemExit("no SE/SM packing satisfies {N,M} = ABD{e,k}"
                             " (best: N rows %.2e, M rows %.2e of their"
                             " own scales) -- report and law do not"
                             " belong together" % (rN, rM))
    return {"perm": perm, "tri": tri, "fs": fs, "fk": fk,
            "residual": score, "res_N": rN, "res_M": rM,
            "margin": (None if not runner else score / max(runner,
                                                           1e-300)),
            "trace_free": trace_free, "degenerate_N": degenerate_N,
            "sN": sN, "sM": sM,
            "i_se": i_se, "i_sk": i_sk, "jN": jN, "jM": jM}


# --------------------------------------------------------- the station
_W1 = {-2: 1.0, -1: -8.0, 0: 0.0, 1: 8.0, 2: -1.0}       # /12h
_W2 = {-2: -1.0, -1: 16.0, 0: -30.0, 1: 16.0, 2: -1.0}   # /12h^2


def station_state(rpt_path, cellmap_path, law_path, x=0.0, y=0.0,
                  di=0, dj=0):
    """The level-1 macro state of the cell containing (x, y), or of the
    cell OFFSET (di, dj) from it.

    di/dj are SIGNED CELL OFFSETS from the cell at (x, y): di = +1 is
    the next cell along +x (row-wise), dj along +y (column-wise);
    di = -2 is two cells toward -x.  So (0, 0) with the default x = y
    = 0 is the plate-centre cell, and (1, 2) is one cell right, two up.

    STENCIL FALLBACK.  Derivatives use the 5-point O(h^4) rules when
    the station has 2 cells of margin on every side, else the 3-point
    O(h^2) rules with 1 cell of margin (printed); a station on the edge
    ring is refused.  On an AR5 (5x5) plate only the centre cell gets
    the 5-point set -- an off-centre station silently costing accuracy
    is exactly the kind of thing this print exists to surface.

    In:  rpt_path, cellmap_path, law_path str; x, y float; di, dj int
    Out: dict {EPS (6,), FF (6,), u (3,), theta (3,), u_node (3,),
         theta_node (3,), derivs {dE1 dE2 dE11 dE12 dE22}, h float,
         cell (Ic, Jc), stencil_order 5|3, packing dict}."""
    T = read_plate_rpt(rpt_path)
    ec, er = T["elem"]
    cells = read_cellmap(cellmap_path)
    ABD = read_abd(law_path)
    sk = measure_sk_order(T, cells)
    if sk is None:
        print("SK order: NOT measurable from this report (no structured"
              " rotation grid) -- file order assumed")
    else:
        print("SK order (nodal rotations, %d asymmetric cells):"
              " (k11,k22) = SK cols %s; err %.1f%% vs %.1f%% swapped"
              % (sk["n"], sk["order"], 100 * sk["err"][0],
                 100 * sk["err"][1]))
    pk = measure_packing(ec, er, ABD, sk_order=(sk and sk["order"]))
    # SAY which membrane convention was chosen and how well the data
    # decided it -- the 2026-09-03 lesson: this choice used to be made
    # silently, by noise, and the wrong one is invisible downstream
    # until a station with real membrane action shows up.
    se_names = [ec[pk["i_se"][t]].replace("SE.", "") for t in pk["tri"]]
    if pk.get("degenerate_N"):
        print("SE membrane slots: MEMBRANE BLOCK DEGENERATE -- max|N| ="
              " %.2e vs max|M| = %.2e; the N rows cannot discriminate"
              " any packing (this deck has no in-plane forcing).  Slots"
              " (e11, e22, g12) = (%s, %s, %s x %g) taken BY COLUMN NAME"
              " -- a convention, not a measurement; eps is ~0 here so"
              " it cannot bias the station.  M rows residual %.2e."
              % (pk["sN"], pk["sM"], se_names[0], se_names[1],
                 se_names[2], pk["fs"], pk["res_M"]))
    else:
        print("SE membrane slots: (e11, e22, g12) = (%s, %s, %s x %g);"
              " residual N rows %.2e, M rows %.2e (own scales)"
              % (se_names[0], se_names[1], se_names[2], pk["fs"],
                 pk["res_N"], pk["res_M"]))
    if pk["trace_free"] is not None:
        tf = [ec[pk["i_se"][t]].replace("SE.", "")
              for t in pk["trace_free"]]
        bad = [n for n in se_names[:2] if n == tf[2] or (
            n in tf and tf.index(n) == 2)]
        print("  SE %s sum to zero elementwise = Abaqus's (e11, e22,"
              " e33) with nu_eff = 0.5%s"
              % ("/".join(tf), "" if not bad else
                 "  <- and one of them is in a MEMBRANE slot"))
    if pk["margin"] is not None and pk["margin"] > 0.1:
        print("  WARNING: the runner-up packing scores within %.1fx --"
              " this station cannot discriminate the membrane slots"
              " (too little membrane action); the choice is a"
              " CONVENTION here, not a measurement" % (1 / pk["margin"]))
    tri, fs, fk = pk["tri"], pk["fs"], pk["fk"]
    i_se, i_sk = pk["i_se"], pk["i_sk"]
    jN, jM, perm = pk["jN"], pk["jM"], pk["perm"]

    def sub_state(v):
        return np.array([v[i_se[tri[0]]], v[i_se[tri[1]]],
                         fs * v[i_se[tri[2]]],
                         v[i_sk[0]], v[i_sk[1]], fk * v[i_sk[2]]])

    def cell_state(key):
        subs = cells[key]["subs"]
        return np.mean([sub_state(er[e]) for e in subs], axis=0)

    def cell_ff(key):
        subs = cells[key]["subs"]
        return np.mean([[er[e][j] for j in jN]
                        + [er[e][jM[perm[0]]], er[e][jM[perm[1]]],
                           er[e][jM[perm[2]]]] for e in subs], axis=0)


    NI = 1 + max(k[0] for k in cells)
    NJ = 1 + max(k[1] for k in cells)
    st0 = min(cells, key=lambda k: math.hypot(cells[k]["xc"] - x,
                                              cells[k]["yc"] - y))
    st = (st0[0] + int(di), st0[1] + int(dj))
    if st not in cells:
        raise SystemExit("offset (%+d, %+d) from cell %s leaves the"
                         " %d x %d grid" % (di, dj, st0, NI, NJ))
    i0, j0 = st
    # derivative source ladder.  Cell-to-cell stencils give the higher
    # orders but need neighbours; the IN-CELL cross-difference of the
    # subdiv sub-element values needs NONE -- it works in any cell,
    # edge ring included -- and supplies dE1/dE2/dE12 (pure second
    # derivatives need 3 points per direction, which 2x2 cannot give;
    # tau-auto drops them anyway).  The in-cell gradient carries the
    # plate FE solution's own O(h) error -- read edge-adjacent stations
    # with that in mind.
    if 2 <= i0 <= NI - 3 and 2 <= j0 <= NJ - 3:
        order = 5
    elif 1 <= i0 <= NI - 2 and 1 <= j0 <= NJ - 2:
        order = 3
        print("station cell (%d, %d): no 5-point margin -- 3-point"
              " O(h^2) cell stencil" % (i0, j0))
    else:
        order = 0
        print("station cell (%d, %d): edge ring -- IN-CELL"
              " sub-element cross differences (dE1/dE2/dE12 only;"
              " d2eps_11/22 unavailable and written as zero)"
              % (i0, j0))
    xs = sorted(set(c["xc"] for c in cells.values()))
    h = xs[1] - xs[0]

    if order:
        m = 2 if order == 5 else 1
        V = {(a, b): cell_state((i0 + a, j0 + b))
             for a in range(-m, m + 1) for b in range(-m, m + 1)}
        if order == 5:
            w1, w2, s1, s2 = _W1, _W2, 12.0 * h, 12.0 * h * h
        else:
            w1 = {-1: -1.0, 0: 0.0, 1: 1.0}
            w2 = {-1: 1.0, 0: -2.0, 1: 1.0}
            s1, s2 = 2.0 * h, h * h

        def stencil(w, axis, scale):
            out = np.zeros(6)
            for o, c in w.items():
                out += c * (V[(o, 0)] if axis == 0 else V[(0, o)])
            return out / scale
        dE1 = stencil(w1, 0, s1)
        dE2 = stencil(w1, 1, s1)
        dE11 = stencil(w2, 0, s2)
        dE22 = stencil(w2, 1, s2)
        dE12 = np.zeros(6)
        for a, ca in w1.items():
            for b, cb in w1.items():
                if ca and cb:
                    dE12 += ca * cb * V[(a, b)] / (s1 * s1)
        EPS0 = V[(0, 0)]
    else:
        # IN-CELL: the k x k sub-element grid of THIS cell alone
        # (k = subdiv), geometry recovered from the element table's own
        # centroids, never from the cellmap's writing order.
        #   k = 2 : dE1/dE2 (cross differences) + dE12; the pure second
        #           derivatives need 3 points per direction and are
        #           written as ZERO with a printed note
        #   k >= 3 (odd) : ALL five derivatives from central
        #           differences on the in-cell grid, O(h_sub^2) --
        #           subdiv = 3 is the general-case setting
        subs = cells[st]["subs"]
        k = int(round(math.sqrt(len(subs))))
        pts = sorted((er[e][0], er[e][1], e) for e in subs)
        xs_c = sorted(set(round(p[0], 9) for p in pts))
        ys_c = sorted(set(round(p[1], 9) for p in pts))
        hs = xs_c[1] - xs_c[0]
        G = {}
        for xe, ye, e in pts:
            G[(xs_c.index(round(xe, 9)),
               ys_c.index(round(ye, 9)))] = sub_state(er[e])
        if k >= 3 and k % 2 == 1:
            m = k // 2
            dE1 = (G[(m + 1, m)] - G[(m - 1, m)]) / (2 * hs)
            dE2 = (G[(m, m + 1)] - G[(m, m - 1)]) / (2 * hs)
            dE11 = (G[(m + 1, m)] - 2 * G[(m, m)]
                    + G[(m - 1, m)]) / (hs * hs)
            dE22 = (G[(m, m + 1)] - 2 * G[(m, m)]
                    + G[(m, m - 1)]) / (hs * hs)
            dE12 = (G[(m + 1, m + 1)] - G[(m + 1, m - 1)]
                    - G[(m - 1, m + 1)]
                    + G[(m - 1, m - 1)]) / (4 * hs * hs)
        else:
            dE1 = (G[(1, 1)] + G[(1, 0)] - G[(0, 1)] - G[(0, 0)]) \
                / (2.0 * hs)
            dE2 = (G[(1, 1)] + G[(0, 1)] - G[(1, 0)] - G[(0, 0)]) \
                / (2.0 * hs)
            dE12 = (G[(1, 1)] - G[(1, 0)] - G[(0, 1)] + G[(0, 0)]) \
                / (hs * hs)
            dE11 = np.zeros(6)
            dE22 = np.zeros(6)
            print("in-cell subdiv = 2: pure second derivatives"
                  " unavailable, written as zero -- use subdiv = 3")
        EPS0 = cell_state(st)

    # LEVEL-1 u/theta: the cell average of the nodal U/UR over every
    # node inside the cell footprint -- the cell value, consistent with
    # the strain being a cell mean.  The centre-node value is kept too:
    # the two differ by O(h^2) curvature, and printing both makes the
    # difference visible instead of assumed.
    u = th = u_node = th_node = np.zeros(3)
    if "node" in T:
        nc, nr = T["node"]

        def ncol(nm):
            j = [i for i, c in enumerate(nc) if c.endswith("." + nm)]
            return j[0] if j else None
        ju = [ncol(n) for n in ("U1", "U2", "U3")]
        jt = [ncol(n) for n in ("UR1", "UR2", "UR3")]
        xc, yc = cells[st]["xc"], cells[st]["yc"]
        # RELATIVE to the cell size (2026-09-09).  An absolute 1e-9 in
        # the deck's length unit is finer than the coordinates the deck
        # writes (%.6f), so whether a node sitting exactly on a cell
        # boundary is counted depended on the model's absolute size: the
        # mm-size panel dropped those nodes while the geometrically
        # similar panel 100x larger kept them, and the station's rigid
        # motion u, theta then differed between two exactly similar
        # problems.  1e-6 * h is far below the node spacing (h/3 for
        # three elements per cell) and far above the printing round-off.
        tol = 1e-6 * h
        inside = [v for v in nr.values()
                  if abs(v[0] - xc) <= h / 2 + tol
                  and abs(v[1] - yc) <= h / 2 + tol]
        if inside and None not in ju:
            u = np.mean([[v[j] for j in ju] for v in inside], axis=0)
        if inside and None not in jt:
            th = np.mean([[v[j] for j in jt] for v in inside], axis=0)
        near = min(nr.values(),
                   key=lambda v: math.hypot(v[0] - xc, v[1] - yc))
        if None not in ju:
            u_node = np.array([near[j] for j in ju])
        if None not in jt:
            th_node = np.array([near[j] for j in jt])

    return {"EPS": EPS0, "FF": cell_ff(st), "u": u, "theta": th,
            "u_node": u_node, "theta_node": th_node,
            "derivs": {"dE1": dE1, "dE2": dE2, "dE11": dE11,
                       "dE12": dE12, "dE22": dE22},
            "h": h, "cell": st, "stencil_order": order, "packing": pk,
            "n_nodes_in_cell": len(inside) if "node" in T else 0,
            "xc": cells[st]["xc"], "yc": cells[st]["yc"]}


def write_station_ff(out, state, q0=None, qb0=None):
    """build_ff on a station_state dict.

    In:  out str; state dict from station_state; q0 float | None --
         TOP-face pressure (uniform -> qt6 = [q0, 0...]); qb0 likewise
         for the bottom face
    Out: the path written."""
    from opensg_solid.sg_plate_derivatives import build_ff
    return build_ff(
        out, FF=state["FF"], EPS=state["EPS"],
        derivs=state["derivs"], u=state["u"], theta=state["theta"],
        qt6=(None if q0 is None else [q0, 0, 0, 0, 0, 0]),
        qb6=(None if qb0 is None else [qb0, 0, 0, 0, 0, 0]))
