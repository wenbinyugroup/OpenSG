"""step07_elem_grid_derivs.py -- strain and ALL first/second strain derivatives at a
station, by fixed-weight central differences on the ELEMENT grid.

    python -m opensg_solid.helper.Plate_deck_analysis.step07_elem_grid_derivs \
           <plate.rpt> --law <law.out> --at XC YC

WHY THE ELEMENT GRID.  sg_plate_station differences on the CELL grid, so on a 5 x 5
plate only the centre cell gets the 5-point rule, the next ring falls back to 3-point
at h = one cell, and the edge ring to the in-cell route.  The plate report is
centroidal PER SUB-ELEMENT: with subdiv = 3 that is 3 x NX points per direction at
h = pitch/3 -- the same smooth macro field sampled three times finer.  Differencing
THERE gives every cell a real stencil:

    5-point (default, O(h^4))   d1 [ 1, -8,  0,  8, -1]/(12 h)
                                d2 [-1, 16, -30, 16, -1]/(12 h^2)
    3-point (edge fallback)     d1 [-1, 0, 1]/(2 h)     d2 [1, -2, 1]/h^2
    mixed d12                   the product of the two d1 stencils

Validated on the TPMS pressure_case (2026-09-03): at the plate centre the element-grid
values reproduce the cell-grid 5-point ones to 1.7 %; subdiv 3 vs 5 moves them < 2 %.

CONVENTIONS ARE MEASURED, NOT ASSUMED.  The SE/SK column mapping comes from
plate_rm's measured packing (measure_sk_order + measure_packing) -- never from the
report's own labels, which are known to lie (the SK swap of 2026-08-31).

In:  the plate .rpt (level-2 sub-element table) and the measured packing
Out: element_grid() -> the structured strain grid; derivs_at() -> the five
     derivative families at any interior station"""
import argparse
import math
import sys

import numpy as np

W1_5 = {-2: 1 / 12., -1: -8 / 12., 0: 0.0, 1: 8 / 12., 2: -1 / 12.}
W2_5 = {-2: -1 / 12., -1: 16 / 12., 0: -30 / 12., 1: 16 / 12., 2: -1 / 12.}
W1_3 = {-1: -0.5, 0: 0.0, 1: 0.5}
W2_3 = {-1: 1.0, 0: -2.0, 1: 1.0}
I0 = {0: 1.0}


def element_grid(er, packing):
    """The structured (nx, ny, 6) plate-strain grid of the sub-element table.

    In:  er dict {label: row list} -- the element table of
         plate_rm.read_plate_rpt (row = [Xc, Yc, ...fields]);
         packing dict -- measure_packing output (i_se, i_sk, tri, fs, fk),
         SK order already folded in
    Out: (G (nx, ny, 6) [e11 e22 2e12 k11 k22 2k12], xs, ys, h)."""
    i_se, i_sk = packing["i_se"], packing["i_sk"]
    tri, fs, fk = packing["tri"], packing["fs"], packing["fk"]
    lab = sorted(er)
    X = np.array([er[e][0] for e in lab])
    Y = np.array([er[e][1] for e in lab])
    xs = np.unique(np.round(X, 6))
    ys = np.unique(np.round(Y, 6))
    hx, hy = xs[1] - xs[0], ys[1] - ys[0]
    if abs(hx - hy) > 1e-9 * max(hx, hy):
        raise SystemExit("element grid pitch differs by direction"
                         " (%.6g vs %.6g)" % (hx, hy))
    G = np.full((len(xs), len(ys), 6), np.nan)
    for e in lab:
        v = er[e]
        st = [v[i_se[tri[0]]], v[i_se[tri[1]]], fs * v[i_se[tri[2]]],
              v[i_sk[0]], v[i_sk[1]], fk * v[i_sk[2]]]
        G[int(np.argmin(np.abs(xs - v[0]))),
          int(np.argmin(np.abs(ys - v[1])))] = st
    if np.isnan(G).any():
        raise SystemExit("element grid is not complete/rectangular")
    return G, xs, ys, hx


def _apply(G, ic, jc, h, wx, wy, order):
    out = np.zeros(6)
    for a, ca in wx.items():
        for b, cb in wy.items():
            if ca and cb:
                out += ca * cb * G[ic + a, jc + b]
    return out / (h ** order)


def derivs_at(G, xs, ys, h, x, y):
    """The five derivative families at the interior station nearest (x, y).

    5-point per direction wherever 2 points of margin exist, else 3-point
    (printed); a station on the grid boundary is refused -- there is no
    central stencil there.

    In:  G, xs, ys, h from element_grid(); x, y float
    Out: dict {dE1, dE2, dE11, dE12, dE22 (6,), stencil (sx, sy) in {3, 5},
         index (ic, jc)}."""
    ic = int(np.argmin(np.abs(xs - x)))
    jc = int(np.argmin(np.abs(ys - y)))
    mx = min(ic, len(xs) - 1 - ic)
    my = min(jc, len(ys) - 1 - jc)
    if mx < 1 or my < 1:
        raise SystemExit("station (%g, %g) is on the element-grid boundary"
                         " -- no central stencil" % (x, y))
    w1x, w2x, sx = (W1_5, W2_5, 5) if mx >= 2 else (W1_3, W2_3, 3)
    w1y, w2y, sy = (W1_5, W2_5, 5) if my >= 2 else (W1_3, W2_3, 3)
    if sx == 3 or sy == 3:
        print("station (%.4g, %.4g): 3-point fallback in %s (margin %d/%d"
              " elements)" % (x, y,
                              "x" if sx == 3 else "y", mx, my))
    return {"dE1": _apply(G, ic, jc, h, w1x, I0, 1),
            "dE2": _apply(G, ic, jc, h, I0, w1y, 1),
            "dE11": _apply(G, ic, jc, h, w2x, I0, 2),
            "dE12": _apply(G, ic, jc, h, w1x, w1y, 2),
            "dE22": _apply(G, ic, jc, h, I0, w2y, 2),
            "stencil": (sx, sy), "index": (ic, jc)}


def station_derivs(rpt_path, law_path, x, y, cellmap_path=None):
    """One-call convenience: measured packing + element grid + derivatives.

    In:  rpt_path, law_path str; x, y float; cellmap_path str | None
         (needed only for the SK measurement's cell structure)
    Out: (derivs dict, packing dict)."""
    from opensg_solid.helper.plate_rm.sg_plate_station import (
        read_plate_rpt, read_cellmap, read_abd, measure_sk_order,
        measure_packing)
    T = read_plate_rpt(rpt_path)
    ec, er = T["elem"]
    ABD = read_abd(law_path)
    sk = None
    if cellmap_path:
        sk = measure_sk_order(T, read_cellmap(cellmap_path))
    pk = measure_packing(ec, er, ABD, sk_order=(sk and sk["order"]))
    G, xs, ys, h = element_grid(er, pk)
    return derivs_at(G, xs, ys, h, x, y), pk


def main(argv=None):
    """CLI: print the families at one station.

    In:  argv list | None
    Out: int exit status."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("rpt")
    p.add_argument("--law", required=True)
    p.add_argument("--at", nargs=2, type=float, required=True)
    p.add_argument("--cellmap", default=None)
    a = p.parse_args(argv)
    d, _ = station_derivs(a.rpt, a.law, a.at[0], a.at[1], a.cellmap)
    print("stencil: %d-point in x, %d-point in y" % d["stencil"])
    for k in ("dE1", "dE2", "dE11", "dE12", "dE22"):
        print("%-6s %s" % (k, " ".join("%+.6e" % v for v in d[k])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
