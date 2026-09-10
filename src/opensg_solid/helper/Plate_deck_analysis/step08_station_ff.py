"""step08_station_ff.py -- the station .ff of ANY plate cell, with the strain
derivatives taken on the ELEMENT grid (step07's 5-point default).

    python -m opensg_solid.helper.Plate_deck_analysis.step08_station_ff \
           <plate.rpt> --law <law.out> --cell DI DJ --cellmap <csv> \
           [--q0 7946.1] [--out <station.ff>]

What it reuses and what it replaces:

    REUSED    plate_rm.sg_plate_station.station_state -- the level-1 cell state
              (EPS, FF, u, theta) with every SE/SK/SM convention MEASURED, and
              write_station_ff for the file itself
    REPLACED  the derivative families only: the cell-grid stencils (5-point at
              the centre, 3-point one ring out, in-cell at the edge) give way to
              step07's element-grid central differences, so every station gets a
              5-point rule wherever two sub-elements of margin exist

The gate that keeps the two routes honest is printed every run: at any station the
element-grid second derivatives must agree with the cell-grid ones at the level the
two resolutions imply (1.7 % at the TPMS plate centre).  A large disagreement means
a convention drifted, and the run should not be trusted.

--cell DI DJ are the SIGNED CELL OFFSETS from the plate-centre cell, the same
convention as plate_rm.station_ff and fea_cell ((0,0) = centre, (1,2) = one cell
along +x, two along +y).

In:  the level-2 plate .rpt, the cellmap, the 8x8 law, the cell offset
Out: the .ff (path printed), stencil + gate lines"""
import argparse
import math
import sys

import numpy as np


def build_station_ff(rpt_path, cellmap_path, law_path, di, dj,
                     q0=None, out=None):
    """The station .ff with element-grid derivatives.

    In:  rpt_path, cellmap_path, law_path str; di, dj int signed cell
         offsets; q0 float | None -- TOP-face pressure; out str | None
    Out: (path written, state dict, derivs dict)."""
    from opensg_solid.helper.plate_rm.sg_plate_station import (
        read_cellmap, station_state, write_station_ff)
    from opensg_solid.helper.Plate_deck_analysis.step07_elem_grid_derivs \
        import station_derivs

    state = station_state(rpt_path, cellmap_path, law_path,
                          di=int(di), dj=int(dj))
    cells = read_cellmap(cellmap_path)
    xc = cells[state["cell"]]["xc"]
    yc = cells[state["cell"]]["yc"]
    d, _ = station_derivs(rpt_path, law_path, xc, yc, cellmap_path)
    print("element-grid stencil: %d-point in x, %d-point in y at"
          " (%+.4f, %+.4f)" % (d["stencil"][0], d["stencil"][1], xc, yc))

    old = state["derivs"]
    for k in ("dE11", "dE12", "dE22"):
        o, n = np.asarray(old.get(k, np.zeros(6)))[3:], d[k][3:]
        ref = float(np.abs(o).max())
        if ref > 1e-12:
            print("  gate %s: cell-grid vs element-grid curvature slots"
                  " differ %.1f %%" % (k, 100 * float(np.abs(n - o).max())
                                       / ref))
        else:
            print("  gate %s: cell grid had no value here (order-%d"
                  " station); element grid supplies it"
                  % (k, state.get("stencil_order", -1)))
    state["derivs"] = {k: d[k] for k in ("dE1", "dE2", "dE11", "dE12",
                                         "dE22")}
    if out is None:
        tag = "i%d%sj%d%s" % (abs(di), "N" if di < 0 else "",
                              abs(dj), "N" if dj < 0 else "")
        out = "station_%s.ff" % tag
    write_station_ff(out, state, q0=q0)
    print("wrote %s" % out)
    return out, state, d


def main(argv=None):
    """Parse the CLI and build one station file.

    In:  argv list | None
    Out: int exit status."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("rpt")
    p.add_argument("--law", required=True)
    p.add_argument("--cell", nargs=2, type=int, required=True,
                   help="signed cell offset DI DJ from the plate centre")
    p.add_argument("--cellmap", required=True)
    p.add_argument("--q0", type=float, default=None,
                   help="TOP-face pressure [Pa] -> qt6 = [q0, 0, ...]")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    build_station_ff(a.rpt, a.cellmap, a.law, a.cell[0], a.cell[1],
                     q0=a.q0, out=a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
