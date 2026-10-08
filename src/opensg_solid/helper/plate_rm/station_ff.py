"""station_ff.py -- build the dehom .ff of ANY plate cell, chosen by
its SIGNED CELL OFFSET from the plate centre.

    python -m opensg_solid.helper.plate_rm.station_ff <rpt> --law <out> \
           --cell 1 1 [--q0 7946.1] [--cellmap <csv>] [--out <ff>]

--cell di dj:  di = +1 is one cell along +x (row-wise), dj along +y
(column-wise); negatives go toward -x/-y.  (0, 0) is the plate-centre
cell.  The cell's OWN footprint is the SG for that station -- its
centre (xc, yc) is printed, and the 3-D FEA comparison must extract the
cell at the same offset and shift the paths by (xc, yc).

Grid-margin honesty: derivatives fall back 5-point -> 3-point as the
margin shrinks (printed), and the edge ring is refused outright.  On an
AR5 (5 x 5) plate that means: centre cell = 5-point; the 8 cells around
it = 3-point; the outer 16 = refused.

In:  the combined plate report + cellmap + the 8x8 .out
Out: the .ff (path printed), with 0:/1:, u/theta (cell averages),
     derivatives and qt6."""
import argparse
import os

from opensg_solid.helper.plate_rm.sg_plate_station import station_state, write_station_ff


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("rpt")
    p.add_argument("--law", required=True)
    p.add_argument("--cell", nargs=2, type=int, default=[0, 0],
                   metavar=("DI", "DJ"))
    p.add_argument("--q0", type=float, default=None)
    p.add_argument("--cellmap", default=None)
    p.add_argument("--out", default=None)
    A = p.parse_args()

    cmap = A.cellmap
    if cmap is None:
        sib = A.rpt.replace("_plate.rpt", "_cellmap.csv")
        par = os.path.join(os.path.dirname(os.path.abspath(A.rpt)),
                           "..", os.path.basename(sib))
        cmap = sib if os.path.exists(sib) else os.path.normpath(par)

    st = station_state(A.rpt, cmap, A.law, di=A.cell[0], dj=A.cell[1])
    pk = st["packing"]
    print("station cell %s at (%+.4f, %+.4f); offset (%+d, %+d);"
          " stencil order %d"
          % (st["cell"], st["xc"], st["yc"], A.cell[0], A.cell[1],
             st["stencil_order"]))
    print("packing (M11,M22,M12) = (%s); residual %.2e"
          % (", ".join("SM%d" % (i + 1) for i in pk["perm"]),
             pk["residual"]))
    print("  0: EPS = %s" % ["%.6e" % v for v in st["EPS"]])
    print("  u (cell avg) = %s" % ["%.6e" % v for v in st["u"]])

    out = A.out or (os.path.splitext(A.rpt)[0]
                    + "_cell%+d%+d.ff" % (A.cell[0], A.cell[1]))
    write_station_ff(out, st, q0=A.q0)
    print("wrote %s" % out)
    print("FEA pairing: extract the 3-D cell centred at (%+.4f, %+.4f)"
          " and shift the paths by the same offset"
          % (st["xc"], st["yc"]))

    # DEFAULT OUTPUT: the station location on the level-1 plate grid,
    # so the chosen cell is visible rather than inferred from indices.
    # Level-1 cells solid, level-2 sub-element lines dashed, the
    # station cell filled; no figure title (caption convention).
    from opensg_solid.helper.plate_rm.sg_plate_station import read_cellmap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cells = read_cellmap(cmap)
    h = st["h"]
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    # the ACTUAL Abaqus plate mesh: every sub-element (from the element
    # table's centroids, k x k per cell) drawn thin, level-1 cell
    # borders heavier, the station cell's own sub-elements filled
    from opensg_solid.helper.plate_rm.sg_plate_station import \
        read_plate_rpt
    ec_, er_ = read_plate_rpt(A.rpt)["elem"]
    k = int(round(len(next(iter(cells.values()))["subs"]) ** 0.5))
    hs = h / k
    for v in er_.values():
        ax.add_patch(plt.Rectangle((v[0] - hs / 2, v[1] - hs / 2),
                                   hs, hs, fill=False, ec="0.75",
                                   lw=0.4))
    for (ic, jc), c in cells.items():
        ax.add_patch(plt.Rectangle((c["xc"] - h / 2, c["yc"] - h / 2),
                                   h, h, fill=False, ec="0.30",
                                   lw=1.0))
    for e in cells[st["cell"]]["subs"]:
        v = er_[e]
        ax.add_patch(plt.Rectangle((v[0] - hs / 2, v[1] - hs / 2),
                                   hs, hs, fc="#d1495b", alpha=0.45,
                                   ec="k", lw=0.8))
    ax.plot([0], [0], "+", color="k", ms=10)
    ax.annotate("plate centre", (0, 0), textcoords="offset points",
                xytext=(6, 6), fontsize=8)
    ax.annotate("station (%+d, %+d)" % (A.cell[0], A.cell[1]),
                (st["xc"], st["yc"]), ha="center", fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$  [m]")
    ax.set_ylabel(r"$x_2$  [m]")
    lim = h * (0.5 + max(1 + max(k[0] for k in cells) ,
                         1 + max(k[1] for k in cells)) / 2.0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    png = os.path.splitext(out)[0] + "_location.png"
    fig.tight_layout()
    fig.savefig(png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("wrote %s" % png)


if __name__ == "__main__":
    main()
