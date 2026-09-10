"""step02_levels.py -- the level-1 / level-2 contract of the plate deck.

    python -m opensg_solid.helper.Plate_deck_analysis.step02_levels \
           <cellmap.csv> [--rpt <plate.rpt>]

THE CONTRACT.  LEVEL 1 = one SG cell = one station: the unit
`opensg_solid` homogenizes and dehomogenizes -- the station .ff, the
dehom run and the 3-D FEA comparison are all cut per level-1 cell, and
the cell's own footprint is the SG for that station.  LEVEL 2 = the
subdiv^2 sub-elements that mesh that one cell in the Abaqus plate deck:
what Abaqus actually solves, and the grid the step07 derivative
stencils difference on (pitch = cell pitch / subdiv).

WHY A GATE.  The cellmap csv written by step01 is the ONLY bridge
between the two levels; every downstream step (06, 07, 08, 10, 12)
trusts it blindly.  So the map is verified once, here: a complete
NX x NY level-1 grid, the same subdiv^2 sub-element count in every
cell, disjoint sub-element labels (the level-2 mesh is a partition of
the level-1 grid), and -- when the deck report exists -- a level-2
element table of exactly NX * NY * subdiv^2 rows.

In:  the ..._cellmap.csv of step01, optionally the combined plate .rpt
Out: verify_levels() -> {NX, NY, subdiv, ok}; one printed line per
     check"""
import argparse
import itertools
import math
import sys


def read_cellmap(path):
    """plate_rm's two-level cellmap reader, REUSED (a lazy delegate,
    not a duplicate -- the package rule is no jax at import time, and
    importing opensg_solid.* pulls the jax setup of its __init__).

    In:  path str -- ..._cellmap.csv
    Out: {(Ic, Jc): {"xc", "yc", "subs": [element labels]}}"""
    from opensg_solid.helper.plate_rm.sg_plate_station import \
        read_cellmap as _read_cellmap
    return _read_cellmap(path)


def verify_levels(cellmap_path, rpt_path=None):
    """The integrity gate on the level-1 / level-2 map.

    Prints one line per check; a failed check flips ok and says what
    broke, it never raises -- the caller decides whether to stop.

    In:  cellmap_path str; rpt_path str | None -- the combined plate
         .rpt (checks the level-2 element-table row count when given)
    Out: dict {NX, NY, subdiv, ok} -- subdiv = -1 when the sub count
         is not a square."""
    cells = read_cellmap(cellmap_path)
    if not cells:
        raise SystemExit("empty cellmap: %s" % cellmap_path)
    ok = True

    ivals = sorted({i for i, _ in cells})
    jvals = sorted({j for _, j in cells})
    NX = ivals[-1] - ivals[0] + 1
    NY = jvals[-1] - jvals[0] + 1
    full = [(i, j)
            for i in range(ivals[0], ivals[-1] + 1)
            for j in range(jvals[0], jvals[-1] + 1)]
    if len(cells) == NX * NY and all(k in cells for k in full):
        print("level 1: %d x %d cells, grid complete" % (NX, NY))
    else:
        ok = False
        print("level 1: %d x %d grid INCOMPLETE (%d/%d cells)"
              % (NX, NY, len(cells), NX * NY))

    counts = sorted({len(c["subs"]) for c in cells.values()})
    n = counts[0]
    k = math.isqrt(n)
    subdiv = k if k * k == n else -1
    if len(counts) != 1:
        ok = False
        print("level 2: sub-element count NOT uniform across cells: %s"
              % counts)
    elif subdiv < 0:
        ok = False
        print("level 2: %d sub-elements per cell is not a subdiv^2"
              " square" % n)
    else:
        print("level 2: subdiv = %d -- %d sub-elements in every cell"
              % (subdiv, n))

    labs = list(itertools.chain.from_iterable(
        c["subs"] for c in cells.values()))
    if len(set(labs)) == len(labs):
        print("level 2: %d sub-element labels, all disjoint"
              % len(labs))
    else:
        ok = False
        print("level 2: %d sub-element labels, only %d distinct --"
              " cells share elements" % (len(labs), len(set(labs))))

    if rpt_path is not None:
        from opensg_solid.helper.plate_rm.sg_plate_station import \
            read_plate_rpt
        tabs = read_plate_rpt(rpt_path)
        if "elem" not in tabs:
            ok = False
            print("rpt: no element table found in %s" % rpt_path)
        else:
            rows = len(tabs["elem"][1])
            want = NX * NY * (subdiv * subdiv if subdiv > 0 else 0)
            if rows == want:
                print("rpt: %d element rows = NX*NY*subdiv^2" % rows)
            else:
                ok = False
                print("rpt: %d element rows != %d expected"
                      " (NX*NY*subdiv^2)" % (rows, want))

    return {"NX": NX, "NY": NY, "subdiv": subdiv, "ok": ok}


def main(argv=None):
    """CLI: run the gate; exit status carries the verdict.

    In:  argv list | None
    Out: int exit status (0 = every check passed)."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("cellmap")
    p.add_argument("--rpt", default=None,
                   help="combined plate .rpt -- adds the level-2 row"
                        " count check")
    a = p.parse_args(argv)
    return 0 if verify_levels(a.cellmap, a.rpt)["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
