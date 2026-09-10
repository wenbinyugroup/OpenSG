"""step13_run_pipeline.py -- the plate-benchmark orchestrator: steps 1-12
as resumable stages under one output root.

    python -m opensg_solid.helper.Plate_deck_analysis.step13_run_pipeline \\
           --sg <sg.yaml> --law <law.out> --out-root <dir> --q 7946.1 \\
           --paths <p1.dat> [<p2.dat> ...] [--cells "0 0,1 1"] \\
           [--nx 5 --ny 5 --pitch 1.0] [--stages 1-12] \\
           [--run-abaqus plate|solid|none] [--solid-odb <stem>] [--force]

Stage map (pipeline_state.json records what finished; --force reruns):

    1  decks        step01_make_decks: the BARE plate deck + cellmap
    2  verify       step02_levels: level-1/level-2 cellmap integrity
    3  withF        make_load_column_deck: F from Yu 2003 Eq. 47
                    (F = V0^T L, no dehom, no integration) -> the withF
                    deck = bare deck + one *Cload block (edge traction
                    -F_N on the free normal DOF).  --fn <F_N> overrides
                    with the legacy step03 inject.
    4  solid deck   step04_solid_inp: the 3-D reference deck (written
                    only; the solve is the user's, house rule)
    5  rpts         Abaqus solves + step05 dumps.  BOTH plate jobs run
                    (bare AND withF, seconds each) when --run-abaqus is
                    plate|solid.  The 3-D side is PER CELL: rpts/
                    fea_cell_<tag>.rpt, taken as already present (copy
                    them in, they come from the 3-D odb), or dumped from
                    --solid-odb with --run-abaqus solid, else the
                    pipeline prints the commands and stops.
    6  fea dats     step06 per (cell, path) with --offset (DI, DJ)*pitch
                    -> dats/<tag>/fea_<pathstem>.dat
    7  stations     step08 TWICE per cell: <tag>_withF.ff from the withF
                    report (drives the refined recovery) and <tag>_bare.ff
                    from the bare report (drives the classical one --
                    Yu Eq. 33 carries no load term, so the classical
                    plate takes no edge load; its membrane block is
                    degenerate and step08 says so)
    8  --           folded into stage 7
    9  dehom        step09 per cell on <tag>_withF.ff -> dehom/<tag>/
    10 opensg dats  step10 per (cell, path) -> dats/<tag>/refined_path<n>.dat
    11 classical    step11 per cell on <tag>_bare.ff -> dehom_classical/
                    <tag>/ and dats/<tag>/classical_path<n>.dat
    12 plots        step12 per (cell, path) -> plots/<tag>/

WHY TWO PLATE JOBS.  The refined law is {N;M} = ABD{eps;kappa} + F and a
general section carries no F, so the refined station needs the
equivalent edge load; the classical law has no F, so the classical
station must NOT have it.  Sharing one station between the twins (the
previous design) lent the classical curve a membrane strain classical
theory cannot produce.

WHY MAIN-OR-SUBPROCESS.  Every step module exposes main(argv); the runner
calls it in-process when the package imports (one interpreter, honest
tracebacks) and falls back to `python -m` otherwise.

Fixed layout under --out-root: decks/ rpts/ stations/ dehom/<tag>/
dehom_classical/<tag>/ dats/<tag>/ plots/<tag>/ + pipeline_state.json,
<tag> = step08's cell tag (i0j0, i1Nj2, ...).

In:  the SG yaml (its folder must hold exactly one .yaml, the .out and
     the _sg.npz), the 8x8 law .out, the path .dat files (rows s y1 y2
     y3, cell-local), q, the cells
Out: every stage artefact under --out-root; terse per-stage prints"""
import argparse
import glob
import importlib
import json
import os
import shutil
import subprocess
import sys
import time

PKG = "opensg_solid.helper.Plate_deck_analysis."


def cell_tag(di, dj):
    """Folder/file tag of a signed cell offset (step08's convention).

    In:  di, dj int signed cell offsets from the plate centre
    Out: str, e.g. (0, 0) -> 'i0j0', (-1, 2) -> 'i1Nj2'."""
    return "i%d%sj%d%s" % (abs(di), "N" if di < 0 else "",
                           abs(dj), "N" if dj < 0 else "")


def run_mod(name, args):
    """Run one step module: in-process main(argv), else `python -m`.

    In:  name str module basename; args list (stringified here)
    Out: None (raises SystemExit/CalledProcessError on failure)."""
    args = [str(a) for a in args]
    print("  %s %s" % (name, " ".join(args)))
    try:
        fn = getattr(importlib.import_module(PKG + name), "main", None)
    except ImportError:
        fn = None
    if fn is not None:
        rc = fn(args)
        if rc not in (None, 0):
            raise SystemExit("%s failed (%s)" % (name, rc))
    else:
        subprocess.check_call([sys.executable, "-m", PKG + name] + args)


def sh(cmd, cwd):
    """Run one shell command (Abaqus launchers are .bat on Windows).

    In:  cmd str; cwd str working directory
    Out: None (raises CalledProcessError on failure)."""
    print("  $ %s" % cmd)
    subprocess.check_call(cmd, shell=True, cwd=cwd)


def parse_stages(spec):
    """'3-5' | '1,4,12' | 'all' -> the sorted stage list.

    In:  spec str
    Out: list of int in 1..12."""
    if spec in (None, "", "all"):
        return list(range(1, 13))
    out = set()
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    bad = sorted(k for k in out if not 1 <= k <= 12)
    if bad:
        raise SystemExit("stages out of range: %s" % bad)
    return sorted(out)


def parse_cells(spec):
    """'DI DJ[,DI DJ...]' -> [(di, dj), ...].

    In:  spec str, e.g. "0 0,1 1"
    Out: list of int 2-tuples."""
    out = []
    for part in spec.split(","):
        di, dj = part.split()
        out.append((int(di), int(dj)))
    return out


class _Ctx(object):
    """Resolved paths + options shared by the stage functions.

    In:  a -- the parsed argparse namespace
    Out: attribute bag (dirs, deck/rpt paths, cells, paths)."""

    def __init__(self, a):
        self.a = a
        r = os.path.abspath(a.out_root)
        self.root = r
        self.decks = os.path.join(r, "decks")
        self.rpts = os.path.join(r, "rpts")
        self.stations = os.path.join(r, "stations")
        self.dehom = os.path.join(r, "dehom")
        self.dehom_cls = os.path.join(r, "dehom_classical")
        self.dats = os.path.join(r, "dats")
        self.plots = os.path.join(r, "plots")
        self.src_dir = os.path.dirname(a.sg)
        self.stem = os.path.splitext(os.path.basename(a.sg))[0]
        self.withf_inp = os.path.join(self.decks, "plate_withF.inp")
        self.solid_inp = os.path.join(self.decks, "solid.inp")
        # the two plate jobs, named by what they carry
        self.plate_rpt = {"bare": os.path.join(self.rpts,
                                               "plate_bare_plate.rpt"),
                          "withF": os.path.join(self.rpts,
                                                "plate_withF_plate.rpt")}
        self.cells = parse_cells(a.cells)
        self.paths = [os.path.abspath(p) for p in a.paths]

    # step01 names its outputs after the SG stem; resolve after stage 1
    @property
    def plate_inp(self):
        """The bare plate deck step01 wrote.  Out: str path."""
        hits = [f for f in glob.glob(os.path.join(self.decks, "*.inp"))
                if not f.endswith(("plate_withF.inp", "solid.inp"))]
        if len(hits) != 1:
            raise SystemExit("expected one step01 deck in %s, found %s"
                             % (self.decks, hits))
        return hits[0]

    @property
    def cellmap(self):
        """step01's cellmap.  Out: str path."""
        hits = glob.glob(os.path.join(self.decks, "*_cellmap.csv"))
        if len(hits) != 1:
            raise SystemExit("expected one cellmap in %s, found %s"
                             % (self.decks, hits))
        return hits[0]

    def solid_rpt(self, di, dj):
        """The 3-D cell report of one cell.  In: di, dj  Out: str."""
        return os.path.join(self.rpts, "fea_cell_%s.rpt" % cell_tag(di, dj))

    def station(self, t, kind):
        """stations/<tag>_<kind>.ff.  In: t tag; kind bare|withF."""
        return os.path.join(self.stations, "%s_%s.ff" % (t, kind))

    def dat(self, t, kind, n):
        """dats/<tag>/ file of one curve kind and path number.

        In:  t str tag; kind str fea|refined|classical; n int 1-based
        Out: str path -- fea files keep step06's fea_<pathstem>.dat name,
             the OpenSG ones are <kind>_path<n>.dat."""
        d = os.path.join(self.dats, t)
        if kind == "fea":
            stem = os.path.splitext(os.path.basename(self.paths[n - 1]))[0]
            return os.path.join(d, "fea_%s.dat" % stem)
        return os.path.join(d, "%s_path%d.dat" % (kind, n))


def st01(C):
    """Stage 1: the bare plate deck + cellmap.  In: C ctx  Out: None."""
    run_mod("step01_make_decks",
            ["--sg", C.a.sg, "--law", C.a.law, "--nx", C.a.nx,
             "--ny", C.a.ny, "--q", C.a.q, "--out", C.decks])


def st02(C):
    """Stage 2: level-1/level-2 integrity.  In: C ctx  Out: None."""
    run_mod("step02_levels", [C.cellmap])


def st03(C):
    """Stage 3: the withF deck.  Eq. 47 by default, legacy --fn on
    request.  BOTH parts of the load column are applied -- the edge
    traction from F_N and the edge moment from F_M -- because the
    Reissner-like law of Yu 2003 Eq. 61 carries both; --no-moment falls
    back to the F_N-only deck the pipeline built before 2026-09-10.
    In: C ctx  Out: None."""
    if C.a.fn is not None:
        run_mod("step03_plate_inp_withF",
                ["inject", "--inp", C.plate_inp, "--fn", C.a.fn,
                 "--out", C.withf_inp])
    else:
        args = ["--inp", C.plate_inp, "--src-dir", C.src_dir,
                "--q", C.a.q, "--out", C.withf_inp]
        if not C.a.no_moment:
            args.append("--with-moment")
        run_mod("make_load_column_deck", args)


def st04(C):
    """Stage 4: the 3-D solid reference deck (written only).
    In: C ctx  Out: None."""
    run_mod("step04_solid_inp",
            ["--sg", C.a.sg, "--nx", C.a.nx, "--ny", C.a.ny,
             "--q", C.a.q, "--out", C.solid_inp])


def st05(C):
    """Stage 5: Abaqus solves + .rpt dumps; may stop for the 3-D side.

    In:  C ctx
    Out: False when the pipeline must stop for a user-run solve,
         None when every needed .rpt exists."""
    ok = C.a.run_abaqus in ("plate", "solid")
    for kind, inp in (("bare", C.plate_inp), ("withF", C.withf_inp)):
        rpt = C.plate_rpt[kind]
        if os.path.exists(rpt):
            print("  %s exists" % rpt)
            continue
        job = "plate_" + kind
        run_cmd = "abaqus job=%s input=%s interactive" % (job, inp)
        if not ok:
            print("  plate job left to you -- in %s run:" % C.rpts)
            print("    %s" % run_cmd)
            print("    python -m %sstep05_dump_rpts plate %s"
                  % (PKG, os.path.join(C.rpts, job)))
            return False
        sh(run_cmd, cwd=C.rpts)
        run_mod("step05_dump_rpts", ["plate", os.path.join(C.rpts, job)])
    for di, dj in C.cells:
        rpt = C.solid_rpt(di, dj)
        if os.path.exists(rpt):
            print("  %s exists" % rpt)
            continue
        if C.a.solid_odb and C.a.run_abaqus == "solid":
            run_mod("step05_dump_rpts",
                    ["solid", C.a.solid_odb, "--cell", di, dj,
                     "--pitch", C.a.pitch])
            src = os.path.join(os.path.dirname(C.a.solid_odb),
                               os.path.basename(rpt))
            shutil.copy2(src, rpt)
            continue
        print("  3-D cell report missing: %s" % rpt)
        print("    either copy fea_cell_%s.rpt (from the 3-D odb) into %s,"
              % (cell_tag(di, dj), C.rpts))
        print("    or rerun with --solid-odb <odb stem> --run-abaqus solid")
        return False


def st06(C):
    """Stage 6: 3-D FEA path .dat per (cell, path), cell offset applied.
    In: C  Out: None."""
    for di, dj in C.cells:
        t = cell_tag(di, dj)
        run_mod("step06_fea_path_dat",
                [C.solid_rpt(di, dj), "--paths"] + C.paths +
                ["--offset", di * C.a.pitch, dj * C.a.pitch,
                 "--out-dir", os.path.join(C.dats, t)])


def st07(C):
    """Stage 7: two stations per cell -- withF for the refined recovery,
    bare for the classical one (step08).  In: C  Out: None."""
    for di, dj in C.cells:
        t = cell_tag(di, dj)
        for kind in ("withF", "bare"):
            run_mod("step08_station_ff",
                    [C.plate_rpt[kind], "--law", C.a.law,
                     "--cell", di, dj, "--cellmap", C.cellmap,
                     "--q0", C.a.q, "--out", C.station(t, kind)])


def st08(C):
    """Stage 8: nothing separate.  In: C ctx  Out: None."""
    print("  folded into stage 7 (step08 consumes step07)")


def st09(C):
    """Stage 9: shear-refined dehom per cell.  In: C  Out: None."""
    for di, dj in C.cells:
        t = cell_tag(di, dj)
        run_mod("step09_run_dehom",
                ["--src-dir", C.src_dir, "--ff", C.station(t, "withF"),
                 "--workdir", os.path.join(C.dehom, t)])


def st10(C):
    """Stage 10: OpenSG path .dat per (cell, path).  In: C  Out: None."""
    for di, dj in C.cells:
        t = cell_tag(di, dj)
        prefix = os.path.join(C.dehom, t, C.stem + "_dehom")
        os.makedirs(os.path.join(C.dats, t), exist_ok=True)
        for n, pf in enumerate(C.paths, 1):
            run_mod("step10_opensg_path_dat",
                    [prefix, pf, C.dat(t, "refined", n)])


def st11(C):
    """Stage 11: the classical twin on the BARE station + its path
    .dat.  In: C  Out: None."""
    for di, dj in C.cells:
        t = cell_tag(di, dj)
        wd = os.path.join(C.dehom_cls, t)
        run_mod("step11_classical",
                ["--src-dir", C.src_dir, "--ff", C.station(t, "bare"),
                 "--workdir", wd, "--paths"] + C.paths +
                ["--out-dir", wd])
        for n in range(1, len(C.paths) + 1):
            for ext in (".dat", "_elements.csv"):
                src = os.path.join(wd, "path%d%s" % (n, ext))
                if os.path.exists(src):
                    dst = C.dat(t, "classical", n)
                    if ext != ".dat":
                        dst = dst[:-4] + ext
                    shutil.copy2(src, dst)


def st12(C):
    """Stage 12: the 3-curve figures per (cell, path).  In: C  Out: None."""
    for di, dj in C.cells:
        t = cell_tag(di, dj)
        for n in range(1, len(C.paths) + 1):
            run_mod("step12_plots",
                    ["--fea", C.dat(t, "fea", n),
                     "--refined", C.dat(t, "refined", n),
                     "--classical", C.dat(t, "classical", n),
                     "--path-label", n,
                     "--out-dir", os.path.join(C.plots, t)])


STAGES = [(1, "decks", st01), (2, "verify", st02), (3, "withF", st03),
          (4, "solid deck", st04), (5, "rpts", st05),
          (6, "fea dats", st06), (7, "stations", st07),
          (8, "(stations)", st08), (9, "refined dehom", st09),
          (10, "opensg dats", st10), (11, "classical", st11),
          (12, "plots", st12)]


def main(argv=None):
    """Parse the CLI and run the requested stages, skipping finished ones.

    In:  argv list | None
    Out: int exit status."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--sg", required=True)
    p.add_argument("--law", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--nx", type=int, default=5)
    p.add_argument("--ny", type=int, default=5)
    p.add_argument("--pitch", type=float, default=1.0,
                   help="cell pitch [m], the offset unit of stage 6")
    p.add_argument("--q", type=float, required=True,
                   help="top-face pressure [Pa]")
    p.add_argument("--paths", nargs="+", required=True,
                   help="path .dat files, rows s y1 y2 y3 (path 1, 2, ...)")
    p.add_argument("--cells", default="0 0",
                   help='signed cell offsets "DI DJ[,DI DJ...]"')
    p.add_argument("--stages", default="all",
                   help="A-B, comma list, or all")
    p.add_argument("--run-abaqus", choices=["plate", "solid", "none"],
                   default="plate")
    p.add_argument("--solid-odb", default=None,
                   help="3-D odb path WITHOUT .odb, for stage 5 cell dumps")
    p.add_argument("--fn", type=float, default=None,
                   help="legacy: F_N [N/m] via step03 inject instead of"
                        " Eq. 47")
    p.add_argument("--no-moment", action="store_true",
                   help="stage 3: apply only F_N, the pre-2026-09-10"
                        " behaviour (the default applies F_N and F_M)")
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    a.sg = os.path.abspath(a.sg)
    a.law = os.path.abspath(a.law)
    if a.solid_odb:
        a.solid_odb = os.path.abspath(a.solid_odb)
    C = _Ctx(a)
    for d in (C.decks, C.rpts, C.stations, C.dehom, C.dehom_cls,
              C.dats, C.plots):
        os.makedirs(d, exist_ok=True)
    sf = os.path.join(C.root, "pipeline_state.json")
    state = {}
    if os.path.exists(sf):
        with open(sf) as f:
            state = json.load(f)
    todo = parse_stages(a.stages)
    print("start %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    for num, name, fn in STAGES:
        if num not in todo:
            continue
        if state.get(str(num)) and not a.force:
            print("stage %2d %-14s done (skip)" % (num, name))
            continue
        print("stage %2d %s" % (num, name))
        if fn(C) is False:
            print("stopped at stage %d -- rerun to resume" % num)
            print("end %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
            return 0
        state[str(num)] = True
        with open(sf, "w") as f:
            json.dump(state, f, indent=1, sort_keys=True)
    print("end %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
