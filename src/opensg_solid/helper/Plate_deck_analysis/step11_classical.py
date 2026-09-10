"""step11_classical.py -- the classical twin of a finished refined
setup: same SG, same station, refined: 0 law and an FF-only macro
state, then steps 09-10 on the copy.

    python -m opensg_solid.helper.Plate_deck_analysis.step11_classical \
           --src-dir <refined homog dir> --ff <station.ff> \
           --workdir <dir> --paths P1 [P2 ...] [--out-dir <dir>] \
           [--stem <stem>] [--opensg opensg_solid]

WHY A TWIN, NOT A FLAG.  The classical (refined: 0) recovery is the
benchmark's GREEN curve: it shares the mesh, the layup and the station
resultants with the refined run and differs ONLY in the yaml header
flag and in what the .ff carries.  The strain-derivative families
(deps_*, d2eps_*), the surface tractions qt6/qb6 and the q_reaction
line are shear-refinement inputs, so the twin strips every one of them
and keeps u/theta/0:/1:/FF -- the exact prep of the validated TPMS run
(pressure_case/fea_driven/scripts/run_withF_dehom.sh).  WHY qt6/qb6 GO
TOO (verified 2026-09-07): the pressure-driven local field is the load
column V1L / V2L of the FIRST-order (refined) warping ladder -- Yu 2003
introduces the load L after Eq. 28 and it enters the first-order
energy, while the zeroth-order (classical) warping problem, Eq. 33,
carries no load term.  A classical recovery therefore has no mechanism
to use qt6 (sg_dehom applies it only through V1Lt/V2Lt, which only a
refined ladder stores); the pressure reaches the classical field only
through the plate solution, i.e. the station's strains/curvatures.  A
flat classical face-sheet stress with sigma33 = 0 under q is the
classical model's known limitation, not a staging error.  (A 2026-09-07
experiment that gave the classical twin the load columns was REJECTED
on these grounds; the numbers are in the case READMEs.)  The _sg.npz
staged by step09 is a parsed-MESH cache (nodes/cells/mat_id, no law),
identical between the twins, so reusing the refined one is safe.

WHY THE 4 KB HEADER REWRITE.  The yaml opens with a few header keys
(msg / n_model / refined) before the mesh, and the mesh can run to
tens of MB -- so the refined: 1 -> refined: 0 rewrite string-replaces
the FIRST occurrence within the first 4 KB only and stream-copies the
remainder untouched.

In:  the refined homogenization dir (<stem>.yaml/.out[/_sg.npz]), the
     station .ff, one or more path .dat files (s y1 y2 y3)
Out: the classical work dir with its dehom exports, and one unified
     path<i>.dat (+ _elements.csv) per path in --out-dir (default the
     work dir)"""
import argparse
import os
import shutil
import sys

FF_STRIP = ("deps_", "d2eps_", "qt6", "qb6", "q_reaction")
HEAD = 4096


def _load_step(name):
    """Import a sibling step module: package-first, file fallback (so
    the twin also runs beside a pip snapshot that lags the tree).

    In:  name str, e.g. "step09_run_dehom"
    Out: the module object."""
    try:
        import importlib
        return importlib.import_module(
            "opensg_solid.helper.Plate_deck_analysis." + name)
    except ImportError:
        import importlib.util
        here = os.path.dirname(os.path.abspath(__file__))
        spec = importlib.util.spec_from_file_location(
            name, os.path.join(here, name + ".py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


def write_classical_yaml(src_yaml, dst_yaml):
    """Copy the SG yaml with its header's first "refined: 1" rewritten
    to "refined: 0"; only the first 4 KB are searched, the rest is
    stream-copied (the file can be tens of MB).

    In:  src_yaml, dst_yaml str
    Out: dst_yaml str (printed; SystemExit when no refined flag is in
         the header)."""
    with open(src_yaml, "rb") as fi:
        head = fi.read(HEAD)
        if b"refined: 0" in head:
            print("note: %s is already refined: 0" % src_yaml)
        elif b"refined: 1" in head:
            head = head.replace(b"refined: 1", b"refined: 0", 1)
        else:
            raise SystemExit("no 'refined: 1' in the first %d bytes of"
                             " %s" % (HEAD, src_yaml))
        with open(dst_yaml, "wb") as fo:
            fo.write(head)
            shutil.copyfileobj(fi, fo, 1024 * 1024)
    print("yaml %s -> %s (refined: 0)" % (src_yaml, dst_yaml))
    return dst_yaml


def write_classical_ff(src_ff, dst_ff):
    """Copy the station .ff with every shear-refinement line stripped
    (deps_*, d2eps_*, qt6, qb6, q_reaction).

    In:  src_ff, dst_ff str
    Out: dst_ff str (kept/dropped counts printed)."""
    kept = dropped = 0
    with open(src_ff) as fi, open(dst_ff, "w") as fo:
        for ln in fi:
            if ln.startswith(FF_STRIP):
                dropped += 1
                continue
            fo.write(ln)
            kept += 1
    print("ff %s -> %s (%d lines kept, %d refinement lines stripped)"
          % (src_ff, dst_ff, kept, dropped))
    return dst_ff


def run_classical(src_dir, ff, workdir, paths, out_dir=None, stem=None,
                  opensg="opensg_solid"):
    """Stage the classical twin, run its dehom and sample every path.

    In:  src_dir str -- the refined homogenization dir; ff str -- the
         refined station file; workdir str; paths list of path .dat;
         out_dir str | None (default workdir); stem str | None
         (default: inferred from src_dir); opensg str
    Out: int exit status (the dehom's when it fails)."""
    s9 = _load_step("step09_run_dehom")
    s10 = _load_step("step10_opensg_path_dat")
    if stem is None:
        stem = s9.infer_stem(src_dir)
    os.makedirs(workdir, exist_ok=True)
    dst_yaml = os.path.join(workdir, stem + ".yaml")
    if os.path.exists(dst_yaml):
        print("yaml %s kept" % dst_yaml)
    else:
        write_classical_yaml(os.path.join(src_dir, stem + ".yaml"),
                             dst_yaml)
    s9.stage_inputs(src_dir, workdir, stem, include_yaml=False)
    write_classical_ff(ff, os.path.join(workdir, stem + ".ff"))
    rc = s9.run_dehom(workdir, stem, opensg=opensg)
    if rc != 0:
        return rc
    prefix = os.path.join(workdir, stem + "_dehom")
    out_dir = out_dir or workdir
    os.makedirs(out_dir, exist_ok=True)
    for i, p in enumerate(paths, 1):
        s10.sample_path(prefix, p,
                        os.path.join(out_dir, "path%d.dat" % i))
    return 0


def main(argv=None):
    """CLI: the classical twin end to end.

    In:  argv list | None
    Out: int exit status."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--src-dir", required=True,
                   help="refined dir holding <stem>.yaml/.out[/npz]")
    p.add_argument("--ff", required=True,
                   help="the refined station file (stripped here)")
    p.add_argument("--workdir", required=True)
    p.add_argument("--paths", nargs="+", required=True,
                   metavar="PATH_DAT")
    p.add_argument("--out-dir", default=None,
                   help="path<i>.dat destination (default --workdir)")
    p.add_argument("--stem", default=None,
                   help="default: the single *.yaml in --src-dir")
    p.add_argument("--opensg", default="opensg_solid")
    a = p.parse_args(argv)
    return run_classical(a.src_dir, a.ff, a.workdir, a.paths,
                         out_dir=a.out_dir, stem=a.stem,
                         opensg=a.opensg)


if __name__ == "__main__":
    sys.exit(main())
