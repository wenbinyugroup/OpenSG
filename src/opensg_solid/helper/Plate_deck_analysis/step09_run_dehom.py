"""step09_run_dehom.py -- stage a work dir and run one shear-refined
dehomogenization: `opensg_solid <stem>.yaml D --global`.

    python -m opensg_solid.helper.Plate_deck_analysis.step09_run_dehom \
           --src-dir <homog dir> --ff <station.ff> --workdir <dir> \
           [--stem <stem>] [--opensg opensg_solid]

WHY A STAGED COPY.  `opensg_solid <yaml> D` resolves every input and
export next to the yaml, so each macro state (each station .ff, each
refined/classical twin) gets its OWN work dir holding copies of the
homogenization outputs -- runs never overwrite each other and every
dehom_console.log stays with its exports.  The layout mirrors the
validated TPMS run (pressure_case/fea_driven/scripts/run_withF_dehom.sh):

    <workdir>/<stem>.yaml        the SG (header flags + mesh + layup)
    <workdir>/<stem>.out         the homogenized law read back by D
    <workdir>/<stem>_sg.npz      parsed-mesh cache (optional; skips the
                                 multi-MB yaml re-parse)
    <workdir>/<stem>.ff          THE macro state -- always re-copied,
                                 it is what distinguishes this run
    <workdir>/dehom_console.log  full console; the macro state /
                                 Time taken / error lines re-printed

The stem is never hard-coded: it is inferred from the single *.yaml in
--src-dir unless --stem is given.  yaml/.out/npz already in the work
dir are kept (resumable staging).

In:  the homogenization outputs (<stem>.yaml/.out[/_sg.npz]) + one .ff
Out: the staged work dir and the dehom exports <workdir>/<stem>_dehom*
     (prefix printed); exit status of the solver"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import time

KEY = re.compile(r"macro state|Time taken|rror")


def infer_stem(src_dir):
    """The stem of the single *.yaml in a homogenization dir.

    In:  src_dir str
    Out: stem str (SystemExit when zero or several yamls are found)."""
    ys = sorted(f for f in os.listdir(src_dir) if f.endswith(".yaml"))
    if len(ys) != 1:
        raise SystemExit("%s holds %d *.yaml files -- pass --stem"
                         % (src_dir, len(ys)))
    return ys[0][:-5]


def stage_inputs(src_dir, workdir, stem, include_yaml=True):
    """Copy the homogenization outputs into the work dir, skipping any
    file already there; the npz cache is optional, yaml/.out are not.

    In:  src_dir, workdir, stem str; include_yaml bool -- False when
         the caller writes its own yaml (step11's refined: 0 rewrite)
    Out: list of file names copied (one line printed)."""
    os.makedirs(workdir, exist_ok=True)
    names = [stem + ".yaml"] if include_yaml else []
    names += [stem + ".out", stem + "_sg.npz"]
    copied = []
    for n in names:
        src = os.path.join(src_dir, n)
        dst = os.path.join(workdir, n)
        if os.path.exists(dst):
            continue
        if not os.path.exists(src):
            if n.endswith("_sg.npz"):
                continue
            raise SystemExit("missing %s" % src)
        shutil.copy2(src, dst)
        copied.append(n)
    print("staged %s -> %s"
          % (", ".join(copied) if copied else "nothing new", workdir))
    return copied


def stage_ff(ff, workdir, stem):
    """Copy the station file to <workdir>/<stem>.ff -- ALWAYS
    overwritten: the .ff is what distinguishes this run.

    In:  ff, workdir, stem str
    Out: destination path str (printed)."""
    dst = os.path.join(workdir, stem + ".ff")
    shutil.copy2(ff, dst)
    print("ff %s -> %s" % (ff, dst))
    return dst


def run_dehom(workdir, stem, opensg="opensg_solid"):
    """Run `<opensg> <stem>.yaml D --global` in the work dir, console
    teed to dehom_console.log; the macro state / Time taken / error
    lines are re-printed and the export prefix printed on success.

    In:  workdir, stem str; opensg str -- the console script
    Out: int exit status (nonzero on solver failure or an error line
         in the console)."""
    log = os.path.join(workdir, "dehom_console.log")
    cmd = [opensg, stem + ".yaml", "D", "--global"]
    print("dehom start %s  (%s in %s)"
          % (time.strftime("%H:%M:%S"), " ".join(cmd), workdir))
    with open(log, "w") as f:
        try:
            rc = subprocess.call(cmd, cwd=workdir, stdout=f,
                                 stderr=subprocess.STDOUT)
        except OSError as e:
            raise SystemExit("cannot run %s: %s" % (opensg, e))
    print("dehom end   %s" % time.strftime("%H:%M:%S"))
    bad = rc != 0
    for ln in open(log):
        if KEY.search(ln):
            print("  " + ln.rstrip())
            bad = bad or ("rror" in ln)
    if bad:
        print("dehom FAILED (rc %d) -- see %s" % (rc, log))
        return rc if rc != 0 else 1
    print("dehom prefix %s" % os.path.join(workdir, stem + "_dehom"))
    return 0


def main(argv=None):
    """CLI: stage and run one shear-refined dehomogenization.

    In:  argv list | None
    Out: int exit status."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--src-dir", required=True,
                   help="dir holding <stem>.yaml/.out[/_sg.npz]")
    p.add_argument("--ff", required=True, help="the station file")
    p.add_argument("--workdir", required=True)
    p.add_argument("--stem", default=None,
                   help="default: the single *.yaml in --src-dir")
    p.add_argument("--opensg", default="opensg_solid")
    a = p.parse_args(argv)
    stem = a.stem or infer_stem(a.src_dir)
    stage_inputs(a.src_dir, a.workdir, stem)
    stage_ff(a.ff, a.workdir, stem)
    return run_dehom(a.workdir, stem, opensg=a.opensg)


if __name__ == "__main__":
    sys.exit(main())
