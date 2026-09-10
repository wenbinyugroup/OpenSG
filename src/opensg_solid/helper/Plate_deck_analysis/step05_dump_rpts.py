"""step05_dump_rpts.py -- run the odb -> .rpt dumper of EITHER benchmark
job from a normal python: the plate run (vendored abq_dump_plate_rpt.py)
or the 3-D solid run (plate_rm's abq_cell_dump.py, reused).

    python -m opensg_solid.helper.Plate_deck_analysis.step05_dump_rpts \
           {plate|solid} <job stem> [--cell DI DJ] [--pitch 1.0] \
           [--abaqus abaqus]

WHY A SUBPROCESS.  odbAccess exists only under `abaqus python`, so the
odb-dependent surface stays in one dumper per job type and everything
downstream (steps 06-12) reads text.  The plate dumper is VENDORED here
(its home, the pressure_case study, sits outside the package); the solid
dumper is REUSED from plate_rm -- the same file fea_cell.py has always
subprocess-launched.  Both are resolved from this file's own location,
never from the working directory.

THE SOLID CELL.  --cell DI DJ are the signed cell offsets from the plate
centre (plate_rm convention: (0, 0) = centre cell, +1 = one cell along
+x/+y, negatives toward -x/-y).  The dump keeps the elements within
+-pitch/2 of (cx, cy) = (DI*pitch, DJ*pitch) and writes
fea_cell_<tag>.rpt beside the job, tag in the plate_rm filename
convention (i1j2; negatives i1Nj2N).  The rpt coordinates stay in the
job frame -- step06's --offset shifts the cell-local paths in.

If the abaqus launcher is not on PATH the step fails and prints the
exact command to run in an Abaqus shell instead.

In:  the solved .odb (job stem, path without .odb) + mode [+ cell]
Out: plate -> <job>_plate.rpt;  solid -> fea_cell_<tag>.rpt beside the
     job; the dumper's output tail is printed"""
import argparse
import os
import shutil
import subprocess
import sys

TAIL = 12


def cell_tag(di, dj):
    """Filename token of a signed cell offset (plate_rm.fea_cell
    convention, kept inline so this step needs no opensg import).

    In:  di, dj int
    Out: str, e.g. (+1, -2) -> 'i1j2N'."""
    return "i%d%sj%d%s" % (abs(di), "N" if di < 0 else "",
                           abs(dj), "N" if dj < 0 else "")


def dumper_paths():
    """Resolve both dumper scripts from this file's location.

    In:  --
    Out: dict {"plate": abs path (vendored), "solid": abs path
         (plate_rm's abq_cell_dump.py)}."""
    here = os.path.dirname(os.path.abspath(__file__))
    return {"plate": os.path.join(here, "abq_dump_plate_rpt.py"),
            "solid": os.path.normpath(os.path.join(
                here, "..", "plate_rm", "abq_cell_dump.py"))}


def run_dumper(cmd, expect):
    """Run one dumper, print its output tail, verify the rpt exists.

    In:  cmd list[str]; expect str -- the rpt the dumper must produce
    Out: int exit status (0 = rpt exists)."""
    print(" ".join(cmd))
    r = subprocess.run(cmd, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT)
    txt = r.stdout.decode(errors="replace").strip()
    for ln in txt.splitlines()[-TAIL:]:
        print("  | " + ln)
    if r.returncode != 0 or not os.path.exists(expect):
        print("dumper failed (rc %d): %s missing"
              % (r.returncode, expect))
        return r.returncode or 1
    print("%s  (%d bytes)" % (expect, os.path.getsize(expect)))
    return 0


def main(argv=None):
    """CLI -- see the module docstring.

    In:  argv list | None
    Out: int exit status."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("mode", choices=("plate", "solid"))
    p.add_argument("job", help="odb path WITHOUT .odb")
    p.add_argument("--cell", nargs=2, type=int, default=[0, 0],
                   metavar=("DI", "DJ"),
                   help="solid only: signed cell offset from the"
                        " plate centre")
    p.add_argument("--pitch", type=float, default=1.0,
                   help="solid only: cell pitch [m]")
    p.add_argument("--abaqus", default="abaqus",
                   help="abaqus launcher command")
    a = p.parse_args(argv)

    script = dumper_paths()[a.mode]
    if not os.path.exists(script):
        print("dumper not found: %s" % script)
        return 2
    if a.mode == "plate":
        cmd = [a.abaqus, "python", script, a.job]
        expect = a.job + "_plate.rpt"
    else:
        tag = cell_tag(a.cell[0], a.cell[1])
        cx = a.cell[0] * a.pitch
        cy = a.cell[1] * a.pitch
        expect = os.path.join(os.path.dirname(os.path.abspath(a.job)),
                              "fea_cell_%s.rpt" % tag)
        cmd = [a.abaqus, "python", script, a.job, expect,
               str(cx), str(cy), str(a.pitch / 2.0)]
    if shutil.which(a.abaqus) is None:
        print("'%s' not found on PATH; run this yourself in an Abaqus"
              " shell:" % a.abaqus)
        print("  " + " ".join(cmd))
        return 2
    return run_dumper(cmd, expect)


if __name__ == "__main__":
    sys.exit(main())
