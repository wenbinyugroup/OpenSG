"""step10_opensg_path_dat.py -- the OpenSG dehom fields along a path,
written as one .dat per path: element coordinates + 6 stress + 3
displacement (the unified path layout shared by steps 06 and 12).

    python -m opensg_solid.helper.Plate_deck_analysis.step10_opensg_path_dat \
           <dehom_prefix> <path.dat> <out.dat>

Vendored 2026-09-03 from the validated TPMS study
(M:\\Abaqus\\pressure_case\\path\\opensg_path_dat.py), reshaped into
functions so step11 and the orchestrator drive it in-process; the
sampling and the output layout are unchanged.

<dehom_prefix> names the dehom exports of the station SG (printed by
step09), e.g. <workdir>/<stem>_dehom:

    <prefix>_elemental_global.SM   per-ELEMENT stress (4-Gauss mean at
      (else _elemental.SM)         the centroid) -- the elemental
                                   quantity that pairs with Abaqus
                                   centroidal S; the _global suffix is
                                   preferred when present (written by a
                                   --global run)
    <prefix>_elemental.U | .U      fluctuation displacement (SG-global
                                   frame, never rotated)

WHY NEAREST-CENTROID.  Sampling mirrors
opensg_solid.io.abq_rpt.sample_path exactly: each path point takes the
NEAREST element centroid's stress row (piecewise constant, so nearest
centroid = containing element to raster accuracy) and the nearest .U
row's displacement.  The chosen element row per point is kept in
<out stem>_elements.csv so the pairing with the 3-D FEA _elements.csv
stays auditable.

Column layout of the output .dat (one row per path point):

    s  x  y  z  S11 S22 S33 S12 S13 S23  U1 U2 U3

s is the path's own non-dimensional parameter, carried through from
<path.dat>; stress order is the rm_plate_1D print order 11 22 33 12 13
23 (both input files already use it).

In:  the dehom exports + the path .dat (s y1 y2 y3)
Out: <out.dat> and <out stem>_elements.csv"""
import argparse
import os
import sys

import numpy as np


def read_field(path, ncomp):
    """One rm_plate_1D-layout field file.

    In:  path str; ncomp int -- components after x y z
    Out: (xyz (n,3), F (n,ncomp))."""
    rows = []
    for ln in open(path):
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        v = s.split()
        try:
            rows.append([float(x) for x in v])
        except ValueError:
            continue
    a = np.array(rows)
    return a[:, :3], a[:, 3:3 + ncomp]


def resolve_exports(prefix):
    """The stress/displacement export pair of one dehom prefix;
    _elemental_global.SM (a --global run) wins over _elemental.SM and
    _elemental.U over .U.

    In:  prefix str
    Out: (sm path, u path) str (SystemExit when either is missing)."""
    sm = prefix + "_elemental_global.SM"
    if not os.path.exists(sm):
        sm = prefix + "_elemental.SM"
    uu = prefix + "_elemental.U"
    if not os.path.exists(uu):
        uu = prefix + ".U"
    for f in (sm, uu):
        if not os.path.exists(f):
            raise SystemExit("missing %s -- run the dehom + elemental"
                             " export first" % f)
    return sm, uu


def sample_path(prefix, path_dat, out_dat):
    """Nearest-centroid stress + nearest-point displacement along one
    path, written in the unified layout with its _elements.csv audit.

    In:  prefix str -- the dehom exports; path_dat str -- the
         s y1 y2 y3 table; out_dat str
    Out: (out_dat, csv path) written (both printed)."""
    sm, uu = resolve_exports(prefix)
    exyz, S = read_field(sm, 6)
    uxyz, U = read_field(uu, 3)
    P = np.atleast_2d(np.loadtxt(path_dat))
    print("elemental.SM: %d elements;  .U: %d points;  path: %d points"
          % (len(exyz), len(uxyz), len(P)))

    rows = []
    eids = np.zeros(len(P), dtype=np.int64)
    for i, (s, x, y, z) in enumerate(P):
        p = np.array([x, y, z])
        k = int(np.argmin(np.einsum("ij,ij->i", exyz - p, exyz - p)))
        j = int(np.argmin(np.einsum("ij,ij->i", uxyz - p, uxyz - p)))
        rows.append([s, x, y, z] + list(S[k]) + list(U[j]))
        eids[i] = k
    nd = 1 + int(np.sum(np.diff(eids) != 0))
    print("distinct elements visited: %d" % nd)

    hdr = ("OpenSG dehom along %s -- source %s\n"
           "%12s %14s %14s %14s" % (os.path.basename(path_dat),
                                    os.path.basename(prefix), "s",
                                    "x[m]", "y[m]", "z[m]")
           + "".join(" %14s" % c for c in
                     ("S11[Pa]", "S22[Pa]", "S33[Pa]", "S12[Pa]",
                      "S13[Pa]", "S23[Pa]", "U1[m]", "U2[m]",
                      "U3[m]")))
    np.savetxt(out_dat, np.array(rows), fmt="%14.6e", header=hdr)
    csv = os.path.splitext(out_dat)[0] + "_elements.csv"
    with open(csv, "w") as f:
        f.write("s,elem_row\n")
        for r, e in zip(rows, eids):
            f.write("%.6e,%d\n" % (r[0], e))
    print("wrote %s" % out_dat)
    print("wrote %s" % csv)
    return out_dat, csv


def main(argv=None):
    """CLI: one path -> one unified .dat (+ _elements.csv).

    In:  argv list | None
    Out: int exit status."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("prefix",
                   help="dehom export prefix, e.g. <wd>/<stem>_dehom")
    p.add_argument("path_dat", help="path table: s y1 y2 y3")
    p.add_argument("out_dat")
    a = p.parse_args(argv)
    sample_path(a.prefix, a.path_dat, a.out_dat)
    return 0


if __name__ == "__main__":
    sys.exit(main())
