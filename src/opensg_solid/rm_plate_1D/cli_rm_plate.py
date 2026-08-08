"""cli_rm_plate.py -- the OpenSG-RM plate COMMAND LINE: homogenize, plot and
dehomogenize a through-thickness 1-D SG without writing any Python.

    opensg-rm-plate homo  SG.yaml [--out PREFIX]
    opensg-rm-plate plot  SG.yaml [--png PATH]
    opensg-rm-plate dehom SG.yaml (--FF N11 N22 N12 M11 M22 M12 Q1 Q2 |
                                   --strain e11 e22 g12 k11 k22 k12 2g13 2g23)
                          [--u2d U1 U2 U3] [--qtop q q1 q2 q11 q12 q22]
                          [--qbot ...] [--n-per-ply N] [--base PREFIX]

(equivalently ``python -m opensg_solid.rm_plate_1D.cli_rm_plate ...``).

homo    prints the labeled 8x8 ABDG and its compliance; --out writes
        PREFIX_8x8.out.
plot    writes the SG mesh figure (defaults next to the YAML).
dehom   the VABS-style recovery: from resultants (--FF, converted through the
        8x8) or plate strains (--strain), plus optional face-pressure
        ladders (--qtop/--qbot, load case 4).  Writes PREFIX.SM / .EM / .U /
        .out: material-frame stress, strain, and the Eq.-65 displacement
        (u_2d from --u2d) at n-per-ply stations per ply.

Functions
---------
_load(yaml_path)          read the SG and homogenize; returns (inp, r)
cmd_homo / cmd_plot /     one function per subcommand; each takes the parsed
cmd_dehom                 argparse namespace and returns an exit code
main(argv=None)           the console entry point (argparse dispatch)

Variables (dehom)
-----------------
FF, E6      the 8-resultant vector and the 6 plate strains (E6 = S6 FF[:6]
            when --FF is given; FF = ABDG strain when --strain is given)
dE1, dE2    None here: a single FF station fixes no strain gradients, so the
            recovery is classical (gradients need an FF field, rm_dehom_dat)
qt6, qb6    the face-pressure ladders (None when the flags are absent)
zs          recovery stations: n-per-ply points per ply, faces included
rows        per-station (z, 6 values) tables written to the output files;
            .SM/.EM are in the PLY (material) frame -- the core default of
            msgrm_strain_at_depth (frame="material")
"""
import argparse
import os
import sys

import numpy as np

from .msg_rm_plate import (rm_plate_msg, msgrm_strain_at_depth,
                           msgrm_warping_at_depth)
from .segment_plate import read_plate_sg_yaml, plot_plate_sg

_ROWS = ("e11", "e22", "g12", "k11", "k22", "k12", "2g13", "2g23")
_FFL = ("N11", "N22", "N12", "M11", "M22", "M12", "Q1", "Q2")


def _load(yaml_path):
    """Read the SG YAML and homogenize (returns the reader dict + the rm dict)."""
    inp = read_plate_sg_yaml(yaml_path)
    r = rm_plate_msg(inp["thick"], inp["angles"], inp["mat_names"],
                     inp["material_db"], n_per_layer=inp["n_per_layer"],
                     elem_order=inp["elem_order"], fraction=inp["fraction"])
    return inp, r


def _mat8(M, rows):
    """Format a square matrix as labeled text lines for the .out files.

    In:
        M: (n, n) array-like, matrix to print.
        rows: sequence of n str, row/column labels.
    Out:
        list of str: header line plus one labeled row per matrix row (%14.6e).
    """
    out = ["        " + "".join("%14s" % k for k in rows)]
    for k, row in zip(rows, np.asarray(M)):
        out.append("%7s " % k + "".join("%14.6e" % v for v in row))
    return out


def cmd_homo(a):
    """Subcommand 'homo': print the labeled 8x8 ABDG and its compliance.

    In:
        a: argparse.Namespace with yaml (SG YAML path) and out (optional
           output prefix; when set, also writes PREFIX_8x8.out).
    Out:
        int: exit code (0 ok, 2 if the transverse-shear fit is not SPD).
    """
    inp, r = _load(a.yaml)
    if r["ABDG"] is None:
        print("ERROR: the transverse-shear fit is not SPD -- inspect the layup")
        return 2
    lines = ["OpenSG-RM plate 8x8 ABDG  (%s)" % os.path.abspath(a.yaml),
             "plies (bottom->top): " + ", ".join(
                 "%s(%g/%g)" % (m, t, x) for m, t, x in
                 zip(inp["mat_names"], inp["thick"], inp["angles"])),
             "reference_fraction = %g" % inp["fraction"],
             ""]
    lines += _mat8(r["ABDG"], _ROWS)
    lines += ["", "compliance inv(ABDG):"]
    lines += _mat8(np.linalg.inv(np.asarray(r["ABDG"])), _ROWS)
    txt = "\n".join(lines)
    print(txt)
    if a.out:
        path = a.out + "_8x8.out"
        with open(path, "w") as f:
            f.write(txt + "\n")
        print("\nwrote %s" % path)
    return 0


def cmd_plot(a):
    """Subcommand 'plot': write the SG mesh figure.

    In:
        a: argparse.Namespace with yaml (SG YAML path) and png (optional
           output path; default next to the YAML).
    Out:
        int: exit code (0).
    """
    png = plot_plate_sg(a.yaml, png_path=a.png)
    print("wrote %s" % png)
    return 0


def cmd_dehom(a):
    """Subcommand 'dehom': recover through-thickness 3-D stress, strain and
    displacement; writes PREFIX.SM / .EM / .U / .out (ply/material frame).

    In:
        a: argparse.Namespace with yaml (SG YAML path), FF (8 resultants) or
           strain (8 plate strains) [mutually exclusive], u2d (3 plate
           displacements), qtop/qbot (6 face-pressure ladder coefficients or
           None), n_per_ply (int stations per ply), base (output prefix or
           None -> YAML path without extension).
    Out:
        int: exit code (0 ok, 2 if the transverse-shear fit is not SPD).

    A single FF station fixes no strain gradients, so dE1 = dE2 = None
    (classical recovery); gradient-aware recovery requires an FF field.
    """
    inp, r = _load(a.yaml)
    if r["ABDG"] is None:
        print("ERROR: non-SPD transverse-shear fit -- inspect the layup")
        return 2
    ABDG = np.asarray(r["ABDG"])
    S6 = np.linalg.inv(np.asarray(r["A6"]))
    if a.FF is not None:
        FF = np.asarray(a.FF, float)
        E6 = S6 @ FF[:6]
    else:
        eps = np.asarray(a.strain, float)
        FF = ABDG @ eps
        E6 = eps[:6]
    dE1 = None
    dE2 = None
    qt6 = np.asarray(a.qtop, float) if a.qtop else None
    qb6 = np.asarray(a.qbot, float) if a.qbot else None

    thick = inp["thick"]
    h = float(sum(thick))
    zpl = np.concatenate([[0.0], np.cumsum(thick)]) - inp["fraction"] * h
    zs = np.concatenate([np.linspace(zpl[k] + 1e-12, zpl[k + 1] - 1e-12,
                                     a.n_per_ply) for k in range(len(thick))])
    SM = []; EM = []; UU = []
    for z in zs:
        Gam, Sig, _ = msgrm_strain_at_depth(r, z, E6, dE1, dE2,
                                            qt6=qt6, qb6=qb6)
        w = msgrm_warping_at_depth(r, z, E6, dE1, dE2, qt6=qt6, qb6=qb6)
        SM.append(Sig)
        EM.append(Gam)
        UU.append([a.u2d[0] + w[0], a.u2d[1] + w[1], a.u2d[2] + w[2]])

    base = a.base or os.path.splitext(a.yaml)[0]
    d = os.path.dirname(os.path.abspath(base))
    if d and not os.path.isdir(d):
        os.makedirs(d)
    hdrs = "x3 " + " ".join("s%s" % c for c in ("11", "22", "33", "23", "13", "12"))
    np.savetxt(base + ".SM", np.column_stack([zs, np.array(SM)]),
               header="material-frame stress (ply axes); " + hdrs, fmt="%15.6e")
    np.savetxt(base + ".EM", np.column_stack([zs, np.array(EM)]),
               header="material-frame strain (ply axes); x3 e11 e22 e33 g23 g13 g12",
               fmt="%15.6e")
    np.savetxt(base + ".U", np.column_stack([zs, np.array(UU)]),
               header="displacement u_2d + warping; x3 U1 U2 U3", fmt="%15.6e")
    lines = ["OpenSG-RM dehomogenization  (%s)" % os.path.abspath(a.yaml),
             "FF     = [" + ", ".join("%g" % v for v in FF) + "]  (" +
             ", ".join(_FFL) + ")",
             "E6     = [" + ", ".join("%.6g" % v for v in E6) + "]"]
    if dE1 is not None:
        lines += ["dE1    = [" + ", ".join("%.6g" % v for v in dE1) + "]   (Q1 route)",
                  "dE2    = [" + ", ".join("%.6g" % v for v in dE2) + "]   (Q2 route)"]
    lines += ["qtop   = %s" % ("none" if qt6 is None else qt6.tolist()),
              "qbot   = %s" % ("none" if qb6 is None else qb6.tolist()),
              "u_2d   = %s" % (list(a.u2d),),
              "outputs: %s.SM / .EM / .U  (%d stations)" % (base, len(zs))]
    txt = "\n".join(lines)
    print(txt)
    with open(base + ".out", "w") as f:
        f.write(txt + "\n")
    return 0


def main(argv=None):
    """Console entry point: parse arguments and dispatch to the subcommand.

    In:
        argv: list of str or None (None -> sys.argv[1:]).
    Out:
        int: exit code returned by the subcommand handler.
    """
    ap = argparse.ArgumentParser(
        prog="opensg-rm-plate",
        description="OpenSG-RM plate: homogenize / plot / dehomogenize a "
                    "through-thickness 1-D SG YAML.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("homo", help="print the 8x8 ABDG (+ compliance)")
    p.add_argument("yaml")
    p.add_argument("--out", default=None, help="also write PREFIX_8x8.out")
    p.set_defaults(fn=cmd_homo)

    p = sub.add_parser("plot", help="write the SG mesh figure")
    p.add_argument("yaml")
    p.add_argument("--png", default=None)
    p.set_defaults(fn=cmd_plot)

    p = sub.add_parser("dehom", help="recover 3-D stress/strain/displacement")
    p.add_argument("yaml")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--FF", nargs=8, type=float, metavar=tuple(_FFL))
    g.add_argument("--strain", nargs=8, type=float, metavar=tuple(_ROWS))
    p.add_argument("--u2d", nargs=3, type=float, default=[0.0, 0.0, 0.0],
                   metavar=("U1", "U2", "U3"),
                   help="plate displacement at the point (added to the warping)")
    p.add_argument("--qtop", nargs=6, type=float, default=None,
                   metavar=("q", "q1", "q2", "q11", "q12", "q22"),
                   help="top-face pressure ladder (manual load case 4)")
    p.add_argument("--qbot", nargs=6, type=float, default=None,
                   metavar=("q", "q1", "q2", "q11", "q12", "q22"))
    p.add_argument("--n-per-ply", type=int, default=21, dest="n_per_ply")
    p.add_argument("--base", default=None,
                   help="output prefix (default: the YAML path without .yaml)")
    p.set_defaults(fn=cmd_dehom)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
