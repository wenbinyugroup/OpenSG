"""abq_rpt.py -- read an Abaqus .rpt (the element/node tables the deck
dumpers write) and extract fields along a path as OpenSG .SM / .U
files.

The .rpt is the COMMON INTERMEDIATE between Abaqus and OpenSG: one
script touches the odb once (abaqus python, e.g. dump_3d_rpt.py), and
everything downstream -- path sampling, comparison, plotting -- runs in
a normal python against the text file.  That keeps the odb-dependent
surface to a single dumper per job type.

Expected layout (written by the dumpers; `**` lines are comments):

    ** ---------------- ELEMENT TABLE ----------------
    Element  Xc Yc Zc  S11 S22 S33 S12 S13 S23
    ** ---------------- NODE TABLE ----------------
    Node     X  Y  Z   U1 U2 U3

Sampling is ELEMENT-WISE: a centroidal stress is piecewise constant, so
a path point takes the value of the NEAREST element centroid (for the
tet sizes here that is the containing element to raster accuracy, and
the chosen element per point is returned so it can be audited).  U is
taken from the nearest node.  Outputs follow the rm_plate_1D .SM/.U
text layout: x y z + components, print order 11 22 33 12 13 23.

names
  read_rpt(path)              -> {"elem": (labels, xyz, S),
                                  "node": (labels, xyz, U)}
  sample_path(tabs, P)        -> dict of sampled arrays + element ids
  write_path_fields(...)      -> <prefix>.SM, <prefix>.U,
                                 <prefix>_elements.csv
"""
import os

import numpy as np


def read_rpt(path):
    """Parse the two tables of a deck-dumper .rpt.

    In:  path str
    Out: dict {"elem": (labels (E,), xyz (E,3), S (E,6)),
               "node": (labels (N,), xyz (N,3), U (N,3))}."""
    kind = None
    rows = {"elem": [], "node": []}
    for ln in open(path):
        s = ln.strip()
        if not s:
            continue
        if s.startswith("**"):
            if "ELEMENT TABLE" in s:
                kind = "elem"
            elif "NODE TABLE" in s:
                kind = "node"
            continue
        v = s.split()
        if not v[0].lstrip("-").isdigit():
            continue                       # header row
        if kind:
            rows[kind].append([float(x) for x in v])
    out = {}
    for k, w in (("elem", 6), ("node", 3)):
        a = np.array(rows[k])
        if not len(a):
            out[k] = (np.zeros(0, np.int64), np.zeros((0, 3)),
                      np.zeros((0, w)))
            continue
        out[k] = (a[:, 0].astype(np.int64), a[:, 1:4], a[:, 4:4 + w])
    return out


def sample_path(tabs, P):
    """Fields along a path, element-wise.

    In:  tabs -- read_rpt output; P (n, 4) -- rows [s y1 y2 y3]
    Out: dict {s (n,), xyz (n,3), S (n,6), U (n,3), elem (n,),
         n_distinct int}."""
    elab, exyz, ES = tabs["elem"]
    nlab, nxyz, NU = tabs["node"]
    pts = P[:, 1:4]
    S = np.zeros((len(P), 6))
    U = np.zeros((len(P), 3))
    eid = np.zeros(len(P), np.int64)
    for i, p in enumerate(pts):
        k = int(np.argmin(np.einsum("ij,ij->i", exyz - p, exyz - p)))
        S[i] = ES[k]
        eid[i] = elab[k]
        j = int(np.argmin(np.einsum("ij,ij->i", nxyz - p, nxyz - p)))
        U[i] = NU[j]
    nd = 1 + int(np.sum(eid[1:] != eid[:-1]))
    return {"s": P[:, 0], "xyz": pts, "S": S, "U": U, "elem": eid,
            "n_distinct": nd}


def write_path_fields(tabs, P, prefix, title=""):
    """Sample a path and write <prefix>.SM / .U / _elements.csv in the
    rm_plate_1D text layout.

    In:  tabs -- read_rpt output; P (n,4) [s y1 y2 y3]; prefix str;
         title str -- first header line
    Out: the sample_path dict (also written to disk)."""
    r = sample_path(tabs, P)
    hdr = ("%s\n lattice: path points; z = 0 is the reference surface"
           % (title or "path fields"))
    cols6 = "".join(" %14s" % c for c in
                    ("S11[Pa]", "S22[Pa]", "S33[Pa]",
                     "S12[Pa]", "S13[Pa]", "S23[Pa]"))
    cols3 = "".join(" %14s" % c for c in ("U1[m]", "U2[m]", "U3[m]"))
    np.savetxt(prefix + ".SM", np.hstack([r["xyz"], r["S"]]),
               fmt="%14.6e",
               header=hdr + "\n%12s %14s %14s" % ("x[m]", "y[m]", "z[m]")
               + cols6)
    np.savetxt(prefix + ".U", np.hstack([r["xyz"], r["U"]]),
               fmt="%14.6e",
               header=hdr + "\n%12s %14s %14s" % ("x[m]", "y[m]", "z[m]")
               + cols3)
    with open(prefix + "_elements.csv", "w") as f:
        f.write("s,x,y,z,element\n")
        for s, p, e in zip(r["s"], r["xyz"], r["elem"]):
            f.write("%.6e,%.6e,%.6e,%.6e,%d\n" % (s, p[0], p[1], p[2], e))
    print("wrote %s.SM / .U / _elements.csv  (%d points, %d distinct"
          " elements)" % (prefix, len(P), r["n_distinct"]))
    return r
