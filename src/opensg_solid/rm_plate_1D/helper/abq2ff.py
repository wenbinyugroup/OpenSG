"""abq2ff.py -- collect an Abaqus shell field .dat into the GENERAL plate
dehomogenization input: the .ff station table.

The .ff format (one row per in-plane station, whitespace-separated, '#'
comments; stations must fill a rectangular lattice for the FD gradients):

    x  y  N11 N22 N12 M11 M22 M12  Q1 Q2  [u1 u2 u3 wx wy]

    cols 1-2    station coordinates [m] (here: the TRUE integration-point
                coordinates from the Abaqus COORD table)
    cols 3-8    the FF: membrane + moment resultants [N/m, N]
    cols 9-10   transverse shear resultants [N/m] -- ONLY used as the
                per-station Q-consistency rescale targets, never inverted
    cols 11-15  OPTIONAL: mid-surface displacement u1 u2 u3 [m] and the
                slopes wx = dw/dx, wy = dw/dy at the station (here: nodal
                U bilinearly interpolated to the ip lattice, slopes by FD
                of nodal w on the corner grid).  Without them the driver
                writes warping-only displacement.

This is the Abaqus-side COLLECTOR.  A user with any other solver ("the
local way") writes the same table themselves -- the driver rm_dehom_ff.py
reads ONLY this file + layup_db.yaml (dehom: block) and never sees Abaqus.
The applied face-pressure ladder is NOT in the table: loads are input, and
they live in layup_db.yaml (dehom: loads:), not in solver output.

Run:  python abq2ff.py Abaqus_results/ex5_shell_S8R_field.dat
      -> Abaqus_results/ex5_shell_S8R_field.ff

The parsers are verbatim copies of rm_dehom_dat.py's (kept self-contained
so this file is portable next to any .dat).
"""
import argparse
import os
import re
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("dat", help="Abaqus whole-plate field .dat")
ap.add_argument("--db", default=None,
                help="layup_db.yaml (default: next to this script)")
args = ap.parse_args()
DAT = os.path.abspath(args.dat)
FF_OUT = os.path.splitext(DAT)[0] + ".ff"

DB = args.db or os.path.join(HERE, "layup_db.yaml")
db = yaml.safe_load(open(DB))
p = db["plate"]
A = float(p["a"])
NX, NY = int(p["nx"]), int(p["ny"])
dx = A / NX


# ---- parsers (verbatim from rm_dehom_dat.py) --------------------------------
def node_table(dat, nset, ncomp, nmax=None):
    """(n_nodes, ncomp), row i = node i+1; fully-constrained nodes are
    omitted by Abaqus and scatter in as zero."""
    rows, active, seen = [], False, False
    for ln in open(dat, errors="replace").read().splitlines():
        if ("NODE SET %s" % nset) in ln and "TABLE IS PRINTED" in ln:
            active, seen = True, False
            continue
        tok = ln.split()
        if not tok:
            continue
        if active and not seen:
            if tok[0] == "NODE":
                seen = True
            continue
        if active and seen:
            if re.fullmatch(r"\d+", tok[0]):
                v = [float(tok[0])]
                for t in tok[1:]:
                    try:
                        v.append(float(t))
                    except ValueError:
                        pass
                rows.append(v[:1 + ncomp])
            elif tok[0] in ("MAXIMUM", "MINIMUM"):
                active = False
    rows = np.array(rows)
    if rows.size == 0:
        raise RuntimeError("no NODE SET %s table in %s" % (nset, dat))
    ids = rows[:, 0].astype(int)
    out = np.zeros((max(nmax or 0, ids.max()), ncomp))
    out[ids - 1] = rows[:, 1:1 + ncomp]
    return out


def element_table_ip(dat, marker, ncomp, which=0):
    """One row PER INTEGRATION POINT, in file order: (ids, pts, values)."""
    ids, pts, vals = [], [], []
    active, seen, hit, done = False, False, -1, False
    for ln in open(dat, errors="replace").read().splitlines():
        if "THE FOLLOWING TABLE IS PRINTED" in ln:
            if done:
                break
            if marker in ln:
                hit += 1
            active, seen = (marker in ln and hit == which), False
            continue
        if not active:
            continue
        tok = ln.split()
        if not tok:
            continue
        if not seen:
            if tok[0] == "ELEMENT":
                seen = True
            continue
        if re.fullmatch(r"\d+", tok[0]):
            v = []
            for t in tok:
                try:
                    v.append(float(t))
                except ValueError:
                    pass
            ids.append(int(v[0]))
            pts.append(int(v[1]))
            vals.append(v[-ncomp:])
            done = True
        elif tok[0] in ("MAXIMUM", "MINIMUM"):
            break
    if not ids:
        raise RuntimeError("no '%s' ip table #%d in %s" % (marker, which,
                                                           dat))
    return np.array(ids), np.array(pts), np.array(vals)


def corner_ids():
    """Corner-node ids of the S8R half-index grid."""
    NID, k = {}, 0
    for J in range(2 * NY + 1):
        for I in range(2 * NX + 1):
            if (I % 2) + (J % 2) <= 1:
                k += 1
                NID[(I, J)] = k
    return np.array([[NID[(2 * i, 2 * j)] for i in range(NX + 1)]
                     for j in range(NY + 1)])


# ---- the per-ip resultant field on its own lattice --------------------------
print("parsing %s (per integration point) ..." % os.path.basename(DAT))
ids_r, pts_r, R_ip = element_table_ip(DAT, "INTEGRATION POINTS", 8, 0)
ids_c, pts_c, XY_ip = element_table_ip(DAT, "INTEGRATION POINTS", 3, 1)
assert np.array_equal(ids_r, ids_c) and np.array_equal(pts_r, pts_c), (
    "SF/SM and COORD tables are not row-aligned")
XY_ip = XY_ip[:, :2]

XL = np.unique(np.round(XY_ip[:, 0], 10))
YL = np.unique(np.round(XY_ip[:, 1], 10))
NLX, NLY = len(XL), len(YL)
assert NLX * NLY == len(ids_r), (
    "%d ip do not fill a %d x %d lattice" % (len(ids_r), NLX, NLY))
IX = np.searchsorted(XL, np.round(XY_ip[:, 0], 10))
IY = np.searchsorted(YL, np.round(XY_ip[:, 1], 10))
Rg = np.zeros((NLY, NLX, 8))
Rg[IY, IX] = R_ip

# Abaqus SF order SF1..SF5 SM1..SM3 = N11 N22 N12 Q1 Q2 M11 M22 M12
# -> .ff column order N11 N22 N12 M11 M22 M12 Q1 Q2
FFg = Rg[:, :, [0, 1, 2, 5, 6, 7, 3, 4]]

# ---- displacement pieces (nodal -> the ip lattice), as rm_dehom_dat.py ------
U_all = node_table(DAT, "NALL", 3)
ug = U_all[corner_ids() - 1]
wy_n, wx_n = np.gradient(ug[:, :, 2], dx, dx)


def bilerp(F):
    f = XL / dx
    i0 = np.clip(f.astype(int), 0, NX - 1)
    t = (f - i0)[None, :]
    Fi = (1 - t) * F[:, i0] + t * F[:, i0 + 1]
    f2 = YL / dx
    j0 = np.clip(f2.astype(int), 0, NY - 1)
    t2 = (f2 - j0)[:, None]
    return (1 - t2) * Fi[j0] + t2 * Fi[j0 + 1]


u0l = np.stack([bilerp(ug[:, :, c]) for c in range(3)], axis=-1)
wxl, wyl = bilerp(wx_n), bilerp(wy_n)

# ---- write the .ff table ----------------------------------------------------
Xl, Yl = np.meshgrid(XL, YL)
rows = np.column_stack([
    Xl.reshape(-1), Yl.reshape(-1),
    FFg.reshape(-1, 8),
    u0l.reshape(-1, 3), wxl.reshape(-1), wyl.reshape(-1)])
hdr = ("plate dehom FF station table (collected from %s)\n"
       "x[m] y[m] N11 N22 N12[N/m] M11 M22 M12[N] Q1 Q2[N/m] "
       "u1 u2 u3[m] wx wy[-]\n"
       "%d x %d rectangular station lattice, x-fastest"
       % (os.path.basename(DAT), NLX, NLY))
np.savetxt(FF_OUT, rows, fmt="%.17e", header=hdr)
print("wrote %s  (%d stations, 15 columns)" % (FF_OUT, len(rows)))
