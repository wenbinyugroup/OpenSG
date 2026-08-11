"""abaqus_inp.py -- layup_db.yaml -> <name>_static_S8R.inp: the STATIC
cylindrical-bending strip deck of this case (Pagano / Yu sec. 6.1 setting),
on the ex5 deck architecture with three deliberate deviations, all data-driven
from the plate: block:

    static: true       *STATIC replaces *DYNAMIC, DIRECT
    load_mode: cyl     q = q0 sin(pi x / a) -- single sine, no y factor
    (plane strain)     v = 0 and UR1 = 0 on ALL nodes: the strip bends
                       cylindrically, nothing varies along y

Everything else IS ex5: S8R on the half-index grid (midside nodes in every
edge set), one homogenized section via the core rm_homo (*SHELL GENERAL
SECTION + *TRANSVERSE SHEAR STIFFNESS, no plies in the model), and the
whole-plate field prints (U at NALL, SF/SM + COORD at EALL) that
rm_dehom_dat.py consumes.  See abaqus_inp_README.md for every keyword.

Run:  python abaqus_inp.py

# ----------------------------------------------------------------------------
# ALL VARIABLES USED IN THIS SCRIPT
# ----------------------------------------------------------------------------
#   HERE, ROOT      this case folder; the repo root (walk up to opensg_jax/)
#   d, db, p        the parsed layup_db (core loader) and its plate block
#   A, B, NX, NY    span, strip width, elements along x and y
#   Q0              load amplitude q0 (net q for the Yu split load)
#   SELT            shell element (S8R)
#   r, AB, G2       core homogenization result, 6x6 ABD, 2x2 shear block
#   rho_h           section mass per area (write it even though static)
#   dx, dy          element sizes;  e(i, j)  element id
#   exists, NID, n  the half-index serendipity node grid and corner lookup
#   CORN, MIDS, OFFS  the S8R connectivity offsets
#   L               the deck lines;  xc  element-centre x for the load
#   edge(pred)      node filter for the X0 / XA / NALL sets
#   out             <name>_static_S8R.inp
# ----------------------------------------------------------------------------
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
try:
    import opensg_solid                      # pip install -e . -- nothing to do
except ImportError:                          # fall back to the in-repo source tree
    ROOT = HERE
    while not os.path.isdir(os.path.join(ROOT, "src", "opensg_solid")):
        parent = os.path.dirname(ROOT)
        if parent == ROOT:                   # hit the filesystem root
            raise ImportError(
                "opensg_solid not installed and no src/ found above " + HERE)
        ROOT = parent
    sys.path.insert(0, os.path.join(ROOT, "src"))

import time as _t
print("start: " + _t.strftime("%Y-%m-%d %H:%M:%S"))

from opensg_solid.rm_plate_1D.rm_homo import load_layup_db, homogenize_layup_db

DB = os.path.join(HERE, "layup_db.yaml")
d = load_layup_db(DB)
p = d["db"]["plate"]
A, B = float(p["a"]), float(p["b"])
NX, NY = int(p["nx"]), int(p["ny"])
Q0 = float(p["q0"])
SELT = str(p.get("shell_element", "S8R")).upper()
assert p.get("static", False) and p.get("load_mode") == "cyl", (
    "this deck generator is the STATIC cylindrical-bending variant")

# the section: homogenize through the core (writes 1dsg.yaml + homo .out too)
r = homogenize_layup_db(DB)
AB = np.asarray(r["A6"])
G2 = np.asarray(r["ABDG"])[6:8, 6:8]
rho_h = sum(d["material_db"][m]["rho"] * t
            for m, t in zip(d["layup"]["mat_names"], d["layup"]["thick"]))

# ---- the S8R half-index mesh (as in ex5) ------------------------------------
dx, dy = A / NX, B / NY
e = lambda i, j: 1 + i + NX * j


def exists(I, J):
    return (I % 2) + (J % 2) <= 1              # serendipity: no face centres


NID, _k = {}, 0
for J in range(2 * NY + 1):
    for I in range(2 * NX + 1):
        if exists(I, J):
            _k += 1
            NID[(I, J)] = _k
CORN = [(0, 0), (2, 0), (2, 2), (0, 2)]
MIDS = [(1, 0), (2, 1), (1, 2), (0, 1)]
OFFS = CORN + MIDS

L = ["*HEADING",
     "OpenSG-RM static cylindrical bending, %s" % d["db"].get("name", "case"),
     "*NODE"]
for (I, J), nid in sorted(NID.items(), key=lambda kv: kv[1]):
    L.append("%d, %.8f, %.8f, 0.0" % (nid, 0.5 * I * dx, 0.5 * J * dy))
L.append("*ELEMENT, TYPE=%s, ELSET=EALL" % SELT)
for j in range(NY):
    for i in range(NX):
        row = [e(i, j)] + [NID[(2 * i + a_, 2 * j + b_)] for a_, b_ in OFFS]
        L.append(", ".join(str(v) for v in row))


def edge(pred):
    return sorted(v for k, v in NID.items() if pred(k))


for nm, ids in (("X0", edge(lambda k: k[0] == 0)),
                ("XA", edge(lambda k: k[0] == 2 * NX))):
    L.append("*NSET, NSET=%s" % nm)
    for s in range(0, len(ids), 12):
        L.append(", ".join(str(v) for v in ids[s:s + 12]))
L.append("*NSET, NSET=NALL, GENERATE")
L.append("1, %d, 1" % len(NID))

L.append("*SHELL GENERAL SECTION, ELSET=EALL, DENSITY=%.6g" % rho_h)
tri = [AB[i, j] for j in range(6) for i in range(j + 1)]
for s in range(0, len(tri), 8):
    L.append(", ".join("%.6e" % v for v in tri[s:s + 8]))
L.append("*TRANSVERSE SHEAR STIFFNESS")
L.append("%.6e, %.6e, %.6e" % (G2[0, 0], G2[1, 1], G2[0, 1]))

L.append("** CYLINDRICAL BENDING: SS at x = 0 and x = a (w = 0), one axial")
L.append("** anchor (u = 0 at x = 0), and PLANE STRAIN on every node:")
L.append("** v = 0 (dof 2), UR1 = 0 (dof 4, twist about x), drilling fixed.")
L.append("*BOUNDARY")
for card in ("X0, 3, 3", "XA, 3, 3", "X0, 1, 1",
             "NALL, 2, 2", "NALL, 4, 4", "NALL, 6, 6"):
    L.append(card)

L.append("*STEP, NAME=STATIC")
L.append("*STATIC")
L.append("1.0, 1.0")
L.append("*DLOAD")                  # single sine, sampled at element centres
for j in range(NY):
    for i in range(NX):
        q = Q0 * np.sin(np.pi * (i + 0.5) * dx / A)
        L.append("%d, P, %.6e" % (e(i, j), q))
# the whole-strip dump rm_dehom_dat.py consumes (one static increment)
L.append("*NODE PRINT, NSET=NALL, FREQUENCY=1")
L.append("U")
L.append("*EL PRINT, ELSET=EALL, FREQUENCY=1")
L.append("SF, SM")
L.append("*EL PRINT, ELSET=EALL, FREQUENCY=1")
L.append("COORD")
L.append("*END STEP")

out = os.path.join(HERE, "%s_static_%s.inp" % (d["db"].get("name", "case"),
                                               SELT))
open(out, "w").write("\n".join(L) + "\n")
print("%s -> %s  (%d %s, %d nodes)" % (os.path.basename(DB),
                                       os.path.basename(out), NX * NY, SELT,
                                       len(NID)))
print("end:   " + _t.strftime("%Y-%m-%d %H:%M:%S"))
