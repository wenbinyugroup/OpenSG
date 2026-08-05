"""abaqus_3dfea.py -- layup_db.yaml  ->  layup_db_3dfea.inp

The 3-D FEA BENCHMARK for the same plate abaqus_inp.py builds as a shell.
Here nothing is homogenized: every ply is meshed as its own layer of solid
bricks carrying its real 3-D material properties and its own *ORIENTATION.
That is the point -- it is the reference the OpenSG-RM shell is judged
against, so it must share the geometry, the load and the time integration
and differ ONLY in how the wall is represented.

    abaqus_inp.py   -> 400 S8R    + one homogenized 8x8 section
    abaqus_3dfea.py -> 6400 C3D20 + 9 real materials, ply by ply

Every keyword this script writes, and why, is in abaqus_3dfea_README.md.

Run:  python abaqus_3dfea.py           the point history at the top centre
      python abaqus_3dfea.py field     one WHOLE-SOLID snapshot at the first
                                       deflection peak, for the 3-D field VTK
"""
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------
# ALL VARIABLES USED IN THIS SCRIPT
# ----------------------------------------------------------------------------
# paths and input
#   HERE        folder holding this script; every path is built from it
#   DB          path to layup_db.yaml, the same single input abaqus_inp.py uses
#   db          that YAML parsed into a dict
#   p           shorthand for db["plate"], the plate/analysis block
#   out         path of the deck written out, <DB stem>_3dfea.inp
#
# case data, all read from layup_db.yaml
#   A, B        plate side lengths in x and y [m]
#   NX, NY      C3D8I elements along x and along y (matched to the shell mesh)
#   Q0          peak pressure q0 [Pa] of the double-sine load
#   DT          fixed time increment of the *DYNAMIC step [s]
#   TTOT        total simulated time [s]
#   mats        per-ply material key, bottom ply first
#   thick       per-ply thickness [m], bottom first
#   angles      per-ply fibre angle [deg], bottom first
#   divs        per-ply number of solid elements through its thickness
#               (the layup's optional `divisions`, default 1)
#   mdb         material database: E[3], G[3], nu[3], rho per material, every
#               entry coerced to float (see the note where it is built)
#
# derived mesh
#   tlay        thickness of every SOLID layer after subdividing plies by
#               divs -- length NZT, and sum(tlay) = the laminate thickness
#   play        which PLY each solid layer came from, so it can be given that
#               ply's material and orientation
#   NZT         total solid layers through the thickness = sum(divs)
#   h           laminate thickness, sum(tlay)
#   zk          the NZT+1 node planes, centred: z runs -h/2 .. +h/2
#   dx, dy      in-plane element size, A/NX and B/NY [m]
#   npl         nodes per z-layer, (NX+1)*(NY+1)
#   n3(i,j,k)   node id    = 1 + i + (NX+1)*j + npl*k
#   e3(i,j,k)   element id = 1 + i + NX*j + NX*NY*k
#   i, j, k     in-plane and through-thickness indices
#   xc, yc      coordinates of the centre line, written into the deck as a
#               comment so NTOP3D says which point it is
#   ids, s      node list being written and the chunk start index
#   L           the deck itself: a list of text lines
#   kk, m, t, v throwaway loop/comprehension variables
# ----------------------------------------------------------------------------

# ---- input -----------------------------------------------------------------
DB = os.path.join(HERE, "layup_db.yaml")
db = yaml.safe_load(open(DB))
p = db["plate"]
A, B = float(p["a"]), float(p["b"])
NX, NY = int(p["nx"]), int(p["ny"])
if NX % 2 or NY % 2:
    raise ValueError("nx and ny must be EVEN (got %d, %d): the deflection "
                     "probe is the node at the plate centre, which only "
                     "exists when both are even" % (NX, NY))
Q0, DT, TTOT = float(p["q0"]), float(p["dt"]), float(p["ttot"])
FIELD = "FIELD" in [a.upper() for a in sys.argv[1:]]
FINC = int(p.get("field_inc", 53))       # snapshot increment (first peak)
if FIELD:
    # stop AT the snapshot: the field deck exists to dump one instant, so the
    # remaining 347 increments would only cost time
    TTOT = FINC * DT
ELTYPE = str(p.get("element", "C3D20R")).upper()
if ELTYPE not in ("C3D20R", "C3D20", "C3D8I"):
    raise ValueError("plate.element must be C3D20R, C3D20 or C3D8I, got %r"
                     % ELTYPE)
# NOTE: `fraction` is deliberately NOT read here.  The solid is centred on its
# own geometry and probes its own top face, so it has no need of the shell's
# reference-surface choice.  Keeping that dependency out means the benchmark
# cannot be perturbed by a modelling decision that belongs to the candidate.
# float() every number: PyYAML's float resolver demands an explicit exponent
# SIGN, so "128.0e9" in the YAML comes back as a str while "128.0e+9" would be
# a float.  Coercing here means the database can be written either way.
mdb = {k: {"E": [float(v) for v in m["E"]],
           "G": [float(v) for v in m["G"]],
           "nu": [float(v) for v in m["nu"]],
           "rho": float(m["rho"])}
       for k, m in db["materials"].items()}
mats = [str(q["material"]) for q in db["layup"]]
thick = [float(q["thickness"]) for q in db["layup"]]
angles = [float(q.get("angle", 0.0)) for q in db["layup"]]
divs = [int(q.get("divisions", 1)) for q in db["layup"]]

# ---- through-thickness mesh: subdivide each ply by its `divisions` ---------
tlay, play = [], []
for kk, (t, d) in enumerate(zip(thick, divs)):
    tlay += [t / d] * d
    play += [kk] * d
NZT = len(tlay)
h = float(sum(tlay))
# The solid is CENTRED ON ITS OWN GEOMETRY: z spans -h/2 .. +h/2, origin at the
# plate mid-plane.  That is the solid's natural origin and it does not depend on
# whatever reference surface the shell happens to use.  Where the origin sits
# changes no result anyway -- a displacement field is invariant under a rigid
# translation of the axes -- it just makes the coordinates say something.
zk = np.concatenate([[0.0], np.cumsum(tlay)]) - 0.5 * h
dx, dy = A / NX, B / NY
e3 = lambda i, j, k: 1 + i + NX * j + NX * NY * k

# ---- the node grid ---------------------------------------------------------
# Indices are HALVED: I in 0..2NX, J in 0..2NY, K in 0..2NZT, so a step of 1 is
# half an element and edge midpoints get integer coordinates too.
#
# C3D20 is a SERENDIPITY element -- 8 corners plus 12 edge midpoints, with NO
# face-centre and NO body-centre node.  So a grid point exists only where at
# most ONE of (I, J, K) is odd:
#     0 odd -> a corner        (needed by every element type)
#     1 odd -> an edge midpoint (quadratic only)
#     2 odd -> a face centre    -- does not exist
#     3 odd -> the body centre  -- does not exist
QUAD = ELTYPE.startswith("C3D20")


def exists(I, J, K):
    odd = (I % 2) + (J % 2) + (K % 2)
    return odd <= 1 if QUAD else odd == 0


def z_of(K):
    """z of a half-index plane: layer boundary if even, its midpoint if odd."""
    return zk[K // 2] if K % 2 == 0 else 0.5 * (zk[K // 2] + zk[K // 2 + 1])


NID = {}
nid = 0
for K in range(2 * NZT + 1):
    for J in range(2 * NY + 1):
        for I in range(2 * NX + 1):
            if exists(I, J, K):
                nid += 1
                NID[(I, J, K)] = nid
n3 = lambda i, j, k: NID[(2 * i, 2 * j, 2 * k)]      # corner-node shorthand

# Abaqus C3D20 ordering, as half-index offsets from the element's base corner:
# 8 corners (bottom CCW then top CCW), then the 4 bottom-face midsides, the 4
# top-face midsides, and LAST the 4 vertical-edge midsides.  Getting this order
# wrong still runs and silently distorts the element.
CORNERS = [(0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0),
           (0, 0, 2), (2, 0, 2), (2, 2, 2), (0, 2, 2)]
MIDSIDE = [(1, 0, 0), (2, 1, 0), (1, 2, 0), (0, 1, 0),
           (1, 0, 2), (2, 1, 2), (1, 2, 2), (0, 1, 2),
           (0, 0, 1), (2, 0, 1), (2, 2, 1), (0, 2, 1)]
OFFSETS = CORNERS + MIDSIDE if QUAD else CORNERS

print("3-D benchmark: %d x %d x %d = %d %s, %d nodes, %d dof"
      % (NX, NY, NZT, NX * NY * NZT, ELTYPE, len(NID), 3 * len(NID)))
print("through-thickness layers: %s"
      % ", ".join("%.4g mm" % (1e3 * t) for t in tlay))

# ---- nodes and elements ----------------------------------------------------
L = ["*HEADING",
     "3-D FEA benchmark from %s -- ply-by-ply %s"
     % (os.path.basename(DB), ELTYPE),
     "*NODE"]
for (I, J, K), n in sorted(NID.items(), key=lambda kv: kv[1]):
    L.append("%d, %.8f, %.8f, %.8f"
             % (n, 0.5 * I * dx, 0.5 * J * dy, z_of(K)))
L.append("*ELEMENT, TYPE=%s, ELSET=EALL" % ELTYPE)
for k in range(NZT):
    for j in range(NY):
        for i in range(NX):
            row = [e3(i, j, k)] + [NID[(2 * i + a, 2 * j + b, 2 * k + c)]
                                   for a, b, c in OFFSETS]
            # Abaqus allows at most 16 items per data line, so a 20-node
            # element needs a continuation line.  The TRAILING COMMA on the
            # first line is what tells Abaqus the element continues; without
            # it each line is read as a complete element and the 20-node
            # connectivity silently becomes 15 nodes + 5 zeros, which comes
            # back as "ELEMENT n NODE NUMBERS MUST BE POSITIVE".
            if len(row) > 16:
                L.append(", ".join(str(v) for v in row[:16]) + ",")
                L.append(", ".join(str(v) for v in row[16:]))
            else:
                L.append(", ".join(str(v) for v in row))

# one ELSET per solid layer, so each gets its ply's material and orientation
for k in range(NZT):
    L.append("*ELSET, ELSET=LAY%d, GENERATE" % (k + 1))
    L.append("%d, %d, 1" % (e3(0, 0, k), e3(NX - 1, NY - 1, k)))

# ---- the side faces (SS-1) and the centre probe ----------------------------
# Selected from the grid, so MIDSIDE nodes on those faces are included too --
# with a quadratic element, constraining only the corners would leave the face
# free to bulge between them.
def face_nodes(pred):
    return sorted(n for p, n in NID.items() if pred(p))


for nm, ids in (("FX0", face_nodes(lambda p: p[0] == 0)),
                ("FXA", face_nodes(lambda p: p[0] == 2 * NX)),
                ("FY0", face_nodes(lambda p: p[1] == 0)),
                ("FYB", face_nodes(lambda p: p[1] == 2 * NY))):
    L.append("*NSET, NSET=%s" % nm)
    for s in range(0, len(ids), 12):
        L.append(", ".join(str(v) for v in ids[s:s + 12]))
# ONE probe: the centre of the TOP surface, z = +h/2, following Nayak.
# It is the loaded face and the station he reports.
#
# The shell cannot match this by reading a node -- an RM shell has one w per
# point (eps33 = 0 by construction), and that w belongs to the reference
# surface.  Reaching z = +h/2 on the RM side means dehomogenizing, which is
# why the shell deck prints SF/SM on a centre patch.  Comparing the shell's
# NODE against this top face would charge it for the through-thickness
# compression it never claimed to model.
xc, yc = (NX // 2) * dx, (NY // 2) * dy
L.append("** NTOP3D -- the ONE deflection probe of this deck:")
L.append("**   centre of the TOP surface, x = %.6f  y = %.6f  z = %+.6f m"
         % (xc, yc, zk[NZT]))
L.append("**   (Nayak's station: the loaded face.  z is measured from the")
L.append("**    plate mid-plane, so the top face is at +h/2 = %+.6f m.)"
         % (0.5 * h))
L.append("**   The shell's counterpart is NOT its node -- it must be")
L.append("**   dehomogenized to this same z from its PATCHC resultants.")
L.append("*NSET, NSET=NTOP3D")
L.append("%d" % n3(NX // 2, NY // 2, NZT))

# The four TOP-LAYER elements that meet at that node.  Their stresses are
# printed with POSITION=NODES, so the values are extrapolated to the element
# nodes and the centre node is sampled AT z = +h/2 exactly -- not at the
# outermost integration point, which sits inside the ply and would have to be
# reported at a different depth than the shell is recovered at.
# Four elements share the node; they are averaged in post-processing, and their
# spread is a discretisation check that costs nothing to look at.
ETOPC = [e3(i, j, NZT - 1)
         for j in (NY // 2 - 1, NY // 2) for i in (NX // 2 - 1, NX // 2)]
L.append("** ETOPC -- the 4 top-layer elements meeting at NTOP3D; stresses are")
L.append("**   extrapolated to nodes so the centre node is read at z = +h/2.")
L.append("**   At that point sigma33 must equal MINUS the applied pressure")
L.append("**   (-%.6e Pa): a traction boundary condition, hence an exact check."
         % Q0)
L.append("*ELSET, ELSET=ETOPC")
L.append(", ".join(str(v) for v in ETOPC))

# ---- materials, orientations, sections -------------------------------------
for m in dict.fromkeys(mats):
    md = mdb[m]
    L.append("*MATERIAL, NAME=%s" % m.upper())
    L.append("*ELASTIC, TYPE=ENGINEERING CONSTANTS")
    L.append("%.6e, %.6e, %.6e, %.4g, %.4g, %.4g, %.6e, %.6e,"
             % (md["E"][0], md["E"][1], md["E"][2],
                md["nu"][0], md["nu"][1], md["nu"][2],
                md["G"][0], md["G"][1]))
    L.append("%.6e" % md["G"][2])
    L.append("*DENSITY")
    L.append("%g," % md["rho"])
for k in range(NZT):
    L.append("*ORIENTATION, NAME=OR%d, SYSTEM=RECTANGULAR" % (k + 1))
    L.append("1.0, 0.0, 0.0, 0.0, 1.0, 0.0")
    L.append("3, %g" % angles[play[k]])
    L.append("*SOLID SECTION, ELSET=LAY%d, MATERIAL=%s, ORIENTATION=OR%d"
             % (k + 1, mats[play[k]].upper(), k + 1))
    L.append(",")

# ---- boundary conditions ---------------------------------------------------
L.append("** SS-1 on the side FACES; solids have no rotational dof, so there")
L.append("** is no drilling constraint and no edge-rotation choice to make")
L.append("*BOUNDARY")
for card in ("FX0, 2, 3", "FXA, 2, 3", "FY0, 1, 1", "FYB, 1, 1",
             "FY0, 3, 3", "FYB, 3, 3"):
    L.append(card)

# ---- step ------------------------------------------------------------------
L.append("*STEP, NAME=PULSE, INC=%d" % int(2 * TTOT / DT))
# DIRECT is ESSENTIAL here, not cosmetic: with the 4-parameter *DYNAMIC form
# Abaqus ran this solid on automatic incrementation -- 437 increments over 40
# distinct dt, down to 1.6e-7 ms at the start -- while the shell happened to
# hold a uniform 50 us.  The two histories then cannot be compared point for
# point.  DIRECT pins dt for both.
L.append("*DYNAMIC, DIRECT")
L.append("%g, %g" % (DT, TTOT))
# Only the TOP SURFACE is loaded: P2 is the top face (nodes 5-8) of the TOP
# layer, k = NZT-1, so 400 of the 6400 elements carry pressure.
#
# The pressure is sampled at the CENTROID OF THAT TOP FACE, taken directly from
# the face's own four nodes rather than from the element.  On this prismatic
# mesh the two happen to share x and y (verified: max difference 1.1e-16 m over
# all 400 loaded elements) -- but they would NOT on a tapered or skewed solid,
# and averaging the face nodes is right either way.
def top_face_centre(i, j):
    """(x, y) of the centroid of element (i, j, NZT-1)'s TOP face, averaged
    over its four top nodes -- the surface the pressure actually acts on."""
    n = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
    return (sum(a * dx for a, _ in n) / 4.0,
            sum(b * dy for _, b in n) / 4.0)


L.append("*DLOAD")                  # positive P2 acts INTO the element, i.e. -z
for j in range(NY):
    for i in range(NX):
        xf, yf = top_face_centre(i, j)
        q = Q0 * np.sin(np.pi * xf / A) * np.sin(np.pi * yf / B)
        L.append("%d, P2, %.6e" % (e3(i, j, NZT - 1), q))
if FIELD:
    # the WHOLE solid, once, at the snapshot increment.  CENTROIDAL gives one
    # value per element, i.e. a 20 x 20 x 16 cell-centred field -- the natural
    # lattice for a volume plot, and it sidesteps having to match C3D20's 27
    # Gauss points against anything.  COORD comes in its own block: merging it
    # with S makes the .dat one table and breaks the reshape.
    L.append("*NSET, NSET=NALL3D, GENERATE")
    L.append("1, %d, 1" % nid)
    L.append("*NODE PRINT, NSET=NALL3D, FREQUENCY=%d" % FINC)
    L.append("U")
    L.append("*EL PRINT, ELSET=EALL, POSITION=CENTROIDAL, FREQUENCY=%d" % FINC)
    L.append("S,")
    L.append("*EL PRINT, ELSET=EALL, POSITION=CENTROIDAL, FREQUENCY=%d" % FINC)
    L.append("COORD,")
else:
    L.append("*NODE PRINT, NSET=NTOP3D, FREQUENCY=1")
    L.append("U")
    # all six stress components at the top-surface centre, every increment
    L.append("*EL PRINT, ELSET=ETOPC, POSITION=NODES, FREQUENCY=1")
    L.append("S,")
L.append("*END STEP")

# ---- output ----------------------------------------------------------------
out = os.path.splitext(DB)[0] + "_3dfea%s.inp" % ("_field" if FIELD else "")
open(out, "w").write("\n".join(L) + "\n")
print("%s -> %s" % (os.path.basename(DB), os.path.basename(out)))
