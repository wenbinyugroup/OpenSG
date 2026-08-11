"""abaqus_inp.py -- layup_db.yaml + 1dsg.yaml  ->  layup_db_abaqus.inp

The plate law is homogenized here from the SG that 1d_sg.py wrote, rather
than parsed back out of its .out -- one rm_plate_msg call is cheaper than
carrying a text parser and keeping it in step with the .out layout, and the
deck gets full precision instead of the printed six figures.

A plain Abaqus S4 plate whose ONLY constitutive input is the homogenized
plate law: the 6x6 goes in as *SHELL GENERAL SECTION (DENSITY = rho h) and,
when model = 1, the 2x2 MSG block as *TRANSVERSE SHEAR STIFFNESS.  No plies,
no orientations, no shear correction factor anywhere in the deck.

Simply supported (SS-1) on all four edges, drilling dof fixed (flat plate),
double-sine pressure q0 sin(pi x/a) sin(pi y/b) held over the step, implicit
*DYNAMIC at a fixed increment.  Printed every increment: the centre-node
deflection, and SF/SM + COORD on a 2x2 patch at the centre so the RM 3-D
recovery can reach any depth -- in particular the TOP SURFACE, which is where
Nayak reports and which a shell node cannot give directly.

Every keyword this script writes, and why each data line looks the way it
does, is documented in abaqus_inp_README.md -- read that rather than
expecting the explanations in here.

Run:  python abaqus_inp.py
"""
import os
import sys

import numpy as np
import yaml

# ----------------------------------------------------------------------------
# ALL VARIABLES USED IN THIS SCRIPT
# ----------------------------------------------------------------------------
# paths and input
#   HERE        folder holding this script; every path is built from it
#   ROOT        repo root, found by walking up until opensg_jax/ appears
#   DB          path to layup_db.yaml, the single user input file
#   db          that YAML parsed into a dict
#   p           shorthand for db["plate"], the plate/analysis block
#   out         path of the deck written out, <DB stem>_abaqus.inp
#
# case data, all read from layup_db.yaml (nothing is defaulted in code)
#   model       0 = classical, write the ABD only
#               1 = shear-refined, also write *TRANSVERSE SHEAR STIFFNESS
#   fraction    reference-surface location as a fraction of thickness:
#               0 = bottom/OML face, 0.5 = mid-surface, 1 = top.  It fixes
#               where the shell's z = 0 plane sits in the laminate.
#   A, B        plate side lengths in x and y [m]
#   NX, NY      number of S4 elements along x and along y (so NX*NY elements
#               and (NX+1)*(NY+1) nodes)
#   Q0          peak pressure q0 [Pa] of the double-sine load
#   DT          fixed time increment of the *DYNAMIC step [s]
#   TTOT        total simulated time [s]
#
# the homogenized section
#   inp         1dsg.yaml parsed back: thick / angles / mat_names /
#               material_db / n_per_layer / elem_order / node_x
#   r           the rm_plate_msg result dict
#   AB          6x6 classical ABD, r["A6"]; rows e11,e22,g12,k11,k22,k12
#   G2          2x2 MSG transverse-shear block, ABDG[6:8, 6:8]; None if
#               model = 0
#   rho_h       section mass PER UNIT AREA, sum(rho_k * t_k) [kg/m^2] --
#               this is what *SHELL GENERAL SECTION's DENSITY wants, not rho
#   tri         the 21 upper-triangle terms of AB, column by column, in the
#               order Abaqus requires
#
# mesh construction
#   dx, dy      element size, A/NX and B/NY [m]
#   n(i, j)     node id   = 1 + i + (NX+1)*j   (x fastest)
#   e(i, j)     element id = 1 + i + NX*j
#   c           NX//2, the centre index; the PATCHC elset straddles it
#   xc, yc      coordinates of the centre node, written into the deck as a
#               comment so NCEN_REF says which point it is
#   i, j        in-plane element/node indices
#   nm, ids     name and node list of each edge *NSET being written
#   s           start index when chunking a long list across data lines
#   card        one *BOUNDARY data line, "nset, first_dof, last_dof"
#   q           the pressure on one element, q0 sin(pi x/a) sin(pi y/b) at
#               that element's centre [Pa]
#   L           the deck itself: a list of text lines, joined and written at
#               the end
#   m, t, v     throwaway comprehension variables: a material key, a ply
#               thickness, and whichever id or stiffness term is being
#               formatted onto the current data line
# ----------------------------------------------------------------------------

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

from opensg_solid.rm_plate_1D.segment_plate import read_plate_sg_yaml
from opensg_solid.rm_plate_1D.msg_rm_plate import rm_plate_msg


def main():
    """Build the homogenized-section Abaqus shell deck from layup_db.yaml + 1dsg.yaml.

    In:
        none -- parses sys.argv (element override, FIELD flag) and reads
        layup_db.yaml and 1dsg.yaml next to this file.
    Out:
        None -- writes <DB stem>_abaqus_<elt>[_field].inp.
    """
    # ---- input -----------------------------------------------------------------
    DB = os.path.join(HERE, "layup_db.yaml")
    db = yaml.safe_load(open(DB))
    model = int(db["model"])                 # 0 = ABD only, 1 = ABD + shear
    fraction = float(db["fraction"])         # no code-side default: the YAML is
                                             # the ONLY place these are set
    p = db["plate"]
    A, B = float(p["a"]), float(p["b"])
    NX, NY = int(p["nx"]), int(p["ny"])
    if NX % 2 or NY % 2:                     # NCEN is node n(NX//2, NY//2), which
        raise ValueError(                    # only lands ON the centre if both are
            "nx and ny must be EVEN (got %d, %d): the deflection probe NCEN is "
            "node n(nx//2, ny//2), and with an odd count there is no node at the "
            "plate centre -- the floor silently puts the probe half an element "
            "off (nx=21 would probe x = 0.7257 m instead of a/2 = 0.762 m)."
            % (NX, NY))
    Q0, DT, TTOT = float(p["q0"]), float(p["dt"]), float(p["ttot"])
    ARGS = [a.upper() for a in sys.argv[1:]]
    FIELD = "FIELD" in ARGS                  # whole-plate snapshot instead of the
    ARGS = [a for a in ARGS if a != "FIELD"]  # centre-patch history
    SELT = str(p.get("shell_element", "S4")).upper()
    if ARGS:                                 # override, for element studies; the
        SELT = ARGS[0]                       # deck is named after the element so
                                             # one never overwrites another
    FINC = int(p.get("field_inc", 53))       # the snapshot increment
    if FIELD:
        # stop AT the snapshot: the field deck exists to dump one instant, so
        # running the remaining increments would only cost time
        TTOT = FINC * DT
    if SELT not in ("S4", "S8R", "S9R5"):
        raise ValueError("plate.shell_element must be S4, S8R or S9R5, got %r"
                         % SELT)
    # S9R5 is a FIVE-dof thin shell: it imposes the Kirchhoff constraint and has no
    # transverse-shear flexibility and no drilling dof.  For this sandwich the core
    # is 0.95 h of soft foam, so transverse shear is the dominant compliance -- the
    # very term the MSG 2x2 block supplies.  S8R is the SIX-dof quadratic THICK
    # shell and is the right quadratic counterpart.  Both are allowed here so the
    # difference can be measured rather than argued about.
    NDOF5 = SELT == "S9R5"

    # the section law: homogenize the SG that 1d_sg.py wrote.  n_per_layer and
    # elem_order come back from the mesh file itself, so the section can never
    # disagree with the mesh it was built on.
    inp = read_plate_sg_yaml(os.path.join(HERE, "1dsg.yaml"))
    r = rm_plate_msg(inp["thick"], inp["angles"], inp["mat_names"],
                     inp["material_db"], n_per_layer=inp["n_per_layer"],
                     elem_order=inp["elem_order"], fraction=fraction)
    AB = np.asarray(r["A6"])
    G2 = np.asarray(r["ABDG"])[6:8, 6:8] if model == 1 else None
    rho_h = sum(inp["material_db"][m]["rho"] * t
                for m, t in zip(inp["mat_names"], inp["thick"]))
    print("model %d: %s -- the deck %s the *TRANSVERSE SHEAR STIFFNESS card"
          % (model, "shear-refined" if model == 1 else "classical",
             "carries" if model == 1 else "omits"))
    print("section from 1dsg.yaml, reference fraction %g, rho*h = %.4f kg/m^2"
          % (fraction, rho_h))

    # ---- mesh helpers ----------------------------------------------------------
    dx, dy = A / NX, B / NY
    e = lambda i, j: 1 + i + NX * j                  # element id

    # Nodes live on a HALF-INDEX grid, I = 0..2*NX, so a step of 1 is half an
    # element.  Which half-index nodes exist is exactly what distinguishes the
    # three elements:
    #   S4    both indices even            -> corners only,            4 / element
    #   S8R   at most one index odd        -> + edge midpoints,        8 / element
    #   S9R5  every index                  -> + the face centre,       9 / element
    # S8R is SERENDIPITY, so it has no face-centre node; S9R5 is LAGRANGE and does.
    def exists(I, J):
        odd = (I % 2) + (J % 2)
        return odd == 0 if SELT == "S4" else (odd <= 1 if SELT == "S8R" else True)


    NID, _k = {}, 0
    for J in range(2 * NY + 1):
        for I in range(2 * NX + 1):
            if exists(I, J):
                _k += 1
                NID[(I, J)] = _k
    n = lambda i, j: NID[(2 * i, 2 * j)]             # corner node of the (i,j) grid

    # Abaqus 2-D order: 4 corners counter-clockwise, then the midside nodes
    # 5=mid(1,2) 6=mid(2,3) 7=mid(3,4) 8=mid(4,1), then 9 = face centre.
    CORN = [(0, 0), (2, 0), (2, 2), (0, 2)]
    MIDS = [(1, 0), (2, 1), (1, 2), (0, 1)]
    OFFS = (CORN if SELT == "S4" else
            CORN + MIDS if SELT == "S8R" else CORN + MIDS + [(1, 1)])

    L = ["*HEADING",
         "OpenSG-RM plate from %s (model %d)" % (os.path.basename(DB), model),
         "*NODE"]
    for (I, J), nid in sorted(NID.items(), key=lambda kv: kv[1]):
        L.append("%d, %.8f, %.8f, 0.0" % (nid, 0.5 * I * dx, 0.5 * J * dy))
    L.append("*ELEMENT, TYPE=%s, ELSET=EALL" % SELT)
    for j in range(NY):
        for i in range(NX):
            row = [e(i, j)] + [NID[(2 * i + a, 2 * j + b)] for a, b in OFFS]
            L.append(", ".join(str(v) for v in row))
    # Edge sets are filtered from the grid, so with a quadratic element the MIDSIDE
    # nodes on each edge are constrained too -- pinning only the corners would let
    # the edge bulge between them and read as a softer plate.
    edge = lambda pred: sorted(v for k, v in NID.items() if pred(k))
    for nm, ids in (("NX0", edge(lambda k: k[0] == 0)),
                    ("NXA", edge(lambda k: k[0] == 2 * NX)),
                    ("NY0", edge(lambda k: k[1] == 0)),
                    ("NYB", edge(lambda k: k[1] == 2 * NY))):
        L.append("*NSET, NSET=%s" % nm)
        for s in range(0, len(ids), 12):
            L.append(", ".join(str(v) for v in ids[s:s + 12]))
    L.append("*NSET, NSET=NALL, GENERATE")
    L.append("1, %d, 1" % len(NID))
    xc, yc = (NX // 2) * dx, (NY // 2) * dy
    L.append("** NCEN_REF -- the deflection probe:")
    L.append("**   centre node, x = %.6f  y = %.6f, ON THE REFERENCE SURFACE"
             % (xc, yc))
    L.append("**   (fraction = %g of the thickness from the bottom face)." % fraction)
    L.append("**   An RM shell has ONE w per point (eps33 = 0), so this is NOT a")
    L.append("**   top- or bottom-surface value.  For the top surface, dehomogenize")
    L.append("**   using the PATCHC resultants below.")
    L.append("*NSET, NSET=NCEN_REF")
    L.append("%d" % n(NX // 2, NY // 2))
    # A 2x2 element patch straddling the centre.  A shell node gives ONE w, on the
    # reference surface; to say anything about the TOP surface the RM route has to
    # dehomogenize, and that needs the section resultants plus their in-plane
    # gradients -- which is what a patch (rather than a single element) provides,
    # by least-squares fitting SF/SM over its integration points.
    c = NX // 2
    L.append("*ELSET, ELSET=PATCHC")
    L.append(", ".join(str(e(i, j)) for j in (c - 1, c) for i in (c - 1, c)))

    # ---- the section: the 21 upper-triangle terms, COLUMN by column ------------
    L.append("*SHELL GENERAL SECTION, ELSET=EALL, DENSITY=%.6g" % rho_h)
    tri = [AB[i, j] for j in range(6) for i in range(j + 1)]
    for s in range(0, len(tri), 8):
        L.append(", ".join("%.6e" % v for v in tri[s:s + 8]))
    if model == 1:                                   # K11, K22, K12
        L.append("*TRANSVERSE SHEAR STIFFNESS")
        L.append("%.6e, %.6e, %.6e" % (G2[0, 0], G2[1, 1], G2[0, 1]))

    L.append("** SS-1: v=w=0 on the x-edges, u=w=0 on the y-edges")
    L.append("*BOUNDARY")
    cards = ["NX0, 2, 3", "NXA, 2, 3", "NY0, 1, 1", "NYB, 1, 1",
             "NY0, 3, 3", "NYB, 3, 3"]
    if NDOF5:
        # S9R5 carries FIVE dof per node -- two in-surface rotations, no rotation
        # about the normal -- so there is no drilling dof to fix and dof 6 does not
        # exist.  Asking for it is an Abaqus error, not a harmless extra card.
        L.append("** %s has 5 dof/node: no drilling dof exists, so none is fixed"
                 % SELT)
    else:
        L.append("** drilling dof: no strain energy on a flat shell, so it is fixed")
        L.append("** to remove the singularity; its reactions are identically zero")
        cards.append("NALL, 6, 6")
    for card in cards:
        L.append(card)

    L.append("*STEP, NAME=PULSE, INC=%d" % int(2 * TTOT / DT))
    # DIRECT = no automatic incrementation.  The 4-parameter form only CAPS dt;
    # Abaqus still cuts back to control the half-increment residual, and then the
    # two models no longer share increments (the solid did exactly this: 437
    # increments over 40 distinct dt).  DIRECT takes just dt and the period.
    L.append("*DYNAMIC, DIRECT")
    L.append("%g, %g" % (DT, TTOT))
    L.append("*DLOAD")                  # "P" = Abaqus's shell pressure load type
    for j in range(NY):                 # q0 sin sin, sampled at the element centre
        for i in range(NX):
            q = Q0 * np.sin(np.pi * (i + 0.5) * dx / A) \
                   * np.sin(np.pi * (j + 0.5) * dy / B)
            L.append("%d, P, %.6e" % (e(i, j), q))
    # SF/SM and COORD must be TWO separate *EL PRINT blocks: the .dat parser keys
    # them as two tables, and merging them into one request breaks the reshape.
    if FIELD:
        # the WHOLE plate, once, at the snapshot increment -- this is what the
        # 3-D field VTK is dehomogenized from
        L.append("*NODE PRINT, NSET=NALL, FREQUENCY=%d" % FINC)
        L.append("U")
        L.append("*EL PRINT, ELSET=EALL, FREQUENCY=%d" % FINC)
        L.append("SF, SM")
        L.append("*EL PRINT, ELSET=EALL, FREQUENCY=%d" % FINC)
        L.append("COORD")
    else:
        L.append("*NODE PRINT, NSET=NCEN_REF, FREQUENCY=1")
        L.append("U")
        L.append("*EL PRINT, ELSET=PATCHC, FREQUENCY=1")
        L.append("SF, SM")
        L.append("*EL PRINT, ELSET=PATCHC, FREQUENCY=1")
        L.append("COORD")
    L.append("*END STEP")

    # ---- output ----------------------------------------------------------------
    out = os.path.splitext(DB)[0] + "_abaqus_%s%s.inp" % (SELT,
                                                         "_field" if FIELD else "")
    open(out, "w").write("\n".join(L) + "\n")
    print("%s -> %s  (%d %s, %d nodes, rho_h = %.4f kg/m^2, %s)"
          % (os.path.basename(DB), os.path.basename(out), NX * NY, SELT,
             len(NID), rho_h,
             "with *TRANSVERSE SHEAR STIFFNESS" if model == 1
             else "ABD only, no shear card"))


if __name__ == "__main__":
    import time as _t
    print("start: " + _t.strftime("%Y-%m-%d %H:%M:%S"))
    main()
    print("end:   " + _t.strftime("%Y-%m-%d %H:%M:%S"))
