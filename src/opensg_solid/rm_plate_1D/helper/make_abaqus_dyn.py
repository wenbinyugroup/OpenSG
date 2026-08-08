"""make_abaqus_dyn.py -- deck generator for the Nayak-Shenoi-Moy Example-5
transient sandwich benchmark (README.md).  Reads sandwich_sg.yaml
(1d_sg.py) and writes FOUR decks -- {RM shell, 3-D solid} x {step, blast}:

    sandwich_RM_S8R_step.inp / sandwich_RM_S8R_blast.inp
        20 x 20 S8R plate (the ex5 standard element; the legacy S4 decks
        remain available via write_rm(..., selt="S4")) carrying the
        OpenSG-RM 8x8 as a *SHELL GENERAL
        SECTION (+ 2x2 shear + section density 40.69 kg/m^2), SS-1 edges,
        the paper's DOUBLE-SINE spatial load q0 sin(pi x/a) sin(pi y/b) with
        F(t) = step or the explosive blast e^(-330 t), q0 = 68.9476 MPa,
        implicit dynamics dt = 50 us for 0.02 s (the paper's step size).
        Per-increment prints: center U; SF/SM + COORD at three 2x2 patches
        (center, x-edge middle, y-edge middle) for the recovery.
    sandwich_SOLID_step.inp / sandwich_SOLID_blast.inp
        the benchmark: 20 x 20 x 16 C3D8I (one element per face PLY -- eight
        plies -- and eight through the core), per-ply orientation/material/
        density, same BCs/load/time.  Center U every increment; centroidal S
        at the matching columns every 10th increment (0.5 ms cadence).

The double-sine spatial load is the plate's own Navier mode: at the center
and edge-middle stations the recovery gradients are closed-form, and the
patch prints provide the same information redundantly for cross-checking.

Variables mirror examples/opensg-rm_dynamic conventions; the blast amplitude
is tabulated as e^(-330 t) at 0.25 ms spacing (t1 = 6 ms decay per the paper).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "src", "opensg_solid", "rm_plate_1D")):
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from opensg_solid.rm_plate_1D.segment_plate import read_plate_sg_yaml
from opensg_solid.rm_plate_1D.msg_rm_plate import rm_plate_msg

A = 1.524                    # plate side a = b = 10 h [m]
NX = 20                      # in-plane elements per side
Q0 = 68.9476e6               # load intensity [Pa] (the paper's q0)
DT, TTOT = 5.0e-5, 0.02      # the paper's 50 us step; 20 ms window
CBLAST = 330.0               # blast decay e^(-c t), c = 330 1/s
NZC = 8                      # solid elements through the core


def n(i, j):
    return 1 + i + (NX + 1) * j


def e(i, j):
    return 1 + i + NX * j


def amp_lines(name, kind):
    """The *AMPLITUDE block: 'step' = 1 from t = 0; 'blast' = e^(-c t)
    tabulated every 0.25 ms."""
    L = ["*AMPLITUDE, NAME=%s" % name]
    if kind == "step":
        L.append("0., 1., %g, 1." % TTOT)
    else:
        ts = np.arange(0.0, TTOT + 1e-12, 2.5e-4)
        vals = []
        for t in ts:
            vals += ["%.6g" % t, "%.6g" % np.exp(-CBLAST * t)]
        for s in range(0, len(vals), 8):
            L.append(", ".join(vals[s:s + 8]))
    return L


def sinsin(i, j):
    """The double-sine load amplitude at the element center."""
    xc = (i + 0.5) * dx
    yc = (j + 0.5) * dx
    return np.sin(np.pi * xc / A) * np.sin(np.pi * yc / A)


def write_rm(path, kind, selt="S8R"):
    """The OpenSG-RM shell deck.  selt = "S8R" (the ex5 standard: thick
    quadratic shell, carries the MSG transverse shear) or "S4" (the legacy
    element the first history runs used).  The nodes live on the HALF-INDEX
    grid of abaqus_inp.py: S4 keeps only the even-even (corner) nodes, S8R
    adds the edge midpoints (serendipity, at most one odd index)."""
    tri = [AB[i, j] for j in range(6) for i in range(j + 1)]

    def exists(I, J):
        odd = (I % 2) + (J % 2)
        return odd == 0 if selt == "S4" else odd <= 1

    NID, k2 = {}, 0
    for J in range(2 * NX + 1):
        for I in range(2 * NX + 1):
            if exists(I, J):
                k2 += 1
                NID[(I, J)] = k2
    nn = lambda i, j: NID[(2 * i, 2 * j)]        # corner node of grid (i, j)
    CORN = [(0, 0), (2, 0), (2, 2), (0, 2)]
    MIDS = [(1, 0), (2, 1), (1, 2), (0, 1)]
    OFFS = CORN if selt == "S4" else CORN + MIDS

    L = ["*HEADING",
         "Nayak-Shenoi-Moy Ex.5 sandwich, OpenSG-RM 8x8 shell (%s), %s pulse"
         % (selt, kind),
         "** (0/90/0/90/core)s, a = b = %g m, h = %g m, q0 = %g Pa" %
         (A, H, Q0),
         "** load q0 F(t) sin(pi x/a) sin(pi y/b); dt = %g s, T = %g s"
         % (DT, TTOT),
         "*NODE"]
    for (I, J), nid in sorted(NID.items(), key=lambda kv: kv[1]):
        L.append("%d, %.8f, %.8f, 0.0" % (nid, 0.5 * I * dx, 0.5 * J * dx))
    L.append("*ELEMENT, TYPE=%s, ELSET=EALL" % selt)
    for j in range(NX):
        for i in range(NX):
            row = [e(i, j)] + [NID[(2 * i + a, 2 * j + b)] for a, b in OFFS]
            L.append(", ".join(str(v) for v in row))
    c = NX // 2
    L.append("*NSET, NSET=NCEN")
    L.append("%d" % nn(c, c))
    for name, ids in (("PATCHC", [e(c - 1, c - 1), e(c, c - 1),
                                  e(c - 1, c), e(c, c)]),
                      ("PATCHX", [e(0, c - 1), e(1, c - 1),
                                  e(0, c), e(1, c)]),
                      ("PATCHY", [e(c - 1, 0), e(c - 1, 1),
                                  e(c, 0), e(c, 1)])):
        L.append("*ELSET, ELSET=%s" % name)
        L.append(", ".join(str(v) for v in ids))
    # edge sets filtered from the HALF-INDEX grid: with a quadratic element
    # the MIDSIDE nodes on each edge must be constrained too (corner-only
    # pinning lets the edge bulge between them and reads as a softer plate)
    edge = lambda pred: sorted(v for k, v in NID.items() if pred(k))
    for name, ids in (("NX0", edge(lambda k: k[0] == 0)),
                      ("NXA", edge(lambda k: k[0] == 2 * NX)),
                      ("NY0", edge(lambda k: k[1] == 0)),
                      ("NYB", edge(lambda k: k[1] == 2 * NX))):
        L.append("*NSET, NSET=%s" % name)
        for s in range(0, len(ids), 12):
            L.append(", ".join(str(v) for v in ids[s:s + 12]))
    L.append("*NSET, NSET=NALL, GENERATE")
    L.append("1, %d, 1" % len(NID))
    L.append("**")
    L.append("*SHELL GENERAL SECTION, ELSET=EALL, DENSITY=%.6g" % rho_h)
    for s in range(0, len(tri), 8):
        L.append(", ".join("%.6e" % v for v in tri[s:s + 8]))
    L.append("*TRANSVERSE SHEAR STIFFNESS")
    L.append("%.6e, %.6e, %.6e" % (G2[0, 0], G2[1, 1], G2[0, 1]))
    L.append("**")
    L.append("** SS-1 simple supports + drilling (cross-ply layup)")
    L.append("*BOUNDARY")
    L.append("NX0, 2, 3")
    L.append("NXA, 2, 3")
    L.append("NY0, 1, 1")
    L.append("NYB, 1, 1")
    L.append("NY0, 3, 3")
    L.append("NYB, 3, 3")
    L.append("NALL, 6, 6")
    L += amp_lines("FT", kind)
    L.append("*STEP, NAME=PULSE, INC=%d" % int(2 * TTOT / DT))
    L.append("*DYNAMIC")
    L.append("%g, %g, %g, %g" % (DT, TTOT, DT * 1e-4, DT))
    L.append("*DLOAD, AMPLITUDE=FT")
    for j in range(NX):
        for i in range(NX):
            L.append("%d, P, %.6e" % (e(i, j), Q0 * sinsin(i, j)))
    L.append("*OUTPUT, FIELD, FREQUENCY=20")
    L.append("*ELEMENT OUTPUT")
    L.append("SF, SM")
    L.append("*NODE OUTPUT")
    L.append("U")
    L.append("*NODE PRINT, NSET=NCEN, FREQUENCY=1")
    L.append("U")
    for es in ("PATCHC", "PATCHX", "PATCHY"):
        L.append("*EL PRINT, ELSET=%s, FREQUENCY=1" % es)
        L.append("SF, SM")
        L.append("*EL PRINT, ELSET=%s, FREQUENCY=1" % es)
        L.append("COORD")
    L.append("*END STEP")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    return path


def write_solid(path, kind, selt="C3D20"):
    """The 3-D solid benchmark deck.  selt = "C3D20" (the ex5 standard:
    quadratic, FULL 3x3x3 integration -- the thin 80:1-aspect face plies
    make the linear C3D8I ~20 %% too stiff) or "C3D8I" (the legacy first
    runs).  C3D20 nodes live on the 3-D HALF-INDEX grid (corners +
    edge midpoints, at most one odd index)."""
    # one element per face PLY (8) + NZC through the core = 16 layers
    zk = [0.0]
    for t in thick[:4]:
        zk.append(zk[-1] + t)
    for s in range(NZC):
        zk.append(zk[-1] + thick[4] / NZC)
    for t in thick[5:]:
        zk.append(zk[-1] + t)
    zk = np.array(zk)
    nzt = len(zk) - 1
    lay_of = [0, 1, 2, 3] + [4] * NZC + [5, 6, 7, 8]
    NPL = (NX + 1) * (NX + 1)

    if selt == "C3D8I":
        def n3(i, j, k):
            return 1 + i + (NX + 1) * j + NPL * k
    else:
        # half-index grid in all three directions; ZH[2k] = the layer
        # interface zk[k], ZH[2k+1] = the layer midplane
        ZH = np.empty(2 * nzt + 1)
        ZH[0::2] = zk
        ZH[1::2] = 0.5 * (zk[:-1] + zk[1:])
        N3ID, _k3 = {}, 0
        for K in range(2 * nzt + 1):
            for J in range(2 * NX + 1):
                for I in range(2 * NX + 1):
                    if (I % 2) + (J % 2) + (K % 2) <= 1:
                        _k3 += 1
                        N3ID[(I, J, K)] = _k3

        def n3(i, j, k):
            return N3ID[(2 * i, 2 * j, 2 * k)]

    def e3(i, j, k):
        return 1 + i + NX * j + NX * NX * k

    L = ["*HEADING",
         "Nayak-Shenoi-Moy Ex.5 sandwich, 3-D SOLID benchmark (%s), %s pulse"
         % (selt, kind),
         "** %d x %d x %d %s; one element per face ply, %d through core"
         % (NX, NX, nzt, selt, NZC),
         "*NODE"]
    if selt == "C3D8I":
        for k in range(nzt + 1):
            for j in range(NX + 1):
                for i in range(NX + 1):
                    L.append("%d, %.8f, %.8f, %.8f"
                             % (n3(i, j, k), i * dx, j * dx, zk[k]))
        L.append("*ELEMENT, TYPE=C3D8I")
        for k in range(nzt):
            for j in range(NX):
                for i in range(NX):
                    L.append("%d, %d, %d, %d, %d, %d, %d, %d, %d"
                             % (e3(i, j, k),
                                n3(i, j, k), n3(i + 1, j, k),
                                n3(i + 1, j + 1, k),
                                n3(i, j + 1, k), n3(i, j, k + 1),
                                n3(i + 1, j, k + 1), n3(i + 1, j + 1, k + 1),
                                n3(i, j + 1, k + 1)))
    else:
        for (I, J, K), nid in sorted(N3ID.items(), key=lambda kv: kv[1]):
            L.append("%d, %.8f, %.8f, %.8f"
                     % (nid, 0.5 * I * dx, 0.5 * J * dx, ZH[K]))
        # Abaqus C3D20 order: 8 corners (bottom CCW, top CCW), then the 12
        # edge midpoints (bottom 9-12, top 13-16, vertical 17-20)
        OFF20 = [(0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0),
                 (0, 0, 2), (2, 0, 2), (2, 2, 2), (0, 2, 2),
                 (1, 0, 0), (2, 1, 0), (1, 2, 0), (0, 1, 0),
                 (1, 0, 2), (2, 1, 2), (1, 2, 2), (0, 1, 2),
                 (0, 0, 1), (2, 0, 1), (2, 2, 1), (0, 2, 1)]
        L.append("*ELEMENT, TYPE=C3D20")
        for k in range(nzt):
            for j in range(NX):
                for i in range(NX):
                    row = [e3(i, j, k)] + [
                        N3ID[(2 * i + a, 2 * j + b, 2 * k + c)]
                        for a, b, c in OFF20]
                    # 21 numbers > the 16-item Abaqus line limit: wrap with
                    # a trailing-comma continuation
                    L.append(", ".join(str(v) for v in row[:16]) + ",")
                    L.append(", ".join(str(v) for v in row[16:]))
    for m in range(9):
        L.append("*ELSET, ELSET=PLY%d" % (m + 1))
        ids = [str(e3(i, j, k)) for k in range(nzt) if lay_of[k] == m
               for j in range(NX) for i in range(NX)]
        for s in range(0, len(ids), 12):
            L.append(", ".join(ids[s:s + 12]))
    c = NX // 2
    L.append("*NSET, NSET=NCEN3D")
    L.append("%d" % n3(c, c, nzt // 2))
    for name, cols in (("COLC", [(c - 1, c - 1), (c, c - 1),
                                 (c - 1, c), (c, c)]),
                       ("COLX", [(0, c - 1), (0, c)]),
                       ("COLY", [(c - 1, 0), (c, 0)])):
        L.append("*ELSET, ELSET=%s" % name)
        ids = [str(e3(ii, jj, k)) for k in range(nzt) for (ii, jj) in cols]
        for s in range(0, len(ids), 12):
            L.append(", ".join(ids[s:s + 12]))
    for name, sel in (("NX0F", lambda i, j: i == 0),
                      ("NXAF", lambda i, j: i == NX),
                      ("NY0F", lambda i, j: j == 0),
                      ("NYBF", lambda i, j: j == NX)):
        L.append("*NSET, NSET=%s" % name)
        if selt == "C3D8I":
            ids = [str(n3(i, j, k)) for k in range(nzt + 1)
                   for j in range(NX + 1) for i in range(NX + 1)
                   if sel(i, j)]
        else:
            # HALF-INDEX face filter: the edge-midpoint nodes on the face
            # must be constrained too (corner-only pinning bulges)
            ids = [str(v) for key, v in sorted(N3ID.items(),
                                               key=lambda kv: kv[1])
                   if sel(key[0] / 2.0, key[1] / 2.0)]
        for s in range(0, len(ids), 12):
            L.append(", ".join(ids[s:s + 12]))
    L.append("**")
    for name in ("ge", "herex"):
        m = db[name]
        L.append("*MATERIAL, NAME=%s" % name.upper())
        L.append("*ELASTIC, TYPE=ENGINEERING CONSTANTS")
        L.append("%.6e, %.6e, %.6e, %.6g, %.6g, %.6g, %.6e, %.6e,"
                 % (m["E"][0], m["E"][1], m["E"][2], m["nu"][0], m["nu"][1],
                    m["nu"][2], m["G"][0], m["G"][1]))
        L.append("%.6e" % m["G"][2])
        L.append("*DENSITY")
        L.append("%g," % m["rho"])
    for m in range(9):
        L.append("*ORIENTATION, NAME=OR%d, SYSTEM=RECTANGULAR" % (m + 1))
        L.append("1.0, 0.0, 0.0, 0.0, 1.0, 0.0")
        L.append("3, %g" % angs[m])
        L.append("*SOLID SECTION, ELSET=PLY%d, MATERIAL=%s, ORIENTATION=OR%d"
                 % (m + 1, mats[m].upper(), m + 1))
    L.append("**")
    L.append("** SS-1 simple supports (same rule as the shell)")
    L.append("*BOUNDARY")
    L.append("NX0F, 2, 3")
    L.append("NXAF, 2, 3")
    L.append("NY0F, 1, 1")
    L.append("NYBF, 1, 1")
    L.append("NY0F, 3, 3")
    L.append("NYBF, 3, 3")
    L += amp_lines("FT", kind)
    L.append("*STEP, NAME=PULSE, INC=%d" % int(2 * TTOT / DT))
    L.append("*DYNAMIC")
    L.append("%g, %g, %g, %g" % (DT, TTOT, DT * 1e-4, DT))
    L.append("*DLOAD, AMPLITUDE=FT")
    for j in range(NX):
        for i in range(NX):
            L.append("%d, P2, %.6e" % (e3(i, j, nzt - 1), Q0 * sinsin(i, j)))
    L.append("*OUTPUT, FIELD, FREQUENCY=50")
    L.append("*ELEMENT OUTPUT")
    L.append("S")
    L.append("*NODE OUTPUT")
    L.append("U")
    L.append("*NODE PRINT, NSET=NCEN3D, FREQUENCY=1")
    L.append("U")
    for es in ("COLC", "COLX", "COLY"):
        L.append("*EL PRINT, ELSET=%s, POSITION=CENTROIDAL, FREQUENCY=10" % es)
        L.append("S")
    L.append("*END STEP")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    return path


def write_fsdt(path, kind, selt="S8R"):
    """The CONVENTIONAL Abaqus 2-D shell route (FSDT): *SHELL SECTION,
    COMPOSITE with all NINE physical layers (4 GE plies / foam core / 4 GE
    plies) -- Abaqus builds A/B/D and ITS OWN transverse-shear stiffness
    (the standard industry first-order treatment).  Identical mesh (selt,
    default S8R -- the same element as the RM deck so the comparison
    isolates the SECTION LAW, not the element), BCs, load and center-U
    print to the OpenSG-RM deck."""
    ge, hx = db["ge"], db["herex"]

    def exists(I, J):
        odd = (I % 2) + (J % 2)
        return odd == 0 if selt == "S4" else odd <= 1

    NID, k2 = {}, 0
    for J in range(2 * NX + 1):
        for I in range(2 * NX + 1):
            if exists(I, J):
                k2 += 1
                NID[(I, J)] = k2
    nn = lambda i, j: NID[(2 * i, 2 * j)]
    CORN = [(0, 0), (2, 0), (2, 2), (0, 2)]
    MIDS = [(1, 0), (2, 1), (1, 2), (0, 1)]
    OFFS = CORN if selt == "S4" else CORN + MIDS

    L = ["*HEADING",
         "Nayak-Shenoi-Moy Ex.5 sandwich, CONVENTIONAL Abaqus composite %s"
         " (FSDT), %s pulse" % (selt, kind),
         "*NODE"]
    for (I, J), nid in sorted(NID.items(), key=lambda kv: kv[1]):
        L.append("%d, %.8f, %.8f, 0.0" % (nid, 0.5 * I * dx, 0.5 * J * dx))
    L.append("*ELEMENT, TYPE=%s, ELSET=EALL" % selt)
    for j in range(NX):
        for i in range(NX):
            row = [e(i, j)] + [NID[(2 * i + a, 2 * j + b)] for a, b in OFFS]
            L.append(", ".join(str(v) for v in row))
    c = NX // 2
    L.append("*NSET, NSET=NCEN")
    L.append("%d" % nn(c, c))
    edge = lambda pred: sorted(v for k, v in NID.items() if pred(k))
    for name, ids in (("NX0", edge(lambda k: k[0] == 0)),
                      ("NXA", edge(lambda k: k[0] == 2 * NX)),
                      ("NY0", edge(lambda k: k[1] == 0)),
                      ("NYB", edge(lambda k: k[1] == 2 * NX))):
        L.append("*NSET, NSET=%s" % name)
        for s in range(0, len(ids), 12):
            L.append(", ".join(str(v) for v in ids[s:s + 12]))
    L.append("*NSET, NSET=NALL, GENERATE")
    L.append("1, %d, 1" % len(NID))
    L.append("*MATERIAL, NAME=GE")
    L.append("*ELASTIC, TYPE=LAMINA")
    L.append("%.6e, %.6e, %.4g, %.6e, %.6e, %.6e"
             % (ge["E"][0], ge["E"][1], ge["nu"][0],
                ge["G"][0], ge["G"][1], ge["G"][2]))
    L.append("*DENSITY")
    L.append("%g," % ge["rho"])
    L.append("*MATERIAL, NAME=HEREX")
    L.append("*ELASTIC, TYPE=LAMINA")
    L.append("%.6e, %.6e, %.4g, %.6e, %.6e, %.6e"
             % (hx["E"][0], hx["E"][1], hx["nu"][0],
                hx["G"][0], hx["G"][1], hx["G"][2]))
    L.append("*DENSITY")
    L.append("%g," % hx["rho"])
    L.append("*SHELL SECTION, ELSET=EALL, COMPOSITE")
    for t, mname, ang in zip(thick,
                             [x.upper() for x in mats], angs):
        L.append("%.8g, 3, %s, %g" % (t, mname, ang))
    L.append("** SS-1 + drilling (same rule as the RM deck)")
    L.append("*BOUNDARY")
    for b in ("NX0, 2, 3", "NXA, 2, 3", "NY0, 1, 1", "NYB, 1, 1",
              "NY0, 3, 3", "NYB, 3, 3", "NALL, 6, 6"):
        L.append(b)
    L += amp_lines("FT", kind)
    L.append("*STEP, NAME=PULSE, INC=%d" % int(2 * TTOT / DT))
    L.append("*DYNAMIC")
    L.append("%g, %g, %g, %g" % (DT, TTOT, DT * 1e-4, DT))
    L.append("*DLOAD, AMPLITUDE=FT")
    for j in range(NX):
        for i in range(NX):
            L.append("%d, P, %.6e" % (e(i, j), Q0 * sinsin(i, j)))
    L.append("*NODE PRINT, NSET=NCEN, FREQUENCY=1")
    L.append("U")
    L.append("*END STEP")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    return path


def main():
    """Homogenize the sandwich SG and write all six transient decks.

    In:
        none -- reads sandwich_sg.yaml next to this file.
    Out:
        None -- writes the {RM, SOLID, FSDT} x {step, blast} .inp decks and
        sets the module-level section data (AB, G2, rho_h, ...) that the
        write_* helpers read.
    """
    global inp, r, ABDG, AB, G2, db, thick, mats, angs, H, rho_h, dx
    inp = read_plate_sg_yaml(os.path.join(HERE, "sandwich_sg.yaml"))
    r = rm_plate_msg(inp["thick"], inp["angles"], inp["mat_names"],
                     inp["material_db"], fraction=inp["fraction"])
    ABDG = np.asarray(r["ABDG"])
    AB = ABDG[:6, :6]; G2 = ABDG[6:, 6:]
    db = inp["material_db"]
    thick = inp["thick"]; mats = inp["mat_names"]; angs = inp["angles"]
    H = float(sum(thick))
    rho_h = sum(db[m]["rho"] * t for m, t in zip(mats, thick))
    dx = A / NX
    # the ex5-standard element pairing: S8R shells (RM + FSDT) judged
    # against the C3D20 full-integration solid.  The legacy spellings
    # (S4 shells, C3D8I solid -- the Aug-1 first runs, both ~20 % too
    # stiff: S4 through its shear treatment, C3D8I through the 80:1
    # aspect face plies) stay available via the selt arguments.
    for kind in ("step", "blast"):
        p1 = write_rm(os.path.join(HERE, "sandwich_RM_S8R_%s.inp" % kind),
                      kind)
        p2 = write_solid(os.path.join(HERE,
                                      "sandwich_SOLID_C3D20_%s.inp" % kind),
                         kind)
        p3 = write_fsdt(os.path.join(HERE,
                                     "sandwich_FSDT_S8R_%s.inp" % kind),
                        kind)
        print("wrote %s, %s, %s" % (os.path.basename(p1),
                                    os.path.basename(p2),
                                    os.path.basename(p3)))
    print("rho*h = %.4f kg/m^2; D11 = %.4e; G11 = %.4e"
          % (rho_h, AB[3, 3], G2[0, 0]))


if __name__ == "__main__":
    main()
