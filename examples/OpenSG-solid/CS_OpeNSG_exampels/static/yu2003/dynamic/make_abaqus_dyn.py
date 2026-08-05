"""make_abaqus_dyn.py -- deck generator for the Yu-2003 SECTION 6.2 DYNAMIC
example (yu62_config.py): Abaqus/Standard implicit dynamics in the DYMORE role.

Two decks:

  yu_dyn_RM.inp     NX x NX S4 plate carrying the MSG-RM 8x8 as a *SHELL
                    GENERAL SECTION (+ MSG 2x2 *TRANSVERSE SHEAR STIFFNESS,
                    DENSITY = rho h as the section mass/area), corner MASS
                    element + triangular-impulse *CLOAD, clamped BC/CD edges.
                    Per-increment prints: U at M and A; SF/SM and COORD at the
                    2x2 element PATCHES around M and Q -- recover_dyn.py fits a
                    quadratic over the 16 patch integration points to get the
                    resultants AND their first/second gradients at the point
                    (exactly Yu's "interpolating the strain values at the Gauss
                    points of a 2x2 patch" procedure), which drive the Eq.-63/66
                    recovery.
  yu_dyn_SOLID.inp  the benchmark: NX x NX x (2 NZ_PLY) C3D8I solid plate,
                    per-ply *ORIENTATION (stresses print in the PLY frame --
                    the post rotates back), *DENSITY, the same corner mass and
                    impulse at the mid-thickness corner node, same clamps.
                    Prints: U at the centre mid-surface node every increment;
                    centroidal S along the four element columns around M and Q
                    every 96th increment (t = 0.0096 s first).

No rotary-inertia line is given for the shell section: the 50 kg corner mass
dominates the plate's own 0.0256 kg, so shell rotary inertia is negligible.

Run:
    python examples/yu2003/dynamic/make_abaqus_dyn.py
then submit both jobs from a space-free directory and give recover_dyn.py the
two job .dat files.

Script variables
----------------
n(i, j)      shell node id 1 + i + (NX+1) j at (i dx, j dy); dx = dy = W/NX
e(i, j)      shell element id 1 + i + NX j spanning [i, i+1] x [j, j+1] cells
n3(i, j, k)  solid node id adding 625 k per thickness plane k (z = k H/nzt)
e3(i, j, k)  solid element id adding 576 k per thickness layer
im, jm /     the node indices of M (12, 12) and Q (6, 18) -- patch elements
iq, jq       are the four cells touching that node
r, ABDG      rm_plate_msg result (mid-surface, fraction 0.5) and its 8x8
tri, G2      the 21 general-section constants (lower triangle of [[A,B],[B,D]]
             by columns) and the 2x2 shear block
amp          the *AMPLITUDE knots of the triangular impulse
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CC = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(CC, "src"))
sys.path.insert(0, HERE)

from opensg_solid.rm_plate_1D.msg_rm_plate import rm_plate_msg               # noqa: E402
from yu62_config import (MATERIAL_DB, W, H, THK, ANG, MATS, PMASS, F0,  # noqa: E402
                         T_RISE, T_END, DT, T_TOTAL, NX, NZ_PLY)

dx = W / NX


def n(i, j):
    """Shell node id at (i dx, j dx); i, j = 0..NX."""
    return 1 + i + (NX + 1) * j


def e(i, j):
    """Shell element id of cell [i, i+1] x [j, j+1]; i, j = 0..NX-1."""
    return 1 + i + NX * j


im = jm = NX // 2                      # M = (W/2, W/2)
iq, jq = round(H / dx), round((W - H) / dx)   # Q = (H, W - H) -> (6, 18)


def _common_dynamic(A):
    """The *AMPLITUDE + dynamic step preamble shared by both decks.

    Variables: A = the deck append shorthand; the *DYNAMIC data line is
    (initial increment, period, minimum, maximum) with max = DT pinning the
    fixed dt = 1e-4 s march; INC bounds the increment count.
    """
    A("*AMPLITUDE, NAME=TRI")
    A("0., 0., %g, 1., %g, 0., 1., 0." % (T_RISE, T_END))
    A("**")
    A("*STEP, NAME=IMPULSE, INC=2000")
    A("*DYNAMIC")
    A("%g, %g, %g, %g" % (DT, T_TOTAL, DT * 1e-4, DT))


def write_rm_dyn(path):
    """The MSG-RM shell dynamics deck (variables in the module docstring)."""
    r = rm_plate_msg(THK, ANG, MATS, MATERIAL_DB, fraction=0.5)
    ABDG = np.asarray(r["ABDG"])
    AB = ABDG[:6, :6]; G2 = ABDG[6:, 6:]
    tri = [AB[i, j] for j in range(6) for i in range(j + 1)]
    rho_h = MATERIAL_DB[MATS[0]]["rho"] * H

    L = []
    A = L.append
    A("*HEADING")
    A("Yu-2003 sec. 6.2 dynamics: clamped [90/0] plate, corner mass, impulse")
    A("** MSG-RM 8x8 general section; w = %g m, h = %g m, S4 %d x %d" %
      (W, H, NX, NX))
    A("** corner mass %g kg at A, impulse to %g N over %g s, dt = %g s" %
      (PMASS, F0, T_END, DT))
    A("** recovery patches: M node (%d,%d), Q node (%d,%d)" % (im, jm, iq, jq))
    A("*NODE")
    for j in range(NX + 1):
        for i in range(NX + 1):
            A("%d, %.8f, %.8f, 0.0" % (n(i, j), i * dx, j * dx))
    A("*ELEMENT, TYPE=S4, ELSET=EALL")
    for j in range(NX):
        for i in range(NX):
            A("%d, %d, %d, %d, %d" % (e(i, j), n(i, j), n(i + 1, j),
                                      n(i + 1, j + 1), n(i, j + 1)))
    A("*ELEMENT, TYPE=MASS, ELSET=CM")
    A("%d, %d" % (NX * NX + 1000, n(0, 0)))
    A("*MASS, ELSET=CM")
    A("%g," % PMASS)
    A("*NSET, NSET=NA")
    A("%d" % n(0, 0))
    A("*NSET, NSET=NM")
    A("%d" % n(im, jm))
    A("*NSET, NSET=NQ")
    A("%d" % n(iq, jq))
    A("*NSET, NSET=NBC")
    ids = [str(n(NX, j)) for j in range(NX + 1)]
    for s in range(0, len(ids), 12):
        A(", ".join(ids[s:s + 12]))
    A("*NSET, NSET=NCD")
    ids = [str(n(i, NX)) for i in range(NX + 1)]
    for s in range(0, len(ids), 12):
        A(", ".join(ids[s:s + 12]))
    A("*ELSET, ELSET=PATCHM")
    A("%d, %d, %d, %d" % (e(im - 1, jm - 1), e(im, jm - 1),
                          e(im - 1, jm), e(im, jm)))
    A("*ELSET, ELSET=PATCHQ")
    A("%d, %d, %d, %d" % (e(iq - 1, jq - 1), e(iq, jq - 1),
                          e(iq - 1, jq), e(iq, jq)))
    A("**")
    A("*SHELL GENERAL SECTION, ELSET=EALL, DENSITY=%g" % rho_h)
    for s in range(0, len(tri), 8):
        A(", ".join("%.6e" % v for v in tri[s:s + 8]))
    A("*TRANSVERSE SHEAR STIFFNESS")
    A("%.6e, %.6e, %.6e" % (G2[0, 0], G2[1, 1], G2[0, 1]))
    A("**")
    A("*BOUNDARY")
    A("NBC, 1, 6")
    A("NCD, 1, 6")
    A("**")
    _common_dynamic(A)
    A("*CLOAD, AMPLITUDE=TRI")
    A("%d, 3, %g" % (n(0, 0), -F0))
    A("*OUTPUT, FIELD, FREQUENCY=10")
    A("*ELEMENT OUTPUT")
    A("SF, SM")
    A("*NODE OUTPUT")
    A("U")
    A("*NODE PRINT, NSET=NM, FREQUENCY=1")
    A("U")
    A("*NODE PRINT, NSET=NA, FREQUENCY=1")
    A("U")
    for es in ("PATCHM", "PATCHQ"):
        A("*EL PRINT, ELSET=%s, FREQUENCY=1" % es)
        A("SF, SM")
        A("*EL PRINT, ELSET=%s, FREQUENCY=1" % es)
        A("COORD")
    A("*END STEP")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    return path


def n3(i, j, k):
    """Solid node id at (i dx, j dx, k H/nzt); k = 0..2 NZ_PLY."""
    return 1 + i + (NX + 1) * j + (NX + 1) * (NX + 1) * k


def e3(i, j, k):
    """Solid element id of cell (i, j) in thickness layer k."""
    return 1 + i + NX * j + NX * NX * k


def write_solid_dyn(path):
    """The 3-D solid dynamics benchmark deck (variables in module docstring)."""
    nzt = 2 * NZ_PLY
    dz = H / nzt
    kmid = nzt // 2
    L = []
    A = L.append
    A("*HEADING")
    A("Yu-2003 sec. 6.2 dynamics SOLID benchmark: [90/0] plate, corner mass")
    A("** C3D8I %d x %d x %d, mass+load at the mid-thickness corner node" %
      (NX, NX, nzt))
    A("** EL PRINT S is in the PLY LOCAL frame (ORIENTATION) -- post rotates")
    A("*NODE")
    for k in range(nzt + 1):
        for j in range(NX + 1):
            for i in range(NX + 1):
                A("%d, %.8f, %.8f, %.8f" % (n3(i, j, k), i * dx, j * dx,
                                            k * dz))
    A("*ELEMENT, TYPE=C3D8I")
    for k in range(nzt):
        for j in range(NX):
            for i in range(NX):
                A("%d, %d, %d, %d, %d, %d, %d, %d, %d"
                  % (e3(i, j, k),
                     n3(i, j, k), n3(i + 1, j, k), n3(i + 1, j + 1, k),
                     n3(i, j + 1, k),
                     n3(i, j, k + 1), n3(i + 1, j, k + 1),
                     n3(i + 1, j + 1, k + 1), n3(i, j + 1, k + 1)))
    for m in range(2):
        A("*ELSET, ELSET=PLY%d, GENERATE" % (m + 1))
        A("%d, %d, 1" % (e3(0, 0, m * NZ_PLY), e3(NX - 1, NX - 1,
                                                  (m + 1) * NZ_PLY - 1)))
    A("*ELEMENT, TYPE=MASS, ELSET=CM")
    A("%d, %d" % (NX * NX * nzt + 1000, n3(0, 0, kmid)))
    A("*MASS, ELSET=CM")
    A("%g," % PMASS)
    A("*NSET, NSET=NM3D")
    A("%d" % n3(im, jm, kmid))
    A("*NSET, NSET=NA3D")
    A("%d" % n3(0, 0, kmid))
    A("*NSET, NSET=NBC3D")
    ids = [str(n3(NX, j, k)) for k in range(nzt + 1) for j in range(NX + 1)]
    for s in range(0, len(ids), 12):
        A(", ".join(ids[s:s + 12]))
    A("*NSET, NSET=NCD3D")
    ids = [str(n3(i, NX, k)) for k in range(nzt + 1) for i in range(NX + 1)]
    for s in range(0, len(ids), 12):
        A(", ".join(ids[s:s + 12]))
    for name, (ic, jc) in (("COLM3D", (im, jm)), ("COLQ3D", (iq, jq))):
        A("*ELSET, ELSET=%s" % name)
        ids = [str(e3(ii, jj, k)) for k in range(nzt)
               for jj in (jc - 1, jc) for ii in (ic - 1, ic)]
        for s in range(0, len(ids), 12):
            A(", ".join(ids[s:s + 12]))
    A("**")
    m0 = MATERIAL_DB[MATS[0]]
    A("*MATERIAL, NAME=YU62")
    A("*ELASTIC, TYPE=ENGINEERING CONSTANTS")
    A("%.6e, %.6e, %.6e, %.6g, %.6g, %.6g, %.6e, %.6e,"
      % (m0["E"][0], m0["E"][1], m0["E"][2], m0["nu"][0], m0["nu"][1],
         m0["nu"][2], m0["G"][0], m0["G"][1]))
    A("%.6e" % m0["G"][2])
    A("*DENSITY")
    A("%g," % m0["rho"])
    for m in range(2):
        A("*ORIENTATION, NAME=OR%d, SYSTEM=RECTANGULAR" % (m + 1))
        A("1.0, 0.0, 0.0, 0.0, 1.0, 0.0")
        A("3, %g" % ANG[m])
        A("*SOLID SECTION, ELSET=PLY%d, MATERIAL=YU62, ORIENTATION=OR%d"
          % (m + 1, m + 1))
    A("**")
    A("*BOUNDARY")
    A("NBC3D, 1, 3")
    A("NCD3D, 1, 3")
    A("**")
    _common_dynamic(A)
    A("*CLOAD, AMPLITUDE=TRI")
    A("%d, 3, %g" % (n3(0, 0, kmid), -F0))
    A("*OUTPUT, FIELD, FREQUENCY=50")
    A("*ELEMENT OUTPUT")
    A("S")
    A("*NODE OUTPUT")
    A("U")
    A("*NODE PRINT, NSET=NM3D, FREQUENCY=1")
    A("U")
    A("*NODE PRINT, NSET=NA3D, FREQUENCY=1")
    A("U")
    A("*EL PRINT, ELSET=COLM3D, POSITION=CENTROIDAL, FREQUENCY=%d"
      % round(0.0096 / DT))
    A("S")
    A("*EL PRINT, ELSET=COLQ3D, POSITION=CENTROIDAL, FREQUENCY=%d"
      % round(0.0096 / DT))
    A("S")
    A("*END STEP")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    return path


if __name__ == "__main__":
    p1 = write_rm_dyn(os.path.join(HERE, "yu_dyn_RM.inp"))
    p2 = write_solid_dyn(os.path.join(HERE, "yu_dyn_SOLID.inp"))
    print("wrote %s and %s" % (os.path.relpath(p1, CC), os.path.relpath(p2, CC)))
