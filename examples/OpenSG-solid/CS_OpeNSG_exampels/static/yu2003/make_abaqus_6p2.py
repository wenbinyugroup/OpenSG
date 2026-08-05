"""make_abaqus_6p2.py -- deck generator for the Yu-2003 SECTION-6.2 analog:
Abaqus in the DYMORE role.

Yu's sec. 6.2 couples VAPAS to DYMORE: VAPAS supplies the 2-D constitutive law,
DYMORE solves the 2-D plate problem, and VAPAS recovers the 3-D fields from
DYMORE's output.  Here Abaqus plays DYMORE for the SAME three cylindrical-bending
cases of sec. 6.1, so every link of the chain is checkable:

  deck 1  yu_<case>_RM.inp     S4 strip carrying the MSG-RM 8x8 as a *SHELL
                               GENERAL SECTION (+ the MSG 2x2 as *TRANSVERSE
                               SHEAR STIFFNESS).  Its printed section forces
                               (SF/SM at the x = 0 and x = L/2 stations) are the
                               FF input to the OpenSG-RM 3-D recovery
                               (recover_6p2.py) -- Abaqus REPLACES the harmonic
                               solve of yu_bench.rm_cyl_bend, nothing else.
  deck 2  yu_<case>_FSDT.inp   the same strip as a community *SHELL SECTION,
                               COMPOSITE (+ *ELASTIC, TYPE=LAMINA): Abaqus's own
                               ABD + its own transverse-shear estimate, i.e. the
                               INBUILT FOSDT baseline; TSHR13/TSHR23 (Abaqus's
                               equilibrium-based shear estimate) is requested.
  deck 3  yu_<case>_SOLID.inp  the BENCHMARK: a 3-D C3D8I solid strip of the
                               laminate, one element wide with node-by-node
                               *EQUATION ties across the width (d/dy = 0, the
                               infinite-plate condition -- REQUIRED for these
                               shear-coupled angle plies), split face load
                               +-p0/2, per-ply *ORIENTATION.  Its centroidal
                               stresses along the x = 0 and x = L/2 element
                               columns are the reference profiles.

All three shell/solid models use the exact-solution-consistent supports: u3 = 0
on the whole end faces (the harmonic solution has w ~ sin(px) -> w = 0 there at
EVERY z), ends free in x (sigma_11 ~ sin -> 0 there), rigid modes pinned at
x = L/2 where u = v = 0 exactly (u, v ~ cos).  The shell decks use the
``coupled=True`` BC mode of helper_inp_plate (u2/ur1 FREE -- the angle plies
respond in v and twist).

Run (writes into each case folder):
    python examples/yu2003/make_abaqus_6p2.py
Submit each job from a SPACE-FREE directory (helper_inp_plate gotcha):
    abaqus job=yu_case1_RM interactive
and copy the three job .dat files back into examples/yu2003/<case>/Abaqus_Plate/
for recover_6p2.py.

Script variables (bottom of file)
---------------------------------
name, lay      the LAYUPS key and layup dict of the current case
thk, ang, mats its plies (bottom -> top)
r, ABDG        rm_plate_msg result and its 8x8 (mid-surface, fraction = 0.5)
outdir         the case folder; hdr = deck header comment lines
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# repo root = the first ancestor holding opensg_jax/ (location-robust: also
# runs from the examples/RM_OpenSG_pagano copy)
CC = HERE
while not os.path.isdir(os.path.join(CC, "src", "opensg_solid", "rm_plate_1D")):
    _up = os.path.dirname(CC)
    if _up == CC:
        raise RuntimeError("rm_plate repo root not found above " + __file__)
    CC = _up
sys.path.insert(0, os.path.join(CC, "src"))
sys.path.insert(0, HERE)

from opensg_solid.rm_plate_1D.msg_rm_plate import rm_plate_msg               # noqa: E402
from opensg_solid.rm_plate_1D.helper_inp_plate import (write_plate_strip_inp,  # noqa: E402
                                                write_plate_strip_inp_fsdt)
from yu_layups import MATERIAL_DB, LAYUPS, H, L_SPAN, P0              # noqa: E402


def write_strip_solid_inp(path, thick, angles_deg, mat_names, material_db,
                          a=4.0, q0=1.0, nx=160, nz_per_ply=10, b=0.1,
                          header_lines=()):
    """The 3-D solid benchmark deck: C3D8I strip, width-tied (infinite plate).

    Model: 0 <= x <= a, one element across the width (0 <= y <= b), the laminate
    stacked in z (bottom face at z = 0).  Every (x, z) node pair across the width
    is tied with *EQUATION on all three dofs -- u(y=b) = u(y=0) -- which enforces
    d/dy = 0 exactly (eps22 = 0, the infinite-plate cylindrical-bending
    condition); free y-faces would instead give a plane-stress strip, WRONG for
    the shear-coupled laminate.  Load: Yu's split faces, pressure P = +q0/2
    sin(px) on the BOTTOM face (pushing +z) and P = -q0/2 sin(px) on the TOP
    (pulling +z), so sigma_33 = -+q0/2 at the faces as in the exact solution.
    Supports: u3 = 0 on both END faces (harmonic-consistent at every z); rigid
    pins u1 = u2 = 0 at one x = a/2 bottom node (u, v ~ cos(px) vanish there).

    Variables
    ---------
    path                 output .inp path
    thick, angles_deg,   the laminate (ply 0 at the BOTTOM), psi/in units
    mat_names,
    material_db
    a, q0                span [in], TOTAL load amplitude p0 [psi]
    nx, nz_per_ply, b    elements along the span / through EACH ply / strip
                         width [in] (one element across)
    p                    wavenumber pi/a [1/in]
    zpl                  (nply+1,) ply interface heights from the bottom face
    zk                   the through-thickness node planes (nz_per_ply per ply,
                         non-uniform if ply thicknesses differ)
    nzt                  total node planes - 1 = total elements through thickness
    nid(i, j, k)         node numbering: span i (0..nx), width row j (0, 1),
                         thickness plane k (0..nzt)
    eid(i, k)            element numbering: 1 + i + k*nx
    elay(k)              the ply index of thickness-element k (for ELSET/section)
    L, A                 the deck lines and the append shorthand
    xc                   element-centroid x for the sine load and the extraction
                         corrections (recorded in the header)
    COL0, COLM           extraction element columns: i = 0 (s13/s23 station,
                         centroid x = dx/2) and i = nx/2 (s33 station, centroid
                         x = a/2 + dx/2); *EL PRINT POSITION=CENTROIDAL writes S
                         [S11,S22,S33,S12,S13,S23] for both into the job .dat
    NW                   the mid-span bottom node (u3 print = deflection check)
    """
    thick = [float(t) for t in thick]
    p = np.pi / a
    nply = len(thick)
    zpl = np.concatenate([[0.0], np.cumsum(thick)])
    zk = np.concatenate([np.linspace(zpl[m], zpl[m + 1], nz_per_ply + 1)[:-1]
                         for m in range(nply)] + [[zpl[-1]]])
    nzt = len(zk) - 1
    dx = a / nx

    def nid(i, j, k):
        return 1 + i + j * (nx + 1) + k * 2 * (nx + 1)

    def eid(i, k):
        return 1 + i + k * nx

    def elay(k):
        return int(np.searchsorted(zpl[1:-1], 0.5 * (zk[k] + zk[k + 1]),
                                   side="right"))

    L = []
    A = L.append
    A("*HEADING")
    A("Cylindrical-bending SOLID benchmark strip (Yu-2003 sec.-6.2 analog)")
    for ln in header_lines:
        A("** " + ln)
    A("** a = %g in, h = %g in, q0 = p0 = %g psi split +-p0/2 on the faces" %
      (a, zpl[-1], q0))
    A("** %d x 1 x %d C3D8I; width ties = infinite plate; ends u3 = 0" %
      (nx, nzt))
    A("** extraction: COL0 centroid x = %.6g (divide s13/s23 by cos(p*x) = %.8f)"
      % (0.5 * dx, np.cos(p * 0.5 * dx)))
    A("**             COLM centroid x = %.6g (divide s33 by sin(p*x) = %.8f)"
      % (a / 2 + 0.5 * dx, np.sin(p * (a / 2 + 0.5 * dx))))
    A("*NODE")
    for k in range(nzt + 1):
        for j in (0, 1):
            for i in range(nx + 1):
                A("%d, %.8f, %.8f, %.8f" % (nid(i, j, k), i * dx, j * b, zk[k]))
    A("*ELEMENT, TYPE=C3D8I")
    for k in range(nzt):
        for i in range(nx):
            A("%d, %d, %d, %d, %d, %d, %d, %d, %d"
              % (eid(i, k),
                 nid(i, 0, k), nid(i + 1, 0, k), nid(i + 1, 1, k), nid(i, 1, k),
                 nid(i, 0, k + 1), nid(i + 1, 0, k + 1), nid(i + 1, 1, k + 1),
                 nid(i, 1, k + 1)))
    for m in range(nply):
        A("*ELSET, ELSET=PLY%d" % (m + 1))
        ids = [str(eid(i, k)) for k in range(nzt) if elay(k) == m
               for i in range(nx)]
        for s in range(0, len(ids), 12):
            A(", ".join(ids[s:s + 12]))
    A("*ELSET, ELSET=COL0")
    for s in range(0, nzt, 12):
        A(", ".join(str(eid(0, k)) for k in range(s, min(s + 12, nzt))))
    A("*ELSET, ELSET=COLM")
    for s in range(0, nzt, 12):
        A(", ".join(str(eid(nx // 2, k)) for k in range(s, min(s + 12, nzt))))
    A("*NSET, NSET=NX0F")
    ids = [str(nid(0, 0, k)) for k in range(nzt + 1)]
    for s in range(0, len(ids), 12):
        A(", ".join(ids[s:s + 12]))
    A("*NSET, NSET=NXAF")
    ids = [str(nid(nx, 0, k)) for k in range(nzt + 1)]
    for s in range(0, len(ids), 12):
        A(", ".join(ids[s:s + 12]))
    A("*NSET, NSET=NW")
    A("%d" % nid(nx // 2, 0, 0))
    A("**")
    used = list(dict.fromkeys(mat_names))
    for m in used:
        E = material_db[m]["E"]; G = material_db[m]["G"]; nu = material_db[m]["nu"]
        A("*MATERIAL, NAME=%s" % m.upper())
        A("*ELASTIC, TYPE=ENGINEERING CONSTANTS")
        A("%.6e, %.6e, %.6e, %.6g, %.6g, %.6g, %.6e, %.6e,"
          % (E[0], E[1], E[2], nu[0], nu[1], nu[2], G[0], G[1]))
        A("%.6e" % G[2])
    for m in range(nply):
        A("*ORIENTATION, NAME=OR%d, SYSTEM=RECTANGULAR" % (m + 1))
        A("1.0, 0.0, 0.0, 0.0, 1.0, 0.0")
        A("3, %g" % angles_deg[m])
        A("*SOLID SECTION, ELSET=PLY%d, MATERIAL=%s, ORIENTATION=OR%d"
          % (m + 1, mat_names[m].upper(), m + 1))
    A("**")
    A("** ---- width ties: u(y=b) = u(y=0), all dofs -> d/dy = 0 exactly ----")
    A("** (the j=1 node is the ELIMINATED first term; BCs touch j=0 nodes only)")
    A("*EQUATION")
    for k in range(nzt + 1):
        for i in range(nx + 1):
            for d in (1, 2, 3):
                A("2")
                A("%d, %d, 1.0, %d, %d, -1.0" % (nid(i, 1, k), d, nid(i, 0, k), d))
    A("**")
    A("** ---- simple supports (harmonic-consistent) + rigid pins ----")
    A("*BOUNDARY")
    A("NX0F, 3, 3")
    A("NXAF, 3, 3")
    A("%d, 1, 2" % nid(nx // 2, 0, 0))
    A("**")
    A("*STEP, NAME=SINELOAD")
    A("*STATIC")
    A("*DLOAD")
    for i in range(nx):
        xc = (i + 0.5) * dx
        A("%d, P1, %.6e" % (eid(i, 0), 0.5 * q0 * np.sin(p * xc)))
        A("%d, P2, %.6e" % (eid(i, nzt - 1), -0.5 * q0 * np.sin(p * xc)))
    A("*OUTPUT, FIELD")
    A("*ELEMENT OUTPUT")
    A("S")
    A("*NODE OUTPUT")
    A("U")
    A("*NODE PRINT, NSET=NW")
    A("U")
    A("*EL PRINT, ELSET=COL0, POSITION=CENTROIDAL")
    A("S")
    A("*EL PRINT, ELSET=COLM, POSITION=CENTROIDAL")
    A("S")
    A("*END STEP")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    return path


def _append_tshr(path):
    """Patch a community-FSDT deck: request Abaqus's own transverse-shear-stress
    estimate TSHR13/TSHR23 (its equilibrium-based composite-shell output) at both
    extraction stations, in front of *END STEP.

    Variables: path = the deck to patch in place; txt = its content; block = the
    two extra *EL PRINT tables (all section points).
    """
    with open(path) as f:
        txt = f.read()
    block = ("*EL PRINT, ELSET=EMID\n"
             "TSHR13, TSHR23\n"
             "*EL PRINT, ELSET=EEND\n"
             "TSHR13, TSHR23\n")
    with open(path, "w") as f:
        f.write(txt.replace("*END STEP", block + "*END STEP"))


if __name__ == "__main__":
    for name, lay in LAYUPS.items():
        thk = lay["thick"]; ang = lay["angles"]; mats = lay["mat_names"]
        outdir = os.path.join(HERE, name)
        r = rm_plate_msg(thk, ang, mats, MATERIAL_DB, fraction=0.5)
        ABDG = np.asarray(r["ABDG"])
        hdr = ["Yu-2003 sec.-6.2 analog, %s: plies (bottom->top) %s" % (name,
               ", ".join("%s(%.3gin/%g)" % (m, t, x)
                         for m, t, x in zip(mats, thk, ang))),
               "L/h = 4 (a = %g in, h = %g in), p0 = %g psi, psi/in units"
               % (L_SPAN, H, P0)]
        p1 = write_plate_strip_inp(
            os.path.join(outdir, "yu_%s_RM.inp" % name), ABDG, a=L_SPAN, q0=P0,
            nel=100, header_lines=hdr + ["MSG-RM 8x8 general section; Abaqus "
                                         "supplies FF to recover_6p2.py"],
            coupled=True)
        p2 = write_plate_strip_inp_fsdt(
            os.path.join(outdir, "yu_%s_FSDT.inp" % name), thk, ang, mats,
            MATERIAL_DB, a=L_SPAN, q0=P0, nel=100,
            header_lines=hdr + ["community FSDT: Abaqus ABD + its own shear; "
                                "TSHR13/23 requested"],
            coupled=True)
        _append_tshr(p2)
        p3 = write_strip_solid_inp(
            os.path.join(outdir, "yu_%s_SOLID.inp" % name), thk, ang, mats,
            MATERIAL_DB, a=L_SPAN, q0=P0, header_lines=hdr + [
                "3-D SOLID BENCHMARK (replaces Pagano in the 6.2 chain)"])
        print("wrote %s, %s, %s" % tuple(os.path.relpath(q, CC)
                                         for q in (p1, p2, p3)))
