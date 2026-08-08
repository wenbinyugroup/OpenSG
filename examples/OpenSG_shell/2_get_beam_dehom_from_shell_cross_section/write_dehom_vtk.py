"""Exploded-mesh VTK of the SHELL cross-section through-thickness stress recovery.

Re-runs the two-step RM dehomogenization by importing the sibling driver
beam_dehom_shell (same bundle, same MSG-RM plate recovery, no physics duplicated)
and re-packages its recovered Gauss values as a legacy VTK UNSTRUCTURED_GRID of
wall-thickness-resolved QUAD cells (VTK cell type 9).

One quad per (contour element e, depth band k): the band spans
zeta in [k/n_depth, (k+1)/n_depth], its centre being the recovery station
zeta_k = (k+0.5)/n_depth, so the n_depth bands tile the FULL layup thickness h of
the element and the wall is drawn with its real thickness.  Each quad owns its four
points (exploded mesh) and carries the stress of its own recovery station as
CELL_DATA -- never averaged onto nodes.

Corner k of element e: p_c + (zeta_edge - frac)*h * e3, with p_c the two contour
nodes, e3 the inward wall normal (OML -> IML) and frac the reference-surface
fraction of the bundle.

In:  beam_dehom_shell (sibling script; reads <name>_shell.yaml + the beam FF table
     from this folder), providing corners (n_nd,2), red_cells (n_el,2), nvec
     (n_el,2), layup thickness h per element, frac float, n_depth int and
     sig (n_el*n_depth, 6) stress in MPa, column order S11 S22 S33 S23 S13 S12
Out: <name>_dehom_shell.vtk -- 4*n_el*n_depth POINTS, n_el*n_depth QUAD cells,
     CELL_DATA scalars S11 S22 S33 S23 S13 S12 (MPa)
"""
import numpy as np

import beam_dehom_shell as R

NAMES = ("S11", "S22", "S33", "S23", "S13", "S12")

corners = R.corners
rc = R.rc
nvec = R.nvec
layups = R.layups
hth = R.hth
frac = R.frac
n_el = R.n_el
n_depth = R.n_depth
sig = R.sig
out = R.name + "_dehom_shell.vtk"

n_cell = n_el * n_depth
edge = np.arange(n_depth + 1) / float(n_depth)          # band bounds, 0=OML..1=IML
xyz = np.zeros((4 * n_cell, 3))
val = np.zeros((n_cell, 6))
for e in range(n_el):
    h = hth[layups[e]]
    p0 = corners[int(rc[e, 0])]
    p1 = corners[int(rc[e, 1])]
    nv = nvec[e]
    for k in range(n_depth):
        c = e * n_depth + k
        zlo = (edge[k] - frac) * h
        zhi = (edge[k + 1] - frac) * h
        xyz[4 * c + 0, :2] = p0 + zlo * nv
        xyz[4 * c + 1, :2] = p1 + zlo * nv
        xyz[4 * c + 2, :2] = p1 + zhi * nv
        xyz[4 * c + 3, :2] = p0 + zhi * nv
        val[c] = sig[c]

conn = np.column_stack([np.full(n_cell, 4, int),
                        np.arange(4 * n_cell, dtype=int).reshape(n_cell, 4)])
with open(out, "w") as f:
    f.write("# vtk DataFile Version 3.0\n"
            "OpenSG %s RM shell dehom, exploded through-thickness quads, "
            "stress in MPa\nASCII\nDATASET UNSTRUCTURED_GRID\n" % R.name)
    f.write("POINTS %d float\n" % (4 * n_cell))
    np.savetxt(f, xyz, fmt="%.6e")
    f.write("\nCELLS %d %d\n" % (n_cell, 5 * n_cell))
    np.savetxt(f, conn, fmt="%d")
    f.write("\nCELL_TYPES %d\n" % n_cell)
    np.savetxt(f, np.full(n_cell, 9, int), fmt="%d")
    f.write("\nCELL_DATA %d\n" % n_cell)
    for i, nm in enumerate(NAMES):
        f.write("SCALARS %s float 1\nLOOKUP_TABLE default\n" % nm)
        np.savetxt(f, val[:, i], fmt="%.6e")

print("wrote %s: %d QUAD cells (%d elements x %d depth bands), %d points"
      % (out, n_cell, n_el, n_depth, 4 * n_cell))
print("CELL_DATA arrays: %s" % ", ".join(NAMES))
for i, nm in enumerate(NAMES):
    print("  %-4s min %12.4f  max %12.4f  MPa" % (nm, val[:, i].min(),
                                                  val[:, i].max()))
