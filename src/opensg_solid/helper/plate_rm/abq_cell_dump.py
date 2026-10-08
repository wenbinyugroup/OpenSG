"""abq_cell_dump.py -- odb -> centre-of-cell rpt.  RUNS UNDER `abaqus
python` ONLY (odbAccess); fea_cell.py subprocess-calls it, nothing else
should.

    abaqus python abq_cell_dump.py <job> <out.rpt> [cx] [cy] [half]

Keeps the elements whose corner centroid lies within +-half of
(cx, cy) and every node they reference.  Element stress is CENTROIDAL,
one row per element; U nodal.
"""
import os
import sys

import numpy as np
from odbAccess import openOdb

JOB = sys.argv[1]
OUT = sys.argv[2]
CX = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
CY = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
HALF = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5

odb = openOdb(JOB + ".odb", readOnly=True)
inst = odb.rootAssembly.instances[odb.rootAssembly.instances.keys()[0]]
fr = odb.steps[odb.steps.keys()[-1]].frames[-1]

nlab = np.array([n.label for n in inst.nodes])
nxyz = np.array([n.coordinates for n in inst.nodes], float)
nmap = np.zeros(int(nlab.max()) + 1, np.int64)
nmap[nlab] = np.arange(len(nlab))

elems, used = [], set()
for e in inst.elements:
    c = list(e.connectivity)
    m = nxyz[nmap[c[:4]]].mean(axis=0)
    if abs(m[0] - CX) <= HALF and abs(m[1] - CY) <= HALF:
        elems.append((e.label, m))
        used.update(c)
keep = set(l for l, _ in elems)
print("cell (%+.3f, %+.3f): %d elements, %d nodes"
      % (CX, CY, len(elems), len(used)))

sv = {}
for v in fr.fieldOutputs["S"].values:
    if v.elementLabel in keep:
        sv[v.elementLabel] = list(v.data)
uv = {}
for v in fr.fieldOutputs["U"].values:
    if v.nodeLabel in used:
        uv[v.nodeLabel] = list(v.data)

with open(OUT, "w") as f:
    f.write("** 3-D FEA cell at (%+.4f, %+.4f) -- job %s, step %s\n"
            % (CX, CY, os.path.basename(JOB), odb.steps.keys()[-1]))
    # S comes back in whatever csys the section carries: GLOBAL for an
    # elset without *Orientation, the section's MATERIAL csys for one
    # with (the HC deck attaches ORI2/ORI3 = about-3 +-45 to its ply
    # elsets, so those rows are PLY-frame).  The old line asserted
    # "GLOBAL" unconditionally; a cell integral that trusted it mixed
    # frames (2026-09-07).
    f.write("** element stress CENTROIDAL; frame AS ABAQUS REPORTS IT: "
            "GLOBAL for elsets without *Orientation, the section's "
            "MATERIAL csys for elsets that carry one -- check the deck\n")
    f.write("** ---------------- ELEMENT TABLE ----------------\n")
    f.write("%12s %14s %14s %14s" % ("Element", "Xc", "Yc", "Zc")
            + "".join(" %14s" % c for c in
                      ("S11", "S22", "S33", "S12", "S13", "S23")) + "\n")
    for lab, m in sorted(elems):
        s = sv.get(lab)
        if s is None:
            continue
        f.write("%12d %14.6e %14.6e %14.6e" % (lab, m[0], m[1], m[2])
                + "".join(" %14.6e" % x for x in s) + "\n")
    f.write("** ---------------- NODE TABLE ----------------\n")
    f.write("%12s %14s %14s %14s %14s %14s %14s\n"
            % ("Node", "X", "Y", "Z", "U1", "U2", "U3"))
    for lab in sorted(used):
        c = nxyz[nmap[lab]]
        u = uv.get(lab, [0.0, 0.0, 0.0])
        f.write("%12d %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e\n"
                % (lab, c[0], c[1], c[2], u[0], u[1], u[2]))
print("wrote %s" % OUT)
odb.close()
