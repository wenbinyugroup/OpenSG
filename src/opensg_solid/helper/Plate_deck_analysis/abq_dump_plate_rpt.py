# abq_dump_plate_rpt.py -- Plate_deck_analysis step05's PLATE dumper.
# VENDORED VERBATIM (2026-09-03) from the validated TPMS pressure_case
# study:
#   M:\Abaqus\pressure_case\abaqus_plate_2d\opensg_dehomo\dump_plate_rpt.py
# Everything below this header is that file, unchanged.  It runs ONLY
# under `abaqus python` (odbAccess) and is standalone on purpose -- no
# opensg imports; step05_dump_rpts.py subprocess-launches it.  Fix bugs
# upstream in pressure_case first, then re-vendor.
"""dump_plate_rpt.py -- ONE self-contained report of a plate job:
element table and node table, each carrying its own COORDINATES, so no
separate coordinate key is needed.

    abaqus python dump_plate_rpt.py <job>          # job in $PWD
    abaqus python dump_plate_rpt.py <dir>/<job>

Out: <job>_plate.rpt with two tables

  ELEMENT TABLE (centroidal, one row per element)
      Element Label, Xc, Yc, SF.*, SM.*, SE.*, SK.*
      SF1/SF2/SF3 = N11/N22/N12      [N/m]   membrane
      SF4/SF5     = Q1/Q2            [N/m]   transverse shear
      SM1/SM2/SM3 = M11/M22/M12      [N]     bending + twist
      SE          = section STRAINS: membrane + transverse shear.
                    It does NOT contain the curvatures -- verified from
                    the odb, where SE's last two components equal Q/G.
      SK          = section CURVATURES kappa11/kappa22/kappa12.  These
                    are what the V2 shear-refined recovery needs;
                    requesting SE alone silently loses them.

  NODE TABLE (one row per node)
      Node Label, X, Y, U.U1..U3, UR.UR1..UR3

Abaqus permutes componentLabels for these fields (SM2 before SM1) and
SE's label array even repeats 'SE3', so names are zipped to data from
the odb rather than assumed positional.

RF is deliberately NOT reported: it is the support reaction, identically
zero anywhere without a support, and nothing in the recovery reads it.
The one thing it was good for -- global equilibrium, sum(RF3) = applied
weight -- is checked here and printed, then dropped.

WHY CENTROIDAL.  Element fields sit at the INTEGRATION POINTS in the
odb -- 4 per S8R.  A reader keyed on element label would silently keep
whichever it saw last, a point offset ~0.29 of the element from the
centre.  At the plate centre that turns a transverse shear which must
vanish by symmetry into ~1.1e3 N/m.  getSubset(position=CENTROID)
gives one row per element, at the centroid.
"""
import os
import sys

from abaqusConstants import CENTROID
from odbAccess import openOdb

JOB = sys.argv[1]
odb = openOdb(JOB + ".odb", readOnly=True)
inst = odb.rootAssembly.instances[odb.rootAssembly.instances.keys()[0]]
step = odb.steps.keys()[-1]
frame = odb.steps[step].frames[-1]
print("instance %s: %d nodes, %d elements  (step %s, last frame)"
      % (inst.name, len(inst.nodes), len(inst.elements), step))

xy = {}
for nd in inst.nodes:
    xy[nd.label] = nd.coordinates

ecen = {}
for el in inst.elements:
    c = [xy[k] for k in el.connectivity[:4] if k in xy]
    if c:
        ecen[el.label] = (sum(p[0] for p in c) / len(c),
                          sum(p[1] for p in c) / len(c))


def grab(name, by):
    """One field -> {label: [values]} plus its component labels.

    In:  name str; by 'element'|'node'
    Out: (cols [str], data {label: [float]})."""
    if name not in frame.fieldOutputs.keys():
        return [], {}
    f = frame.fieldOutputs[name]
    if by == "element":
        f = f.getSubset(position=CENTROID)
    cols = ["%s.%s" % (name, c) for c in f.componentLabels]
    data = {}
    for v in f.values:
        data[v.elementLabel if by == "element" else v.nodeLabel] = \
            list(v.data)
    return cols, data


ecols, edata = [], {}
for nm in ("SF", "SM", "SE", "SK"):
    c, d = grab(nm, "element")
    ecols += c
    for k, v in d.items():
        edata.setdefault(k, []).extend(v)

ncols, ndata = [], {}
for nm in ("U", "UR"):                  # RF deliberately excluded
    c, d = grab(nm, "node")
    ncols += c
    for k, v in d.items():
        ndata.setdefault(k, []).extend(v)

# global equilibrium is the one check needing no reference solution, so
# RF is read for that and then discarded rather than reported
_, rf = grab("RF", "node")

out = JOB + "_plate.rpt"
g = open(out, "w")
g.write("** Abaqus plate result -- %s\n" % os.path.basename(JOB))
g.write("** step %s, last frame.  Element rows are CENTROIDAL.\n" % step)
g.write("** SF1/2/3 = N11/N22/N12 [N/m];  SF4/5 = Q1/Q2 [N/m];\n")
g.write("** SM1/2/3 = M11/M22/M12 [N];\n")
g.write("** SE = section strains (membrane + transverse shear, NOT"
        " curvature);\n")
g.write("** SK = section curvatures kappa11/kappa22/kappa12.\n")
g.write("** RF is not reported -- see the module docstring.\n")
g.write("**\n** ---------------- ELEMENT TABLE ----------------\n")
g.write("%16s  %16s  %16s  " % ("Element Label", "Xc", "Yc")
        + "  ".join("%16s" % c for c in ecols) + "\n")
for lab in sorted(edata):
    c = ecen.get(lab, (0.0, 0.0))
    g.write("%16d  %16.8e  %16.8e  " % (lab, c[0], c[1])
            + "  ".join("%16.8e" % x for x in edata[lab]) + "\n")
g.write("**\n** ------------- INTEGRATION POINT TABLE -------------\n")
g.write("** The 4 in-plane integration points of each S8R.  These carry\n")
g.write("** the in-plane GRADIENT of the resultants inside a single\n")
g.write("** element, so d/dx and d/dy at a cell centre can be formed\n")
g.write("** from ONE element -- no refinement, so one element still\n")
g.write("** equals one unit cell.\n")
g.write("** CAUTION: Xc/Yc below are the ELEMENT CENTROID, not the\n")
g.write("** integration point.  For a rectangular S8R the points sit\n")
g.write("** at xc +- (dx/2)/sqrt(3), yc +- (dy/2)/sqrt(3), but the\n")
g.write("** index -> quadrant mapping is NOT assumed here.  Request\n")
g.write("** COORD in *Element Output and use it before differencing.\n")
ipc, ipd = [], {}
for nm in ("SF", "SM", "SK"):
    if nm not in frame.fieldOutputs.keys():
        continue
    f = frame.fieldOutputs[nm]
    ipc += ["%s.%s" % (nm, c) for c in f.componentLabels]
    for v in f.values:
        key = (v.elementLabel, v.integrationPoint)
        ipd.setdefault(key, []).extend(list(v.data))
g.write("%16s  %6s  %16s  %16s  " % ("Element Label", "IP", "Xc", "Yc")
        + "  ".join("%16s" % c for c in ipc) + "\n")
for (lab, ip) in sorted(ipd):
    c = ecen.get(lab, (0.0, 0.0))
    g.write("%16d  %6d  %16.8e  %16.8e  " % (lab, ip, c[0], c[1])
            + "  ".join("%16.8e" % x for x in ipd[(lab, ip)]) + "\n")
g.write("**\n** ---------------- NODE TABLE ----------------\n")
g.write("%16s  %16s  %16s  " % ("Node Label", "X", "Y")
        + "  ".join("%16s" % c for c in ncols) + "\n")
for lab in sorted(ndata):
    c = xy.get(lab, (0.0, 0.0, 0.0))
    g.write("%16d  %16.8e  %16.8e  " % (lab, c[0], c[1])
            + "  ".join("%16.8e" % x for x in ndata[lab]) + "\n")
g.close()
print("wrote %s : %d elements x %d cols, %d nodes x %d cols"
      % (os.path.basename(out), len(edata), len(ecols), len(ndata),
         len(ncols)))

# global equilibrium: the reactions must carry the whole applied load
if rf:
    tot = sum(v[2] for v in rf.values())
    print("SUM of RF3 over every node = %+.6e N" % tot)
    print("  (must equal the total applied weight; the one check that"
          " needs no reference solution)")
odb.close()
