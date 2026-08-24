"""abq_dump_45pm_linear.py -- dump stress and displacement from the pm45 run.

    abaqus python abq_dump_45pm_linear.py

STRESS IS TRANSFORMED TO THE GLOBAL FRAME.  With *Orientation on a solid
section, Abaqus stores S in the MATERIAL system, so the +45 and -45 face
plies would otherwise arrive rotated while the aluminium core arrives
global -- mixing frames within one cross-section and corrupting any path
or integral.  getTransformedField against a global Cartesian datum puts
every element in one frame.  Both are written so the difference is
auditable.

Out: 45pm_S_global_linear.csv    elem,x,y,z,S11,S22,S33,S12,S13,S23
     45pm_S_material_linear.csv  same, as stored (material frame)
     45pm_U_linear.csv           node,x,y,z,U1,U2,U3
"""
import os

from abaqusConstants import CARTESIAN
from odbAccess import openOdb

ODB = "45pm_L_min_pm45_linear.odb"
odb = openOdb(ODB, readOnly=True)
inst = odb.rootAssembly.instances[odb.rootAssembly.instances.keys()[0]]
print("instance %s: %d nodes, %d elements"
      % (inst.name, len(inst.nodes), len(inst.elements)))

xyz = {}
for n in inst.nodes:
    xyz[n.label] = n.coordinates
xs = [c[0] for c in xyz.values()]
ys = [c[1] for c in xyz.values()]
zs = [c[2] for c in xyz.values()]
print("bbox x %.5f..%.5f  y %.5f..%.5f  z %.5f..%.5f"
      % (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)))
print("span mid-plane x = %.6f" % (0.5 * (min(xs) + max(xs))))

cen = {}
for e in inst.elements:
    cx = cy = cz = 0.0
    m = len(e.connectivity)
    for nl in e.connectivity:
        c = xyz[nl]
        cx += c[0]
        cy += c[1]
        cz += c[2]
    cen[e.label] = (cx / m, cy / m, cz / m)

step = odb.steps[odb.steps.keys()[-1]]
frame = step.frames[-1]
print("step %s, frame %d" % (step.name, len(step.frames) - 1))

csys = odb.rootAssembly.DatumCsysByThreePoints(
    name="GLOBAL_CSYS", coordSysType=CARTESIAN, origin=(0.0, 0.0, 0.0),
    point1=(1.0, 0.0, 0.0), point2=(0.0, 1.0, 0.0))

S = frame.fieldOutputs["S"]
try:
    Sg = S.getTransformedField(datumCsys=csys)
    print("stress transformed to GLOBAL")
except Exception as e:                                      # noqa: BLE001
    Sg = None
    print("TRANSFORM FAILED (%s) -- global csv will be skipped" % e)


def write(field, path):
    g = open(path, "w")
    g.write("elem,x,y,z,S11,S22,S33,S12,S13,S23\n")
    n = 0
    for v in field.values:
        c = cen.get(v.elementLabel)
        if c is None:
            continue
        d = v.data
        g.write("%d,%.6f,%.6f,%.6f,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e\n"
                % (v.elementLabel, c[0], c[1], c[2],
                   d[0], d[1], d[2], d[3], d[4], d[5]))
        n += 1
    g.close()
    print("wrote %s : %d rows" % (path, n))


if Sg is not None:
    write(Sg, "45pm_S_global_linear.csv")
write(S, "45pm_S_material_linear.csv")

U = frame.fieldOutputs["U"]
g = open("45pm_U_linear.csv", "w")
g.write("node,x,y,z,U1,U2,U3\n")
n = 0
for v in U.values:
    c = xyz.get(v.nodeLabel)
    if c is None:
        continue
    d = v.data
    g.write("%d,%.6f,%.6f,%.6f,%.8e,%.8e,%.8e\n"
            % (v.nodeLabel, c[0], c[1], c[2], d[0], d[1], d[2]))
    n += 1
g.close()
odb.close()
print("wrote 45pm_U_linear.csv : %d rows" % n)
