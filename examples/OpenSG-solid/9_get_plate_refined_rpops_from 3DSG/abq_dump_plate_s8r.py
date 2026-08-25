"""abq_dump_plate_s8r.py -- SF/SM element reports + U of an S8R ABDG
plate job, in exactly the shape build_ff.read_rpt parses (header line
carrying `Element Label` and `SF.SF1`-style names, ONE numeric row per
element, columns taken BY NAME).

WHY NOT abq_dump_plate.py: S8R stores SF/SM at its 4 integration
points, so the stock dump writes 4 rows per element -- and 25 x 4 =
100 rows would even PASS ff_grid's square-grid assert as a bogus
10 x 10.  This dump AVERAGES the 4 section values per element label;
SF/SM vary linearly inside a quadratic element, so the Gauss-point
average IS the centroid value.  Rows come out sorted by element label
(= the generators' e = j*NE + i + 1 grid rule).

    abaqus python abq_dump_plate_s8r.py <job>

Out: <job>_SF.rpt   Element Label + every SF component (SF1..SF3
                    membrane, SF4/SF5 transverse shear)
     <job>_SM.rpt   Element Label + SM1/SM2/SM3
     <job>_U.csv    node,x,y,U1,U2,U3,UR1,UR2,UR3 (corner + midside)
"""
import sys

from odbAccess import openOdb

JOB = sys.argv[1]
odb = openOdb(JOB + ".odb", readOnly=True)
inst = odb.rootAssembly.instances[odb.rootAssembly.instances.keys()[0]]
frame = odb.steps[odb.steps.keys()[-1]].frames[-1]
print("instance %s: %d nodes, %d elements"
      % (inst.name, len(inst.nodes), len(inst.elements)))


def rpt(name, path):
    f = frame.fieldOutputs[name]
    labs = list(f.componentLabels)
    acc, cnt = {}, {}
    for v in f.values:
        e = v.elementLabel
        if e in acc:
            a = acc[e]
            for k in range(len(a)):
                a[k] += v.data[k]
            cnt[e] += 1
        else:
            acc[e] = [float(x) for x in v.data]
            cnt[e] = 1
    g = open(path, "w")
    g.write("   Element Label  " +
            "  ".join("%16s" % ("%s.%s" % (name, c)) for c in labs) + "\n")
    for e in sorted(acc.keys()):
        g.write("%16d  " % e +
                "  ".join("%16.8e" % (x / cnt[e]) for x in acc[e]) + "\n")
    g.close()
    ips = sorted(set(cnt.values()))
    print("wrote %s : %d rows (ONE per element), %s integration point"
          " values averaged per element, columns %s"
          % (path, len(acc), ips, labs))


rpt("SF", JOB + "_SF.rpt")
rpt("SM", JOB + "_SM.rpt")

xy = {}
for nd in inst.nodes:
    xy[nd.label] = nd.coordinates
U = frame.fieldOutputs["U"]
UR = frame.fieldOutputs["UR"]
ur = {}
for v in UR.values:
    ur[v.nodeLabel] = v.data
g = open(JOB + "_U.csv", "w")
g.write("node,x,y,U1,U2,U3,UR1,UR2,UR3\n")
n = 0
for v in U.values:
    c = xy[v.nodeLabel]
    r = ur.get(v.nodeLabel, (0.0, 0.0, 0.0))
    g.write("%d,%.6f,%.6f,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e\n"
            % (v.nodeLabel, c[0], c[1], v.data[0], v.data[1], v.data[2],
               r[0], r[1], r[2]))
    n += 1
g.close()
odb.close()
print("wrote %s_U.csv : %d rows" % (JOB, n))
