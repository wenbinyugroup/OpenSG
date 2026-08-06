"""One-time converter: UDcomp_2D.msh (gmsh 2.2) -> UDcomp_2D.yaml (OpenSG dialect).

Writes nodes as '- [x y z]' rows, elements as '- [n1 n2 n3]' (1-based), constant
elementOrientations (e1 = axial z, e2 = +x, e3 = +y), the two materials with
engineering constants (MPa), and sets.element for matrix/fiber phases.
Run (from this folder):  python make_udcomp_yaml.py
"""
import numpy as np

MSH = "UDcomp_2D.msh"
lines = open(MSH).read().split("\n")
i = lines.index("$Nodes"); nn = int(lines[i+1])
nd = np.array([[float(v) for v in lines[i+2+k].split()[1:4]] for k in range(nn)])
i = lines.index("$Elements"); ne_all = int(lines[i+1])
tris, phys = [], []
for k in range(ne_all):
    p = lines[i+2+k].split()
    if int(p[1]) == 2:
        phys.append(int(p[3]))
        tris.append([int(p[-3]), int(p[-2]), int(p[-1])])
phys = np.array(phys)

out = ["nodes:"]
for x, y, z in nd:
    out.append("- [%.12f %.12f %.12f]" % (x, y, z))
out.append("elements:")
for a, b, c in tris:
    out.append("- [%d %d %d]" % (a, b, c))
out.append("elementOrientations:")
for _ in tris:
    out.append("- [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]")
out += ["materials:",
        "- name: matrix",
        "  density: 1200.0",
        "  elastic:",
        "    E: [4760.0, 4760.0, 4760.0]",
        "    G: [%.6f, %.6f, %.6f]" % ((4760.0/2/1.37,)*3),
        "    nu: [0.37, 0.37, 0.37]",
        "- name: fiber",
        "  density: 1800.0",
        "  elastic:",
        "    E: [276000.0, 19500.0, 19500.0]",
        "    G: [70000.0, 70000.0, 5735.0]",
        "    nu: [0.28, 0.28, 0.70]",
        "sets:",
        "  element:"]
for name, tag in (("matrix", 0), ("fiber", 1)):
    out.append("  - name: %s" % name)
    out.append("    labels:")
    for e in np.where(phys == tag)[0]:
        out.append("    - %d" % (e + 1))
with open("UDcomp_2D.yaml", "w") as f:
    f.write("\n".join(out) + "\n")
print("wrote UDcomp_2D.yaml  (%d nodes, %d tris, matrix %d / fiber %d)"
      % (nn, len(tris), (phys == 0).sum(), (phys == 1).sum()))
