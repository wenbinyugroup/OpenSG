"""Convert the Schwarz-P TPMS shell mesh (gmsh 2.2 triangles) to the OpenSG
3-D shell SG yaml.  Aluminum matching the SwiftComp .sc (E = 69 GPa,
nu = 0.3); shell thickness T below (relative density = area x T, reported).

Run (from this folder):  python make_schwarz_yaml.py"""
import numpy as np

MSH = ("../../../tests/08072026_square_shell_mesh/TPMS_SC_files/"
       "schwarz_p_D2_shell.msh")
T = 0.036457           # Sample_2, volume-consistent
E, nu = 69.0e9, 0.30
G = E/(2*(1+nu))

lines = open(MSH).read().split("\n")
i_n = lines.index("$Nodes")
nn = int(lines[i_n+1])
nd = np.array([[float(v) for v in lines[i_n+2+k].split()[1:4]]
               for k in range(nn)])
i_e = lines.index("$Elements")
ne = int(lines[i_e+1])
el = []
for k in range(ne):
    p = lines[i_e+2+k].split()
    ntag = int(p[2])
    el.append([int(v) for v in p[3+ntag:]])
el = np.array(el, int)                          # 3-node tris, 1-based

p1, p2, p3 = nd[el[:, 0]-1], nd[el[:, 1]-1], nd[el[:, 2]-1]
nrm = np.cross(p2-p1, p3-p1)
area = 0.5*np.linalg.norm(nrm, axis=1)
e3 = nrm/(2*area)[:, None]
e2 = (p2-p1)/np.linalg.norm(p2-p1, axis=1)[:, None]
print("%d nodes, %d tris; surface area %.4f; t=%.3f -> relative density"
      " %.4f" % (nn, ne, area.sum(), T, area.sum()*T))

o = ["nodes:"]
o += ["- [%.12f %.12f %.12f]" % tuple(p) for p in nd]
o.append("elements:")
o += ["- [%d %d %d]" % tuple(e) for e in el]
o.append("elementOrientations:")
for k in range(ne):
    e1 = np.cross(e2[k], e3[k])
    o.append("- [%.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f]"
             % (*e1, *e2[k], *e3[k]))
o += ["sections:", "- type: shell", "  elementSet: layup_0", "  layup:",
      "  - - alu", "    - %.6e" % T, "    - 0.0",
      "materials:", "- name: alu", "  density: 2700.0", "  elastic:",
      "    E: [%.6e, %.6e, %.6e]" % (E, E, E),
      "    G: [%.6e, %.6e, %.6e]" % (G, G, G),
      "    nu: [%.6f, %.6f, %.6f]" % (nu, nu, nu),
      "sets:", "  element:", "  - name: layup_0", "    labels:"]
o += ["    - %d" % (e+1) for e in range(ne)]
open("schwarz_p_3Dshell.yaml", "w").write("\n".join(o) + "\n")
print("wrote schwarz_p_3Dshell.yaml")
