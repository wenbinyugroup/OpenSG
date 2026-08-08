"""(1) 8-ply SYMMETRIC laminate [0/45/-45/90]s on the cross cell:
shell+microcell vs ply-resolved full-cell periodic solid.
(2) junction_mechanism.png -- how the junction addition works, drawn from the
REAL m45 merged-X microcell meshes (no new global elements: dC is an energy
matrix added to Dee).

Run (from this folder):  python microcell_8ply_sym_fig.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml as _yaml

from opensg_shell import build_solid_bundle
from opensg_shell.sg_mesh import load_ring_ref
from opensg_shell.sg_junction import microcell_law
from opensg_shell.sg_materials import material_db_from_yaml
from opensg_solid.rm_plate_1D.msg_rm_plate import rm_plate_msg

############### User Input #################################
a = 1.0
PLY = [0.0, 45.0, -45.0, 90.0, 90.0, -45.0, 45.0, 0.0]   # symmetric
TP = 0.00375
E1, E2, G12, nu12, nu23 = 142.0e9, 9.8e9, 9.8e9/(2*(1+0.42))*0 + 6.0e9, 0.30, 0.42
M45_YAML = "../square_tube_1Dshell.yaml"
############################################################

G23 = E2/(2*(1 + nu23))
TERMS = [("D11", 0, 0), ("D22", 1, 1), ("D33", 2, 2), ("D12", 0, 1),
         ("D13", 0, 2), ("D23", 1, 2)]


def msg_G(sections, materials):
    pl = [[str(p[0]), float(p[1]), float(p[2])] for p in sections[0]["layup"]]
    rr = rm_plate_msg([p[1] for p in pl], [p[2] for p in pl],
                      [p[0] for p in pl], material_db_from_yaml(materials),
                      fraction=0.5)
    return np.asarray(rr["G_msg"])


# ---- 8-ply symmetric cross yaml -------------------------------------------
nseg = 40
xs = np.linspace(0.0, a, nseg+1)
c = a/2
h_n = [(x, c) for x in xs]
v_n = [(c, y) for y in xs if abs(y - c) > 1e-12]
pts = h_n + v_n
o = ["nodes:"]
for x, y in pts:
    o.append("- [%.12f %.12f 0.00000000]" % (x, y))
vid = {}
k = len(h_n)
for j, y in enumerate(xs):
    if abs(y - c) < 1e-12:
        vid[j] = nseg//2
    else:
        vid[j] = k; k += 1
cells = ([[i, i+1] for i in range(nseg)]
         + [[vid[j], vid[j+1]] for j in range(nseg)])
o.append("elements:")
for a_, b_ in cells:
    o.append("- [%d %d]" % (a_+1, b_+1))
o.append("elementOrientations:")
o += ["- [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]"]*nseg
o += ["- [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0]"]*nseg
o += ["sections:", "- type: shell", "  elementSet: layup_0", "  layup:"]
for ang in PLY:
    o += ["  - - pl", "    - %.6e" % TP, "    - %.1f" % ang]
o += ["materials:", "- name: pl", "  density: 1800.0", "  elastic:",
      "    E: [%.6e, %.6e, %.6e]" % (E1, E2, E2),
      "    G: [%.6e, %.6e, %.6e]" % (G12, G12, G23),
      "    nu: [%.6f, %.6f, %.6f]" % (nu12, nu12, nu23),
      "sets:", "  element:", "  - name: layup_0", "    labels:"]
o += ["    - %d" % (e+1) for e in range(len(cells))]
o.append("reference: center")
open("_ply8s_cross.yaml", "w").write("\n".join(o) + "\n")

d8 = _yaml.safe_load(open("_ply8s_cross.yaml"))
R8 = load_ring_ref("_ply8s_cross.yaml", "center")
G8 = [msg_G(d8["sections"], d8["materials"])]
s1 = [(0, False)]
b0 = build_solid_bundle("_ply8s_cross.yaml", cell_area=a*a)
b3 = build_solid_bundle("_ply8s_cross.yaml", cell_area=a*a,
                        junction="microcell")
t8 = 8*TP
_, info = microcell_law(s1, s1, d8["sections"], d8["materials"],
                        R8["D_by"], G8, fill="A", Lfac=a/t8, nply_el=2,
                        along_fac=1.0)
Db = info["D_solid_mini"]
print("==== 8-ply SYMMETRIC [0/45/-45/90]s cross cell ====")
print("  %-5s %13s %13s %9s %9s"
      % ("term", "shell+cell", "solid bench", "off/so", "cell/so"))
for nm, i, j in TERMS:
    so = Db[i, j]
    thr = 1e-3*abs(Db[0, 0])
    r0 = b0["D_eff"][i, j]/so if abs(so) > thr else np.inf
    r3 = b3["D_eff"][i, j]/so if abs(so) > thr else np.inf
    print("  %-5s %13.5e %13.5e %9.4f %9.4f"
          % (nm, b3["D_eff"][i, j], so, r0, r3))

# ---- junction mechanism figure (real m45 microcell meshes) ----------------
d1 = _yaml.safe_load(open(M45_YAML))
R = load_ring_ref(M45_YAML, "center")
Gm = [msg_G(d1["sections"], d1["materials"])]
stack = [(0, False), (0, True)]
dC, mi = microcell_law(stack, stack, d1["sections"], d1["materials"],
                       R["D_by"], Gm, fill="mitre4", Lfac=6.0)

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
rx = R["rx"]
ax[0].plot(rx[:, 0], rx[:, 1], ".", ms=2, color="0.4")
for x, y in ((0.5, 0.5), (-0.5, 0.5), (-0.5, -0.5), (0.5, -0.5)):
    ax[0].plot([x], [y], "o", ms=9, mfc="none", mec="crimson", mew=2)
ax[0].set_aspect("equal")
ax[0].set_xlabel("global 1-D midline mesh; 4 corner nodes merge\n"
                 "periodically into ONE X-junction (crimson)")

nd, tris, own = mi["nd"], mi["tris"], mi["own"]
colA = np.array([0.85, 0.55, 0.25])
colB = np.array([0.30, 0.55, 0.80])
for e in range(len(tris)):
    p = nd[tris[e]]
    ax[1].fill(p[:, 0], p[:, 1],
               color=(colA if own[e] == "A" else colB), ec="0.3", lw=0.15)
ax[1].set_aspect("equal")
ax[1].set_xlabel("SOLID mini-cell (periodic, ply-resolved,\n"
                 "merged 2t walls, mitre4 block) -> D_solid_mini")

sp, sc = mi["shell_pts"], mi["shell_cells"]
for k, (n1, n2) in enumerate(sc):
    off = 0.004*(1 if k % 2 == 0 else -1)
    p1, p2 = sp[n1], sp[n2]
    if abs(p1[1] - p2[1]) < 1e-9:
        ax[2].plot([p1[0], p2[0]], [p1[1]+off, p2[1]+off], "-",
                   color="0.2", lw=1)
    else:
        ax[2].plot([p1[0]+off, p2[0]+off], [p1[1], p2[1]], "-",
                   color="0.2", lw=1)
ax[2].plot(sp[:, 0], sp[:, 1], ".", ms=3, color="crimson")
ax[2].set_aspect("equal")
ax[2].set_xlabel("SHELL mini-cell (production ring_solid, stacked\n"
                 "coincident midline elements) -> D_shell_mini")
fig.text(0.5, 0.965,
         "junction addition:  dC = D_solid_mini - D_shell_mini  "
         "(normal block + scaled D44)  ->  ADDED TO Dee.\n"
         "NO new element or DOF in the global model: the correction couples "
         "only through the macro strain channels.",
         ha="center", va="top", fontsize=10)
plt.tight_layout(rect=(0, 0, 1, 0.9))
plt.savefig("junction_mechanism.png", dpi=160)
print("wrote junction_mechanism.png")
