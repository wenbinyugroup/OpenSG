"""(1) Inspect the ACTUAL PreVABS square-tube 2-D solid mesh corner: which
wall family owns the corner material (mitre / horizontal-through /
vertical-through)?  (2) Re-run the m45 merged-X microcell with each fill.

Run (from this folder):  python corner_inspect_fill.py
"""
import numpy as np
import yaml as _yaml

from opensg_shell.sg_mesh import load_ring_ref
from opensg_shell.sg_junction import microcell_law
from opensg_shell.sg_materials import material_db_from_yaml
from opensg_solid.rm_plate_1D.msg_rm_plate import rm_plate_msg

SOLID_YAML = "../square_tube_2Dsolid_t1only.yaml"
M45_YAML = "../square_tube_1Dshell.yaml"
a, t = 1.0, 0.03

d = _yaml.safe_load(open(SOLID_YAML))
row = lambda r: [float(v) for v in
                 " ".join(str(x) for x in (r if isinstance(r, list) else [r])).split()]
nd = np.array([row(r)[:2] for r in d["nodes"]], float)
tri = np.array([[int(v) for v in row(r)] for r in d["elements"]], int) - 1
ori = np.array(d["elementOrientations"], float)
cen = nd[tri].mean(1)
corner = np.array([a/2, a/2])
near = np.where(np.linalg.norm(cen - corner, axis=1) < 1.2*t)[0]
print("elements within 1.2t of corner (+a/2,+a/2): %d" % len(near))
print("  %-6s %8s %8s   e2 (tangent, yaml cols 3:6)" % ("elem", "cx", "cy"))
for e in near[:24]:
    print("  %-6d %8.4f %8.4f   [%+.3f %+.3f %+.3f]"
          % (e, cen[e, 0], cen[e, 1], ori[e, 3], ori[e, 4], ori[e, 5]))
hcnt = sum(1 for e in near if abs(ori[e, 3]) > abs(ori[e, 4]))
vcnt = len(near) - hcnt
print("tangent mostly-horizontal: %d   mostly-vertical: %d" % (hcnt, vcnt))

# ---- m45 merged-X microcell under each fill -------------------------------
d1 = _yaml.safe_load(open(M45_YAML))
R = load_ring_ref(M45_YAML, "center")
pl = [[str(p[0]), float(p[1]), float(p[2])]
      for p in d1["sections"][0]["layup"]]
G_by = [np.asarray(rm_plate_msg([p[1] for p in pl], [p[2] for p in pl],
                                [p[0] for p in pl],
                                material_db_from_yaml(d1["materials"]),
                                fraction=0.5)["G_msg"])]
stack = [(0, False), (0, True)]
print("\nm45 merged-X microcell dC per fill (Lfac=16.67, ~full cell):")
print("  %-8s %12s %12s %12s %12s" % ("fill", "dC22", "dC33", "dC23", "dC12"))
for fill in ("mitre4", "A", "B"):
    dC, info = microcell_law(stack, stack, d1["sections"], d1["materials"],
                             R["D_by"], G_by, fill=fill, Lfac=16.67)
    print("  %-8s %12.4e %12.4e %12.4e %12.4e"
          % (fill, dC[1, 1], dC[2, 2], dC[1, 2], dC[0, 1]))
print("target dC23 (msg_solid benchmark - shell) ~ 1.33e8")
