"""1d_sg.py -- layup_db.yaml  ->  1dsg.yaml  ->  <layup_db>.out

    input   layup_db.yaml   the user's laminate: fraction, materials
                            (with density), stacking sequence, and
                            model = 0 (classical ABD) or 1 (8x8 ABDG)
    mesh    1dsg.yaml       the through-thickness 1-D SG, written by
                            segment_plate.plate_sg_yaml (+ 1dsg.png)
    output  <layup_db>.out  the OpenSG plate homogenization

Run:  python 1d_sg.py [layup_db.yaml]
"""
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
try:
    import opensg_solid                      # pip install -e . -- nothing to do
except ImportError:                          # fall back to the in-repo source tree
    ROOT = HERE
    while not os.path.isdir(os.path.join(ROOT, "src", "opensg_solid")):
        parent = os.path.dirname(ROOT)
        if parent == ROOT:                   # hit the filesystem root
            raise ImportError(
                "opensg_solid not installed and no src/ found above " + HERE)
        ROOT = parent
    sys.path.insert(0, os.path.join(ROOT, "src"))

import time as _t
print("start: " + _t.strftime("%Y-%m-%d %H:%M:%S"))

from opensg_solid.rm_plate_1D.segment_plate import plate_sg_yaml, read_plate_sg_yaml
from opensg_solid.rm_plate_1D.msg_rm_plate import rm_plate_msg

# ----------------------------------------------------------------------------
# ALL VARIABLES USED IN THIS SCRIPT
# ----------------------------------------------------------------------------
#   HERE, ROOT   this folder; the repo root (walk up until opensg_jax/)
#   ROWS         row/column labels of the plate matrix: 3 membrane strains,
#                3 curvatures, then the 2 transverse shears of the 8x8
#   DB, db       layup_db.yaml path (argv[1] overrides) and its parsed dict
#   model        0 = classical ABD only, 1 = shear-refined 8x8 ABDG
#   fraction     reference surface as a fraction of thickness, from the YAML
#   mesh         db["mesh"]; n_per_layer  SG elements per ply;
#   elem_order   SG element order (4 = 5-noded quartic)
#   material_db  materials with every number coerced to float (the PyYAML
#                "128.0e9 is a string" trap) + rho for the section mass
#   layup        mat_names / thick / angles lists, bottom ply first
#   yml          1dsg.yaml -- the SG mesh file plate_sg_yaml writes (+ .png)
#   inp          that file parsed back (the mesh is the source of truth)
#   r            rm_plate_msg result: A6, ABDG, and the nodal warping
#                ladders V0/V11/V12 the later dehomogenization uses
#   n, M         printed size (6 or 8) and the matrix itself
#   rho_h        section mass per unit area, sum(rho_k t_k) [kg/m^2]
#   out, f       the <layup_db>_plate_homo.out path and its handle
#   k, m, p, t, v, row   throwaway comprehension/loop variables
# ----------------------------------------------------------------------------

ROWS = ("e11", "e22", "g12", "k11", "k22", "k12", "2g13", "2g23")

# ---- input -----------------------------------------------------------------
DB = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "layup_db.yaml")
db = yaml.safe_load(open(DB))
model = int(db["model"])                 # 0 = ABD only, 1 = ABD + shear
fraction = float(db["fraction"])         # no code-side default: the YAML is
                                         # the ONLY place these are set
mesh = db.get("mesh") or {}
n_per_layer = int(mesh.get("n_per_layer", 1))
elem_order = int(mesh.get("elem_order", 4))

material_db = {k: {"E": [float(v) for v in m["E"]],
                   "G": [float(v) for v in m["G"]],
                   "nu": [float(v) for v in m["nu"]],
                   "rho": float(m["rho"]),          # density -> section mass
                   "full_name": m.get("full_name") or k}
               for k, m in db["materials"].items()}
layup = {"mat_names": [p["material"] for p in db["layup"]],
         "thick": [float(p["thickness"]) for p in db["layup"]],
         "angles": [float(p.get("angle", 0.0)) for p in db["layup"]]}

# ---- 1-D SG mesh, straight through segment_plate ---------------------------
yml = os.path.join(HERE, "1dsg.yaml")
plate_sg_yaml(yml, layup, material_db, n_per_layer=n_per_layer,
              elem_order=elem_order, fraction=fraction)

# ---- homogenize ------------------------------------------------------------
inp = read_plate_sg_yaml(yml)
r = rm_plate_msg(inp["thick"], inp["angles"], inp["mat_names"],
                 inp["material_db"], n_per_layer=n_per_layer,
                 elem_order=elem_order, fraction=fraction)
n = 6 if model == 0 else 8
M = np.asarray(r["A6"] if model == 0 else r["ABDG"])
rho_h = sum(inp["material_db"][m]["rho"] * t
            for m, t in zip(inp["mat_names"], inp["thick"]))

# ---- output ----------------------------------------------------------------
out = os.path.splitext(DB)[0] + "_plate_homo.out"
with open(out, "w") as f:
    f.write("OpenSG plate homogenization of %s\n"
            % os.path.basename(yml))
    f.write("%d plies, h = %.6f m, reference fraction = %g\n"
            % (len(inp["thick"]), sum(inp["thick"]), fraction))
    f.write("model %d: %s\n\n"
            % (model, "classical 6x6 ABD" if model == 0
               else "shear-refined 8x8 ABDG"))
    f.write("rows/cols: %s\n" % ", ".join(ROWS[:n]))
    for row in M:
        f.write(" ".join("%14.6e" % v for v in row) + "\n")
    f.write("\nsection mass rho*h = %.6f kg/m^2\n" % rho_h)

print("%s + %s  ->  %s" % (os.path.basename(DB), os.path.basename(yml),
                           os.path.basename(out)))
print("end:   " + _t.strftime("%Y-%m-%d %H:%M:%S"))
