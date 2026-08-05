"""rm_homo.py -- the GENERAL layup_db-driven OpenSG-RM plate homogenization,
promoted to the core from the ex5 pipeline (examples/opensg-rm_dynamic/ex5/
1d_sg.py) because nothing in it is example-specific.

    layup_db.yaml  ->  1dsg.yaml (+ .png)  ->  <db-stem>_plate_homo.out
                                               + the rm_plate_msg result dict

The YAML is the single user input: reference fraction, materials (with
density), stacking sequence, and model = 0 (classical 6x6 ABD) or 1
(shear-refined 8x8 ABDG).  See the ex5 layup_db.yaml for the documented
format.

Use as a module:
    from opensg_solid.rm_plate_1D.rm_homo import homogenize_layup_db
    r = homogenize_layup_db("layup_db.yaml")        # writes the three files
or from the command line:
    python -m rm_plate.rm_homo <layup_db.yaml>

# ----------------------------------------------------------------------------
# ALL VARIABLES USED IN THIS MODULE
# ----------------------------------------------------------------------------
#   ROWS          row/col labels: 3 membrane strains, 3 curvatures, 2 shears
#   load_layup_db(path) -> dict with every number float-coerced (the PyYAML
#                 "128.0e9 is a string" trap) and the layup unpacked to the
#                 mat_names/thick/angles lists rm_plate_msg wants
#   homogenize_layup_db(path, out_dir=None, write=True) -> the rm_plate_msg
#                 result dict r (A6, ABDG, the V0/V11/V12 warping ladders);
#                 when write is True also writes 1dsg.yaml/.png next to the
#                 db (or into out_dir) and <db-stem>_plate_homo.out
#   db, model, fraction, mesh, n_per_layer, elem_order, material_db, layup
#                 the parsed pieces of the YAML
#   yml, inp, r   the SG mesh file, its parsed-back dict, the homo result
#   n, M, rho_h   printed size (6 or 8), the matrix, the section mass rho*h
# ----------------------------------------------------------------------------
"""
import os

import numpy as np
import yaml

from .segment_plate import plate_sg_yaml, read_plate_sg_yaml
from .msg_rm_plate import rm_plate_msg

ROWS = ("e11", "e22", "g12", "k11", "k22", "k12", "2g13", "2g23")


def load_layup_db(path):
    """Parse layup_db.yaml with every number coerced to float."""
    db = yaml.safe_load(open(path))
    material_db = {k: {"E": [float(v) for v in m["E"]],
                       "G": [float(v) for v in m["G"]],
                       "nu": [float(v) for v in m["nu"]],
                       "rho": float(m.get("rho", 1.0)),
                       "full_name": m.get("full_name") or k}
                   for k, m in db["materials"].items()}
    layup = {"mat_names": [p["material"] for p in db["layup"]],
             "thick": [float(p["thickness"]) for p in db["layup"]],
             "angles": [float(p.get("angle", 0.0)) for p in db["layup"]]}
    return {"db": db, "model": int(db.get("model", 1)),
            "fraction": float(db["fraction"]),
            "n_per_layer": int((db.get("mesh") or {}).get("n_per_layer", 1)),
            "elem_order": int((db.get("mesh") or {}).get("elem_order", 4)),
            "material_db": material_db, "layup": layup}


def homogenize_layup_db(path, out_dir=None, write=True):
    """The full homo chain of the ex5 pipeline, callable from anywhere.

    Returns the rm_plate_msg result dict (A6, ABDG and the nodal warping
    ladders the dehomogenization rides on).  When `write` is True, also
    writes 1dsg.yaml + 1dsg.png next to the db (or into out_dir) and the
    <db-stem>_plate_homo.out matrix file."""
    path = os.path.abspath(path)
    d = load_layup_db(path)
    out_dir = out_dir or os.path.dirname(path)
    yml = os.path.join(out_dir, "1dsg.yaml")
    if write:
        plate_sg_yaml(yml, d["layup"], d["material_db"],
                      n_per_layer=d["n_per_layer"],
                      elem_order=d["elem_order"], fraction=d["fraction"])
        inp = read_plate_sg_yaml(yml)
    else:
        inp = {"thick": d["layup"]["thick"], "angles": d["layup"]["angles"],
               "mat_names": d["layup"]["mat_names"],
               "material_db": d["material_db"],
               "n_per_layer": d["n_per_layer"],
               "elem_order": d["elem_order"]}
    r = rm_plate_msg(inp["thick"], inp["angles"], inp["mat_names"],
                     inp["material_db"], n_per_layer=d["n_per_layer"],
                     elem_order=d["elem_order"], fraction=d["fraction"])
    if write:
        n = 6 if d["model"] == 0 else 8
        M = np.asarray(r["A6"] if d["model"] == 0 else r["ABDG"])
        rho_h = sum(inp["material_db"][m]["rho"] * t
                    for m, t in zip(inp["mat_names"], inp["thick"]))
        out = os.path.join(out_dir, os.path.splitext(
            os.path.basename(path))[0] + "_plate_homo.out")
        with open(out, "w") as f:
            f.write("OpenSG plate homogenization of %s\n"
                    % os.path.basename(yml))
            f.write("%d plies, h = %.6g m, reference fraction = %g\n"
                    % (len(inp["thick"]), sum(inp["thick"]), d["fraction"]))
            f.write("model %d: %s\n\n"
                    % (d["model"], "classical 6x6 ABD" if d["model"] == 0
                       else "shear-refined 8x8 ABDG"))
            f.write("rows/cols: %s\n" % ", ".join(ROWS[:n]))
            for row in M:
                f.write(" ".join("%14.6e" % v for v in row) + "\n")
            f.write("\nsection mass rho*h = %.6g kg/m^2\n" % rho_h)
    return r


if __name__ == "__main__":
    import sys
    homogenize_layup_db(sys.argv[1] if len(sys.argv) > 1 else "layup_db.yaml")
    print("done")
