"""rm_homo.py -- layup_db-driven OpenSG-RM plate homogenization.

    layup_db.yaml  ->  1dsg.yaml (+ .png)  ->  <db-stem>_plate_homo.out
                                               + the rm_plate_msg result dict

The YAML is the single user input: reference fraction, materials (with
density), stacking sequence, and model = 0 (classical 6x6 ABD) or 1
(shear-refined 8x8 ABDG).

Use as a module:
    from opensg_solid.rm_plate_1D.rm_homo import homogenize_layup_db
    r = homogenize_layup_db("layup_db.yaml")        # writes the three files
or from the command line:
    python -m opensg_solid.rm_plate_1D.rm_homo <layup_db.yaml>

ROWS: row/col labels -- 3 membrane strains, 3 curvatures, 2 transverse shears.
"""
import os

import numpy as np
import yaml

from .segment_plate import plate_sg_yaml, read_plate_sg_yaml
from .msg_rm_plate import rm_plate_msg

ROWS = ("e11", "e22", "g12", "k11", "k22", "k12", "2g13", "2g23")


def load_layup_db(path):
    """Parse a layup_db.yaml with every number coerced to float
    (PyYAML can read values such as 128.0e9 as strings).

    In:
        path: str, path to the layup_db YAML.
    Out:
        dict with keys:
            db: dict, the raw parsed YAML
            model: int, 0 = classical 6x6 ABD, 1 = shear-refined 8x8 ABDG
            fraction: float, reference-surface fraction
            n_per_layer: int, elements per ply
            elem_order: int, element order
            material_db: dict, name -> {E, G, nu, rho, full_name}
            layup: dict with mat_names/thick/angles lists (bottom->top).
    """
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
    """Homogenize the layup described by a layup_db.yaml.

    In:
        path: str, layup_db YAML path.
        out_dir: str or None, output directory (default: the db's directory).
        write: bool, when True write 1dsg.yaml + 1dsg.png and the
               <db-stem>_plate_homo.out matrix file.
    Out:
        dict: the rm_plate_msg result (A6, ABDG, nodal warping ladders).
    """
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
