"""emit_abd.py -- per-station laminate ABD file (windIO-style YAML) for the RM shell.

For a cross-section, compute the MID-REF plate stiffness of every unique layup -- the 6x6 ABD
[[A,B],[B,D]] plus the 2x2 transverse-shear Gs -- via the MSG through-thickness SG, and write ONE
yaml per station.  The homogenization and dehomogenization both need the ABD, and the shell-buckling
tool reads it directly, so it is computed once per station and reused (no recompute).

    from emit_abd import emit_station_abd
    emit_station_abd("iea_s10_shell.yaml", "abd/iea_s10_abd.yaml", station="iea_s10", r=0.20)

YAML layout (one entry per unique layup, keyed by an integer id = its section index):
    station, r, reference (mid/oml/iml), convention
    layups:
      - id: 0
        name: layup_0
        thickness: <m>
        mass_per_area: <kg/m^2>
        plies: [{material, thickness, angle}, ...]     # bottom(OML) -> top(IML)
        ABD: 6x6  ([[A,B],[B,D]]; Voigt membrane/bending order [11,22,12]; A[N/m] B[N] D[N.m])
        Gs:  2x2  transverse shear [N/m]
"""
import os
import numpy as np
import yaml
from .fe_jax.msg_materials import compute_ABD_matrix

_ZREF = {"oml": 0.0, "mid": 0.5, "iml": 1.0}     # fraction of total thickness from the OML face


def material_db_from_yaml(materials):
    db = {}
    for m in materials:
        el = m["elastic"]
        db[m["name"]] = {"E": [float(x) for x in el["E"]], "G": [float(x) for x in el["G"]],
                         "nu": [float(x) for x in el["nu"]], "rho": float(m.get("density", 0.0))}
    return db


def emit_station_abd(shell_yaml, out_yaml, station=None, r=None, ref="mid", g_source="whitney"):
    """Write the per-layup 8x8 RM wall law (default MID reference) for one cross-section.

    Every layup stores the FULL 8x8 storage matrix  ABDG = [[A,B,0],[B,D,0],[0,0,G]]
    (SwiftComp-style RM plate law) plus the legacy ABD 6x6 / Gs 2x2 keys for old readers.

    ``g_source`` selects the 2x2 transverse-shear block G:
      * "whitney" (default) -- coupling-aware complementary-energy shear flow
        (msg_transverse_shear.transverse_shear_stiffness, coupled=True)
      * "msg"     -- the MSG/VAM second-order-energy RM projection (Yu-2002 LS
        construction, msg_rm_plate.rm_plate_msg) -- SwiftComp-like G.
    Returns the dict written."""
    d = yaml.safe_load(open(shell_yaml))
    mdb = material_db_from_yaml(d["materials"])
    frac = _ZREF[ref]
    layups = []
    for sid, sec in enumerate(d["sections"]):
        plies = [[str(p[0]), float(p[1]), float(p[2])] for p in sec["layup"]]   # [material, thickness, angle]
        mats = [p[0] for p in plies]; thk = [p[1] for p in plies]; ang = [p[2] for p in plies]
        h = float(sum(thk))
        ABD8, mass = compute_ABD_matrix(thk, ang, mats, mdb, shear_refined=True, z_ref=frac * h)
        ABD8 = np.asarray(ABD8, float).copy()
        if g_source == "msg":
            from opensg_solid.rm_plate_1D.msg_rm_plate import rm_plate_msg
            rr = rm_plate_msg(thk, ang, mats, mdb, fraction=frac)
            if rr["G_msg"] is not None:
                ABD8[6:, 6:] = rr["G_msg"]
        layups.append({
            "id": sid, "name": str(sec["elementSet"]), "thickness": round(h, 9),
            "mass_per_area": float(np.asarray(mass).ravel()[0]),
            "plies": [{"material": m, "thickness": round(t, 9), "angle": a} for m, t, a in plies],
            "ABDG": [[float(v) for v in row] for row in ABD8],
            "ABD": [[float(v) for v in row] for row in ABD8[:6, :6]],
            "Gs": [[float(v) for v in row] for row in ABD8[6:, 6:]],
        })
    out = {"station": station, "r": r, "reference": ref, "g_source": g_source,
           "convention": "ABDG=[[A,B,0],[B,D,0],[0,0,G]] 8x8 (rows 1-6 Voigt [11,22,12] membrane+bending, rows 7-8 transverse shear [2g13,2g23]); legacy ABD 6x6 + Gs 2x2 kept; plies OML->IML",
           "n_layups": len(layups), "layups": layups}
    dd = os.path.dirname(out_yaml)
    if dd:
        os.makedirs(dd, exist_ok=True)
    with open(out_yaml, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False, default_flow_style=None)
    return out


def load_station_abd(abd_yaml):
    """Read an ABD yaml -> dict {name/id: (ABD 6x6, Gs 2x2, thickness)} for the buckling assembler.
    Also exposes the full 8x8 per layup under ``abdg_by_name`` when the file carries ABDG."""
    d = yaml.safe_load(open(abd_yaml))
    by_id = {}; by_name = {}; abdg_by_name = {}
    for L in d["layups"]:
        if "ABDG" in L:
            M8 = np.array(L["ABDG"], float)
            A = M8[:6, :6]; G = M8[6:, 6:]
            abdg_by_name[str(L["name"])] = M8
        else:
            A = np.array(L["ABD"], float); G = np.array(L["Gs"], float)
        t = float(L["thickness"])
        by_id[int(L["id"])] = (A, G, t); by_name[str(L["name"])] = (A, G, t)
    return {"by_id": by_id, "by_name": by_name, "abdg_by_name": abdg_by_name,
            "reference": d.get("reference"), "g_source": d.get("g_source", "whitney"), "raw": d}
