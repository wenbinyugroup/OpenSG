"""Section laws for the RM shell: per-section wall constitutive matrices (the plate ABD 6x6
plus the 2x2 transverse-shear G via _material_by_section, with analytic isotropic-tube
references for validation) and the per-station wall-law ABDG emitter that writes one
windIO-style YAML of 8x8 RM plate laws per cross-section.

Merged from: solve_segment_jax.py, emit_abd.py

solve_segment_jax.py
========================================================================
STAGE 3 of the hybrid MITC-RM tapered-segment pipeline (isotropic first).

    stage 1  make_cylinder_segment.py        -> surface-quad YAML
    stage 2  extract_boundaries_dolfinx.py   -> boundary rings + maps (.npz)
    stage 3  solve_segment_jax.py            -> RM/MITC V0,V1 + 6x6      <-- THIS FILE

PLAN
----
  PART 1  (implemented, self-validating)
      Solve each END RING as a 1-D Reissner-Mindlin / MITC cross-section SG,
      reusing the VALIDATED opensg_jax RM assembly (msg_rm_timo.timoshenko_rm
      with return_warp=True).  This yields the boundary warping fields
      V0 (4 Euler-Bernoulli modes) and V1 (Timoshenko shear-warping), plus the
      ring Timoshenko 6x6.  We check the ring 6x6 against the closed-form
      isotropic tube (EA, EI, GJ) so the material/geometry wiring is trusted
      before it feeds the segment.

  PART 2  (next: the one genuinely new element)
      A 2-D MITC4 Reissner-Mindlin SHELL element on the curved quad surface
      (5 DOF/node [w1,w2,w3,w1,w2]; bilinear N(xi,eta); 2x2 surface Jacobian;
      transverse-shear tying per Chapelle-Bathe "The Finite Element Analysis of
      Shells" sec 8.2 / plate sec 7.2.4).  Then scatter the ring V0/V1 onto the
      segment boundary DOFs as Dirichlet BCs, solve the segment, and (the check
      you asked for) confirm the segment warping at ANY interior ring equals the
      boundary warping node-for-node -- the prismatic span-invariance identity.

Only the isotropic case is exercised here (per request).

emit_abd.py -- per-station laminate ABD file (windIO-style YAML) for the RM shell.
========================================================================
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
import sys
import json
import numpy as np
import yaml

from .fe_jax.msg_materials import compute_ABD_matrix, shift_abd_reference
from .fe_jax.msg_transverse_shear import transverse_shear_stiffness
from .fe_jax.msg_rm_timo import timoshenko_rm     # 1-D RM solve (return_warp -> V0,V1)


# ============================================================================
# from solve_segment_jax.py
# ============================================================================

LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]


# ============================================================ material (ABD + G)
def build_material(bundle, center_ref=True):
    """Return (D 6x6 ABD, G 2x2 transverse shear, t) for the (single) layup.

    The mesh sits on the wall MID-surface, so the ABD -- which compute_ABD_matrix
    references to the bottom face -- is parallel-axis shifted by t/2 to the centre
    (shift_abd_reference); the transverse-shear block G is reference-independent.
    """
    materials = json.loads(str(bundle["materials"]))
    sections = json.loads(str(bundle["sections"]))
    mat_db = {m["name"]: {"E": m["elastic"]["E"], "G": m["elastic"]["G"],
                          "nu": m["elastic"]["nu"]} for m in materials}
    layup = sections[0]["layup"]                       # [[mat, ply_t, angle], ...]
    mat_names = [ply[0] for ply in layup]
    thick = [float(ply[1]) for ply in layup]
    angles = [float(ply[2]) for ply in layup]
    t = float(sum(thick))

    D = np.asarray(compute_ABD_matrix(thick, angles, mat_names, mat_db)[0])   # 6x6, bottom-ref
    if center_ref:
        D = shift_abd_reference(D, t / 2.0)                                    # -> mid-wall
    G = np.asarray(transverse_shear_stiffness(thick, angles, mat_names, mat_db)[0])  # 2x2
    return D, G, t


# ================================================== PART 1: 1-D boundary ring SG
def order_ring(ring_x):
    """Sort a ring's nodes by hoop angle and return (nodes2d(NC,2), elems(NC,2),
    perm, R).  The dolfinx submesh nodes are unordered; sorting into a consistent
    CCW loop matches the 1-D generator convention (uniform inward normal), so the
    reused element operators see the same orientation they were validated with.
    `perm[i]` = original ring-node index of sorted node i (kept so ring warping
    can later be matched to segment boundary nodes by hoop position).
    """
    y, z = ring_x[:, 1], ring_x[:, 2]                  # cross-section plane = (y, z)
    th = np.arctan2(z, y)
    perm = np.argsort(th)
    nodes2d = np.column_stack([y[perm], z[perm]])
    NC = len(perm)
    elems = np.array([[i, (i + 1) % NC] for i in range(NC)], dtype=int)   # closed loop
    R = float(np.mean(np.hypot(y, z)))
    return nodes2d, elems, perm, R


def solve_ring(ring_x, D, G, shear="mitc_both"):
    """Solve the ring as a 1-D RM/MITC cross-section SG.

    Returns dict with the Timoshenko 6x6 and the warping fields V0 (ndof,4) and
    V1 (ndof,4) in the ring's SORTED node order (5 DOF/node [w1,w2,w3,w1r,w2r]),
    plus `perm` and `R`.  k22 = -1/R is the (uniform) hoop curvature of the tube
    in the code's -1/R convention; the sign is confirmed by the 6x6 check below.
    """
    nodes2d, elems, perm, R = order_ring(ring_x)
    lpe = np.zeros(len(elems), dtype=int)
    D_by = {0: D}; G_by = {0: G}
    k22 = np.full(len(elems), -1.0 / R)
    C6, Deff, V0, V1 = timoshenko_rm(nodes2d, elems, lpe, D_by, G_by, k22,
                                     p=1, return_warp=True, shear=shear)
    return {"C6": np.asarray(C6), "V0": np.asarray(V0), "V1": np.asarray(V1),
            "perm": perm, "R": R, "nodes2d": nodes2d, "elems": elems}


def solve_boundary_yaml(yaml_path, center_ref=True, shear="mitc_both"):
    """Solve a boundary ring from its standalone 1-D cross-section YAML file.

    Reads the YAML directly (node order preserved == node2seg order), builds the
    centre-ref ABD + transverse-shear G, and solves the RM/MITC cross-section for
    V0 (4 EB modes) and V1 (Timoshenko).  shear='mitc' is the discretization the
    2-D MITC4 element reduces to at span-invariance (gamma23 tied at hoop-centre,
    gamma13=omega2 full).  V0[5*i:5*i+5] maps to segment node node2seg[i].
    """
    import yaml as _yaml
    d = _yaml.safe_load(open(yaml_path))
    nd2 = np.array(d["nodes"], float)[:, :2]                 # (m,2) cross-section (y,z)
    elems = np.array(d["elements"], int) - 1                 # (m,2) 0-based, closed loop
    matmap = {mm["name"]: {"E": mm["elastic"]["E"], "G": mm["elastic"]["G"], "nu": mm["elastic"]["nu"]}
              for mm in d["materials"]}
    layup = d["sections"][0]["layup"]
    mat_names = [p[0] for p in layup]; thick = [float(p[1]) for p in layup]; angles = [float(p[2]) for p in layup]
    abd = np.asarray(compute_ABD_matrix(thick, angles, mat_names, matmap)[0])
    if center_ref:
        abd = shift_abd_reference(abd, 0.5 * sum(thick))
    Gm = np.asarray(transverse_shear_stiffness(thick, angles, mat_names, matmap)[0])
    R = float(np.mean(np.hypot(nd2[:, 0], nd2[:, 1]))); k22 = np.full(len(elems), -1.0 / R)
    C6, Deff, V0, V1 = timoshenko_rm(nd2, elems, np.zeros(len(elems), int), {0: abd}, {0: Gm},
                                     k22, p=1, return_warp=True, shear=shear)
    return {"C6": np.asarray(C6), "V0": np.asarray(V0), "V1": np.asarray(V1), "R": R, "m": len(nd2)}


def _material_by_section(sections, materials, center_ref=True):
    """Build the plate ABD and transverse-shear stiffness for each layup section.

    In:
        sections: list of section dicts, each with 'layup' = [[mat_name, ply_t, angle_deg], ...].
        materials: list of material dicts, each with 'name' and 'elastic' {E, G, nu}.
        center_ref: bool; True parallel-axis shifts each ABD to the laminate mid-surface.
    Out:
        D_by: dict {section index: (6,6) ABD matrix}.
        G_by: dict {section index: (2,2) transverse-shear stiffness}.
    """
    matmap = {mm["name"]: {"E": mm["elastic"]["E"], "G": mm["elastic"]["G"], "nu": mm["elastic"]["nu"]}
              for mm in materials}
    D_by, G_by = {}, {}
    for si, sec in enumerate(sections):
        layup = sec["layup"]
        mn = [p[0] for p in layup]; th = [float(p[1]) for p in layup]; an = [float(p[2]) for p in layup]
        abd = np.asarray(compute_ABD_matrix(th, an, mn, matmap)[0])
        if center_ref:
            abd = shift_abd_reference(abd, 0.5 * sum(th))
        D_by[si] = abd
        G_by[si] = np.asarray(transverse_shear_stiffness(th, an, mn, matmap)[0])
    return D_by, G_by


def solve_boundary_bundle(b, side, center_ref=True, shear="mitc_both", k22=None):
    """Solve a boundary ring IN-MEMORY straight from the extraction bundle -- no
    boundary-YAML write/read round-trip (the fast path; pass write_yaml=True to
    boundary_from_yaml.extract only if you also want the 1-D YAML for inspection).

    Multi-material: ABD/G computed once per layup section, keyed by section index;
    per-edge section id = bundle['<side>_subdom'].

    k22 = the CURVATURE argument (initial hoop curvature per edge):
      None       -> uniform -1/R (circular cross-section default)
      'general'  -> per-edge geometric curvature from the parent-quad frames
                    (segment_element.compute_k22: median + flat-element snap-to-0)
      'zero'     -> no initial-curvature terms
      array      -> caller-supplied per-edge values
    """
    ax = int(b["axis"]); cross = [j for j in range(3) if j != ax]
    nd2 = np.asarray(b["%s_x" % side])[:, cross]                 # (m,2) cross-section coords
    elems = np.asarray(b["%s_cells" % side]); lpe = np.asarray(b["%s_subdom" % side])
    sections = json.loads(str(b["sections"])); materials = json.loads(str(b["materials"]))
    D_by, G_by = _material_by_section(sections, materials, center_ref)
    c = nd2.mean(0); R = float(np.mean(np.hypot(nd2[:, 0] - c[0], nd2[:, 1] - c[1])))
    if k22 is None:                                          # circular cross-section default
        k22 = np.full(len(elems), -1.0 / R)
    elif isinstance(k22, str) and k22 == "zero":
        k22 = np.zeros(len(elems))
    elif isinstance(k22, str) and k22 == "general":
        from .sg_assembly import compute_k22
        rx = np.asarray(b["%s_x" % side])
        k22 = compute_k22(rx[np.asarray(b["%s_cells" % side])].mean(axis=1),
                          np.asarray(b["%s_e2" % side]), np.asarray(b["%s_e3" % side]),
                          np.asarray(b["%s_cells" % side]))
    C6, Deff, V0, V1 = timoshenko_rm(nd2, elems, lpe, D_by, G_by, np.asarray(k22),
                                     p=1, return_warp=True, shear=shear)
    return {"C6": np.asarray(C6), "V0": np.asarray(V0), "V1": np.asarray(V1), "R": R}


def analytic_iso_tube(R, t, E, nu):
    """Closed-form thin isotropic-tube Timoshenko diagonal (mid-wall reference)."""
    Gs = E / (2.0 * (1.0 + nu))
    A = 2.0 * np.pi * R * t
    I = np.pi * R**3 * t                 # bending 2nd moment (thin ring)
    J = 2.0 * np.pi * R**3 * t           # torsion constant (thin tube)
    ks = 0.5                             # thin-ring shear coefficient (approx)
    return dict(EA=E * A, GA2=Gs * A * ks, GA3=Gs * A * ks, GJ=Gs * J,
                EI2=E * I, EI3=E * I)


# ================================================== PART 2: 2-D MITC-RM segment
# TODO (next): assemble the segment RM fluctuation system on the quad surface.
#   * bilinear shape functions N(xi,eta) on [-1,1]^2, 2x2 Gauss for the D-energy;
#   * surface Jacobian J = [dX/dxi, dX/deta] (3x2) -> in-plane derivatives d/dx1
#     (axial, .e1) and d/ds (hoop, .e2); e1/e2/e3 come from seg_e1/e2/e3;
#   * D-strain rows [eps11, eps22, 2eps12, k11, k22, 2k12] extend the 1-D BDq by
#     adding the AXIAL (xi1) derivatives (zero for a span-invariant field);
#   * MITC4 transverse-shear tying (Dvorkin-Bathe / Chapelle-Bathe sec 8.2):
#       gamma_xi_zeta  tied at (xi=0, eta=+/-1), linear in eta;
#       gamma_eta_zeta tied at (xi=+/-1, eta=0), linear in xi;
#   * reuse msg_solver (assemble_kkt / solve_fluctuation_field / prepare_v1_rhs /
#     finalize_v1_and_compute_deff) unchanged -- it is element-agnostic.
# Then: Dirichlet the boundary DOFs to the ring V0/V1 (map ring node -> segment
# node via L_node2seg[perm]), solve the segment, and assert
#   V0_seg[interior ring node] == V0_ring[same hoop node]  (prismatic invariance).
def assemble_segment_rm(bundle, D, G):
    raise NotImplementedError("PART 2 -- 2-D MITC4 RM segment element (next step)")


# ============================================================================ run
def main(npz_path):
    b = np.load(npz_path, allow_pickle=True)
    D, G, t = build_material(b, center_ref=True)
    # isotropic engineering constants (for the closed-form check)
    mat = json.loads(str(b["materials"]))[0]["elastic"]
    E, nu = float(mat["E"][0]), float(mat["nu"][0])

    print("=== PART 1: boundary-ring RM/MITC 6x6 (isotropic) ===")
    print("material: E=%.3e  nu=%.2f   wall t=%.4f" % (E, nu, t))
    left = solve_ring(b["L_x"], D, G)
    right = solve_ring(b["R_x"], D, G)
    R = left["R"]
    ana = analytic_iso_tube(R, t, E, nu)

    C6 = left["C6"]; d = np.diag(C6)
    print("ring radius R=%.4f  h/R=%.3f" % (R, t / R))
    print("%-5s %14s %14s %10s" % ("term", "ring 6x6", "analytic", "%err"))
    for i, k in enumerate(LBL):
        err = 100.0 * (d[i] - ana[k]) / ana[k]
        print("%-5s %14.4e %14.4e %+9.1f%%" % (k, d[i], ana[k], err))

    # left/right must be identical for a prismatic segment (same cross-section)
    dLR = np.max(np.abs(left["C6"] - right["C6"])) / max(1.0, np.max(np.abs(left["C6"])))
    print("left-vs-right 6x6 max rel diff: %.2e (expect ~0, both ends identical)" % dLR)
    print("V0 shape", left["V0"].shape, " V1 shape", left["V1"].shape,
          " (ndof = 5 x %d ring nodes)" % (left["V0"].shape[0] // 5))
    print("\nPART 2 (2-D MITC4 RM segment + V0/V1 span-invariance check): next.")
    return left, right


# ============================================================================
# from emit_abd.py
# ============================================================================

_ZREF = {"oml": 0.0, "mid": 0.5, "iml": 1.0}     # fraction of total thickness from the OML face


def material_db_from_yaml(materials):
    """Build a material lookup dict from a windIO-style materials list.

    In:
        materials: list of dicts, each with "name", "elastic" holding "E"/"G"/"nu"
            component lists, and optional "density".
    Out:
        dict: name -> {"E": [3 floats], "G": [3 floats], "nu": [3 floats],
            "rho": float} ("rho" = 0.0 when density is absent).
    """
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
