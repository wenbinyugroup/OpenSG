"""RM-consistent thin-walled dehomogenization.

Step-1 recovery is rebuilt on the SAME Reissner-Mindlin ring used for the homogenization
(C0 Lagrange 6-DOF element with independent drilling omega_3, MITC-tied gamma_23), with the
same element operators (quad_ops_indep / _mitc_shear_indep), so it is the exact
energy-consistent adjoint of the RM 6x6 assembly:

    st  = C6_RM^{-1} FF                       (RM Timoshenko 6x6 inverse)
    a1,a2,a3,a4 = V0 st_m, V1 st_cl1, V1 st_cl2, V0 st_cl1   (RM warping combos)
    s6 = BDe st_m + BDh (a1+a2) + BDl (a4+a3)  -> [e11,e22,2e12,k11,k22,2k12]
    s2 = BGe st_m + BGt (a1+a2) + BGl (a4+a3)  -> [2g13,2g23]  (BGt = MITC-tied g23)

Unlike the Kirchhoff-Love bundle, the recovery carries the wall transverse shears
(2g13, 2g23).  Step 2 (plate through-thickness SG) reuses the opensg_jax plate machinery.

Public entry points: build_rm_bundle, disp_at_points, stress_at_points.
"""
import numpy as np
import yaml
import jax.numpy as jnp

from .segment_indep import quad_ops_indep, _mitc_shear_indep          # RM 8-strain operator
from .run_ring_indep import ring_indep                                 # RM ring solver
from .oml_ring import load_ring_ref                                    # OML/center ring loader
from .fe_jax.msg_hermite import solve_tw_from_yaml          # layup_db / material_db by name
from .fe_jax.msg_dehom import (_macro_recovery, _project_point,
                                         _voigt_to_tensor, _tensor_to_voigt)
from .fe_jax.msg_materials import (compute_ABD_matrix, plate_stress_at_depth,
                                             rotation_6x6)
from .fe_jax.msg_transverse_shear import transverse_shear_stiffness


def _strip(rx3, cells, ax):
    """Reconstruct the one-quad-deep prismatic strip mesh EXACTLY as ring_indep does.

    In:
        rx3: (m,3) float, ring node coordinates.
        cells: (n_el,2) int, ring line-element connectivity.
        ax: int, beam-axis index (0/1/2) the strip is extruded along.
    Out:
        nodes: (2m,3) float, strip node coordinates (ring + extruded copy).
        quads: (n_el,4) int, quad connectivity [a, b, m+b, m+a].
        h: float, extrusion length (mean ring element length).
    """
    rx3 = np.asarray(rx3, float); m = len(rx3)
    h = float(np.mean(np.linalg.norm(rx3[cells[:, 1]] - rx3[cells[:, 0]], axis=1)))
    ez = np.zeros(3); ez[ax] = 1.0
    nodes = np.vstack([rx3, rx3 + h * ez])
    quads = np.array([[a, b, m + b, m + a] for a, b in cells], dtype=int)
    return nodes, quads, h


def build_rm_bundle(shell_yaml, ref=None, shear="mitc4_g23", g_source="msg"):
    """Homogenize with the RM ring and package everything the two-step dehom needs.

    ``ref=None`` reads the reference surface from the yaml's ``reference`` field -- the single
    source of truth, set when the 1-D yaml is created (absent -> "center" = mid-surface) -- so
    homogenization and dehom follow the same reference; pass an explicit ``ref`` only to override.

    In:
        shell_yaml: str, path to the 1-D shell section yaml.
        ref: None | "center" | "oml" | "oml_flip" | "iml", reference-surface override.
        shear: str, RM transverse-shear tying scheme passed to ring_indep.
        g_source: str, wall transverse-shear source: "msg" (Yu-2002 LS projection) or
            "whitney" (complementary-energy shear flow).
    Out:
        dict bundle: "Timo" (6,6) RM Timoshenko matrix; "V0"/"V1" (6m,4) warping modes;
        "corners" (m,2) section contour coords; "red_cells" (n_el,2) connectivity;
        "rx3" (m,3), "re3" (n_el,3), "k22" (n_el,) ring geometry; "ax" int beam axis;
        "cross" axis pair; "strip" (nodes, quads, h); "layup_per_elem" list of layup
        names per element; "layup_db"/"material_db" plate-SG databases (by-name,
        geometry-free); "frac" float; "ref" str; "g_source" str.
    """
    import time as _time
    _t0 = _time.perf_counter()
    d = yaml.safe_load(open(shell_yaml))
    if ref is None:                                       # single source of truth: yaml records its ref
        ref = d.get("reference", "center")                # (set at 1-D-yaml creation; absent -> center)
    R = load_ring_ref(shell_yaml, ref)
    # Single reference decision: ring laminate ref, plate-SG z_ref, layup_db frac, emitted ABD,
    # and the recovery depth conversion in stress_at_points ALL follow ``frac``.
    frac = {"center": 0.5, "oml": 0.0, "oml_flip": 1.0, "iml": 1.0}.get(ref, 0.0)
    # G_msg is reference-independent, but the SG carries the recovery warping, so its z_ref
    # must sit at the chosen reference surface.
    G_by = list(R["G_by"])
    if g_source == "msg":
        from opensg_solid.rm_plate_1D.msg_rm_plate import rm_plate_msg
        from .emit_abd import material_db_from_yaml
        _mdb = material_db_from_yaml(d["materials"])
        for si, sec in enumerate(d["sections"]):
            _pl = [[str(p[0]), float(p[1]), float(p[2])] for p in sec["layup"]]
            _h = sum(p[1] for p in _pl)
            _rr = rm_plate_msg([p[1] for p in _pl], [p[2] for p in _pl], [p[0] for p in _pl],
                               _mdb, fraction=frac)
            if _rr["G_msg"] is not None:
                G_by[si] = np.asarray(_rr["G_msg"])
    from .solid_props import write_abdg_out
    import os as _os2
    write_abdg_out(_os2.path.splitext(shell_yaml)[0] + "_ABDG.out",
                   d["sections"], R["D_by"], G_by)
    C6, V0, V1 = ring_indep(R["rx"], R["cells"], R["rsub"], R["re3"], R["D_by"], G_by,
                            R["k22"], R["ax"], R["cross"], shear=shear, lam_space="elem",
                            return_fields=True)
    C6 = 0.5 * (C6 + C6.T)
    nodes, quads, h = _strip(R["rx"], R["cells"], R["ax"])

    sec_names = [s["elementSet"] for s in d["sections"]]
    layup_per_elem = [sec_names[int(si)] for si in R["rsub"]]
    # reuse the KL bundle ONLY for the by-name plate layup_db + material_db (geometry-free)
    kl = solve_tw_from_yaml(shell_yaml, frac=frac)
    # emit the per-station ABD yaml at the SAME reference (once, cached) for dehom + shell buckling
    try:
        import os as _os
        from .emit_abd import emit_station_abd
        _tag = _os.path.splitext(_os.path.basename(shell_yaml))[0]
        _ay = _os.path.join(_os.path.dirname(shell_yaml) or ".", "abd", _tag + "_abd.yaml")
        if not _os.path.exists(_ay):
            emit_station_abd(shell_yaml, _ay, station=_tag,
                             ref="mid" if ref == "center" else "oml")
    except Exception:
        # intentional best-effort: ABD yaml emission failure must not abort the bundle build
        pass
    # OpenSG default: SwiftComp-format timed .out for the beam model too
    from opensg_solid.sg_homo import write_sc_K
    write_sc_K(_os2.path.splitext(shell_yaml)[0] + "_Timo.out", C6,
               solve_time=_time.perf_counter() - _t0,
               model="msg-shell beam model"
                     " [ext sh2 sh3 twist bend2 bend3]",
               constants=False, name="Timoshenko")
    return {"Timo": C6, "V0": np.asarray(V0), "V1": np.asarray(V1),
            "corners": R["rx"][:, R["cross"]], "red_cells": np.asarray(R["cells"]),
            "rx3": np.asarray(R["rx"]), "re3": np.asarray(R["re3"]), "k22": np.asarray(R["k22"]),
            "ax": int(R["ax"]), "cross": list(R["cross"]), "strip": (nodes, quads, h),
            "layup_per_elem": layup_per_elem, "layup_db": kl["layup_db"],
            "material_db": kl["material_db"], "frac": frac, "ref": ref,
            "g_source": g_source}


def _rm_shell_strain(B, e, xi, st_m, aA, aB, s2_scheme="mitc4_g23"):
    """Evaluate the 8 RM shell strains of element ``e`` at arc coordinate ``xi``.

    In:
        B: dict, RM bundle from build_rm_bundle.
        e: int, ring element index.
        xi: float in [0,1], arc coordinate along the element.
        st_m: (6,) macro beam strain.
        aA: (6m,) nodal warping w.
        aB: (6m,) nodal warping derivative w'.
        s2_scheme: str, tying for the RECOVERED transverse shear: "mitc4_g23" matches the
            homogenization (only g23 tied; raw g13 carries flat-wall drilling), "mitc4_both"
            ties both rows -- the drilling-free physical wall shear for stress recovery.
    Out:
        s6: (6,) [e11, e22, 2e12, k11, k22, 2k12].
        s2: (2,) [2g13, 2g23].
    """
    nodes, quads, _ = B["strip"]
    Xe = nodes[quads[e]]; e3e = B["re3"][e]
    xq = 2.0 * float(xi) - 1.0                                   # arc [0,1] -> element [-1,1]
    BDe, BDh, BDl, BGe, BGh, BGl, DRe, DRh, DRl, dA = quad_ops_indep(
        Xe, e3e, xq, 0.0, float(B["k22"][e]), B["cross"], B["ax"])
    BGt = _mitc_shear_indep(Xe, e3e, xq, 0.0, float(B["k22"][e]), B["cross"], B["ax"],
                            scheme=s2_scheme)
    c0, c1 = int(B["red_cells"][e, 0]), int(B["red_cells"][e, 1])
    g = np.r_[c0 * 6:c0 * 6 + 6, c1 * 6:c1 * 6 + 6, c1 * 6:c1 * 6 + 6, c0 * 6:c0 * 6 + 6]
    wA = aA[g]; wB = aB[g]                                       # (24,) warping / warping'
    s6 = BDe @ st_m + BDh @ wA + BDl @ wB
    s2 = BGe @ st_m + BGt @ wA + BGl @ wB
    return s6, s2


def _macro_fields(B, beam_force_vabs=None, beam_strain=None):
    """Solve the macro beam strain and assemble the nodal RM warping fields.

    In:
        B: dict, RM bundle from build_rm_bundle.
        beam_force_vabs: (6,) beam force/moment in VABS order, or None.
        beam_strain: (6,) beam strain, or None (exactly one of the two must be given).
    Out:
        st: (6,) macro beam strain (given or C6^{-1} @ force).
        st_m: (6,) macro strain from _macro_recovery.
        aA: (6m,) nodal warping w  = V0 st_m + V1 st_cl1.
        aB: (6m,) nodal warping w' = V0 st_cl1 + V1 st_cl2.
    """
    C6 = np.asarray(B["Timo"])
    if (beam_strain is None) == (beam_force_vabs is None):
        raise ValueError("provide exactly one of beam_force_vabs or beam_strain")
    st = (np.asarray(beam_strain, float) if beam_strain is not None
          else np.linalg.inv(C6) @ np.asarray(beam_force_vabs, float))
    st, st_m, st_cl1, st_cl2 = _macro_recovery(C6, st)
    aA = np.asarray(B["V0"]) @ st_m + np.asarray(B["V1"]) @ st_cl1     # w  = a1 + a2
    aB = np.asarray(B["V0"]) @ st_cl1 + np.asarray(B["V1"]) @ st_cl2   # w' = a4 + a3
    return st, st_m, aA, aB


def disp_at_points(B, points_2d, beam_force_vabs=None, beam_strain=None, director=True):
    """RM-recovered warping displacement (u1,u2,u3) at query points.

    RM kinematics: u(z) = u_mid + z (omega x e3), with u_mid the mid-surface warping
    (first 3 nodal DOF) and omega the director rotation (last 3 DOF).

    In:
        B: dict, RM bundle from build_rm_bundle.
        points_2d: (P,2) float, query points (y2, y3) in the section plane.
        beam_force_vabs: (6,) beam force/moment in VABS order, or None.
        beam_strain: (6,) beam strain, or None (exactly one of the two must be given).
        director: bool, include the z*(omega x e3) depth term (needed for
            through-thickness paths; on-contour points z~=0 are unaffected).
    Out:
        (P,3) float, warping displacement [u1,u2,u3] per point.
    """
    pts = np.atleast_2d(np.asarray(points_2d, float))
    st, st_m, aA, aB = _macro_fields(B, beam_force_vabs, beam_strain)
    wn = np.asarray(aA).reshape(-1, 6)                       # per-node [u1,u2,u3,om1,om2,om3]
    corners = np.asarray(B["corners"]); rc = np.asarray(B["red_cells"]); cen = corners.mean(0)
    out = np.zeros((len(pts), 3))
    for i in range(len(pts)):
        e, xi, pr = _project_point(corners, rc, pts[i])
        c0, c1 = int(rc[e, 0]), int(rc[e, 1])
        umid = (1.0 - xi) * wn[c0, 0:3] + xi * wn[c1, 0:3]      # mid-surface warping
        if director:
            om = (1.0 - xi) * wn[c0, 3:6] + xi * wn[c1, 3:6]    # director rotation omega
            t2, t3 = corners[c1] - corners[c0]; tl = np.hypot(t2, t3); t2, t3 = t2 / tl, t3 / tl
            n2, n3 = t3, -t2
            if (cen[0] - pr[0]) * n2 + (cen[1] - pr[1]) * n3 < 0.0:
                n2, n3 = -n2, -n3                               # inward normal (contour -> interior)
            z = (pts[i, 0] - pr[0]) * n2 + (pts[i, 1] - pr[1]) * n3   # depth from the contour
            umid = umid + z * np.cross(om, np.array([0.0, n2, n3]))   # + z (omega x e3)
        out[i] = umid
    return out


def _flow_nodal_avg(B, st_m, aA, aB):
    """Nodal (patch) average of the contour-derivative strain rows 2eps12 (row 2)
    and 2k12 (row 5) along the element chains.

    Element-midpoint values are averaged only at nodes shared by exactly 2 elements
    (junction nodes keep one-sided values); rows 0,1,3,4 are NOT touched -- their
    region-boundary jumps are physical.

    In:
        B: dict, RM bundle from build_rm_bundle.
        st_m: (6,) macro beam strain.
        aA: (6m,) nodal warping w.
        aB: (6m,) nodal warping derivative w'.
    Out:
        emid: (n_el,2) element-midpoint [2eps12, 2k12].
        nodal: (n_nd,2) nodal averages; NaN where node degree != 2.
    """
    rc = np.asarray(B["red_cells"]); nodes, quads, _h = B["strip"]
    n_el = rc.shape[0]; n_nd = int(rc.max()) + 1
    emid = np.zeros((n_el, 2))
    for e in range(n_el):
        s6, _ = _rm_shell_strain(B, e, 0.5, st_m, aA, aB)
        emid[e] = [float(s6[2]), float(s6[5])]
    deg = np.zeros(n_nd, int)
    acc = np.zeros((n_nd, 2))
    for e in range(n_el):
        for nd in (int(rc[e, 0]), int(rc[e, 1])):
            deg[nd] += 1
            acc[nd] += emid[e]
    nodal = np.full((n_nd, 2), np.nan)
    ok = deg == 2
    nodal[ok] = acc[ok] / 2.0
    return emid, nodal


def stress_at_points(B, points_2d, beam_force_vabs=None, beam_strain=None,
                     frame="global", n_per_layer=2, elem_order=2, rm_shear=False,
                     s2_scheme="mitc4_g23", flow_avg=False):
    """RM two-step dehom: 3-D stress/strain at arbitrary section coordinates (y2, y3).

    Step 1 recovers the RM (C0, MITC-g23) shell strains consistent with the ring
    homogenization; step 2 evaluates the plate through-thickness SG (plate_stress_at_depth).

    In:
        B: dict, RM bundle from build_rm_bundle.
        points_2d: (P,2) float, query points (y2, y3) in the section plane.
        beam_force_vabs: (6,) beam force/moment in VABS order, or None.
        beam_strain: (6,) beam strain, or None (exactly one of the two must be given).
        frame: str, output frame: "global" (section), "material" (ply), else plate (wall).
        n_per_layer: int, through-thickness elements per ply in the plate SG.
        elem_order: int, plate-SG element order.
        rm_shear: bool, add constitutive transverse-shear recovery to sigma13/23.  OFF by
            default: sigma13=G13*g13 is not physical where the wall shear is an equilibrium
            (shear-flow) effect, so shipped sigma13/23 is the plate plane-stress limit.
        s2_scheme: str, MITC tying for the recovered transverse shear (see _rm_shell_strain).
        flow_avg: bool, nodally average the contour-derivative rows 2eps12/2k12.
    Out:
        dict with:
            "stress": (P,6) Voigt [S11,S22,S33,S23,S13,S12] in ``frame``.
            "strain": (P,6) Voigt strains in ``frame``.
            "elem": (P,) int, ring element each point projected onto.
            "xi": (P,) float, arc coordinate in [0,1] on that element.
            "depth": (P,) float, signed depth from the reference contour.
            "proj": (P,2) float, projected point on the contour.
            "macro": (6,) macro beam strain used.
    """
    pts = np.atleast_2d(np.asarray(points_2d, float))
    st, st_m, aA, aB = _macro_fields(B, beam_force_vabs, beam_strain)
    corners = np.asarray(B["corners"]); rc = np.asarray(B["red_cells"])
    xd = np.asarray(B["rx3"]); cen = corners.mean(axis=0)
    if flow_avg:
        _emid, _nodal = _flow_nodal_avg(B, st_m, aA, aB)
    layups = B["layup_per_elem"]; ldb = B["layup_db"]; mdb = B["material_db"]
    warp = {ln: compute_ABD_matrix(i["thick"], i["angles"], i["mat_names"], mdb,
            n_per_layer=n_per_layer, return_warping=True, elem_order=elem_order)[2]
            for ln, i in ldb.items()}
    shr = {ln: transverse_shear_stiffness(i["thick"], i["angles"], i["mat_names"], mdb)
           for ln, i in ldb.items()} if rm_shear else {}

    P = len(pts)
    stress = np.zeros((P, 6)); strain = np.zeros((P, 6))
    el = np.zeros(P, int); xia = np.zeros(P); dep = np.zeros(P); proj = np.zeros((P, 2))
    for ip in range(P):
        e, xi, pr = _project_point(corners, rc, pts[ip])
        c0, c1 = int(rc[e, 0]), int(rc[e, 1])
        t2, t3 = corners[c1] - corners[c0]
        tl = float(np.hypot(t2, t3)); t2, t3 = t2 / tl, t3 / tl       # unit arc tangent
        n2, n3 = t3, -t2                                              # inward normal (plate z)
        if (cen[0] - pr[0]) * n2 + (cen[1] - pr[1]) * n3 < 0.0:
            n2, n3 = -n2, -n3
        z = float((pts[ip, 0] - pr[0]) * n2 + (pts[ip, 1] - pr[1]) * n3)

        s6, s2 = _rm_shell_strain(B, e, xi, st_m, aA, aB, s2_scheme=s2_scheme)
        if flow_avg:
            s6 = np.array(s6, float)
            for kk, row in enumerate((2, 5)):
                v0 = _nodal[c0, kk] if np.isfinite(_nodal[c0, kk]) else _emid[e, kk]
                v1 = _nodal[c1, kk] if np.isfinite(_nodal[c1, kk]) else _emid[e, kk]
                s6[row] = (1.0 - xi) * v0 + xi * v1
        # Convert the signed ring depth z (from the frac-ref surface) to the plate-SG OML depth
        # (0..h) and shift the membrane strain to the OML so the z*curvature term stays consistent.
        hth = float(warp[layups[e]]['node_x'][-1])
        frac = float(B.get('frac', 0.0))
        z_oml = z + frac * hth
        s6r = np.array(s6, float); s6r[0:3] = s6r[0:3] - frac * hth * s6r[3:6]
        Gam, Sig, ply = plate_stress_at_depth(warp[layups[e]], s6r, z_oml)
        if rm_shear:
            Gmat, recover, _ = shr[layups[e]]
            Q = Gmat @ s2                                            # [Q1,Q2] resultants (N/m)
            s13, s23 = recover(z_oml, Q)                             # ghat(z)*Q, free-surface
            Sig = np.asarray(Sig, float).copy()
            Sig[4] += s13; Sig[3] += s23                            # Voigt [S11,S22,S33,S23,S13,S12]
        if frame == "global":
            Rm = np.array([[1., 0, 0], [0, t2, n2], [0, t3, n3]])
            Sig = _tensor_to_voigt(Rm @ _voigt_to_tensor(Sig) @ Rm.T)
            Gam = _tensor_to_voigt(Rm @ _voigt_to_tensor(Gam) @ Rm.T)
        elif frame == "material":
            Sig = rotation_6x6(-ply) @ Sig; Gam = rotation_6x6(ply).T @ Gam
        stress[ip] = Sig; strain[ip] = Gam
        el[ip] = e; xia[ip] = xi; dep[ip] = z; proj[ip] = pr
    return {"stress": stress, "strain": strain, "elem": el, "xi": xia,
            "depth": dep, "proj": proj, "macro": st}
