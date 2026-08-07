"""dehom_rm.py -- RM-CONSISTENT thin-walled dehomogenization.

The paper's homogenization is the Reissner-Mindlin ring (mitc_rm_segment/run_ring_indep.py
::ring_indep, shear='mitc4_g23'): a C0 Lagrange 6-DOF element (independent drilling omega_3)
with MITC-tied gamma_23.  The *dehomogenization* must use the SAME model -- not the
Kirchhoff-Love C1 Hermite shell (opensg_jax/fe_jax/msg_dehom.py).  This module rebuilds
step-1 on the RM ring:

    st  = C6_RM^{-1} FF                       (RM Timoshenko 6x6 inverse)
    a1,a2,a3,a4 = V0 st_m, V1 st_cl1, V1 st_cl2, V0 st_cl1   (RM warping combos)
    s6 = BDe st_m + BDh (a1+a2) + BDl (a4+a3)  -> [e11,e22,2e12,k11,k22,2k12]
    s2 = BGe st_m + BGt (a1+a2) + BGl (a4+a3)  -> [2g13,2g23]  (BGt = MITC-tied g23)

with the SAME element operators (quad_ops_indep / _mitc_shear_indep) that assembled the RM
6x6.  This is the exact adjoint of the RM homogenization (mirrors assemble_segment_indep),
so it is energy-consistent AND -- unlike the KL bundle -- it carries the wall transverse
shears (2g13, 2g23), so sigma13/sigma23 are recovered from the physical per-element wall
shear, not a section-level approximation.

STEP 2 (plate through-thickness SG) and STEP 3 (shear-flow sigma13/23) reuse the existing
opensg_jax plate machinery unchanged.
"""
import os
import sys

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
    """Reconstruct the one-quad-deep prismatic strip EXACTLY as ring_indep does."""
    rx3 = np.asarray(rx3, float); m = len(rx3)
    h = float(np.mean(np.linalg.norm(rx3[cells[:, 1]] - rx3[cells[:, 0]], axis=1)))
    ez = np.zeros(3); ez[ax] = 1.0
    nodes = np.vstack([rx3, rx3 + h * ez])
    quads = np.array([[a, b, m + b, m + a] for a, b in cells], dtype=int)
    return nodes, quads, h


def build_rm_bundle(shell_yaml, ref=None, shear="mitc4_g23", g_source="msg"):
    """Homogenize with the RM ring and package everything the two-step dehom needs.

    ``ref`` defaults to ``None`` = read the reference from the yaml's ``reference`` field -- the
    SINGLE source of truth, chosen once when the 1-D yaml is created (emit_opensg_yaml's ``fraction``:
    0.5->"center", 0.0->"oml"); absent -> "center".  So the RM homogenization AND dehom automatically
    follow whatever reference the yaml was built at.  center = mid-surface (frac=0.5, the default); the
    BeamDyn 6x6 / FF are center-ref too, so the dehom stress/disp recovery stays consistent
    (stress_at_points converts the mid-ref depth to the plate OML depth via frac).  Pass an explicit
    ``ref="oml"``/``"center"`` only to override the yaml.

    Returns a bundle dict with the RM Timoshenko 6x6 ``Timo`` (=C6_RM), the RM warping
    ``V0``/``V1`` (6m x 4), the strip geometry, the per-element layup, and the plate-SG
    layup_db / material_db (reused from solve_tw_from_yaml -- geometry-independent, keyed
    by layup name).
    """
    import time as _time
    _t0 = _time.perf_counter()
    d = yaml.safe_load(open(shell_yaml))
    if ref is None:                                       # single source of truth: yaml records its ref
        ref = d.get("reference", "center")                # (set at 1-D-yaml creation; absent -> center)
    R = load_ring_ref(shell_yaml, ref)
    # THE single initial-stage reference decision: everything below (ring laminate reference via
    # load_ring_ref above, plate-SG z_ref for the MSG G / recovery warping, the KL layup_db frac,
    # the emitted ABD yaml, and the recovery depth conversion in stress_at_points) follows ``frac``.
    frac = {"center": 0.5, "oml": 0.0, "oml_flip": 1.0, "iml": 1.0}.get(ref, 0.0)
    # wall transverse-shear source: "msg" (DEFAULT since 2026-07-21, SwiftComp-like Yu-2002 LS
    # projection, msg_rm_plate.rm_plate_msg) or "whitney" (legacy complementary-energy shear flow).
    # Section 6x6 is insensitive (<=0.02% at IEA r=0.2) but the MSG G is the theory-consistent value.
    # (G_msg itself is reference-independent -- validated -- but the SG carries the recovery warping,
    # so its z_ref must sit at the chosen reference surface.)
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
    # compulsory: emit the per-station ABD yaml at the SAME reference (once, cached) for reuse
    # by dehom + shell buckling
    try:
        import os as _os
        from .emit_abd import emit_station_abd
        _tag = _os.path.splitext(_os.path.basename(shell_yaml))[0]
        _ay = _os.path.join(_os.path.dirname(shell_yaml) or ".", "abd", _tag + "_abd.yaml")
        if not _os.path.exists(_ay):
            emit_station_abd(shell_yaml, _ay, station=_tag,
                             ref="mid" if ref == "center" else "oml")
    except Exception:
        pass
    # OpenSG default: SwiftComp-format timed .out for the beam model too
    from opensg_solid.sg_homo import write_sc_K
    write_sc_K(_os2.path.splitext(shell_yaml)[0] + "_Timo.out", C6,
               solve_time=_time.perf_counter() - _t0,
               model="msg-shell beam model (Timoshenko 6x6,"
                     " [ext sh2 sh3 twist bend2 bend3])", constants=False)
    return {"Timo": C6, "V0": np.asarray(V0), "V1": np.asarray(V1),
            "corners": R["rx"][:, R["cross"]], "red_cells": np.asarray(R["cells"]),
            "rx3": np.asarray(R["rx"]), "re3": np.asarray(R["re3"]), "k22": np.asarray(R["k22"]),
            "ax": int(R["ax"]), "cross": list(R["cross"]), "strip": (nodes, quads, h),
            "layup_per_elem": layup_per_elem, "layup_db": kl["layup_db"],
            "material_db": kl["material_db"], "frac": frac, "ref": ref,
            "g_source": g_source}


def _rm_shell_strain(B, e, xi, st_m, aA, aB, s2_scheme="mitc4_g23"):
    """The 8 RM shell strains at element ``e``, arc ``xi`` in [0,1]:
    s6=[e11,e22,2e12,k11,k22,2k12], s2=[2g13,2g23].

    ``s2_scheme`` sets the tying used for the RECOVERED transverse shear s2: 'mitc4_g23'
    matches the homogenization (only g23 tied; g13 raw carries flat-wall drilling), while
    'mitc4_both' returns the drilling-free (tied) transverse shear on both rows -- the
    physical wall shear for stress recovery."""
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
    """RM-recovered warping DISPLACEMENT (u1,u2,u3) at query points.  The RM shell warping w=V0 st_m
    + V1 st_cl1 carries the mid-surface displacement (first 3 DOF) and the director rotation (last 3
    DOF, omega).  A point at through-thickness depth z from the reference contour moves as the RM
    kinematics dictate: u(z) = u_mid + z (omega x e3).  With ``director`` the depth term is included
    (needed for through-thickness paths); the circumferential path lies on the contour (z~=0) so it
    is unaffected.  Both RM and VABS warping are orthogonal to the rigid/classical modes, so they are
    directly comparable."""
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
    """Nodal (patch) average of the two contour-DERIVATIVE strain rows, 2eps12
    (row 2) and 2k12 (row 5), along the element chains.

    The RM ring is a linear 2-node element, so any strain row carrying the
    contour derivative of the warping is element-piecewise -- it oscillates
    about the smooth field (measured on iea_s10: element-to-element jumps of
    32% of the mean on row 2 and 182% on row 5, vs ~10% for the macro terms).
    Standard derivative-field recovery: average the element-midpoint values at
    shared nodes (only where exactly 2 elements meet -- junction nodes keep
    one-sided values), then interpolate linearly.  Rows 0,1,3,4 are NOT
    touched: their region-boundary jumps are physical."""
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
    """RM two-step dehom: 3-D stress at arbitrary section coords (y2,y3).

    Mirrors msg_dehom.stress_at_points but with the RM shell-strain recovery (step 1).  The
    in-plane sigma11/22/12 and sigma33 match VABS as with the KL dehom, but now from the RM
    (C0, MITC-g23) element, consistent with the paper's homogenization.

    ``rm_shear``: OFF by default.  The RM warping DOES carry the wall transverse shear s2, but
    the local *constitutive* recovery sigma13=G13*g13 is NOT physical for a spar cap -- the
    cap's transverse shear is an equilibrium (shear-flow) effect, so the constitutive parabola
    over-predicts ~20x and has the wrong through-thickness shape regardless of the tying
    scheme (validated: sweep_rm_shear.py).  A correct sigma13/23 needs the equilibrium
    shear-flow q(s), which is a separate development.  Left OFF so the shipped stress is the
    validated in-plane field (sigma13/23 = plate plane-stress limit, as in the KL dehom)."""
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
        # The RM ring reference is the frac-surface (mid-surface for center-ref, frac=0.5) but the
        # plate through-thickness SG (plate_stress_at_depth) measures depth from the OML (0..h).  So
        # convert the ring depth z (signed from the frac-ref) to the OML depth, and shift the shell
        # membrane strain from the frac-ref to the OML so the z*curvature term stays consistent.
        # Without this, all z<0 (outer-half) points are clamped to the OML ply -> wrong through-thickness.
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
