"""RM-consistent thin-walled dehomogenization: two-step recovery of 3-D
stress/strain and warping displacement from the RM ring homogenization bundle.
Merged from: dehom_rm.py (build_rm_bundle and _strip moved to sg_homo; the
bundle consumed here is produced by sg_homo.build_rm_bundle).

dehom_rm.py -- RM-consistent thin-walled dehomogenization
---------------------------------------------------------
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

Public entry points: disp_at_points, stress_at_points, ring_wall_strains
(bundle from sg_homo.build_rm_bundle).  ring_wall_strains is the per-ELEMENT
step-1 field builder shared by the CLI D route and the shell dehom example --
it never re-projects a point onto the nearest element, so junction points
keep their own element's layup.
"""
import numpy as np
import jax.numpy as jnp

from .sg_assembly import quad_ops_indep, _mitc_shear_indep          # RM 8-strain operator
from .fe_jax.msg_dehom import (_macro_recovery, _project_point,
                                         _voigt_to_tensor, _tensor_to_voigt)
from .fe_jax.msg_materials import (compute_ABD_matrix, plate_stress_at_depth,
                                             rotation_6x6)
from .fe_jax.msg_transverse_shear import transverse_shear_stiffness


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


def ring_wall_strains(B, beam_force_vabs=None, beam_strain=None):
    """Step-1 (section) RM wall strain fields of the WHOLE ring, per element.

    The exact code path of the shell dehom example (beam_dehom_shell.py),
    lifted here so the CLI and the example share it: _macro_fields, then per
    element the 6 RM shell strains at the mid-arc (_rm_shell_strain), the span
    strain gradient dE1 from the element operators, and the arc gradient dE2
    from the LAYUP-BOUNDARY-AWARE nodal average (element-midpoint values are
    averaged only at nodes shared by exactly two SAME-layup elements -- no
    differencing ever crosses a section change, and no point is ever
    re-projected onto a neighbouring element).

    In:
        B: dict, RM bundle from build_rm_bundle.
        beam_force_vabs: (6,) beam force/moment in VABS order, or None.
        beam_strain: (6,) beam strain, or None (exactly one of the two must be given).
    Out:
        dict with:
            "st": (6,) macro beam strain used.
            "aA"/"aB": (6m,) nodal RM warping w / w'.
            "s6mid": (E,6) shell strains [e11 e22 2e12 k11 k22 2k12] at mid-arc.
            "dE1": (E,6) span gradients dE/dx1.
            "dE2": (E,6) arc gradients dE/dx2.
            "s6n": (N,6) same-layup nodal averages of s6mid (NaN elsewhere).
            "emid": (E,2) element mid-arc points; "nvec": (E,2) inward normals.
            "Lel": (E,) element arc lengths; "h": (E,) layup thicknesses.
    """
    st, st_m, aA, aB = _macro_fields(B, beam_force_vabs, beam_strain)
    _, _sm, st_cl1, _cl2 = _macro_recovery(np.asarray(B["Timo"]), st)

    corners = np.asarray(B["corners"]); rc = np.asarray(B["red_cells"])
    cen = corners.mean(0)
    nodes_s, quads_s, _hs = B["strip"]
    n_el = rc.shape[0]; n_nd = int(rc.max()) + 1
    layups = B["layup_per_elem"]
    hth = {ln: float(sum(i["thick"])) for ln, i in B["layup_db"].items()}
    h = np.array([hth[ln] for ln in layups])

    s6mid = np.zeros((n_el, 6)); dE1 = np.zeros((n_el, 6))
    for e in range(n_el):
        s6, _ = _rm_shell_strain(B, e, 0.5, st_m, aA, aB)
        s6mid[e] = np.asarray(s6, float)
        Xe = nodes_s[quads_s[e]]; e3e = B["re3"][e]
        BDe, BDh, BDl, *_ = quad_ops_indep(Xe, e3e, 0.0, 0.0, float(B["k22"][e]),
                                           B["cross"], B["ax"])
        c0, c1 = int(rc[e, 0]), int(rc[e, 1])
        g = np.r_[c0 * 6:c0 * 6 + 6, c1 * 6:c1 * 6 + 6, c1 * 6:c1 * 6 + 6, c0 * 6:c0 * 6 + 6]
        dE1[e] = np.asarray(BDe @ st_cl1 + BDh @ aB[g], float)

    # layup-boundary-aware nodal average of s6 (only where two SAME-layup elements meet)
    deg = np.zeros(n_nd, int)
    nd_el = [[] for _ in range(n_nd)]
    for e in range(n_el):
        for nd in (int(rc[e, 0]), int(rc[e, 1])):
            deg[nd] += 1; nd_el[nd].append(e)
    s6n = np.full((n_nd, 6), np.nan)
    for nd in range(n_nd):
        if deg[nd] == 2 and layups[nd_el[nd][0]] == layups[nd_el[nd][1]]:
            s6n[nd] = 0.5 * (s6mid[nd_el[nd][0]] + s6mid[nd_el[nd][1]])
    T = corners[rc[:, 1]] - corners[rc[:, 0]]
    Lel = np.sqrt((T ** 2).sum(1))
    dE2 = np.zeros((n_el, 6))
    for e in range(n_el):
        c0, c1 = int(rc[e, 0]), int(rc[e, 1])
        v0 = s6n[c0] if np.isfinite(s6n[c0, 0]) else s6mid[e]
        v1 = s6n[c1] if np.isfinite(s6n[c1, 0]) else s6mid[e]
        dE2[e] = (v1 - v0) / Lel[e]

    # inward wall normal per element (plate +z = OML -> IML)
    tun = T / Lel[:, None]
    nvec = np.column_stack([tun[:, 1], -tun[:, 0]])
    emid = 0.5 * (corners[rc[:, 0]] + corners[rc[:, 1]])
    flip = ((cen - emid) * nvec).sum(1) < 0
    nvec[flip] *= -1.0
    return {"st": np.asarray(st, float), "aA": np.asarray(aA), "aB": np.asarray(aB),
            "s6mid": s6mid, "dE1": dE1, "dE2": dE2, "s6n": s6n,
            "emid": emid, "nvec": nvec, "Lel": Lel, "h": h}


def _wall_band(B, e):
    """The recovery band of ring element ``e`` about its reference surface.

    The wall owns depths z in [-frac*h, (1-frac)*h] (h = its OWN layup
    thickness, frac = B["frac"]); with the default center reference that is
    exactly +-h/2, the wall's own half-thickness.

    In:  B: dict, RM bundle; e: int, ring element index.
    Out: (lo, hi, h) floats.
    """
    h = float(sum(B["layup_db"][B["layup_per_elem"][int(e)]]["thick"]))
    frac = float(B.get("frac", 0.0))
    return -frac * h, (1.0 - frac) * h, h


def _warn_off_wall(bad, api):
    """One aggregated warning for query points that land outside the wall they
    were projected onto.

    ``_project_point`` returns the NEAREST ring element, and the plate SG's
    ``msg_rm_plate._locate`` then CLAMPS the depth into [0, h].  A point that
    really lives in the CROSSING wall of a junction therefore comes back
    silently carrying the wrong wall's surface-ply state -- the origin of a
    measured 274 MPa error at the IEA-22 spar-cap/web junctions.  The hazard is
    REPORTED, never repaired: the return values of the two public APIs are
    unchanged, so no consumer breaks and no result silently moves.  A caller
    that wants the points partitioned instead should flag/exclude them with
    sg_dehom_junction (the yaml `junction:` tier).

    In:
        bad: list of (overshoot, ip, e, z, lo, hi, h) per offending point.
        api: str, the public API name to name in the message.
    Out:
        None (raises nothing; emits at most ONE RuntimeWarning).
    """
    if not bad:
        return
    import warnings
    over, ip, e, z, lo, hi, h = max(bad, key=lambda r: r[0])
    warnings.warn(
        "%s: %d query point(s) project onto a ring element that does NOT own"
        " them -- the depth falls outside that wall's own thickness band, and"
        " the through-thickness SG CLAMPS it to the laminate surface, so the"
        " reported state is the WRONG wall's surface ply (this is the"
        " spar-cap/web junction failure mode).  Worst: point %d -> element %d,"
        " depth %.6g m outside [%.6g, %.6g] (h = %.6g m, overshoot %.3g x h)."
        " Values are returned unchanged; use the yaml `junction:` tier"
        " (sg_dehom_junction) to flag or exclude such stations."
        % (api, len(bad), ip, e, z, lo, hi, h, over / h),
        RuntimeWarning, stacklevel=3)


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

    Warns (RuntimeWarning) when a point's nearest-element projection puts it
    outside that wall's own thickness band -- see :func:`_warn_off_wall`.  The
    returned values are unchanged.
    """
    pts = np.atleast_2d(np.asarray(points_2d, float))
    st, st_m, aA, aB = _macro_fields(B, beam_force_vabs, beam_strain)
    wn = np.asarray(aA).reshape(-1, 6)                       # per-node [u1,u2,u3,om1,om2,om3]
    corners = np.asarray(B["corners"]); rc = np.asarray(B["red_cells"]); cen = corners.mean(0)
    out = np.zeros((len(pts), 3))
    bad = []
    for i in range(len(pts)):
        e, xi, pr = _project_point(corners, rc, pts[i])
        # the projected element must be able to REACH the point: |p - proj|
        # beyond the wall's own band means the nearest element is not the wall
        # that owns it (a junction cross-wall) -- report, never repair
        _lo, _hi, _h = _wall_band(B, e)
        _zmax = max(-_lo, _hi)
        _dpr = float(np.hypot(*(pts[i] - pr)))
        if _dpr > _zmax * (1.0 + 1e-9):
            bad.append((_dpr - _zmax, i, int(e), _dpr, -_zmax, _zmax, _h))
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
    _warn_off_wall(bad, "disp_at_points")
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

    Warns (RuntimeWarning) when a point's nearest-element projection puts its
    depth outside that wall's own band [-frac*h, (1-frac)*h] -- the depth the
    plate SG then CLAMPS to the laminate surface; see :func:`_warn_off_wall`.
    The returned values are unchanged.
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
    bad = []
    for ip in range(P):
        e, xi, pr = _project_point(corners, rc, pts[ip])
        c0, c1 = int(rc[e, 0]), int(rc[e, 1])
        t2, t3 = corners[c1] - corners[c0]
        tl = float(np.hypot(t2, t3)); t2, t3 = t2 / tl, t3 / tl       # unit arc tangent
        n2, n3 = t3, -t2                                              # inward normal (plate z)
        if (cen[0] - pr[0]) * n2 + (cen[1] - pr[1]) * n3 < 0.0:
            n2, n3 = -n2, -n3
        z = float((pts[ip, 0] - pr[0]) * n2 + (pts[ip, 1] - pr[1]) * n3)
        # a depth outside this wall's OWN band is silently clamped by
        # msg_rm_plate._locate -- flag it before the wrong ply answers
        _lo, _hi, _hw = _wall_band(B, e)
        _ov = max(_lo - z, z - _hi)
        if _ov > 1e-9 * _hw:
            bad.append((_ov, ip, int(e), z, _lo, _hi, _hw))

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
    _warn_off_wall(bad, "stress_at_points")
    return {"stress": stress, "strain": strain, "elem": el, "xi": xia,
            "depth": dep, "proj": proj, "macro": st}
