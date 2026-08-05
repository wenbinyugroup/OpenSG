"""
MSG thin-walled (TW) dehomogenization — two-step strain recovery.

Given a solved cross-section (Timoshenko 6x6, plus the EB warping ``V0`` and the
shear warping ``V1``) and an applied beam force, recover the local strain field
in two steps:

**Step 1 — shell strain recovery** (this module, :func:`recover_shell_strains`).
Mirrors the OpenSG solid ``stress_recov.local_strain`` but with the MSG *shell*
strain operators (``eps_h`` / ``eps_l`` / ``eps_e`` from :mod:`msg_hermite`):

    shell_strain = eps_h(V0 @ st_m) + Ge @ st_m              (Euler-Bernoulli)
                 + eps_h(V1 @ st_cl1) + eps_l(V0 @ st_cl1)
                 + eps_l(V1 @ st_cl2)                         (Timoshenko)

``st_m`` is the macro EB strain from the compliance, and ``st_cl1`` / ``st_cl2``
are the Timoshenko derivative-recovery terms built from the ``recov`` chain
(identical to the solid).  The result is the 6 plate/shell strains
[eps11, eps22, gamma12, kappa11, kappa22, kappa12] at every cross-section
quadrature point.

**Step 2 — plate dehomogenization** (:func:`dehomogenize`, using
:func:`msg_materials.plate_dehom_strain`).  For each line element the 6 shell
strains drive the same MSG plate (through-thickness 1D SG) model that produced
the ABD matrix, recovering the pointwise 3D strain/stress across the thickness.

Force convention: VABS order ``[F1, F2, F3, M1, M2, M3]`` = [axial, shear-2,
shear-3, torsion, bending-2, bending-3].  (From BeamDyn reactions use
``FF = [rf[2], -rf[1], rf[0], rf[5], -rf[4], rf[3]]``.)
"""
import numpy as np
import jax.numpy as jnp

from .msg_hermite import hermite_strain_operators, solve_tw_from_yaml
from .msg_materials import (compute_ABD_matrix, plate_dehom_strain,
                           plate_stress_at_depth, rotation_6x6)

# EB<->shear coupling map (same Q as the solid recovery / finalize_v1)
_Q = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, -1.0], [1.0, 0.0]])


def _recov(st):
    """6x6 Timoshenko derivative-recovery matrix (matches solid ``utils.recov``)."""
    R = np.zeros((6, 6))
    R[0, 1], R[0, 2] = st[5], -st[4]
    R[1, 0], R[1, 2] = -st[5], st[3]
    R[2, 0], R[2, 1] = st[4], -st[3]
    R[3:6, 3:6] = R[0:3, 0:3]
    R[3, 1], R[3, 2] = st[2], -st[1]
    R[4, 0], R[4, 2] = -st[2], st[0]
    R[5, 0], R[5, 1] = st[1], -st[0]
    return R


def _macro_recovery(Ceff_srt, st):
    """Macro EB strain ``st_m`` and the two Timoshenko derivative terms.

    Matches the corrected OpenSG-FEniCS ``stress_recov.local_strain``: the
    derivative-recovery operator is the *fixed* ``R1 = recov([1,0,0,0,0,0])``
    and the higher derivatives are its powers, ``F_{k+1} = R1 @ F_k`` (the old
    ``recov(st+e0)`` / R2 / R3 terms are dropped).

    Parameters
    ----------
    Ceff_srt : (6,6) Timoshenko stiffness
    st       : (6,) prescribed beam strain [eps11, gamma12, gamma13, k1, k2, k3]

    Returns
    -------
    (st, st_m, st_cl1, st_cl2) : st unchanged, st_m the Q-corrected 4 EB strains,
    st_cl1 / st_cl2 the Q-corrected first / second derivative recoveries (len 4).
    """
    Comp = np.linalg.inv(Ceff_srt)
    st = np.asarray(st, dtype=float)
    st_m = np.array([st[0], st[3], st[4], st[5]])

    F_1d = Ceff_srt @ st                      # equivalent beam force
    st_lin = np.zeros(6); st_lin[0] = 1.0
    R1 = _recov(st_lin)                       # fixed recovery operator

    F1 = R1 @ F_1d
    stT1 = Comp @ F1
    st_cl1 = np.array([stT1[0], stT1[3], stT1[4], stT1[5]]); gamma1 = stT1[[1, 2]]

    F2 = R1 @ F1
    stT2 = Comp @ F2
    st_cl2 = np.array([stT2[0], stT2[3], stT2[4], stT2[5]]); gamma2 = stT2[[1, 2]]

    F3 = R1 @ F2
    stT3 = Comp @ F3; gamma3 = stT3[[1, 2]]

    st_m = st_m + _Q @ gamma1
    st_cl1 = st_cl1 + _Q @ gamma2
    st_cl2 = st_cl2 + _Q @ gamma3
    return st, st_m, st_cl1, st_cl2


def recover_shell_strains(bundle, beam_force_vabs=None, beam_strain=None,
                          xi_eval=None):
    """Step 1 — recover the 6 shell strains along the cross-section.

    Provide exactly one of ``beam_force_vabs`` or ``beam_strain``.  The beam
    strain ``st`` = [eps11, gamma12, gamma13, kappa1, kappa2, kappa3] is taken
    directly when given (e.g. a prescribed unit strain), otherwise it is
    recovered from the force via the compliance ``inv(Timo) @ force``.

    Parameters
    ----------
    bundle : dict — from :func:`msg_hermite.solve_tw_from_yaml`
    beam_force_vabs : (6,) — applied beam force in VABS order, OR
    beam_strain : (6,) — prescribed beam strain in VABS order
    xi_eval : (P,) — arc positions in [0,1] to evaluate at; default = the bundle
              Gauss points.  Use ``[0.0, 1.0]`` to get the two end-node values.

    Returns
    -------
    dict with:
      ``shell_strain`` (E, P, 6) — [eps11, eps22, gamma12, kappa11, kappa22, kappa12]
      ``shell_strain_elem`` (E, 6) — averaged per element
      ``x2``, ``x3`` (E, P) — arc-point coordinates;  ``xi`` (P,) — the positions
      ``tang`` (E, 2) — unit arc tangent (xd2, xd3) per element
      ``macro`` (6,) — beam strain ``st``;  ``st_m`` (4,) — EB macro strain
    """
    Ceff_srt = np.asarray(bundle["Timo"])
    V0 = np.asarray(bundle["V0"]); V1 = np.asarray(bundle["V1"])
    if (beam_strain is None) == (beam_force_vabs is None):
        raise ValueError("provide exactly one of beam_force_vabs or beam_strain")
    if beam_strain is not None:
        st = np.asarray(beam_strain, dtype=float)
    else:
        st = np.linalg.inv(Ceff_srt) @ np.asarray(beam_force_vabs, dtype=float)

    st, st_m, st_cl1, st_cl2 = _macro_recovery(Ceff_srt, st)

    a1 = V0 @ st_m           # EB warping        (w_1)
    a2 = V1 @ st_cl1         # shear warping     (w1s_1)
    a3 = V1 @ st_cl2         # shear warping     (w1s_2)
    a4 = V0 @ st_cl1         # EB-of-derivative  (w_2)
    st_m_j = jnp.array(st_m)

    rc = np.asarray(bundle["red_cells"]); corners = np.asarray(bundle["corners"])
    k22 = np.asarray(bundle["k22"]); L = np.asarray(bundle["L"])
    xd2 = np.asarray(bundle["xd2"]); xd3 = np.asarray(bundle["xd3"])
    xi = bundle["xi_q"] if xi_eval is None else jnp.array(xi_eval, dtype=float)
    xin = np.asarray(xi)
    n_elem = rc.shape[0]; P = xin.shape[0]

    shell = np.zeros((n_elem, P, 6))
    x2 = np.zeros((n_elem, P)); x3 = np.zeros((n_elem, P))
    for e in range(n_elem):
        c0, c1 = int(rc[e, 0]), int(rc[e, 1])
        g = np.r_[c0 * 6:c0 * 6 + 6, c1 * 6:c1 * 6 + 6]
        n0 = corners[c0]; n1 = corners[c1]
        eps_h, eps_l, eps_e, _ = hermite_strain_operators(
            jnp.array(n0), jnp.array(n1), float(k22[e]), float(L[e]),
            float(xd2[e]), float(xd3[e]), xi)
        s = (eps_h(jnp.array(a1[g])) + eps_e(st_m_j)
             + eps_h(jnp.array(a2[g])) + eps_l(jnp.array(a4[g]))
             + eps_l(jnp.array(a3[g])))
        shell[e] = np.asarray(s)
        x2[e] = (1.0 - xin) * n0[0] + xin * n1[0]
        x3[e] = (1.0 - xin) * n0[1] + xin * n1[1]

    return {"shell_strain": shell, "shell_strain_elem": shell.mean(axis=1),
            "x2": x2, "x3": x3, "xi": xin,
            "tang": np.stack([xd2, xd3], axis=1),
            "macro": st, "st_m": st_m}


def dehomogenize(yaml_path, beam_force_vabs=None, beam_strain=None,
                 n_eval_per_elem=3, bundle=None, elem_order=2, n_per_layer=2):
    """Full two-step dehomogenization for a YAML cross-section.

    Step 1 recovers the shell strains; step 2 runs the MSG plate model per
    element to recover the 3D strain/stress across the thickness (using the
    element-mean shell strain as the plate input).  Provide exactly one of
    ``beam_force_vabs`` or ``beam_strain``.

    Parameters
    ----------
    yaml_path : str — OpenSG cross-section YAML
    beam_force_vabs : (6,) — applied beam force in VABS order, OR
    beam_strain : (6,) — prescribed beam strain in VABS order
    n_eval_per_elem : int — through-thickness sample points per plate sub-element
    bundle : dict, optional — a precomputed :func:`solve_tw_from_yaml` bundle
             (avoids re-solving when dehomogenizing several load cases)
    elem_order : int — plate (step-2) element order: 2 = quadratic 3-node
             (default), 3 = cubic 4-node.  Identical for uniform plies (the
             quadratic plate model is already exact); 3 refines non-uniform layers.
    n_per_layer : int — through-thickness plate sub-elements per ply layer
             (default 2).  The step-2 1D SG spans the whole laminate thickness;
             2 sub-elements per layer refine the warping/stress recovery (and
             double the through-thickness sample points) while staying exact for
             uniform plies — i.e. it never degrades a validated result.

    Returns
    -------
    dict with the step-1 fields plus, for step 2:
      ``elem`` — list (length E) of per-element dicts
                 {``z``, ``strain_3d`` (n,6), ``stress_3d`` (n,6), ``layup``}
      ``bundle`` — the solution bundle (for reuse / inspection)
    """
    if bundle is None:
        bundle = solve_tw_from_yaml(yaml_path)
    shell = recover_shell_strains(bundle, beam_force_vabs, beam_strain)

    # plate warping cache per layup name
    warp_cache = {}
    for ln, info in bundle["layup_db"].items():
        _, _, warp = compute_ABD_matrix(
            info["thick"], info["angles"], info["mat_names"],
            bundle["material_db"], n_per_layer=n_per_layer,
            return_warping=True, elem_order=elem_order)
        warp_cache[ln] = warp

    ss_elem = shell["shell_strain_elem"]
    layups = bundle["layup_per_elem"]
    elem = []
    for e in range(len(layups)):
        ln = layups[e]
        z, Gam, Sig = plate_dehom_strain(warp_cache[ln], ss_elem[e], n_eval_per_elem)
        elem.append({"z": z, "strain_3d": Gam, "stress_3d": Sig, "layup": ln})

    out = dict(shell)
    out["elem"] = elem
    out["bundle"] = bundle
    return out


# ----------------------------------------------------------------------------
# Stress at ARBITRARY cross-section coordinates (point evaluation)
# ----------------------------------------------------------------------------

def _voigt_to_tensor(s):
    return np.array([[s[0], s[5], s[4]], [s[5], s[1], s[3]], [s[4], s[3], s[2]]])


def _tensor_to_voigt(T):
    return np.array([T[0, 0], T[1, 1], T[2, 2], T[1, 2], T[0, 2], T[0, 1]])


def _project_point(corners, rc, p):
    """Project 2D point ``p`` onto the 1D reference mesh.

    Returns (elem, xi, proj) — the nearest element, the arc parameter xi in
    [0,1] along it, and the projected point on the reference curve.
    """
    best = (np.inf, 0, 0.0, corners[0])
    for e in range(rc.shape[0]):
        A = corners[int(rc[e, 0])]; B = corners[int(rc[e, 1])]
        AB = B - A; L2 = float(AB @ AB)
        t = 0.0 if L2 < 1e-30 else float(np.clip((p - A) @ AB / L2, 0.0, 1.0))
        proj = A + t * AB
        d = float(np.hypot(*(p - proj)))
        if d < best[0]:
            best = (d, e, t, proj)
    return best[1], best[2], best[3]


def stress_at_points(bundle, points_2d, beam_force_vabs=None, beam_strain=None,
                     frame="global", elem_order=2, n_per_layer=2):
    """Recover the 3D stress at ARBITRARY cross-section coordinates.

    For each point ``(y2, y3)`` the point is projected onto the 1D reference
    mesh to get its arc position (element + xi) and its through-thickness depth
    (signed distance from the reference surface toward the section interior,
    0 = OML).  The recovered shell strain at that arc position drives the MSG
    plate model at that depth, giving the local 3D stress.  This lets the TW
    model report stress on ANY path/point in the cross-section (e.g. exactly at
    the FEniCS solid sample coordinates).

    The step-2 plate model is a through-thickness 1D SG that spans the WHOLE
    laminate thickness, so any depth ``z in [0, h]`` is covered.  ``n_per_layer``
    sub-elements per ply refine that 1D SG (default 2); it stays exact for
    uniform plies, so refining never changes a validated point value.

    Parameters
    ----------
    bundle : dict — from :func:`solve_tw_from_yaml`
    points_2d : (P,2) cross-section coordinates (y2, y3)
    beam_force_vabs / beam_strain : (6,) — exactly one (VABS order)
    frame : "global" (y1,y2,y3) or "local" (1=beam, 2=tangent, 3=normal)
    elem_order : plate (step-2) element order (2 or 3)
    n_per_layer : through-thickness plate sub-elements per ply layer (default 2)

    Returns
    -------
    dict with:
      ``stress`` (P,6) — [S11,S22,S33,S23,S13,S12]
      ``strain`` (P,6) — 3D strain in the same order
      ``elem`` (P,), ``xi`` (P,), ``depth`` (P,) — projection (depth 0 = OML)
      ``proj`` (P,2) — projected point on the reference curve
    """
    pts = np.atleast_2d(np.asarray(points_2d, dtype=float))
    Ceff_srt = np.asarray(bundle["Timo"])
    V0 = np.asarray(bundle["V0"]); V1 = np.asarray(bundle["V1"])
    if (beam_strain is None) == (beam_force_vabs is None):
        raise ValueError("provide exactly one of beam_force_vabs or beam_strain")
    st = (np.asarray(beam_strain, float) if beam_strain is not None
          else np.linalg.inv(Ceff_srt) @ np.asarray(beam_force_vabs, float))
    st, st_m, st_cl1, st_cl2 = _macro_recovery(Ceff_srt, st)
    a1 = V0 @ st_m; a2 = V1 @ st_cl1; a3 = V1 @ st_cl2; a4 = V0 @ st_cl1
    st_m_j = jnp.array(st_m)

    rc = np.asarray(bundle["red_cells"]); corners = np.asarray(bundle["corners"])
    k22 = np.asarray(bundle["k22"]); L = np.asarray(bundle["L"])
    xd2 = np.asarray(bundle["xd2"]); xd3 = np.asarray(bundle["xd3"])
    layups = bundle["layup_per_elem"]
    cen = corners.mean(axis=0)
    warp = {ln: compute_ABD_matrix(i["thick"], i["angles"], i["mat_names"],
            bundle["material_db"], n_per_layer=n_per_layer,
            return_warping=True, elem_order=elem_order)[2]
            for ln, i in bundle["layup_db"].items()}
    # reference offset (frac x thickness inward from the OML): the plate warp is
    # OML-referenced, so the plate depth adds frac*h and the recovered membrane
    # strain is converted back to the OML (m_OML = m_ref - frac*h*kappa).
    fr = float(bundle.get("frac", 0.0))
    h_elem = {ln: float(sum(i["thick"])) for ln, i in bundle["layup_db"].items()}

    P = len(pts)
    stress = np.zeros((P, 6)); strain = np.zeros((P, 6))
    el = np.zeros(P, int); xia = np.zeros(P); dep = np.zeros(P); proj = np.zeros((P, 2))
    for ip in range(P):
        e, xi, pr = _project_point(corners, rc, pts[ip])
        # inward normal (toward the section interior) = the plate z-direction
        t2, t3 = float(xd2[e]), float(xd3[e]); n2, n3 = t3, -t2
        if (cen[0] - pr[0]) * n2 + (cen[1] - pr[1]) * n3 < 0.0:
            n2, n3 = -n2, -n3
        z = float((pts[ip, 0] - pr[0]) * n2 + (pts[ip, 1] - pr[1]) * n3)

        c0, c1 = int(rc[e, 0]), int(rc[e, 1])
        g = np.r_[c0 * 6:c0 * 6 + 6, c1 * 6:c1 * 6 + 6]
        eps_h, eps_l, eps_e, _ = hermite_strain_operators(
            jnp.array(corners[c0]), jnp.array(corners[c1]), float(k22[e]),
            float(L[e]), t2, t3, jnp.array([xi]))
        ss = np.asarray(eps_h(jnp.array(a1[g])) + eps_e(st_m_j)
                        + eps_h(jnp.array(a2[g])) + eps_l(jnp.array(a4[g]))
                        + eps_l(jnp.array(a3[g])))[0]
        # convert the reference-frac dehom back to the OML-referenced plate model
        he = h_elem[layups[e]]
        if fr:
            ss = ss.copy(); ss[0:3] = ss[0:3] - fr * he * ss[3:6]
        Gam, Sig, ply = plate_stress_at_depth(warp[layups[e]], ss, z + fr * he)
        if frame == "global":
            R = np.array([[1., 0, 0], [0, t2, n2], [0, t3, n3]])
            Sig = _tensor_to_voigt(R @ _voigt_to_tensor(Sig) @ R.T)
            Gam = _tensor_to_voigt(R @ _voigt_to_tensor(Gam) @ R.T)
        elif frame == "material":
            # laminate -> ply/fiber frame (in-plane rotation by the ply angle)
            Rm = rotation_6x6(-ply)
            Sig = Rm @ Sig; Gam = rotation_6x6(ply).T @ Gam
        stress[ip] = Sig; strain[ip] = Gam
        el[ip] = e; xia[ip] = xi; dep[ip] = z + fr * he; proj[ip] = pr
    return {"stress": stress, "strain": strain, "elem": el, "xi": xia,
            "depth": dep, "proj": proj}
