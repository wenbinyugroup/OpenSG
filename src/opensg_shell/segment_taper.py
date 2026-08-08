"""Aperiodic-boundary shell segment homogenization (tapered or prismatic).

The msg-shell counterpart of OpenSG-FEniCSx's boundary/segment flow
(SolidBounMesh + compute_timo_boun -> compute_stiffness(l_submesh, r_submesh)):

  1. the segment's two END CROSS-SECTIONS are extracted topologically from the
     3-D shell yaml (free-edge components) and written as separate 1-D yamls
     (`boundary_from_yaml.extract`);
  2. each boundary ring is solved on its own (`ring_indep`) -> the ring
     Timoshenko 6x6 and the warping fields V0, V1 (six DOFs per node,
     drilling omega_3 included);
  3. the ring fields are MAPPED onto the segment's boundary nodes (node2seg)
     and imposed as Dirichlet data replacing the axial periodicity;
  4. the segment Timoshenko 6x6 follows from the first-order (V1)
     finalization, with the energy divided by the segment length L.

For a PRISMATIC segment the result must reproduce the boundary ring's own
6x6 -- the seg==ring identity is the standing verification.
"""
import io
import contextlib
import json
import os
import time

import numpy as np

NDOF6 = 6            # u1 u2 u3 theta_1 theta_2 omega_3 (independent drilling)
NEPS = 4             # macro strain columns: eps_11, kappa_1, kappa_2, kappa_3


def _augment_drilling(Dhh, Dhe, Dhl, Dll, Dle, Gc, Gl, Ge):
    """Append the element-constant drilling Lagrange multipliers as a
    saddle-point block.

    The drilling rotation omega_3 is an independent 6th DOF; objectivity is
    restored by the element-wise constraint  g_e(V) = 0  enforced with one
    multiplier per element:

        [ Dhh  Gc^T ] [ V ]   [ -Dhe ]
        [ Gc    0   ] [ mu] = [ -Ge  ]        (zeroth order)

    Variables
    ---------
    Dhh, Dhe, Dhl, Dll, Dle : warping/strain energy blocks, 2U =
        V'^T Dll V' + 2 V'^T (Dhl^T V + Dle eps) + V^T Dhh V + 2 V^T Dhe eps
        + eps^T Dee eps   (V' = d V/d x_axis; MSG shell energy functional)
    Gc, Gl, Ge : constraint rows paired with (V, V', eps) respectively
    M, P       : warping DOF count, multiplier count
    returns    : (Dhh_a, Dhe_a, Dhl_a, Dll_a, Dle_a) on the augmented
                 space of size naug = M + P
    """
    M, P = Dhh.shape[0], Gc.shape[0]
    naug = M + P
    Dhh_a = np.zeros((naug, naug))
    Dhh_a[:M, :M] = Dhh
    Dhh_a[:M, M:] = Gc.T
    Dhh_a[M:, :M] = Gc
    Dhe_a = np.zeros((naug, NEPS)); Dhe_a[:M] = Dhe; Dhe_a[M:] = Ge
    Dhl_a = np.zeros((naug, naug)); Dhl_a[:M, :M] = Dhl; Dhl_a[M:, :M] = Gl
    Dll_a = np.zeros((naug, naug)); Dll_a[:M, :M] = Dll
    Dle_a = np.zeros((naug, NEPS)); Dle_a[:M] = Dle
    return Dhh_a, Dhe_a, Dhl_a, Dll_a, Dle_a


def _boundary_dirichlet(rings, b, key):
    """Map a boundary-ring warping field onto segment boundary DOFs.

    Replaces the periodic boundary condition of the ring problem: on each
    end cross-section the segment warping is PRESCRIBED to the separately
    solved ring solution,

        V_seg(x_boundary node, :) = V_ring(ring node, :)     (all 6 DOFs)

    Variables
    ---------
    rings : {"L"/"R": {"V0"/"V1": (6m x 4) ring field}} from ring_indep
    b     : the boundary bundle (npz) -- b["L_node2seg"][i] is the segment
            node index of ring node i (likewise "R_")
    key   : "V0" or "V1"
    V     : ring field reshaped (m_ring, 6, 4)
    bd    : constrained global DOF indices, 6*seg_node + component
    bv    : prescribed values, one (4,) row per constrained DOF
    """
    bd, bv = [], []
    for side in ("L", "R"):
        V = rings[side][key].reshape(-1, NDOF6, NEPS)
        for i, sn in enumerate(np.asarray(b["%s_node2seg" % side])):
            for c in range(NDOF6):
                bd.append(NDOF6*int(sn) + c)
                bv.append(V[i, c, :])
    return np.array(bd), np.array(bv, float)


def segment_timo_from_3dyaml(seg_yaml, workdir=None, lam_space="elem",
                             shear="full", write_boundary_yamls=True,
                             return_full=False):
    """Timoshenko 6x6 of a shell SEGMENT with aperiodic (Dirichlet) ends.

    Zeroth order (Euler-Bernoulli):
        Dhh_a V0 = -Dhe_a   with  V0 = V0_ring  on both end sections
        A_EB    = (Dee + V0^T Dhe_a) / L                       (4x4)
    First order (generalized Timoshenko, Yu et al.):
        Dhh_a V1 = b(V0)    with  V1 = V1_ring  on both end sections
        (b and the auxiliary blocks V0^T Dll V0, Dhl V0, Dhl^T V0 + Dle are
         formed by prepare_v1_rhs; finalize_v1_and_compute_deff performs the
         energy transformation (A_EB, B, C, D) -> S 6x6, each block / L)
    One LU factorization of the constrained Dhh_a serves both solves.

    Variables
    ---------
    seg_yaml  : 3-D shell segment yaml (nodes, quads, sections, materials)
    workdir   : output folder (default: alongside seg_yaml)
    lam_space : drilling-multiplier space; "elem" = element-constant
    shear     : transverse-shear integration for the SEGMENT operators;
                "full" is the production default (MITC tying aliases the
                drilling content on flat-walled/webbed sections)
    ax, cross : axial coordinate index and the two cross-section indices
    D_by, G_by: per-section laminate stiffness / shear blocks (center ref)
    k22_e,kg_e: element hoop-curvature and geometric-curvature corrections
    rings     : per-side ring results {C6, V0, V1} solved SEPARATELY
    Lz        : segment length = extent of nodes along the axis
    S6        : symmetrized segment Timoshenko 6x6
    returns   : dict(S6, C6L, C6R, L, solve_time [, V0, V1, rings, npz])
                and writes `<yaml>_Timo.out` (SwiftComp layout, timed)
    """
    import jax.numpy as jnp
    from .boundary_from_yaml import extract
    from .segment_element import (dirichlet_solve, dirichlet_factor,
                                  dirichlet_solve_fac, compute_k22,
                                  compute_kg)
    from .segment_indep import (assemble_segment_indep, assemble_constraint,
                                build_C_Psi_segment6)
    from .solve_segment_jax import _material_by_section
    from .fe_jax.msg_solver import prepare_v1_rhs, finalize_v1_and_compute_deff
    from .run_ring_indep import ring_indep

    t0 = time.perf_counter()
    base = os.path.splitext(seg_yaml)[0]
    if workdir is not None:
        base = os.path.join(workdir, os.path.basename(base))
    npz = base + "_boundaries.npz"
    # 1. boundary extraction: free-edge components -> 1-D yamls + node2seg
    with contextlib.redirect_stdout(io.StringIO()):
        extract(seg_yaml, npz, write_yaml=write_boundary_yamls)
    b = np.load(npz, allow_pickle=True)
    ax = int(b["axis"])
    cross = tuple(j for j in range(3) if j != ax)
    nodes = np.asarray(b["seg_x"])
    quads = np.asarray(b["seg_cells"])
    sd = np.asarray(b["seg_subdom"])
    e1s, e2s, e3s = (np.asarray(b["seg_e1"]), np.asarray(b["seg_e2"]),
                     np.asarray(b["seg_e3"]))
    D_by, G_by = _material_by_section(json.loads(str(b["sections"])),
                                      json.loads(str(b["materials"])),
                                      center_ref=True)
    cents = nodes[quads].mean(1)
    k22_e = compute_k22(cents, e2s, e3s, quads)
    kg_e = compute_kg(cents, e1s, e2s, e3s, quads)

    # 2. boundary rings solved SEPARATELY -> C6, V0, V1 per side
    rings = {}
    for side in ("L", "R"):
        rx = np.asarray(b["%s_x" % side])
        rc = np.asarray(b["%s_cells" % side])
        rs = np.asarray(b["%s_subdom" % side])
        re3 = np.asarray(b["%s_e3" % side])
        kr = compute_k22(rx[rc].mean(1), np.asarray(b["%s_e2" % side]),
                         re3, rc)
        C6r, V0r, V1r = ring_indep(rx, rc, rs, re3, D_by, G_by, kr, ax,
                                   list(cross), lam_space=lam_space,
                                   return_fields=True)
        rings[side] = dict(C6=C6r, V0=V0r, V1=V1r)

    # 3. segment operators + drilling constraint (saddle point, pen=0)
    Dhh, Dhe, Dee, Dhl, Dll, Dle = assemble_segment_indep(
        nodes, quads, sd, e3s, D_by, G_by, k22_e, cross, ax, kg_e=kg_e,
        pen=0.0, shear=shear)
    Gc, Gl, Ge = assemble_constraint(nodes, quads, sd, e3s, k22_e, cross, ax,
                                     kg_e=kg_e, lam_space=lam_space)
    Dhh, Dhe, Dhl, Dll, Dle = map(np.asarray, (Dhh, Dhe, Dhl, Dll, Dle))
    M = Dhh.shape[0]
    Dhh_a, Dhe_a, Dhl_a, Dll_a, Dle_a = _augment_drilling(
        Dhh, Dhe, Dhl, Dll, Dle, np.asarray(Gc), np.asarray(Gl),
        np.asarray(Ge))
    naug = Dhh_a.shape[0]
    # rigid-body projector C and macro mode shapes Psi on the augmented space
    C, Psi = build_C_Psi_segment6(nodes, quads, cross)
    Psi[3::6, 3] *= -1.0             # torsion mode sign convention
    Psi_a = np.zeros((naug, NEPS)); Psi_a[:M] = Psi
    Dc_a = np.zeros((naug, NEPS)); Dc_a[:M] = C.T

    # 4a. zeroth order:  Dhh_a V0 = -Dhe_a,  V0 = V0_ring on the ends
    bd0, bv0 = _boundary_dirichlet(rings, b, "V0")
    fac = dirichlet_factor(Dhh_a, bd0)
    V0 = dirichlet_solve_fac(fac, -Dhe_a, bd0, bv0)
    Lz = float(nodes[:, ax].max() - nodes[:, ax].min())
    EB = (np.asarray(Dee) + V0.T @ Dhe_a)/Lz           # A_EB (4x4), per length
    # 4b. first order:  Dhh_a V1 = b(V0),  V1 = V1_ring on the ends
    bb, DhlV0, DhlTV0Dle, V0DllV0 = prepare_v1_rhs(
        jnp.array(V0), jnp.array(Dhl_a), jnp.array(Dll_a), jnp.array(Dle_a),
        jnp.array(Psi_a), jnp.array(Dc_a))
    bd1, bv1 = _boundary_dirichlet(rings, b, "V1")
    V1 = (dirichlet_solve_fac(fac, np.asarray(bb), bd1, bv1)
          if np.array_equal(bd0, bd1) else
          dirichlet_solve(Dhh_a, np.asarray(bb), bd1, bv1))
    # generalized-Timoshenko transformation; energy blocks per unit length
    S6, *_ = finalize_v1_and_compute_deff(
        jnp.array(V1), jnp.array(V0), jnp.array(EB),
        jnp.array(np.asarray(V0DllV0)/Lz), jnp.array(np.asarray(DhlV0)/Lz),
        jnp.array(np.asarray(DhlTV0Dle)/Lz), jnp.array(Psi_a),
        jnp.array(Dc_a))
    S6 = 0.5*(np.asarray(S6) + np.asarray(S6).T)
    solve_time = time.perf_counter() - t0

    from opensg_solid.sg_homo import write_sc_K
    write_sc_K(base + "_Timo.out", S6, solve_time=solve_time,
               model="msg-shell tapered/aperiodic segment"
                     " (boundary V0/V1 Dirichlet, L=%.6g)" % Lz,
               constants=False, name="Timoshenko")
    out = dict(S6=S6, C6L=rings["L"]["C6"], C6R=rings["R"]["C6"], L=Lz,
               solve_time=solve_time)
    if return_full:
        out.update(V0=np.asarray(V0[:M]), V1=np.asarray(V1[:M]),
                   rings=rings, npz=npz)
    return out
