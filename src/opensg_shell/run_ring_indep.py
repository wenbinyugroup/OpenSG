"""run_ring_indep.py -- 6-DOF (independent omega_3, element-lambda Lagrange) BOUNDARY RING
solver + shear-scheme comparison against the validated 5-DOF MITC ring and the 3-D solid.

Ring = one-quad-deep prismatic strip with the top node row DOF-mapped onto the bottom row
(exactly ring_general's construction), assembled with the constrained 6-DOF element; the
drilling constraint is enforced by element-constant Lagrange multipliers and the rigid
modes by the standard KKT rows.  All matrices are per-unit-length (/h).

Public entry points: ring_indep (constrained 6-DOF ring solve), main (shear-scheme
comparison study).
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
BENCH = os.path.join(REPO, "examples", "data", "benchmark")


def ring_indep(rx, rcells, rsub, re3, D_by, G_by, k22_edge, ax, cross, h=None,
               shear="mitc4_g23", lam_space="elem", return_fields=False):
    """Constrained 6-DOF ring SG.  Returns the ring Timoshenko C6 (6,6); with
    return_fields=True also the zeroth/first-order warping fields V0, V1 (6m x 4,
    multiplier rows stripped) for the segment Dirichlet transfer -- including the
    drilling omega_3 boundary values.

    PRODUCTION shear scheme (ring): 'mitc4_g23' -- tie ONLY gamma_23.  Under span
    invariance gamma_13 carries no fluctuation gradient (it is algebraic in the
    directors), so only gamma_23 pairs a differentiated displacement with rotations."""
    import jax.numpy as jnp
    from .segment_indep import (assemble_segment_indep, assemble_constraint, NDOF6)
    from .fe_jax.msg_rm_timo import build_C_Psi
    from .fe_jax.msg_solver import prepare_v1_rhs, finalize_v1_and_compute_deff

    m = len(rx)
    if h is None:
        h = float(np.mean(np.linalg.norm(rx[rcells[:, 1]] - rx[rcells[:, 0]], axis=1)))
    ez = np.zeros(3); ez[ax] = 1.0
    nodes = np.vstack([rx, rx + h * ez])
    dof_map = np.concatenate([np.arange(m), np.arange(m)])
    quads = np.array([[a, b, m + b, m + a] for a, b in rcells], dtype=int)
    e3q = np.asarray(re3)

    Dhh, Dhe, Dee, Dhl, Dll, Dle = assemble_segment_indep(
        nodes, quads, rsub, e3q, D_by, G_by, np.asarray(k22_edge), cross, ax,
        kg_e=None, pen=0.0, dof_map=dof_map, shear=shear)
    Gc, Gl, Ge = assemble_constraint(nodes, quads, rsub, e3q, np.asarray(k22_edge),
                                     cross, ax, dof_map=dof_map, lam_space=lam_space)
    Dhh, Dhe, Dhl, Dll, Dle = [np.asarray(A) / h for A in (Dhh, Dhe, Dhl, Dll, Dle)]
    Dee = np.asarray(Dee) / h
    Gc, Gl, Ge = Gc / h, Gl / h, Ge / h

    M = Dhh.shape[0]; P = Gc.shape[0]
    # 5-DOF rigid kernel/constraints on the contour, embedded into 6 DOF (om3 rigid-free)
    C5, Psi5 = build_C_Psi(rx[:, cross], rcells, p=1)          # (4,5m), (5m,4)
    C6 = np.zeros((4, M)); Psi6 = np.zeros((M, 4))
    for n in range(m):
        C6[:, 6 * n:6 * n + 5] = C5[:, 5 * n:5 * n + 5]
        Psi6[6 * n:6 * n + 5, :] = Psi5[5 * n:5 * n + 5, :]
    Psi6[3::6, 3] *= -1.0                                      # validated-kernel om1 sign flip

    naug = M + P
    Dhh_a = np.zeros((naug, naug)); Dhh_a[:M, :M] = Dhh
    Dhh_a[:M, M:] = Gc.T; Dhh_a[M:, :M] = Gc
    Dhe_a = np.zeros((naug, 4)); Dhe_a[:M] = Dhe; Dhe_a[M:] = Ge
    Dhl_a = np.zeros((naug, naug)); Dhl_a[:M, :M] = Dhl; Dhl_a[M:, :M] = Gl
    Dll_a = np.zeros((naug, naug)); Dll_a[:M, :M] = Dll
    Dle_a = np.zeros((naug, 4)); Dle_a[:M] = Dle
    Psi_a = np.zeros((naug, 4)); Psi_a[:M] = Psi6
    Dc_a = np.zeros((naug, 4)); Dc_a[:M] = C6.T

    # full KKT: [Dhh_a, Dc_a; Dc_a^T, 0] -- factorized once for the V0 AND V1 solves
    from scipy.linalg import lu_factor, lu_solve
    A = np.zeros((naug + 4, naug + 4))
    A[:naug, :naug] = Dhh_a; A[:naug, naug:] = Dc_a; A[naug:, :naug] = Dc_a.T
    Alu = lu_factor(A)
    R0 = np.zeros((naug + 4, 4)); R0[:naug] = -Dhe_a
    V0 = lu_solve(Alu, R0)[:naug]
    Deff = Dee + V0.T @ Dhe_a
    bb, DhlV0, DhlTV0Dle, V0DllV0 = prepare_v1_rhs(
        jnp.array(V0), jnp.array(Dhl_a), jnp.array(Dll_a), jnp.array(Dle_a),
        jnp.array(Psi_a), jnp.array(Dc_a))
    R1 = np.zeros((naug + 4, 4)); R1[:naug] = np.asarray(bb)
    V1 = lu_solve(Alu, R1)[:naug]
    C6r, *_ = finalize_v1_and_compute_deff(
        jnp.array(V1), jnp.array(V0), jnp.array(Deff),
        V0DllV0, DhlV0, DhlTV0Dle, jnp.array(Psi_a), jnp.array(Dc_a))
    C6r = np.asarray(C6r)
    C6r = 0.5 * (C6r + C6r.T)
    if return_fields:
        return C6r, np.asarray(V0[:M]), np.asarray(V1[:M])
    return C6r


def main():
    from .segment_element import compute_k22
    from .segment_element_general import ring_general
    from .solve_segment_jax import _material_by_section
    print("BOUNDARY RING (prismatic cross-section SG): 6-DOF constrained element, shear-scheme")
    print("comparison vs the validated 5-DOF MITC ring and the 3-D solid (L ring, R=1.0, thin)\n")
    import io, contextlib
    from boundary_from_yaml import extract
    scratch = os.path.join(HERE, "out", "ring_indep_scratch")
    os.makedirs(scratch, exist_ok=True)
    for geom, res_sub, npzn in [("square", "taper_square", "taper_square_solid_%s.npz"),
                                ("circle", "taper_study", "taper_study_solid_%s.npz")]:
        for mat in ("iso", "m45"):
            tg = "thin_%s_aR070" % mat
            # extract FRESH from the correct geometry's mesh yaml into a geometry-tagged
            # scratch npz (tag-only npz names in shared res dirs are cross-geometry hazards)
            npz = os.path.join(scratch, "%s_%s.npz" % (geom, tg))
            with contextlib.redirect_stdout(io.StringIO()):
                extract(os.path.join(HERE, "out", res_sub, "meshes", "shell_%s.yaml" % tg), npz)
            b = np.load(npz, allow_pickle=True)
            sref = np.load(os.path.join(BENCH, npzn % mat), allow_pickle=True)
            So = 0.5 * (sref[tg + "_L"] + sref[tg + "_L"].T)
            ax = int(b["axis"]); cross = [j for j in range(3) if j != ax]
            rx = np.asarray(b["L_x"]); rc = np.asarray(b["L_cells"])
            rs = np.asarray(b["L_subdom"]); re3 = np.asarray(b["L_e3"])
            D_by, G_by = _material_by_section(json.loads(str(b["sections"])),
                                              json.loads(str(b["materials"])), center_ref=True)
            kr = compute_k22(rx[rc].mean(1), np.asarray(b["L_e2"]), re3, rc)
            rows = []
            C5m, _, _ = ring_general(rx, rc, rs, re3, D_by, G_by, kr, ax, cross, shear="mitc4_both")
            rows.append(("5-DOF elim + MITC (ref)", 0.5 * (C5m + C5m.T)))
            for sch in ("full", "mitc4_g23", "mitc4_both"):
                rows.append(("6-DOF constr, %s" % sch,
                             ring_indep(rx, rc, rs, re3, D_by, G_by, kr, ax, cross, shear=sch)))
            print("== %s thin %s : L ring vs solid ==" % (geom, mat))
            print("%-26s %7s %7s %7s %7s %7s %7s"
                  % ("element", "EA%", "GA2%", "GA3%", "GJ%", "EI2%", "EI3%"))
            for nm, S in rows:
                e = [100 * (S[i, i] - So[i, i]) / So[i, i] for i in range(6)]
                print("%-26s %+6.1f %+7.1f %+7.1f %+6.1f %+6.1f %+6.1f" % (nm, *e))
            print()
