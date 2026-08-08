"""Localize the negative C44 of the PERIODIC shell ring.

Rebuilds exactly what ring_solid(periodic=True) assembles for the square tube
and asks, in order:

  1. is Dhh positive semi-definite after the periodic dof_map merge?
  2. are the three global rigid translations still exact zero-energy modes?
     (they must be: w1,w2,w3 are GLOBAL components in these operators)
  3. what is the true null-space dimension of Dhh, and does the kernel block
     C6 that the KKT imposes match it?
  4. is the KKT matrix nonsingular, and does D_eff computed through the KKT
     agree with D_eff from a projected pseudo-inverse (which needs no kernel
     guess)?  A mismatch means the KKT side conditions are the bug, not the
     periodic assembly.
  5. same four checks WITHOUT periodicity, as the control.

Run (from this folder):  python diag_periodic_shell.py
"""
import numpy as np
import yaml as _yaml

from opensg_shell.sg_mesh import load_ring_ref
from opensg_shell.sg_assembly import (assemble_segment_indep,
                                        assemble_constraint, NDOF6)
from opensg_shell.sg_homo import assemble_solid_macro
from opensg_shell.fe_jax.msg_rm_timo import build_C_Psi
from opensg_shell.sg_periodicity import periodic_node_map

############### User Input #################################
YAML = "square_tube_shell.yaml"     # written by square_tube_periodic.py
A_cell = 1.0
############################################################

R = load_ring_ref(YAML, "center")
rx, rcells = R["rx"], R["cells"]
ax, cross = R["ax"], R["cross"]
G_by = list(R["G_by"])
d_sh = _yaml.safe_load(open(YAML))
from opensg_solid.rm_plate_1D.msg_rm_plate import rm_plate_msg
from opensg_shell.sg_materials import material_db_from_yaml
_mdb = material_db_from_yaml(d_sh["materials"])
for si, sec in enumerate(d_sh["sections"]):
    _pl = [[str(p[0]), float(p[1]), float(p[2])] for p in sec["layup"]]
    _rr = rm_plate_msg([p[1] for p in _pl], [p[2] for p in _pl],
                       [p[0] for p in _pl], _mdb, fraction=0.5)
    if _rr["G_msg"] is not None:
        G_by[si] = np.asarray(_rr["G_msg"])

m = len(rx)
h = float(np.mean(np.linalg.norm(rx[rcells[:, 1]] - rx[rcells[:, 0]], axis=1)))
ez = np.zeros(3); ez[ax] = 1.0
nodes_st = np.vstack([rx, rx + h*ez])
quads = np.array([[a, b, m+b, m+a] for a, b in rcells], int)
e3q = np.asarray(R["re3"])


def build(periodic):
    if periodic:
        nm, nun = periodic_node_map(rx[:, cross], n_model=3)
    else:
        nm, nun = np.arange(m), m
    dof_map = np.concatenate([nm, nm])
    Dhh, _, _, _, _, _ = assemble_segment_indep(
        nodes_st, quads, R["rsub"], e3q, R["D_by"], G_by, np.asarray(R["k22"]),
        cross, ax, kg_e=None, pen=0.0, dof_map=dof_map, shear="mitc4_g23")
    Gc, _, _ = assemble_constraint(nodes_st, quads, R["rsub"], e3q,
                                   np.asarray(R["k22"]), cross, ax,
                                   dof_map=dof_map, lam_space="elem")
    Dhe6, Dee6 = assemble_solid_macro(nodes_st, quads, R["rsub"], e3q,
                                      R["D_by"], G_by, cross, ax,
                                      dof_map=dof_map, shear="mitc4_g23")
    return (np.asarray(Dhh)/h, Gc/h, Dhe6/h, Dee6/h, nm, nun)


def report(tag, periodic):
    Dhh, Gc, Dhe6, Dee6, nm, nun = build(periodic)
    M = Dhh.shape[0]; P = Gc.shape[0]
    print("\n=== %s ===  nodes %d -> %d, M=%d dofs, %d multiplier rows"
          % (tag, m, nun, M, P))

    # 1. PSD?
    ev = np.linalg.eigvalsh(0.5*(Dhh + Dhh.T))
    sc = np.max(np.abs(ev))
    print("1. Dhh eigenvalues: min %.3e  (min/max = %.2e)" % (ev.min(), ev.min()/sc))

    # 2. global rigid translations must be zero-energy
    for k, nmk in enumerate(("w1", "w2", "w3")):
        v = np.zeros(M)
        v[k::NDOF6] = 1.0
        print("   translation %s: v^T Dhh v / |Dhh| = %.3e"
              % (nmk, float(v @ Dhh @ v)/sc))

    # 3. true null space vs the kernel block the KKT imposes
    tolv = 1e-9*sc
    nnull = int(np.sum(np.abs(ev) < tolv))
    C5, _ = build_C_Psi(rx[:, cross], rcells, p=1)
    nk = 3 if periodic else 4
    C6 = np.zeros((nk, M))
    for n in range(m):
        s = NDOF6*nm[n]
        C6[:, s:s+5] += C5[:nk, 5*n:5*n+5]
    print("3. dim null(Dhh) = %d ; KKT imposes %d kernel rows (rank %d)"
          % (nnull, nk, np.linalg.matrix_rank(C6, tol=1e-8*np.abs(C6).max())))

    # null space of the CONSTRAINED operator [Dhh; Gc] -- what actually needs pinning
    stack = np.vstack([Dhh, Gc])
    sv = np.linalg.svd(stack, compute_uv=False)
    nnull_c = int(np.sum(sv < 1e-9*sv.max()))
    print("   dim null([Dhh; Gc]) = %d  <- number of modes the kernel MUST pin"
          % nnull_c)

    # 4. KKT vs projected pseudo-inverse
    naug = M + P
    A = np.zeros((naug+nk, naug+nk))
    A[:M, :M] = Dhh; A[:M, M:naug] = Gc.T; A[M:naug, :M] = Gc
    A[:M, naug:] = C6.T; A[naug:, :M] = C6
    R0 = np.zeros((naug+nk, 6)); R0[:M] = -Dhe6
    condA = np.linalg.cond(A)
    V0 = np.linalg.solve(A, R0)[:naug]
    D_kkt = Dee6 + V0[:M].T @ Dhe6
    D_kkt = 0.5*(D_kkt + D_kkt.T)

    # projected pinv: solve on the range of [Dhh; Gc] with no kernel guess
    KK = np.zeros((M+P, M+P))
    KK[:M, :M] = Dhh; KK[:M, M:] = Gc.T; KK[M:, :M] = Gc
    rhs = np.zeros((M+P, 6)); rhs[:M] = -Dhe6
    V0p = np.linalg.pinv(KK, rcond=1e-10) @ rhs
    D_pinv = Dee6 + V0p[:M].T @ Dhe6
    D_pinv = 0.5*(D_pinv + D_pinv.T)

    print("4. cond(KKT) = %.3e" % condA)
    print("   C44 : KKT %.6e   pinv %.6e" % (D_kkt[3, 3]/A_cell, D_pinv[3, 3]/A_cell))
    print("   min eig(D_eff)/A: KKT %.4e   pinv %.4e"
          % (np.linalg.eigvalsh(D_kkt).min()/A_cell,
             np.linalg.eigvalsh(D_pinv).min()/A_cell))
    print("   max|KKT - pinv| / max|pinv| = %.3e"
          % (np.max(np.abs(D_kkt - D_pinv))/np.max(np.abs(D_pinv))))
    return D_kkt/A_cell, D_pinv/A_cell


Dk_f, Dp_f = report("FREE (control)", False)
Dk_p, Dp_p = report("PERIODIC", True)

print("\nperiodic C3D via projected pinv (Pa):")
for i in range(6):
    print("  " + " ".join("%12.4e" % Dp_p[i, j] for j in range(6)))
