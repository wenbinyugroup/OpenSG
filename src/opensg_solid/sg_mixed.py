"""sg_mixed.py -- MIXED-element (multi-type) FEA support for the general
SG engine: the per-element-type BATCHING layer, kept out of sg_homo so
the core MSG math stays single-purpose.

The doctrine is fe_jax's ElementBatch ("a map can only be created for a
single type of cell, so different cell types need to be organized into
separate batches") and the shell_sg3d tri+quad homogenizer: split the
cells by arity, give every batch its own basis/quadrature tables, run
the per-element kernels batch-by-batch, and SCATTER-ADD every assembled
object into the same global system -- the dof space (nodes x 3, reduced
by the periodic node map, which is built from the POINTS alone and is
therefore identical across batches) is shared by construction.

What is additive over batches: the K COO stream (csr_matrix sums
duplicate (i, j) entries), the Dhe case RHS, D_bar, all six ladder csr
blocks, the ladder dense scatters (D_he/D_l1e/D_l2e), D_ee and the
<w>-weights.  What stays global and single: the pin/Dirichlet unit
rows, the factorizations and solves, and omega (ptp measures over ALL
nodes; integrated measures summed).

sg_homo calls in here for a mixed mesh and stays on its own
single-batch path otherwise; sg_dehom loops the per-batch lists this
module leaves in the result dict.  Supported: plate/solid macro models
with solver='direct' (the beam KKT route and the CG/EBE operators are
single-batch by construction).

# ----------------------------------------------------------------------------
# ALL VARIABLES USED IN THIS MODULE  (suffix _b = one element-type batch)
# ----------------------------------------------------------------------------
# inputs
#   sc, points, n_sg        the loaded SG dict, its (V, n_sg) nodes
#   n_model, bdofs          macro model; aperiodic Dirichlet dofs
#   u_0_g, unique_dofs, n_unique   the shared reduced dof space
#   x/dphi/phi/W/C/cells lists     per-batch ladder assembly inputs
# working
#   lens, idx               cells arity vector; a batch's element rows
#   batches                 [{nn, idx, cells, phi_qn, dphi_dxi_qnp, W_q,
#                             x_end, C_ess?, periodic_cells?}]
#   Dhe_g, rows/cols/data   the accumulated RHS and COO stream
#   D_bar, omega_int        accumulated direct integral / measure
# outputs
#   fe_tables()             (phi, dphi, W) of one arity
#   build_batches()         the batch dicts (basix-ordered cells)
#   attach_periodic_maps()  per-batch reduced connectivity + dof_map
#   homo_direct_batched()   (C_eff, V0, omega) -- _homo_direct, batched
#   ladder_blocks()         the assembled global ladder objects
# ----------------------------------------------------------------------------
"""
import numpy as np
from scipy.sparse import csr_matrix

import jax.numpy as jnp

from fe_jax.basis_quadrature import (ElementFamily, FiniteElementType,
                                     LagrangeVariant, QuadratureType,
                                     eval_basis_and_derivatives,
                                     get_quadrature)
from fe_jax.setup import (mesh_to_jax,
                          mesh_to_periodic_sparse_assembly_map)

from opensg_solid.sg_assembly import (_sparse_direct_solve,
                                      calculate_RHS_and_Ke_batch_periodic,
                                      compute_homogenized_constants,
                                      plate_ladder_element_blocks)
from opensg_solid.sg_mesh import _cell_basis

# enum member names differ across fe_jax vintages (default vs Default)
_QT = getattr(QuadratureType, "default", None) or getattr(
    QuadratureType, "Default")


def as_batches(v):
    """A per-batch LIST view of one r[...] entry (arrays wrap to [v])."""
    return v if isinstance(v, (list, tuple)) else [v]


def fe_tables(n_sg, nn, hi=False):
    """The (phi, dphi, W) basis/quadrature set of ONE element arity --
    the same FiniteElementType construction sg_homo always used, so a
    single-type run through here is digit-identical.

    In:  n_sg int SG dimension; nn int nodes/element; hi bool -- the
         ladder's 2p+2 rule instead of the classical set
    Out: (phi_qn (Q, N), dphi_dxi_qnp (Q, N, p), W_q (Q,))."""
    ctype, bdeg = _cell_basis(n_sg, nn)
    fe = FiniteElementType(
        cell_type=ctype, family=ElementFamily.P, basis_degree=bdeg,
        lagrange_variant=LagrangeVariant.equispaced,
        quadrature_type=_QT,
        quadrature_degree=(2 * bdeg + 2) if hi else
                          (6 if bdeg > 1 else 2))
    xi_qp, W_q = get_quadrature(fe_type=fe)
    phi_qn, dphi_dxi_qnp = eval_basis_and_derivatives(fe_type=fe,
                                                      xi_qp=xi_qp)
    return phi_qn, dphi_dxi_qnp, W_q


def build_batches(sc, points, n_sg):
    """Split a (possibly mixed) SG into per-element-type batches over
    the shared node space, each with basix-ordered cells, its own basis
    tables and element coordinates.  Element order within a batch
    follows the global mesh order, so mat_id/C_ess split by `idx`.

    In:  sc dict (load_sg_input); points (V, n_sg) jnp; n_sg int
    Out: list of dicts {nn, idx (E_b,), cells (E_b, N_b) jnp,
         phi_qn, dphi_dxi_qnp, W_q, x_end (E_b, N_b, n_sg)}."""
    from opensg_solid.sg_homo import _to_basix_order   # call-time: no cycle
    lens = np.array([len(c) for c in sc["cells"]])
    out = []
    for nn_b in sorted(set(lens.tolist())):
        idx = np.where(lens == nn_b)[0]
        cells_b = jnp.array(np.array([sc["cells"][k] for k in idx],
                                     np.uint64))
        cells_b = _to_basix_order(cells_b, n_sg, int(nn_b))
        phi_b, dphi_b, W_b = fe_tables(n_sg, int(nn_b))
        out.append({"nn": int(nn_b), "idx": idx, "cells": cells_b,
                    "phi_qn": phi_b, "dphi_dxi_qnp": dphi_b, "W_q": W_b,
                    "x_end": mesh_to_jax(vertices=points,
                                         cells=cells_b)})
    return out


def attach_periodic_maps(batches, V, points, n_model, boundary):
    """Per-batch reduced connectivity over the ONE shared dof space.

    The periodic node reduction inside
    mesh_to_periodic_sparse_assembly_map is built from the points alone
    (cells only route the reduced ids), so per-batch calls agree on the
    reduced numbering; aperiodic keeps the raw cells and the identity
    dof map.

    In:  batches (build_batches); V int nodes; points (V, n_sg);
         n_model int; boundary 'periodic' | anything else (aperiodic)
    Out: dof_map_np (V*3,) -- each batch gains 'periodic_cells'."""
    if boundary == "periodic":
        dof_map_np = None
        for b in batches:
            b["periodic_cells"], dof_map_np = \
                mesh_to_periodic_sparse_assembly_map(
                    V, b["cells"], points, n_model, atol=1e-6)
    else:
        dof_map_np = np.arange(V * 3)
        for b in batches:
            b["periodic_cells"] = b["cells"]
    return dof_map_np


def homo_direct_batched(batches, u_0_g, unique_dofs, n_unique, points,
                        n_model, n_sg, bdofs=None):
    """sg_homo._homo_direct for a MIXED SG: per-batch element kernels,
    one shared global system, one factorization.

    In:  batches -- build_batches dicts + 'C_ess' (E_b, 6, 6) and
         'periodic_cells' (attach_periodic_maps); u_0_g (V*3,) zero
         seed; unique_dofs/n_unique the shared reduced dof space;
         points (V, n_sg) ALL SG nodes (the global ptp measures);
         n_model 2 plate / 3 solid; n_sg int; bdofs (n_b,) aperiodic
         Dirichlet dofs or None
    Out: C_eff (H, H); V0_matrix (n_unique, H); omega float."""
    H = 4 if n_model == 1 else 6
    Dhe_g = np.zeros((n_unique, H))
    rows_l, cols_l, data_l = [], [], []
    D_bar = None
    omega_int = 0.0
    for b in batches:
        Dhe_b, J_euu = calculate_RHS_and_Ke_batch_periodic(
            b["x_end"], b["dphi_dxi_qnp"], b["phi_qn"], b["W_q"],
            b["C_ess"], b["periodic_cells"], u_0_g[unique_dofs],
            n_model, n_sg)
        Dhe_g += np.asarray(Dhe_b)
        E_b, n_ed, _ = J_euu.shape
        N_b = n_ed // 3
        dof_map = ((np.asarray(b["periodic_cells"], dtype=np.int64) * 3)
                   .reshape(E_b, N_b, 1)
                   + np.arange(3)).reshape(E_b, n_ed).astype(np.int32)
        rows_l.append(np.repeat(dof_map, n_ed, axis=1).ravel())
        cols_l.append(np.tile(dof_map, (1, n_ed)).ravel())
        data_l.append(np.asarray(J_euu).ravel())
        Db_b, om_b = compute_homogenized_constants(
            b["x_end"], b["dphi_dxi_qnp"], b["phi_qn"], b["W_q"],
            b["C_ess"], n_model, n_sg)
        D_bar = Db_b if D_bar is None else D_bar + Db_b
        omega_int += float(om_b)        # only the additive branch uses it

    rows = np.concatenate(rows_l)
    cols = np.concatenate(cols_l)
    data = np.concatenate(data_l)
    if bdofs is None:
        keep = rows >= 3                # pinned rows 0:3 -> unit diagonal
        pin = np.arange(3, dtype=np.int32)
    else:
        pinmask = np.zeros(n_unique, bool)
        pinmask[np.asarray(bdofs, np.int64)] = True
        keep = ~pinmask[rows]
        pin = np.where(pinmask)[0].astype(np.int32)
    rows = np.concatenate([rows[keep], pin])
    cols = np.concatenate([cols[keep], pin])
    data = np.concatenate([data[keep], np.ones(len(pin))])
    A_csr = csr_matrix((data, (rows, cols)), shape=(n_unique, n_unique))
    RHS = -Dhe_g
    RHS[0:3, :] = 0.0                   # re-zero after the += over batches
    if bdofs is not None:
        RHS[pin] = 0.0
    V0_matrix = jnp.asarray(_sparse_direct_solve(A_csr, RHS, sym=True))
    D1 = jnp.einsum('ni,nj->ij', V0_matrix, jnp.asarray(Dhe_g))

    # the global SG measure -- the single-batch expressions over ALL nodes
    pts = np.asarray(points)
    if n_model == 3:
        omega = (float(np.ptp(pts[:, 0]) * np.ptp(pts[:, 1])
                       * np.ptp(pts[:, 2])) if n_sg == 3 else omega_int)
    elif n_model == 2:
        omega = (float(np.ptp(pts[:, 0]) * np.ptp(pts[:, 1]))
                 if n_sg == 3 else
                 float(np.ptp(pts[:, 0])) if n_sg == 2 else 1.0)
    else:
        omega = 1.0
    C_eff = (D_bar + D1) / omega
    return C_eff, V0_matrix, omega


def ladder_blocks(x_end, dphi_hi, phi_hi, W_hi, C_ess, reduced_cells,
                  n_unique, n_sg, omega):
    """The ASSEMBLED global objects of the RM shear ladder, batched:
    every per-element argument may be an array (one batch) or a
    per-batch list; each batch's element blocks scatter-add into the
    same csr/dense globals, so plate_shear_ladder's math never sees the
    batching.

    In:  x_end/dphi_hi/phi_hi/W_hi/C_ess/reduced_cells -- arrays or
         per-batch lists (the ladder's 2p+2 tables); n_unique int;
         n_sg int; omega float in-plane SG measure
    Out: dict {D_hh, D_hl1, D_hl2, D_l11, D_l12, D_l22 (csr),
         D_he, D_l1e, D_l2e (n_unique, 6), D_ee (6, 6),
         w_dof (n_unique,)}."""
    D_hh = D_hl1 = D_hl2 = D_l11 = D_l12 = D_l22 = None
    D_he = np.zeros((n_unique, 6))
    D_l1e = np.zeros((n_unique, 6))
    D_l2e = np.zeros((n_unique, 6))
    D_ee = np.zeros((6, 6))
    w_node = np.zeros(n_unique // 3)
    _acc = lambda A, B: B if A is None else A + B          # noqa: E731
    for x_b, dp_b, ph_b, W_b, C_b, rc_b in zip(
            as_batches(x_end), as_batches(dphi_hi), as_batches(phi_hi),
            as_batches(W_hi), as_batches(C_ess),
            as_batches(reduced_cells)):
        out = plate_ladder_element_blocks(x_b, dp_b, ph_b, W_b,
                                          jnp.asarray(C_b), n_sg)
        (hh_b, he_b, ee_b, hl1_b, hl2_b, l11_b, l12_b, l22_b,
         l1e_b, l2e_b, wN_b) = [np.asarray(o) for o in out]

        E_elem, n_ed = he_b.shape[0], he_b.shape[1]
        N_nodes = n_ed // 3
        dof_map = ((np.asarray(rc_b, dtype=np.int64) * 3)
                   .reshape(E_elem, N_nodes, 1)
                   + np.arange(3)).reshape(E_elem, n_ed)
        rows = np.repeat(dof_map, n_ed, axis=1).ravel()
        cols = np.tile(dof_map, (1, n_ed)).ravel()

        # ONE sparsity analysis per batch for its six ladder matrices:
        # they share the (rows, cols) pattern, so the per-call lexsort +
        # dedup inside csr_matrix((data, (rows, cols))) -- ~1 s of a
        # 19k-tri3 run -- is paid once and each block reduces to a
        # bincount fill.  uniq is sorted by row * n + col = exactly the
        # canonical csr order.
        key = rows * np.int64(n_unique) + cols
        uniq, inv = np.unique(key, return_inverse=True)
        u_indices = (uniq % n_unique).astype(np.int32)
        u_rows = (uniq // n_unique).astype(np.int32)
        u_indptr = np.searchsorted(u_rows, np.arange(n_unique + 1),
                                   side="left").astype(np.int32)

        def nn(Be):
            data = np.bincount(inv, weights=Be.ravel() / omega,
                               minlength=len(uniq))
            return csr_matrix((data, u_indices, u_indptr),
                              shape=(n_unique, n_unique))

        def ns_add(M, Be):
            np.add.at(M, dof_map.ravel(), Be.reshape(-1, 6) / omega)

        D_hh = _acc(D_hh, nn(hh_b))
        D_hl1 = _acc(D_hl1, nn(hl1_b))
        D_hl2 = _acc(D_hl2, nn(hl2_b))
        D_l11 = _acc(D_l11, nn(l11_b))
        D_l12 = _acc(D_l12, nn(l12_b))
        D_l22 = _acc(D_l22, nn(l22_b))
        ns_add(D_he, he_b)
        ns_add(D_l1e, l1e_b)
        ns_add(D_l2e, l2e_b)
        D_ee += ee_b.sum(axis=0) / omega
        np.add.at(w_node, np.asarray(rc_b, dtype=np.int64).ravel(),
                  wN_b.ravel())
    return {"D_hh": D_hh, "D_hl1": D_hl1, "D_hl2": D_hl2,
            "D_l11": D_l11, "D_l12": D_l12, "D_l22": D_l22,
            "D_he": D_he, "D_l1e": D_l1e, "D_l2e": D_l2e, "D_ee": D_ee,
            "w_dof": np.repeat(w_node, 3)}
