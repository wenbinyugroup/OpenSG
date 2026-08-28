"""sg_stream.py -- `--solver iter 3` (stream): chunked element-by-element
CG that runs at ANY mesh size on ANY machine, by BOUNDING resident
memory instead of assuming it.

The iter-1 (cheb) pipeline holds the element tangent blocks J_uu
(E, 3N, 3N) as ONE device array and streams the EBE product over the
whole mesh each iteration; at 4.67M tet10 that array alone is 34 GB and
the jacfwd assembly transient is far larger.  This route re-plumbs the
same pipeline slab-by-slab, in one of TWO storage modes behind the same
solver:

  PACKED     the element blocks are computed once, slab by slab, and
             kept HOST-side as numpy in SYMMETRIC-HALF packing
             (E, n_ed (n_ed+1)/2) float64 -- E n_ed (n_ed+1)/2 8 bytes,
             ~17 GB for 4.67M tet10.  Fast per iteration: a matvec is
             an unpack + batched matmul + scatter-add.
  RECOMPUTE  NOTHING is stored.  Every matvec walks the slabs and
             RECOMPUTES that slab's element blocks with the same
             vmapped kernel, applies them, discards them.  Resident
             memory is O(one slab + the vectors) -- a few hundred MB at
             any mesh size -- at the cost of one element-kernel
             evaluation per iteration (~2-4x the per-iteration time).

Policy (automatic, printed in one line): the projected packed size is
compared against MemAvailable; PACKED is used when it fits inside half
of it, else RECOMPUTE.  Override the budget with the environment
variable OPENSG_ITER3_MEM_GB (the cli's `--mem GB` sets it): the number
is the GB the packed blocks may occupy, so `--mem 0` forces RECOMPUTE
and `--mem 1e6` forces PACKED.

Device: the element kernel and the slab matmul run on whatever jax
device exists -- GPU when jax sees one, CPU otherwise.  Only ONE slab
is ever resident on the device (the slab size is chosen from the
device's free memory on GPU, from host RAM on CPU), so a small-VRAM GPU
runs meshes far past its own memory.  The dof vectors stay host-side.

Kernels are the SAME jitted functions the direct and iter-1 routes call
(calculate_RHS_and_Ke_batch_periodic for the classical system,
plate_ladder_element_blocks for the RM ladder), so the digits are the
kernels' own.  Kernel nullspaces are removed by PROJECTION,
P = I - Q (Q^T Q)^-1 Q^T (the sparse_projected_cg construction): the
classical route projects the 3 rigid translations, the ladder its
<w> = 0 rows.  Both laws are gauge-invariant, so this matches the
direct route's first-node pin in exact arithmetic -- and unlike that
pin it leaves a genuinely SYMMETRIC operator, which CG requires (the
pinned operator has identity ROWS but stiffness COLUMNS: fine for a
factorization, fatal for a Krylov method).

Digits contract: identical element kernels + CG on stationary
functionals -> the law matches the direct factorization to ~tol^2 for
the stationary blocks (C_eff, A6) and ~tol for the ladder's H blocks.
Conditioning caveat: the preconditioner is the iter-1 recipe (node
block-Jacobi, optionally Chebyshev-wrapped), which is weak on very
low-density porous SGs -- see the module's stagnation guard, which
reports rather than hides it.

# ----------------------------------------------------------------------------
# ALL VARIABLES USED IN THIS MODULE  (axis suffixes: e element, n elem-node,
# q quad point, d spatial dim, s Voigt-6, H macro mode, T packed-half slot)
# ----------------------------------------------------------------------------
# inputs
#   x_end               (E, N, d) element-node coordinates
#   dphi_dxi_qnp, phi_qn, W_q     basis/quadrature tables of the arity
#   C_ess               (E, 6, 6) per-element stiffness
#   rc / periodic_cells (E, N) REDUCED master-node connectivity
#   n_unique            int, solve size (3 dofs / reduced node)
#   n_model, n_sg       macro model (2 plate / 3 solid) and SG dimension
#   omega               float, the ladder's in-plane SG measure
#   tol, maxiter        CG controls
# working
#   blocks_fn           (a, b) -> (b-a, n_ed, n_ed) slab element blocks
#   mode                "packed" | "recompute"; S the slab size
#   iu0, iu1, _full     symmetric-half packing / unpacking maps
#   Kp                  (E, T) float64 packed blocks (PACKED mode only)
#   dofs                (E, n_ed) int32 global dof ids 3 rc + c
#   plans               per-slab (a, b, uniq, lid) compact scatter plans
#   inv_blocks          (n_nodes, 3, 3) node block-Jacobi inverses
#   Qc, proj            constraint columns and the projector
#   Dhe / D_bar / omega (n_unique, H) case RHS, direct integral, measure
#   eig_max, theta      Chebyshev spectrum bound and shifts
# outputs
#   C_eff, V0_matrix, omega        stream_homo (the classical law)
#   lad dict (A6, G_msg, X, ev_min, Ustar_rel, V0, V11, V12,
#             V11bar, V12bar)      stream_plate_shear_ladder | None
# ----------------------------------------------------------------------------
"""
import os
import time

import numpy as np

import jax
import jax.numpy as jnp

from opensg_solid.sg_assembly import (calculate_RHS_and_Ke_batch_periodic,
                                      compute_homogenized_constants,
                                      plate_ladder_element_blocks)
from opensg_solid import sg_progress

_HOST_KERNEL_BUDGET = 4e9    # the assembly kernel's per-slab transient cap
_MV_BUDGET = 2.5e8           # unpacked-block transient per PACKED matvec


# ------------------------------------------------------------ machine probing
def _mem_available_bytes():
    """MemAvailable of /proc/meminfo, psutil as the portable fallback.

    In:  --
    Out: int bytes, or -1 when neither is readable."""
    try:
        for ln in open("/proc/meminfo"):
            if ln.startswith("MemAvailable"):
                return int(ln.split()[1]) * 1024
    except OSError:
        pass
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        return -1


def _device_free_bytes():
    """Free memory of the default jax device.

    In:  --
    Out: (device, int bytes) -- bytes -1 when the device reports no
         stats (every CPU backend, and GPU backends without the
         memory_stats API)."""
    try:
        d = jax.devices()[0]
    except Exception:
        return None, -1
    try:
        st = d.memory_stats() or {}
        lim = st.get("bytes_limit")
        use = st.get("bytes_in_use", 0)
        if lim:
            return d, int(lim) - int(use)
    except Exception:
        pass
    return d, -1


def _kernel_slab(Q, n_ed, E):
    """Elements per slab: the element kernel's AD transient scales
    ~ Q (n_ed)^2 doubles per element, so the slab is the memory budget
    divided by that -- the DEVICE's free memory when jax sits on a GPU
    (only one slab is ever resident there), host RAM otherwise.

    In:  Q int quad points; n_ed int dofs/element; E int elements
    Out: (S int elements/slab, note str for the console)."""
    dev, free = _device_free_bytes()
    plat = getattr(dev, "platform", "cpu")
    if plat != "cpu" and free > 0:
        budget = min(0.4 * free, _HOST_KERNEL_BUDGET)
        note = "%s, %.1f GB free" % (plat, free / 1e9)
    else:
        budget = _HOST_KERNEL_BUDGET
        note = str(plat)
    return max(1, min(E, int(budget / (Q * n_ed ** 2 * 8)))), note


def choose_mode(E, n_ed, label=""):
    """PACKED or RECOMPUTE, and say why in one line.

    Policy: PACKED when the projected packed size fits inside HALF of
    MemAvailable, else RECOMPUTE.  OPENSG_ITER3_MEM_GB overrides the
    budget outright (`--mem 0` forces RECOMPUTE).

    In:  E int elements; n_ed int dofs/element; label str
    Out: (mode str, packed_bytes int)."""
    T = n_ed * (n_ed + 1) // 2
    need = E * T * 8
    env = os.environ.get("OPENSG_ITER3_MEM_GB")
    if env is not None:
        budget = float(env) * 1e9
        why = "budget %.2f GB (OPENSG_ITER3_MEM_GB)" % (budget / 1e9)
    else:
        avail = _mem_available_bytes()
        budget = 0.5 * avail if avail > 0 else 0.0
        why = ("%.2f GB available" % (avail / 1e9) if avail > 0
               else "available RAM unknown")
    mode = "packed" if need <= budget else "recompute"
    print(" iter3%s: %s mode (packed blocks would need %.2f GB, %s)"
          % ((" " + label) if label else "", mode, need / 1e9, why))
    return mode, need


# ---------------------------------------------------------------- the operator
def _pack_maps(n_ed):
    """Packing maps of the symmetric half of an (n_ed, n_ed) block.

    In:  n_ed int, element dof count (3N)
    Out: (iu0, iu1) triu row/col indices (T,) each;
         full (n_ed*n_ed,) int32 -- the packed slot of every (i, j), so
         Kp[:, full].reshape(-1, n_ed, n_ed) unpacks."""
    iu0, iu1 = np.triu_indices(n_ed)
    slot = np.zeros((n_ed, n_ed), np.int32)
    slot[iu0, iu1] = np.arange(len(iu0), dtype=np.int32)
    ii, jj = np.meshgrid(np.arange(n_ed), np.arange(n_ed), indexing="ij")
    full = slot[np.minimum(ii, jj), np.maximum(ii, jj)].ravel()
    return iu0, iu1, full


class StreamKe:
    """The streamed EBE operator y = K z in either storage mode.

    PACKED holds the symmetric halves of every element block in host
    numpy and unpacks per matvec slab; RECOMPUTE holds nothing and
    calls blocks_fn again inside every matvec.  Both share the dof map,
    the compact scatter plans and the node-diagonal extraction, so the
    two modes are the same operator with different residency.

    In:  blocks_fn callable (a, b) -> (b-a, n_ed, n_ed) element blocks
         (jax or numpy); rc (E, N) reduced connectivity; n_unique int;
         S int slab size; mode "packed" | "recompute"; label str."""

    def __init__(self, blocks_fn, rc, n_unique, S, mode, label=""):
        rc = np.asarray(rc, np.int64)
        E, N = rc.shape
        self.blocks_fn = blocks_fn
        self.E, self.n_ed, self.n_unique = E, 3 * N, n_unique
        self.mode = mode
        self.T = self.n_ed * (self.n_ed + 1) // 2
        self.iu0, self.iu1, self._full = _pack_maps(self.n_ed)
        self.dofs = ((rc * 3)[:, :, None]
                     + np.arange(3)).reshape(E, self.n_ed).astype(np.int32)
        self.Kp = np.empty((E, self.T), np.float64) if mode == "packed" \
            else None
        # PACKED matvecs are unpack-bound, RECOMPUTE matvecs are
        # kernel-bound: each mode walks the slab size of its own cost
        S_mv = (max(1, int(_MV_BUDGET / (self.n_ed ** 2 * 8)))
                if mode == "packed" else S)
        self.plans = []
        for a in range(0, E, S_mv):
            b = min(a + S_mv, E)
            uniq, lid = np.unique(self.dofs[a:b].ravel(),
                                  return_inverse=True)
            self.plans.append((a, b, uniq.astype(np.int64),
                               lid.astype(np.int32)))
        self.S_build = S
        self.n_apply = 0
        self.label = label

    def build(self):
        """PACKED: fill Kp slab by slab (RECOMPUTE: nothing to do).

        In:  --
        Out: None."""
        if self.mode != "packed":
            return
        t0 = time.perf_counter()
        for a in range(0, self.E, self.S_build):
            b = min(a + self.S_build, self.E)
            J = np.asarray(self.blocks_fn(a, b))
            self.Kp[a:b] = J[:, self.iu0, self.iu1]
            del J
        print(" iter3%s pack: %d slabs stored (%.1f s)"
              % ((" " + self.label) if self.label else "",
                 -(-self.E // self.S_build), time.perf_counter() - t0))

    def _slab_blocks(self, a, b):
        """The (b-a, n_ed, n_ed) blocks of one slab, from store or
        kernel.

        In:  a, b int element range
        Out: array (b-a, n_ed, n_ed) -- numpy (packed) or jax
             (recompute, still on the device)."""
        if self.mode == "packed":
            return self.Kp[a:b].take(self._full, axis=1) \
                       .reshape(b - a, self.n_ed, self.n_ed)
        return self.blocks_fn(a, b)

    def matmul(self, Z):
        """The streamed multi-column EBE product.

        In:  Z (n_unique, H) host numpy
        Out: (n_unique, H) = K_assembled @ Z (no pin, no projection)."""
        H = Z.shape[1]
        Y = np.zeros_like(Z)
        for a, b, uniq, lid in self.plans:
            K = self._slab_blocks(a, b)
            zl = Z[self.dofs[a:b].ravel()].reshape(b - a, self.n_ed, H)
            if self.mode == "packed":
                yl = np.matmul(K, zl).reshape(-1, H)
            else:                       # keep the product on the device
                yl = np.asarray(jnp.einsum("eij,ejh->eih", K,
                                           jnp.asarray(zl))
                                ).reshape(-1, H)
            acc = np.empty((len(uniq), H))
            for c in range(H):
                acc[:, c] = np.bincount(lid, weights=yl[:, c],
                                        minlength=len(uniq))
            Y[uniq] += acc
            del K
        self.n_apply += 1
        return Y

    def diag_blocks(self):
        """Assembled node-diagonal 3x3 blocks (block-Jacobi setup); one
        streaming pass in either mode.

        In:  --
        Out: (n_unique//3, 3, 3) float64."""
        n_nodes = self.n_unique // 3
        N = self.n_ed // 3
        blocks = np.zeros((n_nodes, 3, 3))
        step = self.S_build if self.mode == "recompute" else None
        rng = (range(0, self.E, step) if step
               else [p[0] for p in self.plans])
        ends = ([min(a + step, self.E) for a in rng] if step
                else [p[1] for p in self.plans])
        for a, b in zip(rng, ends):
            K = np.asarray(self._slab_blocks(a, b)).reshape(
                b - a, N, 3, N, 3)
            bl = np.einsum("enpmq,nm->enpq", K, np.eye(N))
            nodes = (self.dofs[a:b, ::3] // 3).ravel().astype(np.int64)
            for p in range(3):
                for q in range(3):
                    blocks[:, p, q] += np.bincount(
                        nodes, weights=bl[:, :, p, q].ravel(),
                        minlength=n_nodes)
            del K, bl
        return blocks


# ------------------------------------------------------------- the CG machinery
def _inv_blocks(blocks):
    """In:  blocks (n_nodes, 3, 3)
    Out: (n_nodes, 3, 3) inverses of blocks + 1e-8 I (the
         compute_block_inv_diag regularization)."""
    return np.linalg.inv(blocks + 1e-8 * np.eye(3)[None])


def _block_apply(inv_blocks, X):
    """In:  inv_blocks (n_nodes, 3, 3); X (n, H)
    Out: (n, H) per-node 3x3 inverse-block multiply."""
    n, H = X.shape
    return np.matmul(inv_blocks, X.reshape(-1, 3, H)).reshape(n, H)


def _estimate_eig_max(A_op, M_op, n, num_iters=30, start=None):
    """Power-method bound of M^-1 A -- estimate_max_eigenvalue's recipe
    with a wider safety margin (1.15) and more iterations, because a
    Chebyshev interval whose top falls BELOW the true spectrum top
    makes the preconditioner indefinite and stalls CG.  `start`
    overrides the ones start: a PROJECTED operator must start outside
    its constraint span (ones IS the translation direction, and locks
    the iteration onto the harmless identity eigenvalue 1.0).

    In:  A_op/M_op callables on (n, 1); n int; num_iters int;
         start (n, 1) | None
    Out: float eig_max."""
    v = (np.ones((n, 1)) / np.sqrt(n) if start is None
         else start / (np.linalg.norm(start) + 1e-300))
    lam = 0.0
    for _ in range(num_iters):
        w = M_op(A_op(v))
        lam = float(v.ravel() @ w.ravel())
        v = w / (np.linalg.norm(w) + 1e-12)
    return lam * 1.15


def _make_cheb(A_op, M_blk, eig_max, eig_min, degree=4):
    """Chebyshev(degree) wrap of the block preconditioner --
    apply_chebyshev_precond in numpy, multi-column.

    In:  A_op/M_blk callables on (n, H); eig_max/eig_min floats;
         degree int (0 -> M_blk itself, the SPD-safe plain block
         Jacobi)
    Out: callable X -> approximate A^-1 X."""
    if not degree:
        return M_blk
    d = (eig_max + eig_min) / 2.0
    c = (eig_max - eig_min) / 2.0
    i = np.arange(1, degree + 1)
    theta = d + c * np.cos(np.pi * (2 * i - 1) / (2 * degree))

    def cheb(X):
        Z = np.zeros_like(X)
        for th in theta:
            Z = Z + M_blk(X - A_op(Z)) / th
        return Z
    return cheb


def _pcg(A_op, M_op, B, tol, maxiter, label=""):
    """Multi-column preconditioned CG: every column advances against
    the SAME streamed A applies (one slab pass serves all columns).
    A stagnation guard (no 10 % progress of the worst active column
    over 400 iterations) breaks out and SAYS SO rather than burning
    maxiter silently.

    In:  A_op/M_op callables on (n, H); B (n, H) right-hand sides;
         tol float relative-residual target; maxiter int; label str
    Out: (X (n, H), iters (H,), rel (H,), converged bool)."""
    t0 = time.perf_counter()
    n, H = B.shape
    X = np.zeros_like(B)
    R = B.copy()
    Z = M_op(R)
    P = Z.copy()
    rz = np.einsum("nh,nh->h", R, Z)
    bnorm = np.linalg.norm(B, axis=0)
    bnorm = np.where(bnorm == 0.0, 1.0, bnorm)
    iters = np.zeros(H, int)
    done = np.linalg.norm(R, axis=0) <= tol * bnorm
    best, k_best, k = np.inf, 0, 0
    # the sg_progress bar rides `worst` below -- this loop is ALREADY
    # host-resident and reads every column residual each iteration, so
    # the bar costs one log10 per iteration and nothing when dark
    bar, ltol = sg_progress.active(), np.log10(tol)
    while k < maxiter and not done.all():
        k += 1
        AP = A_op(P)
        pAp = np.einsum("nh,nh->h", P, AP)
        alpha = np.where(done | (pAp == 0.0), 0.0,
                         rz / np.where(pAp == 0.0, 1.0, pAp))
        X += alpha * P
        R -= alpha * AP
        rn = np.linalg.norm(R, axis=0)
        newly = (~done) & (rn <= tol * bnorm)
        iters[newly] = k
        done |= newly
        if done.all():
            break
        worst = float((rn / bnorm)[~done].max())
        if bar:
            sg_progress.solve(np.log10(max(worst, 1e-300)) / ltol)
        if worst < 0.9 * best:
            best, k_best = worst, k
        elif k - k_best > 400:
            print(" iter3 %s: STAGNATED at k=%d (worst rel r %.1e"
                  " unimproved since k=%d)" % (label, k, best, k_best))
            break
        if k % 250 == 0:
            print(" iter3 %s: k=%d  rel r %s" % (
                label, k, "/".join("%.1e" % v for v in rn / bnorm)))
        Z = M_op(R)
        rz_new = np.einsum("nh,nh->h", R, Z)
        beta = np.where(done, 0.0, rz_new / np.where(rz == 0.0, 1.0, rz))
        rz = np.where(done, rz, rz_new)
        P = Z + beta * P
    iters[~done] = k
    sg_progress.solve(1.0)
    rel = np.linalg.norm(R, axis=0) / bnorm
    print(" iter3 %s: %d cols, iters %s, final rel r %s%s  (%.1f s)"
          % (label, H, "/".join(str(i) for i in iters),
             "/".join("%.1e" % r for r in rel),
             "" if done.all() else "  [NOT CONVERGED at tol %.1e]" % tol,
             time.perf_counter() - t0))
    return X, iters, rel, bool(done.all())


def make_projected_solver(op, Qc, label="", degree=4):
    """The iter-3 constrained solve: CG on P K P + (I - P) with
    P = I - Qc (Qc^T Qc)^-1 Qc^T, node block-Jacobi + Chebyshev(4)
    preconditioned (the iter-1 recipe).  The classical route passes the
    3 TRANSLATION columns (the operator's exact kernel on a periodic
    SG), the ladder its <w> = 0 rows.

    In:  op StreamKe; Qc (n_unique, m) constraint columns; label str;
         degree int Chebyshev degree (0 = plain block-Jacobi, SPD-safe)
    Out: solve(B, tol, maxiter, label2) -> (X, iters, rel, converged),
         X satisfying Qc^T X = 0."""
    Gi = np.linalg.inv(Qc.T @ Qc)

    def proj(X):
        return X - Qc @ (Gi @ (Qc.T @ X))

    inv_b = _inv_blocks(op.diag_blocks())

    def A_op(X):
        PX = proj(X)
        return proj(op.matmul(PX)) + (X - PX)

    def M_blk(X):
        PX = proj(X)
        return proj(_block_apply(inv_b, PX)) + (X - PX)

    v0 = proj(np.random.default_rng(0).standard_normal((op.n_unique, 1)))
    eig_max = _estimate_eig_max(A_op, M_blk, op.n_unique, start=v0)
    M_op = _make_cheb(A_op, M_blk, eig_max, eig_max / 25.0, degree)
    print(" iter3%s spectrum: eig_max %.4e (block-Jacobi%s, %d-column"
          " projection)" % ((" " + label) if label else "", eig_max,
                            " + Chebyshev %d" % degree if degree else "",
                            Qc.shape[1]))

    def solve(B, tol, maxiter, label2=""):
        X, it, rel, ok = _pcg(A_op, M_op, proj(B), tol, maxiter,
                              label=label2)
        return proj(X), it, rel, ok
    return solve


# ------------------------------------------------------------- the build passes
def _padder(arr, S):
    """A slab slice padded to exactly S rows by REPEATING the last
    element (valid geometry; the padded rows are discarded).  A fixed
    slab shape means ONE XLA compile serves every slab.

    In:  arr (E, ...) numpy; S int slab size
    Out: callable (a, b) -> (S, ...) numpy."""
    def take(a, b):
        sl = arr[a:b]
        if len(sl) == S:
            return sl
        return np.concatenate([sl, np.repeat(sl[-1:], S - len(sl), 0)], 0)
    return take


def make_classical_ops(x_end, dphi_dxi_qnp, phi_qn, W_q, C_ess, rc,
                       n_unique, n_model, n_sg):
    """The classical fluctuation system, streamed: the element-block
    callable (either mode), the assembled case RHS and the operator.

    The RHS pass runs calculate_RHS_and_Ke_batch_periodic over a
    slab-LOCAL dof space (node 0 = dummy: it absorbs the kernel's
    rows-0:3 zeroing AND the tail-slab padding), so the global assembly
    is exact.  The element tangent depends only on (x, C) -- it is the
    jacfwd of a LINEAR residual -- so the recompute callable can pass a
    trivial connectivity and skip the dof bookkeeping entirely.

    In:  x_end (E, N, d); basis tables; C_ess (E, 6, 6); rc (E, N)
         reduced connectivity; n_unique int; n_model 2|3; n_sg int
    Out: (op StreamKe built, Dhe (n_unique, H) assembled case RHS)."""
    t0 = time.perf_counter()
    x_np = np.asarray(x_end)
    C_np = np.asarray(C_ess)
    rc_np = np.asarray(rc, np.int64)
    E, N, _ = x_np.shape
    n_ed = 3 * N
    Q = int(np.asarray(W_q).shape[0])
    S, dnote = _kernel_slab(Q, n_ed, E)
    H = 4 if n_model == 1 else 6
    mode, _ = choose_mode(E, n_ed)
    print(" iter3 device: %s, slab %d elements (E=%d, n_ed=%d, Q=%d)"
          % (dnote, S, E, n_ed, Q))
    px, pC = _padder(x_np, S), _padder(C_np, S)
    cl_pad = jnp.zeros((S, N), jnp.int32)
    u_pad = jnp.zeros(3)

    def blocks_fn(a, b):
        _, J = calculate_RHS_and_Ke_batch_periodic(
            jnp.asarray(px(a, b)), dphi_dxi_qnp, phi_qn, W_q,
            jnp.asarray(pC(a, b)), cl_pad, u_pad, n_model, n_sg)
        return J[:b - a]

    op = StreamKe(blocks_fn, rc_np, n_unique, S, mode)
    # the RHS pass: one walk with the real (slab-local) connectivity
    RHS_nodes = np.zeros((n_unique // 3, 3, H))
    u_zero = jnp.zeros(3 * (S * N + 1))
    for a in range(0, E, S):
        b = min(a + S, E)
        uniq, inv = np.unique(rc_np[a:b], return_inverse=True)
        cl = (inv.astype(np.int32) + 1).reshape(b - a, N)
        if b - a < S:                # padded elements -> all-dummy node 0
            cl = np.concatenate([cl, np.zeros((S - (b - a), N), np.int32)],
                                0)
        rhs_loc, J = calculate_RHS_and_Ke_batch_periodic(
            jnp.asarray(px(a, b)), dphi_dxi_qnp, phi_qn, W_q,
            jnp.asarray(pC(a, b)), jnp.asarray(cl), u_zero, n_model, n_sg)
        if mode == "packed":
            op.Kp[a:b] = np.asarray(J[:b - a])[:, op.iu0, op.iu1]
        np.add.at(RHS_nodes, uniq,
                  np.asarray(rhs_loc)[3:3 * (len(uniq) + 1)]
                  .reshape(-1, 3, H))
        del J, rhs_loc
    print(" iter3 assembly pass: %d slabs (%.1f s)"
          % (-(-E // S), time.perf_counter() - t0))
    return op, RHS_nodes.reshape(n_unique, H)


def _stream_D_bar_omega(x_np, dphi_dxi_qnp, phi_qn, W_q, C_np, n_model,
                        n_sg, S):
    """Slab-streamed D_bar (compute_homogenized_constants per slab) and
    the SG measure composed on the host: ptp measures from the global
    element-node extents (order-free min/max = digit-identical),
    integrated measures summed over slabs.

    In:  x_np (E, N, d); basis tables; C_np (E, 6, 6); n_model 2|3;
         n_sg int; S int slab size
    Out: (D_bar (H, H) np, omega float)."""
    E = x_np.shape[0]
    D_bar, om_int = None, 0.0
    for a in range(0, E, S):
        b = min(a + S, E)
        Db, om = compute_homogenized_constants(
            jnp.asarray(x_np[a:b]), dphi_dxi_qnp, phi_qn, W_q,
            jnp.asarray(C_np[a:b]), n_model, n_sg)
        D_bar = np.asarray(Db) if D_bar is None else D_bar + np.asarray(Db)
        om_int += float(om)          # only the additive branches use it
    lo, hi = x_np.min(axis=(0, 1)), x_np.max(axis=(0, 1))
    if n_model == 3:
        omega = (float((hi[0] - lo[0]) * (hi[1] - lo[1])
                       * (hi[2] - lo[2])) if n_sg == 3 else om_int)
    elif n_model == 2:
        omega = (float((hi[0] - lo[0]) * (hi[1] - lo[1])) if n_sg == 3
                 else float(hi[0] - lo[0]) if n_sg == 2 else 1.0)
    else:
        omega = 1.0
    return D_bar, omega


def stream_homo(x_end, dphi_dxi_qnp, phi_qn, W_q, C_ess, periodic_cells,
                n_unique, n_model, n_sg, tol=1e-8, maxiter=5000):
    """iter-3 classical homogenization: _homo_direct's system solved by
    the streamed EBE CG in whichever storage mode the machine affords.

    In:  x_end (E, N, d); basis tables; C_ess (E, 6, 6);
         periodic_cells (E, N); n_unique int; n_model 2|3; n_sg int;
         tol/maxiter CG controls
    Out: (C_eff (H, H), V0_matrix (n_unique, H) np, omega float)."""
    op, Dhe = make_classical_ops(x_end, dphi_dxi_qnp, phi_qn, W_q, C_ess,
                                 periodic_cells, n_unique, n_model, n_sg)
    T = np.zeros((n_unique, 3))
    T[0::3, 0] = T[1::3, 1] = T[2::3, 2] = 1.0
    solve = make_projected_solver(op, T)
    V0, _it, _rel, _ok = solve(-Dhe, tol, maxiter, label2="V0 classical")
    D1 = V0.T @ Dhe
    D_bar, omega = _stream_D_bar_omega(
        np.asarray(x_end), dphi_dxi_qnp, phi_qn, W_q, np.asarray(C_ess),
        n_model, n_sg, op.S_build)
    C_eff = (D_bar + D1) / omega
    print(" iter3: %d streamed applies (%s mode)"
          % (op.n_apply, op.mode))
    return C_eff, V0, omega


# ------------------------------------------------------------- the RM ladder
def _ladder_dense_pass(x_np, C_np, rc_np, dphi_hi, phi_hi, W_hi, n_sg,
                       omega, S, px, pC, dense, op=None):
    """One streaming pass filling the ladder's DENSE accumulators
    (D_he, D_l1e, D_l2e, D_ee, the <w> weights) and, in PACKED mode,
    the hh store.  All block contributions carry the ladder_blocks
    1/omega normalization; the <w> weights stay raw.

    In:  the slab inputs + padders; dense dict; op StreamKe | None
    Out: None (in-place)."""
    E, N = rc_np.shape
    n_ed = 3 * N
    for a in range(0, E, S):
        b = min(a + S, E)
        out = plate_ladder_element_blocks(jnp.asarray(px(a, b)), dphi_hi,
                                          phi_hi, W_hi,
                                          jnp.asarray(pC(a, b)), n_sg)
        hh, he, ee, l1e, l2e, wN = (np.asarray(out[0][:b - a]),
                                    np.asarray(out[1][:b - a]),
                                    np.asarray(out[2][:b - a]),
                                    np.asarray(out[8][:b - a]),
                                    np.asarray(out[9][:b - a]),
                                    np.asarray(out[10][:b - a]))
        dof = ((rc_np[a:b] * 3)[:, :, None]
               + np.arange(3)).reshape(b - a, n_ed)
        if op is not None and op.mode == "packed":
            op.Kp[a:b] = (hh / omega)[:, op.iu0, op.iu1]
        for key, Be in (("D_he", he), ("D_l1e", l1e), ("D_l2e", l2e)):
            np.add.at(dense[key], dof.ravel(), Be.reshape(-1, 6) / omega)
        dense["D_ee"] += ee.sum(axis=0) / omega
        np.add.at(dense["w_node"], rc_np[a:b].ravel(), wN.ravel())
        del out, hh


def _ladder_prod_pass(x_np, C_np, rc_np, dphi_hi, phi_hi, W_hi, n_sg,
                      omega, S, px, pC, V0, prods):
    """One streaming pass forming the ladder's block PRODUCTS with V0
    (D_hl1 V0, D_hl1^T V0, D_hl2 V0, D_hl2^T V0, D_l11 V0, D_l12 V0,
    D_l22 V0) -- so the hl/l blocks are never stored in either mode.

    In:  the slab inputs + padders; V0 (n, 6); prods dict of (n, 6)
    Out: None (in-place)."""
    E, N = rc_np.shape
    n_ed = 3 * N
    for a in range(0, E, S):
        b = min(a + S, E)
        out = plate_ladder_element_blocks(jnp.asarray(px(a, b)), dphi_hi,
                                          phi_hi, W_hi,
                                          jnp.asarray(pC(a, b)), n_sg)
        hl1, hl2, l11, l12, l22 = [np.asarray(out[k][:b - a])
                                   for k in (3, 4, 5, 6, 7)]
        dof = ((rc_np[a:b] * 3)[:, :, None]
               + np.arange(3)).reshape(b - a, n_ed)
        vl = V0[dof.ravel()].reshape(b - a, n_ed, 6)
        for key, B, tr in (("hl1v", hl1, False), ("hl1tv", hl1, True),
                           ("hl2v", hl2, False), ("hl2tv", hl2, True),
                           ("l11v", l11, False), ("l12v", l12, False),
                           ("l22v", l22, False)):
            Bm = np.swapaxes(B, 1, 2) if tr else B
            np.add.at(prods[key], dof.ravel(),
                      np.matmul(Bm, vl).reshape(-1, 6) / omega)
        del out


def stream_plate_shear_ladder(x_end, dphi_hi, phi_hi, W_hi, C_ess,
                              reduced_cells, n_unique, n_sg, omega,
                              tol=1e-10, maxiter=5000):
    """iter-3 RM first-order warping ladder: plate_shear_ladder's Eq.
    30-61 flow with the hh operator streamed (packed or recomputed),
    the hl/l blocks consumed on the fly, and the <w> = 0 KKT rows as a
    projection.  HOMOGENIZATION-ONLY: no V2 chains, no load columns
    (the iter-3 delivery scope).

    In:  plate_shear_ladder's single-batch signature (hi-quadrature
         tables); tol/maxiter CG controls
    Out: dict {A6, G_msg (None unless X SPD), X, ev_min, Ustar_rel,
         V0, V11, V12, V11bar, V12bar}, or None when a CG stage fails
         to converge (the deferral is printed; the classical law
         stands)."""
    from opensg_solid.sg_homo import _rm_ls_reduction   # call-time: no cycle
    t0 = time.perf_counter()
    x_np = np.asarray(x_end)
    C_np = np.asarray(C_ess)
    rc_np = np.asarray(reduced_cells, np.int64)
    E, N, _ = x_np.shape
    n_ed = 3 * N
    Q = int(np.asarray(W_hi).shape[0])
    S, dnote = _kernel_slab(Q, n_ed, E)
    mode, _ = choose_mode(E, n_ed, label="ladder")
    print(" iter3 ladder device: %s, slab %d elements (Q_hi=%d)"
          % (dnote, S, Q))
    px, pC = _padder(x_np, S), _padder(C_np, S)

    def blocks_fn(a, b):
        out = plate_ladder_element_blocks(jnp.asarray(px(a, b)), dphi_hi,
                                          phi_hi, W_hi,
                                          jnp.asarray(pC(a, b)), n_sg)
        return out[0][:b - a] / omega

    op = StreamKe(blocks_fn, rc_np, n_unique, S, mode, label="ladder")
    dense = {"D_he": np.zeros((n_unique, 6)),
             "D_l1e": np.zeros((n_unique, 6)),
             "D_l2e": np.zeros((n_unique, 6)),
             "D_ee": np.zeros((6, 6)),
             "w_node": np.zeros(n_unique // 3)}
    _ladder_dense_pass(x_np, C_np, rc_np, dphi_hi, phi_hi, W_hi, n_sg,
                       omega, S, px, pC, dense, op=op)
    print(" iter3 ladder assembly pass: done (%.1f s)"
          % (time.perf_counter() - t0))

    w_dof = np.repeat(dense["w_node"], 3)
    Qc = np.zeros((n_unique, 3))
    for j in range(3):
        Qc[j::3, j] = w_dof[j::3]
    solve = make_projected_solver(op, Qc, label="ladder")

    def solve_constrained(rhs, tag):
        X, _it, _rel, ok = solve(-np.asarray(rhs, float), tol, maxiter,
                                 label2="ladder %s" % tag)
        return X, ok

    V0, ok0 = solve_constrained(dense["D_he"], "V0 (6 col)")
    if not ok0:
        print(" iter3 ladder DEFERRED: the V0 stage did not converge --"
              " the G ladder is skipped; the classical law stands")
        return None
    A6 = dense["D_ee"] + V0.T @ dense["D_he"]

    prods = {k: np.zeros((n_unique, 6)) for k in
             ("hl1v", "hl1tv", "hl2v", "hl2tv", "l11v", "l12v", "l22v")}
    _ladder_prod_pass(x_np, C_np, rc_np, dphi_hi, phi_hi, W_hi, n_sg,
                      omega, S, px, pC, V0, prods)
    D1bar = (prods["hl1v"] - prods["hl1tv"]) - dense["D_l1e"]
    D2bar = (prods["hl2v"] - prods["hl2tv"]) - dense["D_l2e"]
    V1, ok1 = solve_constrained(np.concatenate([D1bar, D2bar], axis=1),
                                "V1 (12 col)")
    if not ok1:
        print(" iter3 ladder DEFERRED: the V1 stage did not converge --"
              " the G ladder is skipped; the classical law stands")
        return None
    V11, V12 = V1[:, :6], V1[:, 6:]

    H11 = V0.T @ prods["l11v"] + D1bar.T @ V11
    H12 = V0.T @ prods["l12v"] + 0.5 * (D1bar.T @ V12 + V11.T @ D2bar)
    H22 = V0.T @ prods["l22v"] + D2bar.T @ V12
    kernel = np.zeros((n_unique, 3))
    kernel[0::3, 0] = kernel[1::3, 1] = kernel[2::3, 2] = 1.0
    S1 = (kernel.T @ D1bar)[:2]
    S2 = (kernel.T @ D2bar)[:2]
    G, X, ev_min, Ustar_rel, c1, c2 = _rm_ls_reduction(A6, H11, H12,
                                                       H22, S1, S2)
    c1_ks = np.zeros((3, 6)); c1_ks[:2] = c1
    c2_ks = np.zeros((3, 6)); c2_ks[:2] = c2
    print(" iter3 ladder: done (%.1f s, %d streamed applies, %s mode)"
          % (time.perf_counter() - t0, op.n_apply, op.mode))
    return {"A6": A6, "G_msg": (G if ev_min > 0 else None), "X": X,
            "ev_min": ev_min, "Ustar_rel": float(Ustar_rel),
            "V0": V0, "V11": V11, "V12": V12,
            "V11bar": V11 + kernel @ c1_ks,
            "V12bar": V12 + kernel @ c2_ks}
