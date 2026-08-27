"""sg_amg.py -- iter 2 of the --solver menu: smoothed-aggregation
AMG-preconditioned CG for the fluctuation solves of the SSDM
plate/solid routes (sg_homo dispatch solver="amg").

Split build/apply: pyamg constructs the hierarchy ONCE on the CPU from
the assembled pinned CSR (sg_assembly.assemble_rhs_and_pinned_csr,
sym=True -- the exact system _homo_direct factorizes, slab-streamed by
default), then every level's operators
move to jax arrays on the DEFAULT DEVICE (GPU when jax sees one, CPU
otherwise) and the symmetric Chebyshev(3)-smoothed V-cycle + the outer
CG (vmapped over the RHS columns) run entirely in jax, FP64.  ONE hierarchy serves every column: the V0
macro columns and, for the shear-refined plate, the whole warping
ladder (make_constrained_solver replaces plate_shear_ladder's KKT
factorization -- gauge note there).  Memory ~ the assembled CSR + the
coarse levels (~1.6x): no factorization fill-in, so it runs where
direct dies; on high-contrast SGs the V-cycle replaces the cheb-CG
conditioning wall with near-mesh-independent iteration counts.

# ----------------------------------------------------------------------------
# ALL VARIABLES USED IN THIS MODULE
# ----------------------------------------------------------------------------
# inputs (amg_homo mirrors _homo_direct; points/cells feed the B modes)
#   x_end, u_0_g, dphi_dxi_qnp, phi_qn, W_q, C_ess, periodic_cells,
#   unique_dofs, n_unique, n_model, n_sg     see sg_homo._homo_direct
#   points (V, n_sg), cells (E, N)           the mesh (near-nullspace)
#   rtol                CG relative-residual tolerance (law error is
#                       SECOND order in it: 1e-8 -> ~1e-16)
# working
#   A_csr, pin          pinned SPD csr + pinned dof ids (shared helper)
#   B                   (n_unique, m) rigid-mode candidates, m in 3..6
#   ml                  the pyamg SA hierarchy (CPU, dropped after copy)
#   levels              tuple of per-level dicts, jnp on-device:
#                       d/c/r csr triplets of A_l, Di = 1/diag(A_l),
#                       th Chebyshev nodes of [rho/30, 1.1 rho],
#                       Pd/Pc/Pr triplets of P_l
#   coarse              {"Ainv": (nc, nc)} dense coarsest inverse
#   fine                {d, c, r} the TRUE outer operator (level-0 A
#                       for V0; the pinned hi-quadrature D_hh for the
#                       ladder -- the hierarchy only preconditions)
#   ctx                 {"levels", "coarse", "pin", "rtol"} -- the
#                       handle plate_shear_ladder reuses
# outputs
#   C_eff, V0_matrix, omega                  as _homo_direct
#   X, iters, worst     per-column solutions, CG counts, worst relres
# ----------------------------------------------------------------------------
"""
import os
import time
from functools import partial

import numpy as np

import jax
import jax.numpy as jnp
from scipy.sparse import diags

from opensg_solid.sg_assembly import (assemble_rhs_and_pinned_csr,
                                      compute_homogenized_constants)
from opensg_solid import sg_progress

# env-overridable stopping pair -- a tolerance certificate is one rerun
# with OPENSG_AMG_RTOL tightened 100x: the law must hold its digits
_RTOL = float(os.environ.get("OPENSG_AMG_RTOL", 1e-8))
_MAXITER = int(os.environ.get("OPENSG_AMG_MAXITER", 500))
_CHEB_DEG = 3      # Chebyshev smoothing degree each side of the V-cycle
_CHEB_LO = 1/30.   # smoothing interval [rho/30, 1.1 rho] (pyamg default)
_CHEB_HI = 1.1
_MAX_COARSE = 200  # dense-inverse size of the coarsest level
_COL_BYTES = 4e9   # RHS-block transient budget of the batched CG


def _require_pyamg():
    """In:  --
    Out: the pyamg module (ImportError with the install hint when it is
         missing -- pyamg is the optional [amg] extra, CPU-only)."""
    try:
        import pyamg
    except ImportError as e:
        raise ImportError(
            "--solver amg (iter 2) needs pyamg for the CPU hierarchy"
            " build: pip install pyamg  (or pip install .[amg])") from e
    return pyamg


def _near_nullspace(points, cells, periodic_cells, n_unique, pin):
    """Rigid-body candidate modes B for the smoothed-aggregation setup:
    3 translations + the rotations the SG coordinates support (coords
    pad into the LAST slots -- thickness is always the last SG
    coordinate), zeroed on the pinned dofs, unit-normalized, degenerate
    columns dropped.  A preconditioner ingredient only: the solve is
    gated by the CG residual, never by B.

    In:  points (V, n_sg) node coords; cells (E, N) mesh connectivity;
         periodic_cells (E, N) reduced connectivity; n_unique int;
         pin (n_pin,) pinned dof ids
    Out: B (n_unique, m) float64 C-contiguous, 3 <= m <= 6."""
    pts = np.asarray(points, float)
    n_red = n_unique // 3
    X = np.zeros((n_red, 3))
    # reduced-node coords: scatter mesh nodes onto their masters (a
    # periodic slave overwrites by a lattice vector -- immaterial for a
    # smooth-mode hint)
    X[np.asarray(periodic_cells, np.int64).ravel(), 3 - pts.shape[1]:] = \
        pts[np.asarray(cells, np.int64).ravel(), :]
    cols = [np.zeros((n_red, 3)) for _ in range(6)]
    for a in range(3):
        cols[a][:, a] = 1.0                        # translations
    ex = np.eye(3)
    for a in range(3):                             # rotations e_a x X
        cols[3 + a] = np.cross(np.broadcast_to(ex[a], X.shape), X)
    B = np.stack([c.ravel() for c in cols], axis=1)
    B[np.asarray(pin, np.int64)] = 0.0
    nrm = np.linalg.norm(B, axis=0)
    keep = nrm > 1e-8 * max(float(nrm.max()), 1.0)
    return np.ascontiguousarray(B[:, keep] / nrm[keep])


def _csr_triplets(A):
    """In:  A scipy sparse matrix
    Out: dict {d, c, r} jnp csr-order triplets on the default device
         (rows sorted -- the matvec segment_sum runs sorted)."""
    A = A.tocsr()
    A.sort_indices()
    r = np.repeat(np.arange(A.shape[0], dtype=np.int32),
                  np.diff(A.indptr))
    return {"d": jnp.asarray(A.data),
            "c": jnp.asarray(A.indices.astype(np.int32)),
            "r": jnp.asarray(r)}


def build_hierarchy(A_csr, B):
    """pyamg smoothed-aggregation setup (CPU) -> device-resident levels.
    Per level: the Galerkin A_l as csr triplets, the inverse diagonal +
    the Chebyshev nodes of [rho/30, 1.1 rho] (rho = spectral radius of
    D^-1 A_l) for the smoother, and the prolongator P_l triplets (its
    transpose scatter IS the restriction, so R is never stored); the
    coarsest level becomes a dense symmetric inverse.

    In:  A_csr (n, n) pinned SPD csr; B (n, m) near-nullspace modes
    Out: (levels, coarse) -- levels tuple of dicts {d, c, r, Di, th,
         Pd, Pc, Pr}, coarse {"Ainv": (nc, nc)}; all jnp FP64 on the
         default device."""
    pyamg = _require_pyamg()
    from pyamg.util.linalg import approximate_spectral_radius
    # energy-minimized prolongation: 92 -> 34 CG iterations on the G1
    # spike-plate anchor vs jacobi-smoothed P, at ~3x the (small) setup.
    # OPENSG_AMG_SMOOTH=jacobi trades iterations for a ~3x cheaper CPU
    # setup -- the winning trade when iterations run on a GPU and the
    # serial setup dominates wall clock (big-dof Colab runs)
    ml = pyamg.smoothed_aggregation_solver(
        A_csr, B=B, max_coarse=_MAX_COARSE,
        smooth=os.environ.get("OPENSG_AMG_SMOOTH", "energy"), keep=False)
    levels = []
    for lv in ml.levels[:-1]:
        A = lv.A.tocsr()
        d = A.diagonal()
        rho = float(approximate_spectral_radius(diags(1.0 / d) @ A))
        ld = _csr_triplets(A)
        ld["Di"] = jnp.asarray(1.0 / d)
        lo, hi = _CHEB_LO * rho, _CHEB_HI * rho
        ld["th"] = jnp.asarray(
            0.5 * (hi + lo) + 0.5 * (hi - lo)
            * np.cos(np.pi * (2 * np.arange(1, _CHEB_DEG + 1) - 1)
                     / (2 * _CHEB_DEG)))
        P = lv.P.tocsr()
        P.sort_indices()
        ld["Pd"] = jnp.asarray(P.data)
        ld["Pc"] = jnp.asarray(P.indices.astype(np.int32))
        ld["Pr"] = jnp.asarray(np.repeat(
            np.arange(P.shape[0], dtype=np.int32), np.diff(P.indptr)))
        levels.append(ld)
    Ainv = np.linalg.inv(ml.levels[-1].A.toarray())
    coarse = {"Ainv": jnp.asarray(0.5 * (Ainv + Ainv.T))}
    return tuple(levels), coarse


def _spmv(T, x, n):
    """y = A x through csr triplets, device-resident.

    In:  T dict {d, c, r} (extra keys ignored); x (n_col,); n static
    Out: y (n,)."""
    return jax.ops.segment_sum(T["d"] * x[T["c"]], T["r"],
                               num_segments=n, indices_are_sorted=True)


def _smooth(lv, r, x, n):
    """Chebyshev(_CHEB_DEG) product-form sweep toward A^-1 r from the
    iterate x: the diagonally-preconditioned polynomial with roots at
    the Chebyshev nodes of [rho/30, 1.1 rho] (lv["th"]).  A polynomial
    in Di A, so pre/post applications commute -- the V-cycle stays an
    SPD preconditioner.

    In:  lv level dict; r (n,) rhs; x (n,) start; n static
    Out: x (n,) smoothed."""
    for i in range(_CHEB_DEG):
        x = x + lv["Di"] * (r - _spmv(lv, x, n)) / lv["th"][i]
    return x


def _smooth0(lv, r, n):
    """_smooth from a ZERO start (the first matvec is skipped).

    In:  lv level dict; r (n,) rhs; n static
    Out: x (n,)."""
    x = lv["Di"] * r / lv["th"][0]
    for i in range(1, _CHEB_DEG):
        x = x + lv["Di"] * (r - _spmv(lv, x, n)) / lv["th"][i]
    return x


def _vcycle(levels, coarse, r0):
    """One symmetric V-cycle with Chebyshev(_CHEB_DEG) smoothing:
    z ~ A^-1 r0.  Pre-smoothing starts from zero; post-smoothing
    applies the same polynomial, so the cycle is an SPD preconditioner
    for CG.  Unrolled over the (static) level count under jit.

    In:  levels/coarse from build_hierarchy; r0 (n,) residual
    Out: z (n,)."""
    ns = [lv["Di"].shape[0] for lv in levels]
    ns.append(coarse["Ainv"].shape[0])
    xs, rs = [], []
    r = r0
    for i, lv in enumerate(levels):
        x = _smooth0(lv, r, ns[i])
        res = r - _spmv(lv, x, ns[i])
        xs.append(x)
        rs.append(r)
        # restrict: R = P^T is the transpose scatter of the P triplets
        r = jax.ops.segment_sum(lv["Pd"] * res[lv["Pr"]], lv["Pc"],
                                num_segments=ns[i + 1])
    x = coarse["Ainv"] @ r
    for i in range(len(levels) - 1, -1, -1):
        lv = levels[i]
        xf = xs[i] + jax.ops.segment_sum(
            lv["Pd"] * x[lv["Pc"]], lv["Pr"], num_segments=ns[i],
            indices_are_sorted=True)
        x = _smooth(lv, rs[i], xf, ns[i])
    return x


def _pcg_one(levels, coarse, fine, b, rtol, maxiter):
    """Preconditioned CG on ONE column, fully on the jax default
    device.  `fine` is the TRUE operator of the outer iteration (the
    ladder passes its own hi-quadrature matrix; the hierarchy only
    preconditions, so the answer is exact for whichever system `fine`
    is).

    In:  levels/coarse from build_hierarchy; fine {d, c, r}; b (n,);
         rtol float; maxiter int
    Out: x (n,); k int iterations; relres float."""
    n = b.shape[0]
    normb = jnp.linalg.norm(b)
    tolb = rtol * jnp.where(normb > 0, normb, 1.0)
    z0 = _vcycle(levels, coarse, b)

    def cond(s):
        _, r, _, _, _, k = s
        return (jnp.linalg.norm(r) > tolb) & (k < maxiter)

    def body(s):
        x, r, z, p, rz, k = s
        Ap = _spmv(fine, p, n)
        alpha = rz / (p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        z = _vcycle(levels, coarse, r)
        rz2 = r @ z
        p = z + (rz2 / rz) * p
        return (x, r, z, p, rz2, k + 1)

    x, r, _, _, _, k = jax.lax.while_loop(
        cond, body, (jnp.zeros_like(b), b, z0, z0, b @ z0,
                     jnp.zeros((), jnp.int32)))
    return x, k, jnp.linalg.norm(r) / jnp.where(normb > 0, normb, 1.0)


@partial(jax.jit, static_argnames=("maxiter",))
def _pcg_block(levels, coarse, fine, Bcols, rtol, maxiter):
    """vmapped _pcg_one over a block of RHS columns: the matvecs batch
    (bandwidth-amortized on CPU, parallel on GPU) and the block runs
    LOCK-STEP -- every column iterates until the slowest converges, so
    the reported count is the block maximum.

    In:  levels/coarse/fine as _pcg_one; Bcols (n, m); rtol; maxiter
         static int
    Out: X (n, m); k (m,) iterations; relres (m,)."""
    return jax.vmap(
        lambda b: _pcg_one(levels, coarse, fine, b, rtol, maxiter),
        in_axes=1, out_axes=(1, 0, 0))(Bcols)


def solve_columns(levels, coarse, fine, RHS, rtol=_RTOL,
                  maxiter=_MAXITER):
    """Solve fine x = b for every RHS column, in vmapped blocks sized
    by the _COL_BYTES matvec-transient budget (the jitted _pcg_block
    compiles once per pattern-and-width and is reused).

    In:  levels/coarse/fine as _pcg_one; RHS (n, m) host float64;
         rtol; maxiter
    Out: X (n, m) np; iters list[int] per column (block maxima); worst
         float worst relative residual."""
    n, m = RHS.shape
    bs = max(1, min(m, int(_COL_BYTES / (8 * fine["d"].shape[0] + 1))))
    X = np.zeros_like(RHS)
    iters, worst = [], 0.0
    for j0 in range(0, m, bs):
        Xb, kb, rb = _pcg_block(levels, coarse, fine,
                                jnp.asarray(RHS[:, j0:j0 + bs]),
                                rtol, maxiter)
        X[:, j0:j0 + bs] = np.asarray(Xb)
        iters += [int(k) for k in np.asarray(kb)]
        worst = max(worst, float(np.max(np.asarray(rb))))
    if worst > rtol:
        print(" amg WARNING: CG stopped at maxiter=%d, worst relres"
              " %.2e (asked %.0e)" % (maxiter, worst, rtol))
    return X, iters, worst


def amg_homo(x_end, u_0_g, dphi_dxi_qnp, phi_qn, W_q, C_ess,
             periodic_cells, unique_dofs, n_unique, n_model, n_sg,
             points, cells, rtol=_RTOL):
    """iter 2 variant of sg_homo._homo_direct: the SAME element kernels
    and the SAME pinned-first-node system, solved by AMG-preconditioned
    CG instead of a factorization.  Digit-safe swap: C_eff is
    stationary in V0, so the rtol solver error enters second order.

    In:  x_end (E, N, d); u_0_g (V*3,) zero seed; dphi_dxi_qnp/phi_qn/
         W_q quadrature basis tables; C_ess (E, 6, 6); periodic_cells
         (E, N); unique_dofs (n_unique,); n_unique int; n_model 2
         plate / 3 solid; n_sg SG dimension; points (V, n_sg) and
         cells (E, N) the mesh (near-nullspace modes); rtol float
    Out: C_eff (H, H); V0_matrix (n_unique, H) jnp; omega float;
         ctx dict {levels, coarse, pin, rtol} -- the device hierarchy
         the refined-plate ladder reuses (make_constrained_solver)."""
    t0 = time.perf_counter()
    # slab-streamed assembly by default (OPENSG_ASSEMBLY=device for the
    # historical all-device program) -- live memory stays slab-bounded
    Dhe, A_csr, pin = assemble_rhs_and_pinned_csr(
        x_end, dphi_dxi_qnp, phi_qn, W_q, C_ess, periodic_cells,
        u_0_g[unique_dofs], n_model, n_sg, n_unique, sym=True)
    B = _near_nullspace(points, cells, periodic_cells, n_unique, pin)
    levels, coarse = build_hierarchy(A_csr, B)
    fine = {k: levels[0][k] for k in ("d", "c", "r")}
    nnz = A_csr.nnz
    del A_csr
    # terse-console rule: no setup/progress chatter -- the law and the
    # timed .out are the results; warnings alone are loud
    del nnz, t0
    RHS = -np.asarray(Dhe)
    RHS[np.asarray(pin, np.int64)] = 0.0
    V0, iters, _ = solve_columns(levels, coarse, fine, RHS, rtol)
    sg_progress.tick("solve")
    V0_matrix = jnp.asarray(V0)
    D1 = jnp.einsum('ni,nj->ij', V0_matrix, Dhe)
    D_bar, omega = compute_homogenized_constants(
        x_end, dphi_dxi_qnp, phi_qn, W_q, C_ess, n_model, n_sg)
    C_eff = (D_bar + D1) / omega
    ctx = {"levels": levels, "coarse": coarse, "pin": pin, "rtol": rtol}
    return C_eff, V0_matrix, omega, ctx


def make_constrained_solver(ctx, D_hh, w_dof, scl):
    """AMG replacement for plate_shear_ladder's KKT factorization
    (solver="amg"): solve D_hh x + scl Hpsi mu = -rhs with
    Hpsi^T x = 0, never forming the KKT.  K (the 3 translation
    columns) spans null(D_hh) on the periodic SG, so
    mu = K^T b / (scl sum_w) absorbs the kernel-incompatible part of
    the load (EXACTLY the KKT multiplier), the pinned SPD system
    Z D_hh Z + (I - Z) returns THE solution member with x[0:3] = 0,
    and the closing kernel shift lands the <w> = 0 gauge -- identical
    to the KKT solution at the CG tolerance.  The V0-stage hierarchy
    preconditions (same operator up to quadrature degree and the pin);
    the CG matvec runs on THIS matrix, so the answer is exact for the
    ladder system regardless.

    In:  ctx dict from amg_homo; D_hh (n, n) csr ladder stiffness;
         w_dof (n,) <w> weights (node weights repeated per component);
         scl float KKT row scale
    Out: solve_constrained(rhs (n, m)) -> (n, m) np -- the drop-in."""
    n = D_hh.shape[0]
    pin = np.asarray(ctx["pin"], np.int64)
    mask = np.ones(n)
    mask[pin] = 0.0
    Z = diags(mask)
    fine = _csr_triplets(Z @ D_hh @ Z + diags(1.0 - mask))
    comp = np.arange(n) % 3
    sw = float(w_dof[0::3].sum())

    def solve_constrained(rhs):
        t0 = time.perf_counter()
        b = -np.asarray(rhs, float)
        m = b.shape[1]
        mu = b.reshape(-1, 3, m).sum(axis=0)           # K^T b, (3, m)
        b2 = b - w_dof[:, None] * mu[comp] / sw        # scl cancels
        b2[pin] = 0.0
        X, iters, _ = solve_columns(ctx["levels"], ctx["coarse"], fine,
                                    b2, ctx["rtol"])
        c = -(w_dof[:, None] * X).reshape(-1, 3, m).sum(axis=0) / sw
        X += c[comp]                                   # <w> = 0 gauge
        del iters, t0                    # terse console: results only
        return X

    return solve_constrained
