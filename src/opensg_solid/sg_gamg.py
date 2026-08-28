"""sg_gamg.py -- iter 4 of the --solver menu: PETSc GAMG
(smoothed-aggregation)-preconditioned CG for the fluctuation solves of
the SSDM plate/solid routes (sg_homo dispatch solver="gamg"), through
the vendored jetsci PETSc layer.

Same contract as sg_amg.amg_homo: the exact pinned SPD CSR that
_homo_direct factorizes (sg_assembly.assemble_rhs_and_pinned_csr,
sym=True) is handed to ONE PETSc KSP (CG, unpreconditioned norm --
jetsci's convention) with PC GAMG; setup runs ONCE and every RHS
column reuses it (KSPMatSolve when the build carries it, else a
per-column loop on the same KSP).  jetsci supplies the option plumbing
(petsc_ksp.options.PETScMethodOptions: aij on CPU, aijcusparse on GPU
exactly as its default) and the Mat construction recipe (COO
preallocation + setValuesCOO, the buildKSP_Keith sequence); the
Mat/KSP objects themselves are built with petsc4py directly because
jetsci's buffer-callback lifecycle is cupy/CUDA-bound and has no hook
for PC config, tolerances or near-nullspace attachment.

DOCTRINE.  OpenSG removes rigid-body modes via the KKT / pinned-dof
construction ONLY -- never nullspace deflation.  The rigid-body
vectors passed to PETSc MatSetNearNullSpace here are PRECONDITIONER
COARSE-SPACE HINTS (they seed GAMG's aggregation coarse spaces) and
nothing else: the solved system is always the pinned SPD CSR from
assemble_rhs_and_pinned_csr, unchanged.

# ----------------------------------------------------------------------------
# ALL VARIABLES USED IN THIS MODULE
# ----------------------------------------------------------------------------
# inputs (gamg_homo mirrors sg_amg.amg_homo)
#   x_end, u_0_g, dphi_dxi_qnp, phi_qn, W_q, C_ess, periodic_cells,
#   unique_dofs, n_unique, n_model, n_sg      see sg_homo._homo_direct
#   points (V, n_sg), cells (E, N)            the mesh (rigid-mode hints)
#   rtol                CG relative-residual tolerance (law error is
#                       SECOND order in it: 1e-8 -> ~1e-16)
# working
#   A_csr, pin          pinned SPD csr + pinned dof ids (shared helper)
#   B                   (n_unique, m) rigid-mode candidates (sg_amg.
#                       _near_nullspace), QR-orthonormalized for PETSc
#   mat, ksp            PETSc Mat (COO recipe) + KSP CG / PC GAMG
#                       (a read-only sg_progress monitor rides the KSP
#                       for the duration of a solve, armed bar only)
#   X, iters, worst     per-column solutions, iteration counts, worst
#                       relres (checked host-side against the scipy A)
#   ctx                 {"backend": "gamg", "pin", "rtol", "B"} -- the
#                       handle plate_shear_ladder routes back here
# outputs
#   C_eff, V0_matrix, omega                   as _homo_direct
# ----------------------------------------------------------------------------
"""
import os

import numpy as np

import jax
import jax.numpy as jnp
from scipy.sparse import diags

from opensg_solid.sg_assembly import (assemble_rhs_and_pinned_csr,
                                      compute_homogenized_constants)
from opensg_solid import sg_progress

# same stopping pair as sg_amg (a tolerance certificate is one rerun
# with OPENSG_AMG_RTOL tightened 100x: the law must hold its digits)
_RTOL = float(os.environ.get("OPENSG_AMG_RTOL", 1e-8))
_MAXITER = int(os.environ.get("OPENSG_AMG_MAXITER", 500))


def _require_petsc():
    """In:  --
    Out: the petsc4py PETSc namespace (ImportError with the install
         hint when it is missing -- petsc4py is the optional [gamg]
         extra)."""
    try:
        from petsc4py import PETSc
    except ImportError as e:
        raise ImportError(
            "--solver gamg (iter 4) needs petsc4py for the PETSc GAMG"
            " preconditioner: conda install -c conda-forge petsc4py"
            "  (or pip install .[gamg])") from e
    return PETSc


def _petsc_type_names():
    """Mat/KSP/PC PETSc type names through the vendored jetsci options
    plumbing: aijcusparse on GPU (jetsci's own default), aij on CPU,
    CG + GAMG.  Falls back to the literal names if the jetsci package
    is unimportable (e.g. a snapshot without the CPU import guard).

    In:  --
    Out: (mat, ksp, pc) str PETSc type names."""
    on_gpu = jax.default_backend() == "gpu"
    mt = "aijcusparse" if on_gpu else "aij"
    try:
        from jetsci.petsc_ksp.options import PETScMethodOptions
        o = PETScMethodOptions(mat_type=mt, ksp_type="cg",
                               pc_type="gamg")
        return (o.matrix_construction_options()[0],
                o.ksp_construction_options()[0],
                o.pc_construction_options()[0])
    except ImportError:
        return mt, "cg", "gamg"


def _usable_mat_type(PETSc, mat_type):
    """The requested Mat type if THIS petsc build provides it, else aij.

    jax seeing a GPU says nothing about PETSc: the conda-forge petsc is
    built without CUDA, so aijcusparse raises 'Unknown Mat type' (code
    86) at create/convert.  Probe once on a 1x1 mat -- a GPU gamg run
    needs a CUDA-enabled PETSc (examples/colab/petsc_gpu_setup.md).

    In:  PETSc namespace; mat_type str
    Out: str -- mat_type or "aij" (one loud line when it falls back)."""
    if mat_type == "aij":
        return mat_type
    m = PETSc.Mat().create(PETSc.COMM_SELF)
    try:
        m.setSizes((1, 1))
        m.setType(mat_type)
        return mat_type
    except PETSc.Error:
        print(" NOTE: this petsc has no %s (built without CUDA) -- gamg"
              " runs on the CPU; see examples/colab/petsc_gpu_setup.md"
              % mat_type)
        return "aij"
    finally:
        m.destroy()


def _mat_from_csr(PETSc, A, mat_type):
    """PETSc Mat from a scipy sparse matrix.  The jetsci buildKSP COO
    recipe (create / setSizes / setType / setPreallocationCOO /
    setValuesCOO -- one sequence for aij and aijcusparse) when this
    petsc4py build carries the COO api; older builds (<= 3.21) take
    the CSR route (createAIJ, then convert to the requested type).
    Index arrays are cast to PETSc.IntType -- PetscInt is fixed when
    petsc is BUILT (int64 only with --with-64-bit-indices), so this is
    the widest the ABI accepts, not a doctrine int32.

    In:  PETSc namespace; A scipy sparse (n, n); mat_type str
    Out: mat PETSc.Mat, assembled."""
    IT = PETSc.IntType
    mat = PETSc.Mat().create(PETSc.COMM_SELF)
    if hasattr(mat, "setPreallocationCOO"):
        C = A.tocoo()
        mat.setSizes(C.shape)
        mat.setType(mat_type)
        mat.setPreallocationCOO(C.row.astype(IT), C.col.astype(IT))
        mat.setValuesCOO(np.ascontiguousarray(C.data, dtype=np.float64))
        return mat
    mat.destroy()
    C = A.tocsr()
    C.sort_indices()
    mat = PETSc.Mat().createAIJ(
        size=C.shape, csr=(C.indptr.astype(IT), C.indices.astype(IT),
                           np.ascontiguousarray(C.data, np.float64)),
        comm=PETSc.COMM_SELF)
    if mat_type != "aij":
        mat = mat.convert(mat_type)
    return mat


def _attach_near_nullspace(PETSc, mat, B):
    """Attach the rigid-body vectors as PRECONDITIONER COARSE-SPACE
    HINTS via MatSetNearNullSpace.  OpenSG removes rigid-body modes via
    the KKT / pinned-dof construction ONLY -- these vectors never alter
    the solved system (the pinned SPD CSR), they only seed GAMG's
    smoothed-aggregation coarse spaces.  PETSc wants the set
    orthonormal, so the sg_amg candidates are QR-reduced first.

    In:  PETSc namespace; mat PETSc.Mat; B (n, m) rigid-mode candidates
    Out: -- (mat carries the near-nullspace)."""
    Q, R = np.linalg.qr(np.asarray(B, float))
    keep = np.abs(np.diag(R)) > 1e-10 * max(float(np.abs(R).max()), 1.0)
    vecs = [PETSc.Vec().createWithArray(
        np.ascontiguousarray(Q[:, j]), comm=PETSc.COMM_SELF)
        for j in range(Q.shape[1]) if keep[j]]
    ns = PETSc.NullSpace().create(constant=False, vectors=vecs,
                                  comm=PETSc.COMM_SELF)
    mat.setNearNullSpace(ns)
    ns.destroy()                        # mat holds its own reference
    for v in vecs:
        v.destroy()


def _build_ksp(PETSc, mat, ksp_name, pc_name, rtol, maxiter):
    """KSP CG + PC GAMG (smoothed aggregation) on an assembled Mat,
    set up ONCE -- every subsequent solve reuses the hierarchy.
    Unpreconditioned norm (the jetsci convention), so rtol gates the
    same relative residual sg_amg reports.

    GAMG's aggressive coarsening SQUARES the graph (a sparse A*A whose
    result carries 10-30x the nonzeros of A); on a GPU mat that product
    runs through cuSPARSE and dies at scale --
    CUSPARSE_STATUS_INSUFFICIENT_RESOURCES at 9.4M dofs / ~760M nnz.
    It is off here: one extra level, no A*A.  Both option spellings are
    set because PETSc renamed it (square_graph -> aggressive_coarsening
    at 3.20); an unknown option is inert.

    In:  PETSc namespace; mat PETSc.Mat; ksp_name/pc_name str;
         rtol float; maxiter int
    Out: ksp PETSc.KSP, set up."""
    opts = PETSc.Options()
    opts["pc_gamg_aggressive_coarsening"] = 0
    opts["pc_gamg_square_graph"] = 0
    ksp = PETSc.KSP().create(PETSc.COMM_SELF)
    ksp.setOperators(mat)
    ksp.setType(ksp_name)
    ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
    ksp.setTolerances(rtol=rtol, atol=1e-50, max_it=maxiter)
    pc = ksp.getPC()
    pc.setType(pc_name)
    pc.setGAMGType(PETSc.PC.GAMGType.AGG)      # smoothed aggregation
    pc.setFromOptions()
    ksp.setUp()
    return ksp


def _mat_and_ksp(PETSc, A_csr, B, mat_t, ksp_t, pc_t, rtol):
    """Upload the matrix and set the hierarchy up.  A GPU setup that
    exhausts device resources ABORTS with the reason -- GAMG's coarse
    operators are sparse matrix products whose cuSPARSE workspace does
    not fit at this size.  No silent CPU retry: that decision (and the
    hours it would cost) belongs to whoever started the run.

    PC GAMG's setup is ONE opaque call with no callback, so the bar
    carries the INDETERMINATE marker across it (never a faked
    percentage) -- sg_progress; the caller has opened the phase.

    In:  PETSc namespace; A_csr pinned SPD csr; B rigid-mode hints;
         mat_t/ksp_t/pc_t str type names; rtol float
    Out: (mat, ksp) both live, hierarchy set up."""
    sg_progress.busy()
    try:
        mat = _mat_from_csr(PETSc, A_csr, mat_t)
        _attach_near_nullspace(PETSc, mat, B)
        try:
            return mat, _build_ksp(PETSc, mat, ksp_t, pc_t, rtol,
                                   _MAXITER)
        except PETSc.Error as e:
            if mat_t == "aij":
                raise
            _why = [ln.strip() for ln in str(e).splitlines()
                    if "error" in ln.lower() and ":" in ln]
            raise SystemExit(
                "\n gamg: the GPU ran out of resources building the"
                " hierarchy for %d dofs / %d nonzeros\n %s"
                % (A_csr.shape[0], A_csr.nnz,
                   _why[-1] if _why else "cuSPARSE: insufficient"
                   " resources")) from None
    finally:
        sg_progress.idle()


def _progress_monitor(m, rtol):
    """The sg_progress hook of a KSP: a monitor callback mapping each
    iteration's residual to the SAME log fraction sg_amg reports.  The
    KSP norm type is UNPRECONDITIONED (_build_ksp), so normalizing by
    the rnorm the monitor sees at its=0 (= ||b||, the initial guess is
    zero on both solve paths) makes rnorm/r0 exactly amg's relative
    residual.  Columns solve SEQUENTIALLY here (KSPMatSolve loops
    KSPSolve internally), so each its=0 opens the next column and the
    fraction is scaled (j + p) / m across them -- one sweep per solve,
    as chunked amg does across its RHS blocks.

    In:  m int RHS columns of this solve; rtol float
    Out: cb(ksp, its, rnorm) -- a PETSc KSP monitor callback."""
    st = {"r0": 0.0, "j": -1}

    def cb(_ksp, its, rnorm):
        if its == 0:
            st["r0"], st["j"] = float(rnorm), st["j"] + 1
            return
        if st["r0"] <= 0.0:
            return
        p = (np.log10(max(float(rnorm) / st["r0"], 1e-300))
             / np.log10(rtol))
        sg_progress.solve((st["j"] + min(1.0, max(0.0, p))) / m)

    return cb


def solve_columns(PETSc, ksp, mat, A_csr, RHS, rtol=_RTOL):
    """Solve A x = b for every RHS column on ONE set-up KSP:
    KSPMatSolve when the build carries it, else a per-column loop on
    the same KSP.  The relative residual is re-checked host-side
    against the scipy CSR (authoritative regardless of the path).
    The sg_progress bar rides a KSP MONITOR (_progress_monitor),
    attached for the duration of THIS call and only when the bar is
    armed -- a per-iteration callback is cheap but not free, and a
    monitor is read-only, so the iterates cannot move.

    In:  PETSc namespace; ksp set-up KSP; mat its Mat; A_csr scipy csr
         (the SAME operator, residual check); RHS (n, m) host float64;
         rtol float
    Out: X (n, m) np; iters list[int] (per column for the loop, the
         final KSP count for MatSolve); worst float relres."""
    n, m = RHS.shape
    X = np.zeros_like(RHS)
    iters = []
    bar = sg_progress.active()
    if bar:
        ksp.setMonitor(_progress_monitor(m, rtol))
    try:
        Bm = PETSc.Mat().createDense((n, m), array=np.asfortranarray(RHS),
                                     comm=PETSc.COMM_SELF)
        Xm = PETSc.Mat().createDense((n, m), comm=PETSc.COMM_SELF)
        Xm.setUp()
        ksp.matSolve(Bm, Xm)
        X[:] = Xm.getDenseArray()
        iters = [int(ksp.getIterationNumber())]
        Bm.destroy(); Xm.destroy()
    except (AttributeError, PETSc.Error):
        if bar:                   # a fresh sweep: matSolve may have run
            ksp.cancelMonitor()   # partway before falling back here
            ksp.setMonitor(_progress_monitor(m, rtol))
        x_vec, b_vec = mat.createVecs()
        for j in range(m):
            b_vec.array[:] = RHS[:, j]
            x_vec.set(0.0)
            ksp.solve(b_vec, x_vec)
            X[:, j] = x_vec.array
            iters.append(int(ksp.getIterationNumber()))
        x_vec.destroy(); b_vec.destroy()
    finally:
        if bar:
            ksp.cancelMonitor()   # the next solve installs its own
            sg_progress.solve(1.0)
    nb = np.linalg.norm(RHS, axis=0)
    nb[nb == 0.0] = 1.0
    worst = float(np.max(np.linalg.norm(RHS - A_csr @ X, axis=0) / nb))
    if worst > rtol:
        print(" gamg WARNING: CG stopped early, worst relres %.2e"
              " (asked %.0e)" % (worst, rtol))
    if os.environ.get("OPENSG_GAMG_VERBOSE"):    # benchmarking aid only
        print(" gamg: %d col(s), iters %s, worst relres %.2e"
              % (m, iters, worst))
    return X, iters, worst


def gamg_homo(x_end, u_0_g, dphi_dxi_qnp, phi_qn, W_q, C_ess,
              periodic_cells, unique_dofs, n_unique, n_model, n_sg,
              points, cells, rtol=_RTOL):
    """iter 4 variant of sg_homo._homo_direct: the SAME element kernels
    and the SAME pinned-first-node system, solved by PETSc GAMG-
    preconditioned CG instead of a factorization.  Digit-safe swap:
    C_eff is stationary in V0, so the rtol solver error enters second
    order.  Rigid-body vectors ride along solely as GAMG coarse-space
    hints (MatSetNearNullSpace) -- see the module doctrine note.

    In:  x_end (E, N, d); u_0_g (V*3,) zero seed; dphi_dxi_qnp/phi_qn/
         W_q quadrature basis tables; C_ess (E, 6, 6); periodic_cells
         (E, N); unique_dofs (n_unique,); n_unique int; n_model 2
         plate / 3 solid; n_sg SG dimension; points (V, n_sg) and
         cells (E, N) the mesh (rigid-mode hints); rtol float
    Out: C_eff (H, H); V0_matrix (n_unique, H) jnp; omega float;
         ctx dict {backend, pin, rtol, B} -- the handle the
         refined-plate ladder routes to make_constrained_solver."""
    PETSc = _require_petsc()
    from opensg_solid.sg_amg import _near_nullspace
    Dhe, A_csr, pin = assemble_rhs_and_pinned_csr(
        x_end, dphi_dxi_qnp, phi_qn, W_q, C_ess, periodic_cells,
        u_0_g[unique_dofs], n_model, n_sg, n_unique, sym=True)
    B = _near_nullspace(points, cells, periodic_cells, n_unique, pin)
    mat_t, ksp_t, pc_t = _petsc_type_names()
    mat_t = _usable_mat_type(PETSc, mat_t)
    sg_progress.stage(sg_progress.W_SETUP, "setup")     # opaque phase
    mat, ksp = _mat_and_ksp(PETSc, A_csr, B, mat_t, ksp_t, pc_t, rtol)
    RHS = -np.asarray(Dhe)
    RHS[np.asarray(pin, np.int64)] = 0.0
    # the solve phase: residual-driven (the KSP monitor), so an ETA
    # rides it whenever the window runs to 100% -- sg_progress
    sg_progress.stage(sg_progress.solve_window(), "solve", eta=True)
    V0, iters, _ = solve_columns(PETSc, ksp, mat, A_csr, RHS, rtol)
    del iters                # terse console: results only
    ksp.destroy()
    mat.destroy()
    V0_matrix = jnp.asarray(V0)
    D1 = jnp.einsum('ni,nj->ij', V0_matrix, Dhe)
    D_bar, omega = compute_homogenized_constants(
        x_end, dphi_dxi_qnp, phi_qn, W_q, C_ess, n_model, n_sg)
    C_eff = (D_bar + D1) / omega
    ctx = {"backend": "gamg", "pin": pin, "rtol": rtol, "B": B}
    return C_eff, V0_matrix, omega, ctx


def make_constrained_solver(ctx, D_hh, w_dof, scl):
    """GAMG replacement for plate_shear_ladder's KKT factorization
    (solver="gamg"): solve D_hh x + scl Hpsi mu = -rhs with
    Hpsi^T x = 0, never forming the KKT.  The gauge projection is
    sg_amg.make_constrained_solver's EXACTLY -- K (the 3 translation
    columns) spans null(D_hh) on the periodic SG, so
    mu = K^T b / (scl sum_w) absorbs the kernel-incompatible part of
    the load (EXACTLY the KKT multiplier), the pinned SPD system
    Z D_hh Z + (I - Z) returns THE solution member with x[0:3] = 0,
    and the closing kernel shift lands the <w> = 0 gauge -- identical
    to the KKT solution at the CG tolerance.  Only the inner linear
    solve differs: a SECOND KSP (CG + GAMG) set up ONCE on the pinned
    ladder matrix itself, with the same rigid-body vectors reattached
    as preconditioner coarse-space hints (MatSetNearNullSpace; the
    solved system stays this pinned SPD matrix, unchanged -- module
    doctrine note).

    In:  ctx dict from gamg_homo; D_hh (n, n) csr ladder stiffness;
         w_dof (n,) <w> weights (node weights repeated per component);
         scl float KKT row scale
    Out: solve_constrained(rhs (n, m)) -> (n, m) np -- the drop-in."""
    PETSc = _require_petsc()
    n = D_hh.shape[0]
    pin = np.asarray(ctx["pin"], np.int64)
    mask = np.ones(n)
    mask[pin] = 0.0
    Z = diags(mask)
    A_lad = (Z @ D_hh @ Z + diags(1.0 - mask)).tocsr()
    comp = np.arange(n) % 3
    sw = float(w_dof[0::3].sum())
    mat_t, ksp_t, pc_t = _petsc_type_names()
    mat_t = _usable_mat_type(PETSc, mat_t)
    mat, ksp = _mat_and_ksp(PETSc, A_lad, ctx["B"], mat_t, ksp_t, pc_t,
                            ctx["rtol"])

    def solve_constrained(rhs):
        b = -np.asarray(rhs, float)
        m = b.shape[1]
        mu = b.reshape(-1, 3, m).sum(axis=0)           # K^T b, (3, m)
        b2 = b - w_dof[:, None] * mu[comp] / sw        # scl cancels
        b2[pin] = 0.0
        X, iters, _ = solve_columns(PETSc, ksp, mat, A_lad, b2,
                                    ctx["rtol"])
        c = -(w_dof[:, None] * X).reshape(-1, 3, m).sum(axis=0) / sw
        X += c[comp]                                   # <w> = 0 gauge
        del iters                        # terse console: results only
        return X

    return solve_constrained
