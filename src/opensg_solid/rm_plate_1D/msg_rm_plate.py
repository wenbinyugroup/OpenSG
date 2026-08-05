"""msg_rm_plate.py -- MSG Reissner-Mindlin plate law + 3-D recovery (Yu-2002/2003).

Core OpenSG module, JAX implementation.  Equation numbers = Yu, Hodges &
Volovoi, Computers & Structures 81:439-454 (2003).  The ladder:

  V0 (Eq. 39)  -> A6            V11/V12 (Eq. 45)  -> H, U*-fit -> G_msg (Eq. 61)
  V21/22/23 + V21t/22t/23t (Eq. 64, two variants) + V1L/V2L load columns
  recovery: Eq. (63) first order; Eq. (66) second order with a ROW SPLIT
  (in-plane stress rows from the detilted chain, through-thickness rows
  from the tilted chain).

All rationale, conventions (tilt/detilt rules, V2 variant choice, load-column
signs), history and validation numbers: msg_rm_plate_README.md next to this
file.  Tests: tests/test_msg_rm_plate.py; recovery gate:
examples/opensg_rm_static/validate_v2.py.

Voigt strain order [11,22,33,23,13,12]; plate strains E = [e11,e22,g12,k11,
k22,k12]; shears gamma = [2g13,2g23].  x = through-thickness coordinate from
the reference plane (`fraction`: 0 = bottom/OML, 0.5 = center, 1 = IML).

DIMENSION SUFFIXES on internal arrays (the public dict keeps paper symbols):
    n  ndofs (3 per SG node)      l  3*(p+1), one element's dofs
    k  3, kernel/psi modes        v  6, 3-D Voigt components
    s  6, plate strains           i  2, in-plane warping components
    g  2, transverse shears       t  12, stacked [eps,1; eps,2]
    q  144, raveled (t,t)         m  27, LS unknowns [X(3), c1(12), c2(12)]
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import jax
import jax.numpy as jnp

from jax.scipy.linalg import lu_factor, lu_solve

from .msg_materials import rotated_stiffness_6x6, rotation_6x6, _plate_B


def _lagrange_N(nodes_xi: np.ndarray, xi: float) -> np.ndarray:
    """Lagrange shape-function values.
    In:  nodes_xi (p+1,) float -- element node positions on [-1, 1]
         xi       float        -- evaluation point on [-1, 1]
    Out: N        (p+1,) float -- N_a(xi), one value per node."""
    npn = len(nodes_xi)
    N = np.ones(npn)
    for i in range(npn):
        for j in range(npn):
            if j != i:
                N[i] *= (xi - nodes_xi[j]) / (nodes_xi[i] - nodes_xi[j])
    return N


def _grad_ops(nodes_xi: np.ndarray, xi: float) -> Tuple[np.ndarray, np.ndarray]:
    """Strain operators of the IN-PLANE warping gradients:
    w,1: e11 += w1,1 ; 2g13 += w3,1 ; g12 += w2,1
    w,2: e22 += w2,2 ; 2g23 += w3,2 ; g12 += w1,2
    In:  nodes_xi (p+1,) float, xi float -- as _lagrange_N
    Out: (Gamma_l1, Gamma_l2), each (6, 3*(p+1)) float -- [Gamma_la S]."""
    N = _lagrange_N(nodes_xi, xi)
    npn = len(nodes_xi)
    Gamma_l1 = np.zeros((6, 3 * npn)); Gamma_l2 = np.zeros((6, 3 * npn))
    for n in range(npn):
        Gamma_l1[0, 3 * n + 0] = N[n]      # eps11 <- w1,1
        Gamma_l1[4, 3 * n + 2] = N[n]      # 2g13  <- w3,1
        Gamma_l1[5, 3 * n + 1] = N[n]      # g12   <- w2,1
        Gamma_l2[1, 3 * n + 1] = N[n]      # eps22 <- w2,2
        Gamma_l2[3, 3 * n + 2] = N[n]      # 2g23  <- w3,2
        Gamma_l2[5, 3 * n + 0] = N[n]      # g12   <- w1,2
    return Gamma_l1, Gamma_l2


# Gamma_eps = E0 + x3*E1 (Eq. 11 direct plate-strain term, code strain order)
_E0 = np.zeros((6, 6)); _E0[0, 0] = _E0[1, 1] = _E0[5, 2] = 1.0
_E1 = np.zeros((6, 6)); _E1[0, 3] = _E1[1, 4] = _E1[5, 5] = 1.0

_BUCKETS = {}

# Relative singular-value cutoff for the truncated-SVD minimum-norm solve of the U* least
# squares (the "sigma_i = 0" test of Ascher & Greif Sec. 8.2).  The U* system is rank 26 of
# 27; its null singular value sits ~1e-16 of the largest, so anything in 1e-14..1e-10 works.
_LS_RCOND = 1e-12


def _bucket(nlay: int, n_per_layer: int, p: int) -> Dict[str, Any]:
    """Static structures + jitted kernels for one (n_ply, n_per_layer, elem_order) bucket.
    In:  nlay int -- number of plies; n_per_layer int -- SG elements per ply;
         p int -- element order (nodes per element = p + 1)
    Out: dict {nlay, n_per_layer, p, n_elem, ndofs, elem_layer (n_elem,) int,
         jit_single, jit_batch} -- cached in _BUCKETS, one entry per key."""
    key = (int(nlay), int(n_per_layer), int(p))
    if key in _BUCKETS:
        return _BUCKETS[key]
    nodes_xi = np.linspace(-1.0, 1.0, p + 1)
    xi_g, w_g = np.polynomial.legendre.leggauss(max(3, p + 1))
    # static reference operator tables per Gauss point (element-independent):
    #   [Gamma_h S] scales as 2/he -> store at he=2 (unit scale); [Gamma_l_a S] has no he
    Gh_ref = np.stack([_plate_B(nodes_xi, xi, 2.0) for xi in xi_g])
    Gl1_ref, Gl2_ref = zip(*[_grad_ops(nodes_xi, xi) for xi in xi_g])
    Gl1_ref = np.stack(Gl1_ref); Gl2_ref = np.stack(Gl2_ref)

    n_elem = nlay * n_per_layer
    n_node = p * n_elem + 1
    ndofs = 3 * n_node
    dofs_e = np.stack([np.arange(3 * p * e, 3 * p * e + 3 * (p + 1)) for e in range(n_elem)])
    elem_layer = np.repeat(np.arange(nlay), n_per_layer)
    sub_of_elem = np.arange(n_elem) % n_per_layer

    # kernel of D_hh = the 3 constant (rigid-translation) through-thickness modes; the
    # paper's psi (Eq. 31, psi^T H psi = I with H = <S^T S>) is kernel/sqrt(h), built
    # in-trace because h depends on the laminate.
    kernel_nk = np.zeros((ndofs, 3))
    kernel_nk[0::3, 0] = 1.0; kernel_nk[1::3, 1] = 1.0; kernel_nk[2::3, 2] = 1.0
    nodes_of_e = np.stack([np.arange(p * e, p * e + p + 1) for e in range(n_elem)])
    S_int_ref = np.array([sum(w_g[q] * _lagrange_N(nodes_xi, xi_g[q])[a] for q in range(len(xi_g)))
                          for a in range(p + 1)])   # int N_a dxi over [-1,1] (for H = <S^T S>)

    # D_1, D_2 (Eq. 51): Boolean selectors of eps = R - D_a gamma,_a, code strain order
    D1_sg = np.zeros((6, 2)); D2_sg = np.zeros((6, 2))
    D1_sg[3, 0] = 1.0; D1_sg[5, 1] = 1.0     # k11 <- 2g13,1 ; k12 <- 2g23,1
    D2_sg[4, 1] = 1.0; D2_sg[5, 0] = 1.0     # k22 <- 2g23,2 ; k12 <- 2g13,2

    # unit directions of the 27 LS unknowns [x11,x12,x22, c1(12), c2(12)] (Eq. 58: 3 + 24)
    units = []
    for j in range(27):
        pj = np.zeros(27); pj[j] = 1.0
        units.append((np.array([[pj[0], pj[1]], [pj[1], pj[2]]]),
                      pj[3:15].reshape(2, 6), pj[15:27].reshape(2, 6)))

    jGh = jnp.asarray(Gh_ref); jGl1 = jnp.asarray(Gl1_ref); jGl2 = jnp.asarray(Gl2_ref)
    jE0 = jnp.asarray(_E0); jE1 = jnp.asarray(_E1)
    jdofs = jnp.asarray(dofs_e); jlayer = jnp.asarray(elem_layer); jsub = jnp.asarray(sub_of_elem)
    jkernel_nk = jnp.asarray(kernel_nk)
    jnodes_e = jnp.asarray(nodes_of_e); jS_int = jnp.asarray(S_int_ref)
    jD1_sg = jnp.asarray(D1_sg); jD2_sg = jnp.asarray(D2_sg)
    junits = [tuple(jnp.asarray(u) for u in unit) for unit in units]
    jw_g = jnp.asarray(w_g); jxi_g = jnp.asarray(xi_g)

    def single(thick, C_layers, fraction):
        """The whole ladder for ONE laminate (jit-traced; all jnp arrays).
        In:  thick    (nlay,)     -- ply thicknesses
             C_layers (nlay,6,6)  -- rotated 3-D ply stiffnesses
             fraction scalar      -- reference plane, 0=OML .. 1=IML
        Out: 25-tuple (A6 (6,6), G (2,2), X (2,2), H (12,12), Ustar_rel
             scalar, V0/V11/V12 (ndofs,6), c1/c2 (2,6), ev_min scalar,
             V11bar/V12bar/V11barD/V12barD (ndofs,6), V21/22/23 and
             V21t/22t/23t (ndofs,6), V1Lt/V1Lb (ndofs,), V2Lt/V2Lb
             (ndofs,5)) -- unpacked by _pack."""
        h = jnp.sum(thick)
        z_ref = fraction * h                                   # reference plane (0=OML .. 1=IML)
        he = jnp.repeat(thick / n_per_layer, n_per_layer)      # (n_elem,) element thickness
        layer_bot = jnp.concatenate([jnp.zeros(1), jnp.cumsum(thick)])
        x_left = layer_bot[jlayer] + jsub * he - z_ref         # (n_elem,) element bottom x3
        Ck = C_layers[jlayer]                                  # (n_elem,6,6)

        # nodal x3 grid (== _node_grid) + trapezoid weights for the TILT
        # projection: w -> w - x3 <x3 w>/<x3^2> on the IN-PLANE components.
        # Built here (not in _pack) so the V2 drivers below can use the SAME
        # detilted columns the recovery uses -- the V2 natural BC then cancels
        # the Gamma_l face traction of exactly the columns fed to Gamma_l,
        # keeping the recovered face sigma_33 exact (see _detilt_inplane for
        # the tilt rule itself).
        offs_p = jnp.arange(p) / p
        x_nodes = jnp.concatenate(
            [(x_left[:, None] + he[:, None] * offs_p[None, :]).reshape(-1),
             (x_left[-1] + he[-1])[None]])                     # (nnode,)
        dx_n = x_nodes[1:] - x_nodes[:-1]
        wtr_n = (jnp.zeros_like(x_nodes).at[:-1].add(0.5 * dx_n)
                 .at[1:].add(0.5 * dx_n))
        z2_mom = jnp.sum(wtr_n * x_nodes * x_nodes)

        def detilt_cols(V_ns):
            """Project the x3-linear content out of the in-plane components of a
            warping column block (w3 keeps its tilt).
            In:  V_ns (ndofs, ncols) jnp -- nodal warping columns
            Out: (ndofs, ncols) jnp -- same columns, in-plane tilt removed."""
            Vw = V_ns.reshape(len(x_nodes), 3, -1)
            m1 = jnp.einsum("n,nak->ak", wtr_n * x_nodes, Vw)  # (3, ncols)
            corr = x_nodes[:, None, None] * (m1 / z2_mom)[None, :, :]
            corr = corr.at[:, 2, :].set(0.0)
            return (Vw - corr).reshape(V_ns.shape)

        def elem_D_mats(he_e, xl_e, C_e):
            """Elemental Eq.-30 integrals.
            In:  he_e scalar -- element thickness; xl_e scalar -- element
                 bottom x3; C_e (6,6) jnp -- the ply stiffness
            Out: 10-tuple of local blocks, (3(p+1),3(p+1)) for the _ll pairs,
                 (3(p+1),6) for _ls, (6,6) for D_ee_ss."""
            D_hh_ll = jnp.zeros((3 * (p + 1), 3 * (p + 1)))
            D_he_ls = jnp.zeros((3 * (p + 1), 6)); D_ee_ss = jnp.zeros((6, 6))
            D_hl1_ll = jnp.zeros_like(D_hh_ll); D_hl2_ll = jnp.zeros_like(D_hh_ll)
            D_l1l1_ll = jnp.zeros_like(D_hh_ll); D_l1l2_ll = jnp.zeros_like(D_hh_ll)
            D_l2l2_ll = jnp.zeros_like(D_hh_ll)
            D_l1e_ls = jnp.zeros_like(D_he_ls); D_l2e_ls = jnp.zeros_like(D_he_ls)
            for q in range(len(xi_g)):                          # unrolled Gauss loop
                dw = 0.5 * he_e * jw_g[q]
                Gamma_h_vl = (2.0 / he_e) * jGh[q]              # [Gamma_h S]: d/dx3 strains
                Gamma_l1_vl = jGl1[q]; Gamma_l2_vl = jGl2[q]    # [Gamma_l1 S], [Gamma_l2 S]
                x_q = xl_e + he_e * 0.5 * (1.0 + jxi_g[q])
                Gamma_eps_vs = jE0 + x_q * jE1                  # Gamma_eps: e + x3*k rows
                D_hh_ll += Gamma_h_vl.T @ C_e @ Gamma_h_vl * dw
                D_he_ls += Gamma_h_vl.T @ C_e @ Gamma_eps_vs * dw
                D_ee_ss += Gamma_eps_vs.T @ C_e @ Gamma_eps_vs * dw
                D_hl1_ll += Gamma_h_vl.T @ C_e @ Gamma_l1_vl * dw
                D_hl2_ll += Gamma_h_vl.T @ C_e @ Gamma_l2_vl * dw
                D_l1l1_ll += Gamma_l1_vl.T @ C_e @ Gamma_l1_vl * dw
                D_l1l2_ll += Gamma_l1_vl.T @ C_e @ Gamma_l2_vl * dw
                D_l2l2_ll += Gamma_l2_vl.T @ C_e @ Gamma_l2_vl * dw
                D_l1e_ls += Gamma_l1_vl.T @ C_e @ Gamma_eps_vs * dw
                D_l2e_ls += Gamma_l2_vl.T @ C_e @ Gamma_eps_vs * dw
            return (D_hh_ll, D_he_ls, D_ee_ss, D_hl1_ll, D_hl2_ll,
                    D_l1l1_ll, D_l1l2_ll, D_l2l2_ll, D_l1e_ls, D_l2e_ls)

        # vmapped elemental blocks (the beam-solid pattern), _b = batched over elements
        (D_hh_b, D_he_b, D_ee_b, D_hl1_b, D_hl2_b,
         D_l1l1_b, D_l1l2_b, D_l2l2_b, D_l1e_b, D_l2e_b) = jax.vmap(elem_D_mats)(he, x_left, Ck)

        rows = jdofs[:, :, None]; cols = jdofs[:, None, :]

        def scat_nn(Be):
            """In: Be (n_elem, 3(p+1), 3(p+1)) jnp elemental blocks.
            Out: (ndofs, ndofs) jnp assembled matrix."""
            return jnp.zeros((ndofs, ndofs)).at[rows, cols].add(Be)

        def scat_ns(Be):
            """In: Be (n_elem, 3(p+1), 6) jnp.  Out: (ndofs, 6) jnp."""
            return jnp.zeros((ndofs, 6)).at[jdofs].add(Be)

        # assembled SG matrices, paper Eq. 30 names (dimension suffixes: see module docstring)
        D_hh_nn = scat_nn(D_hh_b)                               # E     = <Gamma_h^T C Gamma_h>
        D_he_ns = scat_ns(D_he_b)                               # D_he  = <Gamma_h^T C Gamma_eps>
        D_ee_ss = jnp.sum(D_ee_b, axis=0)                       # D_ee  = <Gamma_eps^T C Gamma_eps>
        D_hl1_nn = scat_nn(D_hl1_b); D_hl2_nn = scat_nn(D_hl2_b)
        D_l1l1_nn = scat_nn(D_l1l1_b); D_l1l2_nn = scat_nn(D_l1l2_b); D_l2l2_nn = scat_nn(D_l2l2_b)
        D_l1e_ns = scat_ns(D_l1e_b); D_l2e_ns = scat_ns(D_l2e_b)

        # ---- the constraint machinery of Eqs. (31)-(39) ----
        # H = <S^T S> enters only through H @ psi: the through-thickness integration
        # functional (per-node weights int N_a dx3, one copy per warping component).
        w_node = jnp.zeros(n_node).at[jnodes_e].add(0.5 * he[:, None] * jS_int)
        w_dof_n = jnp.repeat(w_node, 3)                        # <.> weights on the dofs
        psi_nk = jkernel_nk / jnp.sqrt(h)                      # Eq. (31): psi^T H psi = I3
        Hpsi_nk = w_dof_n[:, None] * psi_nk                    # H @ psi  (n, k)
        # Eq. (34) solved DIRECTLY as the Lagrange-multiplier saddle-point system:
        #   [[D_hh, H psi], [psi^T H, 0]] [V; Lam] = [-rhs; 0]
        # the multiplier row reproduces Eq. (35) automatically (Lam = -psi^T rhs: zero at
        # zeroth order since Gamma_h of a constant vanishes, = -S_a/sqrt(h) at first
        # order), and the constraint row enforces <w_i> = 0 exactly, so Eq. (39) is
        # built into the solve.  scl balances the two blocks for conditioning only.
        scl = jnp.max(jnp.abs(jnp.diag(D_hh_nn)))
        KKT = jnp.block([[D_hh_nn, scl * Hpsi_nk],
                         [scl * Hpsi_nk.T, jnp.zeros((3, 3))]])
        # ONE factorization shared by all 18 unit load cases (the zeroth- and first-order
        # problems differ only in the forcing).  All three formulations give bit-identical
        # results; timings under the outer laminate vmap (min of paired interleaved trials,
        # shared machine) put this one ~15-25% ahead of two independent multi-RHS solves,
        # with one lu_solve vmapped per load-case column the slowest of the three -- so the
        # case columns are kept as a matrix (level-3 BLAS) rather than vmapped individually.
        lu_piv = lu_factor(KKT)

        def solve_constrained(rhs):
            """Constrained warping solve, all load cases at once: E V = -rhs
            subject to <w_i> = 0 (the shared KKT factorization).
            In:  rhs (ndofs, n_case) jnp -- one column per unit driver
            Out: V (ndofs, n_case) jnp -- the warping columns (multiplier
                 rows stripped)."""
            return lu_solve(lu_piv, jnp.vstack([-rhs, jnp.zeros((3, rhs.shape[1]))]))[:ndofs]

        # zeroth order (Eq. 39-40): V0 columns per unit plate strain; A6 = classical ABD.
        # The multiplier vanishes identically here: kernel^T D_he = <(Gamma_h kernel)^T C
        # Gamma_eps> = 0 because Gamma_h of a through-thickness constant is zero.  So no
        # correction to the forcing is needed at zeroth order (kept as a check below).
        # The CONSTRAINT row is still load-bearing: it selects the unique V0 with <w>=0,
        # and that gauge propagates into D_abar (D_hla @ kernel != 0) and hence into G.
        V0_ns = solve_constrained(D_he_ns)
        A6_ss = D_ee_ss + V0_ns.T @ D_he_ns


        D1bar_ns = (D_hl1_nn - D_hl1_nn.T) @ V0_ns - D_l1e_ns
        D2bar_ns = (D_hl2_nn - D_hl2_nn.T) @ V0_ns - D_l2e_ns

        V1_n2s = solve_constrained(jnp.concatenate([D1bar_ns, D2bar_ns], axis=1))  # 12 at once
        V11_ns = V1_n2s[:, :6]; V12_ns = V1_n2s[:, 6:]

        # gradient energy blocks B, C, D of Eq. (47) -> H = [[B,C],[C^T,D]] over [E,1; E,2]
        H11_ss = V0_ns.T @ D_l1l1_nn @ V0_ns + D1bar_ns.T @ V11_ns
        H12_ss = V0_ns.T @ D_l1l2_nn @ V0_ns + 0.5 * (D1bar_ns.T @ V12_ns + V11_ns.T @ D2bar_ns)
        H22_ss = V0_ns.T @ D_l2l2_nn @ V0_ns + D2bar_ns.T @ V12_ns

        H11_ss = 0.5 * (H11_ss + H11_ss.T); H22_ss = 0.5 * (H22_ss + H22_ss.T)
        H_tt = jnp.block([[H11_ss, H12_ss], [H12_ss.T, H22_ss]])

        S1_is = (jkernel_nk.T @ D1bar_ns)[:2]
        S2_is = (jkernel_nk.T @ D2bar_ns)[:2]
        AD1_sg = A6_ss @ jD1_sg; AD2_sg = A6_ss @ jD2_sg

        def blocks(X_gg, c1_is, c2_is):
            """[[Bhat,Chat],[Chat^T,Dhat]] of Eqs. (57)+(60); its 78 entries
            must -> 0.  DIAGONAL blocks carry the symmetric representative
            c_a^T S_a + S_a^T c_a (never the raw 2 c^T S); the OFF-diagonal
            Chat stays unsymmetrized (real bilinear asymmetry).  Why, and
            what breaks otherwise: README sec. 2.
            In:  X_gg (2,2) jnp shear compliance; c1_is, c2_is (2,6) jnp
                 relaxation constants
            Out: (12,12) jnp residual operator over [eps,1; eps,2]."""
            Bs_ss = H11_ss + AD1_sg @ X_gg @ AD1_sg.T + c1_is.T @ S1_is + S1_is.T @ c1_is
            Cs_ss = H12_ss + AD1_sg @ X_gg @ AD2_sg.T + c1_is.T @ S2_is + S1_is.T @ c2_is
            Ds_ss = H22_ss + AD2_sg @ X_gg @ AD2_sg.T + c2_is.T @ S2_is + S2_is.T @ c2_is
            return jnp.block([[Bs_ss, Cs_ss], [Cs_ss.T, Ds_ss]])

        # linear LS ("78 equations, 27 unknowns", text after Eq. 57): columns by unit probing
        M0_tt = blocks(jnp.zeros((2, 2)), jnp.zeros((2, 6)), jnp.zeros((2, 6)))  # == H exactly
        b0_q = -M0_tt.ravel()
        Amat_qm = jnp.stack([blocks(*unit).ravel() + b0_q for unit in junits], axis=1)
        # Column equilibration: the X-columns scale like A^2 (~1e13) and the c-columns like
        # S (~1e9), a ~1e4 spread that pushes cond(Amat) to ~5e19.  Scaling each column to
        # unit norm brings it to ~8e15.  Only the redundant constants move; X is invariant.
        cs_m = jnp.linalg.norm(Amat_qm, axis=0)
        cs_m = jnp.where(cs_m == 0, 1.0, cs_m)

        U_qm, sig_m, Vt_mm = jnp.linalg.svd(Amat_qm / cs_m, full_matrices=False)
        sig_ok = sig_m > _LS_RCOND * sig_m[0]
        # guard the reciprocal so the discarded branch never holds inf (would poison jax.grad)
        sig_inv = jnp.where(sig_ok, 1.0 / jnp.where(sig_ok, sig_m, 1.0), 0.0)  # Sigma^dagger
        sol_m = (Vt_mm.T @ (sig_inv * (U_qm.T @ b0_q))) / cs_m
        X_gg = jnp.array([[sol_m[0], sol_m[1]], [sol_m[1], sol_m[2]]])
        c1_is = sol_m[3:15].reshape(2, 6); c2_is = sol_m[15:27].reshape(2, 6)
        res_tt = blocks(X_gg, c1_is, c2_is)
        Ustar_rel = jnp.linalg.norm(res_tt) / (jnp.linalg.norm(H_tt) + 1e-30)
        ev_min = jnp.linalg.eigvalsh(X_gg).min()               # SPD gate evaluated by caller
        G_gg = jnp.linalg.inv(X_gg)                            # Eq. (61) transverse-shear G

        # ---- second-order warping V2 (Eq. 64), for the Eq. (65)-(66) recovery only ----
        # Energy collection (README sec. 4):
        #     2 Pi2* = V2^T E V2 + 2 V2^T (Dbar21 e,11 + Dbar22 e,12 + Dbar23 e,22)
        #     Dbar21 = (D_hl1 - D_hl1^T) V1bar1 - D_l1l1 V0
        #     Dbar22 = (D_hl1 - D_hl1^T) V1bar2 + (D_hl2 - D_hl2^T) V1bar1
        #              - (D_l1l2 + D_l1l2^T) V0
        #     Dbar23 = (D_hl2 - D_hl2^T) V1bar2 - D_l2l2 V0
        # V2's energy content is O((h/l)^4): A6/G untouched; V2 exists purely to
        # recover the through-thickness fields at their leading order.
        #
        # V1bar = V1 + kernel c_a (Eq. 58; the c_a genuinely enter: D_hla@kernel != 0)
        c1_ks = jnp.zeros((3, 6)).at[:2].set(c1_is)
        c2_ks = jnp.zeros((3, 6)).at[:2].set(c2_is)
        V11bar_ns = V11_ns + jkernel_nk @ c1_ks
        V12bar_ns = V12_ns + jkernel_nk @ c2_ks
        V11barD_ns = detilt_cols(V11bar_ns)                    # detilted value-use variants
        V12barD_ns = detilt_cols(V12bar_ns)
        # TWO V2 VARIANTS -- the source must MATCH the recovery's Gamma_l columns
        # (else the V2 natural BC misses the Gamma_l face traction by the +-0.73q0
        # tilt mode), and the two consistent pairs split by output row family:
        # D-chain (detilted) serves the in-plane stress rows, T-chain (tilted)
        # the through-thickness rows.  Falsification sweep, gate results and the
        # row-split rationale: README sec. 4.
        D21D_ns = (D_hl1_nn - D_hl1_nn.T) @ V11barD_ns - D_l1l1_nn @ V0_ns
        D22D_ns = ((D_hl1_nn - D_hl1_nn.T) @ V12barD_ns + (D_hl2_nn - D_hl2_nn.T) @ V11barD_ns
                   - (D_l1l2_nn + D_l1l2_nn.T) @ V0_ns)
        D23D_ns = (D_hl2_nn - D_hl2_nn.T) @ V12barD_ns - D_l2l2_nn @ V0_ns
        D21T_ns = (D_hl1_nn - D_hl1_nn.T) @ V11bar_ns - D_l1l1_nn @ V0_ns
        D22T_ns = ((D_hl1_nn - D_hl1_nn.T) @ V12bar_ns + (D_hl2_nn - D_hl2_nn.T) @ V11bar_ns
                   - (D_l1l2_nn + D_l1l2_nn.T) @ V0_ns)
        D23T_ns = (D_hl2_nn - D_hl2_nn.T) @ V12bar_ns - D_l2l2_nn @ V0_ns
        V2_n6s = solve_constrained(jnp.concatenate(
            [D21D_ns, D22D_ns, D23D_ns, D21T_ns, D22T_ns, D23T_ns], axis=1))
        V21_ns = V2_n6s[:, :6]; V22_ns = V2_n6s[:, 6:12]; V23_ns = V2_n6s[:, 12:18]
        V21t_ns = V2_n6s[:, 18:24]; V22t_ns = V2_n6s[:, 24:30]; V23t_ns = V2_n6s[:, 30:]

        # ---- LOAD COLUMNS (Yu Eqs. 29/45/64): the pressure-driven warping ladder ----
        # Unit face traction -> one entry at that face node's w3 dof; TWO columns
        # (top/bottom) so split-face loads compose.  First order: E V1L = -L on the
        # same LU; second order (V2L quintets): q,a drivers (D_hla - D_hla^T) V1L,
        # q,ab drivers -D_lalb V1L.  Signs fixed by the machine-exact face checks.
        # History and rationale: README sec. 5.
        Lt_n = jnp.zeros(ndofs).at[ndofs - 1].set(-1.0)        # top-node w3 dof
        Lb_n = jnp.zeros(ndofs).at[2].set(1.0)                 # bottom-node w3 dof
        V1L_n2 = solve_constrained(jnp.stack([Lt_n, Lb_n], axis=1))
        V1Lt_n = V1L_n2[:, 0]; V1Lb_n = V1L_n2[:, 1]

        def v2l(V1L_n):
            """The five second-order load columns of one face (q,1 q,2 q,11
            q,12 q,22 drivers), Eq.-64 analog with V1L as the lower rung.
            In:  V1L_n (ndofs,) jnp -- that face's first-order load column
            Out: (ndofs, 5) jnp -- the V2L quintet."""
            return solve_constrained(jnp.stack(
                [(D_hl1_nn - D_hl1_nn.T) @ V1L_n,
                 (D_hl2_nn - D_hl2_nn.T) @ V1L_n,
                 -D_l1l1_nn @ V1L_n,
                 -(D_l1l2_nn + D_l1l2_nn.T) @ V1L_n,
                 -D_l2l2_nn @ V1L_n], axis=1))

        V2Lt_n5 = v2l(V1Lt_n)
        V2Lb_n5 = v2l(V1Lb_n)

        # returned in the PAPER symbols (the public dict keys carry no dimension suffixes)
        return (A6_ss, G_gg, X_gg, H_tt, Ustar_rel, V0_ns, V11_ns, V12_ns, c1_is, c2_is,
                ev_min, V11bar_ns, V12bar_ns, V11barD_ns, V12barD_ns,
                V21_ns, V22_ns, V23_ns, V21t_ns, V22t_ns, V23t_ns,
                V1Lt_n, V2Lt_n5, V1Lb_n, V2Lb_n5)

    bk = dict(nlay=nlay, n_per_layer=n_per_layer, p=p, n_elem=n_elem, ndofs=ndofs,
              elem_layer=elem_layer,
              jit_single=jax.jit(single),
              jit_batch=jax.jit(jax.vmap(single, in_axes=(0, 0, None))))
    _BUCKETS[key] = bk
    return bk

def _node_grid(thick: Sequence[float], n_per_layer: int, p: int, z_ref: float) -> np.ndarray:
    """node_x exactly as the recovery expects (reference-plane origin).
    In:  thick (nlay,) float -- ply thicknesses; n_per_layer int; p int;
         z_ref float -- reference-plane offset from the bottom face
    Out: node_x (p*nlay*n_per_layer + 1,) float -- SG node coordinates."""
    nlay = len(thick)
    layer_bot = np.concatenate([[0.0], np.cumsum(thick)])
    n_elem = nlay * n_per_layer
    node_x = np.empty(p * n_elem + 1)
    idx = 0
    for k in range(nlay):
        for s in range(n_per_layer):
            xl = layer_bot[k] + thick[k] * s / n_per_layer
            xr = layer_bot[k] + thick[k] * (s + 1) / n_per_layer
            for j in range(p):
                node_x[p * idx + j] = xl + (xr - xl) * j / p
            idx += 1
    node_x[p * n_elem] = layer_bot[-1]
    return node_x - z_ref


def _detilt_inplane(cols: np.ndarray, node_x: np.ndarray) -> np.ndarray:
    """Project the TILT (x3-linear content) out of the IN-PLANE components of
    a warping-column block: w_a -> w_a - x3 <x3 w_a>/<x3^2>  (w3 keeps its
    tilt).  cols (ndofs, 6), node_x = through-thickness node coordinates.

    The tilt IS the transverse-shear deformation; detilted columns in the
    Gamma_l VALUE terms perform Yu's Eq.-50 R -> eps conversion implicitly,
    so callers pass the RM measures R, never a pre-converted eps.  Full
    per-use rules and validation numbers: README sec. 3."""
    W = np.asarray(cols).reshape(len(node_x), 3, 6).copy()
    order = np.argsort(node_x)
    xs = np.asarray(node_x)[order]
    z2 = np.trapezoid(xs * xs, xs)
    for comp in (0, 1):
        m1 = np.trapezoid(xs[:, None] * W[order, comp, :], xs, axis=0)
        W[:, comp, :] -= np.outer(np.asarray(node_x), m1 / z2)
    return W.reshape(-1, 6)


def _pack(out: Sequence, thick: Sequence[float], angles_deg: Sequence[float], C_layers,
          n_per_layer: int, p: int, fraction: float, elem_layer: np.ndarray) -> Dict[str, Any]:
    """Unpack the kernel's 25-tuple into the public result dict.
    In:  out -- single()'s return (see its docstring); thick/angles_deg per
         ply; C_layers (nlay,6,6); n_per_layer/p int; fraction float;
         elem_layer (n_elem,) int
    Out: the rm_plate_msg dict (paper-symbol keys; see rm_plate_msg)."""
    (A6, G, X, H, Ustar_rel, V0, V11, V12, c1, c2, ev_min,
     V11b, V12b, V11bD, V12bD, V21, V22, V23, V21t, V22t, V23t,
     V1Lt, V2Lt, V1Lb, V2Lb) = [np.asarray(o) for o in out]
    thick = [float(t) for t in thick]
    spd = float(ev_min) > 0
    # the full RM plate law, Eqs. (40) + (61):  ABDG = [[A6, 0], [0, G]]
    # (rows/cols: e11, e22, g12, k11, k22, k12, 2g13, 2g23; None if X is not SPD)
    ABDG = None
    if spd:
        ABDG = np.zeros((8, 8))
        ABDG[:6, :6] = A6
        ABDG[6:, 6:] = G
    node_x = _node_grid(thick, n_per_layer, p, float(fraction) * sum(thick))
    # Ustar_rel (the U*-fit residual) is deliberately NOT in the public dict
    # (removed 2026-08-04 on request): it is a fit diagnostic, not a model
    # quantity -- consumers that reported it now compute or skip it.
    return {"A6": A6, "G_msg": (G if spd else None), "ABDG": ABDG, "X": X, "H": H,
            "V0": V0, "V11": V11, "V12": V12,
            "V11bar": V11b, "V12bar": V12b, "V21": V21, "V22": V22, "V23": V23,
            # the TILTED-chain V2 variants (through-thickness rows; see the
            # source-block comment in _bucket.single)
            "V21t": V21t, "V22t": V22t, "V23t": V23t,
            # detilted VALUE-use variants (Eq. 65 / the Gamma_l terms of Eq. 66;
            # see _detilt_inplane for the tilt rule) -- computed INSIDE the
            # kernel so the V2 drivers are built on the identical projection
            "V11barD": V11bD, "V12barD": V12bD,
            # the LOAD ladder, per face: V1L (ndofs,) per unit face pressure and
            # V2L (ndofs, 5) for its (q,1 q,2 q,11 q,12 q,22) gradients
            "V1Lt": V1Lt, "V2Lt": V2Lt, "V1Lb": V1Lb, "V2Lb": V2Lb,
            "node_x": node_x,
            "elem_layer": elem_layer, "C_layers": list(C_layers), "elem_order": p,
            "angles": [float(a) for a in angles_deg], "c1": c1, "c2": c2}


def rm_plate_msg(thick: Sequence[float],            # ply thicknesses, bottom first
                 angles_deg: Sequence[float],       # ply angles [deg]
                 mat_names: Sequence[str],          # material_db key per ply
                 material_db: Dict[str, dict],      # {name: {E, G, nu, [rho]}}
                 n_per_layer: int = 1,              # SG elements per ply
                 elem_order: int = 4,               # 4 = 5-noded quartic (exact ladder)
                 fraction: float = 0.0              # reference plane, 0=OML..1=IML
                 ) -> Dict[str, Any]:               # the plate-law + warping dict
    """Build the MSG-RM plate law for ONE laminate.

    Returns a dict: A6 (6x6 ABD), G_msg (2x2, None if not SPD), ABDG (8x8
    [[A6,0],[0,G]]), X = G^-1, H (12x12), the warping ladders
    (V0, V11/V12 + bar/barD variants, V21/22/23 + t variants, V1L/V2L per
    face), node_x, elem_layer, C_layers, elem_order, angles, c1, c2.

    elem_order = 4 represents the whole warping ladder exactly (degrees
    V0: 2, V1: 3, V2: 4); exactness numbers and performance notes: README
    sec. 1.  Batch many laminates with ``rm_plate_msg_batch``."""
    C_layers = np.array([rotated_stiffness_6x6(material_db[mat_names[k]]['E'],
                                               material_db[mat_names[k]]['G'],
                                               material_db[mat_names[k]]['nu'],
                                               angles_deg[k]) for k in range(len(thick))])
    bk = _bucket(len(thick), n_per_layer, elem_order)
    out = bk["jit_single"](jnp.asarray(np.asarray(thick, float)), jnp.asarray(C_layers),
                           jnp.asarray(float(fraction)))
    return _pack(out, thick, angles_deg, C_layers, n_per_layer, int(elem_order), fraction,
                 bk["elem_layer"])


def rm_plate_msg_batch(layups: List[Dict[str, Any]],    # [{mat_names, thick, angles}]
                       material_db: Dict[str, dict],
                       n_per_layer: int = 1, elem_order: int = 4,
                       fraction: float = 0.0
                       ) -> List[Dict[str, Any]]:       # rm_plate_msg dicts, input order
    """Vmapped fast path: MSG-RM plate law for MANY laminates at once.

    ``layups`` = list of dicts with keys mat_names / thick / angles (the ``layup_db``
    values of ``msg_mesh.load_yaml``).  Laminates are grouped into (n_ply) buckets --
    the hex/tet element-type-batch analog -- one jitted vmap call per bucket.
    Returns a list of ``rm_plate_msg`` dicts in input order."""
    groups = {}
    for i, l in enumerate(layups):
        groups.setdefault(len(l["thick"]), []).append(i)
    results = [None] * len(layups)
    for nlay, idxs in groups.items():
        thick_b = np.array([[float(t) for t in layups[i]["thick"]] for i in idxs])
        C_b = np.array([[rotated_stiffness_6x6(material_db[m]['E'], material_db[m]['G'],
                                               material_db[m]['nu'], float(a))
                         for m, a in zip(layups[i]["mat_names"], layups[i]["angles"])]
                        for i in idxs])
        bk = _bucket(nlay, n_per_layer, elem_order)
        outs = bk["jit_batch"](jnp.asarray(thick_b), jnp.asarray(C_b),
                               jnp.asarray(float(fraction)))
        for bi, i in enumerate(idxs):
            out_i = [o[bi] for o in outs]
            results[i] = _pack(out_i, layups[i]["thick"], layups[i]["angles"], C_b[bi],
                               n_per_layer, int(elem_order), fraction, bk["elem_layer"])
    return results


def _locate(obj: Dict[str, Any], z: float) -> Tuple[int, float, float, np.ndarray, np.ndarray, float]:
    """Locate depth z in the SG mesh.
    In:  obj dict -- the rm_plate_msg result; z float -- depth (node_x origin)
    Out: (e int element index, xi float local coord [-1,1], he float element
         thickness, nodes_xi (p+1,) float, dofs (3(p+1),) int element dof
         ids, x_q float -- the physical x3 of xi)."""
    node_x = obj["node_x"]; p = obj["elem_order"]
    n_elem = len(obj["elem_layer"])
    e = int(np.clip(np.searchsorted(node_x[::p][1:], z, side="right"), 0, n_elem - 1))
    xl = node_x[p * e]; xr = node_x[p * e + p]; he = xr - xl
    xi = np.clip(2.0 * (z - xl) / he - 1.0, -1.0, 1.0)
    nodes_xi = np.linspace(-1.0, 1.0, p + 1)
    dofs = np.arange(3 * p * e, 3 * p * e + 3 * (p + 1))
    return e, xi, he, nodes_xi, dofs, 0.5 * (xl + xr) + 0.5 * he * xi


def _warp_terms(obj: Dict[str, Any], dofs: np.ndarray, E6: np.ndarray,
                dE1: np.ndarray, dE2: np.ndarray, dE11: np.ndarray,
                dE12: np.ndarray, dE22: np.ndarray,
                qt6: Optional[np.ndarray] = None,
                qb6: Optional[np.ndarray] = None
                ) -> Tuple[np.ndarray, ...]:
    """Element-local warping (Gamma_h use) and Gamma_l arguments, BOTH chains:

    w   = V0 e + V1bar e,a + V2 e,ab + V1L q + V2L (q,a; q,ab)
    g_a = V0 e,a + V1bar(D) e,ab + V1L q,a

    Returns (w_loc, g1, g2) of the detilted chain + (w_t, g1t, g2t) of the
    tilted chain (the row-split consumer is msgrm_strain_at_depth).  Column
    variants per use -- raw V1bar in Gamma_h, detilted in the D-chain
    Gamma_l -- and the load-ladder semantics: README secs. 3-5."""
    w_loc = (obj["V0"][dofs] @ E6 + obj["V11bar"][dofs] @ dE1 + obj["V12bar"][dofs] @ dE2
             + obj["V21"][dofs] @ dE11 + obj["V22"][dofs] @ dE12 + obj["V23"][dofs] @ dE22)
    g1 = obj["V0"][dofs] @ dE1 + obj["V11barD"][dofs] @ dE11 + obj["V12barD"][dofs] @ dE12
    g2 = obj["V0"][dofs] @ dE2 + obj["V11barD"][dofs] @ dE12 + obj["V12barD"][dofs] @ dE22
    # the TILTED chain (through-thickness rows 33/23/13 of the second-order
    # recovery): V1bar in Gamma_l, matched by the V2*t source columns
    w_t = (obj["V0"][dofs] @ E6 + obj["V11bar"][dofs] @ dE1 + obj["V12bar"][dofs] @ dE2
           + obj["V21t"][dofs] @ dE11 + obj["V22t"][dofs] @ dE12
           + obj["V23t"][dofs] @ dE22)
    g1t = obj["V0"][dofs] @ dE1 + obj["V11bar"][dofs] @ dE11 + obj["V12bar"][dofs] @ dE12
    g2t = obj["V0"][dofs] @ dE2 + obj["V11bar"][dofs] @ dE12 + obj["V12bar"][dofs] @ dE22
    for q6, v1, v2 in ((qt6, "V1Lt", "V2Lt"), (qb6, "V1Lb", "V2Lb")):
        if q6 is not None:
            q6 = np.asarray(q6, float)
            w_loc = (w_loc + obj[v1][dofs] * q6[0]
                     + obj[v2][dofs] @ q6[1:])
            w_t = (w_t + obj[v1][dofs] * q6[0]
                   + obj[v2][dofs] @ q6[1:])
            g1 = g1 + obj[v1][dofs] * q6[1]
            g2 = g2 + obj[v1][dofs] * q6[2]
            g1t = g1t + obj[v1][dofs] * q6[1]
            g2t = g2t + obj[v1][dofs] * q6[2]
    return w_loc, g1, g2, w_t, g1t, g2t


def msgrm_strain_at_depth(obj: Dict[str, Any],          # the rm_plate_msg result
                          z: float,                     # depth, node_x origin
                          E6: np.ndarray,               # plate strains (RM measures R)
                          dE1: Optional[np.ndarray] = None,   # E6,1
                          dE2: Optional[np.ndarray] = None,   # E6,2
                          dE11: Optional[np.ndarray] = None,  # E6,11 (2nd order on)
                          dE12: Optional[np.ndarray] = None,  # E6,12
                          dE22: Optional[np.ndarray] = None,  # E6,22
                          qt6: Optional[np.ndarray] = None,   # top-face load ladder
                          qb6: Optional[np.ndarray] = None,   # bottom-face ladder
                          frame: str = "material"             # "material" | "plate"
                          ) -> Tuple[np.ndarray, np.ndarray, float]:
    """3-D Voigt strain + stress at depth z (same origin as node_x).

    dE1/dE2 only -> first-order recovery, Eq. (63).  dE11/dE12/dE22 as well
    -> second-order Eq. (66), ROW-SPLIT between the two V2 chains (in-plane
    stress rows from the detilted chain, rows 33/23/13 from the tilted one).
    qt6/qb6 = [q, q,1, q,2, q,11, q,12, q,22] of the top/bottom FACE
    pressures: with them sigma33 is constitutive with machine-exact faces.

    frame="material" (DEFAULT -- failure criteria live there): Gam/Sig are
    rotated into the ply axes of the layer containing z, Sig_mat =
    rotation_6x6(-angle) @ Sig, Gam_mat = rotation_6x6(angle).T @ Gam
    (engineering shear).  frame="plate": the laminate/SG frame -- REQUIRED
    by anything that integrates or differentiates across plies (build_ops
    operators, Q-consistency rescale, momentum/equilibrium integrals,
    comparisons against global-frame exact/FE fields).  sigma33/eps33 are
    invariant either way.
    Returns (Gam6, Sig6, ply_angle_deg).  Conventions: README secs. 3-5."""
    zeros = np.zeros(6)
    dE1 = zeros if dE1 is None else np.asarray(dE1, float)
    dE2 = zeros if dE2 is None else np.asarray(dE2, float)
    dE11 = zeros if dE11 is None else np.asarray(dE11, float)
    dE12 = zeros if dE12 is None else np.asarray(dE12, float)
    dE22 = zeros if dE22 is None else np.asarray(dE22, float)
    e, xi, he, nodes_xi, dofs, x_q = _locate(obj, z)
    Gamma_h = _plate_B(nodes_xi, xi, he)
    Gamma_l1, Gamma_l2 = _grad_ops(nodes_xi, xi)
    Gamma_eps = _E0 + x_q * _E1
    w_loc, g1, g2, w_t, g1t, g2t = _warp_terms(obj, dofs, E6, dE1, dE2,
                                               dE11, dE12, dE22,
                                               qt6=qt6, qb6=qb6)
    Gam = Gamma_h @ w_loc + Gamma_eps @ E6 + Gamma_l1 @ g1 + Gamma_l2 @ g2
    k = obj["elem_layer"][e]
    C = np.asarray(obj["C_layers"][k])
    Sig = C @ Gam
    if np.any(dE11) or np.any(dE12) or np.any(dE22):
        # ROW SPLIT of the second-order recovery: the through-thickness
        # stress rows (33, 23, 13 = Voigt 2, 3, 4) come from the TILTED
        # chain, whose (source, Gamma_l) pair keeps the face tractions
        # exact AND the cross-ply interior sigma33 correct; the in-plane
        # rows keep the detilted chain (raw tilt would double-count z*phi
        # there).  See the source-block comment in _bucket.single.
        Gam_t = Gamma_h @ w_t + Gamma_eps @ E6 + Gamma_l1 @ g1t + Gamma_l2 @ g2t
        Sig_t = C @ Gam_t
        Sig = Sig.copy(); Gam = Gam.copy()
        Sig[2:5] = Sig_t[2:5]
        Gam[2:5] = Gam_t[2:5]
    ang = obj["angles"][k]
    if frame == "material":
        Sig = rotation_6x6(-ang) @ Sig
        Gam = rotation_6x6(ang).T @ Gam
    elif frame != "plate":
        raise ValueError("frame must be 'material' or 'plate', got %r" % (frame,))
    return Gam, Sig, ang


def msgrm_strain_at_depth_batch(obj: Dict[str, Any],       # rm_plate_msg result
                                z_n: np.ndarray,           # (n,) depths
                                E6_n: np.ndarray,          # (n,6) plate strains
                                dE1_n=None, dE2_n=None,    # (n,6) first grads
                                dE11_n=None, dE12_n=None,  # (n,6) second grads
                                dE22_n=None,
                                frame: str = "material"
                                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """VECTORIZED msgrm_strain_at_depth: n points of ONE laminate at once.

    Same algebra as the scalar function, evaluated with array shape ops --
    the point loop, the per-point element search and the per-point operator
    builds all collapse into batched einsums.  Use it whenever many points
    share a layup but sit at different depths (the shell -> plate dehom of a
    ring: ~1e5 recovery points over a handful of layups).

    In:  obj   the rm_plate_msg dict; z_n (n,) depths in the node_x origin;
         E6_n (n,6) plate strains; dE1_n/dE2_n (n,6) first gradients (None ->
         zero); dE11_n/dE12_n/dE22_n (n,6) second gradients (None -> zero,
         which selects the Eq.-63 first-order recovery exactly as the scalar
         call does); frame "material" (ply axes, default) | "plate"
    Out: (Gam (n,6), Sig (n,6), ply_angle (n,)) -- identical to looping the
         scalar function, to solver precision (gated in the test suite)."""
    z_n = np.atleast_1d(np.asarray(z_n, float))
    n = len(z_n)
    zero_n = np.zeros((n, 6))
    E6_n = np.asarray(E6_n, float).reshape(n, 6)
    dE1, dE2 = (zero_n if a is None else np.asarray(a, float).reshape(n, 6)
                for a in (dE1_n, dE2_n))
    d11, d12, d22 = (zero_n if a is None else np.asarray(a, float).reshape(n, 6)
                     for a in (dE11_n, dE12_n, dE22_n))
    second = bool(np.any(d11) or np.any(d12) or np.any(d22))

    node_x = np.asarray(obj["node_x"], float)
    p = int(obj["elem_order"])
    n_elem = len(obj["elem_layer"])
    npn = p + 1
    nodes_xi = np.linspace(-1.0, 1.0, npn)

    # ---- locate every depth at once (the vector form of _locate) ----
    e_n = np.clip(np.searchsorted(node_x[::p][1:], z_n, side="right"),
                  0, n_elem - 1)
    xl = node_x[p * e_n]
    xr = node_x[p * e_n + p]
    he = xr - xl
    xi = np.clip(2.0 * (z_n - xl) / he - 1.0, -1.0, 1.0)
    x_q = 0.5 * (xl + xr) + 0.5 * he * xi
    dofs = (3 * p * e_n)[:, None] + np.arange(3 * npn)[None, :]   # (n, 3npn)

    # ---- Lagrange values and derivatives at every xi ----
    N = np.ones((n, npn))
    for i in range(npn):
        for j in range(npn):
            if j != i:
                N[:, i] *= (xi - nodes_xi[j]) / (nodes_xi[i] - nodes_xi[j])
    dN = np.zeros((n, npn))
    for i in range(npn):
        s = np.zeros(n)
        for j in range(npn):
            if j == i:
                continue
            term = np.full(n, 1.0 / (nodes_xi[i] - nodes_xi[j]))
            for m in range(npn):
                if m == i or m == j:
                    continue
                term = term * ((xi - nodes_xi[m])
                               / (nodes_xi[i] - nodes_xi[m]))
            s += term
        dN[:, i] = s
    dNx = dN * (2.0 / he)[:, None]                 # the _plate_B scaling

    # Per-ELEMENT warping blocks, built once and indexed by e_n: a 1-D take
    # over n_elem small blocks instead of an (n, 3npn) fancy index per call.
    _blk = {}

    def _block(col):
        B = _blk.get(col)
        if B is None:
            V = np.asarray(obj[col], float)
            B = np.stack([V[3 * p * e: 3 * p * e + 3 * npn]
                          for e in range(n_elem)])       # (n_elem, 3npn, ncol)
            _blk[col] = B
        return B

    def warp(col, drv):
        """(n, npn, 3) nodal warping of column `col` contracted with `drv`."""
        V = _block(col)[e_n]                       # (n, 3npn, ncol)
        return np.einsum("nkc,nc->nk", V, drv).reshape(n, npn, 3)

    def chain(v1a, v1b, tilt_cols):
        """w, g1, g2 of one V1 variant (raw = tilted, barD = detilted)."""
        w = (warp("V0", E6_n) + warp(v1a, dE1) + warp(v1b, dE2)
             + warp(tilt_cols[0], d11) + warp(tilt_cols[1], d12)
             + warp(tilt_cols[2], d22))
        g1 = warp("V0", dE1) + warp(v1a, d11) + warp(v1b, d12)
        g2 = warp("V0", dE2) + warp(v1a, d12) + warp(v1b, d22)
        return w, g1, g2

    def strain(w, g1, g2):
        """Gamma = Gamma_h w + Gamma_eps E6 + Gamma_l1 g1 + Gamma_l2 g2."""
        G = np.zeros((n, 6))
        G[:, 2] = np.einsum("nk,nk->n", dNx, w[:, :, 2])      # eps33 <- w3,3
        G[:, 3] = np.einsum("nk,nk->n", dNx, w[:, :, 1])      # 2g23  <- w2,3
        G[:, 4] = np.einsum("nk,nk->n", dNx, w[:, :, 0])      # 2g13  <- w1,3
        G[:, 0] += np.einsum("nk,nk->n", N, g1[:, :, 0])      # eps11 <- w1,1
        G[:, 4] += np.einsum("nk,nk->n", N, g1[:, :, 2])      # 2g13  <- w3,1
        G[:, 5] += np.einsum("nk,nk->n", N, g1[:, :, 1])      # g12   <- w2,1
        G[:, 1] += np.einsum("nk,nk->n", N, g2[:, :, 1])      # eps22 <- w2,2
        G[:, 3] += np.einsum("nk,nk->n", N, g2[:, :, 2])      # 2g23  <- w3,2
        G[:, 5] += np.einsum("nk,nk->n", N, g2[:, :, 0])      # g12   <- w1,2
        G += E6_n @ _E0.T + x_q[:, None] * (E6_n @ _E1.T)     # Gamma_eps E6
        return G

    # detilted chain (in-plane rows) -- V11bar/V12bar in Gamma_h, barD in Gamma_l
    w_d, g1_d, g2_d = chain("V11bar", "V12bar", ("V21", "V22", "V23"))
    g1_d = (warp("V0", dE1) + warp("V11barD", d11) + warp("V12barD", d12))
    g2_d = (warp("V0", dE2) + warp("V11barD", d12) + warp("V12barD", d22))
    Gam = strain(w_d, g1_d, g2_d)

    k_n = np.asarray(obj["elem_layer"])[e_n]
    C = np.asarray(obj["C_layers"], float)[k_n]                # (n, 6, 6)
    Sig = np.einsum("nij,nj->ni", C, Gam)

    if second:
        # ROW SPLIT: rows 33/23/13 come from the TILTED chain (see the scalar
        # function and README sec. 4 -- the source/Gamma_l pairing that keeps
        # the face tractions exact and the cross-ply interior sigma33 right)
        w_t, g1_t, g2_t = chain("V11bar", "V12bar", ("V21t", "V22t", "V23t"))
        Gam_t = strain(w_t, g1_t, g2_t)
        Sig_t = np.einsum("nij,nj->ni", C, Gam_t)
        Gam[:, 2:5] = Gam_t[:, 2:5]
        Sig[:, 2:5] = Sig_t[:, 2:5]

    ang = np.asarray(obj["angles"], float)[k_n]
    if frame == "material":
        th = np.deg2rad(ang)
        c, s = np.cos(th), np.sin(th)
        Rs = np.zeros((n, 6, 6)); Re = np.zeros((n, 6, 6))
        for i, (cc, ss) in enumerate(zip(c, s)):
            Rs[i] = rotation_6x6(-ang[i])
            Re[i] = rotation_6x6(ang[i]).T
        Sig = np.einsum("nij,nj->ni", Rs, Sig)
        Gam = np.einsum("nij,nj->ni", Re, Gam)
    elif frame != "plate":
        raise ValueError("frame must be 'material' or 'plate', got %r" % (frame,))
    return Gam, Sig, ang


def msgrm_warping_at_depth(obj: Dict[str, Any], z: float, E6: np.ndarray,
                           dE1: Optional[np.ndarray] = None,
                           dE2: Optional[np.ndarray] = None,
                           dE11: Optional[np.ndarray] = None,
                           dE12: Optional[np.ndarray] = None,
                           dE22: Optional[np.ndarray] = None,
                           qt6: Optional[np.ndarray] = None,
                           qb6: Optional[np.ndarray] = None) -> np.ndarray:
    """The 3-D warping displacement at depth z -- the SG part of the Eq.-(65)
    recovery; the caller composes  U_a = u_a^2d - x3 w,a + w_a,  U3 = w + w3
    (the KIRCHHOFF z-term, never x3*phi, and the MEAN-ZERO columns V11/V12,
    never the relaxed ones -- both rules and their validation: README sec. 3).
    qt6/qb6 as in msgrm_strain_at_depth.  Returns w (3,), ALWAYS in the
    plate/laminate frame -- warping composes with the GLOBAL plate
    displacement, so a per-ply frame would make U discontinuous; rotate the
    composed U afterwards if a material-frame displacement is wanted."""
    zeros = np.zeros(6)
    dE1 = zeros if dE1 is None else np.asarray(dE1, float)
    dE2 = zeros if dE2 is None else np.asarray(dE2, float)
    dE11 = zeros if dE11 is None else np.asarray(dE11, float)
    dE12 = zeros if dE12 is None else np.asarray(dE12, float)
    dE22 = zeros if dE22 is None else np.asarray(dE22, float)
    e, xi, he, nodes_xi, dofs, x_q = _locate(obj, z)
    w_loc = (obj["V0"][dofs] @ E6
             + obj["V11"][dofs] @ dE1 + obj["V12"][dofs] @ dE2
             + obj["V21"][dofs] @ dE11 + obj["V22"][dofs] @ dE12
             + obj["V23"][dofs] @ dE22)
    for q6, v1, v2 in ((qt6, "V1Lt", "V2Lt"), (qb6, "V1Lb", "V2Lb")):
        if q6 is not None:
            q6 = np.asarray(q6, float)
            w_loc = w_loc + obj[v1][dofs] * q6[0] + obj[v2][dofs] @ q6[1:]
    N = _lagrange_N(nodes_xi, xi)
    w_nodes = w_loc.reshape(-1, 3)                    # (p+1, 3) nodal warping in the element
    return N @ w_nodes


