"""
RM Timoshenko (V1) step: produce the 6x6 stiffness (incl. C22, C33) from the
RM C0 cross-section model, reusing the validated msg_solver condensation.

Pipeline (mirrors the Kirchhoff path):
  EB:   Dhh,Dhe,Dee  + constraints C, rigid kernel Psi  -> V0, D_eff(4x4)
  V1:   eps_l (shear-warping) -> Dhl,Dll,Dle -> V1 -> 6x6 [EA,GA12,GA13,GJ,EI2,EI3]

RM rigid kernel (Psi, derived from eq 4.23, zero-strain modes):
  3 translations (w1,w2,w3=const) + twist (w2=-y3, w3=y2, omega1=-1).
Constraints C: <w1>=<w2>=<w3>=<omega1>=0 (conjugate to Psi).
"""
import os, sys
import numpy as np
from scipy.sparse import coo_matrix
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from .msg_solver import (solve_fluctuation_field, prepare_v1_rhs,
                               finalize_v1_and_compute_deff)
from .msg_rm import _lagrange, _shape, _macro_BD, _macro_BG


def _elem_BD_BG_BL(nodes_xi, xi, X, dN_unused, k22, p, g13mode="omega"):
    """Return BDq(6,5n), BGq(2,5n), BLq(6,5n) and geometry at xi.

    g13mode controls the 2*eps13 (BGq row 0) operator:
      'omega' (default) -- 2*eps13 = omega2 (algebraic; the validated baseline).
      'eq12'            -- 2*eps13 = -wdot1*xdot3/(2 xdot2) + omega2/xdot2  (the full Eq.12
                           warping-derivative form, for the tie-both 'mitc_both' scheme).
    """
    N, dNr = _shape(nodes_xi, xi)
    dxds = dNr @ X; Jac = np.hypot(*dxds); dN = dNr / Jac
    x2, x3 = N @ X; t2, t3 = dxds / Jac; n2, n3 = t3, -t2
    t2f = t2 if abs(t2) > 1e-3 else (1e-3 if t2 >= 0 else -1e-3)   # floor 1/xdot2 (vertical walls)
    npn = p + 1
    BDq = np.zeros((6, 5*npn)); BGq = np.zeros((2, 5*npn)); BLq = np.zeros((6, 5*npn))
    for a in range(npn):
        o = 5*a
        # EB warping strain (eps_h)
        BDq[1, o+1] += t2*dN[a]; BDq[1, o+2] += t3*dN[a]
        BDq[2, o+0] += dN[a]
        BDq[4, o+3] += dN[a]   # kappa22 = +dN(omega1): curvature of the rotation fluctuation
        # (was -dN; the wrong sign made the closed-tube shear-bend coupling over-count +66%)
        BDq[5, o+4] += dN[a]; BDq[5, o+0] += 0.5*k22*dN[a]
        if g13mode == "omega":
            BGq[0, o+4] += N[a]                                   # 2 eps13 = omega2 (baseline)
        else:                                                    # 'eq12': add the warping-derivative term
            BGq[0, o+0] += -(t3/(2.0*t2f))*dN[a]                  # -wdot1 * xdot3/(2 xdot2)
            BGq[0, o+4] += N[a]/t2f                               # + omega2/xdot2
        BGq[1, o+1] += n2*dN[a]; BGq[1, o+2] += n3*dN[a]; BGq[1, o+3] += -N[a]
        # shear-warping operator (eps_l), Kirchhoff-style on the w DOFs
        BLq[0, o+0] += N[a]                                   # eps11 = w1
        BLq[2, o+1] += t2*N[a]; BLq[2, o+2] += t3*N[a]        # 2eps12 = t.w
        BLq[5, o+1] += 2*t3*dN[a] - 0.5*k22*t2*N[a]           # kappa12
        BLq[5, o+2] += -2*t2*dN[a] - 0.5*k22*t3*N[a]
    return BDq, BGq, BLq, (x2, x3, t2, t3, Jac)


def assemble_all(nodes, elems, layup_per_elem, D_by, G_by, k22_e, p=1, reduced=True,
                 shear="mitc"):
    """shear: transverse-shear (G-energy) integration scheme.
      'mitc'    -- SELECTIVE assumed-strain (default, the soft-core fix): full-integrate the
                   NON-locking gamma13=omega2 row, assumed-strain (tying point xi=0 for p=1,
                   +-1/sqrt(3) for p=2) the locking-prone gamma23=n.dw/ds-omega1 row.
                   See docs/MITC_transverse_shear.md (Dvorkin-Bathe MITC / Prathap field-consistency).
      'reduced' -- legacy uniform reduced integration of the whole G-energy (leaves the omega2
                   antisymmetric mode unpenalized -> soft-core hourglass that over-softens GA2).
      'full'    -- full integration of the whole G-energy (locks thin walls)."""
    Nn = len(nodes); ndof = 5*Nn
    nodes_xi = _lagrange(p)
    g13mode = "eq12" if shear == "mitc_both" else "omega"   # mitc_both: eps13 carries the Eq.12 deriv
    tie_both = (shear == "mitc_both")                       # and is tied too (both shear rows assumed)
    is_mitc = shear in ("mitc", "mitc_both")
    Dhh = np.zeros((ndof, ndof)); Dhe = np.zeros((ndof, 4)); Dee = np.zeros((4, 4))
    Dhl = np.zeros((ndof, ndof)); Dll = np.zeros((ndof, ndof)); Dle = np.zeros((ndof, 4))
    Dhh_mem = np.zeros((ndof, ndof))   # membrane/bending-only fluctuation stiffness (NO transverse-shear G)
    xgD, wgD = np.polynomial.legendre.leggauss(p+1)
    xgG, wgG = np.polynomial.legendre.leggauss(max(1, p))
    for e, el in enumerate(elems):
        X = nodes[el]; k22 = float(k22_e[e]); ln = layup_per_elem[e]
        D = D_by[ln]; G = G_by[ln]
        g = np.concatenate([[5*nd, 5*nd+1, 5*nd+2, 5*nd+3, 5*nd+4] for nd in el])
        # D-energy (full int) + eps_l matrices
        for xi, w in zip(xgD, wgD):
            BDq, BGq, BLq, geo = _elem_BD_BG_BL(nodes_xi, xi, X, None, k22, p)
            x2, x3, t2, t3, Jac = geo; dl = Jac*w
            BDe = _macro_BD(x2, x3, t2, t3, k22)
            kdd = BDq.T @ D @ BDq * dl
            Dhh[np.ix_(g, g)] += kdd
            Dhh_mem[np.ix_(g, g)] += kdd          # membrane/bending part (shared with Dhh)
            Dhe[g] += BDq.T @ D @ BDe * dl
            Dee += BDe.T @ D @ BDe * dl
            Dhl[np.ix_(g, g)] += BDq.T @ D @ BLq * dl
            Dll[np.ix_(g, g)] += BLq.T @ D @ BLq * dl
            Dle[g] += BLq.T @ D @ BDe * dl
        # ---- G-energy: transverse shear ----
        if is_mitc:
            # SELECTIVE assumed-strain. gamma13 = omega2 (BGq row 0) is algebraic in the DOF and
            # does NOT lock -> integrate it FULLY (reduced int leaves the omega2 antisymmetric mode
            # unpenalised -> the soft-core hourglass). gamma23 = n.dw/ds - omega1 (BGq row 1) IS
            # locking-prone -> sample it at the Barlow/tying points and re-interpolate (MITC),
            # then full-integrate.  (Dvorkin-Bathe 1984/86; Prathap field-consistency.)
            if p == 1:
                ty = [0.0]; Nas = lambda xi: (1.0,)                       # constant assumed gamma23
            else:
                aa = 1.0/np.sqrt(3.0); ty = [-aa, aa]
                Nas = lambda xi: ((1.0 - np.sqrt(3.0)*xi)/2.0, (1.0 + np.sqrt(3.0)*xi)/2.0)
            BG13 = []; BG23 = []
            for xt in ty:
                _, BGq_t, _, _ = _elem_BD_BG_BL(nodes_xi, xt, X, None, k22, p, g13mode)
                BG13.append(BGq_t[0:1, :].copy()); BG23.append(BGq_t[1:2, :].copy())   # both rows at tying pt
            for xi, w in zip(xgD, wgD):                                   # FULL integration
                BDq, BGq, BLq, geo = _elem_BD_BG_BL(nodes_xi, xi, X, None, k22, p, g13mode)
                x2, x3, t2, t3, Jac = geo; dl = Jac*w
                Nk = Nas(xi)
                BG23bar = sum(Nk[k]*BG23[k] for k in range(len(ty)))     # assumed gamma23
                if tie_both:                                             # mitc_both: tie gamma13 too
                    BG13bar = sum(Nk[k]*BG13[k] for k in range(len(ty)))
                    BGb = np.vstack([BG13bar, BG23bar])                  # both rows assumed
                else:
                    BGb = np.vstack([BGq[0:1, :], BG23bar])              # row0 full, row1 assumed
                BGe = _macro_BG(x2, x3, t2, t3)
                Dhh[np.ix_(g, g)] += BGb.T @ G @ BGb * dl
                Dhe[g] += BGb.T @ G @ BGe * dl
                Dee += BGe.T @ G @ BGe * dl
        else:
            for xi, w in zip((xgG, xgD)[shear == "full"], (wgG, wgD)[shear == "full"]):
                BDq, BGq, BLq, geo = _elem_BD_BG_BL(nodes_xi, xi, X, None, k22, p)
                x2, x3, t2, t3, Jac = geo; dl = Jac*w
                BGe = _macro_BG(x2, x3, t2, t3)
                Dhh[np.ix_(g, g)] += BGq.T @ G @ BGq * dl
                Dhe[g] += BGq.T @ G @ BGe * dl
                Dee += BGe.T @ G @ BGe * dl
    return Dhh, Dhe, Dee, Dhl, Dll, Dle, Dhh_mem


def build_C_Psi(nodes, elems, p=1, w2null=False):
    """4 rigid-kernel modes (w1,w2,w3 translations + twist) and conjugate <.> constraints.
    w2null=True adds a 5th mode for the director omega2 (constant-omega2 Psi column + <omega2>
    constraint row) -- the Eq.85/100 V1s projection otherwise misses omega2's near-null mode for a
    soft core (it is in NO rigid mode), leaking the soft-Gs director into the section shear."""
    Nn = len(nodes); ndof = 5*Nn
    idmode = (w2null == "id")                    # 'id' = constrain EVERY omega2 DOF (full subspace)
    nm = (4 + Nn) if idmode else (5 if w2null else 4)
    nodes_xi = _lagrange(p)
    C = np.zeros((nm, ndof)); Psi = np.zeros((ndof, nm))
    xg, wg = np.polynomial.legendre.leggauss(p+1)
    for el in elems:
        X = nodes[el]
        for xi, w in zip(xg, wg):
            N, dNr = _shape(nodes_xi, xi); Jac = np.hypot(*(dNr @ X)); dl = Jac*w
            for a, nd in enumerate(el):
                C[0, 5*nd+0] += N[a]*dl          # <w1>
                C[1, 5*nd+1] += N[a]*dl          # <w2>
                C[2, 5*nd+2] += N[a]*dl          # <w3>
                C[3, 5*nd+3] += N[a]*dl          # <omega1>
                if w2null is True:
                    C[4, 5*nd+4] += N[a]*dl      # <omega2>  (5th constraint, average)
    for nd in range(Nn):
        y2, y3 = nodes[nd]
        Psi[5*nd+0, 0] = 1.0                     # w1 translation
        Psi[5*nd+1, 1] = 1.0                     # w2 translation
        Psi[5*nd+2, 2] = 1.0                     # w3 translation
        Psi[5*nd+1, 3] = -y3; Psi[5*nd+2, 3] = y2; Psi[5*nd+3, 3] = -1.0  # twist
        if w2null is True:
            Psi[5*nd+4, 4] = 1.0                 # constant-omega2 mode
        elif idmode:
            C[4+nd, 5*nd+4] = 1.0                # identity: constrain this omega2 DOF
            Psi[5*nd+4, 4+nd] = 1.0
    return C, Psi


def timoshenko_rm(nodes, elems, layup_per_elem, D_by, G_by, k22_e, p=1, reduced=True,
                  return_warp=False, shear="mitc_both"):
    Dhh, Dhe, Dee, Dhl, Dll, Dle, Dhh_mem = assemble_all(
        nodes, elems, layup_per_elem, D_by, G_by, k22_e, p, reduced, shear=shear)
    C, Psi = build_C_Psi(nodes, elems, p)
    Dc = C.T
    Dhh_coo = coo_matrix(Dhh)
    V0, D1, A_aug = solve_fluctuation_field(Dhh_coo, -Dhe, Dc)   # G stays in the EB warping
    Deff = Dee + np.asarray(D1)
    bb, DhlV0, DhlTV0Dle, V0DllV0 = prepare_v1_rhs(
        jnp.array(V0), jnp.array(Dhl), jnp.array(Dll), jnp.array(Dle),
        jnp.array(Psi), jnp.array(Dc))
    n = Dhh.shape[0]
    R_v1 = np.concatenate([np.array(bb), np.zeros((4, bb.shape[1]))], axis=0)
    import pypardiso
    V_aug = pypardiso.spsolve(A_aug, R_v1)
    C6, *_ = finalize_v1_and_compute_deff(
        jnp.array(V_aug[:n, :]), jnp.array(V0), jnp.array(Deff),
        V0DllV0, DhlV0, DhlTV0Dle, jnp.array(Psi), jnp.array(Dc))
    if return_warp:
        return np.asarray(C6), np.asarray(Deff), np.asarray(V0), np.asarray(V_aug[:n, :])
    return np.asarray(C6), np.asarray(Deff)
