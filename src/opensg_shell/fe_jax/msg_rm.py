"""
MSG Reissner-Mindlin thin-walled beam -- cross-section homogenization (CG / C0).

Same MSG flow as the Kirchhoff model (assemble fluctuation stiffness, minimise,
recover the beam stiffness), but:
  * C0 Lagrange (CG) line elements -- 5 DOF/node [w1,w2,w3,omega1,omega2];
  * curvatures use FIRST derivatives of the rotation fluctuations (eq 4.23),
    so no C1/penalty;
  * a transverse-shear block G enters the energy -> shear LOCKING for thin walls,
    handled by SELECTIVE REDUCED INTEGRATION of the G-energy (the standard
    commercial fix; cf. assumed-strain/MITC).

Beam strains  eb = [gamma11, kappa1(twist), kappa2, kappa3]  (4, Euler-Bernoulli).
DOF field     q  = [w1,w2,w3, omega1,omega2]  per node.

See scripts/rm_research/RM_DERIVATION.md (sections 4-8) for the strain field and the algorithm.
This file validates on the isotropic circular tube (Opensg_MSG Table 3.1) first.
"""
import numpy as np


# ---------------------------------------------------------------- element maths
def _lagrange(p):
    """nodes in [-1,1] for order p (1 or 2)."""
    return np.linspace(-1.0, 1.0, p + 1)


def _shape(nodes_xi, xi):
    n = len(nodes_xi); N = np.ones(n); dN = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            N[i] *= (xi - nodes_xi[j]) / (nodes_xi[i] - nodes_xi[j])
        s = 0.0
        for j in range(n):
            if j == i:
                continue
            t = 1.0 / (nodes_xi[i] - nodes_xi[j])
            for m in range(n):
                if m == i or m == j:
                    continue
                t *= (xi - nodes_xi[m]) / (nodes_xi[i] - nodes_xi[m])
            s += t
        dN[i] = s
    return N, dN


# coefficient of k1 in the kappa12 macro.  The physical plate-twist measure for a
# beam twisting at rate k1 is kappa12+kappa21 = -2 k1 (ASC25 Eq. 24; Roy-Yu Eq. V),
# so the macro must carry the full -2.  The omega2 fluctuation that a -1 split
# relied on to supply the other -1 is locked to ~0 by the transverse-shear
# stiffness of a thin wall, so a -1 macro under-counts OPEN-section torsion by 4x
# (St-Venant J = W h^3/3 collapses to W h^3/12).  Closed cells hide this -- their
# torsion is A66-membrane (Bredt) dominated and the plate-twist term is a <1%
# correction -- which is why the closed tube/airfoil passed with -1; the open flat
# strip is the discriminating test.  -2 fixes the strip (centre GJ -75% -> -0.6%
# at h/W=0.01) and PRESERVES the tube (centre GJ -1.0% -> -0.3% at h/R=0.2).
KAPPA12_MACRO = -2.0


def _macro_BD(x2, x3, t2, t3, k22):
    """6x4 plate-strain macro map B_D(eb) ; eb=[g11,k1,k2,k3].  (== Kirchhoff Ge)"""
    Rn = x2 * t3 - x3 * t2
    B = np.zeros((6, 4))
    B[0] = [1.0, 0.0, x3, -x2]              # eps11
    B[2, 1] = Rn                            # 2eps12 = k1 Rn
    B[3, 2] = t2; B[3, 3] = t3              # kappa11 = t.k
    B[5, 1] = KAPPA12_MACRO + 0.5 * k22 * Rn   # kappa12+ macro (curvature sign matches the k22=-1/R convention)
    return B


def _macro_BG(x2, x3, t2, t3):
    """2x4 transverse-shear macro map (2g13, 2g23) from eb. (probe; see debug)"""
    return np.zeros((2, 4))


def assemble_rm(nodes, elems, node_dofs, D, Gs, k22_e, p=2, reduced=True):
    """Assemble the RM fluctuation system on a closed contour.

    nodes (Nn,2), elems (Ne,p+1) node indices, node_dofs = 5*Nn.
    D (6,6) plate ABD, Gs (2,2) transverse shear.  Returns Kqq, Kqe, Kee."""
    nodes_xi = _lagrange(p)
    ndof = 5 * len(nodes)
    Kqq = np.zeros((ndof, ndof)); Kqe = np.zeros((ndof, 4)); Kee = np.zeros((4, 4))
    xg2, wg2 = np.polynomial.legendre.leggauss(p + 1)     # full integration
    xg1, wg1 = np.polynomial.legendre.leggauss(max(1, p)) # reduced (one order less)

    for e, el in enumerate(elems):
        X = nodes[el]                                     # (p+1,2)
        k22 = float(k22_e[e])
        gdofs = np.concatenate([[5*n, 5*n+1, 5*n+2, 5*n+3, 5*n+4] for n in el])

        def quad(xs, ws, which):
            Kq = np.zeros((5*(p+1), 5*(p+1)))
            Kqe_e = np.zeros((5*(p+1), 4)); Kee_e = np.zeros((4, 4))
            for xi, w in zip(xs, ws):
                N, dNr = _shape(nodes_xi, xi)
                dxds_vec = dNr @ X                        # dX/dxi
                Jac = np.hypot(*dxds_vec); dN = dNr / Jac # arc-length derivative
                x2, x3 = N @ X
                t2, t3 = dxds_vec / Jac                   # unit tangent
                n2, n3 = t3, -t2                          # unit normal
                npn = p + 1
                BDq = np.zeros((6, 5*npn)); BGq = np.zeros((2, 5*npn))
                for a in range(npn):
                    o = 5*a
                    BDq[1, o+1] += t2*dN[a]; BDq[1, o+2] += t3*dN[a]   # eps22
                    BDq[2, o+0] += dN[a]                              # 2eps12 = w1'
                    BDq[4, o+3] += dN[a]                              # kappa22 = +dN(omega1) (rotation curvature; was -dN)
                    BDq[5, o+4] += dN[a]; BDq[5, o+0] += 0.5*k22*dN[a]  # kappa12+
                    BGq[0, o+4] += N[a]                               # 2g13 = omega2
                    BGq[1, o+1] += n2*dN[a]; BGq[1, o+2] += n3*dN[a]   # 2g23 = dwn/ds
                    BGq[1, o+3] += -N[a]                              # 2g23 -= omega1
                BDe = _macro_BD(x2, x3, t2, t3, k22)
                BGe = _macro_BG(x2, x3, t2, t3)
                dl = Jac * w
                if which == "D":
                    Kq += BDq.T @ D @ BDq * dl
                    Kqe_e += BDq.T @ D @ BDe * dl
                    Kee_e += BDe.T @ D @ BDe * dl
                else:
                    Kq += BGq.T @ Gs @ BGq * dl
                    Kqe_e += BGq.T @ Gs @ BGe * dl
                    Kee_e += BGe.T @ Gs @ BGe * dl
            return Kq, Kqe_e, Kee_e

        for which, (xs, ws) in [("D", (xg2, wg2)),
                                ("G", (xg1, wg1) if reduced else (xg2, wg2))]:
            Kq, Kqe_e, Kee_e = quad(xs, ws, which)
            Kqq[np.ix_(gdofs, gdofs)] += Kq
            Kqe[gdofs] += Kqe_e; Kee += Kee_e
    return Kqq, Kqe, Kee


def solve_eb(Kqq, Kqe, Kee, nodes):
    """Minimise -> V0 -> 4x4 EB stiffness.  Nullspace: rigid translations of
    w1,w2,w3 (3) handled by regularisation."""
    ndof = Kqq.shape[0]
    null = np.zeros((ndof, 3))
    null[0::5, 0] = 1.0; null[1::5, 1] = 1.0; null[2::5, 2] = 1.0
    Q, _ = np.linalg.qr(null)
    alpha = np.max(np.abs(np.diag(Kqq))) * 1e8
    Kreg = Kqq + alpha * Q @ Q.T
    V0 = np.linalg.solve(Kreg, -Kqe)
    V0 -= Q @ (Q.T @ V0)
    return Kee + V0.T @ Kqe
