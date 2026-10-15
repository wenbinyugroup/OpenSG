"""
Transverse-shear stiffness of a laminate -- the 2x2 G block of the
Reissner-Mindlin 8x8 plate stiffness [[A,B,0],[B,D,0],[0,0,G]].  This is the
RM-specific addition to the Kirchhoff 6x6 ABD; computed by the FSDT /
3D-equilibrium shear-flow method with complementary-energy equivalence, NOT the
crude (5/6) sum(G t).

DEFAULT for the RM cross-section work is the coupled MSG form (coupled=True).

References (the papers this 2x2 G is referred to):
  * Whitney, J.M. (1973), "Shear correction factors for orthotropic laminates
    under static load", J. Appl. Mech. 40(1):302-304 -- the complementary-energy /
    3D-equilibrium shear-flow construction implemented here (the ``coupled`` route
    is its coupling-aware 2x2 generalization).
  * Yu, W. (2005), "Mathematical construction of a Reissner-Mindlin plate theory
    for composite laminates", Int. J. Solids Struct. 42:6680-6699; and
    Yu, Hodges & Volovoi (2002), CMAME 191:5087-5109 -- the MSG Reissner-Mindlin
    plate model this G is the transverse-shear block of (correction-factor-free).
  NB: this G is the Whitney complementary-energy stiffness in the spirit of the
  Yu-2005 RM plate; it is NOT the Yu-2005 minimum-information-loss G itself.

Physics (per direction; x = ply-1 / beam, y = ply-2 / tangent):
  pure bending    : sigma_11(z) = Q11(z) (z - z_na) kappa        (z_na = neutral axis)
  3D equilibrium  : d sigma_13/dz = - d sigma_11/dx
  -> shear flow   : sigma_13(z) = (g(z)/D_eff) V , g(z)=INT_bot^z Q11(zeta-z_na) dzeta
  energy equiv.   : F = D_eff^2 / INT g(z)^2 / G13(z) dz
The SAME g(z)/D_eff recovers the transverse-shear STRESS from the beam shear force.

``coupled`` (default True, the MSG form) uses the full through-thickness shear
flexibility Gbar^{-1}(z) and returns the full 2x2; it reduces to the per-direction
FSDT/Whitney(1973) diagonal whenever Gbar is diagonal (G13=G23 / specially
orthotropic), and differs only for G13 != G23 with off-axis plies.
Validates: isotropic plate -> F = (5/6) G h exactly.
"""
import numpy as np


def _ply_Q_and_G(E, G, nu, theta_deg):
    """In-plane Q11,Q22 (reduced) and transverse shear Gbar (2x2) for a rotated ply."""
    E1, E2 = E[0], E[1]; G12, G13, G23 = G[0], G[1], G[2]
    v12 = nu[0]; v21 = v12 * E2 / E1
    den = 1 - v12 * v21
    Q11, Q22, Q12 = E1/den, E2/den, v12*E2/den
    Q66 = G12
    th = np.deg2rad(theta_deg); c, s = np.cos(th), np.sin(th)
    Qb11 = Q11*c**4 + 2*(Q12+2*Q66)*s**2*c**2 + Q22*s**4
    Qb22 = Q11*s**4 + 2*(Q12+2*Q66)*s**2*c**2 + Q22*c**4
    Gp = np.array([[G13, 0.0], [0.0, G23]])
    R2 = np.array([[c, s], [-s, c]])
    Gbar = R2 @ Gp @ R2.T
    return Qb11, Qb22, Gbar


def transverse_shear_stiffness(thick, angles, mat_names, mat_db, nz_per_ply=40,
                               coupled=True):
    """2x2 transverse-shear stiffness G_2x2 (rows/cols = [2g13, 2g23]) and the
    shear-flow shapes for stress recovery.  Returns (Gmat, recover, aux) where
    recover(z, V) -> [sigma13, sigma23].

    ``coupled`` selects the transverse-shear constitutive route:
      * True (default) -- the coupling-aware ("no-shear-correction-factor") MSG
        form: the full 2x2 from the laminate complementary shear energy with the
        FULL through-thickness shear flexibility Gbar^{-1}(z),
        F = M^{-1}, M_ab = INT ghat_a ghat_b [Gbar^{-1}]_ab dz, ghat = g/D_eff.
      * False -- the FSDT / Whitney(1973) energy-equivalence stiffness, per
        direction with the diagonal G13'/G23', F = D_eff^2 / INT g^2/G dz.
    The two coincide when Gbar(z) is diagonal (G13=G23 / specially orthotropic);
    they differ only when the rotated transverse shear couples (G13 != G23 with
    off-axis plies)."""
    nlay = len(thick)
    zb = np.concatenate([[0.0], np.cumsum(thick)])
    htot = zb[-1]
    zs, Qb11, Qb22, Gb = [], [], [], []
    for k in range(nlay):
        zk = np.linspace(zb[k], zb[k+1], nz_per_ply, endpoint=False)
        q11, q22, gbar = _ply_Q_and_G(mat_db[mat_names[k]]['E'], mat_db[mat_names[k]]['G'],
                                      mat_db[mat_names[k]]['nu'], angles[k])
        for z in zk:
            zs.append(z); Qb11.append(q11); Qb22.append(q22); Gb.append(gbar)
    zs = np.array(zs); dz = htot/len(zs)
    Qb11 = np.array(Qb11); Qb22 = np.array(Qb22); Gb = np.array(Gb)

    def direction(Q, Gss):
        z_na = np.sum(Q*zs*dz)/np.sum(Q*dz)
        zc = zs - z_na
        D_eff = np.sum(Q*zc**2*dz)
        g = np.cumsum(Q*zc*dz)
        F = D_eff**2 / np.sum(g**2/Gss*dz)
        return F, g, D_eff

    Fx, gx, Dx = direction(Qb11, Gb[:, 0, 0])
    Fy, gy, Dy = direction(Qb22, Gb[:, 1, 1])
    if coupled:
        gh1, gh2 = gx/Dx, gy/Dy
        Ginv = np.linalg.inv(Gb)
        M = np.array([[np.sum(gh1*gh1*Ginv[:, 0, 0]*dz), np.sum(gh1*gh2*Ginv[:, 0, 1]*dz)],
                      [np.sum(gh1*gh2*Ginv[:, 1, 0]*dz), np.sum(gh2*gh2*Ginv[:, 1, 1]*dz)]])
        Gmat = np.linalg.inv(M)
    else:
        Gmat = np.array([[Fx, 0.0], [0.0, Fy]])

    def recover(z, V):
        i = min(int(z/htot*len(zs)), len(zs)-1)
        return np.array([gx[i]/Dx*V[0], gy[i]/Dy*V[1]])
    return Gmat, recover, (zs, gx/Dx, gy/Dy)


def plate_8x8(D6, Gmat):
    """Assemble the RM 8x8 plate stiffness [[A,B,0],[B,D,0],[0,0,G]] from the 6x6
    Kirchhoff ABD ``D6`` and the 2x2 transverse-shear block ``Gmat``."""
    P = np.zeros((8, 8)); P[:6, :6] = D6; P[6:, 6:] = Gmat
    return P
