"""
MSG Shell — Material Stiffness and ABD Plate Homogenization

Provides:
  - 6x6 orthotropic stiffness (Voigt ordering: [s11,s22,s33,s23,s13,s12])
  - Fiber-angle rotation matching OpenSG R_sig convention
  - MSG 1D through-thickness SG ABD (quadratic Lagrange, exact for uniform layers)
  - Classical Lamination Theory ABD (for comparison)
"""

import numpy as np
from .msg_transverse_shear import transverse_shear_stiffness, plate_8x8


def build_stiffness_6x6(E, G, nu):
    """6x6 stiffness C = S^{-1} from orthotropic elastic constants.

    Parameters
    ----------
    E  : [E1, E2, E3]  Young's moduli
    G  : [G12, G13, G23]  shear moduli
    nu : [nu12, nu13, nu23]  Poisson ratios

    Voigt ordering: [sigma11, sigma22, sigma33, sigma23, sigma13, sigma12]
    """
    E1, E2, E3 = E[0], E[1], E[2]
    G12, G13, G23 = G[0], G[1], G[2]
    v12, v13, v23 = nu[0], nu[1], nu[2]

    S = np.zeros((6, 6))
    S[0, 0] = 1.0 / E1
    S[1, 1] = 1.0 / E2
    S[2, 2] = 1.0 / E3
    S[0, 1] = S[1, 0] = -v12 / E1
    S[0, 2] = S[2, 0] = -v13 / E1
    S[1, 2] = S[2, 1] = -v23 / E2
    S[3, 3] = 1.0 / G23
    S[4, 4] = 1.0 / G13
    S[5, 5] = 1.0 / G12
    return np.linalg.inv(S)


def rotation_6x6(theta_deg):
    """6x6 rotation matrix for fiber angle (degrees), matching OpenSG R_sig."""
    th = np.deg2rad(theta_deg)
    c, s = np.cos(th), np.sin(th)
    cs = c * s
    return np.array([
        [c**2,   s**2,  0,  0,  0, -2*cs       ],
        [s**2,   c**2,  0,  0,  0,  2*cs       ],
        [0,      0,     1,  0,  0,  0           ],
        [0,      0,     0,  c,  s,  0           ],
        [0,      0,     0, -s,  c,  0           ],
        [cs,    -cs,    0,  0,  0,  c**2-s**2  ],
    ])


def rotated_stiffness_6x6(E, G, nu, theta_deg):
    """Rotated 6x6 stiffness for a ply at fiber angle theta_deg."""
    C = build_stiffness_6x6(E, G, nu)
    R = rotation_6x6(theta_deg)
    return R @ C @ R.T


def _lagrange_dN(nodes_xi, xi):
    """d/dxi of the 1D Lagrange basis at ``xi`` for nodes at ``nodes_xi``."""
    n = len(nodes_xi)
    dN = np.zeros(n)
    for i in range(n):
        s = 0.0
        for j in range(n):
            if j == i:
                continue
            term = 1.0 / (nodes_xi[i] - nodes_xi[j])
            for m in range(n):
                if m == i or m == j:
                    continue
                term *= (xi - nodes_xi[m]) / (nodes_xi[i] - nodes_xi[m])
            s += term
        dN[i] = s
    return dN


def _plate_B(nodes_xi, xi, he):
    """Through-thickness gamma_h operator B (6, 3*(p+1)) at ``xi``.

    Rows: eps33 <- dv3/dx, gamma23 <- dv2/dx, gamma13 <- dv1/dx.
    """
    dN = _lagrange_dN(nodes_xi, xi) * (2.0 / he)
    npn = len(nodes_xi)
    B = np.zeros((6, 3 * npn))
    for n_idx in range(npn):
        B[2, n_idx * 3 + 2] = dN[n_idx]
        B[3, n_idx * 3 + 1] = dN[n_idx]
        B[4, n_idx * 3 + 0] = dN[n_idx]
    return B


def compute_ABD_matrix(thick, angles_deg, mat_names, material_db, n_per_layer=1,
                       return_warping=False, elem_order=2, shear_refined=False,
                       z_ref=0.0):
    """Plate stiffness via MSG 1D through-thickness SG.

    By default returns the 6x6 ABD (the Kirchhoff plate law). Set
    ``shear_refined=True`` to instead return the 8x8 Reissner-Mindlin plate
    stiffness [[A,B,0],[B,D,0],[0,0,G]], i.e. the same 6x6 ABD plus the 2x2
    transverse-shear block G (the MSG/coupling-aware value, see
    :func:`msg_transverse_shear.transverse_shear_stiffness`). The 6x6 default is
    untouched so the Kirchhoff path is unaffected; the 8x8 is the RM-model input.

    Uses quadratic Lagrange elements (3-node, 3-pt Gauss). With n_per_layer=1
    the result is exact to machine precision for uniform layers.

    Reference surface at the bottom face (x=0, outer layer). The transverse-shear
    block G is reference-independent, so a later parallel-axis shift of the ABD
    (e.g. to the mid-surface) leaves the 8x8's G unchanged.

    Variational principle:
      gamma_h(v) = [0, 0, dv3/dx, dv2/dx, dv1/dx, 0]
      gamma_e    = [[1,0,0,x,0,0],[0,1,0,0,x,0],[0,0,0,0,0,0],
                    [0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,1,0,0,x]]
      D_eff = D_ee + V0^T @ F  (V0 = K^{-1}(-F), so D1 < 0, reduces stiffness)

    Parameters
    ----------
    thick      : list[float] — layer thicknesses, bottom to top
    angles_deg : list[float] — fiber angles in degrees per layer
    mat_names  : list[str]   — material name per layer (keys in material_db)
    material_db: dict        — {name: {E:[3], G:[3], nu:[3], rho:float}}
    n_per_layer: int         — quadratic sub-elements per layer (default 1)
    z_ref      : float        — through-thickness coordinate of the REFERENCE
                  surface, measured from the bottom/OML face (default 0 = OML).
                  The 1D SG mesh origin is shifted by z_ref so the membrane and
                  bending are referenced there: z_ref = 0 (OML), sum(thick)/2
                  (mid-surface / centre), sum(thick) (IML).  This is the mesh-based
                  reference change: the plies keep their order along the inward
                  through-thickness direction and only the origin moves, exactly as
                  the FEniCS OpenSG plate solver does (preferred over a
                  parallel-axis transform of the assembled ABD).
    return_warping : bool     — also return the through-thickness fluctuation
                  field for plate dehomogenization (see :func:`plate_dehom_strain`)

    Returns
    -------
    stiff : (6,6) MSG ABD stiffness (default), or (8,8) RM plate stiffness
            [[A,B,0],[B,D,0],[0,0,G]] when ``shear_refined=True``
    mass  : [mu, mu*xm3, i22] — mass per unit area, first/second moment
    warp  : dict (only if return_warping) — {V0, node_x, elem_layer, C_layers}
            the plate warping V0 (ndofs,6) and geometry needed to recover the
            3D through-thickness strain from the 6 plate (shell) strains.
    """
    nlay = len(thick)
    layer_bot = np.zeros(nlay + 1)
    for k in range(nlay):
        layer_bot[k + 1] = layer_bot[k] + thick[k]

    C_layers = [rotated_stiffness_6x6(
        material_db[mat_names[k]]['E'],
        material_db[mat_names[k]]['G'],
        material_db[mat_names[k]]['nu'],
        angles_deg[k]) for k in range(nlay)]
    rho_layers = [material_db[mat_names[k]].get('rho',
                  material_db[mat_names[k]].get('density', 0.0))
                  for k in range(nlay)]

    p = int(elem_order)                       # element polynomial order (2 or 3)
    nodes_xi = np.linspace(-1.0, 1.0, p + 1)
    n_elem = nlay * n_per_layer
    n_node = p * n_elem + 1
    ndofs = 3 * n_node

    node_x = np.empty(n_node)
    elem_layer = np.empty(n_elem, dtype=int)
    idx = 0
    for k in range(nlay):
        for s in range(n_per_layer):
            xl = layer_bot[k] + thick[k] * s / n_per_layer
            xr = layer_bot[k] + thick[k] * (s + 1) / n_per_layer
            for j in range(p):                # p new nodes (last is shared)
                node_x[p * idx + j] = xl + (xr - xl) * j / p
            elem_layer[idx] = k
            idx += 1
    node_x[p * n_elem] = layer_bot[-1]
    node_x = node_x - z_ref                   # mesh-based reference shift (origin at z_ref)

    xi_g, w_g = np.polynomial.legendre.leggauss(max(3, p + 1))

    K = np.zeros((ndofs, ndofs))
    F_load = np.zeros((ndofs, 6))
    D_ee = np.zeros((6, 6))
    mu, mu_x, i22_m = 0.0, 0.0, 0.0

    for e in range(n_elem):
        xl = node_x[p * e]
        xr = node_x[p * e + p]
        he = xr - xl
        k = elem_layer[e]
        Ck = C_layers[k]
        dofs = np.arange(3 * p * e, 3 * p * e + 3 * (p + 1))

        for q in range(len(xi_g)):
            xi = xi_g[q]
            x_q = 0.5 * (xl + xr) + 0.5 * he * xi
            dw = 0.5 * he * w_g[q]

            B = _plate_B(nodes_xi, xi, he)

            Ge = np.zeros((6, 6))
            Ge[0, 0] = 1.0;  Ge[0, 3] = x_q
            Ge[1, 1] = 1.0;  Ge[1, 4] = x_q
            Ge[5, 2] = 1.0;  Ge[5, 5] = x_q

            K[np.ix_(dofs, dofs)] += B.T @ Ck @ B * dw
            F_load[dofs, :] += B.T @ Ck @ Ge * dw
            D_ee += Ge.T @ Ck @ Ge * dw

            mu += rho_layers[k] * dw
            mu_x += rho_layers[k] * x_q * dw
            i22_m += rho_layers[k] * x_q**2 * dw

    # Nullspace: 3 rigid-body translations (v1, v2, v3 = const)
    null = np.zeros((ndofs, 3))
    null[0::3, 0] = 1.0
    null[1::3, 1] = 1.0
    null[2::3, 2] = 1.0
    Q, _ = np.linalg.qr(null)

    # Remove the rigid-body nullspace by PROJECTION, not a giant penalty.  A penalty
    # K + alpha*Q Q^T leaks alpha onto EVERY node through Q Q^T and swamps the physical
    # stiffness of a near-zero ply (gelcoat E=10 Pa, ~1e5 vs alpha~1e22) -> numerically
    # singular.  Projecting keeps the physical block (incl. soft plies) well-scaled; V0 is
    # the same physical fluctuation, so D_eff is unchanged for all well-conditioned cases.
    P = np.eye(ndofs) - Q @ Q.T
    beta = np.max(np.abs(np.diag(K)))
    K_proj = P @ K @ P + beta * (Q @ Q.T)

    V0 = np.linalg.solve(K_proj, -(P @ F_load))
    V0 = P @ V0

    D_eff = D_ee + V0.T @ F_load

    xm3 = mu_x / mu if mu > 0 else 0.0

    stiff = D_eff
    if shear_refined:                            # RM: append the 2x2 transverse-shear G
        Gmat = transverse_shear_stiffness(thick, angles_deg, mat_names, material_db)[0]
        stiff = plate_8x8(D_eff, Gmat)           # 8x8 [[A,B,0],[B,D,0],[0,0,G]]

    if return_warping:
        warp = {"V0": V0, "node_x": node_x, "elem_layer": elem_layer,
                "C_layers": C_layers, "elem_order": p,
                "angles": list(angles_deg)}
        return stiff, [mu, mu * xm3, i22_m], warp
    return stiff, [mu, mu * xm3, i22_m]


def plate_dehom_strain(warp, shell_strain, n_eval_per_elem=3):
    """Plate dehomogenization: 3D strain/stress across the thickness.

    Given the 6 plate (shell) strains [eps11, eps22, gamma12, kappa11, kappa22,
    kappa12] on the reference surface, recover the pointwise 3D strain through
    the plate thickness via the MSG plate warping::

        Gamma_3D(z) = (B(z) @ V0_e + Ge(z)) @ shell_strain   (strain concentration)
        Sigma_3D(z) = C_layer @ Gamma_3D(z)

    where ``B`` is the through-thickness gamma_h operator and ``Ge`` the macro
    map (membrane + z*curvature) — the exact transposes of the operators that
    built the ABD matrix, so ``int Gamma:Sigma dz == shell_strain^T ABD
    shell_strain`` to machine precision.

    Parameters
    ----------
    warp : dict — from ``compute_ABD_matrix(..., return_warping=True)``
    shell_strain : (6,) — the recovered plate/shell strains at one arc point
    n_eval_per_elem : int — through-thickness sample points per quadratic element

    Returns
    -------
    z    : (npts,)   through-thickness coordinate (0 = bottom face)
    Gam  : (npts, 6) 3D strain   [e11, e22, e33, g23, g13, g12]
    Sig  : (npts, 6) 3D stress (material/ply Voigt order matching C_layers)
    """
    V0 = warp["V0"]; node_x = warp["node_x"]
    elem_layer = warp["elem_layer"]; C_layers = warp["C_layers"]
    p = warp.get("elem_order", 2)
    nodes_xi = np.linspace(-1.0, 1.0, p + 1)
    ss = np.asarray(shell_strain, dtype=float)
    n_elem = len(elem_layer)
    xi_eval = np.linspace(-1.0, 1.0, n_eval_per_elem)

    z_all, Gam_all, Sig_all = [], [], []
    for e in range(n_elem):
        xl = node_x[p * e]; xr = node_x[p * e + p]; he = xr - xl
        k = elem_layer[e]; Ck = C_layers[k]
        dofs = np.arange(3 * p * e, 3 * p * e + 3 * (p + 1))
        V0e = V0[dofs, :]
        for xi in xi_eval:
            x_q = 0.5 * (xl + xr) + 0.5 * he * xi
            B = _plate_B(nodes_xi, xi, he)
            Ge = np.zeros((6, 6))
            Ge[0, 0] = 1.0;  Ge[0, 3] = x_q
            Ge[1, 1] = 1.0;  Ge[1, 4] = x_q
            Ge[5, 2] = 1.0;  Ge[5, 5] = x_q
            SC = B @ V0e + Ge            # 6x6 strain concentration
            Gam = SC @ ss
            z_all.append(x_q); Gam_all.append(Gam); Sig_all.append(Ck @ Gam)
    return np.array(z_all), np.array(Gam_all), np.array(Sig_all)


def shift_abd_reference(ABD, z0):
    """Move the ABD reference surface by ``z0`` along the through-thickness axis
    (parallel-axis), KEEPING the e3 (through-thickness) direction unchanged.

    Voigt order [eps11,eps22,gamma12, kappa11,kappa22,kappa12].  Membrane strain
    at the shifted reference is ``m + z0*k`` (T = [[I, z0 I],[0, I]]), so
    ``ABD_new = T^{-T} ABD T^{-1}``.  Use this to reference the IML (z0 = laminate
    thickness) instead of reversing the layup — reversal flips e3 and breaks the
    agreement with the material orientation (see ``check-e3-orientation``).
    """
    Tinv = np.eye(6)
    Tinv[:3, 3:] = -z0 * np.eye(3)
    return Tinv.T @ np.asarray(ABD) @ Tinv


def plate_stress_at_depth(warp, shell_strain, z):
    """3D strain/stress at a SPECIFIC through-thickness depth ``z`` (0 = OML).

    Same MSG plate strain concentration as :func:`plate_dehom_strain`, evaluated
    at one depth instead of a sweep — used to recover the stress at an arbitrary
    cross-section point (see ``msg_dehom.stress_at_points``).  ``z`` is clamped
    to [0, total thickness].

    Returns (Gam (6,), Sig (6,), angle) in the laminate Voigt order
    [e11, e22, e33, g23, g13, g12]; ``angle`` is the fiber angle (deg) of the ply
    at this depth, so a caller can rotate to the material/ply frame with
    ``rotation_6x6(-angle) @ Sig``.
    """
    V0 = warp["V0"]; node_x = warp["node_x"]
    elem_layer = warp["elem_layer"]; C_layers = warp["C_layers"]
    angles = warp.get("angles")
    p = warp.get("elem_order", 2)
    nodes_xi = np.linspace(-1.0, 1.0, p + 1)
    ss = np.asarray(shell_strain, dtype=float)
    h = float(node_x[-1])
    z = min(max(float(z), 0.0), h)

    e = len(elem_layer) - 1                      # plate sub-element containing z
    for ee in range(len(elem_layer)):
        if z <= node_x[p * ee + p] + 1e-12:
            e = ee; break
    xl = node_x[p * e]; xr = node_x[p * e + p]; he = xr - xl
    xi = np.clip(2.0 * (z - xl) / he - 1.0, -1.0, 1.0)
    layer = elem_layer[e]; Ck = C_layers[layer]
    V0e = V0[np.arange(3 * p * e, 3 * p * e + 3 * (p + 1)), :]
    B = _plate_B(nodes_xi, xi, he)
    Ge = np.zeros((6, 6))
    Ge[0, 0] = 1.0;  Ge[0, 3] = z
    Ge[1, 1] = 1.0;  Ge[1, 4] = z
    Ge[5, 2] = 1.0;  Ge[5, 5] = z
    Gam = (B @ V0e + Ge) @ ss
    angle = 0.0 if angles is None else float(angles[layer])
    return Gam, Ck @ Gam, angle


def compute_ABD_CLT(thick, angles_deg, mat_names, material_db):
    """Classical Lamination Theory 6x6 ABD with reference at bottom face.

    Returns [[A, B], [B, D]] (3x3 blocks) stacked into 6x6.
    """
    A = np.zeros((3, 3))
    B = np.zeros((3, 3))
    D = np.zeros((3, 3))
    z = 0.0

    for k in range(len(thick)):
        mat = material_db[mat_names[k]]
        E1, E2 = mat['E'][0], mat['E'][1]
        G12 = mat['G'][0]
        v12 = mat['nu'][0]

        denom = 1.0 - v12**2 * E2 / E1
        Q = np.array([
            [E1 / denom,        v12 * E2 / denom, 0.0],
            [v12 * E2 / denom,  E2 / denom,       0.0],
            [0.0,               0.0,               G12],
        ])

        th = np.deg2rad(angles_deg[k])
        c, s = np.cos(th), np.sin(th)
        R3 = np.array([
            [c**2,  s**2,  -2*c*s       ],
            [s**2,  c**2,   2*c*s       ],
            [c*s,  -c*s,    c**2-s**2  ],
        ])
        Qr = R3 @ Q @ R3.T

        zt = z + thick[k]
        A += Qr * (zt - z)
        B += 0.5 * Qr * (zt**2 - z**2)
        D += (1.0 / 3.0) * Qr * (zt**3 - z**3)
        z = zt

    return np.block([[A, B], [B, D]])
