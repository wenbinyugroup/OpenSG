"""solid_props.py -- equivalent 3-D SOLID properties from the shell cross-section SG.

The same RM shell SG and fluctuation operators as the Timoshenko ring,
homogenized into classical 3-D elasticity.  Public entry points:
build_solid_bundle (yaml -> C3D bundle) and ring_solid (assembled KKT solve).

    eps_shell = Gamma_e * ebar + Gamma_h * w
    ebar = [G11, G22, G33, 2G23, 2G13, 2G12]   (1 = beam axis, 2/3 = cross axes)
    w    = [w1, w2, w3 | om1, om2, om3]

FE process: one KKT solve for V0 with the 4-mode rigid kernel;
D_eff = Dee + V0^T Dhe (6x6); C3D = D_eff / w_SG.  No V1 step.

# ----------------------------------------------------------------------------
# ALL VARIABLES USED IN THIS MODULE   (ne = #elements, m = #contour nodes,
# nred = #master nodes after the periodic tie, ndof = 6*nred, NDOF6 = 6)
# ----------------------------------------------------------------------------
# geometry / frames
#   rx        (m, 3) float     contour nodes (cross-section plane, axial = 0)
#   rcells    (ne, 2) int      line-element connectivity (0-based)
#   rsub      (ne,) int        section/layup id per element
#   re3       (ne, 3) float    wall normal e3 per element
#   h         float            strip depth along the axial direction
#   nodes     (2m, 3) float    strip nodes = [rx ; rx + h*ez]
#   quads     (ne, 4) int      strip quad connectivity [a, b, m+b, m+a]
#   Xe        (ne, 4, 3) float quad corner coordinates per element
#   e3e       (ne, 3) float    = re3
#   xi, eta   float            Gauss-point coordinates on the quad
#   cross     (2,) list int    indices of the two cross-section axes
#   ax        int              index of the axial direction
#   N         (4,) float       bilinear shape functions at (xi, eta)
#   D1, D2    (ne, 4) float    dN/dzeta1, dN/dzeta2 (curvilinear derivatives)
#   c         dict of (ne,)    direction cosines: x11..x32 = X_{i,alpha};
#                              y1, y2, y3 = C_{3i}
#   xi1, xi2  (ne, 3) float    X_{i1}, X_{i2} stacked
#   yv        (ne, 3) float    C_{3i} stacked
#   den       (ne,) float      C33 divisor of DR (1 where |C33| < 1e-8)
#   dA        (ne,) float      area weight at the Gauss point
# operators
#   B -> BDh  (ne, 6, 24)      Gamma_h membrane+curvature rows (slots w | om)
#   Bg -> BGh (ne, 2, 24)      Gamma_h transverse-shear rows
#   Dr -> DRh (ne, 24)         drilling residual row DR (divided by C33)
#   BDe6      (ne, 6, 6)       Gamma_e membrane+curvature rows on ebar
#   BGe6      (ne, 2, 6)       Gamma_e transverse-shear rows on ebar
#   tie       dict of (ne,1,24) g23 Dvorkin-Bathe tying rows (only g23 is tied
#                              for a cross-section; g13 stays at Gauss)
# material laws
#   D_by      list (6, 6)      wall ABD per section (about the reference line)
#   G_by      list (2, 2)      wall transverse-shear law per section
#   De, Gm    (ne,6,6)/(ne,2,2) per-element laws (D_by/G_by gathered by rsub)
#   k22_edge  (ne,) float      edge curvature factor (loader output; unused here)
# assembly
#   node_master (m,) int       node -> master map (periodic tie; identity free)
#   dof_map   (2m,) int        strip dof map = [node_master ; node_master]
#   g         (ne, 24) int     global dof columns per element
#   Ehh, Ehe  (ne,24,24)/(ne,24,6)  per-element blocks before scatter
#   Dhh       (ndof, ndof)     int Gamma_h^T K Gamma_h dA
#   Dhe       (ndof, 6)        int Gamma_h^T K Gamma_e dA
#   Dee       (6, 6)           int Gamma_e^T K Gamma_e dA
#   Gc        (ne, ndof)       int_e DR dA rows (element-constant multipliers)
#   C5        (4, 5m)          rigid-mode rows from build_C_Psi (5 dof/node)
#   C6        (nk, ndof)       kernel rows on 6-dof slots; nk = 3 periodic
#                              (translations), 4 free (+ in-plane rotation)
# solve / outputs
#   A         (naug+nk, naug+nk)  KKT matrix, naug = ndof + ne
#   V0        (ndof, 6)        fluctuation per unit macro strain (min-norm lsq)
#   Deff      (6, 6)           Dee + V0^T Dhe   (per unit axial length)
#   C3D       (6, 6)           Deff / cell_area, order = GBAR_ORDER
#   cell_area float            normalizing area (convex hull if not given)
# ----------------------------------------------------------------------------
"""
import numpy as np

from .segment_indep import _surf_frame_batch, _shear_batch, NDOF6

GBAR_ORDER = ("strains [e11 e22 e33 2e23 2e13 2e12] -> stiffness C11..C66  "
              "(1=beam axis, 2/3=cross-section axes)")


def solid_fluct_ops_batch(Xe, e3e, xi, eta, cross, ax):
    """Gamma_h for the SOLID route (standalone -- no Gamma_l, no beam macro).
    Returns BDh (ne,6,24), BGh (ne,2,24), DRh (ne,24), dA (ne,).
    DOF slots per node: 0:3 = w_i, 3:6 = om_i."""
    N, D1, D2, dA, c = _surf_frame_batch(Xe, e3e, xi, eta, cross, ax)
    ne = Xe.shape[0]
    xi1 = np.stack([c["x11"], c["x21"], c["x31"]], axis=1)     # X_{i1}
    xi2 = np.stack([c["x12"], c["x22"], c["x32"]], axis=1)     # X_{i2}
    yv = np.stack([c["y1"], c["y2"], c["y3"]], axis=1)         # C_{3i}

    B = np.zeros((ne, 6, 4, NDOF6))
    B[:, 0, :, 0:3] = D1[:, :, None] * xi1[:, None, :]                  # eps11 = X_i1 w_i,1
    B[:, 1, :, 0:3] = D2[:, :, None] * xi2[:, None, :]                  # eps22 = X_i2 w_i,2
    B[:, 2, :, 0:3] = (D2[:, :, None] * xi1[:, None, :]
                       + D1[:, :, None] * xi2[:, None, :])              # 2eps12 = X_i1 w_i,2 + X_i2 w_i,1
    B[:, 3, :, 3:6] = D1[:, :, None] * xi2[:, None, :]                  # K11 = X_i2 om_i,1
    B[:, 4, :, 3:6] = -D2[:, :, None] * xi1[:, None, :]                 # K22 = -X_i1 om_i,2
    B[:, 5, :, 3:6] = (D2[:, :, None] * xi2[:, None, :]
                       - D1[:, :, None] * xi1[:, None, :])              # K12+K21 = X_i2 om_i,2 - X_i1 om_i,1

    Bg = np.zeros((ne, 2, 4, NDOF6))
    Bg[:, 0, :, 0:3] = D1[:, :, None] * yv[:, None, :]                  # 2g13 = C_3i w_i,1 ...
    Bg[:, 0, :, 3:6] = N[None, :, None] * xi2[:, None, :]               # ... + X_i2 om_i
    Bg[:, 1, :, 0:3] = D2[:, :, None] * yv[:, None, :]                  # 2g23 = C_3i w_i,2 ...
    Bg[:, 1, :, 3:6] = N[None, :, None] * (-xi1)[:, None, :]            # ... - X_i1 om_i

    # DR = om3 + [ C31 om1 + C32 om2 - 1/2 (X_i2 w_i,1 - X_i1 w_i,2) ] / C33
    # (C33 -> 0: multiplied-through form kept)
    den = np.where(np.abs(yv[:, 2]) > 1e-8, yv[:, 2], 1.0)
    Dr = np.zeros((ne, 4, NDOF6))
    Dr[:, :, 0:3] = -0.5 * (D1[:, :, None] * xi2[:, None, :]
                            - D2[:, :, None] * xi1[:, None, :]) \
        / den[:, None, None]                    # -1/2 (X_i2 w_i,1 - X_i1 w_i,2)/C33
    Dr[:, :, 3:6] = (N[None, :, None] * yv[:, None, :]
                     / den[:, None, None])      # om3 + (C31 om1 + C32 om2)/C33
    return (B.reshape(ne, 6, 24), Bg.reshape(ne, 2, 24),
            Dr.reshape(ne, 24), dA)


def _tie_rows_solid(Xe, e3e, cross, ax):
    """Dvorkin-Bathe MITC tying rows for the solid route: only g23 is tied
    for a cross-section (g13 stays at the Gauss value).

    In:  Xe: (ne, 4, 3) float quad corner coordinates per element;
         e3e: (ne, 3) float wall normal per element;
         cross: (2,) int indices of the two cross-section axes;
         ax: int index of the axial direction.
    Out: dict {"g23m", "g23p"}: (ne, 1, 24) float g23 Gamma_h rows sampled
         at tying points (0, -1) and (0, +1)."""
    f = lambda xi, eta: solid_fluct_ops_batch(Xe, e3e, xi, eta, cross, ax)[1]
    return {"g23m": f(0.0, -1.0)[:, 1:2, :], "g23p": f(0.0, 1.0)[:, 1:2, :]}


def solid_macro_enn_batch(Xe, e3e, xi, eta, cross, ax):
    """The e_nn completion row of Gamma_e (through-thickness normal strain):
        e_nn = C_3i G_ij C_3j
    -> row [C31^2, C32^2, C33^2, C32*C33, C31*C33, C31*C32]  (ne, 6).
    Its Gamma_h partner is zero on a midline (no thickness-stretch DOF)."""
    N, D1, D2, dA, c = _surf_frame_batch(Xe, e3e, xi, eta, cross, ax)
    C31, C32, C33 = c["y1"], c["y2"], c["y3"]
    return np.stack([C31**2, C32**2, C33**2,
                     C32*C33, C31*C33, C31*C32], 1), dA


def solid_macro_ops_batch(Xe, e3e, xi, eta, cross, ax):
    """The document's Gamma_e at (xi,eta): BDe6 (ne,6,6) on the membrane+curvature
    rows [e11,e22,2e12,K11,K22,K12+K21], BGe6 (ne,2,6) on the shear rows
    [2g13,2g23]; dA (ne,).  Entries verbatim from Eq. (17) of the document."""
    N, D1, D2, dA, c = _surf_frame_batch(Xe, e3e, xi, eta, cross, ax)
    ne = Xe.shape[0]
    X11, X21, X31 = c["x11"], c["x21"], c["x31"]        # X_{i1}
    X12, X22, X32 = c["x12"], c["x22"], c["x32"]        # X_{i2}
    C31, C32, C33 = c["y1"], c["y2"], c["y3"]           # C_{3i}
    z = np.zeros(ne)

    # Gamma_e = Gamma_h on  w_i -> G_ij y_j ,  om_i -> 0
    # columns: [ G11  G22  G33  2G23  2G13  2G12 ]
    BDe6 = np.stack([
        # eps11 = X_i1 G_ij X_j1
        np.stack([X11**2, X21**2, X31**2,
                  X31*X21, X11*X31, X11*X21], 1),
        # eps22 = X_i2 G_ij X_j2
        np.stack([X12**2, X22**2, X32**2,
                  X32*X22, X32*X12, X12*X22], 1),
        # 2eps12 = X_i1 G_ij X_j2 + X_i2 G_ij X_j1
        np.stack([2*X11*X12, 2*X22*X21, 2*X31*X32,
                  X22*X31 + X32*X21, X12*X31 + X32*X11,
                  X12*X21 + X11*X22], 1),
        np.stack([z, z, z, z, z, z], 1),                # K11     (om -> 0)
        np.stack([z, z, z, z, z, z], 1),                # K22     (om -> 0)
        np.stack([z, z, z, z, z, z], 1),                # K12+K21 (om -> 0)
    ], axis=1)
    # G_ij is the TENSOR macro strain: off-diagonals = engineering/2, so the
    # engineering-shear columns carry HALF the pair-sum.  This makes the rows
    # literally Gamma_h on (w_i -> G_ij y_j, om_i -> 0); the om-embedding that
    # would restore full pair-sums is inadmissible (its fiber tilt jumps
    # between wall families at junctions while nodal om is single-valued).
    BGe6 = np.stack([
        # 2g13 = C_3i G_ij X_j1
        np.stack([C31*X11, C32*X21, C33*X31,
                  0.5*(X31*C32 + X21*C33), 0.5*(X31*C31 + X11*C33),
                  0.5*(X21*C31 + X11*C32)], 1),
        # 2g23 = C_3i G_ij X_j2
        np.stack([C31*X12, C32*X22, C33*X32,
                  0.5*(X32*C32 + X22*C33), 0.5*(X32*C31 + X12*C33),
                  0.5*(X22*C31 + X12*C32)], 1),
    ], axis=1)
    return BDe6, BGe6, dA


def _C_from_eng(E, G, nu):
    """6x6 stiffness from engineering constants; Voigt [11,22,33,23,13,12],
    G = [G12,G13,G23], nu = [nu12,nu13,nu23]."""
    S = np.zeros((6, 6))
    S[0, 0], S[1, 1], S[2, 2] = 1/E[0], 1/E[1], 1/E[2]
    S[0, 1] = S[1, 0] = -nu[0]/E[0]
    S[0, 2] = S[2, 0] = -nu[1]/E[0]
    S[1, 2] = S[2, 1] = -nu[2]/E[1]
    S[3, 3], S[4, 4], S[5, 5] = 1/G[2], 1/G[1], 1/G[0]
    return np.linalg.inv(S)


def _rot_inplane(C, th_deg):
    """Rotate a 6x6 Voigt stiffness by th about axis 3 (ply angle about the
    wall normal, 1 -> 2 ccw)."""
    th = np.radians(th_deg)
    c, s = np.cos(th), np.sin(th)
    cs = c*s
    R = np.array([[c**2, s**2, 0, 0, 0, 2*cs],
                  [s**2, c**2, 0, 0, 0, -2*cs],
                  [0, 0, 1, 0, 0, 0],
                  [0, 0, 0, c, -s, 0],
                  [0, 0, 0, s, c, 0],
                  [-cs, cs, 0, 0, 0, c**2 - s**2]])
    return R @ C @ R.T


def wall_solid_law(sections, materials):
    """Per-section through-thickness-integrated UN-CONDENSED wall law
    sum_k t_k * C_ply(theta_k), wall Voigt [11,22,nn,2n,1n,12].

    In:  sections: list of yaml section dicts, each with "layup" =
         [[material_name, thickness, angle_deg], ...];
         materials: list of yaml material dicts with E/G/nu (possibly under
         an "elastic" key).
    Out: list of (6, 6) float, one t*C per section.
    This is the law the e_nn completion contracts with."""
    mats = {str(m["name"]): m for m in materials}
    out = []
    for sec in sections:
        C = np.zeros((6, 6))
        for p in sec["layup"]:
            m = mats[str(p[0])]
            el = m["elastic"] if "elastic" in m else m
            Cp = _C_from_eng([float(v) for v in el["E"]],
                             [float(v) for v in el["G"]],
                             [float(v) for v in el["nu"]])
            C += float(p[1]) * _rot_inplane(Cp, float(p[2]))
        out.append(C)
    return out


_VOIGT_IDX = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]


def _voigt_rotate(C, Q):
    """C' = Q C Q^T on the 4th-order tensor; C engineering Voigt
    [11,22,33,23,13,12] (stiffness entries equal tensor components)."""
    C4 = np.zeros((3, 3, 3, 3))
    for I, (i, j) in enumerate(_VOIGT_IDX):
        for J, (k, l) in enumerate(_VOIGT_IDX):
            C4[i, j, k, l] = C4[j, i, k, l] = C4[i, j, l, k] \
                = C4[j, i, l, k] = C[I, J]
    C4r = np.einsum('ai,bj,ck,dl,ijkl->abcd', Q, Q, Q, Q, C4)
    Cr = np.zeros((6, 6))
    for I, (i, j) in enumerate(_VOIGT_IDX):
        for J, (k, l) in enumerate(_VOIGT_IDX):
            Cr[I, J] = C4r[i, j, k, l]
    return Cr


def junction_inventory(rx, cells, rsub, cross, ang_tol_deg=1.0):
    """Junction nodes of the midline mesh: a node shared by walls with
    distinct tangents.  Per junction: walls = [(section, tangent2d, weight)],
    weight = 1 if the wall runs THROUGH the node (elements on both sides,
    counted overlap length t_other), 1/2 if it ENDS there (counted t_other/2)."""
    ctol = np.cos(np.radians(ang_tol_deg))
    adj = {}
    for e, (n1, n2) in enumerate(cells):
        adj.setdefault(int(n1), []).append(e)
        adj.setdefault(int(n2), []).append(e)
    out = []
    for nd, elist in adj.items():
        if len(elist) < 2:
            continue
        groups = []
        for e in elist:
            n1, n2 = cells[e]
            other = int(n2) if int(n1) == nd else int(n1)
            v = np.asarray(rx[other])[cross] - np.asarray(rx[nd])[cross]
            d = v/np.linalg.norm(v)
            side = 1
            if d[0] < -1e-9 or (abs(d[0]) < 1e-9 and d[1] < 0):
                d, side = -d, -1
            for g in groups:
                if float(np.dot(g["dir"], d)) > ctol:
                    g["sides"].add(side)
                    g["secs"].add(int(rsub[e]))
                    break
            else:
                groups.append({"dir": d, "sides": {side},
                               "secs": {int(rsub[e])}})
        if len(groups) < 2:
            continue
        walls = [(sorted(g["secs"])[0], g["dir"],
                  1.0 if len(g["sides"]) == 2 else 0.5) for g in groups]
        out.append({"node": int(nd), "walls": walls})
    return out


def _wall_normal_integrand(d2, ABD, G):
    """Per-unit-length Dee integrand of a wall, NORMAL (1..3) block, section
    frame.  Nonzero Gamma_e rows on the normal columns for a cross-section
    wall (tangent (0,c,s), normal (0,-s,c)):
        e11 -> [1, 0, 0]          x A11
        e22 -> [0, c^2, s^2]      x A22 (+A12 cross)
        2g23 -> [0, -sc, cs]      x G22   (diagonal columns, unhalved)"""
    c, s = float(d2[0]), float(d2[1])
    r1 = np.array([1.0, 0.0, 0.0])
    r2 = np.array([0.0, c*c, s*s])
    r3 = np.array([0.0, -s*c, c*s])
    A11, A12, A22 = ABD[0, 0], ABD[0, 1], ABD[1, 1]
    W = (A11*np.outer(r1, r1) + A12*(np.outer(r1, r2) + np.outer(r2, r1))
         + A22*np.outer(r2, r2) + G[1, 1]*np.outer(r3, r3))
    return W


def junction_census_correction(inv, Cw_by, t_by, D_by, G_by):
    """Junction census correction to Dee: on each wall-overlap block
    A_j = t_A t_B / sin(theta) swap the walls' condensed law for the full
    un-condensed 3-D law,
        dDee = A_j C_j  -  sum_i w_i (t_other/sin) W_i,     normal block only,
    C_j = mean of the two walls' per-volume un-condensed laws rotated to the
    section frame (Q columns: axial, tangent, normal).

    In:  inv: junction_inventory output, [{"node": int, "walls":
         [(section, tangent2d, weight)]}];
         Cw_by: list of (6, 6) float wall t*C per section (wall_solid_law);
         t_by: list of float total wall thickness per section;
         D_by: list of (6, 6) float wall ABD per section;
         G_by: list of (2, 2) float wall transverse-shear law per section.
    Out: dD (6, 6) float additive Deff correction; only the normal (1..3)
         block is nonzero."""
    dD = np.zeros((6, 6))
    for J in inv:
        ws = J["walls"]
        for iA in range(len(ws)):
            for iB in range(iA + 1, len(ws)):
                (sA, dA, wA), (sB, dB, wB) = ws[iA], ws[iB]
                tA, tB = t_by[sA], t_by[sB]
                sinth = abs(dA[0]*dB[1] - dA[1]*dB[0])
                if sinth < 1e-6:
                    continue
                Aj = tA*tB/sinth
                Cj = np.zeros((6, 6))
                for sec, d2, tk in ((sA, dA, tA), (sB, dB, tB)):
                    c, s = float(d2[0]), float(d2[1])
                    Q = np.array([[1.0, 0.0, 0.0],
                                  [0.0, c, -s],
                                  [0.0, s, c]])
                    Cj += 0.5*_voigt_rotate(Cw_by[sec]/tk, Q)
                blk = Aj*Cj[:3, :3] \
                    - wA*(tB/sinth)*_wall_normal_integrand(dA, D_by[sA],
                                                           G_by[sA]) \
                    - wB*(tA/sinth)*_wall_normal_integrand(dB, D_by[sB],
                                                           G_by[sB])
                dD[:3, :3] += blk
    return dD


def junction_inventory_merged(rx, cells, rsub, re3, cross, master):
    """Junctions of the PERIODICALLY REDUCED midline mesh: lattice-coincident
    nodes merge (a ring's four L-corners become ONE X-crossing of the merged
    walls), and coincident periodic partner walls appear as stacked members
    [(sec, mirrored)] of one wall family, ordered bottom (-n) to top;
    mirrored = the partner's frame is the 180-deg-about-axial image
    (e3 . n_canonical < 0), i.e. its ply angles negate in the merged frame."""
    rx = np.asarray(rx, float)
    cells = np.asarray(cells, int)
    adj = {}
    for e, (n1, n2) in enumerate(cells):
        adj.setdefault(int(master[n1]), []).append((e, int(n1)))
        adj.setdefault(int(master[n2]), []).append((e, int(n2)))
    ctol = np.cos(np.radians(1.0))
    out = []
    for mnode, elist in adj.items():
        if len(elist) < 2:
            continue
        groups = []
        for e, nd in elist:
            n1, n2 = cells[e]
            other = int(n2) if int(n1) == nd else int(n1)
            v = rx[other][cross] - rx[nd][cross]
            dcan = v/np.linalg.norm(v)
            side = 1
            if dcan[0] < -1e-9 or (abs(dcan[0]) < 1e-9 and dcan[1] < 0):
                dcan, side = -dcan, -1
            ncan = np.array([-dcan[1], dcan[0]])
            dn = float(np.dot(np.asarray(re3[e], float)[cross], ncan))
            for g in groups:
                if abs(float(np.dot(g["dir"], dcan))) > ctol:
                    g["mem"].append((side, int(rsub[e]), dn))
                    # a member node with this family on BOTH sides = the wall
                    # runs THROUGH its own node (owns the junction block)
                    if (nd, -side) in g["ndside"]:
                        g["thru"] = True
                    g["ndside"].add((nd, side))
                    break
            else:
                groups.append({"dir": dcan,
                               "mem": [(side, int(rsub[e]), dn)],
                               "ndside": {(nd, side)}, "thru": False})
        if len(groups) < 2:
            continue
        walls = []
        for g in groups:
            sides = {m[0] for m in g["mem"]}
            w = 1.0 if len(sides) == 2 else 0.5
            sd = 1 if 1 in sides else -1
            stack = tuple((sec, dn < 0) for (s, sec, dn) in
                          sorted([m for m in g["mem"] if m[0] == sd],
                                 key=lambda m: -m[2]))
            walls.append({"dir": g["dir"], "weight": w, "stack": stack,
                          "thru": bool(g.get("thru", False))})
        out.append({"node": int(mnode), "walls": walls})
    return out


def assemble_solid_macro(nodes, quads, subdom, e3s, D_by, G_by, cross, ax,
                         dof_map=None, shear="mitc4_g23", Cw_by=None):
    """Dhe6 (ndof,6), Dee6 (6,6): the solid-macro blocks, mirroring
    assemble_segment_indep's quadrature, wall-law lookup, tied-shear scheme and
    dof_map.  The drilling residual has NO solid-strain column, so no multiplier or
    penalty cross terms arise here.

    Cw_by (list of 6x6 wall t*C per section, from wall_solid_law) activates the
    e_nn COMPLETION in Dee: the membrane normal block [e11,e22,2e12] switches
    from the plane-stress A to the un-condensed t*C block on
    [e11,e22,e_nn,2e12], with the e_nn macro row C_3i G_ij C_3j.  Dee's normal
    block then equals A_mat*C exactly (the solid's).  Dhe is unchanged (the
    e_nn row has no midline fluctuation partner).  Cw_by=None reproduces the
    previous behaviour identically."""
    if dof_map is None:
        dof_map = np.arange(len(nodes))
    dof_map = np.asarray(dof_map, int)
    nodes = np.asarray(nodes, float); quads = np.asarray(quads, int)
    ne = len(quads)
    Nn = int(np.max(dof_map)) + 1; ndof = NDOF6 * Nn
    Xe = nodes[quads]; e3e = np.asarray(e3s, float)
    sd = np.asarray(subdom, int)
    keys = sorted(set(int(s) for s in sd))
    Darr = np.stack([np.asarray(D_by[k], float) for k in keys])
    Garr = np.stack([np.asarray(G_by[k], float) for k in keys])
    pos = {k: i for i, k in enumerate(keys)}
    sdi = np.array([pos[int(s)] for s in sd])
    De = Darr[sdi]; Gm = Garr[sdi]

    g = (NDOF6 * dof_map[quads])[:, :, None] + np.arange(NDOF6)[None, None, :]
    g = g.reshape(ne, 24)
    tie = None if shear == "full" else _tie_rows_solid(Xe, e3e, cross, ax)

    if Cw_by is not None:
        Cw_arr = np.stack([np.asarray(Cw_by[k], float) for k in keys])
        # wall-law sub-block on [e11, e22, e_nn, 2e12] (wall Voigt 0,1,2,5)
        Cn4e = Cw_arr[sdi][:, [0, 1, 2, 5]][:, :, [0, 1, 2, 5]]
        A3e = De[:, 0:3, 0:3]                  # plane-stress block it replaces

    Ehe = np.zeros((ne, 24, 6)); Dee = np.zeros((6, 6))
    gpv = 1.0 / np.sqrt(3.0)
    for (xi, eta) in [(-gpv, -gpv), (gpv, -gpv), (gpv, gpv), (-gpv, gpv)]:
        BDh, BGh, _, dA = solid_fluct_ops_batch(Xe, e3e, xi, eta, cross, ax)
        BDe6, BGe6, _ = solid_macro_ops_batch(Xe, e3e, xi, eta, cross, ax)
        BGt = BGh if shear == "full" else _shear_batch(xi, eta, shear, BGh, tie)
        w = dA[:, None, None]
        DBe = np.einsum('eij,ejb->eib', De, BDe6)
        GBe = np.einsum('eij,ejb->eib', Gm, BGe6)
        Ehe += w * (np.einsum('eia,eib->eab', BDh, DBe)
                    + np.einsum('eia,eib->eab', BGt, GBe))
        Dee += np.einsum('e,eab->ab', dA,
                         np.einsum('eia,eib->eab', BDe6, DBe)
                         + np.einsum('eia,eib->eab', BGe6, GBe))
        if Cw_by is not None:
            # e_nn COMPLETION of Dee: swap the membrane normal block
            # [e11,e22,2e12]xA  ->  [e11,e22,e_nn,2e12] x (t*C)
            Benn, _ = solid_macro_enn_batch(Xe, e3e, xi, eta, cross, ax)
            Bn4 = np.stack([BDe6[:, 0, :], BDe6[:, 1, :],
                            Benn, BDe6[:, 2, :]], axis=1)      # (ne,4,6)
            Bm3 = BDe6[:, 0:3, :]                              # (ne,3,6)
            Dee += np.einsum('e,eab->ab', dA,
                             np.einsum('eia,eij,ejb->eab', Bn4, Cn4e, Bn4)
                             - np.einsum('eia,eij,ejb->eab', Bm3, A3e, Bm3))
    Dhe = np.zeros((ndof, 6))
    np.add.at(Dhe, g.reshape(-1), Ehe.reshape(-1, 6))
    return Dhe, Dee


def assemble_solid_strip(nodes, quads, subdom, e3s, D_by, G_by, cross, ax,
                         dof_map=None, shear="mitc4_g23"):
    """Dhh (ndof,ndof) and Gc (ne,ndof) for the SOLID route -- standalone.

    Carries only what the zeroth-order solid theory needs: Gamma_h, the
    element-constant drilling rows, and dA.  No Gamma_l / V1 ladder, no beam
    macro columns, no drilling penalty (om3 is enforced exactly by the
    multipliers, not by a penalty).

    Gc row per element e:  int_e DR dA = 0,  with
        DR = C_3i om_i - 1/2 (X_i2 w_i,1 - X_i1 w_i,2)
    i.e. the drilling residual is written on the SOLID-side om3 through C_3i,
    with NO macro-strain column (the macro contributions cancel identically)."""
    if dof_map is None:
        dof_map = np.arange(len(nodes))
    dof_map = np.asarray(dof_map, int)
    nodes = np.asarray(nodes, float); quads = np.asarray(quads, int)
    ne = len(quads)
    Nn = int(np.max(dof_map)) + 1; ndof = NDOF6 * Nn
    Xe = nodes[quads]; e3e = np.asarray(e3s, float)
    sd = np.asarray(subdom, int)
    keys = sorted(set(int(s) for s in sd))
    Darr = np.stack([np.asarray(D_by[k], float) for k in keys])
    Garr = np.stack([np.asarray(G_by[k], float) for k in keys])
    pos = {k: i for i, k in enumerate(keys)}
    sdi = np.array([pos[int(s)] for s in sd])
    De = Darr[sdi]; Gm = Garr[sdi]

    g = (NDOF6 * dof_map[quads])[:, :, None] + np.arange(NDOF6)[None, None, :]
    g = g.reshape(ne, 24)
    tie = None if shear == "full" else _tie_rows_solid(Xe, e3e, cross, ax)

    Ehh = np.zeros((ne, 24, 24)); Gel = np.zeros((ne, 24))
    gpv = 1.0 / np.sqrt(3.0)
    for (xi, eta) in [(-gpv, -gpv), (gpv, -gpv), (gpv, gpv), (-gpv, gpv)]:
        BDh, BGh, DRh, dA = solid_fluct_ops_batch(Xe, e3e, xi, eta, cross, ax)
        BGt = BGh if shear == "full" else _shear_batch(xi, eta, shear, BGh, tie)
        w = dA[:, None, None]
        DB = np.einsum('eij,ejb->eib', De, BDh)
        GB = np.einsum('eij,ejb->eib', Gm, BGt)
        Ehh += w * (np.einsum('eia,eib->eab', BDh, DB)
                    + np.einsum('eia,eib->eab', BGt, GB))
        Gel += DRh * dA[:, None]

    Dhh = np.zeros((ndof, ndof))
    np.add.at(Dhh, (g[:, :, None], g[:, None, :]), Ehh)
    Gc = np.zeros((ne, ndof))
    np.add.at(Gc, (np.arange(ne)[:, None], g), Gel)
    return Dhh, Gc


def ring_solid(rx, rcells, rsub, re3, D_by, G_by, k22_edge, ax, cross, h=None,
               shear="mitc4_g23", lam_space="elem", return_fields=False,
               periodic=True, Cw_by=None, Dhh_extra=None):
    """Equivalent-solid homogenization of the ring SG.  Returns D_eff (6,6), the
    contour-integrated stiffness per unit axial length on GBAR_ORDER; with
    return_fields=True also the 6-column warping V0.

    Mirrors ring_indep: one-quad-deep prismatic strip, top row DOF-mapped onto the
    bottom, element-constant drilling Lagrange multipliers (zero macro column),
    rigid-kernel KKT, single V0 solve, D_eff = Dee + V0^T Dhe.  No V1.

    periodic=True (THE DEFAULT -- periodicity is always part of the msg_shell
    solid-property route) ties opposite bounding-box faces of the SG through
    `periodic_multiscale.mesh_to_periodic_sparse_assembly_map`, the shell mirror
    of the solid side's map: for a 2-D SG both in-plane directions are tied, for
    a 3-D SG all three, and the repeated dof_map[dof_map] resolves the edge and
    corner chains, so opposite faces, edges AND corners are all periodic.  The
    tie rides in the assembly map (element connectivity re-pointed at master
    nodes), so no constraint rows are added, and the kernel drops the in-plane
    rotation, keeping the 3 translations.

    periodic=False leaves the SG FREE, whose equivalent-solid stiffness is rank
    one by construction: every macro strain except Gamma_11 is cancelled at zero
    energy by an affine fluctuation, which only periodicity forbids.  Kept for
    diagnostics only."""
    from .fe_jax.msg_rm_timo import build_C_Psi
    from scipy.linalg import lu_factor, lu_solve

    m = len(rx)
    if h is None:
        h = float(np.mean(np.linalg.norm(rx[rcells[:, 1]] - rx[rcells[:, 0]], axis=1)))
    ez = np.zeros(3); ez[ax] = 1.0
    nodes = np.vstack([rx, rx + h * ez])
    if periodic:
        from .periodic_multiscale import mesh_to_periodic_sparse_assembly_map
        # feed one "cell" per node so the returned reduced connectivity IS the
        # node -> master map the assemblers take as dof_map
        rc, _ = mesh_to_periodic_sparse_assembly_map(
            m, np.arange(m)[:, None], rx[:, cross], 3, NDOF6)
        node_master = np.asarray(rc, int).ravel()
    else:
        node_master = np.arange(m)
    dof_map = np.concatenate([node_master, node_master])
    quads = np.array([[a, b, m + b, m + a] for a, b in rcells], dtype=int)
    e3q = np.asarray(re3)

    Dhh, Gc = assemble_solid_strip(nodes, quads, rsub, e3q, D_by, G_by,
                                   cross, ax, dof_map=dof_map, shear=shear)
    Dhe6, Dee6 = assemble_solid_macro(nodes, quads, rsub, e3q, D_by, G_by,
                                      cross, ax, dof_map=dof_map, shear=shear,
                                      Cw_by=Cw_by)
    Dhh = np.asarray(Dhh) / h; Gc = Gc / h
    Dhe6 = Dhe6 / h; Dee6 = Dee6 / h
    if Dhh_extra is not None:
        # caller-supplied fluctuation stiffening (e.g. rigid junction blocks);
        # acts on w only, so Dee/Dhe and every drive channel are untouched
        Dhh = Dhh + Dhh_extra

    M = Dhh.shape[0]; P = Gc.shape[0]
    C5, Psi5 = build_C_Psi(rx[:, cross], rcells, p=1)
    # rigid kernel: 3 translations + the in-plane rotation for a free SG; tying
    # opposite edges removes the rotation, so a periodic cell carries only 3
    nk = 3 if periodic else 4
    C6 = np.zeros((nk, M))
    for n in range(m):
        s = NDOF6 * node_master[n]
        C6[:, s:s + 5] += C5[:nk, 5 * n:5 * n + 5]

    naug = M + P
    A = np.zeros((naug + nk, naug + nk))
    A[:M, :M] = Dhh; A[:M, M:naug] = Gc.T; A[M:naug, :M] = Gc
    A[:M, naug:] = C6.T; A[naug:, :M] = C6
    R0 = np.zeros((naug + nk, 6)); R0[:M] = -Dhe6         # zero macro drilling column
    # lstsq min-norm is load-bearing: the drilling null-space makes an LU
    # solve of the rank-deficient KKT unreliable (do NOT swap for a direct solve).
    V0 = np.linalg.lstsq(A, R0, rcond=None)[0][:naug]
    Deff = Dee6 + V0[:M].T @ Dhe6
    Deff = 0.5 * (Deff + Deff.T)
    if return_fields:
        return Deff, np.asarray(V0[:M])
    return Deff


def elastic_constants(C3D):
    """The 9 engineering constants from the compliance S = inv(C3D):
    E1,E2,E3 = 1/S11,1/S22,1/S33; G23,G13,G12 = 1/S44,1/S55,1/S66;
    nu12 = -S12*E1, nu13 = -S13*E1, nu23 = -S23*E2.  Also returns cond(C3D)."""
    C = np.asarray(C3D, float)
    S = np.linalg.inv(C)
    out = {
        "E1": 1.0 / S[0, 0], "E2": 1.0 / S[1, 1], "E3": 1.0 / S[2, 2],
        "G23": 1.0 / S[3, 3], "G13": 1.0 / S[4, 4], "G12": 1.0 / S[5, 5],
        "nu12": -S[0, 1] / S[0, 0], "nu13": -S[0, 2] / S[0, 0],
        "nu23": -S[1, 2] / S[1, 1],
        "cond": float(np.linalg.cond(C)),
    }
    return out, S


def write_abdg_out(out_path, sections, D_by, G_by):
    """Write each section's wall plate law in the SwiftComp .K layout: the
    8x8 Reissner-Mindlin ABDG = [[A B 0; B^T D 0; 0 0 G]] stiffness and its
    compliance per section, no engineering constants.

    In:  out_path str; sections list of yaml section dicts; D_by, G_by
         per-section ABD (6, 6) and transverse-shear G (2, 2)
    Out: the .out file at out_path; returns out_path (str)."""
    from opensg_solid.sg_homo import _knum
    with open(out_path, "w") as f:
        f.write(" OpenSG msg-shell wall plate laws, one block per section\n"
                " rows [eps11 eps22 2eps12 | K11 K22 K12+K21 | 2g13 2g23]\n")
        for si, sec in enumerate(sections):
            name = sec.get("elementSet", "section_%d" % si)
            lay = [[str(p[0]), float(p[1]), float(p[2])] for p in sec["layup"]]
            ABDG = np.zeros((8, 8))
            ABDG[:6, :6] = np.asarray(D_by[si], float)
            ABDG[6:, 6:] = np.asarray(G_by[si], float)
            S = np.linalg.inv(ABDG)
            f.write("\n section %d: %s  layup %s\n" % (si, name, lay))
            f.write(" The Effective Reissner-Mindlin Plate Stiffness Matrix\n"
                    " --------------------------------------------\n")
            for i in range(8):
                f.write("".join(_knum(ABDG[i, j]) for j in range(8)) + "\n")
            f.write("\n The Effective Reissner-Mindlin Plate Compliance"
                    " Matrix\n"
                    " --------------------------------------------\n")
            for i in range(8):
                f.write("".join(_knum(S[i, j]) for j in range(8)) + "\n")
    return out_path


def build_solid_bundle(shell_yaml, ref=None, shear="mitc4_g23", g_source="msg",
                       cell_area=None, periodic=True, junction=None):
    """Load the shell yaml exactly as build_rm_bundle does (same reference logic,
    same MSG wall transverse-shear upgrade), run ring_solid, and package:

        {"C3D", "D_eff", "cell_area", "area_source", "V0", geometry..., "order"}

    C3D = D_eff / cell_area.  cell_area=None -> convex-hull area of the contour."""
    import time as _time
    import yaml as _yaml
    from .oml_ring import load_ring_ref

    _t0 = _time.perf_counter()
    d = _yaml.safe_load(open(shell_yaml))
    if ref is None:
        ref = d.get("reference", "center")
    R = load_ring_ref(shell_yaml, ref)
    frac = {"center": 0.5, "oml": 0.0, "oml_flip": 1.0, "iml": 1.0}.get(ref, 0.0)
    G_by = list(R["G_by"])
    if g_source == "msg":
        from opensg_solid.rm_plate_1D.msg_rm_plate import rm_plate_msg
        from .emit_abd import material_db_from_yaml
        _mdb = material_db_from_yaml(d["materials"])
        for si, sec in enumerate(d["sections"]):
            _pl = [[str(p[0]), float(p[1]), float(p[2])] for p in sec["layup"]]
            _rr = rm_plate_msg([p[1] for p in _pl], [p[2] for p in _pl],
                               [p[0] for p in _pl], _mdb, fraction=frac)
            if _rr["G_msg"] is not None:
                G_by[si] = np.asarray(_rr["G_msg"])
    import os as _os
    write_abdg_out(_os.path.splitext(shell_yaml)[0] + "_ABDG.out",
                   d["sections"], R["D_by"], G_by)
    # e_nn channel NOT included: the condensed ABD already carries the
    # pointwise thickness relaxation; freezing e_nn in Gamma_e without a
    # Gamma_h partner double-counts that energy.
    Deff, V0 = ring_solid(R["rx"], R["cells"], R["rsub"], R["re3"], R["D_by"], G_by,
                          R["k22"], R["ax"], R["cross"], shear=shear,
                          lam_space="elem", return_fields=True, periodic=periodic)
    jinfo = None
    if junction == "census":
        # sigma_nn is blocked on the t_A x t_B wall-overlap blocks: swap the
        # condensed wall law for the full 3-D law there (normal block only;
        # junction shear relaxes with the joint and is left to the walls)
        t_by = [sum(float(p[1]) for p in sec["layup"]) for sec in d["sections"]]
        Cw_by = wall_solid_law(d["sections"], d["materials"])
        inv = junction_inventory(R["rx"], R["cells"], R["rsub"], R["cross"])
        dD = junction_census_correction(inv, Cw_by, t_by, R["D_by"], G_by)
        Deff = Deff + dD
        jinfo = {"n_junctions": len(inv), "inventory": inv, "dDee": dD}
    elif junction in ("micro", "microcell"):
        # level-2: per-junction-type local 2-D corner micro-solve (general
        # laminate, dC = E_solid_patch - E_shell_patch, mirror-line lattice
        # environment).  Inventory on the PERIODICALLY REDUCED mesh: a ring's
        # four L-corners merge into one X-crossing of the stacked 2t walls
        # (the mirrored partner's plies negate: m45 -> antisymmetric [+-45]).
        from .junction_micro import corner_micro_law
        from .periodic_multiscale import mesh_to_periodic_sparse_assembly_map
        rxJ = np.asarray(R["rx"])
        mJ = len(rxJ)
        if periodic:
            rc, _ = mesh_to_periodic_sparse_assembly_map(
                mJ, np.arange(mJ)[:, None], rxJ[:, R["cross"]], 3, NDOF6)
            masterJ = np.asarray(rc, int).ravel()
        else:
            masterJ = np.arange(mJ)
        inv = junction_inventory_merged(rxJ, R["cells"], R["rsub"], R["re3"],
                                        R["cross"], masterJ)
        dD = np.zeros((6, 6))
        cache = {}
        for J in inv:
            ws = sorted(J["walls"],
                        key=lambda w: (-w["weight"], -float(w["thru"])))
            if len(ws) != 2:
                raise NotImplementedError(">2 wall families at a junction")
            A, B = ws
            if abs(float(np.dot(A["dir"], B["dir"]))) > 0.05:
                raise NotImplementedError("non-orthogonal junction")
            topo = {(1.0, 1.0): "X", (1.0, 0.5): "T",
                    (0.5, 0.5): "L"}[(A["weight"], B["weight"])]
            # block fill: the family running THROUGH its own member node owns
            # the block (I-beam flange); merged L-corners (neither) -> mitre4
            fill = "A" if (A["thru"] or topo != "X") else \
                   ("mitre4" if not B["thru"] else "A")
            key = (topo, A["stack"], B["stack"], fill, junction)
            if key not in cache:
                if junction == "microcell" and topo == "X":
                    from .junction_micro import microcell_law
                    cache[key] = microcell_law(list(A["stack"]),
                                               list(B["stack"]),
                                               d["sections"], d["materials"],
                                               R["D_by"], G_by, fill=fill)
                else:
                    cache[key] = corner_micro_law(topo, list(A["stack"]),
                                                  list(B["stack"]),
                                                  d["sections"],
                                                  d["materials"], g_source,
                                                  fill=fill)
            dCloc, jinf = cache[key]
            c, s = float(A["dir"][0]), float(A["dir"][1])
            Q = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
            d6 = np.zeros((6, 6))
            d6[:3, :3] = dCloc
            dD += _voigt_rotate(d6, Q)
            # scaled D44 (in-plane shear) merge-census correction: both mini
            # D44s are wall-bending dominated (~1/span), so the mini
            # difference transfers by Lm/span when the two cross spans agree
            if junction == "microcell" and "Lm" in jinf:
                ext = rxJ[:, R["cross"]].max(0) - rxJ[:, R["cross"]].min(0)
                if abs(ext[0] - ext[1]) < 0.05*max(ext):
                    dD[3, 3] += (jinf["D_solid_mini"][3, 3]
                                 - jinf["D_shell_mini"][3, 3]) \
                        * jinf["Lm"]/float(ext[0])
        Deff = Deff + dD
        jinfo = {"n_junctions": len(inv), "inventory": inv, "dDee": dD,
                 "types": sorted(cache.keys())}
    area_source = "user"
    if cell_area is None:
        from scipy.spatial import ConvexHull
        cell_area = float(ConvexHull(R["rx"][:, R["cross"]]).volume)
        area_source = "hull"
    solve_time = _time.perf_counter() - _t0
    # every OpenSG run writes its timed .out by default
    _C = Deff / float(cell_area)
    from opensg_solid.sg_homo import write_sc_K
    write_sc_K(_os.path.splitext(shell_yaml)[0] + "_C3D.out", _C,
               solve_time=solve_time,
               model="msg-shell equivalent 3D solid (cross-section SG),"
                     " omega %.8g (%s)" % (float(cell_area), area_source))
    return {"C3D": Deff / float(cell_area), "D_eff": Deff,
            "solve_time": solve_time,
            "cell_area": float(cell_area), "area_source": area_source,
            "junction": jinfo, "V0": V0, "rx3": np.asarray(R["rx"]),
            "red_cells": np.asarray(R["cells"]), "rsub": np.asarray(R["rsub"]),
            "re3": np.asarray(R["re3"]), "k22": np.asarray(R["k22"]),
            "ax": int(R["ax"]), "cross": list(R["cross"]), "ref": ref,
            "g_source": g_source, "order": GBAR_ORDER}
