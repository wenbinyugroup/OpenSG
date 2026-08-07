"""Level-2 junction law: local 2-D corner micro-solve, general laminate.

Replaces the frozen-block census law with the TRUE corner correction
    dC_j[c,d] = E_patch(ebar_c, ebar_d) - sum_walls L_counted * e_wall[c,d]
over the three normal macro channels c,d in {e11, e22, e33} (local frame):

  * e_wall  : exact far-field energy density per unit wall length from a 1-D
              through-thickness generalized-plane-strain solve (in-plane
              strains = macro projections, sigma on n-faces free; one linear
              element per ply is exact).
  * E_patch : ply-resolved 2-D prismatic solid patch of the junction (stubs
              ~5t, L-corner mitred / T,X through-wall fill), Dirichlet cut
              faces carrying the EXACT far-field displacement profile (affine
              macro + 1-D fluctuation) -> no spurious cut boundary layers.
  * L_counted: midline census length inside the patch (node -> cut face),
              so the subtraction is exactly what the wall integrals count.

Junction shear is left to relax with the joint (normal channels only); for a
rotated junction (e.g. +-45 X-cell) the local normal block is tensor-rotated
to the section frame, which legitimately populates the section 2e23 row.
"""
import numpy as np

from .solid_props import _C_from_eng, _rot_inplane, _voigt_rotate


def _ply_C_wall(mat, ang_deg):
    """3-D 6x6 of one ply in the WALL frame [1=axial, 2=tangent, 3=normal],
    Voigt [11,22,33,23,13,12]; ply angle about the wall normal."""
    el = mat["elastic"] if "elastic" in mat else mat
    Cp = _C_from_eng([float(v) for v in el["E"]],
                     [float(v) for v in el["G"]],
                     [float(v) for v in el["nu"]])
    return _rot_inplane(Cp, float(ang_deg))


def _wall_layup(section, materials, mirrored=False):
    """Plies of one wall as [(C_wall_frame, t_k)]; a MIRRORED wall (the
    coincident periodic partner, frame rotated 180 deg about the axial axis:
    a2 -> -a2, n -> -n) has reversed stacking and negated ply angles."""
    mats = {str(m["name"]): m for m in materials}
    lay = list(section["layup"])
    if mirrored:
        lay = [[p[0], p[1], -float(p[2])] for p in lay[::-1]]
    plies = [(_ply_C_wall(mats[str(p[0])], p[2]), float(p[1])) for p in lay]
    t = sum(tk for _, tk in plies)
    return plies, t


def stack_plies(stack, sections, materials):
    """Merged solid laminate of coincident periodic walls: stack is
    [(sec_idx, mirrored)], ordered bottom (-n side) to top."""
    plies = []
    for sec, mir in stack:
        plies += _wall_layup(sections[sec], materials, mirrored=mir)[0]
    t = sum(tk for _, tk in plies)
    return plies, t


def stack_shell_law(stack, sections, materials, g_source="msg", frac=0.5):
    """What the GLOBAL shell model carries for the stacked coincident walls:
    the SUM of each member's production ABD and (msg) G, the mirrored member
    built from its angle-negated section -- coincident midlines, laws add."""
    from .solve_segment_jax import _material_by_section
    D6 = np.zeros((6, 6))
    G2 = np.zeros((2, 2))
    for sec, mir in stack:
        s = dict(sections[sec])
        if mir:
            s["layup"] = [[p[0], p[1], -float(p[2])]
                          for p in list(sections[sec]["layup"])[::-1]]
        D_by, G_by = _material_by_section([s], materials, center_ref=True)
        Dk = np.asarray(D_by[0], float)
        Gk = np.asarray(G_by[0], float)
        if g_source == "msg":
            from opensg_solid.rm_plate_1D.msg_rm_plate import rm_plate_msg
            from .emit_abd import material_db_from_yaml
            pl = [[str(p[0]), float(p[1]), float(p[2])] for p in s["layup"]]
            rr = rm_plate_msg([p[1] for p in pl], [p[2] for p in pl],
                              [p[0] for p in pl],
                              material_db_from_yaml(materials), fraction=frac)
            if rr["G_msg"] is not None:
                Gk = np.asarray(rr["G_msg"], float)
        D6 += Dk
        G2 += Gk
    return D6, G2


def _wall_farfield(plies):
    """1-D through-thickness far field of a laminate wall (wall frame).
    Unknown nodal (u1, ut, un)(n); total strain = eps_mac (wall frame) plus
    [0,0,dun/dn, dut/dn, du1/dn, 0].  Returns:
      e_bil (3x3): mutual energy density per unit length for unit wall-frame
                   macro channels [e11, e_tt, e_nn]
      prof(ch)   : nodal fluctuation profiles per channel, pinned u(mid)=0
      zs         : nodal n-coordinates (centered)."""
    ts = [tk for _, tk in plies]
    t = sum(ts)
    zs = np.concatenate([[0.0], np.cumsum(ts)]) - t/2
    npl = len(plies)
    nn = npl + 1
    # DOF order per node: (un, ut, u1) -> strain rows (nn=2, 2tn=3, 21n=4)
    ROWS = [2, 3, 4]
    CH = [np.array([1, 0, 0, 0, 0, 0.]), np.array([0, 1, 0, 0, 0, 0.]),
          np.array([0, 0, 1, 0, 0, 0.])]          # wall [e11, e_tt, e_nn]
    K = np.zeros((3*nn, 3*nn))
    F = np.zeros((3*nn, 3))
    for k, (C, tk) in enumerate(plies):
        Cs = C[np.ix_(ROWS, ROWS)]
        Cm = C[ROWS, :]
        g = np.zeros((3, 3*nn))
        for j in range(3):                        # d/dn of (un, ut, u1)
            g[j, 3*k + j] = -1.0/tk
            g[j, 3*(k+1) + j] = 1.0/tk
        K += tk * (g.T @ Cs @ g)
        for c in range(3):
            F[:, c] -= tk * (g.T @ (Cm @ CH[c]))
    fix = 3*(nn//2)
    keep = [i for i in range(3*nn) if i not in (fix, fix+1, fix+2)]
    U = np.zeros((3*nn, 3))
    U[keep, :] = np.linalg.solve(K[np.ix_(keep, keep)], F[keep, :])
    e_bil = np.zeros((3, 3))
    for c in range(3):
        for d in range(3):
            # mutual energy: (eps_c + g u_c)^T C (eps_d + g u_d) summed
            e = 0.0
            for k, (C, tk) in enumerate(plies):
                g = np.zeros((3, 3*nn))
                for j in range(3):
                    g[j, 3*k + j] = -1.0/tk
                    g[j, 3*(k+1) + j] = 1.0/tk
                ec = CH[c].copy(); ec[ROWS] += g @ U[:, c]
                ed = CH[d].copy(); ed[ROWS] += g @ U[:, d]
                e += tk * float(ec @ C @ ed)
            e_bil[c, d] = e
    prof = U.reshape(nn, 3, 3)                    # [node, (un,ut,u1), channel]
    return e_bil, prof, zs


def _grade(a, b, hmin, hmax):
    """1-D points from a to b, size hmin at a growing to hmax."""
    pts = [a]
    h, x = hmin, a
    while x + h < b - 1e-12:
        x += h
        pts.append(x)
        h = min(h*1.35, hmax)
    pts.append(b)
    return np.array(pts)


def _patch_mesh(topo, pliesA, tA, pliesB, tB, Ls, nply_el=2, fill="A",
                extA=0.0, extB=0.0):
    """Structured tri mesh of the junction patch in LOCAL coords: wall A along
    x (band |y|<=tA/2), wall B along y (band |x|<=tB/2).
    topo: 'L' (A:+x, B:+y, mitre fill), 'T' (A through +-x, B:-y, fill A),
          'X' (both through; fill 'A' or 'mitre4').
    extA/extB: homologous-cut extensions of the SOLID stubs (the solid
    lattice's junction spacing exceeds the midline node spacing when the
    other family is an adjacent-merged stack: (1-1/n)*t_other/2 per end).
    Returns nodes, tris, Celem, cut faces list [(axis,val,wall)], counted."""
    tsA = np.concatenate([[0.], np.cumsum([tk for _, tk in pliesA])]) - tA/2
    tsB = np.concatenate([[0.], np.cumsum([tk for _, tk in pliesB])]) - tB/2
    hA = min(tk for _, tk in pliesA)/nply_el
    hB = min(tk for _, tk in pliesB)/nply_el
    hmax = max(tA, tB)/2

    def ply_lines(ts, h):
        out = [np.array([0.0])]                     # midline (mirror line)
        for k in range(len(ts)-1):
            n = max(nply_el, int(round((ts[k+1]-ts[k])/h)))
            out.append(np.linspace(ts[k], ts[k+1], n+1))
        return np.unique(np.concatenate(out))

    ylin = ply_lines(tsA, hA)                       # wall A ply resolution
    xlin = ply_lines(tsB, hB)                       # wall B ply resolution
    LA, LB = Ls + extA, Ls + extB
    if topo == "L":
        xs = np.unique(np.concatenate([xlin, _grade(tB/2, LA, hB, hmax)]))
        ys = np.unique(np.concatenate([ylin, _grade(tA/2, LB, hA, hmax)]))
        cuts = [("x", LA, "A"), ("y", LB, "B")]
        counted = {"A": Ls, "B": Ls}
    elif topo == "T":
        xs = np.unique(np.concatenate([-_grade(tB/2, LA, hB, hmax)[::-1],
                                       xlin, _grade(tB/2, LA, hB, hmax)]))
        ys = np.unique(np.concatenate([-_grade(tA/2, LB, hA, hmax)[::-1],
                                       ylin]))
        cuts = [("x", -LA, "A"), ("x", LA, "A"), ("y", -LB, "B")]
        counted = {"A": 2*Ls, "B": Ls}
    else:                                           # X
        xs = np.unique(np.concatenate([-_grade(tB/2, LA, hB, hmax)[::-1],
                                       xlin, _grade(tB/2, LA, hB, hmax)]))
        ys = np.unique(np.concatenate([-_grade(tA/2, LB, hA, hmax)[::-1],
                                       ylin, _grade(tA/2, LB, hA, hmax)]))
        cuts = [("x", -LA, "A"), ("x", LA, "A"),
                ("y", -LB, "B"), ("y", LB, "B")]
        counted = {"A": 2*Ls, "B": 2*Ls}

    def inA(cx, cy):
        ok = abs(cy) < tA/2
        if topo == "L":
            ok = ok and cx > -tB/2
        return ok

    def inB(cx, cy):
        ok = abs(cx) < tB/2
        if topo == "L":
            ok = ok and cy > -tA/2
        elif topo == "T":
            ok = ok and cy < tA/2
        return ok

    def owner(cx, cy):
        a, b = inA(cx, cy), inB(cx, cy)
        if a and b:                                 # overlap block
            if topo == "L":                         # mitre diagonal
                xi = (cx + tB/2)/tB
                eta = (cy + tA/2)/tA
                return "A" if xi > eta else "B"
            if fill == "mitre4":                    # merged L-corners (ring)
                return "A" if abs(cx)/tB > abs(cy)/tA else "B"
            return "A"                              # through wall owns block
        return "A" if a else ("B" if b else None)

    def plyC(w, cx, cy):
        if w == "A":
            k = min(max(int(np.searchsorted(tsA, cy)-1), 0), len(pliesA)-1)
            return pliesA[k][0]                     # wall A frame == local
        k = min(max(int(np.searchsorted(tsB, cx)-1), 0), len(pliesB)-1)
        # wall B frame -> local: a2_B = +y3loc? B along y: tangent (0,0->)..
        QB = np.array([[1.0, 0.0, 0.0],
                       [0.0, 0.0, -1.0],
                       [0.0, 1.0, 0.0]])            # cols: axial, a2_B, n_B
        return _voigt_rotate(pliesB[k][0], QB)

    nx, ny = len(xs), len(ys)
    gid = -np.ones((nx, ny), int)
    quads = []
    for i in range(nx-1):
        for j in range(ny-1):
            cx, cy = 0.5*(xs[i]+xs[i+1]), 0.5*(ys[j]+ys[j+1])
            w = owner(cx, cy)
            if w is not None:
                quads.append((i, j, w, cx, cy))
    nid = 0
    for (i, j, w, cx, cy) in quads:
        for di, dj in ((0, 0), (1, 0), (0, 1), (1, 1)):
            if gid[i+di, j+dj] < 0:
                gid[i+di, j+dj] = nid
                nid += 1
    nodes = np.zeros((nid, 2))
    for i in range(nx):
        for j in range(ny):
            if gid[i, j] >= 0:
                nodes[gid[i, j]] = (xs[i], ys[j])
    tris, Ce, blk = [], [], []
    for (i, j, w, cx, cy) in quads:
        C = plyC(w, cx, cy)
        inblk = inA(cx, cy) and inB(cx, cy)
        n00, n10, n01, n11 = gid[i, j], gid[i+1, j], gid[i, j+1], gid[i+1, j+1]
        if (i + j) % 2 == 0:
            tris += [[n00, n10, n11], [n00, n11, n01]]
        else:
            tris += [[n00, n10, n01], [n10, n11, n01]]
        Ce += [C, C]
        blk += [inblk, inblk]
    return nodes, np.array(tris, int), Ce, cuts, counted, np.array(blk)


def _cut_bc(nodes, cuts, profA, zsA, profB, zsB, ch):
    """Dirichlet values on cut faces for local channel ch (0:e11,1:e22,2:e33):
    affine macro part + far-field fluctuation profile of the wall being cut.
    Local macro: e22 -> u2 = y2 on ch 1; e33 -> u3 = y3 on ch 2; e11 -> 0.
    Wall A profile (un,ut,u1) at n=y: (u3,u2,u1) local, wall channels
    (e11,e_tt,e_nn) = local (ch0, ch1, ch2).
    Wall B (n=-x, t=+y): un -> -u2, ut -> u3; wall channels = local
    (ch0, ch2, ch1)."""
    bc = {}
    tol = 1e-9
    for ax, val, w in cuts:
        for nd in range(len(nodes)):
            x, y = nodes[nd]
            on = (abs(x - val) < tol) if ax == "x" else (abs(y - val) < tol)
            if not on:
                continue
            # FLUCTUATION-only Dirichlet (the macro strain is the eigen load;
            # the affine part must NOT appear here)
            u = np.zeros(3)                          # (u1, u2, u3) local
            if w == "A":
                p = np.array([np.interp(y, zsA, profA[:, k, ch])
                              for k in range(3)])    # (un,ut,u1) wall A
                u[2] += p[0]                         # un -> u3
                u[1] += p[1]                         # ut -> u2
                u[0] += p[2]                         # u1
            else:
                chB = {0: 0, 1: 2, 2: 1}[ch]
                p = np.array([np.interp(-x, zsB, profB[:, k, chB])
                              for k in range(3)])
                u[1] += -p[0]                        # un -> -u2
                u[2] += p[1]                         # ut -> u3
                u[0] += p[2]
            bc[nd] = u
    return bc


def _solve_patch(nodes, tris, Ce, cuts, profA, zsA, profB, zsB, blk):
    """Solve the 3 local normal channels; return mutual-energy 3x3 E_bil.
    Lattice environment: the block-average fluctuation is constrained to zero
    (periodicity enforces wall-average strains -> zero junction fluctuation;
    consistent with the far-field profiles, which vanish at wall midlines)."""
    nn, ne = len(nodes), len(tris)
    p1, p2, p3 = nodes[tris[:, 0]], nodes[tris[:, 1]], nodes[tris[:, 2]]
    det = ((p2[:, 0]-p1[:, 0])*(p3[:, 1]-p1[:, 1])
           - (p3[:, 0]-p1[:, 0])*(p2[:, 1]-p1[:, 1]))
    area = 0.5*np.abs(det)
    bb = np.stack([p2[:, 1]-p3[:, 1], p3[:, 1]-p1[:, 1],
                   p1[:, 1]-p2[:, 1]], 1)/det[:, None]
    cc = np.stack([p3[:, 0]-p2[:, 0], p1[:, 0]-p3[:, 0],
                   p2[:, 0]-p1[:, 0]], 1)/det[:, None]
    B = np.zeros((ne, 6, 9))                        # local Voigt rows; DOF (u1,u2,u3)/node
    for k in range(3):
        B[:, 1, 3*k+1] = bb[:, k]                   # e22 = u2,2
        B[:, 2, 3*k+2] = cc[:, k]                   # e33 = u3,3
        B[:, 3, 3*k+1] = cc[:, k]; B[:, 3, 3*k+2] = bb[:, k]   # 2e23
        B[:, 4, 3*k] = cc[:, k]                     # 2e13 = u1,3
        B[:, 5, 3*k] = bb[:, k]                     # 2e12 = u1,2
    Carr = np.stack(Ce)
    Ke = np.einsum('e,eia,eij,ejb->eab', area, B, Carr, B)
    gd = (3*tris[:, :, None] + np.arange(3)[None, None, :]).reshape(ne, 9)
    ndof = 3*nn
    K = np.zeros((ndof, ndof))
    np.add.at(K, (gd[:, :, None].repeat(9, 2), gd[:, None, :].repeat(9, 1)), Ke)
    CH = [np.array([1, 0, 0, 0, 0, 0.]), np.array([0, 1, 0, 0, 0, 0.]),
          np.array([0, 0, 1, 0, 0, 0.])]
    eps_tot = []
    for ch in range(3):
        F = -np.einsum('e,eia,eij,j->ea', area, B, Carr, CH[ch])
        Fg = np.zeros(ndof)
        np.add.at(Fg, gd.ravel(), F.ravel())
        bc = _cut_bc(nodes, cuts, profA, zsA, profB, zsB, ch)
        uvals = np.zeros(ndof)
        mask = np.zeros(ndof, bool)
        for nd in sorted(bc.keys()):
            for k in range(3):
                mask[3*nd+k] = True
                uvals[3*nd+k] = bc[nd][k]
        # mirror-line constraints (lattice reflection symmetry about each
        # wall plane, exact for symmetric-material lattices under normal
        # channels): u2 = 0 on the x=0 line, u3 = 0 on the y=0 line
        tol = 1e-9
        for nd in range(len(nodes)):
            if abs(nodes[nd, 0]) < tol:
                mask[3*nd+1] = True
            if abs(nodes[nd, 1]) < tol:
                mask[3*nd+2] = True
        free = ~mask
        rhs = Fg[free] - K[np.ix_(free, mask)] @ uvals[mask]
        u = uvals.copy()
        u[free] = np.linalg.solve(K[np.ix_(free, free)], rhs)
        et = CH[ch][None, :] + np.einsum('eia,ea->ei', B, u[gd])
        eps_tot.append(et)
    E = np.zeros((3, 3))
    for c in range(3):
        for d in range(3):
            E[c, d] = float(np.einsum('e,ei,eij,ej->', area, eps_tot[c],
                                      Carr, eps_tot[d]))
    return E


def _shell_patch(topo, tA, tB, Ls, secA, secB, D_by, G_by):
    """The SAME junction patch solved with the production shell operators
    (assemble_solid_macro / assemble_solid_strip on a midline frame), cut
    nodes fluctuation-fixed to zero.  Returns mutual-energy 3x3 over the
    local normal channels -- everything the shell model CAN represent of the
    patch (membrane far field, bending escape, drilling), so that
    dC = E_solid - E_shell keeps only sub-shell junction physics."""
    from .solid_props import (assemble_solid_macro, assemble_solid_strip,
                              NDOF6)
    h_el = min(tA, tB)/2.0
    nA = max(4, int(round(Ls/h_el)))

    def line(p0, p1, n):
        return [np.array(p0) + (np.array(p1, float)-np.array(p0))*k/n
                for k in range(n+1)]

    pts, cells, secs, e3s, cutnd = [], [], [], [], []
    idx = {}

    def add(p):
        key = (round(p[0], 12), round(p[1], 12))
        if key not in idx:
            idx[key] = len(pts)
            pts.append(np.array(p, float))
        return idx[key]

    def wall(p0, p1, n, sec, e3):
        ids = [add(q) for q in line(p0, p1, n)]
        for k in range(n):
            cells.append([ids[k], ids[k+1]])
            secs.append(sec)
            e3s.append(e3)
        return ids

    nA_ = nA
    if topo == "L":
        a = wall([0, 0], [Ls, 0], nA_, secA, [0, 0, 1.0])
        b = wall([0, 0], [0, Ls], nA_, secB, [0, -1.0, 0])
        cutnd = [a[-1], b[-1]]
    elif topo == "T":
        a1 = wall([-Ls, 0], [0, 0], nA_, secA, [0, 0, 1.0])
        a2 = wall([0, 0], [Ls, 0], nA_, secA, [0, 0, 1.0])
        b = wall([0, 0], [0, -Ls], nA_, secB, [0, -1.0, 0])
        cutnd = [a1[0], a2[-1], b[-1]]
    else:
        a1 = wall([-Ls, 0], [0, 0], nA_, secA, [0, 0, 1.0])
        a2 = wall([0, 0], [Ls, 0], nA_, secA, [0, 0, 1.0])
        b1 = wall([0, 0], [0, -Ls], nA_, secB, [0, -1.0, 0])
        b2 = wall([0, 0], [0, Ls], nA_, secB, [0, -1.0, 0])
        cutnd = [a1[0], a2[-1], b1[-1], b2[-1]]

    rx = np.array([[0.0, p[0], p[1]] for p in pts])
    m = len(rx)
    cells = np.array(cells, int)
    h_st = float(np.mean(np.linalg.norm(rx[cells[:, 1]] - rx[cells[:, 0]],
                                        axis=1)))
    ez = np.array([1.0, 0.0, 0.0])
    nodes_st = np.vstack([rx, rx + h_st*ez])
    quads = np.array([[q1, q2, m+q2, m+q1] for q1, q2 in cells], int)
    rsub = np.array(secs, int)
    re3 = np.array(e3s, float)
    # prismatic tie: top node layer DOF-mapped onto the bottom (as ring_solid)
    dof_map = np.concatenate([np.arange(m), np.arange(m)])
    Dhe, Dee = assemble_solid_macro(nodes_st, quads, rsub, re3, D_by, G_by,
                                    [1, 2], 0, dof_map=dof_map)
    Dhh, Gc = assemble_solid_strip(nodes_st, quads, rsub, re3, D_by, G_by,
                                   [1, 2], 0, dof_map=dof_map)
    Dhe, Dee, Dhh, Gc = (Dhe/h_st, Dee/h_st, Dhh/h_st, Gc/h_st)
    ndof = NDOF6*m
    fixed = np.zeros(ndof, bool)
    for nd in cutnd:
        fixed[NDOF6*nd:NDOF6*nd + NDOF6] = True
    # mirror-line constraints matching the solid patch: w2 = 0 on the x=0
    # wall line, w3 = 0 on the y=0 wall line; rotations stay free
    for nd in range(m):
        if abs(rx[nd, 1]) < 1e-9:
            fixed[NDOF6*nd + 1] = True
        if abs(rx[nd, 2]) < 1e-9:
            fixed[NDOF6*nd + 2] = True
    fr = ~fixed
    Gf = Gc[:, fr]
    keep = np.linalg.norm(Gf, axis=1) > 1e-12*max(np.max(np.abs(Gf)), 1e-30)
    Gf = Gf[keep]
    nf, nc = int(fr.sum()), int(Gf.shape[0])
    A = np.zeros((nf + nc, nf + nc))
    A[:nf, :nf] = Dhh[np.ix_(fr, fr)]
    A[:nf, nf:] = Gf.T
    A[nf:, :nf] = Gf
    R = np.zeros((nf + nc, 3))
    R[:nf, :] = -Dhe[fr][:, :3]
    try:
        sol = np.linalg.solve(A, R)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(A, R, rcond=None)[0]
    q = np.zeros((ndof, 3))
    q[fr, :] = sol[:nf, :]
    E = np.zeros((3, 3))
    for c in range(3):
        for d in range(3):
            E[c, d] = (Dee[c, d] + q[:, c] @ Dhe[:, d]
                       + q[:, d] @ Dhe[:, c] + q[:, c] @ (Dhh @ q[:, d]))
    return E


def microcell_law(stackA, stackB, sections, materials, D_by, G_by,
                  fill="A", Lfac=10.0, nply_el=3, along_fac=0.5):
    """Periodic MICRO-CELL junction law for anisotropic laminates (no
    symmetry assumptions): solve the junction's own mini-lattice -- walls
    through a small periodic cell L_m = Lfac*max(t) -- with BOTH models:
        dC = (D_eff_solid_mini - D_eff_shell_mini)[:3,:3]
    shell mini: the PRODUCTION periodic ring_solid on coincident stacked
    midline elements (each member its own section ABD and +-n e3);
    solid mini: ply-resolved periodic CST with members side by side.
    Far fields cancel between the two, so dC localizes at the junction and
    is L_m-independent once L_m >> t."""
    from .solid_props import ring_solid
    from .segment_element import compute_k22
    from .periodic_multiscale import mesh_to_periodic_sparse_assembly_map

    mats = {str(m["name"]): m for m in materials}
    tsecs = [sum(float(p[1]) for p in s["layup"]) for s in sections]
    tA = sum(tsecs[s] for s, _ in stackA)
    tB = sum(tsecs[s] for s, _ in stackB)
    Lm = Lfac*max(tA, tB)
    c0 = Lm/2.0

    # ---- shell mini cell -------------------------------------------------
    # both endpoint nodes are kept and the PERIODIC MAP ties the opposite
    # faces (as the validated cross-cell shell); an explicit wrap element
    # would span the whole cell geometrically and double the wall census
    nseg = max(16, int(round(Lm/(min(tA, tB)/2.0))))
    nseg += nseg % 2                                # center node must exist
    xs = np.linspace(0.0, Lm, nseg+1)
    hpts = [(x, c0) for x in xs]
    vpts = [(c0, y) for y in xs]
    pts, idx = [], {}

    def add(p):
        key = (round(p[0], 12), round(p[1], 12))
        if key not in idx:
            idx[key] = len(pts)
            pts.append(p)
        return idx[key]

    hid = [add(p) for p in hpts]
    vid = [add(p) for p in vpts]
    cells, rsub, re3l, e2l = [], [], [], []
    for k in range(nseg):
        n1, n2 = hid[k], hid[k+1]
        for sec, mir in stackA:
            cells.append([n1, n2])
            rsub.append(sec)
            re3l.append([0.0, 0.0, -1.0] if mir else [0.0, 0.0, 1.0])
            e2l.append([0.0, 1.0, 0.0])
    for k in range(nseg):
        n1, n2 = vid[k], vid[k+1]
        for sec, mir in stackB:
            cells.append([n1, n2])
            rsub.append(sec)
            re3l.append([0.0, 1.0, 0.0] if mir else [0.0, -1.0, 0.0])
            e2l.append([0.0, 0.0, 1.0])
    rx = np.array([[0.0, p[0], p[1]] for p in pts])
    cells = np.array(cells, int)
    rsub = np.array(rsub, int)
    re3 = np.array(re3l)
    e2 = np.array(e2l)
    k22 = compute_k22(rx[cells].mean(1), e2, re3, cells)
    Dsh = ring_solid(rx, cells, rsub, re3, D_by, G_by, k22, 0, [1, 2],
                     periodic=True)

    # ---- solid mini cell (periodic CST) ---------------------------------
    def layers(stack, tw):
        z, out = -tw/2.0, []
        for sec, mir in stack:
            lay = list(sections[sec]["layup"])
            if mir:
                lay = [[p[0], p[1], -float(p[2])] for p in lay[::-1]]
            for p in lay:
                C = _ply_C_wall(mats[str(p[0])], p[2])
                out.append((z, z + float(p[1]), C))
                z += float(p[1])
        return out

    layA = layers(stackA, tA)                       # wall A: frame == local
    QB = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    # wall B's frame normal runs along -x: convert its layer intervals to
    # LOCAL x (true side-by-side placement; the member with e3 = -x, the
    # right tube's left wall, sits on the +x side)
    layB = [(-z1, -z0, _voigt_rotate(C, QB)) for z0, z1, C in
            layers(stackB, tB)]

    def lines(lay, cw):
        out = [np.array([0.0, Lm])]
        for z0, z1, _ in lay:
            n = max(nply_el, 1)
            out.append(c0 + np.linspace(z0, z1, n+1))
        gap = np.arange(cw/2 + cw/4, Lm - cw/2, cw*along_fac)
        out.append(c0 + gap)
        out.append(c0 - gap)
        return np.unique(np.round(np.concatenate(out), 12))

    xa = lines(layB, tB)
    ya = lines(layA, tA)
    xa = xa[(xa >= -1e-12) & (xa <= Lm + 1e-12)]
    ya = ya[(ya >= -1e-12) & (ya <= Lm + 1e-12)]
    nx, ny = len(xa), len(ya)
    gid = -np.ones((nx, ny), int)
    quads = []
    for i in range(nx-1):
        for j in range(ny-1):
            cx, cy = 0.5*(xa[i]+xa[i+1]) - c0, 0.5*(ya[j]+ya[j+1]) - c0
            inA_ = abs(cy) < tA/2
            inB_ = abs(cx) < tB/2
            if not (inA_ or inB_):
                continue
            if inA_ and inB_:
                if fill == "mitre4":
                    w = "A" if abs(cx)/tB > abs(cy)/tA else "B"
                elif fill == "B":
                    w = "B"
                else:
                    w = "A"
            else:
                w = "A" if inA_ else "B"
            if w == "A":
                C = next(C for z0, z1, C in layA if z0 - 1e-12 <= cy < z1)
            else:
                C = next(C for z0, z1, C in layB if z0 - 1e-12 <= cx < z1)
            quads.append((i, j, C, w))
    nid = 0
    for (i, j, _, _) in quads:
        for di, dj in ((0, 0), (1, 0), (0, 1), (1, 1)):
            if gid[i+di, j+dj] < 0:
                gid[i+di, j+dj] = nid
                nid += 1
    nd = np.zeros((nid, 2))
    for i in range(nx):
        for j in range(ny):
            if gid[i, j] >= 0:
                nd[gid[i, j]] = (xa[i], ya[j])
    # in-code structured Q4 mesh: one bilinear QUAD per grid cell (rows and
    # columns set by the junction dimensions: ply interfaces across t_A/t_B,
    # uniform divisions along the stubs) -- systematic, general, no overlap
    els, Ce, own = [], [], []
    for (i, j, C, w) in quads:
        els.append([gid[i, j], gid[i+1, j], gid[i+1, j+1], gid[i, j+1]])
        Ce.append(C)
        own.append(w)
    els = np.array(els, int)
    nn, ne = len(nd), len(els)
    ae = nd[els[:, 1], 0] - nd[els[:, 0], 0]        # rectangle x-size
    be = nd[els[:, 3], 1] - nd[els[:, 0], 1]        # rectangle y-size
    Carr = np.stack(Ce)
    ndof = 3*nn
    gd = (3*els[:, :, None] + np.arange(3)[None, None, :]).reshape(ne, 12)
    K = np.zeros((ndof, ndof))
    Dhe = np.zeros((ndof, 6))
    Dee = np.zeros((6, 6))
    gpt = 1.0/np.sqrt(3.0)
    sgn = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    for xi, eta in [(-gpt, -gpt), (gpt, -gpt), (gpt, gpt), (-gpt, gpt)]:
        wJ = ae*be/4.0
        B = np.zeros((ne, 6, 12))
        for k, (sx, sy) in enumerate(sgn):
            dNx = (sx*(1.0 + sy*eta)/4.0)*(2.0/ae)
            dNy = (sy*(1.0 + sx*xi)/4.0)*(2.0/be)
            B[:, 1, 3*k+1] = dNx
            B[:, 2, 3*k+2] = dNy
            B[:, 3, 3*k+1] = dNy; B[:, 3, 3*k+2] = dNx
            B[:, 4, 3*k] = dNy
            B[:, 5, 3*k] = dNx
        Ke = np.einsum('e,eia,eij,ejb->eab', wJ, B, Carr, B)
        Fe = np.einsum('e,eia,eij->eaj', wJ, B, Carr)
        np.add.at(K, (gd[:, :, None].repeat(12, 2),
                      gd[:, None, :].repeat(12, 1)), Ke)
        np.add.at(Dhe, gd.ravel(), Fe.reshape(-1, 6))
        Dee += np.einsum('e,eij->ij', wJ, Carr)
    rc, _ = mesh_to_periodic_sparse_assembly_map(nn, np.arange(nn)[:, None],
                                                 nd, 3, 3)
    master = np.asarray(rc, int).ravel()
    uniq, inv = np.unique(master, return_inverse=True)
    nred = 3*len(uniq)
    T = np.zeros((ndof, nred))
    for n_ in range(nn):
        for k in range(3):
            T[3*n_+k, 3*inv[n_]+k] = 1.0
    wA_ = np.zeros(nn)
    np.add.at(wA_, els.ravel(), np.repeat(ae*be/4.0, 4))
    wAr = np.zeros(len(uniq))
    np.add.at(wAr, inv, wA_)
    Cc = np.zeros((3, nred))
    Cc[0, 0::3] = wAr; Cc[1, 1::3] = wAr; Cc[2, 2::3] = wAr
    A_ = np.zeros((nred+3, nred+3))
    A_[:nred, :nred] = T.T @ K @ T
    A_[:nred, nred:] = Cc.T
    A_[nred:, :nred] = Cc
    Rr = np.zeros((nred+3, 6))
    Rr[:nred] = -(T.T @ Dhe)
    V0 = T @ np.linalg.lstsq(A_, Rr, rcond=None)[0][:nred]
    Dso = Dee + V0.T @ Dhe
    Dso = 0.5*(Dso + Dso.T)
    dC = (Dso - np.asarray(Dsh))[:3, :3]
    # global homologous-span far-field term (mini cells share one spacing)
    eA = _wall_farfield(stack_plies(stackA, sections, materials)[0])[0]
    eB = _wall_farfield(stack_plies(stackB, sections, materials)[0])[0]
    P = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0.]])
    dC = dC + (1.0 - 1.0/max(len(stackB), 1))*tB*eA \
        + (1.0 - 1.0/max(len(stackA), 1))*tA*(P.T @ eB @ P)
    return 0.5*(dC + dC.T), {"Lm": Lm, "n_tris": ne,
                             "D_shell_mini": np.asarray(Dsh),
                             "D_solid_mini": Dso, "nd": nd, "tris": els,
                             "own": own, "shell_pts": np.array(pts),
                             "shell_cells": cells}


def corner_micro_law(topo, stackA, stackB, sections, materials,
                     g_source="msg", Ls_fac=5.0, nply_el=4, fill="A"):
    """dC_j (3x3, LOCAL normal channels [e11, e22, e33]):
        dC = E_solid_patch - E_shell_patch
    with matched cut BCs (solid: far-field fluctuation profiles; shell: zero
    fluctuation) and mirror-line constraints (the lattice environment).  The
    bending escape and the wall far fields exist in BOTH patches and cancel;
    only sub-shell junction physics survives.  stackA/stackB describe the
    MERGED coincident walls of the periodic lattice: [(sec_idx, mirrored)]."""
    pliesA, tA = stack_plies(stackA, sections, materials)
    pliesB, tB = stack_plies(stackB, sections, materials)
    eA, profA, zsA = _wall_farfield(pliesA)
    eB, profB, zsB = _wall_farfield(pliesB)
    DshA, GshA = stack_shell_law(stackA, sections, materials, g_source)
    DshB, GshB = stack_shell_law(stackB, sections, materials, g_source)
    Ls = Ls_fac*max(tA, tB)
    nodes, tris, Ce, cuts, counted, blk = _patch_mesh(topo, pliesA, tA,
                                                      pliesB, tB, Ls, nply_el,
                                                      fill=fill)
    E_solid = _solve_patch(nodes, tris, Ce, cuts, profA, zsA, profB, zsB, blk)
    E_shell = _shell_patch(topo, tA, tB, Ls, 0, 1, [DshA, DshB],
                           [GshA, GshB])
    dC = E_solid - E_shell
    # homologous-span correction, ANALYTIC: the solid lattice's junction
    # spacing exceeds the midline node spacing by (1 - 1/n_other)*t_other per
    # span of adjacent-merged stacks; add that extra length of pure far-field
    # energy (1-D laminate density) -- no mesh change, no cut layers
    nendA, nendB = {"L": (1, 1), "T": (2, 1), "X": (2, 2)}[topo]
    LxA = nendA*(1.0 - 1.0/max(len(stackB), 1))*tB/2.0
    LxB = nendB*(1.0 - 1.0/max(len(stackA), 1))*tA/2.0
    P = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0.]])
    dC = dC + LxA*eA + LxB*(P.T @ eB @ P)
    return 0.5*(dC + dC.T), {"n_nodes": len(nodes), "n_tris": len(tris),
                             "Ls": Ls, "counted": counted,
                             "E_solid": E_solid, "E_shell": E_shell}
