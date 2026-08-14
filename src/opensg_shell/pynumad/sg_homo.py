"""pynumad.sg_homo -- standalone Blade -> Timoshenko + mass, fully in memory.

The pyNuMAD-route homogenizer: takes the EDITABLE Blade object, builds the
station ring arrays directly (the same station transformation the yaml
emitter performs: reference-surface offset, web chains, reference-axis
shift, orientation frames) and calls the core RM ring solver -- NO yaml,
no .out.  The yaml-artifacts route (pynumad.sg_props.beam_props over an
emitted station yaml) stays available for SG-engine studies; this module
is the in-memory blade workflow:

    from opensg_shell.pynumad.sg_homo import timo

    K, M = timo(blade, 4)              # st-id (0-based) or span r

"""
from collections import defaultdict

import numpy as np
import jax.numpy as jnp

from ..sg_assembly import assemble_segment_indep, assemble_constraint
from ..fe_jax.msg_materials import compute_ABD_matrix, shift_abd_reference
from ..fe_jax.msg_transverse_shear import transverse_shear_stiffness
from ..fe_jax.msg_rm_timo import build_C_Psi
from ..fe_jax.msg_solver import finalize_v1_and_compute_deff

_GP = np.array([-1.0, 1.0]) / np.sqrt(3.0)               # 2-pt Gauss (exact: quadratic)
_FRAC = {"center": 0.5, "oml": 0.0, "oml_flip": 1.0, "iml": 1.0}


def compute_k22(centroids, e2s, e3s, elems, flat_tol=1e-3, k22_max=None,
                cop_tol=0.5):
    """Per-element hoop curvature k22 = d e3/ds . e2 (pynumad-owned copy).

    Estimated from hoop-aligned neighbours; a COPLANARITY filter keeps
    folds (near-perpendicular normals) out of the curvature, per-neighbour
    estimates combine with the MEDIAN, flat elements snap to exactly zero.

    In:  centroids (ne,3); e2s/e3s (ne,3) element frames; elems (ne,2);
         flat_tol float snap threshold; k22_max float | None cap;
         cop_tol float coplanarity dot threshold.
    Out: k22 (ne,) float."""
    node2el = defaultdict(list)
    for ei, el in enumerate(elems):
        for n in el:
            node2el[int(n)].append(ei)
    centroids = np.asarray(centroids)
    e2s = np.asarray(e2s); e3s = np.asarray(e3s)
    k22 = np.zeros(len(elems))
    for ei in range(len(elems)):
        neigh = {ej for n in elems[ei] for ej in node2el[int(n)] if ej != ei}
        ks = []
        for ej in neigh:
            if float(np.dot(e3s[ej], e3s[ei])) < cop_tol:
                continue
            disp = centroids[ej] - centroids[ei]
            nd = float(np.linalg.norm(disp))
            if nd < 1e-12:
                continue
            ds = float(np.dot(disp, e2s[ei]))
            if abs(ds) < 0.5 * nd:
                continue
            ks.append(float(np.dot(e3s[ej] - e3s[ei], e2s[ei])) / ds)
        k = float(np.median(ks)) if ks else 0.0
        if abs(k) < flat_tol:
            k = 0.0
        elif k22_max is not None:
            k = float(np.clip(k, -k22_max, k22_max))
        k22[ei] = k
    return k22


def _material_by_section(sections, materials, center_ref=True):
    """Per-section plate ABD 6x6 + transverse-shear 2x2 (pynumad-owned copy).

    In:  sections list of dicts with 'layup' = [[mat, t, ang], ...];
         materials list of dicts with 'name' and 'elastic' {E, G, nu};
         center_ref bool -- True shifts each ABD to the laminate mid-surface.
    Out: (D_by, G_by): dicts {section index: (6,6)} / {(2,2)}."""
    matmap = {mm["name"]: {"E": mm["elastic"]["E"], "G": mm["elastic"]["G"],
                           "nu": mm["elastic"]["nu"]} for mm in materials}
    D_by, G_by = {}, {}
    for si, sec in enumerate(sections):
        layup = sec["layup"]
        mn = [p[0] for p in layup]
        th = [float(p[1]) for p in layup]
        an = [float(p[2]) for p in layup]
        abd = np.asarray(compute_ABD_matrix(th, an, mn, matmap)[0])
        if center_ref:
            abd = shift_abd_reference(abd, 0.5 * sum(th))
        D_by[si] = abd
        G_by[si] = np.asarray(transverse_shear_stiffness(th, an, mn,
                                                         matmap)[0])
    return D_by, G_by


def material_db_from_yaml(materials):
    """Material lookup dict from a yaml-style materials list (owned copy).

    In:  materials list of dicts with "name", "elastic" {E, G, nu} 3-lists
         and optional "density".
    Out: dict name -> {"E", "G", "nu" 3-lists, "rho" float (0.0 absent)}."""
    db = {}
    for m in materials:
        el = m["elastic"]
        db[m["name"]] = {"E": [float(x) for x in el["E"]],
                         "G": [float(x) for x in el["G"]],
                         "nu": [float(x) for x in el["nu"]],
                         "rho": float(m.get("density", 0.0))}
    return db


def ring_indep(rx, rcells, rsub, re3, D_by, G_by, k22_edge, ax, cross,
               h=None, shear="mitc4_g23", lam_space="elem",
               return_fields=False):
    """Constrained 6-DOF ring SG -> Timoshenko C6 (pynumad-owned driver).

    The one-quad-deep dof-mapped prismatic strip over the contour, the
    element-constant drilling Lagrange constraint, one dense LU of the KKT
    for the V0 AND V1 solves, and the generalized-Timoshenko finalization
    (all energy blocks /h).  Dense LU is deliberate: the saddle point
    carries a near-null drilling/kernel mode that a sparse factorization
    resolves along a different direction.

    In:  rx (m,3) contour nodes; rcells (ne,2) edges; rsub (ne,) section
         id; re3 (ne,3) wall normals; D_by/G_by per-section ABD 6x6 /
         G 2x2; k22_edge (ne,); ax int axial index; cross list[2];
         h float | None strip depth (None = mean edge length); shear str
         ('mitc4_g23' = the production ring scheme: tie ONLY gamma_23);
         lam_space str; return_fields bool.
    Out: C6 (6,6); with return_fields also V0, V1 (6m,4) warping fields
         (multiplier rows stripped, drilling omega_3 included)."""
    m = len(rx)
    if h is None:
        h = float(np.mean(np.linalg.norm(rx[rcells[:, 1]] - rx[rcells[:, 0]],
                                         axis=1)))
    ez = np.zeros(3); ez[ax] = 1.0
    nodes = np.vstack([rx, rx + h * ez])
    dof_map = np.concatenate([np.arange(m), np.arange(m)])
    quads = np.array([[a, b, m + b, m + a] for a, b in rcells], dtype=int)
    e3q = np.asarray(re3)

    Dhh, Dhe, Dee, Dhl, Dll, Dle = assemble_segment_indep(
        nodes, quads, rsub, e3q, D_by, G_by, np.asarray(k22_edge), cross, ax,
        kg_e=None, dof_map=dof_map, shear=shear)
    Gc, Gl, Ge = assemble_constraint(nodes, quads, rsub, e3q,
                                     np.asarray(k22_edge), cross, ax,
                                     dof_map=dof_map, lam_space=lam_space)
    Dhh, Dhe, Dhl, Dll, Dle = [np.asarray(A) / h
                               for A in (Dhh, Dhe, Dhl, Dll, Dle)]
    Dee = np.asarray(Dee) / h
    Gc, Gl, Ge = Gc / h, Gl / h, Ge / h

    M = Dhh.shape[0]; P = Gc.shape[0]
    # 5-DOF rigid kernel embedded into 6 DOF (om3 rigid-free)
    C5, Psi5 = build_C_Psi(rx[:, cross], rcells, p=1)
    C6 = np.zeros((4, M)); Psi6 = np.zeros((M, 4))
    for n in range(m):
        C6[:, 6 * n:6 * n + 5] = C5[:, 5 * n:5 * n + 5]
        Psi6[6 * n:6 * n + 5, :] = Psi5[5 * n:5 * n + 5, :]
    Psi6[3::6, 3] *= -1.0                    # validated-kernel om1 sign flip

    naug = M + P
    Dhe_a = np.vstack([Dhe, Ge])
    Dle_a = np.vstack([Dle, np.zeros((P, 4))])
    Psi_a = np.vstack([Psi6, np.zeros((P, 4))])
    Dc_a = np.vstack([C6.T, np.zeros((P, 4))])

    from scipy.linalg import lu_factor, lu_solve
    A = np.zeros((naug + 4, naug + 4))
    A[:M, :M] = Dhh
    A[:M, M:naug] = Gc.T; A[M:naug, :M] = Gc
    A[:M, naug:] = C6.T; A[naug:, :M] = C6
    Alu = lu_factor(A)
    R0 = np.zeros((naug + 4, 4)); R0[:naug] = -Dhe_a
    V0 = lu_solve(Alu, R0)[:naug]
    Deff = Dee + V0.T @ Dhe_a
    # V1 RHS projected onto range(Dhh) (block form of prepare_v1_rhs)
    V0m, V0p = V0[:M], V0[M:naug]
    DhlV0 = np.vstack([Dhl @ V0m, Gl @ V0m])
    DhlTV0Dle = np.vstack([Dhl.T @ V0m + Gl.T @ V0p,
                           np.zeros((P, 4))]) + Dle_a
    V0DllV0 = V0m.T @ (Dll @ V0m)
    b_unproj = DhlV0 - DhlTV0Dle
    bb = Dc_a @ (np.linalg.inv(Psi_a.T @ Dc_a) @ (Psi_a.T @ b_unproj)) \
        - b_unproj
    R1 = np.zeros((naug + 4, 4)); R1[:naug] = bb
    V1 = lu_solve(Alu, R1)[:naug]
    C6r, *_ = finalize_v1_and_compute_deff(
        jnp.array(V1), jnp.array(V0), jnp.array(Deff),
        jnp.array(V0DllV0), jnp.array(DhlV0), jnp.array(DhlTV0Dle),
        jnp.array(Psi_a), jnp.array(Dc_a))
    C6r = np.asarray(C6r)
    C6r = 0.5 * (C6r + C6r.T)
    if return_fields:
        return C6r, np.asarray(V0[:M]), np.asarray(V1[:M])
    return C6r


def _mat_card(blade, name):
    """OpenSG material card (elastic-nested 3-lists) from the blade reader.

    In:  blade: reader (for blade.mats); name: str material name.
    Out: dict(name, density, elastic={E, G, nu} 3-lists)."""
    m = blade.mats[name]
    E, G, nu = m.get("E"), m.get("G"), m.get("nu")
    if not isinstance(E, (list, tuple)):                 # isotropic -> replicate
        nu = 0.3 if nu is None else nu
        G = E / (2.0 * (1.0 + nu)) if G is None else G
        E = [E, E, E]; G = [G, G, G]; nu = [nu, nu, nu]
    return dict(name=name, density=float(m.get("rho", 1.0)),
                elastic=dict(E=[float(x) for x in E], G=[float(x) for x in G],
                             nu=[float(x) for x in nu]))


def station_arrays(cs, reference="oml", web_mesh=None):
    """The station transformation, straight to ring arrays (no file).

    Performs exactly what the yaml emitter does before dumping: OML ->
    reference-surface offset of the skin, web chain meshing, chordwise
    shift to the windIO reference axis, per-element e2/e3 frames with the
    closed-loop inward normal -- then quantizes the node coordinates to
    the emitter's own %.8f so the file route and this route are
    digit-identical.

    In:
        cs: dict from blade.cross_section (build_cross_section schema).
        reference: "center" | "oml", shell reference surface.
        web_mesh: float | None, web element length [m] (default 0.01*chord).
    Out:
        dict: "rx" (n,3) node coords (z=0); "cells" (ne,2) 0-based;
        "rsub" (ne,) section index; "re2"/"re3" (ne,3) element frames;
        "sections" list of yaml-style section dicts; "materials" list of
        yaml-style material cards; "n_webs" int.
    """
    blade = cs["blade"]; chord = cs["chord"]
    fraction = {"center": 0.5, "oml": 0.0}[reference]
    nodes = [np.asarray(p, float) for p in cs["nodes"]]
    elems = list(cs["elems"]); elem_lam = list(cs["elem_lam"])
    web_mesh = web_mesh if web_mesh else 0.01 * chord
    set_of_lam = {v: k for k, v in cs["laminates"].items()}

    # reference-surface offset: OML -> laminate reference surface at `fraction`
    nskin = len(cs["nodes"])
    web_sets_pre = {w["lam"] for w in cs["webs"]}
    skin_xy0 = np.asarray(cs["nodes"])
    area2 = float(np.sum(skin_xy0[:, 0] * np.roll(skin_xy0[:, 1], -1)
                         - np.roll(skin_xy0[:, 0], -1) * skin_xy0[:, 1]))
    inward = 1.0 if area2 > 0 else -1.0                  # e3=[-e2y,e2x] inward for a CCW loop
    if fraction:
        acc_n = np.zeros((nskin, 2)); acc_t = np.zeros(nskin); acc_c = np.zeros(nskin)
        for ei, (n1, n2) in enumerate(elems):
            if elem_lam[ei] in web_sets_pre:
                continue
            e2 = nodes[n2] - nodes[n1]; e2 = e2 / (np.linalg.norm(e2) + 1e-30)
            nin = inward * np.array([-e2[1], e2[0]])
            tlam = float(sum(t for (_m, t, _a) in set_of_lam[elem_lam[ei]]))
            for nd in (n1, n2):
                if nd < nskin:
                    acc_n[nd] += nin; acc_t[nd] += tlam; acc_c[nd] += 1.0
        offs = [np.asarray(p, float) for p in nodes]
        for i in range(nskin):
            if acc_c[i] > 0 and np.linalg.norm(acc_n[i]) > 1e-12:
                nrm = acc_n[i] / np.linalg.norm(acc_n[i])
                offs[i] = nodes[i] + fraction * (acc_t[i] / acc_c[i]) * nrm
        nodes = offs

    # web chains: bottom(-y3) -> top(+y3) so e2 = +y3, e3 = e1 x e2 = -y2
    for w in cs["webs"]:
        a, b = w["a"], w["b"]
        if nodes[a][1] > nodes[b][1]:
            a, b = b, a
        Pa, Pb = nodes[a].copy(), nodes[b].copy()
        nseg = max(2, int(round(np.linalg.norm(Pb - Pa) / web_mesh)))
        ts = np.linspace(0, 1, nseg + 1)
        chain = [a]
        for t in ts[1:-1]:
            nodes.append(Pa + t * (Pb - Pa)); chain.append(len(nodes) - 1)
        chain.append(b)
        for ia, ib in zip(chain[:-1], chain[1:]):
            elems.append((ia, ib)); elem_lam.append(w["lam"])

    # reference-axis origin: chordwise shift LE -> x1
    dx = blade.offset_y(cs["r"])
    nodes = np.array(nodes); nodes[:, 0] -= dx

    # element frames from the UN-quantized nodes (the emitter writes these
    # at full float precision), inward normal for the skin only
    nsets = len(cs["laminates"])
    web_sets = {w["lam"] for w in cs["webs"]}
    ne = len(elems)
    re2 = np.zeros((ne, 3)); re3 = np.zeros((ne, 3))
    for ei, (n1, n2) in enumerate(elems):
        P1, P2 = nodes[n1], nodes[n2]
        e2 = (P2 - P1) / (np.linalg.norm(P2 - P1) + 1e-30)
        e3 = np.array([-e2[1], e2[0]])
        if elem_lam[ei] not in web_sets:
            e3 = inward * e3
        re2[ei, :2] = e2; re3[ei, :2] = e3

    # node quantization: exactly the emitter's "%.8f" print
    rx = np.array([[float("%.8f" % v) for v in p] + [0.0] for p in nodes])
    cells = np.array(elems, dtype=int)

    sections = [dict(elementSet="layup_%d" % k,
                     layup=[[mat, float(t), float(ang)]
                            for (mat, t, ang) in set_of_lam[k]])
                for k in range(nsets)]
    used = []
    for k in range(nsets):
        for (mat, _t, _a) in set_of_lam[k]:
            if mat not in used:
                used.append(mat)
    materials = [_mat_card(blade, name) for name in used]
    return dict(rx=rx, cells=cells, rsub=np.asarray(elem_lam, int),
                re2=re2, re3=re3, sections=sections, materials=materials,
                n_webs=len(cs["webs"]))


def ring_laws(sections, materials, reference):
    """Per-section wall laws: ABD 6x6 at the chosen reference + the MSG G.

    The same construction the file route performs (loader ABD/G, then the
    Yu-2002 least-squares G with the SPD acceptance guard).

    In:
        sections: list of yaml-style section dicts (layup lists).
        materials: list of yaml-style material cards.
        reference: "center" | "oml" | "oml_flip" | "iml".
    Out:
        (D_by, G_by): per-section ABD (6,6) container and G (2,2) list.
    """
    if reference == "center":
        D_by, G_by = _material_by_section(sections, materials, center_ref=True)
    else:
        D_by, G_by = _material_by_section(sections, materials,
                                          center_ref=False)
        if reference in ("oml_flip", "iml"):
            for si, sec in enumerate(sections):
                t = sum(float(p[1]) for p in sec["layup"])
                D_by[si] = shift_abd_reference(np.asarray(D_by[si]), t)
    frac = _FRAC.get(reference, 0.0)
    G_by = [np.asarray(G_by[si], float) for si in range(len(sections))]
    from opensg_solid.rm_plate_1D.msg_rm_plate import rm_plate_msg
    _mdb = material_db_from_yaml(materials)
    for si, sec in enumerate(sections):
        _pl = [[str(p[0]), float(p[1]), float(p[2])] for p in sec["layup"]]
        _rr = rm_plate_msg([p[1] for p in _pl], [p[2] for p in _pl],
                           [p[0] for p in _pl], _mdb, fraction=frac)
        # accept the MSG G only when it is a finite SPD 2x2; otherwise keep
        # the energy-consistent loader G (borderline U*-fits on thin tips)
        _Gm = _rr["G_msg"]
        if _Gm is not None:
            _Gm = np.asarray(_Gm, float)
            if (_Gm.shape == (2, 2) and np.all(np.isfinite(_Gm))
                    and np.linalg.det(_Gm) > 0.0 and _Gm[0, 0] > 0.0):
                G_by[si] = _Gm
    return D_by, G_by


def _density_moments(layup, rho, frac):
    """Through-thickness density moments of one laminate about its reference.

    In:  layup list of (mat, t, ang) outer -> inner; rho {mat: density};
         frac float reference fraction (0 = OML .. 1 = IML).
    Out: (mu, mx3, i22): int rho dz, int rho z dz, int rho z^2 dz."""
    h = sum(t for (_m, t, _a) in layup)
    mu = mx3 = i22 = 0.0
    z0 = -frac * h
    for (m, t, _a) in layup:
        zb = z0 + 0.5 * t
        r = rho[m]
        mu += r * t
        mx3 += r * t * zb
        i22 += r * (t ** 3 / 12.0 + t * zb * zb)
        z0 += t
    return mu, mx3, i22


def mass_ring(rx, cells, re3, rsub, sections, materials, reference):
    """6x6 mass matrix of the ring (VABS frame, about the section origin).

    The same 2-pt Gauss integration as the file route's mass_matrix_ring,
    on the in-memory arrays.

    In:
        rx (n,3), cells (ne,2), re3 (ne,3), rsub (ne,): station_arrays out.
        sections/materials: yaml-style lists (laminates + density cards).
        reference: str, sets the reference fraction for the moments.
    Out:
        (M (6,6), info dict) -- mass_matrix_ring's schema (mass_center,
        M_center, principal inertias, mpus, area, geometric_center, ref).
    """
    frac = _FRAC.get(reference, 0.0)
    rho = {m["name"]: float(m["density"]) for m in materials}
    layups = [[(str(p[0]), float(p[1]), float(p[2])) for p in s["layup"]]
              for s in sections]
    x = rx[:, :2]; nrm = re3[:, :2]; sec = np.asarray(rsub, int)
    mom = np.array([_density_moments(L, rho, frac) for L in layups])
    geo = np.array([_density_moments(L, {m: 1.0 for m in rho}, frac)
                    for L in layups])

    P1 = jnp.asarray(x[cells[:, 0]]); P2 = jnp.asarray(x[cells[:, 1]])
    Le = jnp.linalg.norm(P2 - P1, axis=1)
    mid = 0.5 * (P1 + P2); half = 0.5 * (P2 - P1)
    n2 = jnp.asarray(nrm[:, 0]); n3 = jnp.asarray(nrm[:, 1])
    mu = jnp.asarray(mom[sec, 0]); mx3 = jnp.asarray(mom[sec, 1])
    i22 = jnp.asarray(mom[sec, 2])
    a0 = jnp.asarray(geo[sec, 0]); az = jnp.asarray(geo[sec, 1])

    m = S2 = S3 = I22 = I33 = I23 = A = G2 = G3 = 0.0
    for g in _GP:
        w = 0.5 * Le
        x2 = mid[:, 0] + g * half[:, 0]
        x3 = mid[:, 1] + g * half[:, 1]
        m += jnp.sum(w * mu)
        S2 += jnp.sum(w * (x2 * mu + n2 * mx3))
        S3 += jnp.sum(w * (x3 * mu + n3 * mx3))
        I22 += jnp.sum(w * (mu * x3 ** 2 + 2.0 * x3 * n3 * mx3 + n3 ** 2 * i22))
        I33 += jnp.sum(w * (mu * x2 ** 2 + 2.0 * x2 * n2 * mx3 + n2 ** 2 * i22))
        I23 += jnp.sum(w * (mu * x2 * x3 + (x2 * n3 + x3 * n2) * mx3
                            + n2 * n3 * i22))
        A += jnp.sum(w * a0)
        G2 += jnp.sum(w * (x2 * a0 + n2 * az))
        G3 += jnp.sum(w * (x3 * a0 + n3 * az))
    m, S2, S3, I22, I33, I23, A, G2, G3 = (float(v) for v in
                                           (m, S2, S3, I22, I33, I23,
                                            A, G2, G3))

    M = np.array([[m,   0,   0,   0,        S3,  -S2],
                  [0,   m,   0,  -S3,       0,    0],
                  [0,   0,   m,   S2,       0,    0],
                  [0,  -S3,  S2,  I22 + I33, 0,   0],
                  [S3,  0,   0,   0,        I22, -I23],
                  [-S2, 0,   0,   0,       -I23,  I33]])

    x2m, x3m = S2 / m, S3 / m
    I22c = I22 - m * x3m ** 2
    I33c = I33 - m * x2m ** 2
    I23c = I23 - m * x2m * x3m
    Mc = np.array([[m, 0, 0, 0, 0, 0], [0, m, 0, 0, 0, 0], [0, 0, m, 0, 0, 0],
                   [0, 0, 0, I22c + I33c, 0, 0], [0, 0, 0, 0, I22c, -I23c],
                   [0, 0, 0, 0, -I23c, I33c]])
    T = np.array([[I22c, -I23c], [-I23c, I33c]])
    wv, V = np.linalg.eigh(T)
    ang = float(np.degrees(np.arctan2(V[1, 0], V[0, 0]))) % 180.0
    info = dict(mass_center=np.array([x2m, x3m]), M_center=Mc, mpus=m,
                i11=I22c + I33c, i22p=float(wv[0]), i33p=float(wv[1]),
                angle_deg=ang,
                rgyr=float(np.sqrt((I22c + I33c) / m)), area=A,
                geometric_center=np.array([G2 / A, G3 / A]), ref=reference)
    return M, info


def timo_cs(cs, reference, shear="mitc4_g23"):
    """Cross-section dict -> Timoshenko 6x6 + mass 6x6, fully in memory.

    The homogenization core both entries share: station arrays, ring laws,
    hoop curvature, the ring driver, the ring mass.

    In:
        cs: dict from build_cross_section (any reader).
        reference: "center" | "oml", shell reference surface.
        shear: str, RM transverse-shear tying scheme (production default).
    Out:
        (K (6,6), M (6,6), info dict, arrays dict) -- stiffness (VABS
        order), mass matrix, mass/geometry info, and the station_arrays
        output (rx/cells/sections/... for mesh bookkeeping).
    """
    A = station_arrays(cs, reference=reference)
    D_by, G_by = ring_laws(A["sections"], A["materials"], reference)
    k22 = compute_k22(A["rx"][A["cells"]].mean(1), A["re2"], A["re3"],
                      A["cells"])
    C6 = ring_indep(A["rx"], A["cells"], A["rsub"], A["re3"], D_by, G_by,
                    k22, 2, [0, 1], shear=shear, lam_space="elem")
    C6 = 0.5 * (np.asarray(C6) + np.asarray(C6).T)
    M6, info = mass_ring(A["rx"], A["cells"], A["re3"], A["rsub"],
                         A["sections"], A["materials"], reference)
    return C6, M6, info, A


def timo(blade, st, shear="mitc4_g23", full=False):
    """Blade -> Timoshenko 6x6 + mass 6x6, fully in memory.

    The bypass the optimization loop wants: no yaml, no .out, no windio --
    the Blade's own cross-section (Section overrides included) goes
    straight into the ring solver.

        from opensg_shell.pynumad.sg_homo import timo
        K, M = timo(blade, 4)

    In:
        blade: opensg_shell.Blade (any dialect the Blade reads).
        st: st-id token (0-based station index | span r | str).
        shear: str, RM transverse-shear tying scheme (production default).
        full: bool, True additionally returns the mass/geometry info dict.
    Out:
        (K, M) -- (6,6) Timoshenko stiffness (VABS order) and mass matrix;
        with full=True, (K, M, info).
    """
    r = blade.resolve(st)
    C6, M6, info, _ = timo_cs(blade.cross_section(r), "oml", shear=shear)
    return (C6, M6, info) if full else (C6, M6)


def _principal_2x2(J22, J33, J23):
    """Principal values/axis of the plane tensor [[J22, -J23], [-J23, J33]].

    In:  J22, J33, J23 floats -- the plane moments about the center.
    Out: (j22p, j33p, angle_deg) -- j22p the smaller principal value;
         angle_deg the rotation of its axis from +x2 about +x1 in
         [0, 180)."""
    T = np.array([[J22, -J23], [-J23, J33]])
    w, V = np.linalg.eigh(T)
    return (float(w[0]), float(w[1]),
            float(np.degrees(np.arctan2(V[1, 0], V[0, 0]))) % 180.0)


def _fmt66(Mtx, fmt="  %18.10E"):
    """6x6 -> VABS-style text rows.

    In:  Mtx (6,6); fmt str per-entry format.
    Out: str, 6 joined rows."""
    return "\n".join("".join(fmt % Mtx[i, j] for j in range(6))
                     for i in range(6))


def write_vabs_k(path, K, M, info, model="OpenSG msg-shell beam model",
                 solve_time=None):
    """Write a VABS .K-layout output (pynumad-owned copy): mass blocks +
    Timoshenko stiffness/compliance + center/principal summaries.

    In:
        path: str, output file.
        K: (6,6) Timoshenko stiffness (VABS order).
        M: (6,6) mass matrix; info: dict from mass_ring.
        model: str banner; solve_time: float | None seconds.
    Out:
        str path.
    """
    K = np.asarray(K, float); S = np.linalg.inv(K)
    bar = " ========================================================\n"
    L = [" %s [F1 F2 F3 M1 M2 M3] (1 = axial)\n" % model]
    L.append("\n The 6X6 Mass Matrix\n" + bar + "\n" + _fmt66(M) + "\n")
    L.append("\n The Mass Center of the Cross Section\n" + bar +
             "\n  %18.10E  %18.10E\n" % tuple(info["mass_center"]))
    L.append("\n The 6X6 Mass Matrix at the Mass Center\n" + bar + "\n"
             + _fmt66(info["M_center"]) + "\n")
    L.append("\n The Mass Properties with respect to Principal Inertial"
             " Axes\n" + bar +
             "\n Mass per unit span                     =  %14.10E\n"
             % info["mpus"] +
             " Mass moment of inertia i11             =  %14.10E\n"
             % info["i11"] +
             " Principal mass moments of inertia i22  =  %14.10E\n"
             % info["i22p"] +
             " Principal mass moments of inertia i33  =  %14.10E\n"
             % info["i33p"] +
             " The principal inertial axes rotated from user coordinate"
             " system by\n"
             "   %.10f degrees about the positive direction of x1 axis.\n"
             % info["angle_deg"] +
             " The mass-weighted radius of gyration   =  %14.10E\n"
             % info["rgyr"])
    L.append("\n The Geometric Center of the Cross Section\n" + bar +
             "\n  %18.10E  %18.10E\n" % tuple(info["geometric_center"]))
    L.append("\n The Area of the Cross Section\n" + bar +
             "\n Area =  %18.10E\n" % info["area"])
    L.append("\n Timoshenko Stiffness Matrix (1-extension; 2,3-shear,"
             " 4-twist; 5,6-bending)\n" + bar + "\n" + _fmt66(K) + "\n")
    L.append("\n Timoshenko Compliance Matrix (1-extension; 2,3-shear,"
             " 4-twist; 5,6-bending)\n" + bar + "\n" + _fmt66(S) + "\n")

    x2t, x3t = -K[0, 5] / K[0, 0], K[0, 4] / K[0, 0]
    L.append("\n The Tension Center of the Cross Section\n" + bar +
             "\n  %18.10E  %18.10E\n" % (x2t, x3t))
    EI22c = K[4, 4] - K[0, 4] ** 2 / K[0, 0]
    EI33c = K[5, 5] - K[0, 5] ** 2 / K[0, 0]
    EI23c = K[4, 5] - K[0, 4] * K[0, 5] / K[0, 0]
    ei22p, ei33p, eang = _principal_2x2(EI22c, EI33c, -EI23c)
    L.append("\n The extension stiffness EA           %18.10E\n" % K[0, 0] +
             " The torsional stiffness GJ           %18.10E\n" % K[3, 3] +
             " Principal bending stiffness EI22     %18.10E\n" % ei22p +
             " Principal bending stiffness EI33     %18.10E\n" % ei33p +
             " The principal bending axes rotated from the user coordinate"
             " system by\n"
             "   %.10f degrees about the positive direction of x1 axis.\n"
             % eang)
    x2s, x3s = -S[2, 3] / S[3, 3], S[1, 3] / S[3, 3]
    L.append("\n The Shear Center of the Cross Section\n" + bar +
             "\n  %18.10E  %18.10E\n" % (x2s, x3s))
    ga22p, ga33p, gang = _principal_2x2(K[1, 1], K[2, 2], -K[1, 2])
    L.append("\n Principal shear stiffness GA22 =  %18.10E\n" % ga22p +
             " Principal shear stiffness GA33 =  %18.10E\n" % ga33p +
             " The principal shear axes rotated from user coordinate"
             " system by\n"
             "   %.10f degrees about the positive direction of x1 axis.\n"
             % gang)
    if solve_time is not None:
        L.append("\n Time taken: %.2f sec\n" % solve_time)
    open(path, "w").write("".join(L))
    return path
