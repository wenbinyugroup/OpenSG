"""pynumad.sg_homo -- standalone Blade -> Timoshenko + mass, fully in memory.

The pyNuMAD-route homogenizer: takes the EDITABLE Blade object, builds the
station ring arrays directly (the same station transformation the yaml
emitter performs: reference-surface offset, web chains, reference-axis
shift, orientation frames) and calls the core RM ring solver -- NO yaml,
no .out, no windio import.  The file-based pipeline (opensg_shell.windio)
is untouched; this module is the in-memory pyNuMAD workflow:

    from opensg_shell.pynumad.sg_homo import timo

    K, M = timo(blade, 4)              # st-id (0-based) or span r

Digit-parity with the file route: node coordinates are quantized exactly
as the yaml emitter prints them (%.8f), and the element frames are built
from the un-quantized nodes exactly as the emitter writes them, so the
ring solver sees the same numbers the yaml round-trip would deliver.
Section overrides (blade.section) are honored because the cross-section
comes from blade.cross_section.
"""
import numpy as np
import jax.numpy as jnp

from ..sg_assembly import compute_k22
from ..sg_homo import ring_indep
from ..sg_materials import _material_by_section, material_db_from_yaml
from ..fe_jax.msg_materials import shift_abd_reference

_GP = np.array([-1.0, 1.0]) / np.sqrt(3.0)               # 2-pt Gauss (exact: quadratic)
_FRAC = {"center": 0.5, "oml": 0.0, "oml_flip": 1.0, "iml": 1.0}


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


def timo(blade, st, shear="mitc4_g23", full=False):
    """Blade -> Timoshenko 6x6 + mass 6x6, fully in memory.

    The bypass the optimization loop wants: no yaml, no .out, no windio --
    the Blade's own cross-section (Section overrides included) goes
    straight into the core RM ring solver.

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
    A = station_arrays(blade.cross_section(r), reference=blade.reference)
    D_by, G_by = ring_laws(A["sections"], A["materials"], blade.reference)
    k22 = compute_k22(A["rx"][A["cells"]].mean(1), A["re2"], A["re3"],
                      A["cells"])
    C6 = ring_indep(A["rx"], A["cells"], A["rsub"], A["re3"], D_by, G_by,
                    k22, 2, [0, 1], shear=shear, lam_space="elem")
    C6 = 0.5 * (np.asarray(C6) + np.asarray(C6).T)
    M6, info = mass_ring(A["rx"], A["cells"], A["re3"], A["rsub"],
                         A["sections"], A["materials"], blade.reference)
    return (C6, M6, info) if full else (C6, M6)
