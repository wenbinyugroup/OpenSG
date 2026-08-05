"""
MSG Shell Timoshenko — Hermite C1 cubic elements (the mandatory JAX-TW method).

Displacement interpolation along the cross-section arc is **Hermite C1 cubic**:
each corner node carries value + arc-slope DOFs for all three displacement
components  ->  6 DOF/node, 12 DOF/element.  Because the slope is continuous
across elements (C1), no interior penalty is needed for the V1 (shear-warping)
solve.

Geometry is the quadratic line cross-section: the corner nodes give the chord
tangent / arc length, and the curvature ``k22`` comes from the 3-node geometry
via :func:`mesh_curvature` (flat 2-node -> 0, curved 3-node -> -1/R).  The
strain operators ``gamma_h`` / ``gamma_l`` / ``gamma_e`` are identical to the
quadratic module; only the shape functions and DOF layout differ.

Frame: VABS plane cross-section (e1 = [1,0,0] beam axis, e2 = geometric tangent,
e3 = e1 x e2).

The KKT solve (:func:`solve_fluctuation_field`), V1 RHS
(:func:`prepare_v1_rhs`) and Timoshenko assembly
(:func:`finalize_v1_and_compute_deff`) are reused unchanged from ``msg_shell``.
"""
import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse
import numpy as np

jax.config.update("jax_enable_x64", True)

# component DOF indices within the 12-DOF element [n0: w1,w1',w2,w2',w3,w3'; n1: ...]
_I1 = jnp.array([0, 1, 6, 7])    # w1 value+slope at the two corners
_I2 = jnp.array([2, 3, 8, 9])    # w2
_I3 = jnp.array([4, 5, 10, 11])  # w3


# =============================================================================
# Hermite C1 cubic shape functions
# =============================================================================

def hermite_shape_functions(xi_q, L):
    """Cubic Hermite N, N', N'' on xi in [0, 1] for DOFs [v0, s0, v1, s1].

    psi1 = 1-3z^2+2z^3, psi2 = z-2z^2+z^3, psi3 = 3z^2-2z^3, psi4 = -z^2+z^3;
    slope shape functions are scaled by the element length L so the slope DOF
    is the physical arc derivative dw/ds.

    Returns phi_val, phi_d1 (d/ds), phi_d2 (d^2/ds^2), each (Q, 4).
    """
    t = xi_q; t2 = t * t; t3 = t2 * t
    phi_val = jnp.stack([1 - 3*t2 + 2*t3, L*(t - 2*t2 + t3),
                         3*t2 - 2*t3,     L*(-t2 + t3)], axis=1)
    phi_d1 = jnp.stack([(-6*t + 6*t2)/L, 1 - 4*t + 3*t2,
                        (6*t - 6*t2)/L,  -2*t + 3*t2], axis=1)
    phi_d2 = jnp.stack([(-6 + 12*t)/L**2, (-4 + 6*t)/L,
                        (6 - 12*t)/L**2,  (-2 + 6*t)/L], axis=1)
    return phi_val, phi_d1, phi_d2


# =============================================================================
# Hermite mesh + DOF map (corner nodes only; 6 DOF/node)
# =============================================================================

def make_hermite_mesh(nodes_2d, cells):
    """Corner nodes + 2-node connectivity from the quadratic (3-node) mesh.

    The displacement lives on the corner nodes (even rows of ``nodes_2d``); the
    midside node is used only for geometry/curvature.

    Returns
    -------
    corners : (n_elem+1, 2)  corner coordinates (last == first if closed)
    hcells  : (n_elem, 2)    corner connectivity [corner_i, corner_{i+1}]
    """
    n_elem = cells.shape[0]
    corners = np.asarray(nodes_2d)[0::2]
    hcells = np.column_stack([np.arange(n_elem), np.arange(1, n_elem + 1)]).astype(np.int64)
    return corners, hcells


def build_hermite_dof_map(n_corner, is_closed=True):
    """Periodic DOF map for Hermite corners (6 DOF/node, value+slope x3)."""
    dof_map = np.arange(n_corner * 6, dtype=np.int64)
    if is_closed:
        last = n_corner - 1
        for k in range(6):
            dof_map[last * 6 + k] = k       # merge last corner -> first (value & slope)
        n_unique = n_corner - 1
    else:
        n_unique = n_corner
    return dof_map, n_unique


def compress_hermite_dofs(dof_map, hcells):
    """Renumber Hermite corner DOFs to unique ids; returns reduced cells & n_primal."""
    master = dof_map[::6] // 6
    unique = np.unique(master)
    f2r = np.full(len(master), -1, dtype=np.int64)
    for i, m in enumerate(unique):
        f2r[m] = i
    reduced_cells = f2r[master[hcells]]
    return reduced_cells, len(unique) * 6


# =============================================================================
# Shell strain operators (shared by assembly and dehomogenization)
# =============================================================================

def hermite_strain_operators(n0, n1, k22, L, xd2, xd3, xi_q):
    """Per-element MSG shell strain operators at the quadrature points.

    Returns three callables ``eps_h(u)``, ``eps_l(u)``, ``eps_e(ue)`` and the
    macro-strain map ``Ge`` (Q, 6, 4).  All produce the 6-component shell strain
    [eps11, eps22, gamma12, kappa11, kappa22, kappa12] per quad point:

    * ``eps_h(u)``  — bending fluctuation from the 12-DOF Hermite warping ``u``
    * ``eps_l(u)``  — Timoshenko shear-warping operator
    * ``eps_e(ue)`` — Ge @ ue, the macro EB strain map (ue = [eps11,k1,k2,k3])

    This is the single source of truth for the strain kinematics: both
    :func:`assemble_system_matrices_hermite` (which builds Dhh/.../Dle by
    autodiff of the strain energy) and the dehomogenization shell-strain
    recovery call it, so the two never diverge.
    """
    Q_pts = xi_q.shape[0]
    phi_val, phi_d1, phi_d2 = hermite_shape_functions(xi_q, L)
    x2_q = (1.0 - xi_q) * n0[0] + xi_q * n1[0]
    x3_q = (1.0 - xi_q) * n0[1] + xi_q * n1[1]
    Rn = x2_q * xd3 - x3_q * xd2

    Ge = jnp.zeros((Q_pts, 6, 4))
    Ge = Ge.at[:, 0, 0].set(1.0)
    Ge = Ge.at[:, 0, 2].set(x3_q)
    Ge = Ge.at[:, 0, 3].set(-x2_q)
    Ge = Ge.at[:, 2, 1].set(Rn)
    Ge = Ge.at[:, 3, 2].set(xd2)
    Ge = Ge.at[:, 3, 3].set(xd3)
    Ge = Ge.at[:, 5, 1].set(-2.0 + k22 / 2.0 * Rn)

    def eps_h(u):
        dw1 = phi_d1 @ u[_I1]; dw2 = phi_d1 @ u[_I2]; dw3 = phi_d1 @ u[_I3]
        d2w2 = phi_d2 @ u[_I2]; d2w3 = phi_d2 @ u[_I3]
        return jnp.stack([
            jnp.zeros(Q_pts), xd2*dw2 + xd3*dw3, dw1, jnp.zeros(Q_pts),
            -k22*xd2*dw2 + xd3*d2w2 - k22*xd3*dw3 - xd2*d2w3,
            (k22/2.0)*dw1], axis=1)

    def eps_l(u):
        w1 = phi_val @ u[_I1]; w2 = phi_val @ u[_I2]; w3 = phi_val @ u[_I3]
        dw2 = phi_d1 @ u[_I2]; dw3 = phi_d1 @ u[_I3]
        return jnp.stack([
            w1, xd2*dw2 + xd3*dw3, xd2*w2 + xd3*w3, jnp.zeros(Q_pts),
            jnp.zeros(Q_pts),
            2.0*xd3*dw2 - 2.0*xd2*dw3 + (k22/2.0)*(xd2*w2 + xd3*w3)], axis=1)

    def eps_e(ue):
        return jnp.einsum("qip,p->qi", Ge, ue)

    return eps_h, eps_l, eps_e, Ge


# =============================================================================
# System matrix assembly (energy autodiff, vmap over Hermite elements)
# =============================================================================

def assemble_system_matrices_hermite(
    corners, hcells, reduced_cells, ABD_elems, k22_elems,
    L_elems, xd2_elems, xd3_elems, xi_q, W_q, n_primal, dof_map=None, return_blocks=False
):
    """Assemble Dhh, Dhe, Dee, Dll, Dhl, Dle with Hermite C1 elements.

    ``dof_map`` (E x 12, int) optionally overrides the default node->DOF scatter
    ``6*reduced_cells + arange(6)``.  Used to split the slope (dw/ds) DOFs per
    element at T-junctions (see junction_kirchhoff) while keeping the value DOFs
    shared; ``None`` reproduces the original shared-C1 behaviour exactly."""
    E_elem = hcells.shape[0]
    Q_pts = xi_q.shape[0]

    def get_elem(data):
        n0, n1, ABD, k22, L, xd2, xd3 = data
        dV_q = L * W_q
        ABD_q = jnp.repeat(ABD[None], Q_pts, axis=0)
        eps_h, eps_l, eps_e, Ge = hermite_strain_operators(
            n0, n1, k22, L, xd2, xd3, xi_q)

        z_u = jnp.zeros(12); z_e = jnp.zeros(4)
        q = lambda e, f: 0.5 * jnp.einsum("qi,qij,qj,q->", f(e), ABD_q, f(e), dV_q)
        b = lambda ea, fa, eb, fb: jnp.einsum("qi,qij,qj,q->", fa(ea), ABD_q, fb(eb), dV_q)
        D_hh = jax.hessian(lambda u: q(u, eps_h))(z_u)
        D_he = jax.jacfwd(jax.grad(lambda u, ue: b(u, eps_h, ue, eps_e), 0), 1)(z_u, z_e)
        D_ee = jax.hessian(lambda ue: q(ue, eps_e))(z_e)
        D_ll = jax.hessian(lambda u: q(u, eps_l))(z_u)
        D_hl = jax.jacfwd(jax.grad(lambda u, ul: b(u, eps_h, ul, eps_l), 0), 1)(z_u, z_u)
        D_le = jax.jacfwd(jax.grad(lambda u, ue: b(u, eps_l, ue, eps_e), 0), 1)(z_u, z_e)
        return D_hh, D_he, D_ee, D_ll, D_hl, D_le

    out = jax.vmap(get_elem)((
        jnp.array(corners[hcells[:, 0]]), jnp.array(corners[hcells[:, 1]]),
        ABD_elems, k22_elems, L_elems, xd2_elems, xd3_elems))
    D_hh_b, D_he_b, D_ee_b, D_ll_b, D_hl_b, D_le_b = out
    if return_blocks:                                 # per-element (E,12,12)/(E,12,4)/(E,4,4) blocks
        return D_hh_b, D_he_b, D_ee_b, D_ll_b, D_hl_b, D_le_b

    rc = jnp.array(reduced_cells)
    if dof_map is None:
        dof = (rc.reshape(E_elem, 2, 1) * 6 + jnp.arange(6)).reshape(E_elem, 12)
    else:
        dof = jnp.asarray(dof_map, dtype=jnp.int64)
    rs = jnp.repeat(dof, 12, axis=1).ravel(); cs = jnp.tile(dof, (1, 12)).ravel()
    rr = jnp.repeat(dof, 4, axis=1).ravel(); cr = jnp.tile(jnp.arange(4), (E_elem, 12)).ravel()

    sq = lambda d: jsparse.COO((d.ravel(), rs, cs), shape=(n_primal, n_primal))._sort_indices()
    rect = lambda d: jsparse.COO((d.ravel(), rr, cr), shape=(n_primal, 4))._sort_indices()
    return (sq(D_hh_b), rect(D_he_b), jnp.sum(D_ee_b, axis=0),
            sq(D_ll_b), sq(D_hl_b), rect(D_le_b))


# =============================================================================
# Constraints (derivative twist) + rigid-body kernel Psi
# =============================================================================

def build_constraints_hermite(corners, hcells, reduced_cells, L_elems,
                              xd2_elems, xd3_elems, xi_q, W_q, n_primal, n_unique,
                              dof_map=None):
    """4 x N constraint matrix C and N x 4 rigid kernel Psi (Hermite layout).

    Constraints: <w1>=<w2>=<w3>=0 and the DERIVATIVE twist
    INT(dw2/ds*xd3 - dw3/ds*xd2) ds = 0 (matches the quadratic path, keeps the
    V1 RHS orthogonal to Psi so no penalty is needed).  Psi: 3 translations
    (value DOFs) + twist (value w2=-y3,w3=y2 and slope = tangent).

    ``dof_map`` (E x 12) mirrors :func:`assemble_system_matrices_hermite`: when
    given, the per-element-corner DOFs come from it (split slope DOFs at
    junctions) and the twist slope uses each element's own tangent; ``None``
    reproduces the original shared-C1 layout (nodal-average tangent) exactly."""
    E_elem = hcells.shape[0]
    rc = np.asarray(reduced_cells)
    xd2n = np.asarray(xd2_elems); xd3n = np.asarray(xd3_elems); Ln = np.asarray(L_elems)
    dm = None if dof_map is None else np.asarray(dof_map)

    def cdofs(e, loc, cn):
        return [cn * 6 + k for k in range(6)] if dm is None else [int(dm[e, 6 * loc + k]) for k in range(6)]

    C = np.zeros((4, n_primal))
    for e in range(E_elem):
        val, d1, _ = hermite_shape_functions(xi_q, Ln[e])
        dV = Ln[e] * np.asarray(W_q)
        iv = np.asarray(jnp.einsum("qn,q->n", val, jnp.array(dV)))    # int phi
        idp = np.asarray(jnp.einsum("qn,q->n", d1, jnp.array(dV)))    # int dphi/ds
        for loc, cn in enumerate(rc[e]):
            d = cdofs(e, loc, cn)
            C[0, d[0]] += iv[2*loc];     C[0, d[1]] += iv[2*loc + 1]
            C[1, d[2]] += iv[2*loc];     C[1, d[3]] += iv[2*loc + 1]
            C[2, d[4]] += iv[2*loc];     C[2, d[5]] += iv[2*loc + 1]
            C[3, d[2]] += xd3n[e]*idp[2*loc];  C[3, d[3]] += xd3n[e]*idp[2*loc + 1]
            C[3, d[4]] += -xd2n[e]*idp[2*loc]; C[3, d[5]] += -xd2n[e]*idp[2*loc + 1]

    cc = np.asarray(corners[:n_unique])
    Psi = np.zeros((n_primal, 4))
    if dm is None:
        tang = np.zeros((n_unique, 2))
        for e in range(E_elem):
            for cn in rc[e]:
                tang[cn] += [xd2n[e], xd3n[e]]
        tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-30
        for nd in range(n_unique):
            Psi[nd*6 + 0, 0] = 1.0; Psi[nd*6 + 2, 1] = 1.0; Psi[nd*6 + 4, 2] = 1.0
            y2, y3 = cc[nd]
            Psi[nd*6 + 2, 3] = -y3; Psi[nd*6 + 4, 3] = y2
            Psi[nd*6 + 3, 3] = -tang[nd, 1]; Psi[nd*6 + 5, 3] = tang[nd, 0]
    else:
        for e in range(E_elem):
            for loc, cn in enumerate(rc[e]):
                d = cdofs(e, loc, cn); y2, y3 = cc[cn]
                Psi[d[0], 0] = 1.0; Psi[d[2], 1] = 1.0; Psi[d[4], 2] = 1.0  # translations (value, shared)
                Psi[d[2], 3] = -y3; Psi[d[4], 3] = y2                        # twist values (shared)
                Psi[d[3], 3] = -xd3n[e]; Psi[d[5], 3] = xd2n[e]              # twist slope (per-element tangent)
    return jnp.array(C), jnp.array(Psi)


# =============================================================================
# Full Hermite C1 MSG-TW solve from a YAML cross-section (the only TW path)
# =============================================================================

def solve_tw_from_yaml(yaml_path, reference="OML", frac=None):
    """Full Hermite C1 MSG-shell Timoshenko solve, returning every field.

    Used by both :func:`timoshenko_from_yaml` (which keeps just the stiffness)
    and the dehomogenization (which also needs V0, V1, the mesh/geometry and the
    per-element layups).  Returns a plain ``dict`` bundle with keys:

    ``EB`` (4,4), ``Timo`` (6,6 sorted), ``V0`` (n_primal,4), ``V1`` (n_primal,4
    finalized shear warping), ``corners`` (n_unique,2), ``red_cells`` (E,2),
    ``k22`` (E,), ``L`` (E,), ``xd2`` (E,), ``xd3`` (E,), ``xi_q``, ``W_q``,
    ``ABD_elems`` (E,6,6), ``layup_per_elem`` (E,), ``layup_db``,
    ``material_db``, ``elements`` (1-based connectivity), ``n_primal``.

    ``reference`` : "OML" (default — YAML nodes are the outer mold line),
    "CENTROID"/"MID" (offset inward by half the laminate thickness to the
    material mid-surface — the most accurate reference; OML and IML bracket it),
    or "IML" (offset by the full thickness to the inner mold line).  The offset
    is along the material e3 and the plate ABD is reference-shifted by the same
    amount (e3 kept inward), reducing the spar-cap/web overlap the OML
    double-counts at junctions.
    """
    import numpy as _np
    import jax.numpy as _jnp
    import pypardiso
    from .msg_mesh import (load_yaml, read_mesh, mesh_curvature,
                           offset_oml_to_iml, element_e3_from_yaml)
    from .msg_materials import compute_ABD_matrix, shift_abd_reference
    from .msg_solver import (gauss_legendre_01, compute_element_geometry,
        solve_fluctuation_field, prepare_v1_rhs, finalize_v1_and_compute_deff)

    nodes_3d, elements, material_db, layup_db, elem_to_layup = load_yaml(yaml_path)
    # reference-surface offset fraction of the laminate thickness (inward from
    # the OML): 0 = OML (default), 0.5 = centroid/mid, 1 = IML.  ``frac`` (if
    # given) overrides the named ``reference``.
    if frac is None:
        frac = {"OML": 0.0, "MID": 0.5, "CENTROID": 0.5, "IML": 1.0}[reference.upper()]
    frac = float(frac)

    # Compute the plate ABD ONCE per unique physical laminate.  A station can
    # name several layups that are the same (thick, angles, materials); caching
    # on that signature avoids redundant compute_ABD calls.  A non-OML reference
    # adds a parallel-axis shift by frac x thickness (e3 kept inward; see the
    # check-e3-orientation skill).
    def _sig(i):
        return (tuple(i['thick']), tuple(i['angles']), tuple(i['mat_names']))
    _abd_cache = {}

    def _abd(i):
        s = _sig(i)
        if s not in _abd_cache:
            a = compute_ABD_matrix(i['thick'], i['angles'], i['mat_names'], material_db)[0]
            _abd_cache[s] = shift_abd_reference(a, frac * float(sum(i['thick']))) if frac else a
        return _abd_cache[s]
    ABD_dict = {ln: _abd(i) for ln, i in layup_db.items()}

    # Mesh = YAML connectivity verbatim (every element kept; webs included).
    nodes, cells, layup_per_elem = read_mesh(nodes_3d, elements, elem_to_layup)
    if frac:
        # offset inward along the material e3 (true OML->IML normal per element)
        elem_e3 = element_e3_from_yaml(yaml_path)
        nodes = offset_oml_to_iml(nodes, cells, layup_per_elem, layup_db,
                                  elem_e3=elem_e3, frac=frac)
    k22 = _jnp.array(mesh_curvature(nodes, cells, elements, is_closed=False))
    ABD_elems = _jnp.stack([_jnp.array(ABD_dict[ln], dtype=_jnp.float64) for ln in layup_per_elem])

    # Hermite uses the corner endpoints; renumber to the nodes actually used
    # (6 DOF/node, no periodic merge — nodes are shared via the connectivity).
    hcells = cells[:, [0, -1]]
    used = _np.unique(hcells)
    f2r = _np.full(nodes.shape[0], -1, dtype=_np.int64)
    f2r[used] = _np.arange(len(used))
    red_cells = f2r[hcells]
    corners = nodes[used]
    n_unique = len(used)
    n_primal = 6 * n_unique

    L_e, xd2, xd3 = compute_element_geometry(corners, red_cells)
    xi_q, W_q = gauss_legendre_01(4)

    Dhh, Dhe, Dee, Dll, Dhl, Dle = assemble_system_matrices_hermite(
        corners, red_cells, red_cells, ABD_elems, k22, L_e, xd2, xd3, xi_q, W_q, n_primal)
    C, Psi = build_constraints_hermite(
        corners, red_cells, red_cells, L_e, xd2, xd3, xi_q, W_q, n_primal, n_unique)
    Dc = C.T

    V0, D1, A_aug = solve_fluctuation_field(Dhh, -_np.array(Dhe.todense()), C)
    Ceff = Dee + D1
    bb, DhlV0, DhlTV0Dle, V0DllV0 = prepare_v1_rhs(
        V0, Dhl, Dll, _jnp.array(Dle.todense()), Psi, Dc)
    R_v1 = _np.concatenate([_np.array(bb), _np.zeros((4, bb.shape[1]))], axis=0)
    V_aug = pypardiso.spsolve(A_aug, R_v1)
    C6, _Btim, _Ctim, V1 = finalize_v1_and_compute_deff(
        _jnp.array(V_aug[:n_primal, :]), V0, Ceff, V0DllV0, DhlV0, DhlTV0Dle, Psi, Dc)
    C6.block_until_ready()

    return {
        "EB": _np.array(Ceff), "Timo": _np.array(C6),
        "Btim": _np.array(_Btim), "Ctim": _np.array(_Ctim),
        "V0": _np.array(V0), "V1": _np.array(V1),
        "corners": _np.array(corners), "red_cells": _np.array(red_cells),
        "k22": _np.array(k22), "L": _np.array(L_e),
        "xd2": _np.array(xd2), "xd3": _np.array(xd3),
        "xi_q": xi_q, "W_q": W_q, "ABD_elems": _np.array(ABD_elems),
        "layup_per_elem": list(layup_per_elem), "layup_db": layup_db,
        "material_db": material_db, "elements": elements, "n_primal": n_primal,
        "frac": frac,
    }


def timoshenko_from_yaml(yaml_path, reference="OML", frac=None):
    """Hermite C1 MSG-shell Timoshenko solve for an OpenSG YAML cross-section.

    The mesh is taken straight from the YAML connectivity (``read_mesh`` — no
    chaining), so all elements are kept and shear-webbed / multi-component
    cross-sections are handled.  ``reference`` = "OML" (default) or "IML"
    (inward-offset reference; see :func:`solve_tw_from_yaml`).

    Returns
    -------
    EB   : (4,4) Euler-Bernoulli stiffness  [EA, GJ, EI2, EI3]
    Timo : (6,6) sorted Timoshenko stiffness [EA, GA12, GA13, GJ, EI2, EI3]
    complete : bool  always True (every YAML element is used)
    """
    out = solve_tw_from_yaml(yaml_path, reference=reference, frac=frac)
    return out["EB"], out["Timo"], True
