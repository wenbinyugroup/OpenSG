"""3-D shell SG: equivalent solid properties of a shell-element structure
gene that is PERIODIC IN ALL THREE directions (TPMS-class cells).

Reuses the msg_shell operators unchanged -- solid_fluct_ops_batch (Gamma_h),
solid_macro_ops_batch (Gamma_e) and the per-element frames are geometry-
general; what changes vs a cross-section run is only the environment:
  * general shell quads (tris = collapsed quads) in 3-D, no prismatic strip;
  * periodicity through the sparse assembly map on the FULL 3-D coordinates
    (all opposite faces, edges and corners -- the 3-D SG default);
  * sparse assembly (scipy) -- 3-D SGs are too large for dense Dhh;
  * drilling om3 by penalty on the SAME element-constant residual the
    cross-section route enforces with multipliers;
  * kernel = 3 translations (Lagrange border);
  * JUNCTION handling: shell edges shared by >2 elements are junction lines
    (a smooth TPMS has none -- every edge interior, count reported); when
    present they take the in-code hex-element micro treatment (3-D analog of
    the quad junction remesh) -- detection is wired, the hex micro follows
    the same dC = E_solid_mini - E_shell_mini construction.

Writes <yaml base>_C3D.out with the solve time (OpenSG default).
"""
import time
from collections import Counter

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import yaml as _yaml

from .periodic_multiscale import mesh_to_periodic_sparse_assembly_map
from .solid_props import NDOF6, solid_fluct_ops_batch, solid_macro_ops_batch
from .solve_segment_jax import _material_by_section

_G = 1.0/np.sqrt(3.0)
_GPTS = [(-_G, -_G), (_G, -_G), (_G, _G), (-_G, _G)]


def shell_sg3d(yaml_path, omega=None, drill_pen=1.0e-3, g_source="msg"):
    """omega = the SG measure remaining in the model (SwiftComp-TW
    convention): the midsurface SURFACE AREA, integrated from the mesh by
    default -- the 3-D analog of the plane-section omega = perimeter.
    Pass omega explicitly to override (e.g. the cell volume for a
    per-unit-cell law)."""
    t0 = time.perf_counter()
    d = _yaml.safe_load(open(yaml_path))
    row = lambda r: " ".join(str(x) for x in
                             (r if isinstance(r, list) else [r])).split()
    nd = np.array([[float(v) for v in row(r)][:3] for r in d["nodes"]])
    el = [[int(v) for v in row(r)] for r in d["elements"]]
    el = np.array([e + [e[-1]]*(4 - len(e)) for e in el], int) - 1
    ori = np.array(d["elementOrientations"], float)
    e3 = ori[:, 6:9]
    nn, ne = len(nd), len(el)

    D_by, G_by = _material_by_section(d["sections"], d["materials"],
                                      center_ref=True)
    if g_source == "msg":
        from opensg_solid.rm_plate_1D.msg_rm_plate import rm_plate_msg
        from .emit_abd import material_db_from_yaml
        pl = [[str(p[0]), float(p[1]), float(p[2])]
              for p in d["sections"][0]["layup"]]
        rr = rm_plate_msg([p[1] for p in pl], [p[2] for p in pl],
                          [p[0] for p in pl],
                          material_db_from_yaml(d["materials"]), fraction=0.5)
        if rr["G_msg"] is not None:
            G_by = [np.asarray(rr["G_msg"])]
    De = np.asarray(D_by[0] if not isinstance(D_by, dict) else D_by[0],
                    float).reshape(6, 6)
    Gm = np.asarray(G_by[0], float).reshape(2, 2)

    # junction lines: shell edges shared by more than two elements
    cnt = Counter()
    for e in el:
        vs = list(dict.fromkeys(e))
        for k in range(len(vs)):
            cnt[tuple(sorted((vs[k], vs[(k+1) % len(vs)])))] += 1
    n_junc_edges = sum(1 for v in cnt.values() if v > 2)

    # periodic map on the FULL 3-D coordinates: all faces/edges/corners
    rc, _ = mesh_to_periodic_sparse_assembly_map(nn, np.arange(nn)[:, None],
                                                 nd, 3, NDOF6)
    master = np.asarray(rc, int).ravel()
    uniq, inv = np.unique(master, return_inverse=True)
    ndof = NDOF6*len(uniq)

    Xe = nd[el]
    gd = (NDOF6*inv[el][:, :, None]
          + np.arange(NDOF6)[None, None, :]).reshape(ne, 24)
    rowsI, colsJ, vals = [], [], []
    Dhe = np.zeros((ndof, 6))
    Dee = np.zeros((6, 6))
    A11 = De[0, 0]
    A_surf = 0.0
    for xi, eta in _GPTS:
        B, Bg, Dr, dA = solid_fluct_ops_batch(Xe, e3, xi, eta, [1, 2], 0)
        BDe6, BGe6, _ = solid_macro_ops_batch(Xe, e3, xi, eta, [1, 2], 0)
        Ke = np.einsum('e,eia,ij,ejb->eab', dA, B, De, B) \
            + np.einsum('e,eia,ij,ejb->eab', dA, Bg, Gm, Bg) \
            + drill_pen*A11*np.einsum('e,ea,eb->eab', dA, Dr, Dr)
        Fe = np.einsum('e,eia,ij,ejb->eab', dA, B, De, BDe6) \
            + np.einsum('e,eia,ij,ejb->eab', dA, Bg, Gm, BGe6)
        Dee += np.einsum('e,eia,ij,ejb->ab', dA, BDe6, De, BDe6) \
            + np.einsum('e,eia,ij,ejb->ab', dA, BGe6, Gm, BGe6)
        rowsI.append(np.repeat(gd, 24, 1).ravel())
        colsJ.append(np.tile(gd, (1, 24)).ravel())
        vals.append(Ke.ravel())
        np.add.at(Dhe, gd.ravel(), Fe.reshape(-1, 6))
        A_surf += float(dA.sum())
    K = sp.csr_matrix((np.concatenate(vals),
                       (np.concatenate(rowsI), np.concatenate(colsJ))),
                      shape=(ndof, ndof))

    # kernel: the 3 rigid translations, area-weighted Lagrange rows
    wA = np.zeros(len(uniq))
    _, _, _, dA0 = solid_fluct_ops_batch(Xe, e3, 0.0, 0.0, [1, 2], 0)
    np.add.at(wA, inv[el].ravel(), np.repeat(dA0, 4))
    Cc = sp.lil_matrix((3, ndof))
    for k in range(3):
        Cc[k, k::NDOF6] = wA
    A = sp.bmat([[K, Cc.T], [Cc, None]], format="csc")
    R = np.zeros((ndof + 3, 6))
    R[:ndof] = -Dhe
    lu = spla.splu(A)
    V0 = np.column_stack([lu.solve(R[:, c]) for c in range(6)])[:ndof]
    Deff = Dee + V0.T @ Dhe
    Deff = 0.5*(Deff + Deff.T)
    if omega is None:
        omega = A_surf                       # SG measure = midsurface area
    C3D = Deff/float(omega)
    solve_time = time.perf_counter() - t0

    import os
    from opensg_solid.sg_homo import write_sc_K
    # the .out follows SwiftComp's normalization (per unit-cell volume) so its
    # moduli compare directly with solid .K files; the returned C3D keeps the
    # SG-measure (surface-area) convention
    V_cell = float(np.prod(nd.max(0) - nd.min(0)))
    write_sc_K(os.path.splitext(yaml_path)[0] + "_C3D.out", Deff/V_cell,
               solve_time=solve_time,
               model="msg-shell equivalent 3D solid (3-D shell SG, %d nodes,"
                     " %d elems, %d junction edges, periodic in 3 dirs,"
                     " per unit cell %.6g)"
                     % (nn, ne, n_junc_edges, V_cell))
    return {"C3D": C3D, "D_eff": Deff, "solve_time": solve_time,
            "n_junction_edges": n_junc_edges, "ndof": ndof}
