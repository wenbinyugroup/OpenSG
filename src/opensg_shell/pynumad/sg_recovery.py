"""Per-station two-step RM shell dehomogenization driven by a .ff file, with VABS-layout
.SM / .EM / .U output (the msgrm production route).

step 1 (section): st = C6^-1 FF and the RM warping recombination, then per element the
  6 shell strains [e11 e22 2e12 k11 k22 2k12] plus the span/arc strain gradients
  dE1/dE2 (arc gradient from LAYUP-BOUNDARY-AWARE nodal averaging -- no differencing
  across section changes).
step 2 (wall through-thickness): MSG-RM first-order plate recovery at depth z over the
  FULL cross-section -- every ring element (skins AND webs) x n_depth ply-interior
  through-thickness stations, via the 1-D RM plate SG (opensg_solid.rm_plate_1D).

Output files (VABS recovery layout, section units = Pa, m):
  <base>.SM  x2 x3 s11 s12 s13 s22 s23 s33   (material frame, 2-line header)
  <base>.EM  x2 x3 e11 e12 e13 e22 e23 e33   (material frame, 2-line header)
  <base>.U   id x2 x3 u1 u2 u3               (total displacement, no header)

Total displacement composes the beam motion from the .ff (reference displacement u0 +
finite section rotation R) with the RM warping w:  v = u0 + (R - I) x + R w,
x = (0, x2, x3).  R = C^T by default (VABS .glb stores C_bB = (C_Bb)^T).
"""
import os

import numpy as np

# msgrm order emitted by msgrm_strain_at_depth: [11, 22, 33, 23, 13, 12]
_MSGRM_TO_VABS = [0, 5, 4, 1, 3, 2]                       # -> [11, 12, 13, 22, 23, 33]


def dehom_station(shell_yaml, ff, n_depth=9, frame="material", bundle=None,
                  rot="transpose"):
    """Two-step RM dehomogenization of one station over the full cross-section.

    In:
        shell_yaml: str, 1-D shell SG yaml (its `reference` field sets frac).
        ff: str .ff path (read_ff) | dict from read_ff | (6,) [F1 F2 F3 M1 M2 M3].
        n_depth: int, through-thickness recovery points per element (ply-interior,
            zeta = (k + 0.5)/n_depth, 0 = OML .. 1 = IML).
        frame: "material" (ply axes, failure criteria) | "plate" (wall axes).
        bundle: dict | None, reuse an existing build_rm_bundle result.
        rot: "transpose" | "direct", section rotation R = C^T | C for the rigid part
            of the displacement (only used when ff carries u/C).
    Out:
        dict: "pts" (P,2) recovery coords (x2,x3); "sig"/"gam" (P,6) stress [Pa] /
        strain in msgrm order [11,22,33,23,13,12]; "u" (P,3) total displacement [m];
        "elem" (P,), "zeta" (P,), "z" (P,), "ply" (P,) recovery metadata; "FF" (6,);
        "eta" float | nan; "bundle" the RM bundle (reusable).
    """
    from ..sg_homo import build_rm_bundle
    from ..sg_dehom import _macro_fields, _rm_shell_strain
    from ..sg_assembly import quad_ops_indep
    from ..fe_jax.msg_dehom import _macro_recovery
    from opensg_solid.rm_plate_1D.msg_rm_plate import rm_plate_msg, msgrm_strain_at_depth
    from .sg_beamdyn import read_ff

    B = bundle if bundle is not None else build_rm_bundle(shell_yaml)
    C6 = np.asarray(B["Timo"])
    u0 = R3 = None; eta = float("nan")
    if isinstance(ff, str):
        ff = read_ff(ff)
    if isinstance(ff, dict):
        FF = np.asarray(ff["FF"], float)
        u0 = np.asarray(ff["u"], float)
        C = np.asarray(ff["C"], float)
        R3 = C.T if rot == "transpose" else C
        eta = float(ff.get("eta", float("nan")))
    else:
        FF = np.asarray(ff, float)

    # ---------------- step 1: RM shell section strains + gradients ----------------
    st, st_m, aA, aB = _macro_fields(B, beam_force_vabs=FF)
    _, _sm, st_cl1, _st_cl2 = _macro_recovery(C6, np.linalg.inv(C6) @ FF)

    corners = np.asarray(B["corners"]); rc = np.asarray(B["red_cells"]); cen = corners.mean(0)
    nodes_s, quads_s, _hs = B["strip"]
    n_el = rc.shape[0]; n_nd = int(rc.max()) + 1
    layups = B["layup_per_elem"]; ldb = B["layup_db"]; mdb = B["material_db"]
    frac = float(B.get("frac", 0.0))
    hth = {ln: float(sum(i["thick"])) for ln, i in ldb.items()}

    s6mid = np.zeros((n_el, 6)); dE1e = np.zeros((n_el, 6))
    for e in range(n_el):
        s6, _ = _rm_shell_strain(B, e, 0.5, st_m, aA, aB)
        s6mid[e] = np.asarray(s6, float)
        Xe = nodes_s[quads_s[e]]; e3e = B["re3"][e]
        BDe, BDh, BDl, *_ = quad_ops_indep(Xe, e3e, 0.0, 0.0, float(B["k22"][e]),
                                           B["cross"], B["ax"])
        c0, c1 = int(rc[e, 0]), int(rc[e, 1])
        g = np.r_[c0 * 6:c0 * 6 + 6, c1 * 6:c1 * 6 + 6, c1 * 6:c1 * 6 + 6, c0 * 6:c0 * 6 + 6]
        dE1e[e] = np.asarray(BDe @ st_cl1 + BDh @ aB[g], float)

    # layup-boundary-aware nodal average (only where two SAME-layup elements meet)
    deg = np.zeros(n_nd, int); nd_el = [[] for _ in range(n_nd)]
    for e in range(n_el):
        for nd in (int(rc[e, 0]), int(rc[e, 1])):
            deg[nd] += 1; nd_el[nd].append(e)
    s6n = np.full((n_nd, 6), np.nan)
    for nd in range(n_nd):
        if deg[nd] == 2 and layups[nd_el[nd][0]] == layups[nd_el[nd][1]]:
            s6n[nd] = 0.5 * (s6mid[nd_el[nd][0]] + s6mid[nd_el[nd][1]])
    T = corners[rc[:, 1]] - corners[rc[:, 0]]
    Lel = np.sqrt((T ** 2).sum(1))
    dE2e = np.zeros((n_el, 6))
    for e in range(n_el):
        c0, c1 = int(rc[e, 0]), int(rc[e, 1])
        v0 = s6n[c0] if np.isfinite(s6n[c0, 0]) else s6mid[e]
        v1 = s6n[c1] if np.isfinite(s6n[c1, 0]) else s6mid[e]
        dE2e[e] = (v1 - v0) / Lel[e]

    # inward wall normal per element (plate +z = reference -> IML)
    tun = T / Lel[:, None]
    nvec = np.column_stack([tun[:, 1], -tun[:, 0]])
    emid = 0.5 * (corners[rc[:, 0]] + corners[rc[:, 1]])
    flip = ((cen - emid) * nvec).sum(1) < 0
    nvec[flip] *= -1.0

    # ---------------- step 2: MSG-RM through-thickness over the full section --------
    warpM = {ln: rm_plate_msg(i["thick"], i["angles"], i["mat_names"], mdb, fraction=frac)
             for ln, i in ldb.items()}
    zeta = (np.arange(n_depth) + 0.5) / n_depth
    P = n_el * n_depth
    pts = np.zeros((P, 2)); sig = np.zeros((P, 6)); gam = np.zeros((P, 6))
    ely = np.zeros(P, int); zta = np.zeros(P); zz = np.zeros(P); ply = np.zeros(P)
    wn = np.asarray(aA).reshape(-1, 6)                    # per-node [u1,u2,u3,om1,om2,om3]
    u = np.zeros((P, 3))
    for e in range(n_el):
        ln = layups[e]; h = hth[ln]
        c0, c1 = int(rc[e, 0]), int(rc[e, 1])
        s6 = s6mid[e].copy()
        for row in (2, 5):                                # contour-derivative rows: nodal interp
            v0 = s6n[c0, row] if np.isfinite(s6n[c0, row]) else s6mid[e, row]
            v1 = s6n[c1, row] if np.isfinite(s6n[c1, row]) else s6mid[e, row]
            s6[row] = 0.5 * (v0 + v1)
        umid = 0.5 * (wn[c0, 0:3] + wn[c1, 0:3])          # element-mid warping
        om = 0.5 * (wn[c0, 3:6] + wn[c1, 3:6])            # director rotation
        e3v = np.array([0.0, nvec[e, 0], nvec[e, 1]])
        for k in range(n_depth):
            z = (zeta[k] - frac) * h
            p = emid[e] + z * nvec[e]
            Gam, Sig, ply_ang = msgrm_strain_at_depth(warpM[ln], z, s6, dE1e[e], dE2e[e],
                                                      frame=frame)
            i = e * n_depth + k
            pts[i] = p; sig[i] = np.asarray(Sig, float); gam[i] = np.asarray(Gam, float)
            ely[i] = e; zta[i] = zeta[k]; zz[i] = z; ply[i] = ply_ang
            w = umid + z * np.cross(om, e3v)              # RM warping u(z) = u_mid + z (om x e3)
            x = np.array([0.0, p[0], p[1]])
            if u0 is not None:
                u[i] = u0 + (R3 - np.eye(3)) @ x + R3 @ w
            else:
                u[i] = w
    return dict(pts=pts, sig=sig, gam=gam, u=u, elem=ely, zeta=zta, z=zz, ply=ply,
                FF=FF, eta=eta, bundle=B)


def write_vabs_sm(path, pts, sig):
    """Write the VABS-layout .SM (stresses, material frame, Gauss-point cloud).

    In:
        path: str; pts: (P,2) (x2,x3); sig: (P,6) stress [Pa] in msgrm order
        [11,22,33,23,13,12] (reordered to the VABS columns here).
    Out:
        str path.
    """
    s = np.asarray(sig, float)[:, _MSGRM_TO_VABS]
    with open(path, "w") as f:
        f.write(" Stresses for each Gauss point in material coordinate system for load case #\n")
        f.write("           1\n")
        for i in range(len(pts)):
            f.write("    %.10E    %.10E" % (pts[i, 0], pts[i, 1]) +
                    "".join("    %.10E" % v for v in s[i]) + "\n")
    return path


def write_vabs_em(path, pts, gam):
    """Write the VABS-layout .EM (strains, material frame, Gauss-point cloud).

    In:
        path: str; pts: (P,2) (x2,x3); gam: (P,6) strain in msgrm order
        [11,22,33,23,13,12] (reordered to the VABS columns here).
    Out:
        str path.
    """
    g = np.asarray(gam, float)[:, _MSGRM_TO_VABS]
    with open(path, "w") as f:
        f.write(" Strains for each Gauss point in material coordinate system for load case #\n")
        f.write("           1\n")
        for i in range(len(pts)):
            f.write("    %.10E    %.10E" % (pts[i, 0], pts[i, 1]) +
                    "".join("    %.10E" % v for v in g[i]) + "\n")
    return path


def write_vabs_u(path, pts, u):
    """Write the VABS-layout .U (total displacement at the recovery cloud).

    In:
        path: str; pts: (P,2) (x2,x3); u: (P,3) displacement [m].
    Out:
        str path.
    """
    with open(path, "w") as f:
        for i in range(len(pts)):
            f.write("%9d    %.10E    %.10E    %.10E    %.10E    %.10E\n"
                    % (i + 1, pts[i, 0], pts[i, 1], u[i, 0], u[i, 1], u[i, 2]))
    return path


def dehom_to_files(shell_yaml, ff, out_base=None, n_depth=9, frame="material",
                   bundle=None):
    """Run dehom_station and write the .SM / .EM / .U triple.

    In:
        shell_yaml: str, 1-D shell SG yaml.
        ff: str | dict | (6,), see dehom_station.
        out_base: str | None, output base path (default <yaml base>, `_shell` stripped).
        n_depth, frame, bundle: see dehom_station.
    Out:
        dict from dehom_station, plus "files" [.SM, .EM, .U paths].
    """
    D = dehom_station(shell_yaml, ff, n_depth=n_depth, frame=frame, bundle=bundle)
    if out_base is None:
        out_base = os.path.splitext(shell_yaml)[0]
        if out_base.endswith("_shell"):
            out_base = out_base[:-6]
    D["files"] = [write_vabs_sm(out_base + ".SM", D["pts"], D["sig"]),
                  write_vabs_em(out_base + ".EM", D["pts"], D["gam"]),
                  write_vabs_u(out_base + ".U", D["pts"], D["u"])]
    return D
