"""Beam properties of a 1-D shell SG: RM Timoshenko 6x6 + 6x6 mass matrix, VABS .K output.

The mass matrix is the shell mass model of OpenSG-FEniCSx (utils/shell.py get_mass_shell)
translated to JAX over the 1-D ring: per-section through-thickness density moments
(mu = int rho dz, mx3 = int rho z dz, i22 = int rho z^2 dz) line-integrated along the
contour arc.  The ring generalisation projects the thickness moments on the wall normal
(n2, n3) -- get_mass_shell's plate form is the special case n = (0, 1):

    x(z) = x_ref + z n  ->  int rho x3 dA   = int (mu x3 + n3 mx3) ds
                            int rho x3^2 dA = int (mu x3^2 + 2 x3 n3 mx3 + n3^2 i22) ds

z runs from the yaml's `reference` surface along +e3 (inward), plies outer -> inner,
so OML sits at z = -frac*h and IML at z = +(1-frac)*h.

All matrices are in the VABS frame/order (1 = axial x1, 2-3 = section x2, x3):
rows/cols [F1 F2 F3 M1 M2 M3].
"""
import numpy as np
import jax.numpy as jnp
import yaml

try:
    from yaml import CSafeLoader as _Loader
except ImportError:
    from yaml import SafeLoader as _Loader

_GP = np.array([-1.0, 1.0]) / np.sqrt(3.0)               # 2-pt Gauss (exact: quadratic)


def _row(v):
    """Yaml row -> list of floats (rows are single space-separated strings or lists)."""
    if isinstance(v, (list, tuple)) and len(v) == 1 and isinstance(v[0], str):
        return [float(x) for x in v[0].split()]
    if isinstance(v, str):
        return [float(x) for x in v.split()]
    return [float(x) for x in v]


def _ring_arrays(shell_yaml):
    """Parse the 1-D shell SG yaml into ring arrays for the mass/geometry integrals.

    In:
        shell_yaml: str, 1-D shell SG yaml path.
    Out:
        dict: "x" (n_nd,2) node coords (x2,x3); "cells" (n_el,2) 0-based; "n" (n_el,2)
        wall normal (e3 in-plane components); "sec" (n_el,) section index; "layups"
        list per section of [(mat, t, ang)]; "rho" {mat: density}; "frac" float,
        reference-surface fraction (0 = OML, 0.5 = center); "ref" str.
    """
    d = yaml.load(open(shell_yaml), Loader=_Loader)
    x = np.array([_row(r)[:2] for r in d["nodes"]], float)
    cells = np.array([[int(v) for v in _row(e)] for e in d["elements"]], int)
    if cells.min() == 1:
        cells = cells - 1
    ori = np.array([_row(o) for o in d["elementOrientations"]], float)
    n = ori[:, 6:8]                                       # e3 in-plane = wall normal (n2, n3)
    sec_of_set = {s["elementSet"]: i for i, s in enumerate(d["sections"])}
    sec = np.zeros(len(cells), int)
    for grp in d["sets"]["element"]:
        si = sec_of_set[grp["name"]]
        for lab in grp["labels"]:
            sec[int(lab) - 1] = si
    layups = [[(str(p[0]), float(p[1]), float(p[2])) for p in s["layup"]]
              for s in d["sections"]]
    rho = {m["name"]: float(m.get("density", 1.0)) for m in d["materials"]}
    ref = d.get("reference", "center")
    frac = {"center": 0.5, "oml": 0.0, "oml_flip": 1.0, "iml": 1.0}.get(ref, 0.0)
    return dict(x=x, cells=cells, n=n, sec=sec, layups=layups, rho=rho, frac=frac, ref=ref)


def _density_moments(layup, rho, frac):
    """Through-thickness density moments of one laminate about its reference surface.

    In:
        layup: list of (mat, t, ang) outer -> inner.
        rho: {mat: density}; frac: float reference fraction (0 = OML .. 1 = IML).
    Out:
        (mu, mx3, i22): int rho dz, int rho z dz, int rho z^2 dz  [z from the
        reference surface along +e3; per-ply z^3/12 term included].
    """
    h = sum(t for (_m, t, _a) in layup)
    mu = mx3 = i22 = 0.0
    z0 = -frac * h                                        # OML face
    for (m, t, _a) in layup:
        zb = z0 + 0.5 * t                                 # ply mid-plane
        r = rho[m]
        mu += r * t
        mx3 += r * t * zb
        i22 += r * (t ** 3 / 12.0 + t * zb * zb)
        z0 += t
    return mu, mx3, i22


def mass_matrix_ring(shell_yaml):
    """6x6 mass matrix of the 1-D shell ring (VABS frame, about the section origin).

    JAX translation of OpenSG-FEniCSx get_mass_shell with the ring's wall-normal
    projection; 2-pt Gauss per line element (exact for the quadratic integrands).

    In:
        shell_yaml: str, 1-D shell SG yaml path.
    Out:
        (M, info): M (6,6) mass matrix
            [[ m    0    0    0    S3  -S2 ]
             [ 0    m    0   -S3   0    0  ]
             [ 0    0    m    S2   0    0  ]
             [ 0   -S3   S2  I22+I33  0  0 ]
             [ S3   0    0    0    I22 -I23]
             [-S2   0    0    0   -I23  I33]];
        info dict: "mass_center" (2,), "M_center" (6,6), "i11"/"i22p"/"i33p" principal
        inertias about the mass center, "angle_deg" principal rotation about +x1,
        "rgyr" mass-weighted radius of gyration, "area" material area,
        "geometric_center" (2,), "mpus" mass per unit span.
    """
    R = _ring_arrays(shell_yaml)
    x, cells, nrm, sec = R["x"], R["cells"], R["n"], R["sec"]
    mom = np.array([_density_moments(L, R["rho"], R["frac"]) for L in R["layups"]])
    geo = np.array([_density_moments(L, {m: 1.0 for m in R["rho"]}, R["frac"])
                    for L in R["layups"]])                # unit density -> area moments

    P1 = jnp.asarray(x[cells[:, 0]]); P2 = jnp.asarray(x[cells[:, 1]])
    Le = jnp.linalg.norm(P2 - P1, axis=1)                 # (n_el,)
    mid = 0.5 * (P1 + P2); half = 0.5 * (P2 - P1)
    n2 = jnp.asarray(nrm[:, 0]); n3 = jnp.asarray(nrm[:, 1])
    mu = jnp.asarray(mom[sec, 0]); mx3 = jnp.asarray(mom[sec, 1]); i22 = jnp.asarray(mom[sec, 2])
    a0 = jnp.asarray(geo[sec, 0]); az = jnp.asarray(geo[sec, 1])

    m = S2 = S3 = I22 = I33 = I23 = A = G2 = G3 = 0.0
    for g in _GP:                                         # sum over the 2 Gauss points
        w = 0.5 * Le                                      # weight * |J| per point
        x2 = mid[:, 0] + g * half[:, 0]
        x3 = mid[:, 1] + g * half[:, 1]
        m += jnp.sum(w * mu)
        S2 += jnp.sum(w * (x2 * mu + n2 * mx3))
        S3 += jnp.sum(w * (x3 * mu + n3 * mx3))
        I22 += jnp.sum(w * (mu * x3 ** 2 + 2.0 * x3 * n3 * mx3 + n3 ** 2 * i22))
        I33 += jnp.sum(w * (mu * x2 ** 2 + 2.0 * x2 * n2 * mx3 + n2 ** 2 * i22))
        I23 += jnp.sum(w * (mu * x2 * x3 + (x2 * n3 + x3 * n2) * mx3 + n2 * n3 * i22))
        A += jnp.sum(w * a0)
        G2 += jnp.sum(w * (x2 * a0 + n2 * az))
        G3 += jnp.sum(w * (x3 * a0 + n3 * az))
    m, S2, S3, I22, I33, I23, A, G2, G3 = (float(v) for v in
                                           (m, S2, S3, I22, I33, I23, A, G2, G3))

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
    i22p, i33p, ang = _principal_2x2(I22c, I33c, I23c)
    info = dict(mass_center=np.array([x2m, x3m]), M_center=Mc, mpus=m,
                i11=I22c + I33c, i22p=i22p, i33p=i33p, angle_deg=ang,
                rgyr=float(np.sqrt((I22c + I33c) / m)), area=A,
                geometric_center=np.array([G2 / A, G3 / A]), ref=R["ref"])
    return M, info


def _principal_2x2(J22, J33, J23):
    """Principal values/axis of the plane tensor [[J22, -J23], [-J23, J33]].

    In:
        J22, J33, J23: floats, int x3^2, int x2^2, int x2 x3 moments about the center.
    Out:
        (j22p, j33p, angle_deg): j22p = the smaller principal value; angle_deg = the
        rotation of its principal axis from +x2 about +x1, in [0, 180) (VABS report).
    """
    T = np.array([[J22, -J23], [-J23, J33]])
    w, V = np.linalg.eigh(T)                              # ascending
    j22p, j33p = float(w[0]), float(w[1])
    v = V[:, 0]
    ang = float(np.degrees(np.arctan2(v[1], v[0]))) % 180.0
    return j22p, j33p, ang


def _fmt66(Mtx, fmt="  %18.10E"):
    """6x6 -> VABS-style text rows."""
    return "\n".join("".join(fmt % Mtx[i, j] for j in range(6)) for i in range(6))


def write_vabs_k(path, K, M, info, model="OpenSG msg-shell beam model", solve_time=None):
    """Write a VABS .K-layout output: mass blocks + Timoshenko stiffness/compliance.

    Blocks (in VABS .K order): 6x6 mass matrix, mass center, 6x6 mass matrix at the
    mass center, principal mass properties, geometric center, area, Timoshenko
    stiffness/compliance, tension center, extension/torsion/principal-bending summary,
    shear center, principal shear stiffness.

    In:
        path: str, output file (.K).
        K: (6,6) Timoshenko stiffness (VABS order).
        M: (6,6) mass matrix from mass_matrix_ring.
        info: dict from mass_matrix_ring.
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
    L.append("\n The Mass Properties with respect to Principal Inertial Axes\n" + bar +
             "\n Mass per unit span                     =  %14.10E\n" % info["mpus"] +
             " Mass moment of inertia i11             =  %14.10E\n" % info["i11"] +
             " Principal mass moments of inertia i22  =  %14.10E\n" % info["i22p"] +
             " Principal mass moments of inertia i33  =  %14.10E\n" % info["i33p"] +
             " The principal inertial axes rotated from user coordinate system by\n"
             "   %.10f degrees about the positive direction of x1 axis.\n" % info["angle_deg"] +
             " The mass-weighted radius of gyration   =  %14.10E\n" % info["rgyr"])
    L.append("\n The Geometric Center of the Cross Section\n" + bar +
             "\n  %18.10E  %18.10E\n" % tuple(info["geometric_center"]))
    L.append("\n The Area of the Cross Section\n" + bar +
             "\n Area =  %18.10E\n" % info["area"])
    L.append("\n Timoshenko Stiffness Matrix (1-extension; 2,3-shear, 4-twist; 5,6-bending)\n"
             + bar + "\n" + _fmt66(K) + "\n")
    L.append("\n Timoshenko Compliance Matrix (1-extension; 2,3-shear, 4-twist; 5,6-bending)\n"
             + bar + "\n" + _fmt66(S) + "\n")

    x2t, x3t = -K[0, 5] / K[0, 0], K[0, 4] / K[0, 0]      # tension center
    L.append("\n The Tension Center of the Cross Section\n" + bar +
             "\n  %18.10E  %18.10E\n" % (x2t, x3t))
    EI22c = K[4, 4] - K[0, 4] ** 2 / K[0, 0]              # bending block at the tension center
    EI33c = K[5, 5] - K[0, 5] ** 2 / K[0, 0]
    EI23c = K[4, 5] - K[0, 4] * K[0, 5] / K[0, 0]
    ei22p, ei33p, eang = _principal_2x2(EI22c, EI33c, -EI23c)
    L.append("\n The extension stiffness EA           %18.10E\n" % K[0, 0] +
             " The torsional stiffness GJ           %18.10E\n" % K[3, 3] +
             " Principal bending stiffness EI22     %18.10E\n" % ei22p +
             " Principal bending stiffness EI33     %18.10E\n" % ei33p +
             " The principal bending axes rotated from the user coordinate system by\n"
             "   %.10f degrees about the positive direction of x1 axis.\n" % eang)
    x2s, x3s = -S[2, 3] / S[3, 3], S[1, 3] / S[3, 3]      # shear center (from compliance)
    L.append("\n The Shear Center of the Cross Section\n" + bar +
             "\n  %18.10E  %18.10E\n" % (x2s, x3s))
    ga22p, ga33p, gang = _principal_2x2(K[1, 1], K[2, 2], -K[1, 2])
    L.append("\n Principal shear stiffness GA22 =  %18.10E\n" % ga22p +
             " Principal shear stiffness GA33 =  %18.10E\n" % ga33p +
             " The principal shear axes rotated from user coordinate system by\n"
             "   %.10f degrees about the positive direction of x1 axis.\n" % gang)
    if solve_time is not None:
        L.append("\n Time taken: %.2f sec\n" % solve_time)
    open(path, "w").write("".join(L))
    return path


def _append_mass_to_timo_out(timo_out, M):
    """Insert the VABS-style 6x6 mass block into an existing Timo .out (before the
    ' Time taken' line), so the SwiftComp-layout beam .out also carries the mass
    matrix like a VABS .K.  Idempotent: skipped when a mass block is already present.

    In:
        timo_out: str, the <base>_Timo.out written by build_rm_bundle.
        M: (6,6) mass matrix (VABS frame).
    Out:
        str timo_out (unchanged path), or None when the file does not exist.
    """
    import os
    if not os.path.exists(timo_out):
        return None
    txt = open(timo_out).read()
    if "The 6X6 Mass Matrix" in txt:
        return timo_out
    block = ("\n The 6X6 Mass Matrix\n"
             " ========================================================\n\n"
             + _fmt66(np.asarray(M, float)) + "\n")
    lines = txt.splitlines(keepends=True)
    for i, ln in enumerate(lines):
        if ln.strip().startswith("Time taken"):
            lines.insert(i, block + "\n")
            break
    else:
        lines.append(block)
    open(timo_out, "w").write("".join(lines))
    return timo_out


def beam_props(shell_yaml, out_k=None, shear="mitc4_g23", g_source=None):
    """Timoshenko 6x6 + mass 6x6 of a 1-D shell SG, written as a VABS .K-layout file.

    Also inserts the same 6x6 mass block into the station's <base>_Timo.out
    (build_rm_bundle's SwiftComp-layout output), VABS .K style.

    In:
        shell_yaml: str, 1-D shell SG yaml (its `reference` field sets the surface).
        out_k: str | None, output .K path (default <yaml base>.K, `_shell` stripped).
        shear: str, passed to build_rm_bundle (production RM default).
        g_source: DEPRECATED, accepted and ignored -- the wall transverse-shear G is
            always the MSG construction (see sg_materials.check_g_source).
    Out:
        dict: "Timo" (6,6), "Mass" (6,6), "info" mass/geometry dict, "bundle" the RM
        bundle (reusable by the dehom), "k_file" str path.
    """
    import os
    import time
    from ..sg_homo import build_rm_bundle

    t0 = time.perf_counter()
    B = build_rm_bundle(shell_yaml, shear=shear, g_source=g_source)
    C6 = np.asarray(B["Timo"])
    M6, info = mass_matrix_ring(shell_yaml)
    if out_k is None:
        base = os.path.splitext(shell_yaml)[0]
        if base.endswith("_shell"):
            base = base[:-6]
        out_k = base + ".K"
    write_vabs_k(out_k, C6, M6, info,
                 model="OpenSG msg-shell beam model (RM ring, reference=%s)" % B["ref"],
                 solve_time=time.perf_counter() - t0)
    _append_mass_to_timo_out(os.path.splitext(shell_yaml)[0] + "_Timo.out", M6)
    return dict(Timo=C6, Mass=M6, info=info, bundle=B, k_file=out_k)
