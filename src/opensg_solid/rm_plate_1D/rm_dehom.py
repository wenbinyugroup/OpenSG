"""rm_dehom.py -- the GENERAL OpenSG-RM dehomogenization core, promoted to
the core from the ex5 pipeline (examples/opensg-rm_dynamic/ex5/rm_dehom.py)
because nothing in it is example-specific: every recovery script imports
from here.

What lives here (all documented in the ex5 dehom_README.md and derived in
dehom_theory.pdf):

    load_sg(sg_yaml)         read the 1-D SG and RE-RUN the homogenization
                             (the V0/V11/V12 nodal warping ladders) -- never
                             cached, milliseconds on a line mesh
    z_stations(sg, lattice)  the output depths: "gauss" (default; the
                             NPE-point Gauss abscissae of every SG element,
                             strictly inside plies) or "nodes"
    build_ops(r, z_sg)       (nz, 6, 42) strain + stress and (nz, 3, 42)
                             warping operators from 42 unit evaluations --
                             the recovery is LINEAR in its drivers
    assemble_driver(...)     the 42-vector [E6 dE1 dE2 d11 d12 d22 qt6]
    write_field(...)         the .SM/.EM/.U text writer
    write_vtk(...)           structured-grid VTK from the same arrays (a
                             single station becomes a 1 x 1 x nz line)
    SGIDX, ORDER             SG Voigt positions and the printed order

The single-station USER-INPUT dehom and the Abaqus-driven whole-plate vmap
live in the example pipelines (ex5/rm_dehom.py, ex5/rm_dehom_dat.py) and
import everything above.
"""
import os

import numpy as np

from .segment_plate import read_plate_sg_yaml
from .msg_rm_plate import (rm_plate_msg, msgrm_strain_at_depth,
                           msgrm_warping_at_depth)

SGIDX = {"11": 0, "22": 1, "33": 2, "12": 5, "13": 4, "23": 3}
ORDER = ("11", "22", "33", "12", "13", "23")


def load_sg(sg_yaml):
    """Read the 1-D SG and RE-RUN the homogenization (V0/V11/V12 nodal
    warping ladders).  Milliseconds on a line mesh -- never cached."""
    inp = read_plate_sg_yaml(sg_yaml)
    r = rm_plate_msg(inp["thick"], inp["angles"], inp["mat_names"],
                     inp["material_db"], n_per_layer=inp["n_per_layer"],
                     elem_order=inp["elem_order"], fraction=inp["fraction"])
    H = float(sum(inp["thick"]))
    return {"inp": inp, "r": r, "S6": np.linalg.inv(np.asarray(r["A6"])),
            "H": H, "OFF": (0.5 - inp["fraction"]) * H,
            "ZNODE": np.array(inp["node_x"], dtype=float),
            "NPE": int(inp["elem_order"]) + 1}


def z_stations(sg, lattice="gauss"):
    """The through-thickness output depths, SG frame (0 = reference plane).
    gauss: NPE-point Gauss-Legendre inside every SG element -- strictly
    interior, no ply-interface double values.  nodes: the SG nodes."""
    ZN, NPE = sg["ZNODE"], sg["NPE"]
    if lattice == "gauss":
        xi, _ = np.polynomial.legendre.leggauss(NPE)
        nel = (len(ZN) - 1) // (NPE - 1)
        return np.concatenate([
            0.5 * (ZN[e * (NPE - 1)] + ZN[(e + 1) * (NPE - 1)])
            + 0.5 * (ZN[(e + 1) * (NPE - 1)] - ZN[e * (NPE - 1)]) * xi
            for e in range(nel)])
    ZS = ZN.copy()
    ZS[0] += 1e-9
    ZS[-1] -= 1e-9
    return ZS


def build_ops(r, z_sg):
    """(nz, 6, 42) strain + stress and (nz, 3, 42) warping operators: the
    recovery is LINEAR in its 42 drivers, so 42 unit evaluations per depth
    capture it completely.

    The operators are built in the PLATE/LAMINATE frame (frame="plate"
    explicitly) -- the Q-consistency rescale and any equilibrium/momentum
    integral contract laminate-frame sigma13/sigma23 across plies, so the
    per-ply material rotation must happen AFTER those steps, at write time
    (material_rotations below)."""
    TE = np.zeros((len(z_sg), 6, 42))
    TS = np.zeros((len(z_sg), 6, 42))
    TW = np.zeros((len(z_sg), 3, 42))
    for iz, z in enumerate(z_sg):
        for j in range(42):
            d = np.zeros(42)
            d[j] = 1.0
            a = (r, z, d[0:6], d[6:12], d[12:18], d[18:24], d[24:30],
                 d[30:36])
            gam, sig, _ = msgrm_strain_at_depth(*a, qt6=d[36:42],
                                                frame="plate")
            TE[iz, :, j], TS[iz, :, j] = gam, sig
            TW[iz, :, j] = msgrm_warping_at_depth(*a, qt6=d[36:42])
    return TE, TS, TW


def material_rotations(sg, z_sg):
    """Per-station laminate -> ply-material rotation matrices.

    In:  sg dict (load_sg), z_sg (nz,) depths (node_x origin, as passed to
         build_ops)
    Out: (Rs, Re): (nz, 6, 6) each -- Sig_mat[iz] = Rs[iz] @ Sig[iz],
         Gam_mat[iz] = Re[iz] @ Gam[iz] (engineering shear).  Apply AFTER
         the Q-consistency rescale, right before the writers.  sigma33/
         eps33 rows are untouched by construction."""
    from .msg_materials import rotation_6x6
    inp = sg["inp"]
    npl = int(inp["n_per_layer"])
    node_x = np.asarray(inp["node_x"], float)
    p = int(inp["elem_order"])
    n_elem = (len(node_x) - 1) // p
    Rs = np.zeros((len(z_sg), 6, 6))
    Re = np.zeros((len(z_sg), 6, 6))
    for iz, z in enumerate(z_sg):
        e = int(np.clip(np.searchsorted(node_x[::p][1:], z, side="right"),
                        0, n_elem - 1))
        ang = float(inp["angles"][e // npl])
        Rs[iz] = rotation_6x6(-ang)
        Re[iz] = rotation_6x6(ang).T
    return Rs, Re


def mls_gradients(x, y, F, k=12):
    """Meshfree gradient estimator for a GENERAL station cloud (no lattice).

    In:  x, y (n,) float -- station coordinates; F (n, m) float -- the
         per-station fields (e.g. the 6 plate strains); k int -- patch
         size: the k NEAREST stations by distance (element membership is
         irrelevant, exactly as on the lattice), k >= 6
    Out: (dFx, dFy, dxx, dxy, dyy) -- (n, m) each: first and second
         derivatives at every station

    Method: distance-weighted local QUADRATIC least squares (moving least
    squares -- the SPR patch idea; lattice FD is its tensor-grid special
    case).  Per station the centered monomial basis [1, xi, eta, xi^2/2,
    xi*eta, eta^2/2] is fitted over the patch; the derivatives ARE the
    fitted coefficients, first and second from ONE solve.  Offsets are
    normalized by the patch radius and the geometry normal matrix is
    solved once per station for all m components (batched 6x6).  Exact
    for fields quadratic in (x, y), any weights.  Edge stations have
    one-sided patches and degrade exactly like one-sided FD; degenerate
    (collinear) patches raise LinAlgError."""
    from scipy.spatial import cKDTree
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    F = np.asarray(F, float)
    n = len(x)
    k = int(min(k, n))
    if k < 6:
        raise ValueError("the quadratic patch fit needs >= 6 stations "
                         "(got %d); a single station is classical recovery "
                         "-- leave the gradients zero" % n)
    d, idx = cKDTree(np.column_stack([x, y])).query(
        np.column_stack([x, y]), k=k)
    rc = np.maximum(d[:, -1], 1e-30)               # patch radius (conditioning)
    xi = (x[idx] - x[:, None]) / rc[:, None]
    eta = (y[idx] - y[:, None]) / rc[:, None]
    w = np.exp(-4.0 * (d / rc[:, None]) ** 2)
    P = np.stack([np.ones_like(xi), xi, eta,
                  0.5 * xi ** 2, xi * eta, 0.5 * eta ** 2], axis=-1)
    PW = P * w[:, :, None]
    A = np.einsum("nka,nkb->nab", PW, P)           # (n, 6, 6) geometry
    B = np.einsum("nka,nkm->nam", PW, F[idx])      # (n, 6, m) data
    c = np.linalg.solve(A, B)
    s1 = rc[:, None]
    s2 = (rc ** 2)[:, None]
    return c[:, 1] / s1, c[:, 2] / s1, c[:, 3] / s2, c[:, 4] / s2, c[:, 5] / s2


def assemble_driver(E6, dE1=None, dE2=None, d11=None, d12=None, d22=None,
                    qt6=None):
    """The 42-vector the operators contract with."""
    z = np.zeros(6)
    return np.concatenate([np.asarray(E6, float)] +
                          [np.asarray(v, float) if v is not None else z
                           for v in (dE1, dE2, d11, d12, d22, qt6)])


def write_field(path, name, unit, F, xs, ys, ZS, lattice_note, title):
    """The .SM/.EM/.U text writer.  F: (n_station, nz, 6) in SG Voigt order,
    or (n_station, nz, 3) for displacement; xs/ys the station coordinates."""
    with open(path, "w") as f:
        cols = (["%s%s[%s]" % (name, c, unit) for c in ORDER]
                if F.shape[-1] == 6 else ["U1[m]", "U2[m]", "U3[m]"])
        f.write("# %s\n# lattice: %s; z = 0 is the reference surface\n"
                % (title, lattice_note))
        f.write("# %12s %14s %14s" % ("x[m]", "y[m]", "z[m]")
                + "".join(" %14s" % c for c in cols) + "\n")
        for n in range(F.shape[0]):
            for k in range(F.shape[1]):
                row = (F[n, k, [SGIDX[c] for c in ORDER]]
                       if F.shape[-1] == 6 else F[n, k])
                f.write("%14.6e %14.6e %14.6e" % (xs[n], ys[n], ZS[k])
                        + "".join(" %14.6e" % v for v in row) + "\n")


def write_vtk(path, X, Y, ZS, D, S, title):
    """Structured-grid VTK from the recovery arrays -- BY DEFAULT for every
    dehom, single station included (then the grid is a 1 x 1 x nz line).
    X, Y: the 1-D in-plane coordinate lists; D: (ny, nx, nz, 3)
    displacement; S: (ny, nx, nz, 6) stress in SG Voigt order."""
    nx, ny, nz = len(X), len(Y), len(ZS)
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\n%s\nASCII\n"
                "DATASET STRUCTURED_GRID\nDIMENSIONS %d %d %d\n"
                "POINTS %d float\n" % (title, nx, ny, nz, nx * ny * nz))
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    f.write("%.6e %.6e %.6e\n" % (X[i], Y[j], ZS[k]))
        f.write("POINT_DATA %d\nVECTORS disp float\n" % (nx * ny * nz))
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    f.write("%.6e %.6e %.6e\n" % tuple(D[j, i, k]))
        for nm in ORDER:
            f.write("SCALARS S%s float 1\nLOOKUP_TABLE default\n" % nm)
            for k in range(nz):
                for j in range(ny):
                    for i in range(nx):
                        f.write("%.6e\n" % S[j, i, k, SGIDX[nm]])
    print("wrote %s  (%d x %d x %d)" % (os.path.basename(path), nx, ny, nz))
