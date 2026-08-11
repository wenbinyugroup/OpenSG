"""rm_dehom_ff.py -- the GENERAL plate dehomogenization driver: recovery
from a .ff station table + layup_db.yaml ONLY.  No Abaqus anywhere.

    input   layup_db.yaml     the SG (via 1dsg.yaml next to it), the V2
                              flag, and the dehom: block:
                                dehom:
                                  ff: <path>.ff        # the station table
                                  loads:
                                    top: {q0: ..., shape: sinsin}
                                    bottom: none
            <path>.ff         x y N11 N22 N12 M11 M22 M12 Q1 Q2
                              [u1 u2 u3 wx wy]   (see abq2ff.py for the
                              full column spec; any solver can write it)
    output  <path>_ff.SM/.EM/.U (+ _ff_gauss.vtk on a lattice) --
            stress/strain in the PLY MATERIAL frame, displacement global

THE MESH IS GENERAL.  Two gradient paths, chosen automatically:
  * lattice  -- the station coordinates form a complete rectangular
    lattice (any row order): EXACT tensor-grid finite differences
    (np.gradient on the true non-uniform spacings), as rm_dehom_dat.py.
  * meshfree -- any other station cloud (unstructured mesh, mixed
    element sizes): core mls_gradients -- k-nearest-neighbour patches
    (KD-tree) + distance-weighted local QUADRATIC least squares; the
    derivatives are the fitted coefficients, first AND second from one
    batched solve.  Exact for locally quadratic fields; reduces to the
    lattice stencil on a grid.  --scatter forces this path (testing).

THE RULES this driver embodies (dehom_README.md):
  * FD is THE method: every gradient driver is a finite difference /
    patch fit of the .ff stations.  One row -> classical recovery.
    No equilibrium closure anywhere.
  * loads are INPUT: the applied face-pressure ladder qt6 comes from the
    dehom: loads: block (uniform | sin_x | sin_y | sinsin, differentiated
    analytically with a, b from plate:), NEVER from the solver output.
    bottom loads need a 48-driver operator extension -> NotImplementedError.
  * Q1/Q2 are rescale targets only (per-station integral consistency).

Run:  python rm_dehom_ff.py                    # paths from layup_db.yaml
      python rm_dehom_ff.py path/to/table.ff   # explicit table
      python rm_dehom_ff.py --scatter          # force the meshfree path
"""
import argparse
import os
import sys
import time

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import time as _t
print("start: " + _t.strftime("%Y-%m-%d %H:%M:%S"))

from opensg_solid.rm_plate_1D.rm_dehom import (
    load_sg, z_stations, build_ops, material_rotations,
    mls_gradients, write_field, write_vtk, SGIDX, ORDER)

import jax
import jax.numpy as jnp

ap = argparse.ArgumentParser()
ap.add_argument("ff", nargs="?", default=None,
                help=".ff station table (default: dehom: ff in the db)")
ap.add_argument("--db", default=None,
                help="layup_db.yaml (default: next to this script)")
ap.add_argument("--scatter", action="store_true",
                help="force the meshfree (MLS patch) gradient path even "
                     "when the stations form a lattice")
args = ap.parse_args()

DB = args.db or os.path.join(HERE, "layup_db.yaml")
db = yaml.safe_load(open(DB))
dh = db.get("dehom") or {}
FF_PATH = os.path.abspath(args.ff or os.path.join(os.path.dirname(DB),
                                                  dh.get("ff", "")))
if not os.path.isfile(FF_PATH):
    raise ValueError(
        "no .ff table: pass it on the command line or set dehom: ff in %s"
        % DB)
STEM = os.path.splitext(FF_PATH)[0] + "_ff"

pl = db.get("plate") or {}
A = float(pl.get("a", 0.0)) or None
Bside = float(pl.get("b", pl.get("a", 0.0))) or None

sg = load_sg(os.path.join(os.path.dirname(DB), "1dsg.yaml"))
ZS = z_stations(sg, "gauss")
NZS = len(ZS)
print("homo re-run: %d plies -> %d SG-element Gauss depths"
      % (len(sg["inp"]["thick"]), NZS))

# ---- the station table ------------------------------------------------------
T = np.atleast_2d(np.loadtxt(FF_PATH))
if T.shape[1] not in (10, 15):
    raise ValueError(".ff must have 10 or 15 columns, got %d" % T.shape[1])
HAS_DISP = T.shape[1] == 15
x_st, y_st = T[:, 0], T[:, 1]
NST = len(T)
V2 = int(db.get("V2", 1))

XL = np.unique(np.round(x_st, 10))
YL = np.unique(np.round(y_st, 10))
NLX, NLY = len(XL), len(YL)
structured = (not args.scatter) and NLX * NLY == NST

# ---- gradient drivers (the path chosen by the station geometry) -------------
t0 = time.perf_counter()
if structured:
    # EXACT tensor-grid FD on the true non-uniform spacings (row order of
    # the .ff is irrelevant: the scatter below rebuilds the lattice).
    IX = np.searchsorted(XL, np.round(x_st, 10))
    IY = np.searchsorted(YL, np.round(y_st, 10))
    G = np.zeros((NLY, NLX, T.shape[1] - 2))
    G[IY, IX] = T[:, 2:]
    E6g = np.einsum("ab,ijb->ija", sg["S6"], G[:, :, 0:6])

    def grad(F):
        gy = (np.gradient(F, YL, axis=0) if NLY > 1 else np.zeros_like(F))
        gx = (np.gradient(F, XL, axis=1) if NLX > 1 else np.zeros_like(F))
        return gx, gy

    dE1g, dE2g = grad(E6g)
    if V2:
        d11g, d21g = grad(dE1g)
        d12g, d22g = grad(dE2g)
        d12g = 0.5 * (d12g + d21g)
    else:
        d11g = d12g = d22g = np.zeros_like(E6g)
    # per-station arrays, LATTICE order (x fastest)
    xs, ys = np.tile(XL, NLY), np.repeat(YL, NLX)
    E6s, dE1s, dE2s = (a.reshape(-1, 6) for a in (E6g, dE1g, dE2g))
    d11s, d12s, d22s = (a.reshape(-1, 6) for a in (d11g, d12g, d22g))
    Q1s, Q2s = G[:, :, 6].reshape(-1), G[:, :, 7].reshape(-1)
    DISP = G[:, :, 8:13].reshape(-1, 5) if HAS_DISP else None
    mode = "lattice FD (%d x %d)" % (NLX, NLY)
else:
    # meshfree: any station cloud (core mls_gradients)
    xs, ys = x_st, y_st
    E6s = T[:, 2:8] @ sg["S6"].T
    gx, gy, gxx, gxy, gyy = mls_gradients(xs, ys, E6s, k=12)
    dE1s, dE2s = gx, gy
    if V2:
        d11s, d12s, d22s = gxx, gxy, gyy
    else:
        d11s = d12s = d22s = np.zeros_like(E6s)
    Q1s, Q2s = T[:, 8], T[:, 9]
    DISP = T[:, 10:15] if HAS_DISP else None
    mode = "meshfree MLS patch (k=12, %d stations)" % NST
print("gradient drivers: %s in %.3f s" % (mode, time.perf_counter() - t0))


# ---- the APPLIED-load ladder from dehom: loads: (input, never closure) ------
def load_ladder(spec, px, py):
    """(n, 6) ladder [q, q,1, q,2, q,11, q,12, q,22] of one face at the
    station coordinates px, py."""
    if spec in (None, "none", {}):
        return None
    q0 = float(spec["q0"])
    shape = str(spec.get("shape", "uniform"))
    if shape != "uniform" and (A is None or Bside is None):
        raise ValueError("shape '%s' needs plate: a (and b) in the db"
                         % shape)
    zero = np.zeros_like(px)
    if shape == "uniform":
        return np.stack([q0 + zero, zero, zero, zero, zero, zero], axis=-1)
    Px, Py = np.pi / A, np.pi / Bside
    if shape == "sin_x":
        s, c = np.sin(Px * px), np.cos(Px * px)
        return q0 * np.stack([s, Px * c, zero, -Px * Px * s, zero, zero],
                             axis=-1)
    if shape == "sin_y":
        s, c = np.sin(Py * py), np.cos(Py * py)
        return q0 * np.stack([s, zero, Py * c, zero, zero, -Py * Py * s],
                             axis=-1)
    if shape == "sinsin":
        s1, c1 = np.sin(Px * px), np.cos(Px * px)
        s2, c2 = np.sin(Py * py), np.cos(Py * py)
        return q0 * np.stack([s1 * s2, Px * c1 * s2, Py * s1 * c2,
                              -Px * Px * s1 * s2, Px * Py * c1 * c2,
                              -Py * Py * s1 * s2], axis=-1)
    raise ValueError("unknown load shape '%s' (uniform|sin_x|sin_y|sinsin)"
                     % shape)


loads = dh.get("loads") or {}
qt6 = load_ladder(loads.get("top"), xs, ys)
if load_ladder(loads.get("bottom"), xs, ys) is not None:
    raise NotImplementedError(
        "bottom-face loads need the 48-driver operator extension (the "
        "42-driver block carries qt6 only); split-face recovery is "
        "available through msgrm_strain_at_depth(qb6=...) directly")
if qt6 is None:
    qt6 = np.zeros_like(E6s)

D = np.concatenate([E6s, dE1s, dE2s, d11s, d12s, d22s, qt6], axis=1)
NIP = NST

# ---- operators (core) + one vmap contraction --------------------------------
zf = np.linspace(-sg["H"] / 2 + 1e-9, sg["H"] / 2 - 1e-9, 61)
print("building operators: (%d + %d fine) x 42 unit drivers ..."
      % (NZS, len(zf)))
t0 = time.perf_counter()
TE, TS, TW = build_ops(sg["r"], ZS)
_, TSf, _ = build_ops(sg["r"], zf + sg["OFF"])
print("operators built in %.1f s" % (time.perf_counter() - t0))
TEj, TSj, TWj, TSfj = (jnp.asarray(a) for a in (TE, TS, TW, TSf))


@jax.jit
def station(d):
    return (jnp.einsum("zcj,j->zc", TEj, d), jnp.einsum("zcj,j->zc", TSj, d),
            jnp.einsum("zcj,j->zc", TWj, d), jnp.einsum("zcj,j->zc", TSfj, d))


print("jax.vmap dehomogenization over %d stations ..." % NIP)
Eps, Sig, Warp, Sigf = (np.array(a) for a in jax.vmap(station)(jnp.asarray(D)))

# ---- transverse-shear consistency (per-station Q1/Q2 targets) ---------------
I13 = np.trapezoid(Sigf[:, :, 4], zf, axis=1)
I23 = np.trapezoid(Sigf[:, :, 3], zf, axis=1)
if np.abs(Q1s).max() > 0:
    sc13 = np.where(np.abs(I13) > 1e-3 * np.abs(Q1s).max(), Q1s / I13, 1.0)
    Sig[:, :, 4] *= sc13[:, None]
if np.abs(Q2s).max() > 0:
    sc23 = np.where(np.abs(I23) > 1e-3 * np.abs(Q2s).max(), Q2s / I23, 1.0)
    Sig[:, :, 3] *= sc23[:, None]

# ---- ply MATERIAL frame for the OUTPUT fields (after the rescale) -----------
Rs, Re = material_rotations(sg, ZS)
Sig = np.einsum("zab,nzb->nza", Rs, Sig)
Eps = np.einsum("zab,nzb->nza", Re, Eps)

# ---- 3-D displacement -------------------------------------------------------
if HAS_DISP:
    u0 = DISP[:, 0:3]
    wxl, wyl = DISP[:, 3], DISP[:, 4]
    U1 = u0[:, 0:1] - ZS[None, :] * wxl[:, None] + Warp[:, :, 0]
    U2 = u0[:, 1:2] - ZS[None, :] * wyl[:, None] + Warp[:, :, 1]
    U3 = u0[:, 2:3] + Warp[:, :, 2]
    UU = np.stack([U1, U2, U3], axis=-1)
    unote = ""
else:
    UU = Warp.copy()
    unote = " (WARPING ONLY: no displacement columns in the .ff)"

# ---- output -----------------------------------------------------------------
LAT = ("%d .ff stations [%s] x the %d SG-element Gauss depths; ALL "
       "gradient drivers from the stations; S/E in the PLY MATERIAL frame "
       "(U global%s)" % (NST, mode, NZS, unote))
title = "OpenSG-RM .ff-table recovery of %s" % os.path.basename(FF_PATH)
write_field(STEM + ".SM", "S", "Pa", Sig, xs, ys, ZS, LAT, title)
write_field(STEM + ".EM", "E", "-", Eps, xs, ys, ZS, LAT, title)
write_field(STEM + ".U", "U", "m", UU, xs, ys, ZS, LAT, title)
if structured:
    write_vtk(STEM + "_gauss.vtk", XL, YL, ZS,
              UU.reshape(NLY, NLX, NZS, 3), Sig.reshape(NLY, NLX, NZS, 6),
              title)
    print("wrote %s.SM / .EM / .U / _gauss.vtk  (%d points)"
          % (os.path.basename(STEM), NIP * NZS))
else:
    print("wrote %s.SM / .EM / .U  (%d points; scattered stations -> no "
          "structured-grid VTK)" % (os.path.basename(STEM), NIP * NZS))
for nm in ORDER:
    print("  max|S%s| = %.4e Pa" % (nm, np.abs(Sig[:, :, SGIDX[nm]]).max()))
print("end:   " + _t.strftime("%Y-%m-%d %H:%M:%S"))
