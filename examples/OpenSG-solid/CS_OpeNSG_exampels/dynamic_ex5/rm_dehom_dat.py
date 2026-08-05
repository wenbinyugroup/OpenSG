"""rm_dehom_dat.py -- the ABAQUS-DRIVEN OpenSG-RM dehomogenizer.  THE METHOD:
every strain-gradient driver comes from FINITE DIFFERENCES of the strain
field sampled at the element integration points.  No equilibrium closure
anywhere -- not for the gradients, not for sigma33.

WHY FD IS THE METHOD.  Plate equilibrium supplies five equations against the
thirty gradient components the recovery needs (6 x 2 first + 6 x 3 second):
an indeterminate closure except under special symmetry, and the older
mode-consistent seconds assumed the double-sine solution.  The FE solution
itself carries the field, so the strains are sampled at MULTIPLE POINTS
WITHIN EACH ELEMENT -- the 2x2 integration points Abaqus prints, a 40 x 40
lattice with true coordinates from the COORD table -- and differenced.
Validated side by side against the mode-consistent closure on ex5: < 1 %
agreement on every component over the interior (the closures confirm each
other where both are valid); the FD cost is a ~2-element edge band from the
one-sided double-FD stencils.

sigma33 is CONSTITUTIVE: C(z) Gamma(z) with the FD drivers and the applied-
load ladder qt6 carrying the face tractions.  (The removed momentum route
integrated equilibrium through the thickness; under strong inertia its
rho*u3dd term is the one contribution the constitutive route does not see.)

Outputs, next to the input:

    <stem>.U / .EM / .SM       x y z + 3 / 6 / 6 components at every ip
                               station x every SG-element Gauss depth
    <stem>_gauss.vtk           the same field as a structured grid, for
                               render_fields.py / render_deformed.py

    python rm_dehom_dat.py Abaqus_results/ex5_shell_S8R_field.dat

WHAT THE .dat MUST CONTAIN (a "field dump", one increment):
    *NODE PRINT, NSET=NALL          U        all nodes of the plate
    *EL PRINT, ELSET=EALL           SF, SM   all elements (integration points)
    *EL PRINT, ELSET=EALL           COORD    same block structure
Both deck generators write exactly this in `field` mode.

THE LOCAL FRAME.  Abaqus reports shell SF/SM in each element's LOCAL basis.
On this flat plate the default projection puts local 1 along global x for
every element, so local IS global.  On a curved or skewed mesh a per-element
rotation would be needed here.

Everything per-depth (the homo re-run, the Gauss stations, the 42-driver
operators, the writers) is imported from rm_dehom.py (the core).

# ----------------------------------------------------------------------------
# ALL VARIABLES USED IN THIS SCRIPT
# ----------------------------------------------------------------------------
#   args, DAT, STEM     command line; outputs = STEM.SM/.EM/.U + _gauss.vtk
#   DB, db, p           layup_db.yaml and its plate block
#   A, NX, NY, Q0, FINC, P, dx   case data
#   sg, ZS, NZS         core load_sg / z_stations (SG-element Gauss depths)
#   node_table          nodal-U parser (constrained-out nodes scatter as 0)
#   element_table_ip    parser keeping ONE ROW PER INTEGRATION POINT,
#                       (ids, pts, values), row-aligned across tables
#   R_ip, XY_ip         per-ip resultants and their true coordinates
#   XL, YL, NLX, NLY    the sorted unique ip-lattice coordinates
#   IX, IY, Rg          lattice indices; the (NLY, NLX, 8) resultant field
#   E6                  (NLY, NLX, 6) strains S6 R
#   dE1, dE2            FD first gradients (np.gradient, non-uniform lattice)
#   d11, d12, d22       FD second gradients (gradient of gradient; the mixed
#                       term symmetrised, E,12 = E,21)
#   Q1l, Q2l            transverse shear resultants at the ip lattice
#   qt6                 the closed-form APPLIED-load ladder (input, always)
#   D, NIP              the (NIP, 42) driver vectors
#   corner_ids, ug, wx_n, wy_n, bilerp, u0l, wxl, wyl
#                       the nodal displacement pieces -> ip lattice
#   zf                  fine internal depth grid (Q-rescale integrals only)
#   TE, TS, TW, TSf, *j, station   core operators + the vmapped contraction
#   Eps, Sig, Warp, Sigf           the recovery (SG Voigt order)
#   I13, I23, sc13, sc23, q1, q2   the Q-consistency rescale
#   U1, U2, U3, UU      the composed 3-D displacement
#   LAT, title, xs, ys  writer inputs
# ----------------------------------------------------------------------------
"""
import argparse
import os
import re
import sys

import numpy as np
import jax
import jax.numpy as jnp
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from rm_dehom import (load_sg, z_stations, build_ops, material_rotations,
                      write_field, write_vtk, SGIDX, ORDER)

ap = argparse.ArgumentParser()
ap.add_argument("dat", help="Abaqus whole-plate field .dat")
ap.add_argument("--db", default=None,
                help="layup_db.yaml (default: next to this script)")
args = ap.parse_args()
DAT = os.path.abspath(args.dat)
STEM = os.path.splitext(DAT)[0]

DB = args.db or os.path.join(HERE, "layup_db.yaml")
db = yaml.safe_load(open(DB))
p = db["plate"]
A, Q0 = float(p["a"]), float(p["q0"])
NX, NY = int(p["nx"]), int(p["ny"])
FINC = int(p.get("field_inc", 53))
P = np.pi / A
dx = A / NX

sg = load_sg(os.path.join(os.path.dirname(DB), "1dsg.yaml"))
ZS = z_stations(sg, "gauss")
NZS = len(ZS)
print("homo re-run: %d plies -> %d SG-element Gauss depths"
      % (len(sg["inp"]["thick"]), NZS))


# ---- parsers ----------------------------------------------------------------
def node_table(dat, nset, ncomp, nmax=None):
    """(n_nodes, ncomp), row i = node i+1; fully-constrained nodes are
    omitted by Abaqus and scatter in as zero."""
    rows, active, seen = [], False, False
    for ln in open(dat, errors="replace").read().splitlines():
        if ("NODE SET %s" % nset) in ln and "TABLE IS PRINTED" in ln:
            active, seen = True, False
            continue
        tok = ln.split()
        if not tok:
            continue
        if active and not seen:
            if tok[0] == "NODE":
                seen = True
            continue
        if active and seen:
            if re.fullmatch(r"\d+", tok[0]):
                v = [float(tok[0])]
                for t in tok[1:]:
                    try:
                        v.append(float(t))
                    except ValueError:
                        pass
                rows.append(v[:1 + ncomp])
            elif tok[0] in ("MAXIMUM", "MINIMUM"):
                active = False
    rows = np.array(rows)
    if rows.size == 0:
        raise RuntimeError("no NODE SET %s table in %s" % (nset, dat))
    ids = rows[:, 0].astype(int)
    out = np.zeros((max(nmax or 0, ids.max()), ncomp))
    out[ids - 1] = rows[:, 1:1 + ncomp]
    return out


def element_table_ip(dat, marker, ncomp, which=0):
    """One row PER INTEGRATION POINT, in file order: (ids, pts, values).
    The points are the FD sample lattice -- they are never averaged."""
    ids, pts, vals = [], [], []
    active, seen, hit, done = False, False, -1, False
    for ln in open(dat, errors="replace").read().splitlines():
        if "THE FOLLOWING TABLE IS PRINTED" in ln:
            if done:
                break
            if marker in ln:
                hit += 1
            active, seen = (marker in ln and hit == which), False
            continue
        if not active:
            continue
        tok = ln.split()
        if not tok:
            continue
        if not seen:
            if tok[0] == "ELEMENT":
                seen = True
            continue
        if re.fullmatch(r"\d+", tok[0]):
            v = []
            for t in tok:
                try:
                    v.append(float(t))
                except ValueError:
                    pass
            ids.append(int(v[0]))
            pts.append(int(v[1]))
            vals.append(v[-ncomp:])
            done = True
        elif tok[0] in ("MAXIMUM", "MINIMUM"):
            break
    if not ids:
        raise RuntimeError("no '%s' ip table #%d in %s" % (marker, which,
                                                           dat))
    return np.array(ids), np.array(pts), np.array(vals)


def corner_ids():
    """21x21 corner-node ids of the S8R half-index grid."""
    NID, k = {}, 0
    for J in range(2 * NY + 1):
        for I in range(2 * NX + 1):
            if (I % 2) + (J % 2) <= 1:
                k += 1
                NID[(I, J)] = k
    return np.array([[NID[(2 * i, 2 * j)] for i in range(NX + 1)]
                     for j in range(NY + 1)])


# ---- the per-ip resultant field on its own lattice --------------------------
print("parsing %s (per integration point) ..." % os.path.basename(DAT))
ids_r, pts_r, R_ip = element_table_ip(DAT, "INTEGRATION POINTS", 8, 0)
ids_c, pts_c, XY_ip = element_table_ip(DAT, "INTEGRATION POINTS", 3, 1)
assert np.array_equal(ids_r, ids_c) and np.array_equal(pts_r, pts_c), (
    "SF/SM and COORD tables are not row-aligned")
XY_ip = XY_ip[:, :2]

XL = np.unique(np.round(XY_ip[:, 0], 10))
YL = np.unique(np.round(XY_ip[:, 1], 10))
NLX, NLY = len(XL), len(YL)
assert NLX * NLY == len(ids_r), (
    "%d ip do not fill a %d x %d lattice" % (len(ids_r), NLX, NLY))
IX = np.searchsorted(XL, np.round(XY_ip[:, 0], 10))
IY = np.searchsorted(YL, np.round(XY_ip[:, 1], 10))
Rg = np.zeros((NLY, NLX, 8))
Rg[IY, IX] = R_ip
print("ip lattice: %d x %d (true Gauss coordinates from the COORD table)"
      % (NLX, NLY))

# ---- strains and ALL 30 gradient components BY FINITE DIFFERENCES -----------
E6 = np.einsum("ab,ijb->ija", sg["S6"], Rg[:, :, [0, 1, 2, 5, 6, 7]])


def grad(F):
    gy, gx = np.gradient(F, YL, XL, axis=(0, 1))
    return gx, gy


dE1, dE2 = grad(E6)              # the 12 first-gradient drivers (FD)
# The recovery ORDER is the layup_db V2 flag:
#   V2: 0 (default)  Yu Eq. 63 first order -- second-gradient drivers zero.
#                    Measured on ex5: face sigma_xx peak +0.32% (the full
#                    second order overshot it to +8.67%).
#   V2: 1            Yu Eq. 66 second order -- the FD seconds engage the
#                    V21/V22/V23 columns (implemented in the core with the
#                    V2L load quintets; see validate_v2.py: homogeneous and
#                    angle-ply sections excellent, symmetric cross-ply
#                    stacks currently degrade -- known Dbar-source defect).
zero6 = np.zeros_like(E6)
if int(db.get("V2", 1)):
    d11, d21g = grad(dE1)
    d12g, d22 = grad(dE2)
    d12 = 0.5 * (d12g + d21g)    # E,12 = E,21
    print("V2: 1 -- second-order recovery (Eq. 66), FD second gradients on")
else:
    d11 = d12 = d22 = zero6

Q1l, Q2l = Rg[:, :, 3], Rg[:, :, 4]

# the APPLIED-load ladder stays closed-form: q is input, not closure
Xl, Yl = np.meshgrid(XL, YL)
s1, c1 = np.sin(P * Xl), np.cos(P * Xl)
s2, c2 = np.sin(P * Yl), np.cos(P * Yl)
qt6 = Q0 * np.stack([s1 * s2, P * c1 * s2, P * s1 * c2, -P * P * s1 * s2,
                     P * P * c1 * c2, -P * P * s1 * s2], axis=-1)

D = np.concatenate([E6, dE1, dE2, d11, d12, d22, qt6],
                   axis=-1).reshape(-1, 42)
NIP = NLX * NLY

# ---- displacement pieces (nodal -> the ip lattice) --------------------------
U_all = node_table(DAT, "NALL", 3)
ug = U_all[corner_ids() - 1]
wy_n, wx_n = np.gradient(ug[:, :, 2], dx, dx)


def bilerp(F):
    f = XL / dx
    i0 = np.clip(f.astype(int), 0, NX - 1)
    t = (f - i0)[None, :]
    Fi = (1 - t) * F[:, i0] + t * F[:, i0 + 1]
    f2 = YL / dx
    j0 = np.clip(f2.astype(int), 0, NY - 1)
    t2 = (f2 - j0)[:, None]
    return (1 - t2) * Fi[j0] + t2 * Fi[j0 + 1]


u0l = np.stack([bilerp(ug[:, :, c]) for c in range(3)], axis=-1)
wxl, wyl = bilerp(wx_n), bilerp(wy_n)

# ---- operators (core) + one vmap contraction --------------------------------
zf = np.linspace(-sg["H"] / 2 + 1e-9, sg["H"] / 2 - 1e-9, 61)
print("building operators: (%d + %d fine) x 42 unit drivers ..."
      % (NZS, len(zf)))
TE, TS, TW = build_ops(sg["r"], ZS)
_, TSf, _ = build_ops(sg["r"], zf + sg["OFF"])
TEj, TSj, TWj, TSfj = (jnp.asarray(a) for a in (TE, TS, TW, TSf))


@jax.jit
def station(d):
    return (jnp.einsum("zcj,j->zc", TEj, d), jnp.einsum("zcj,j->zc", TSj, d),
            jnp.einsum("zcj,j->zc", TWj, d), jnp.einsum("zcj,j->zc", TSfj, d))


print("jax.vmap dehomogenization over %d ip stations ..." % NIP)
Eps, Sig, Warp, Sigf = (np.array(a) for a in jax.vmap(station)(jnp.asarray(D)))
# SG Voigt order everywhere: (11, 22, 33, 23, 13, 12)

# transverse-shear consistency: each station's integral carries its own Q1/Q2
I13 = np.trapezoid(Sigf[:, :, 4], zf, axis=1)
I23 = np.trapezoid(Sigf[:, :, 3], zf, axis=1)
q1, q2 = Q1l.reshape(-1), Q2l.reshape(-1)
sc13 = np.where(np.abs(I13) > 1e-3 * np.abs(q1).max(), q1 / I13, 1.0)
sc23 = np.where(np.abs(I23) > 1e-3 * np.abs(q2).max(), q2 / I23, 1.0)
Sig[:, :, 4] *= sc13[:, None]
Sig[:, :, 3] *= sc23[:, None]

# ---- ply MATERIAL frame for the OUTPUT fields -------------------------------
# The whole chain above (FD gradients, operators, Q-rescale) runs in the
# plate/laminate frame -- it must (the rescale integrates laminate sigma13/23
# across the 90-deg plies).  The DELIVERED stress/strain are in the ply axes
# of each depth (failure criteria live there): rotate per z-station AFTER the
# rescale.  sigma33/eps33 rows are invariant; displacement stays global.
Rs, Re = material_rotations(sg, ZS)
Sig = np.einsum("zab,nzb->nza", Rs, Sig)
Eps = np.einsum("zab,nzb->nza", Re, Eps)

# ---- 3-D displacement -------------------------------------------------------
u0 = u0l.reshape(-1, 3)
U1 = u0[:, 0:1] - ZS[None, :] * wxl.reshape(-1)[:, None] + Warp[:, :, 0]
U2 = u0[:, 1:2] - ZS[None, :] * wyl.reshape(-1)[:, None] + Warp[:, :, 1]
U3 = u0[:, 2:3] + Warp[:, :, 2]

# ---- output: the three files AND the VTK ------------------------------------
LAT = ("the %d x %d integration points x the %d SG-element Gauss depths; "
       "ALL 30 gradient drivers by finite differences; S/E in the PLY "
       "MATERIAL frame (U global)" % (NLX, NLY, NZS))
title = ("OpenSG-RM FD recovery of %s  (increment %d)"
         % (os.path.basename(DAT), FINC))
xs = np.tile(XL, NLY)
ys = np.repeat(YL, NLX)
UU = np.stack([U1, U2, U3], axis=-1)
write_field(STEM + ".SM", "S", "Pa", Sig, xs, ys, ZS, LAT, title)
write_field(STEM + ".EM", "E", "-", Eps, xs, ys, ZS, LAT, title)
write_field(STEM + ".U", "U", "m", UU, xs, ys, ZS, LAT, title)
write_vtk(STEM + "_gauss.vtk", XL, YL, ZS,
          UU.reshape(NLY, NLX, NZS, 3), Sig.reshape(NLY, NLX, NZS, 6), title)
print("wrote %s.SM / .EM / .U / _gauss.vtk  (%d points)"
      % (os.path.basename(STEM), NIP * NZS))
for nm in ORDER:
    print("  max|S%s| = %.4e Pa   max|E%s| = %.4e"
          % (nm, np.abs(Sig[:, :, SGIDX[nm]]).max(),
             nm, np.abs(Eps[:, :, SGIDX[nm]]).max()))
