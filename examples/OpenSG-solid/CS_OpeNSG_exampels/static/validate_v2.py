"""validate_v2.py -- IS THE SECOND-ORDER RECOVERY (V2 + V2L) CORRECT?
The four Pagano cylindrical-bending statics as the gate.

For every case the harmonic drivers are EXACT closed forms (E6 = Es sin(px),
so E,1 = p Es cos, E,11 = -p^2 Es sin -- no FD anywhere), the face load
ladders qt6/qb6 are the applied tractions (split faces for the Yu cases), and
sigma33 is recovered CONSTITUTIVELY at x = a/2 with

    V2 ON    the full Eq.-(66) second-order recovery: dE11/dE12/dE22 passed,
             engaging the V21/V22/V23 columns AND the V2L load quintets
    V2 OFF   the Eq.-(63) first-order recovery (seconds zero; the load
             ladders still passed -- the pipeline's V2: 0 semantics)

against the EXACT 3-D profiles already tabulated in the engine .dat files.
Also reported: the face-traction closure (sigma33 at both faces must equal
the applied +-q, machine-exact when V2L is right -- THE V2L correctness
check), sigma13 at x = 0 (V2-insensitive there: all seconds vanish at the
cos peak -- a sanity row), and the top-face sigma11 trade.

Run:  python validate_v2.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "src", "opensg_solid", "rm_plate_1D")):
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "examples", "OpenSG-solid",
                                "CS_OpeNSG_exampels", "static", "yu2003"))

from opensg_solid.rm_plate_1D.rm_homo import load_layup_db
from opensg_solid.rm_plate_1D.msg_rm_plate import rm_plate_msg, msgrm_strain_at_depth
from yu_bench import rm_cyl_bend

CASES = [
    # folder, engine dat, S list, split-face load?  The S list must match the
    # .dat files present under <folder>/<case>/ -- the folders carry L/h = 4
    # only (bench.S = [4]); widen bench.S and rerun run_case.py to regenerate
    # thinner references before adding S values here.
    ("garg_caseA", "caseA/pagano_S%g.dat", (4,), False),
    ("garg_caseC", "caseC/pagano_S%g.dat", (4,), False),
    ("yu2003_case1", "case1/yu_case1.dat", (4,), True),
    ("yu2003_case2", "case2/yu_case2.dat", (4,), True),
]


def gauss_depths(r):
    """The SG-element Gauss depths (strictly inside plies)."""
    zn = np.asarray(r["node_x"], float)
    npe = int(r["elem_order"]) + 1
    xi, _ = np.polynomial.legendre.leggauss(npe)
    nel = (len(zn) - 1) // (npe - 1)
    return np.concatenate([
        0.5 * (zn[e * (npe - 1)] + zn[(e + 1) * (npe - 1)])
        + 0.5 * (zn[(e + 1) * (npe - 1)] - zn[e * (npe - 1)]) * xi
        for e in range(nel)])


def profile(r, zs, E6, dE1, dE2, d2, qt6, qb6):
    """(nz, 6) stress profile; d2 = (d11, d12, d22) or None for Eq. 63."""
    out = np.empty((len(zs), 6))
    for i, z in enumerate(zs):
        d11, d12, d22 = d2 if d2 is not None else (None, None, None)
        out[i] = np.asarray(msgrm_strain_at_depth(
            r, z, E6, dE1, dE2, d11, d12, d22, qt6=qt6, qb6=qb6,
            frame="plate")[1])
    return out


def rel_l2(m, e):
    return 100 * np.sqrt(np.trapezoid((m - e) ** 2)) / \
        max(np.sqrt(np.trapezoid(e ** 2)), 1e-30)


print("%-14s %4s | %9s %9s | %11s %11s | %8s" %
      ("case", "S", "s33 V2on", "s33 V2off", "face top", "face bot",
       "s11 top"))
print("-" * 88)
for folder, datpat, S_list, split in CASES:
    d = load_layup_db(os.path.join(HERE, folder, "layup_db.yaml"))
    p_blk = d["db"]["plate"]
    a, q0 = float(p_blk["a"]), float(p_blk["q0"])
    p = np.pi / a
    base_h = sum(d["layup"]["thick"])
    for S in S_list:
        h = a / S
        thick = [t * h / base_h for t in d["layup"]["thick"]]
        r = rm_plate_msg(thick, d["layup"]["angles"], d["layup"]["mat_names"],
                         d["material_db"], n_per_layer=d["n_per_layer"],
                         elem_order=d["elem_order"], fraction=0.5)
        A6 = np.asarray(r["A6"])
        G2 = np.asarray(r["ABDG"])[6:8, 6:8]
        y, Es, gs, R6, Q = rm_cyl_bend(A6, G2, p, q0)

        # face load ladders (LOCAL face pressure; sigma33 = q on that face)
        if split:
            qt6 = np.array([q0 / 2, 0, 0, -p * p * q0 / 2, 0, 0])
            qb6 = np.array([-q0 / 2, 0, 0, p * p * q0 / 2, 0, 0])
        else:
            qt6 = np.array([q0, 0, 0, -p * p * q0, 0, 0])
            qb6 = None

        zs = gauss_depths(r)
        z6 = np.zeros(6)
        # station x = a/2 (the sin peak): E6 = Es, dE = 0, E,11 = -p^2 Es
        d2 = (-p * p * Es, z6, z6)
        s_on = profile(r, zs, Es, z6, z6, d2, qt6, qb6)
        s_off = profile(r, zs, Es, z6, z6, None, qt6, qb6)
        # face closure of the V2 route (the V2L correctness check):
        zb, zt = r["node_x"][0] + 1e-9, r["node_x"][-1] - 1e-9
        f_top = np.asarray(msgrm_strain_at_depth(
            r, zt, Es, z6, z6, *d2, qt6=qt6, qb6=qb6)[1])[2]
        f_bot = np.asarray(msgrm_strain_at_depth(
            r, zb, Es, z6, z6, *d2, qt6=qt6, qb6=qb6)[1])[2]
        want_t = qt6[0]
        want_b = qb6[0] if qb6 is not None else 0.0

        # exact reference from the engine .dat
        dat = os.path.join(HERE, folder,
                           datpat % S if "%" in datpat else datpat)
        D = np.loadtxt(dat)
        zex = D[:, 0]
        s33_ex = np.interp(zs, zex, D[:, 5] if not split else D[:, 7])
        s33_on = rel_l2(s_on[:, 2], s33_ex)
        s33_off = rel_l2(s_off[:, 2], s33_ex)

        # top-face sigma11 trade (in-plane; exact from the state solver is
        # not tabulated in the .dat, so report the ON-vs-OFF shift instead)
        s11_shift = 100 * (s_on[-1, 0] - s_off[-1, 0]) / \
            max(abs(s_off[-1, 0]), 1e-30)

        print("%-14s %4g | %8.2f%% %8.2f%% | %11.4g %11.4g | %+7.2f%%"
              % (folder, S, s33_on, s33_off,
                 f_top - want_t, f_bot - want_b, s11_shift))
print()
print("face columns = recovered sigma33 MINUS the applied traction at that")
print("face (0 = machine-exact V2L); s11 top = ON-vs-OFF in-plane shift.")
