"""blade.timo() vs TRUE VABS -- the IEA-22 benchmark (example AND unit test).

Is the Timoshenko 6x6 the pynumad Blade route computes actually
benchmarked against VABS?  This file answers it directly: the IEA-22
blade runs through `opensg.read(...)` -> `blade.timo(r)` (the in-memory
pynumad homogenizer, no yaml) at the three stations whose TRUE VABS 2-D
solid .K files are vendored under examples/OpenSG_shell/windio/vabs_K/
(the 51-station VABS set, eta = i/50: s10/s25/s35 = r 0.2/0.5/0.7), and
both the beam STIFFNESS (Timoshenko diagonals, VABS order [EA GA2 GA3 GJ
EI2 EI3]) and the MASS matrix (mass per span, mass center, inertia
diagonals) are gated.

The blade route computes at the OML reference surface -- its ONLY
reference, by design: the airfoil contour the blade yaml describes IS the
outer mold line -- while VABS models the true 2-D solid section, so
the tolerance bands below absorb BOTH the shell-vs-solid modeling
difference and the OML-vs-mid-surface geometry difference.  Each gate is
the MEASURED worst deviation of this exact route over the three stations
(2026-08-14: EA 1.52 %, GA2 2.54 %, GA3 14.33 %, GJ 2.99 %, EI2 1.37 %,
EI3 2.45 %; mass mu 2.26 %, i11 2.53 %, i22 2.31 %, i33 2.57 %; mass
centers <= 1.0 % chord), padded ~1.5x -- a doubling of any modeling
difference fails.  GA3 (chordwise transverse shear) carries the wide
band: it is the term most sensitive to the one-sided OML laminate offset
on the thin aft panels and grows outboard (6.2 -> 12.4 -> 14.3 %).  The
bands cannot be tightened to numerical parity: VABS solves a DIFFERENT
model.  (The tighter center-reference shell-vs-VABS study lives in
tests/msg_shell_windio/test_vabs_k_beam_props.py through the SG-engine
yaml route, which keeps the reference parameter.)

Run as a script (prints the full comparison table):
    python test_timo_pynumad.py
Run as a unit test:
    pytest test_timo_pynumad.py -q          (env opensg_2_0; ~1 min)
"""
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
WDIR = os.path.join(ROOT, "examples", "OpenSG_shell", "windio")
BLADE_YAML = os.path.join(WDIR, "IEA-22-280-RWT.yaml")

LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
TOL = {"EA": 0.025, "GA2": 0.040, "GA3": 0.22,
       "GJ": 0.045, "EI2": 0.020, "EI3": 0.040}
TOL_MASS = 0.040                     # mu / i11 / i22 / i33
TOL_CENTER = 0.02                    # mass center, fraction of chord

# (span r, TRUE VABS 2-D solid reference .K)
CASES = [(0.2, "iea_s10.sg.K"), (0.5, "iea_s25.sg.K"), (0.7, "iea_s35.sg.K")]

_cache = {}


def blade_timo_at(r):
    """blade.timo(r) on the IEA-22 Blade (cached; OML reference, always)."""
    if "blade" not in _cache:
        import opensg
        _cache["blade"] = opensg.read(BLADE_YAML)
    if r not in _cache:
        _cache[r] = _cache["blade"].timo(r)
    return _cache[r]


def compare_station(r, kref):
    """One station: blade.timo(r) vs the TRUE VABS .K.

    In:  r float -- span station; kref str -- vabs_K file name.
    Out: (rows, ok) -- rows [(label, opensg, vabs, rel, tol)], ok bool."""
    from opensg_shell.pynumad import read_k_file

    P = blade_timo_at(r)
    K, M = np.asarray(P["Timo"]), np.asarray(P["Mass"])
    Kv, Mv = read_k_file(os.path.join(WDIR, "vabs_K", kref))
    rows = []
    for i, lab in enumerate(LBL):
        rows.append((lab, K[i, i], Kv[i, i],
                     abs(K[i, i] / Kv[i, i] - 1.0), TOL[lab]))
    rows.append(("mu", M[0, 0], Mv[0, 0],
                 abs(M[0, 0] / Mv[0, 0] - 1.0), TOL_MASS))
    for (i, lab) in [(3, "i11"), (4, "i22"), (5, "i33")]:
        rows.append((lab, M[i, i], Mv[i, i],
                     abs(M[i, i] / Mv[i, i] - 1.0), TOL_MASS))
    # mass center, gated as a fraction of the local chord
    chord = float(P["chord"])
    x2, x3 = M[2, 3] / M[0, 0], M[0, 4] / M[0, 0]
    x2v, x3v = Mv[2, 3] / Mv[0, 0], Mv[0, 4] / Mv[0, 0]
    rows.append(("x2_mc/c", x2 / chord, x2v / chord,
                 abs(x2 - x2v) / chord, TOL_CENTER))
    rows.append(("x3_mc/c", x3 / chord, x3v / chord,
                 abs(x3 - x3v) / chord, TOL_CENTER))
    return rows, all(rel < tol for (_, _, _, rel, tol) in rows)


@pytest.mark.parametrize("r,kref", CASES, ids=["r0.2", "r0.5", "r0.7"])
def test_blade_timo_vs_vabs(r, kref):
    rows, ok = compare_station(r, kref)
    lines = ["%-8s  OpenSG %.6g  VABS %.6g  (%.2f%% > %.1f%%)"
             % (lab, a, b, 100 * rel, 100 * tol)
             for (lab, a, b, rel, tol) in rows if rel >= tol]
    assert ok, "blade.timo(%.1f) vs %s:\n%s" % (r, kref, "\n".join(lines))
    # the stiffness must stay symmetric on this route
    K = np.asarray(blade_timo_at(r)["Timo"])
    assert np.allclose(K, K.T)


def test_gate_rejects_regression():
    """NEGATIVE CONTROL: a 25 % transverse-shear error and a bending-index
    (5 <-> 6) swap must both fail the same gates."""
    rows, ok = compare_station(0.2, "iea_s10.sg.K")
    assert ok
    P = blade_timo_at(0.2)
    K = np.array(P["Timo"], float)
    bad = abs(1.25 * K[1, 1] / rows[1][2] - 1.0)      # GA2 row: (lab, a, b, ...)
    assert bad > TOL["GA2"]
    swap = abs(K[5, 5] / rows[4][2] - 1.0)            # EI3 against the EI2 ref
    assert swap > TOL["EI2"]


if __name__ == "__main__":
    import time

    print("start:", time.strftime("%Y-%m-%d %H:%M:%S"))
    print("blade.timo() [pynumad in-memory, OML reference] vs TRUE VABS"
          " 2-D solid .K")
    print("blade: %s" % os.path.basename(BLADE_YAML))
    n_bad = 0
    for r, kref in CASES:
        t0 = time.perf_counter()
        rows, ok = compare_station(r, kref)
        dt = time.perf_counter() - t0
        print("\nstation r = %.1f  vs  vabs_K/%s   [%.1f s]" % (r, kref, dt))
        print("  %-8s %15s %15s %9s %7s  %s"
              % ("term", "OpenSG", "VABS", "diff", "tol", "verdict"))
        for (lab, a, b, rel, tol) in rows:
            verdict = "ok" if rel < tol else "FAIL"
            n_bad += rel >= tol
            print("  %-8s %15.6g %15.6g %8.2f%% %6.1f%%  %s"
                  % (lab, a, b, 100 * rel, 100 * tol, verdict))
    print("\n%s" % ("ALL STATIONS WITHIN THE MEASURED VABS BANDS"
                    if n_bad == 0 else "%d TERM(S) OUT OF BAND" % n_bad))
    print("end:", time.strftime("%Y-%m-%d %H:%M:%S"))
    raise SystemExit(1 if n_bad else 0)
