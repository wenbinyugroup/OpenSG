"""Unit test: OpenSG msg-shell beam properties vs the VABS .K reference (IEA-22).

The VABS 51-station set (OpenSG-TW dehom51/out/VABS_iea51, 2-D solid .sg per station,
eta = i/50) is the reference: three station .K files are vendored under
examples/OpenSG_shell/windio/vabs_K/.  The test runs the standalone windIO pipeline
end-to-end for each station -- windIO yaml -> 1-D shell SG yaml (center reference,
reference-axis origin) -> RM ring homogenization + ring mass integration -- and gates
the Timoshenko diagonals, the mass per span, the mass center, and the mass-matrix
inertia block against VABS.

Tolerances come from MEASURED agreement, not aspiration.  The worst
relative difference over the three stations is EA 1.45 %, GA2 3.33 %,
GA3 2.05 %, GJ 2.23 %, EI2 0.14 %, EI3 2.70 % (mass: mu 1.46 %, i11
1.22 %, i22 0.63 %, i33 1.45 %); each gate below is ~1.5x its measured
worst, so a doubling of any modeling difference fails.  These are
shell-vs-2D-solid MODELING tolerances -- the reference is VABS's 2-D
solid section, a different model, so they cannot be tightened to
numerical parity the way the msg-solid .K gate in
tests/msg_k_reference/test_swiftcomp_k_solid_props.py is.

Run:  pytest tests/msg_shell_windio -q      (env opensg_2_0; ~1 min)
"""
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
WDIR = os.path.join(ROOT, "examples", "OpenSG_shell", "windio")
WINDIO = os.path.join(WDIR, "IEA-22-280-RWT.yaml")

LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]

# measured worst over the three stations -> gate at ~1.5x (see the docstring)
TOL = {"EA": 0.025, "GA2": 0.050, "GA3": 0.035,
       "GJ": 0.035, "EI2": 0.005, "EI3": 0.040}
TOL_MASS = 0.025                     # mu / i11 / i22 / i33, ~1.7x measured

# (station r, VABS reference .K, per-diagonal rel-tolerance [EA GA2 GA3 GJ EI2 EI3])
CASES = [
    (0.2, "iea_s10.sg.K", [TOL[k] for k in LBL]),
    (0.5, "iea_s25.sg.K", [TOL[k] for k in LBL]),
    (0.7, "iea_s35.sg.K", [TOL[k] for k in LBL]),
]


@pytest.fixture(scope="module")
def blade():
    from opensg_shell.windio import load_blade
    return load_blade(WINDIO)


@pytest.mark.parametrize("r,kref,tol", CASES, ids=["r0.2", "r0.5", "r0.7"])
def test_beam_props_vs_vabs(blade, tmp_path, r, kref, tol):
    from opensg_shell.windio import (build_cross_section, emit_shell_yaml,
                                     beam_props, read_k_file)
    from opensg_solid.helper.k_file import compare_to_K

    cs = build_cross_section(blade, r, mesh_size=0.01)
    y = str(tmp_path / ("st_r%04d_shell.yaml" % round(r * 1000)))
    emit_shell_yaml(cs, y, reference="center")
    P = beam_props(y, out_k=str(tmp_path / "st.K"))
    kpath = os.path.join(WDIR, "vabs_K", kref)
    Kv, Mv = read_k_file(kpath)
    C6, M6 = P["Timo"], P["Mass"]

    # Timoshenko diagonals vs the VABS 2-D solid, through the shared
    # verifier: the VABS beam convention (1 extension, 2,3 shear, 4 twist,
    # 5,6 bending) and the SI unit are STATED, and a failure prints the
    # whole term table rather than the first bad entry
    rep = compare_to_K(C6, kpath, "vabs_beam", ref_unit="SI",
                       value_unit="SI", rtol=dict(zip(LBL, tol)),
                       label="msg-shell RM ring r=%.1f" % r)
    assert rep["passed"], "\n" + rep["table"]
    for i in range(6):
        rel = abs(C6[i, i] / Kv[i, i] - 1.0)
        assert rel < tol[i], "%s at r=%.1f: RM %.5g vs VABS %.5g (%.2f%% > %.0f%%)" % (
            LBL[i], r, C6[i, i], Kv[i, i], 100 * rel, 100 * tol[i])

    # NEGATIVE CONTROL: the gate must reject a real regression -- a 25 %
    # transverse-shear error and a bending-index (5 <-> 6) swap
    Kbad = np.array(C6, float)
    Kbad[1, 1] *= 1.25
    Kbad[2, 2] *= 1.25
    assert not compare_to_K(Kbad, kpath, "vabs_beam", ref_unit="SI",
                            rtol=dict(zip(LBL, tol)))["passed"]
    Q = [0, 1, 2, 3, 5, 4]
    assert not compare_to_K(np.asarray(C6)[np.ix_(Q, Q)], kpath, "vabs_beam",
                            ref_unit="SI",
                            rtol=dict(zip(LBL, tol)))["passed"]

    # mass per unit span, mass center, inertia block
    assert abs(M6[0, 0] / Mv[0, 0] - 1.0) < TOL_MASS, \
        "mass/span at r=%.1f: %.5g vs VABS %.5g" % (r, M6[0, 0], Mv[0, 0])
    x2m, x3m = M6[2, 3] / M6[0, 0], M6[0, 4] / M6[0, 0]
    x2v, x3v = Mv[2, 3] / Mv[0, 0], Mv[0, 4] / Mv[0, 0]
    chord = cs["chord"]
    assert abs(x2m - x2v) < 0.02 * chord and abs(x3m - x3v) < 0.02 * chord, \
        "mass center at r=%.1f: (%.4f, %.4f) vs VABS (%.4f, %.4f)" % (r, x2m, x3m, x2v, x3v)
    for (i, j, lab) in [(3, 3, "i11"), (4, 4, "i22"), (5, 5, "i33")]:
        rel = abs(M6[i, j] / Mv[i, j] - 1.0)
        assert rel < TOL_MASS, "mass %s at r=%.1f: %.5g vs VABS %.5g (%.2f%%)" % (
            lab, r, M6[i, j], Mv[i, j], 100 * rel)

    # the emitted .K must round-trip through the reader
    Kr, Mr = read_k_file(P["k_file"])
    assert np.allclose(Kr, C6, rtol=1e-8) and np.allclose(Mr, M6, rtol=1e-8)


# ---------------------------------------------------------------------------------
# centers / principal-property FORMULAS vs the values VABS itself prints in the .K
# ---------------------------------------------------------------------------------
def _vabs_k_blocks(path):
    """Parse the printed center/principal blocks of a native VABS .K."""
    lines = open(path).read().splitlines()
    out = {}

    def floats_after(tag, n=1):
        for i, ln in enumerate(lines):
            if tag in ln:
                vals = []
                for j in range(i + 1, len(lines)):
                    for tok in lines[j].split():
                        try:
                            vals.append(float(tok))
                        except ValueError:
                            pass                      # mixed text line ("8.73 degrees ...")
                    if len(vals) >= n:
                        return vals[:n]
        raise KeyError(tag)

    def value_on(tag):
        for ln in lines:
            if tag in ln:
                return float(ln.split()[-1])
        raise KeyError(tag)

    out["tc"] = floats_after("The Tension Center", 2)
    out["sc"] = floats_after("The Shear Center", 2)
    out["ei22"] = value_on("Principal bending stiffness EI22")
    out["ei33"] = value_on("Principal bending stiffness EI33")
    out["ei_ang"] = floats_after("principal bending axes rotated", 1)[0]
    out["ga22"] = value_on("Principal shear stiffness GA22")
    out["ga33"] = value_on("Principal shear stiffness GA33")
    out["ga_ang"] = floats_after("principal shear axes rotated", 1)[0]
    out["i22"] = value_on("Principal mass moments of inertia i22")
    out["i33"] = value_on("Principal mass moments of inertia i33")
    out["m_ang"] = floats_after("principal inertial axes rotated", 1)[0]
    return out


def _dang(a, b):
    return min(abs(a - b) % 180.0, 180.0 - abs(a - b) % 180.0)


@pytest.mark.parametrize("kref", ["iea_s10.sg.K", "iea_s25.sg.K", "iea_s35.sg.K"])
def test_vabs_k_center_formulas(kref):
    """write_vabs_k's derived blocks must reproduce what VABS prints, given VABS's
    own Timoshenko and mass matrices."""
    from opensg_shell.windio import read_k_file
    from opensg_shell.windio.sg_props import _principal_2x2

    path = os.path.join(WDIR, "vabs_K", kref)
    K, M = read_k_file(path)
    ref = _vabs_k_blocks(path)

    # tension center
    x2t, x3t = -K[0, 5] / K[0, 0], K[0, 4] / K[0, 0]
    assert np.allclose([x2t, x3t], ref["tc"], rtol=1e-6, atol=1e-10)
    # shear center (from the Timoshenko compliance)
    S = np.linalg.inv(K)
    x2s, x3s = -S[2, 3] / S[3, 3], S[1, 3] / S[3, 3]
    assert np.allclose([x2s, x3s], ref["sc"], rtol=1e-5, atol=1e-8)
    # principal bending at the tension center
    EI22c = K[4, 4] - K[0, 4] ** 2 / K[0, 0]
    EI33c = K[5, 5] - K[0, 5] ** 2 / K[0, 0]
    EI23c = K[4, 5] - K[0, 4] * K[0, 5] / K[0, 0]
    ei22p, ei33p, eang = _principal_2x2(EI22c, EI33c, -EI23c)
    assert abs(ei22p / ref["ei22"] - 1) < 1e-6 and abs(ei33p / ref["ei33"] - 1) < 1e-6
    assert _dang(eang, ref["ei_ang"]) < 1e-3
    # principal shear
    ga22p, ga33p, gang = _principal_2x2(K[1, 1], K[2, 2], -K[1, 2])
    assert abs(ga22p / ref["ga22"] - 1) < 1e-6 and abs(ga33p / ref["ga33"] - 1) < 1e-6
    assert _dang(gang, ref["ga_ang"]) < 1e-3
    # principal mass inertia at the mass center
    m = M[0, 0]
    S2, S3 = -M[0, 5], M[0, 4]
    I22, I33, I23 = M[4, 4], M[5, 5], -M[4, 5]
    x2m, x3m = S2 / m, S3 / m
    i22p, i33p, mang = _principal_2x2(I22 - m * x3m ** 2, I33 - m * x2m ** 2,
                                      I23 - m * x2m * x3m)
    assert abs(i22p / ref["i22"] - 1) < 1e-6 and abs(i33p / ref["i33"] - 1) < 1e-6
    assert _dang(mang, ref["m_ang"]) < 1e-3
