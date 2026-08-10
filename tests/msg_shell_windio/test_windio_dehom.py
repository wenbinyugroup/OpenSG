"""Unit test: windIO-based two-step dehomogenization (msg-shell) at IEA-22 r = 0.2.

Self-contained (no BeamDyn run needed): the station is built from the committed windIO
yaml, homogenized (RM ring), and dehomogenized under known beam resultants.

Gates:
  1. STRESS CLOSURE -- the recovered sigma11 cloud (wall/plate frame) must integrate
     back to the applied beam resultants: F1 = int s11 dA, M2 = int s11 x3 dA,
     M3 = -int s11 x2 dA (point weight = element arc length x thickness / n_depth).
  2. VABS cross-check -- under the production FF of the 51-station BeamDyn run
     (ff51_rmc_reform.dat row 10), the p99 |sigma11| of the material-frame cloud must
     match the committed VABS 2-D-solid .SM within 20 %.
  3. .ff round-trip + Wiener-Milenkovic <-> DCM round-trip + .SM/.EM/.U files parse.

Run:  pytest tests/msg_shell_windio -q      (env opensg_2_0)
"""
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
WDIR = os.path.join(ROOT, "examples", "OpenSG_shell", "windio")
WINDIO = os.path.join(WDIR, "IEA-22-280-RWT.yaml")
FF51 = os.path.join(ROOT, "examples", "OpenSG_shell",
                    "2_get_beam_dehom_from_shell_cross_section", "ff51_rmc_reform.dat")
VABS_SM = os.path.join(ROOT, "examples", "OpenSG_shell", "IEA_blade_beam", "iea_s10.sg.SM")


@pytest.fixture(scope="module")
def station(tmp_path_factory):
    from opensg_shell.windio import (load_blade, build_cross_section, emit_shell_yaml,
                                     beam_props)
    tmp = tmp_path_factory.mktemp("windio_dehom")
    blade = load_blade(WINDIO)
    cs = build_cross_section(blade, 0.2, mesh_size=0.01)
    y = str(tmp / "st_r0200_shell.yaml")
    emit_shell_yaml(cs, y, reference="center")
    P = beam_props(y, out_k=str(tmp / "st.K"))
    return dict(yaml=y, bundle=P["bundle"], tmp=tmp)


def _cloud_weights(D):
    """Per-point area weight of the recovery cloud: L_elem * h_elem / n_depth."""
    B = D["bundle"]
    corners = np.asarray(B["corners"]); rc = np.asarray(B["red_cells"])
    Lel = np.linalg.norm(corners[rc[:, 1]] - corners[rc[:, 0]], axis=1)
    hth = {ln: float(sum(i["thick"])) for ln, i in B["layup_db"].items()}
    h_e = np.array([hth[ln] for ln in B["layup_per_elem"]])
    n_depth = int(len(D["elem"]) / len(rc))
    return (Lel * h_e / n_depth)[D["elem"]]


def test_stress_closure(station):
    from opensg_shell.windio import dehom_station
    FF = np.array([5.0e6, 0.0, 0.0, 0.0, -6.0e7, 2.0e7])   # F1, M2, M3
    D = dehom_station(station["yaml"], FF, n_depth=9, frame="plate",
                      bundle=station["bundle"])
    w = _cloud_weights(D)
    s11 = D["sig"][:, 0]                                    # plate frame: 11 = span
    x2, x3 = D["pts"][:, 0], D["pts"][:, 1]
    F1c = float(np.sum(s11 * w))
    M2c = float(np.sum(s11 * x3 * w))
    M3c = float(-np.sum(s11 * x2 * w))
    # F1 carries the systematic web/skin junction double-count of the shell idealization
    # (web chains span OML->OML, overlapping the skin thickness at the T-joints) on top
    # of the first-order membrane recovery -- gate at 10 %; the moments are tighter.
    assert abs(F1c / FF[0] - 1.0) < 0.10, "F1 closure: %.4g vs %.4g" % (F1c, FF[0])
    assert abs(M2c / FF[4] - 1.0) < 0.06, "M2 closure: %.4g vs %.4g" % (M2c, FF[4])
    assert abs(M3c / FF[5] - 1.0) < 0.08, "M3 closure: %.4g vs %.4g" % (M3c, FF[5])


def test_dehom_vs_vabs_sm(station):
    from opensg_shell.windio import dehom_to_files
    tab = np.loadtxt(FF51)
    row = tab[np.argmin(np.abs(tab[:, 0] - 0.2))]
    D = dehom_to_files(station["yaml"], row[1:7],
                       out_base=str(station["tmp"] / "st"), n_depth=9,
                       frame="material", bundle=station["bundle"])
    p99 = np.percentile(np.abs(D["sig"][:, 0]), 99)
    vs = np.loadtxt(VABS_SM, skiprows=2)
    p99v = np.percentile(np.abs(vs[:, 2]), 99)              # VABS col 2 = s11 (material)
    rel = abs(p99 / p99v - 1.0)
    assert rel < 0.20, "p99|s11|: RM %.4g vs VABS %.4g (%.1f%%)" % (p99, p99v, 100 * rel)
    # the .SM/.EM/.U triple parses back with the right shapes
    sm = np.loadtxt(str(station["tmp"] / "st.SM"), skiprows=2)
    em = np.loadtxt(str(station["tmp"] / "st.EM"), skiprows=2)
    uu = np.loadtxt(str(station["tmp"] / "st.U"))
    P = len(D["pts"])
    assert sm.shape == (P, 8) and em.shape == (P, 8) and uu.shape == (P, 6)


def test_ff_and_wm_roundtrip(station):
    from opensg_shell.windio import write_ff, read_ff, wm_to_dcm, dcm_to_wm
    rng = np.random.default_rng(7)
    c = 0.2 * rng.standard_normal(3)
    C = wm_to_dcm(c)
    assert np.allclose(dcm_to_wm(C), c, atol=1e-12), "WM <-> DCM round-trip"
    u = rng.standard_normal(3); F = 1e6 * rng.standard_normal(3)
    M = 1e7 * rng.standard_normal(3)
    p = str(station["tmp"] / "rt.ff")
    write_ff(p, 0.2, u, c, C, F, M)
    ff = read_ff(p)
    assert np.allclose(ff["u"], u) and np.allclose(ff["C"], C, atol=1e-14)
    assert np.allclose(ff["FF"], np.r_[F, M])
    assert abs(ff["eta"] - 0.2) < 1e-12
