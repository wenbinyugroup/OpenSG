"""Unit tests: the ONE shared `.K` reader and the compare_to_K verifier.

Fast (no homogenization).  These gate the READER against every `.K`
dialect vendored in the repo and the VERIFIER against the two ways it
could give a confidently wrong answer -- a unit slip and an index
permutation -- plus the loud-failure contract that keeps it from ever
guessing a convention.

Dialects covered:
    examples/OpenSG_shell/windio/vabs_K/iea_s{10,25,35}.sg.K   native VABS
    examples/OpenSG_shell/windio/props/iea22_r*.K              write_vabs_k
    examples/OpenSG-solid/6_get_solid_props_from_3D_SG/
        sample_{1,2}/Sample_{1,2}.sc.k                         SwiftComp
        sample_1/Sample_1.out                                  write_sc_K

Run:  pytest tests/msg_k_reference -q      (env opensg_2_0)
"""
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
WDIR = os.path.join(ROOT, "examples", "OpenSG_shell", "windio")
VABS_K = os.path.join(WDIR, "vabs_K")
PROPS = os.path.join(WDIR, "props")
S6 = os.path.join(ROOT, "examples", "OpenSG-solid",
                  "6_get_solid_props_from_3D_SG")

VABS_FILES = ["iea_s10.sg.K", "iea_s25.sg.K", "iea_s35.sg.K"]
SC_K = {"1": os.path.join(S6, "sample_1", "Sample_1.sc.k"),
        "2": os.path.join(S6, "sample_2", "Sample_2.sc.k")}


# ------------------------------------------------------------------- reader
@pytest.mark.parametrize("name", VABS_FILES)
def test_vabs_beam_block_structure(name):
    """A native VABS cross-section .K: every block, and the printed
    scalars must equal the entries of the matrices they summarize."""
    from opensg_solid.helper.k_file import read_k_blocks, read_beam_k

    path = os.path.join(VABS_K, name)
    b = read_k_blocks(path)
    m, s, v = b["matrices"], b["scalars"], b["vectors"]

    for key in ("mass", "mass_at_mass_center", "mass_at_shear_center",
                "timoshenko_stiffness", "timoshenko_compliance"):
        assert m[key].shape == (6, 6), key
    for key in ("classical_stiffness", "classical_compliance",
                "classical_stiffness_at_shear_center",
                "classical_compliance_at_shear_center"):
        assert m[key].shape == (4, 4), key
    for key in ("mass_center", "geometric_center", "tension_center",
                "shear_center", "mass_center_wrt_shear_center"):
        assert len(v[key]) == 2, key

    K, M = read_beam_k(path)
    assert np.allclose(K, m["timoshenko_stiffness"])
    assert np.allclose(M, m["mass"])
    # every 6x6 must be symmetric and the stiffness positive definite
    for name6 in ("timoshenko_stiffness", "mass"):
        A = m[name6]
        assert np.abs(A - A.T).max() <= 1e-9 * np.abs(A).max(), name6
    assert np.all(np.linalg.eigvalsh(K) > 0)
    # stiffness and compliance are inverses (to the file's print precision)
    assert np.abs(K @ m["timoshenko_compliance"] - np.eye(6)).max() < 1e-6

    # the inline scalars restate matrix entries -- the parser must agree
    assert abs(s["ea"] / K[0, 0] - 1) < 1e-9
    assert abs(s["gj"] / m["classical_stiffness"][1, 1] - 1) < 1e-9
    assert abs(s["mpus"] / M[0, 0] - 1) < 1e-9
    # a VABS .K restates i11 / i22 / i33 / the angles at the shear centre;
    # first-occurrence-wins must keep the USER-ORIGIN (mass-centre) set
    assert abs(s["i11"] / m["mass_at_mass_center"][3, 3] - 1) < 1e-9
    assert abs(s["i11"] / m["mass_at_shear_center"][3, 3] - 1) > 1e-3
    assert s["area"] > 0 and s["rgyr"] > 0
    for key in ("ei22", "ei33", "ga22", "ga33", "i22p", "i33p"):
        assert s[key] > 0, key
    for key in ("ei_angle_deg", "ga_angle_deg", "m_angle_deg"):
        assert -360.0 < s[key] < 360.0, key
    # the tension center is -K16/K11, K15/K11 -- a parse-order check
    assert np.allclose(v["tension_center"],
                       [-K[0, 5] / K[0, 0], K[0, 4] / K[0, 0]],
                       rtol=1e-6, atol=1e-10)


def test_write_vabs_k_station_files_read_back():
    """The 16 OpenSG-written station .K files (write_vabs_k layout, no
    classical 4x4 block) go through the same reader."""
    from opensg_solid.helper.k_file import read_beam_k, read_k_blocks

    files = sorted(f for f in os.listdir(PROPS) if f.endswith(".K"))
    assert len(files) >= 10, files
    for f in files:
        p = os.path.join(PROPS, f)
        K, M = read_beam_k(p)
        assert K.shape == (6, 6) and M.shape == (6, 6), f
        assert np.all(np.diag(K) > 0) and M[0, 0] > 0, f
        b = read_k_blocks(p)
        assert "classical_stiffness" not in b["matrices"], f
        assert abs(b["scalars"]["ea"] / K[0, 0] - 1) < 1e-9, f


@pytest.mark.parametrize("s", ["1", "2"])
def test_swiftcomp_solid_block_structure(s):
    """A SwiftComp 3-D solid .sc.k: 6x6 stiffness + compliance + the nine
    orthotropic-approximated engineering constants + effective density."""
    from opensg_solid.helper.k_file import read_solid_k

    d = read_solid_k(SC_K[s])
    C, S = d["C"], d["S"]
    assert C.shape == (6, 6) and S.shape == (6, 6)
    assert np.abs(C - C.T).max() <= 1e-6 * np.abs(C).max()
    assert np.all(np.linalg.eigvalsh(C) > 0)
    assert np.abs(C @ S - np.eye(6)).max() < 1e-6
    c = d["constants"]
    assert set(c) == {"E1", "E2", "E3", "G12", "G13", "G23",
                      "nu12", "nu13", "nu23"}
    # the constants are the compliance read out -- SwiftComp's own definition
    assert abs(c["E1"] * S[0, 0] - 1) < 1e-6
    assert abs(c["G23"] * S[3, 3] - 1) < 1e-6
    assert abs(c["G12"] * S[5, 5] - 1) < 1e-6
    assert abs(c["nu12"] + S[0, 1] / S[0, 0]) < 1e-6
    assert d["density"] is not None


def test_opensg_out_reads_with_the_macro_law_infix():
    """OpenSG's write_sc_K splices the macro-law name into the title
    ('The Effective Cauchy Continuum Stiffness Matrix'); the reader must
    strip it and land on the same key as a bare SwiftComp .sc.k."""
    from opensg_solid.helper.k_file import read_solid_k, read_k_blocks

    p = os.path.join(S6, "sample_1", "Sample_1.out")
    b = read_k_blocks(p)
    assert set(b["matrices"]) == {"effective_stiffness",
                                  "effective_compliance"}
    assert any("Cauchy Continuum" in t for t in b["titles"])
    d = read_solid_k(p)
    assert d["C"].shape == (6, 6) and len(d["constants"]) == 9


def test_write_sc_K_round_trip(tmp_path):
    """write_sc_K -> read_solid_k is lossless to the format's 8 printed
    significant digits, so a driver's .out is a usable reference."""
    from opensg_solid.sg_homo import write_sc_K
    from opensg_solid.helper.k_file import read_solid_k

    rng = np.random.default_rng(0)
    A = rng.normal(size=(6, 6))
    C = A @ A.T + 6.0 * np.eye(6)
    p = str(tmp_path / "rt.out")
    write_sc_K(p, C, solve_time=1.25, model="unit test", name="Cauchy "
               "Continuum")
    d = read_solid_k(p)
    assert np.abs(d["C"] / C - 1.0).max() < 1e-7


def test_legacy_read_k_file_import_path_unchanged():
    """Gate (e): every existing caller imports read_k_file from
    opensg_shell.pynumad -- that path must keep working and must return
    exactly what the promoted reader returns."""
    from opensg_shell.pynumad import read_k_file
    from opensg_shell.pynumad.sg_beamdyn import read_k_file as rk2
    from opensg_solid.helper.k_file import read_beam_k

    p = os.path.join(VABS_K, "iea_s10.sg.K")
    K, M = read_k_file(p)
    Kb, Mb = read_beam_k(p)
    assert np.array_equal(K, Kb) and np.array_equal(M, Mb)
    assert rk2 is read_k_file
    assert K.shape == (6, 6) and M.shape == (6, 6)
    with pytest.raises(ValueError):
        read_k_file(SC_K["1"])           # a solid .sc.k has no beam blocks


# ----------------------------------------------------------------- verifier
def test_verifier_identity_and_report_shape():
    from opensg_solid.helper.k_file import compare_to_K, read_beam_k

    p = os.path.join(VABS_K, "iea_s10.sg.K")
    K, _ = read_beam_k(p)
    rep = compare_to_K(K, p, "vabs_beam", ref_unit="SI", value_unit="SI",
                       rtol=1e-14, label="self")
    assert rep["passed"] and rep["n_fail"] == 0 and rep["n_terms"] == 6
    assert rep["worst"]["rel_diff"] == 0.0
    for x in rep["terms"]:
        assert set(x) == {"term", "i", "j", "opensg", "reference",
                          "abs_diff", "rel_diff", "rtol", "pass"}
    assert "EA" in rep["table"] and "worst term" in rep["table"]
    # a path is accepted in place of a matrix
    rep2 = compare_to_K(p, p, "vabs_beam", ref_unit="SI", rtol=1e-14)
    assert rep2["passed"]


def test_verifier_unit_convention_is_load_bearing():
    """A SwiftComp .sc.k is MPa while opensg_solid writes Pa: state it and
    the comparison passes, get it wrong and it fails by 1e6."""
    from opensg_solid.helper.k_file import compare_to_K, read_solid_k

    C_mpa = read_solid_k(SC_K["1"])["C"]
    C_pa = C_mpa * 1.0e6
    ok = compare_to_K(C_pa, SC_K["1"], "swiftcomp_solid", ref_unit="MPa",
                      value_unit="Pa", rtol=1e-12)
    assert ok["passed"] and abs(ok["scale"] - 1e-6) < 1e-18
    bad = compare_to_K(C_pa, SC_K["1"], "swiftcomp_solid", ref_unit="MPa",
                       value_unit="MPa", rtol=1e-2)
    assert not bad["passed"] and bad["n_fail"] == 9
    assert bad["worst"]["rel_diff"] > 1e5


def test_verifier_catches_an_index_permutation():
    """Swapping the two bending indices is a silent-wrong-answer mode: the
    default gate must reject it, and declaring the permutation must undo
    it exactly."""
    from opensg_solid.helper.k_file import compare_to_K, read_beam_k

    p = os.path.join(VABS_K, "iea_s10.sg.K")
    K, _ = read_beam_k(p)
    P = [0, 1, 2, 3, 5, 4]
    K_swapped = K[np.ix_(P, P)]
    bad = compare_to_K(K_swapped, p, "vabs_beam", ref_unit="SI", rtol=1e-3)
    assert not bad["passed"]
    assert {x["term"] for x in bad["terms"] if not x["pass"]} == {"EI2",
                                                                  "EI3"}
    good = compare_to_K(K_swapped, p, "vabs_beam", ref_unit="SI",
                        permutation=P, rtol=1e-14)
    assert good["passed"] and good["permutation"] == (0, 1, 2, 3, 5, 4)


def test_verifier_catches_a_relative_density_rescale():
    """The omega regression in miniature: C scaled by the relative density
    must fail every term of the solid gate."""
    from opensg_solid.helper.k_file import compare_to_K, read_solid_k

    C = read_solid_k(SC_K["1"])["C"]
    rep = compare_to_K(C * 0.3, SC_K["1"], "swiftcomp_solid",
                       ref_unit="MPa", value_unit="MPa", rtol=1e-6)
    assert not rep["passed"] and rep["n_fail"] == 9
    assert abs(rep["worst"]["rel_diff"] - 0.7) < 1e-6


def test_verifier_refuses_what_it_cannot_resolve():
    """Every unresolvable request raises instead of comparing."""
    from opensg_solid.helper.k_file import compare_to_K, read_beam_k

    p = os.path.join(VABS_K, "iea_s10.sg.K")
    K, _ = read_beam_k(p)
    with pytest.raises(ValueError, match="unknown convention"):
        compare_to_K(K, p, "beam", ref_unit="SI")
    with pytest.raises(ValueError, match="ref_unit is required"):
        compare_to_K(K, p, "vabs_beam")
    with pytest.raises(ValueError, match="unknown ref_unit"):
        compare_to_K(K, p, "vabs_beam", ref_unit="furlong")
    with pytest.raises(ValueError, match="expects a 6x6 law"):
        compare_to_K(K[:4, :4], p, "vabs_beam", ref_unit="SI")
    with pytest.raises(ValueError, match="permutation"):
        compare_to_K(K, p, "vabs_beam", ref_unit="SI",
                     permutation=[0, 0, 2, 3, 4, 5])
    with pytest.raises(ValueError, match="unknown term"):
        compare_to_K(K, p, "vabs_beam", ref_unit="SI", terms=["EA", "XX"])
    with pytest.raises(ValueError, match="no 'timoshenko_stiffness' block"
                                         "|has no"):
        compare_to_K(K, SC_K["1"], "vabs_beam", ref_unit="SI")
    with pytest.raises(ValueError):
        compare_to_K(K, p, "vabs_beam", ref_unit="SI", terms="sideways")


def test_verifier_term_selectors():
    from opensg_solid.helper.k_file import compare_to_K, read_beam_k

    p = os.path.join(VABS_K, "iea_s10.sg.K")
    K, _ = read_beam_k(p)
    kw = dict(convention="vabs_beam", ref_unit="SI", rtol=1e-14)
    assert compare_to_K(K, p, terms="full", **kw)["n_terms"] == 21
    assert compare_to_K(K, p, terms="diagonal", **kw)["n_terms"] == 6
    assert compare_to_K(K, p, terms=["EA", "GJ"], **kw)["n_terms"] == 2
    r = compare_to_K(K, p, terms=[("K15", 0, 4)], **kw)
    assert r["terms"][0]["reference"] == K[0, 4]
