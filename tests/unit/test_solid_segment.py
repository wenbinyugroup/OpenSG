"""The solid segment route (opensg_solid/sg_segment.py): a 3-D solid beam
segment with aperiodic ends gives its Timoshenko 6x6 and its mass matrix.

(a) prismatic identity: an end section equals OpenSG's own 2-D beam route
    (n_model 1) on its end-face mesh, and a prismatic segment equals its
    end section, in hex8, tet4, mixed hex8 + tet4 and tet10.
(b) parity with the JAX port this route comes from (OpenSG-TW-claude,
    opensg_jax/fe_jax/solid_taper.py), whose numbers on the same yamls are
    stored in fixtures/solid_segment/tw_claude_ref.json (hex8, tet4 and
    mixed: the port has no tet10), with the port's nodal end gauge.  The
    default area gauge against FEniCS OpenSG v1.2 on two tapered tubes
    (fixtures/solid_segment/fenics_v12_ref.json).
(c) tapered segments: the stored numbers, the mass matrix against its
    closed form, and the law between the two end laws.
(d) frames: rotated rows equal the block-angle twin.
(e) the command line (<base>.out, <base>_ends.out, the route line) and the
    refusals.
The segments are small generated boxes (fixtures/solid_segment/
make_segments.py writes the yamls of that folder)."""
import importlib.util
import json
import os
import re
import shutil
import struct

import numpy as np
import pytest

assert struct.calcsize("P") == 8

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures",
                   "solid_segment")
_spec = importlib.util.spec_from_file_location(
    "make_segments", os.path.join(FIX, "make_segments.py"))
MS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(MS)

PRISMATIC = ["iso_rect_hex8", "comp_rect_hex8", "comp_rect_tet4",
             "comp_rect_mixed", "comp_rect_tet10", "iso_rect_tet10"]
TAPERED = ["comp_taper_hex8", "comp_taper_tet4", "comp_taper_mixed",
           "comp_taper_tet10"]
# the port has hex8 and tet4 cells only
PARITY = [n for n in PRISMATIC + TAPERED if not n.endswith("tet10")]
# measured on one CPU: 1.7e-15 (an end section against the 2-D route),
# 7.2e-15 (a prismatic hex8, tet4 or mixed segment against its end
# section), 3.8e-15 (parity with the port).  The gates leave room for
# other BLAS builds.
GATE_2D = 1e-12
GATE_ID = 1e-12
GATE_PARITY = 1e-10
# A prismatic tet10 segment differs from its end section by the
# discretization: the 3-D quadratic space holds warping fields that the
# end section's tri6 space does not.  Measured on the composite: 6.3e-4
# on the stored 2 x 4 x 2 mesh, 1.2e-4 at 4 x 8 x 2, 2.0e-4 at 2 x 4 x 4.
# The isotropic EA, EI and their couplings are exact (1.1e-15).
GATE_TET10 = 1e-3


def _gate_id(name):
    return GATE_TET10 if name.endswith("tet10") else GATE_ID

_RUNS = {}


def _rel(A, B):
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    return float(np.max(np.abs(A - B)) / np.max(np.abs(B)))


def _run(path, workdir, **kw):
    from opensg_solid.sg_homo import plate_homo_2d
    return plate_homo_2d(str(path), plot=False, workdir=str(workdir), **kw)


def _fixture_run(name, tmp_path_factory):
    if name not in _RUNS:
        wd = tmp_path_factory.mktemp(name)
        _RUNS[name] = _run(os.path.join(FIX, name + ".yaml"), wd)
    return _RUNS[name]


def _face_yaml(folder, name, side=0, header=None):
    """The end-face mesh of a fixture segment as a 2-D beam SG yaml: nodes
    [x2 x3 0], each face with its parent's material and frame row."""
    from opensg_solid.sg_mesh import load_sg_input
    from opensg_solid.sg_segment import end_sections, segment_mesh
    builder, kw = MS.CASES[name]
    nodes, elems, names, frames, mats = builder(**kw)
    sc = load_sg_input(os.path.join(FIX, name + ".yaml"))
    E = len(sc["cells"])
    seg = segment_mesh(sc, np.tile(np.eye(6), (E, 1, 1)), np.zeros(E))
    end = end_sections(seg)[side]
    fn = np.column_stack([end["nodes"][:, 1], end["nodes"][:, 2],
                          np.zeros(len(end["nodes"]))])
    fe, fnames, fframes = [], [], []
    for conn, _C, _rho, ix in end["batches"].values():
        fe += [list(c) for c in conn]
        fnames += [names[k] for k in ix]
        fframes += [frames[k] for k in ix]
    hdr = header or {"msg": "solid", "n_model": 1, "refined": 1}
    return MS.write_yaml(os.path.join(str(folder), name + "_face.yaml"), fn,
                         fe, fnames, fframes, mats, header=hdr)


# ----------------------------------------------------- (a) prismatic identity
@pytest.mark.parametrize("name", ["iso_rect_hex8", "comp_rect_hex8",
                                  "comp_rect_tet4", "comp_rect_tet10"])
def test_end_section_equals_the_2d_beam_route(name, tmp_path,
                                              tmp_path_factory):
    r = _fixture_run(name, tmp_path_factory)
    r2 = _run(_face_yaml(tmp_path, name), tmp_path)
    assert r2["n_model"] == 1 and r2["n_sg"] == 2
    for end in r["ends"]:
        assert _rel(end["C_eff"], r2["C_eff"]) < GATE_2D
        assert _rel(end["Mass"], r2["Mass"]) < 1e-12
    assert _rel(r["C_eff"], r2["C_eff"]) < _gate_id(name)
    assert _rel(r["C_eff_EB"], r2["C_eff_EB"]) < _gate_id(name)
    assert _rel(r["Mass"], r2["Mass"]) < 1e-12


@pytest.mark.parametrize("name", PRISMATIC)
def test_prismatic_segment_equals_its_end_section(name, tmp_path_factory):
    r = _fixture_run(name, tmp_path_factory)
    left, right = r["ends"]
    assert _rel(left["C_eff"], right["C_eff"]) < GATE_ID
    assert _rel(r["C_eff"], left["C_eff"]) < _gate_id(name)
    assert _rel(r["C_eff_EB"], left["C_eff_EB"]) < _gate_id(name)
    assert _rel(r["Mass"], left["Mass"]) < 1e-12
    K = np.asarray(r["C_eff"])
    assert np.allclose(K, K.T, rtol=0, atol=1e-12 * np.abs(K).max())
    assert np.linalg.eigvalsh(K).min() > 0


def test_isotropic_tet10_hits_the_closed_forms(tmp_path_factory):
    """Extension and bending of an isotropic prism have polynomial fields
    (quadratic Poisson warping), so the tet10 segment gives E A, E I and
    their couplings about the origin exactly."""
    r = _fixture_run("iso_rect_tet10", tmp_path_factory)
    E, W, H, x0, y0 = 210000.0, 2.0, 1.0, -0.5, -0.3
    A = W * H
    xc, yc = x0 + W / 2, y0 + H / 2
    I22 = W * H ** 3 / 12 + A * yc ** 2
    I33 = H * W ** 3 / 12 + A * xc ** 2
    I23 = A * xc * yc
    EB = np.asarray(r["C_eff_EB"])
    exact = np.array([[E * A, E * A * yc, -E * A * xc],
                      [E * A * yc, E * I22, -E * I23],
                      [-E * A * xc, -E * I23, E * I33]])
    assert _rel(EB[np.ix_([0, 2, 3], [0, 2, 3])], exact) < 1e-12


def test_mixed_ends_carry_both_face_types(tmp_path_factory):
    r = _fixture_run("comp_rect_mixed", tmp_path_factory)
    for end in r["ends"]:
        assert set(end["counts"]) == {"quad4", "tri3"}


# -------------------------------------------------- (b) parity with the port
def _run_gauge(name, gauge, workdir):
    from opensg_solid.sg_materials import elem_rotation_from_yaml
    from opensg_solid.sg_mesh import load_sg_input
    from opensg_solid.sg_segment import segment_homo
    sc = load_sg_input(os.path.join(FIX, name + ".yaml"))
    er = (None if sc.get("orientation") is None
          else elem_rotation_from_yaml(sc["orientation"]))
    return segment_homo(sc, os.path.join(str(workdir), name),
                        elem_rotation=er, gauge=gauge)


@pytest.mark.parametrize("name", PARITY)
def test_parity_with_the_jax_port(name, tmp_path, tmp_path_factory):
    """The port pins the end fluctuations to the nodal rigid modes: with
    that gauge the route is the port.  The default area gauge (FEniCS
    OpenSG v1.2) leaves a prismatic segment alone and moves a tapered one:
    1.7e-2 of max|K| on comp_taper_hex8 (coarse, off-centre, taper 0.6),
    1.3e-4 on the +-45 tube of fenics_v12_ref.json (measured)."""
    ref = json.load(open(os.path.join(FIX, "tw_claude_ref.json")))["cases"]
    r = _run_gauge(name, "nodal", tmp_path)
    assert _rel(r["C_eff"], ref[name]["segment"]) < GATE_PARITY
    assert _rel(r["ends"][0]["C_eff"], ref[name]["left"]) < GATE_PARITY
    assert _rel(r["ends"][1]["C_eff"], ref[name]["right"]) < GATE_PARITY
    rd = _fixture_run(name, tmp_path_factory)
    for e, en in zip(rd["ends"], r["ends"]):
        assert _rel(e["C_eff"], en["C_eff"]) < GATE_ID
    if name in PRISMATIC:
        assert _rel(rd["C_eff"], r["C_eff"]) < GATE_ID
    else:
        assert _rel(rd["C_eff"], r["C_eff"]) > 1e-7


@pytest.mark.parametrize("name", ["c_tube_taper_hex8",
                                  "e_tube_taper_pm45_hex8"])
def test_tapered_tube_equals_fenics_v12(name, tmp_path_factory):
    """FEniCS OpenSG v1.2 on the same tapered tubes (stored numbers).  At
    this route's quadrature the laws agree to 6e-14, at the degree FEniCS
    picks for the non-affine hexes to 0.22e-6 (measured)."""
    ref = json.load(open(os.path.join(FIX, "fenics_v12_ref.json")))["cases"]
    r = _fixture_run(name, tmp_path_factory)
    f = ref[name]
    assert _rel(r["C_eff"], f["segment_quadrature_degree_2"]) < GATE_PARITY
    assert _rel(r["C_eff"], f["segment"]) < 1e-6
    assert _rel(r["Mass"], f["segment_mass"]) < 1e-12
    assert _rel(r["ends"][0]["C_eff"], f["end0"]) < 1e-6
    assert _rel(r["ends"][1]["C_eff"], f["end1"]) < 1e-6


# ------------------------------------------------------- (c) tapered segments
@pytest.mark.parametrize("name", TAPERED)
def test_tapered_segment(name, tmp_path_factory):
    ref = json.load(open(os.path.join(FIX, "segment_ref.json")))["cases"]
    r = _fixture_run(name, tmp_path_factory)
    assert _rel(r["C_eff"], ref[name]["C_eff"]) < GATE_PARITY
    assert _rel(r["C_eff_EB"], ref[name]["C_eff_EB"]) < GATE_PARITY
    # every section is the z = 0 section scaled by s(z) = 1 - 0.4 z / L
    # about the origin: the mass per unit length is the z = 0 section
    # mass times the average of s^2, the first moments of s^3 and the
    # second moments of s^4
    t = 0.6
    a2 = (1 + t + t ** 2) / 3.0
    a3 = (1 + t) * (1 + t ** 2) / 4.0
    a4 = (1 + t + t ** 2 + t ** 3 + t ** 4) / 5.0
    M0 = np.asarray(r["ends"][0]["Mass"])
    scale = np.array([[a2, 0, 0, 0, a3, a3], [0, a2, 0, a3, 0, 0],
                      [0, 0, a2, a3, 0, 0], [0, a3, a3, a4, 0, 0],
                      [a3, 0, 0, 0, a4, a4], [a3, 0, 0, 0, a4, a4]])
    assert _rel(r["Mass"], M0 * scale) < 1e-12
    assert _rel(r["Mass"], ref[name]["Mass"]) < 1e-12
    # the law lies between the two end laws (a section that shrinks)
    KL = np.diag(r["ends"][0]["C_eff"])
    KR = np.diag(r["ends"][1]["C_eff"])
    K = np.diag(r["C_eff"])
    assert np.all(K < KL) and np.all(K > KR)


# ---------------------------------------------------------------- (d) frames
def test_rotated_rows_equal_the_block_angle_twin(tmp_path):
    out = []
    for blk in (False, True):
        nodes, elems, names, frames, mats = MS.composite(taper=0.6,
                                                         block_angles=blk)
        p = MS.write_yaml(str(tmp_path / ("blk%d.yaml" % blk)), nodes, elems,
                          names, frames, mats)
        out.append(_run(p, tmp_path))
    assert _rel(out[0]["C_eff"], out[1]["C_eff"]) < 1e-12
    for e0, e1 in zip(out[0]["ends"], out[1]["ends"]):
        assert _rel(e0["C_eff"], e1["C_eff"]) < 1e-12
    assert _rel(out[0]["Mass"], out[1]["Mass"]) < 1e-14


def test_the_api_boundary_argument_selects_the_route(tmp_path,
                                                     tmp_path_factory):
    nodes, elems, names, frames, mats = MS.composite()
    p = MS.write_yaml(str(tmp_path / "noheader.yaml"), nodes, elems, names,
                      frames, mats, header={"msg": "solid"})
    r = _run(p, tmp_path, n_model=1, boundary="aperiodic", refined=1)
    ref = _fixture_run("comp_rect_hex8", tmp_path_factory)
    assert r["route"] == "solid segment"
    assert _rel(r["C_eff"], ref["C_eff"]) < 1e-12


# ------------------------------------------------- (e) command line, refusals
def _block(text, title, n):
    lines = text.splitlines()
    i = next(k for k, ln in enumerate(lines) if title in ln)
    rows = [ln for ln in lines[i + 1:] if re.search(r"\dE[+-]\d", ln)][:n]
    return np.array([[float(v) for v in ln.split()] for ln in rows])


def test_command_line_writes_both_out_files(tmp_path, capsys,
                                            tmp_path_factory):
    from opensg.cli import main
    p = str(tmp_path / "comp_taper_hex8.yaml")
    shutil.copy(os.path.join(FIX, "comp_taper_hex8.yaml"), p)
    try:
        rc = main([p])
    except SystemExit as e:                     # pragma: no cover
        rc = e.code
    assert not rc
    log = capsys.readouterr().out
    route = [ln for ln in log.splitlines() if ln.startswith(" route     :")]
    assert len(route) == 1 and "solid segment" in route[0]
    assert "z = 0 (16 quad4)" in route[0] and "z = 2 (16 quad4)" in route[0]
    r = _fixture_run("comp_taper_hex8", tmp_path_factory)
    out = open(p[:-5] + ".out").read()
    assert "aperiodic" in out.splitlines()[0]
    K = _block(out, "The Effective Timoshenko Beam Stiffness Matrix", 6)
    M = _block(out, "The 6X6 Mass Matrix", 6)
    assert _rel(K, r["C_eff"]) < 1e-7 and _rel(M, r["Mass"]) < 1e-7
    ends = open(p[:-5] + "_ends.out").read()
    assert ends.count("The Effective Timoshenko Beam Stiffness Matrix") == 2
    KR = _block(ends.split("End section 2")[1],
                "The Effective Timoshenko Beam Stiffness Matrix", 6)
    assert _rel(KR, r["ends"][1]["C_eff"]) < 1e-7


def test_refined_0_writes_the_classical_law(tmp_path, tmp_path_factory):
    r0 = _run(os.path.join(FIX, "comp_taper_hex8.yaml"), tmp_path,
              refined=0)
    r = _fixture_run("comp_taper_hex8", tmp_path_factory)
    assert r0["law"].shape == (4, 4)
    assert _rel(r0["law"], r["C_eff_EB"]) < 1e-12


def test_dehomogenization_is_refused(tmp_path):
    from opensg.cli import main
    p = str(tmp_path / "seg.yaml")
    shutil.copy(os.path.join(FIX, "comp_rect_hex8.yaml"), p)
    with pytest.raises(SystemExit, match="no recovery yet"):
        main([p, "D"])


def test_a_2d_beam_sg_with_aperiodic_is_refused(tmp_path):
    hdr = {"msg": "solid", "n_model": 1, "refined": 1, "aperiodic": 1}
    p = _face_yaml(tmp_path, "comp_rect_hex8", header=hdr)
    with pytest.raises(ValueError, match="solid segment route"):
        _run(p, tmp_path)


def test_iterative_solvers_are_refused(tmp_path):
    with pytest.raises(ValueError, match="SuperLU"):
        _run(os.path.join(FIX, "comp_rect_hex8.yaml"), tmp_path,
             solver="cg")


def _one_cell_yaml(path, nodes, cell):
    MS.write_yaml(str(path), np.asarray(nodes, float), [cell], ["steel"],
                  [MS.IDENTITY], [MS.STEEL])
    return str(path)


def test_mixed_tet10_and_open_ends_are_refused(tmp_path):
    from opensg_solid.sg_segment import SegmentInputError
    c = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)], float)
    mids = [(0, 1), (1, 2), (0, 2), (0, 3), (2, 3), (1, 3)]
    n10 = np.vstack([c, [(c[a] + c[b]) / 2 for a, b in mids]])
    p = str(tmp_path / "mix10.yaml")
    MS.write_yaml(p, n10, [list(range(10)), [0, 1, 2, 3]], ["steel"] * 2,
                  [MS.IDENTITY] * 2, [MS.STEEL])
    with pytest.raises(SegmentInputError, match="tet10"):
        _run(p, tmp_path)
    p = _one_cell_yaml(tmp_path / "tip.yaml", c, [0, 1, 2, 3])
    with pytest.raises(SegmentInputError, match="no element face"):
        _run(p, tmp_path)
    p = _one_cell_yaml(tmp_path / "wedge.yaml",
                       np.vstack([c, [(1, 1, 1), (2, 2, 2)]]),
                       [0, 1, 2, 3, 4, 5])
    with pytest.raises(SegmentInputError, match="6-node"):
        _run(p, tmp_path)
