"""The ONE name of the per-element frame block, `orientation` (io.sg_keys), on the solid side.

`elementOrientations` -- the legacy spelling of OpenSG 2.1 and earlier -- is still read: every solid loader renames it to `orientation`
with a one-line note, and a file or SG dict that carries both spellings is refused.  The dialect detectors accept
either spelling, every solid writer writes `orientation:` only, and the shipped solid examples use the one name.
Gates: the law of a framed cell is the same in both spellings and equals its block `angle: 45` twin to 1e-12 of
max|K| (OpenSG <= 2.2 given the new key drops the frames: 2.228 relative); the frames read in the two spellings are
the same arrays (==); the msg-shell ring is `shell` and is refused by the solid SG-input reader in both spellings.
The shell loaders have their own test (tests/msg_shell_windio/test_orientation_key_shell.py).  The guard: in src/
only io/sg_keys.py names the legacy spelling; outside src/ only the legacy-read tests and the docs' note do."""
import os
import re
import struct
import subprocess

import numpy as np
import pytest

assert struct.calcsize("P") == 8

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SOLID_EX = os.path.join(ROOT, "examples", "OpenSG-solid")
SQ_YAML = os.path.join(SOLID_EX, "5_get_solid_props_from_SG", "square_section", "square_tube_2Dsolid.yaml")
SHIPPED = [SQ_YAML,
           os.path.join(SOLID_EX, "5_get_solid_props_from_SG", "UDcomp_2D.yaml"),
           os.path.join(SOLID_EX, "4_get_plate_dehom_from_2DSG", "RHC_SW_2UC_45.yaml")]
SQ_SG = os.path.join(ROOT, "tests", "msg_k_reference", "fixtures", "prevabs", "square_tube.sg")
IEA_RING = os.path.join(ROOT, "examples", "OpenSG_shell", "IEA_blade_beam", "iea_s10_shell.yaml")

NEW = "orientation"
LEGACY = "elementOrientations"           # the legacy spelling: read, never written
KEY_LINE = re.compile(r"^(?:orientation|elementOrientations):", re.M)

# the canonical +45 two-quad cell of test_yaml_frames_applied.py (cells / mat_id / materials mapping)
CANON = """n_model: %d
refined: 0
msg: solid
nodes:
- [0.0, 0.0, 0.0]
- [1.0, 0.0, 0.0]
- [2.0, 0.0, 0.0]
- [0.0, 1.0, 0.0]
- [1.0, 1.0, 0.0]
- [2.0, 1.0, 0.0]
cells:
- [0, 1, 4, 3]
- [1, 2, 5, 4]
mat_id: [1, 1]
materials:
  1: {type: 1, engineering: [150.0e9, 10.0e9, 10.0e9, 5.0e9, 5.0e9, 3.5e9, 0.3, 0.3, 0.4]%s}
"""
# the same cell and material in the MESH spelling (elements 1-based / sets / materials list)
MESH = """n_model: %d
refined: 0
msg: solid
materials:
- name: ply
  density: 0.0
  elastic:
    E: [150.0e9, 10.0e9, 10.0e9]
    G: [5.0e9, 5.0e9, 3.5e9]
    nu: [0.3, 0.3, 0.4]
nodes:
- [0.0 0.0 0.0]
- [1.0 0.0 0.0]
- [2.0 0.0 0.0]
- [0.0 1.0 0.0]
- [1.0 1.0 0.0]
- [2.0 1.0 0.0]
elements:
- [1 2 5 4]
- [2 3 6 5]
%ssets:
  element:
  - name: ply
    labels: [1, 2]
"""
c = s = np.sqrt(0.5)
ROW45 = "- [%.16g, 0.0, %.16g, %.16g, 0.0, %.16g, 0.0, 1.0, 0.0]" % (s, c, c, -s)   # 45 deg about the thickness (y)


def _block(key, rows=ROW45, n=2):
    return "%s:\n%s\n" % (key, "\n".join([rows] * n))


def _write(tmp_path, name, text):
    p = os.path.join(str(tmp_path), name + ".yaml")
    with open(p, "w") as f:
        f.write(text)
    return p


def _respell(src, key, tmp_path, name, drop_msg=False):
    """A tmp copy of `src` with its frame-block key line spelled `key` (the rows untouched)."""
    text = open(src).read()
    assert len(KEY_LINE.findall(text)) == 1, src
    text = KEY_LINE.sub(key + ":", text)
    if drop_msg:
        text = re.sub(r"^msg:.*\n", "", text, flags=re.M)
    return _write(tmp_path, name, text)


def _law(path, n_model, tmp_path):
    from opensg_solid.sg_homo import plate_homo_2d
    r = plate_homo_2d(path, n_model=n_model, refined=0, plot=False, workdir=str(tmp_path), recovery=False)
    return np.asarray(r["law"], float)


def test_the_names():
    from opensg_solid.io import sg_keys
    assert sg_keys.ORIENTATION_KEY == NEW
    assert sg_keys.LEGACY_ORIENTATION_KEYS == (LEGACY,)
    assert sg_keys.ORIENTATION_KEYS == (NEW, LEGACY)
    assert sg_keys.has_orientation_block({"nodes", NEW}) and sg_keys.has_orientation_block([LEGACY])
    assert not sg_keys.has_orientation_block({"nodes", "elements", "sections"})


def test_normalize_renames_in_place_with_one_note_and_refuses_both(capsys):
    from opensg_solid.io.sg_keys import normalize_orientation_key
    rows = [[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]
    d = {"nodes": [], LEGACY: rows}
    assert normalize_orientation_key(d, "a.yaml") is d                      # in place, same object
    assert d == {"nodes": [], NEW: rows} and d[NEW] is rows                 # values untouched
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 1 and "NOTE" in out[0] and "a.yaml" in out[0] and LEGACY in out[0]
    d = {NEW: rows}
    assert normalize_orientation_key(d) == {NEW: rows}                      # the one name: silent no-op
    assert normalize_orientation_key({LEGACY: None}, "n.yaml") == {NEW: None}   # null stays "no frames"
    assert normalize_orientation_key(["not", "a", "mapping"]) == ["not", "a", "mapping"]
    capsys.readouterr()
    with pytest.raises(ValueError, match=r"b\.yaml carries the per-element frame block twice"):
        normalize_orientation_key({NEW: rows, LEGACY: rows}, "b.yaml")
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("n_model", [2, 3])
def test_law_is_one_in_both_spellings_and_both_dialects(tmp_path, n_model, capsys):
    """law(orientation) == law(elementOrientations) == law(angle 45), canonical and mesh spelling alike."""
    K0 = _law(_write(tmp_path, "angle0", CANON % (n_model, "")), n_model, tmp_path)
    K45 = _law(_write(tmp_path, "angle45", CANON % (n_model, ", angle: 45.0")), n_model, tmp_path)
    scale = np.abs(K45).max()
    assert np.max(np.abs(K45 - K0)) > 1e-3 * scale           # the frame matters for this material
    capsys.readouterr()
    for spelling, body in (("canon", lambda k: CANON % (n_model, "") + _block(k)),
                           ("mesh", lambda k: MESH % (n_model, _block(k)))):
        laws = {}
        for key in (NEW, LEGACY):
            laws[key] = _law(_write(tmp_path, "%s_%s" % (spelling, key), body(key)), n_model, tmp_path)
            out = capsys.readouterr().out
            assert " frames    : 2 per-element frames from the file (applied)" in out, (spelling, key, out)
            assert ("NOTE" in out and LEGACY in out) == (key == LEGACY), (spelling, key, out)
            assert np.max(np.abs(laws[key] - K45)) <= 1e-12 * scale, (spelling, key)
        assert np.max(np.abs(laws[NEW] - laws[LEGACY])) <= 1e-12 * scale, spelling


def test_both_spellings_in_one_file_or_dict_are_refused(tmp_path):
    from opensg_solid.io.sg_input import read_opensg_yaml
    from opensg_solid.sg_mesh import load_sg_input
    both = _block(NEW) + _block(LEGACY)
    canon = _write(tmp_path, "canon_both", CANON % (2, "") + both)
    mesh = _write(tmp_path, "mesh_both", MESH % (2, both))
    twice = "carries the per-element frame block twice"
    for loader, p in ((load_sg_input, canon), (load_sg_input, mesh), (read_opensg_yaml, mesh)):
        with pytest.raises(ValueError, match=twice):
            loader(p)
    frames = np.ones((2, 9))
    sc = {"dim": 2, "nodes": np.zeros((6, 3)), "cells": [[0, 1, 4, 3], [1, 2, 5, 4]],
          "mat_id": np.array([1, 1]), "materials": {}, NEW: frames, LEGACY: frames}
    for loader in (load_sg_input, read_opensg_yaml):
        with pytest.raises(ValueError, match=twice):
            loader(dict(sc))


def test_the_dict_entries_keep_legacy_frames(capsys):
    """An SG dict spelled the file's legacy way keeps its frames on both API entries (it lost them before)."""
    from opensg_solid.io.sg_input import read_opensg_yaml
    from opensg_solid.sg_mesh import load_sg_input
    frames = np.arange(18.0).reshape(2, 9)
    sc = {"dim": 2, "nodes": np.zeros((6, 3)), "cells": [[0, 1, 4, 3], [1, 2, 5, 4]],
          "mat_id": np.array([1, 1]), "materials": {}, LEGACY: frames}
    sg = read_opensg_yaml(sc)
    assert sg[NEW] is frames and LEGACY not in sg
    assert LEGACY in sc and NEW not in sc                     # read_opensg_yaml works on a copy
    assert read_opensg_yaml({k: v for k, v in sc.items() if k != LEGACY})[NEW] is None
    out = load_sg_input(sc)                                   # the pass-through entry renames a copy too
    assert out is not sc and out[NEW] is frames and LEGACY not in out
    assert LEGACY in sc and NEW not in sc                     # the caller's dict is left as it was
    assert capsys.readouterr().out.count("NOTE") == 2         # an API dict is noted every time


def test_the_legacy_note_is_printed_once_per_file(tmp_path, capsys):
    """The solid command line parses its yaml twice (the mesh-size line, then the homogenization): one NOTE."""
    from opensg.cli import main
    from opensg_solid.sg_mesh import load_sg_input
    p = _write(tmp_path, "once", CANON % (2, "") + _block(LEGACY))
    capsys.readouterr()
    assert main([p]) == 0
    out = capsys.readouterr().out
    assert out.count("NOTE") == 1 and LEGACY in out, out
    assert load_sg_input(p)[NEW].shape == (2, 9)              # read again: renamed, no second note
    assert "NOTE" not in capsys.readouterr().out


@pytest.mark.parametrize("analysis", ["H", "D"])
def test_the_command_line_refuses_both_spellings_in_one_line(tmp_path, analysis):
    """`opensg <both.yaml> [D]` ends with the one-line reason (SystemExit), not a traceback."""
    from opensg.cli import main
    p = _write(tmp_path, "both_" + analysis, CANON % (2, "") + _block(NEW) + _block(LEGACY))
    with open(os.path.splitext(p)[0] + ".ff", "w") as f:
        f.write("1: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]\n")
    with pytest.raises(SystemExit, match="^refused: .*carries the per-element frame block twice"):
        main([p, analysis])


def test_the_prevabs_square_tube_frames_are_read_in_both_spellings(tmp_path):
    """square_tube_2Dsolid.yaml (PreVABS, 5384 tri3): the same (5384, 9) frames from both readers and spellings."""
    from opensg_solid.io.sg_input import read_opensg_yaml
    from opensg_solid.sg_mesh import block_keys, load_sg_input, read_yaml_header, sg_dialect
    got = {}
    for key in (NEW, LEGACY):
        p = _respell(SQ_YAML, key, tmp_path, "tube_" + key)
        assert key in block_keys(p) and sg_dialect(p) == "solid"
        assert read_yaml_header(p) == {"msg": "solid"}
        a = load_sg_input(p)["orientation"]
        b = read_opensg_yaml(p)["orientation"]
        assert a.shape == (5384, 9) and np.array_equal(a, b)
        got[key] = a
    assert np.array_equal(got[NEW], got[LEGACY])
    R = got[NEW].reshape(-1, 3, 3)                            # proper rotations, i.e. real frames
    assert np.abs(np.einsum("nij,nkj->nik", R, R) - np.eye(3)).max() < 1e-5


@pytest.mark.parametrize("drop_msg", [False, True])
@pytest.mark.parametrize("key", [NEW, LEGACY])
def test_the_shell_ring_is_shell_and_refused_in_both_spellings(tmp_path, key, drop_msg):
    """The IEA msg-shell ring: detected `shell` with and without `msg:`, refused by the solid readers."""
    from opensg_solid.io.sg_input import read_opensg_yaml
    from opensg_solid.sg_mesh import load_sg_input, resolve_msg, sg_dialect
    p = _respell(IEA_RING, key, tmp_path, "ring", drop_msg=drop_msg)
    assert sg_dialect(p) == "shell"
    assert resolve_msg(p) == "shell"
    for loader in (read_opensg_yaml, load_sg_input):          # never the laminate route of load_sg_input
        with pytest.raises(ValueError, match="msg-SHELL"):
            loader(p)


def test_header_stops_at_the_frame_block_in_both_spellings(tmp_path):
    from opensg_solid.sg_mesh import read_yaml_header
    for key in (NEW, LEGACY):
        p = _write(tmp_path, "head_" + key, "msg: solid\nn_model: 1\n" + _block(key) + "refined: 1\n")
        assert read_yaml_header(p) == {"msg": "solid", "n_model": 1}, key


def test_every_solid_writer_writes_orientation_only(tmp_path):
    from opensg_solid.io.msh_to_yaml import convert
    from opensg_solid.io.sc_to_yaml import sg_to_yaml
    from opensg_solid.io.sg_input import read_opensg_yaml
    from opensg_solid.sg_mesh import load_sg_input
    msh = os.path.join(str(tmp_path), "two_hex.msh")
    with open(msh, "w") as f:                                 # two hex8 on two physical tags (gmsh 2.2)
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$PhysicalNames\n2\n3 1 \"core\"\n3 2 \"skin\"\n"
                "$EndPhysicalNames\n$Nodes\n12\n1 0 0 0\n2 1 0 0\n3 1 1 0\n4 0 1 0\n5 0 0 1\n6 1 0 1\n"
                "7 1 1 1\n8 0 1 1\n9 0 0 2\n10 1 0 2\n11 1 1 2\n12 0 1 2\n$EndNodes\n$Elements\n2\n"
                "1 5 2 1 1 1 2 3 4 5 6 7 8\n2 5 2 2 1 5 6 7 8 9 10 11 12\n$EndElements\n")
    mats = [{"name": "core", "density": 100.0, "E": 1e8, "G": 4e7, "nu": 0.25},
            {"name": "skin", "density": 2700.0, "E": 69e9, "G": 26.5e9, "nu": 0.3}]
    y_msh = convert(msh, materials=mats, n_model=3)["path"]
    y_sg = sg_to_yaml(SQ_SG, out_path=os.path.join(str(tmp_path), "square_tube.yaml"))["yaml"]
    for y, n in ((y_msh, 2), (y_sg, 5384)):
        text = open(y).read()
        assert len(re.findall(r"^orientation:", text, re.M)) == 1, y
        assert LEGACY not in text, y
        assert read_opensg_yaml(y)["orientation"].shape == (n, 9)
        assert load_sg_input(y)["orientation"].shape == (n, 9)


@pytest.mark.parametrize("path", SHIPPED, ids=os.path.basename)
def test_the_shipped_solid_examples_use_the_one_name(path):
    text = open(path).read()
    assert len(re.findall(r"^orientation:", text, re.M)) == 1 and LEGACY not in text


# the tests whose job is to READ the legacy spelling: they write it on purpose (a file, a dict key, a parameter)
LEGACY_READ_TESTS = ("tests/unit/test_orientation_key.py", "tests/unit/test_yaml_frames_applied.py",
                     "tests/msg_k_reference/test_sg_input_writer.py",
                     "tests/msg_shell_windio/test_orientation_key_shell.py")
NOTE = re.compile(r"legacy")        # a docs line names the legacy spelling only as the note that it is read


def _repo_files(top):
    """The repository's files under ROOT/top, as ROOT-relative paths: git's list (tracked, plus untracked files
    that are not ignored) when ROOT is a git checkout -- a local run's ignored outputs are not repository content
    -- else every file of a walk that skips __pycache__."""
    try:
        out = subprocess.run(["git", "-C", ROOT, "ls-files", "-z", "--cached", "--others", "--exclude-standard",
                              "--", top], capture_output=True, check=True).stdout
        rels = [p for p in out.decode("utf-8", "replace").split("\0") if p]
    except (OSError, subprocess.CalledProcessError):
        rels = []
        for root, dirs, files in os.walk(os.path.join(ROOT, top)):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            rels += [os.path.relpath(os.path.join(root, fn), ROOT).replace(os.sep, "/") for fn in files]
        if os.path.isfile(os.path.join(ROOT, top)):
            rels = [top]
    return sorted(set(p for p in rels if os.path.isfile(os.path.join(ROOT, p))))


def _legacy_lines(rel):
    """[(line number, line)] of ROOT/rel that name the legacy spelling (any file, read as text)."""
    with open(os.path.join(ROOT, rel), encoding="utf-8", errors="replace") as f:
        return [(i, ln.strip()) for i, ln in enumerate(f, 1) if LEGACY in ln]


def test_the_legacy_name_lives_only_in_sg_keys():
    hits = [rel for rel in _repo_files("src") if _legacy_lines(rel)]
    assert hits == ["src/opensg_solid/io/sg_keys.py"], hits


def test_outside_src_the_legacy_name_is_only_in_legacy_read_tests_and_the_note():
    """examples/ never names the legacy spelling (every shipped yaml and generator writes `orientation`); a test
    names it only when its job is to read it (LEGACY_READ_TESTS); docs/ and README.md name it only in the note
    that it is the legacy spelling, still read."""
    bad = []
    for top in ("examples", "tests", "docs", "README.md"):
        for rel in _repo_files(top):
            if rel in LEGACY_READ_TESTS:
                continue
            lines = _legacy_lines(rel)
            if top in ("docs", "README.md"):
                lines = [(i, ln) for i, ln in lines if not NOTE.search(ln)]
            bad += ["%s:%d: %s" % (rel, i, ln[:120]) for i, ln in lines]
    assert not bad, bad
