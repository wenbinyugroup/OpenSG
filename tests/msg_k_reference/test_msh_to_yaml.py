"""Unit test: gmsh `.msh` -> solid SG yaml (io.msh_to_yaml).

Template mode: a synthetic two-hex gmsh 2.2 mesh with a $PhysicalNames
block, one physical tag per hex.  Without materials the converter must emit
a FILL_IN template whose sets are named from $PhysicalNames and whose
material entries carry the SAME names (the solid reader binds sets to
materials by name); check_filled must refuse the template and accept the
filled file; `opensg msh_to_yaml` is the CLI face of the same thing.

Reader contract on what gmsh writes (msh text built in the tests, no gmsh
dependency): lower-dimensional cells dropped and counted, node ids honoured
in any row order and with gaps, orphan nodes compacted, the elementary-tag
fallback without physical tags (tag 0 still covering the whole mesh), the
legacy-ASCII-2.2 format gate, and refusals of cells written twice and of
2-D cells spanning z.  End to
end: a periodic curved tet10 (and its tet4 corner twin) two-phase laminate
cube with boundary tri6 faces and shuffled node rows, converted and
homogenized, against the closed-form periodic laminate C to 1e-9.

Run:  pytest tests/msg_k_reference -q      (env opensg_2_0)
"""
import itertools
import os

import numpy as np
import pytest

MSH = """$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
2
3 1 "core"
3 2 "skin"
$EndPhysicalNames
$Nodes
12
1 0 0 0
2 1 0 0
3 1 1 0
4 0 1 0
5 0 0 1
6 1 0 1
7 1 1 1
8 0 1 1
9 0 0 2
10 1 0 2
11 1 1 2
12 0 1 2
$EndNodes
$Elements
2
1 5 2 1 1 1 2 3 4 5 6 7 8
2 5 2 2 1 5 6 7 8 9 10 11 12
$EndElements
"""


@pytest.fixture()
def msh(tmp_path):
    p = tmp_path / "two_hex.msh"
    p.write_text(MSH)
    return str(p)


def test_template_names_sets_from_physicalnames(msh):
    from opensg_solid.io.msh_to_yaml import FILL, convert

    r = convert(msh)
    assert r["filled"] is False
    assert r["n_nodes"] == 12 and r["n_elements"] == 2 and r["npe"] == 8
    assert r["sets"] == {"core": 1, "skin": 1}
    txt = open(r["path"]).read()
    # sets and material entries carry the SAME $PhysicalNames names ...
    assert "- name: core" in txt and "- name: skin" in txt
    # ... and every number of the material block is a placeholder, as is
    # the macro model (the engine's silent default would be plate)
    assert FILL in txt and ("n_model: %s_N_MODEL" % FILL) in txt
    assert "msg: solid" in txt.splitlines()[0]
    # --n_model pins the macro model; the materials stay placeholders
    r3 = convert(msh, n_model=3)
    txt3 = open(r3["path"]).read()
    assert "n_model: 3" in txt3 and FILL in txt3
    with pytest.raises(ValueError, match="n_model"):
        convert(msh, n_model=4)


def test_check_filled_refuses_the_template_and_passes_a_filled_file(msh):
    from opensg_solid.io.msh_to_yaml import check_filled, convert

    r = convert(msh)
    msg = check_filled(r["path"])
    assert msg is not None and "UNFILLED" in msg and "materials" in msg

    mats = [{"name": "core", "density": 100.0, "E": 1e8, "G": 4e7,
             "nu": 0.25},
            {"name": "skin", "density": 2700.0, "E": 69e9, "G": 26.5e9,
             "nu": 0.3}]
    filled = convert(msh, materials=mats)
    assert filled["filled"] is True
    assert check_filled(filled["path"]) is None
    # the filled file parses in the solid reader and binds sets by name
    from opensg_solid.io.sg_input import read_opensg_yaml

    sg = read_opensg_yaml(filled["path"])
    assert len(sg["nodes"]) == 12 and len(sg["cells"]) == 2


def test_cli_emits_template_and_engine_refuses_it(msh, capsys):
    from opensg.cli import main

    assert main(["msh_to_yaml", msh]) == 0
    assert "MATERIALS / LAYUP NOT ADDED" in capsys.readouterr().out
    yml = os.path.splitext(msh)[0] + ".yaml"
    assert os.path.exists(yml)
    # freshness: a second run without --force skips
    assert main(["msh_to_yaml", msh]) == 0
    assert "up to date" in capsys.readouterr().out
    # the engine names the unfilled fields instead of running
    with pytest.raises(SystemExit, match="UNFILLED"):
        main([yml])


def test_cli_missing_file_fails_cleanly(capsys):
    from opensg.cli import main

    assert main(["msh_to_yaml", "does_not_exist.msh"]) == 1
    assert "FAILED" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# reader contract on what gmsh writes
# ---------------------------------------------------------------------------
# gmsh midside order: tet10 edges (01, 12, 02, 03, 23, 13), tri6 (01, 12, 02)
_TET10_EDGES = ((0, 1), (1, 2), (0, 2), (0, 3), (2, 3), (1, 3))
_TRI6_EDGES = ((0, 1), (1, 2), (0, 2))


def _msh_text(nodes, elems, names=(), fmt="2.2 0 8", rows=None):
    """A gmsh legacy ASCII mesh as text.

    In:  nodes {id int: (3,) float}; elems list of (type int, tags list[int],
         node ids list[int]); names list of (dim, tag, name); fmt str -- the
         $MeshFormat line; rows list[int] | None -- the node ids in $Nodes
         row order (None: ascending)
    Out: str."""
    out = ["$MeshFormat", fmt, "$EndMeshFormat"]
    if names:
        out += ["$PhysicalNames", str(len(names))]
        out += ['%d %d "%s"' % nm for nm in names]
        out.append("$EndPhysicalNames")
    rows = sorted(nodes) if rows is None else rows
    out += ["$Nodes", str(len(rows))]
    out += ["%d %r %r %r" % ((i,) + tuple(float(v) for v in nodes[i]))
            for i in rows]
    out += ["$EndNodes", "$Elements", str(len(elems))]
    out += [" ".join(str(v) for v in [k + 1, t, len(tg)] + list(tg) + list(c))
            for k, (t, tg, c) in enumerate(elems)]
    out.append("$EndElements")
    return "\n".join(out) + "\n"


def _two_tet10(big_id=False, tet_tags=([3, 1], [5, 1, 1, 2]), phys=True):
    """Two face-sharing tet10 cells plus a tri6 face, a line3 and a point
    element on an orphan node (id 2); node ids gapped (3k + 5), rows
    shuffled.

    In:  big_id bool -- renumber one node to 10**9 (the sorted id map);
         tet_tags ([int], [int]) -- the tag lists of the two tet10 lines;
         phys bool -- False writes the boundary cells' tags as
         `0 <elementary>`
    Out: dict {xyz {id: (3,) float}, elems [(type, tags, ids)], names
         [(dim, tag, name)], rows [id] shuffled, tets [[10 ids], [10 ids]]}."""
    corner = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                       [1, 1, 1 / 3.0]])
    pts, mid = list(corner), {}

    def m(a, b):
        """The midside point index of edge (a, b), created on first use.

        In:  a int, b int -- corner indices
        Out: int -- index into pts."""
        key = (min(a, b), max(a, b))
        if key not in mid:
            mid[key] = len(pts)
            pts.append(0.5 * (corner[a] + corner[b]))
        return mid[key]

    conn = [list(t) + [m(t[a], t[b]) for a, b in _TET10_EDGES]
            for t in ((0, 1, 2, 3), (1, 2, 4, 3))]
    ids = [3 * k + 5 for k in range(len(pts))]
    if big_id:
        ids[-1] = 10 ** 9
    xyz = {i: np.asarray(p, float) for i, p in zip(ids, pts)}
    xyz[2] = np.array([9.0, 9.0, 9.0])
    tri = [conn[0][a] for a in (0, 1, 2, 4, 5, 6)]
    elems = [(15, [], [2]),
             (8, [7, 1] if phys else [0, 1],
              [ids[conn[0][0]], ids[conn[0][1]], ids[conn[0][4]]]),
             (9, [7, 2, 0] if phys else [0, 2], [ids[k] for k in tri]),
             (11, list(tet_tags[0]), [ids[k] for k in conn[0]]),
             (11, list(tet_tags[1]), [ids[k] for k in conn[1]])]
    rows = [int(v) for v in np.random.default_rng(7).permutation(sorted(xyz))]
    return {"xyz": xyz, "elems": elems, "rows": rows,
            "names": [(3, 3, "core"), (3, 5, "skin"), (2, 3, "face")],
            "tets": [[ids[k] for k in c] for c in conn]}


def _write(tmp_path, d, name="two_tet10.msh", fmt="2.2 0 8"):
    """Write a _two_tet10-style dict as a .msh.

    In:  tmp_path Path; d dict (xyz, elems, names, rows); name str; fmt str
    Out: str -- the path."""
    p = tmp_path / name
    p.write_text(_msh_text(d["xyz"], d["elems"], d["names"], fmt, d["rows"]))
    return str(p)


@pytest.mark.parametrize("big_id", [False, True])
def test_reader_keeps_top_dimension_and_honours_node_ids(tmp_path, big_id):
    """Only the tet10 cells kept, every cell node at its id's coordinates,
    the orphan compacted, drops counted, names of the kept dimension only.

    In:  tmp_path Path; big_id bool -- dense or sorted id map
    Out: None (asserts)."""
    from opensg_solid.io.msh_to_yaml import read_msh22

    d = _two_tet10(big_id)
    m = read_msh22(_write(tmp_path, d))
    assert m["etype"] == 11 and m["npe"] == 10 and m["cells"].shape == (2, 10)
    for c, ids in zip(m["cells"], d["tets"]):
        assert np.array_equal(m["nodes"][c - 1],
                              np.array([d["xyz"][i] for i in ids]))
    assert m["nodes"].shape == (14, 3) and m["orphans"] == 1
    assert m["dropped"] == {15: 1, 8: 1, 9: 1}
    assert m["phys"].tolist() == [3, 5] and m["tag_source"] == "physical"
    assert m["phys_names"] == {3: "core", 5: "skin"}


def test_convert_notes_drops_and_writes_full_precision(tmp_path):
    """convert() reports drops/orphans in `note` (empty for a clean mesh)
    and its node rows read back bit-identical.

    In:  tmp_path Path
    Out: None (asserts)."""
    from opensg_solid.io.msh_to_yaml import convert, read_msh22
    from opensg_solid.io.sg_input import read_opensg_yaml

    p = _write(tmp_path, _two_tet10())
    mats = [{"name": "core", "density": 1.0, "E": 1e9, "G": 4e8, "nu": 0.25},
            {"name": "skin", "density": 1.0, "E": 2e9, "G": 8e8, "nu": 0.25}]
    r = convert(p, materials=mats, n_model=3)
    assert r["note"] == ("dropped 1 tri6 + 1 line3 + 1 point lower-dim cells;"
                         " compacted 1 orphan node")
    assert r["sets"] == {"core": 1, "skin": 1} and r["n_nodes"] == 14
    assert r["dropped"] == {15: 1, 8: 1, 9: 1} and r["orphans"] == 1
    assert r["tag_source"] == "physical"
    sg = read_opensg_yaml(r["path"])
    assert np.array_equal(sg["nodes"], read_msh22(p)["nodes"])
    d = _two_tet10()
    d["elems"] = d["elems"][3:]
    d["xyz"].pop(2)
    d["rows"].remove(2)
    assert convert(_write(tmp_path, d, "clean.msh"))["note"] == ""


@pytest.mark.parametrize("fmt, word", [("4.1 0 8", "gmsh 4.1 ascii"),
                                       ("2.2 1 8", "gmsh 2.2 binary")])
def test_reader_refuses_msh4_and_binary(tmp_path, fmt, word):
    """msh 4.1 and binary 2.2 -> ValueError naming the format found and a
    msh22 re-save that keeps the input file.

    In:  tmp_path Path; fmt str -- $MeshFormat line; word str -- expected
         format words
    Out: None (asserts)."""
    from opensg_solid.io.msh_to_yaml import read_msh22

    with pytest.raises(ValueError, match="legacy ASCII 2.2") as e:
        read_msh22(_write(tmp_path, _two_tet10(), fmt=fmt))
    assert word in str(e.value)
    assert "-format msh22 -o two_tet10_msh22.msh -save" in str(e.value)


def test_reader_header_and_section_layout(tmp_path):
    """A headerless file is refused; section names with trailing blanks
    read the same as clean ones.

    In:  tmp_path Path
    Out: None (asserts)."""
    from opensg_solid.io.msh_to_yaml import read_msh22

    txt = open(_write(tmp_path, _two_tet10())).read()
    bare = tmp_path / "bare.msh"
    bare.write_text(txt.split("$EndMeshFormat\n", 1)[1])
    with pytest.raises(ValueError, match=r"no \$MeshFormat"):
        read_msh22(str(bare))
    pad = tmp_path / "pad.msh"
    pad.write_text(txt.replace("\n", "  \n"))
    a, b = read_msh22(str(tmp_path / "two_tet10.msh")), read_msh22(str(pad))
    assert np.array_equal(a["nodes"], b["nodes"])
    assert np.array_equal(a["cells"], b["cells"])
    cut = tmp_path / "cut.msh"
    cut.write_text(txt.split("$EndElements", 1)[0])
    with pytest.raises(ValueError, match=r"^cut\.msh: \$Elements has no"):
        read_msh22(str(cut))


def test_reader_refuses_corrupt_ids_and_mixed_cells(tmp_path):
    """Missing / duplicate node ids, two 3-D cell types, a prism-only SG and
    an unsupported higher-order cell -> ValueErrors with cell words.

    In:  tmp_path Path
    Out: None (asserts)."""
    from opensg_solid.io.msh_to_yaml import read_msh22

    d = _two_tet10()
    gone = d["tets"][1][9]
    d["xyz"].pop(gone)
    d["rows"].remove(gone)
    with pytest.raises(ValueError, match="node id %d," % gone):
        read_msh22(_write(tmp_path, d, "gone.msh"))
    d = _two_tet10()
    d["rows"].append(d["rows"][0])
    with pytest.raises(ValueError, match="appears twice"):
        read_msh22(_write(tmp_path, d, "dup.msh"))
    d = _two_tet10()
    d["elems"].append((4, [3, 1], d["tets"][0][:4]))
    with pytest.raises(ValueError, match="dominant tet10 plus tet4 "):
        read_msh22(_write(tmp_path, d, "mixed.msh"))
    d = _two_tet10()
    d["elems"][3:] = [(6, [3, 1], d["tets"][0][:6])]
    with pytest.raises(ValueError, match="it has 6-node prism$"):
        read_msh22(_write(tmp_path, d, "prism.msh"))
    d = _two_tet10()
    d["elems"].append((17, [3, 1], d["tets"][0] + d["tets"][1]))
    with pytest.raises(ValueError, match=r"plus 20-node hex \(hex20\)"):
        read_msh22(_write(tmp_path, d, "hex20.msh"))
    d = _two_tet10()
    d["elems"].append((99, [3, 1], d["tets"][0]))
    with pytest.raises(ValueError, match="carries gmsh type 99"):
        read_msh22(_write(tmp_path, d, "unknown.msh"))
    d = _two_tet10()
    d["elems"] = d["elems"][:2]
    with pytest.raises(ValueError, match=r"only 0-/1-D cells \(1 line3,"
                                         r" 1 point\)"):
        read_msh22(_write(tmp_path, d, "lines.msh"))


def test_reader_refuses_cells_written_twice(tmp_path):
    """One volume in two physical groups (gmsh 2.2 writes its cells once per
    group) and a repeated element id are refused; one entity under two tags
    with distinct cells is not.

    In:  tmp_path Path
    Out: None (asserts)."""
    from opensg_solid.io.msh_to_yaml import read_msh22

    d = _two_tet10(tet_tags=([3, 1], [3, 1]))
    d["elems"] += [(11, [9, 1], c) for c in d["tets"]]
    with pytest.raises(ValueError, match="elementary volume 1 is in physical"
                                         " groups 3 and 9"):
        read_msh22(_write(tmp_path, d, "twice.msh"))
    p = _write(tmp_path, _two_tet10(), "same_id.msh")
    txt = open(p).read().splitlines()
    k = max(i for i, s in enumerate(txt) if s.startswith("5 11 "))
    txt.insert(k + 1, txt[k])
    txt[txt.index("$Elements") + 1] = "6"
    open(p, "w").write("\n".join(txt) + "\n")
    with pytest.raises(ValueError, match="element id 5 appears twice"):
        read_msh22(p)
    assert read_msh22(_write(tmp_path, _two_tet10()))["phys"].tolist() \
        == [3, 5]


def test_reader_refuses_surfaces_spanning_z_and_drops_hiorder_faces(
        tmp_path):
    """Only 2-D cells off the x-y plane (a 3-D mesh saved with physical
    surfaces only) -> ValueError; a tri10 face beside tet10 cells is
    dropped by dimension.

    In:  tmp_path Path
    Out: None (asserts)."""
    from opensg_solid.io.msh_to_yaml import read_msh22

    d = _two_tet10()
    d["elems"] = [(9, [7, 2], [d["tets"][0][k] for k in (0, 1, 3, 4, 9, 7)])]
    with pytest.raises(ValueError, match=r"only 2-D cells \(1 tri6\)"
                                         " spanning z: not a 2-D SG.*"
                                         "Physical Volume"):
        read_msh22(_write(tmp_path, d, "skin.msh"))
    d = _two_tet10()
    d["elems"].insert(0, (21, [7, 2], d["tets"][0][:10]))
    m = read_msh22(_write(tmp_path, d, "tri10.msh"))
    assert m["dropped"] == {21: 1, 15: 1, 8: 1, 9: 1} and m["etype"] == 11


def test_reader_2d_sg_drops_boundary_lines_only(tmp_path):
    """A 2-D SG keeps its triangles verbatim and drops its boundary lines.

    In:  tmp_path Path
    Out: None (asserts)."""
    from opensg_solid.io.msh_to_yaml import read_msh22

    xyz = {1: (0, 0, 0), 2: (1, 0, 0), 3: (1, 1, 0), 4: (0, 1, 0)}
    elems = [(1, [9, 1], [1, 2]), (1, [9, 2], [2, 3]),
             (2, [4, 1], [1, 2, 3]), (2, [4, 1], [1, 3, 4])]
    p = tmp_path / "sq.msh"
    p.write_text(_msh_text(xyz, elems, [(1, 9, "edge"), (2, 4, "m")]))
    m = read_msh22(str(p))
    assert m["etype"] == 2 and m["cells"].tolist() == [[1, 2, 3], [1, 3, 4]]
    assert m["dropped"] == {1: 2} and m["orphans"] == 0
    assert m["phys_names"] == {4: "m"} and m["phys"].tolist() == [4, 4]


@pytest.mark.parametrize("tet_tags, src, phys", [
    (([0, 1], [0, 1]), "elementary", [1, 1]),
    (([0, 4], [0, 6]), "elementary", [4, 6]),
    (([0], [0]), "none", [1, 1]),
    (([], []), "none", [1, 1])])
def test_no_physical_groups_fall_back(tmp_path, tet_tags, src, phys):
    """All-zero physical tags -> elementary tags (n_tags >= 2) else tag 1.

    In:  tmp_path Path; tet_tags ([int], [int]) -- the tet10 tag lists;
         src str -- expected tag_source; phys [int] -- expected set tags
    Out: None (asserts)."""
    from opensg_solid.io.msh_to_yaml import read_msh22

    m = read_msh22(_write(tmp_path, _two_tet10(tet_tags=tet_tags,
                                               phys=False)))
    assert m["tag_source"] == src and m["phys"].tolist() == phys
    assert m["phys_names"] == {}


def test_cli_mat1_on_a_mesh_without_physical_groups(tmp_path, capsys,
                                                    monkeypatch):
    """`opensg msh_to_yaml --mat1 Al` fills a single-volume mesh without
    physical groups; the console line carries the one-line note.

    In:  tmp_path Path; capsys, monkeypatch (pytest)
    Out: None (asserts)."""
    from opensg.cli import main
    from opensg_solid.io.msh_to_yaml import check_filled

    monkeypatch.chdir(tmp_path)
    p = _write(tmp_path, _two_tet10(tet_tags=([0, 1], [0, 1]), phys=False))
    assert main(["msh_to_yaml", p, "--mat1", "Al", "--n_model", "3"]) == 0
    out = capsys.readouterr().out
    assert ("(tet10; 14 nodes / 42 dofs; dropped 1 tri6 + 1 line3 + 1 point"
            " lower-dim cells; compacted 1 orphan node; no physical tags:"
            " sets from elementary tags [1])") in out
    assert check_filled(os.path.splitext(p)[0] + ".yaml") is None


def test_mesh_without_physical_tags_tag0_covers_everything(tmp_path, capsys,
                                                         monkeypatch):
    """A 2-D gmsh mesh without physical groups (two surfaces): the template
    splits by elementary tag, while --mat0 / {0: m} / a one-material list
    cover every cell as before; a stray flag names the elementary tags.

    In:  tmp_path Path; capsys, monkeypatch (pytest)
    Out: None (asserts)."""
    from opensg.cli import main
    from opensg_solid.io.msh_to_yaml import convert

    p = str(tmp_path / "sq2.msh")
    open(p, "w").write(_msh_text(
        {1: (0, 0, 0), 2: (1, 0, 0), 3: (1, 1, 0), 4: (0, 1, 0)},
        [(15, [0, 1], [1]), (1, [0, 5], [1, 2]),
         (2, [0, 1], [1, 2, 3]), (2, [0, 2], [1, 3, 4])]))
    al = {"name": "Al", "density": 2700.0, "E": 7e10, "G": 2.6e10,
          "nu": 0.33}
    assert convert(p)["sets"] == {"mat_1": 1, "mat_2": 1}
    assert convert(p, materials=[al])["sets"] == {"Al": 2}
    r = convert(p, materials={0: al}, n_model=2)
    assert r["sets"] == {"Al": 2} and r["filled"]
    assert r["note"] == ("dropped 1 line2 + 1 point lower-dim cells;"
                         " no physical tags: one set (tag 0)")
    with pytest.raises(ValueError, match=r"has elementary tags \[1, 2\] \(no"
                       r" physical tags.* for \[3\]; --mat0 alone"):
        convert(p, materials={3: al})
    monkeypatch.chdir(tmp_path)
    assert main(["msh_to_yaml", p, "--mat0", "Al", "--n_model", "2"]) == 0
    assert "mat 0 <- Al" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# end to end: periodic laminate cube vs the closed-form laminate C
# ---------------------------------------------------------------------------
_VO = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


def _c_iso(E, nu):
    """Isotropic 6x6 stiffness from (E, G = E/2(1+nu), nu), Voigt
    (11, 22, 33, 23, 13, 12), engineering shear.

    In:  E float; nu float
    Out: (6, 6) float."""
    S = np.zeros((6, 6))
    S[:3, :3] = -nu / E
    S[[0, 1, 2], [0, 1, 2]] = 1.0 / E
    S[[3, 4, 5], [3, 4, 5]] = 2 * (1 + nu) / E
    return np.linalg.inv(S)


def _to_t(C):
    """Voigt (6, 6) -> the 3x3x3x3 stiffness tensor.

    In:  C (6, 6) float
    Out: (3, 3, 3, 3) float."""
    T = np.zeros((3, 3, 3, 3))
    for I, (i, j) in enumerate(_VO):
        for J, (k, l) in enumerate(_VO):
            for a, b in ((i, j), (j, i)):
                for c, e in ((k, l), (l, k)):
                    T[a, b, c, e] = C[I, J]
    return T


def _laminate_C(Cs, fs, n):
    """The exact periodic C* of a layered medium (Backus / Hill): strain
    Eb + sym(a_k x n) in phase k, traction continuity and <a> = 0.

    In:  Cs list of (6, 6) phase stiffnesses; fs list of volume fractions;
         n (3,) layer normal
    Out: (6, 6) float, Voigt (11, 22, 33, 23, 13, 12)."""
    n = np.asarray(n, float) / np.linalg.norm(n)
    Ts = [_to_t(C) for C in Cs]
    Ni = [np.linalg.inv(np.einsum("j,ijkl,l->ik", n, T, n)) for T in Ts]
    A = np.linalg.inv(sum(f * x for f, x in zip(fs, Ni)))
    out = np.zeros((6, 6))
    for J, (p, q) in enumerate(_VO):
        Eb = np.zeros((3, 3))
        Eb[p, q] = Eb[q, p] = 1.0 if p == q else 0.5
        s0 = [np.einsum("ijkl,kl->ij", T, Eb) for T in Ts]
        t = A @ sum(f * x @ (s @ n) for f, x, s in zip(fs, Ni, s0))
        sb = sum(f * (s + np.einsum("ijkl,k,l->ij", T, x @ (t - s @ n), n))
                 for f, T, s, x in zip(fs, Ts, s0, Ni))
        out[:, J] = [sb[i, j] for i, j in _VO]
    return out


def _kuhn_cube(n, quadratic, curve=0.0, seed=3):
    """A periodic structured tet mesh of the unit cube: n^3 cubes, each cut
    into the 6 right-handed Kuhn tets along its main diagonal (translation
    invariant, so opposite faces match), two phases split at x = 0.5.

    In:  n int -- cubes per side (even); quadratic bool -- tet10 (gmsh
         midside order) else tet4; curve float -- random displacement
         amplitude of the midside nodes off the boundary and off the
         x = 0.5 interface (curved cells; periodic faces and the planar
         interface kept); seed int
    Out: (X (N, 3) float, tets (E, 4|10) int 0-based, faces (F, 3|6) int
         0-based boundary faces, phase (E,) int 1 | 2)."""
    g = np.arange(n + 1) / n
    X = [g[list(c)] for c in itertools.product(range(n + 1), repeat=3)]
    rng = np.random.default_rng(seed)
    mid = {}

    def m(a, b):
        """The midside node index of edge (a, b), created (and possibly
        displaced) on first use.

        In:  a int, b int -- node indices
        Out: int -- index into X."""
        key = (min(a, b), max(a, b))
        if key not in mid:
            p = 0.5 * (X[a] + X[b])
            if curve and (p > 0).all() and (p < 1).all() and p[0] != 0.5:
                p = p + curve * rng.uniform(-1.0, 1.0, 3)
            mid[key] = len(X)
            X.append(p)
        return mid[key]

    tets, faces, phase = [], [], []
    for c in itertools.product(range(n), repeat=3):
        for perm in itertools.permutations(range(3)):
            v = [np.array(c)]
            for a in perm:
                v.append(v[-1] + np.eye(3, dtype=int)[a])
            t = [(p[0] * (n + 1) + p[1]) * (n + 1) + p[2] for p in v]
            if np.linalg.det(np.array([X[q] - X[t[0]] for q in t[1:]])) < 0:
                t[1], t[2] = t[2], t[1]
            phase.append(1 if np.mean([X[q][0] for q in t]) < 0.5 else 2)
            for f in itertools.combinations(t, 3):
                P = np.array([X[q] for q in f])
                if any((P[:, a] == P[0, a]).all() and P[0, a] in (0.0, 1.0)
                       for a in range(3)):
                    faces.append(list(f))
            tets.append(t)
    if quadratic:
        tets = [t + [m(t[a], t[b]) for a, b in _TET10_EDGES] for t in tets]
        faces = [f + [m(f[a], f[b]) for a, b in _TRI6_EDGES] for f in faces]
    return np.array(X), np.array(tets), np.array(faces), np.array(phase)


@pytest.mark.parametrize("quadratic", [True, False])
def test_periodic_laminate_cube_matches_closed_form(tmp_path, quadratic):
    """gmsh-shaped periodic two-phase cube (boundary faces on a physical
    surface, shuffled gapped node rows; tet10 curved) -> convert ->
    plate_homo_2d (the CLI's library call) -> all 21 C terms vs Backus.

    In:  tmp_path Path; quadratic bool -- tet10 (curved) else tet4
    Out: None (asserts)."""
    from opensg_solid.io.msh_to_yaml import convert
    from opensg_solid.sg_homo import plate_homo_2d

    X, tets, faces, phase = _kuhn_cube(2, quadratic,
                                       curve=0.04 if quadratic else 0.0)
    ids = 2 * np.arange(len(X)) + 7
    ft, tt = (9, 11) if quadratic else (2, 4)
    elems = ([(ft, [10, 7], ids[f].tolist()) for f in faces]
             + [(tt, [int(p), int(p)], ids[t].tolist())
                for t, p in zip(tets, phase)])
    rows = [int(v) for v in np.random.default_rng(11).permutation(ids)]
    msh = tmp_path / "lam.msh"
    msh.write_text(_msh_text(dict(zip(ids.tolist(), X)), elems,
                             [(2, 10, "skin"), (3, 1, "stiff"),
                              (3, 2, "soft")], rows=rows))
    E1, n1, E2, n2 = 70e9, 0.33, 3e9, 0.35
    mats = [{"name": "stiff", "density": 2700.0, "E": E1,
             "G": E1 / (2 * (1 + n1)), "nu": n1},
            {"name": "soft", "density": 1200.0, "E": E2,
             "G": E2 / (2 * (1 + n2)), "nu": n2}]
    r = convert(str(msh), out_path=str(tmp_path / "lam.yaml"),
                materials=mats, n_model=3)
    assert r["dropped"] == {ft: len(faces)} and r["orphans"] == 0
    assert r["sets"] == {"stiff": 24, "soft": 24}
    h = plate_homo_2d(r["path"], refined=0, solver="auto", recovery=False,
                      plot=False)
    C = np.asarray(h["C_eff"], float)
    R = _laminate_C([_c_iso(E1, n1), _c_iso(E2, n2)], [0.5, 0.5], [1, 0, 0])
    iu = np.triu_indices(6)
    assert np.abs(C - R)[iu].max() / np.abs(R).max() < 1e-9
