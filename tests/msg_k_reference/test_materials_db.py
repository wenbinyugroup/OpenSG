"""Unit test: the material LIBRARY (io.materials_db + io/materials.yaml)
and its `opensg msh_to_yaml --mat<K> NAME[:ANGLE]` CLI face, plus the
--p_refine one-grade elevation route.

Fixtures are synthetic gmsh 2.2 meshes: the two-hex/two-tag mesh of
test_msh_to_yaml (library fill per physical tag) and a two-tet4 mesh
(--p_refine -> conforming tet10 twin).  The library tests read the
PACKAGED default opensg_solid/io/materials.yaml (the SG yamls' own
material-block dialect -- filling is a passthrough); the CLI tests must
emit yamls that pass check_filled when every tag and n_model are
supplied, keep FILL_IN placeholders when they are not, name every
available material on an unknown-name miss, and keep the console output
to the terse two-line shape.

Run:  pytest tests/msg_k_reference/test_materials_db.py -q   (opensg_2_0)
"""
import os

import pytest

MSH_2HEX = """$MeshFormat
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

MSH_2TET = """$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
1
3 1 "body"
$EndPhysicalNames
$Nodes
5
1 0 0 0
2 1 0 0
3 0 1 0
4 0 0 1
5 1 1 1
$EndNodes
$Elements
2
1 4 2 1 1 1 2 3 4
2 4 2 1 1 2 3 4 5
$EndElements
"""


@pytest.fixture()
def hex_msh(tmp_path):
    p = tmp_path / "two_hex.msh"
    p.write_text(MSH_2HEX)
    return str(p)


@pytest.fixture()
def tet_msh(tmp_path):
    p = tmp_path / "two_tet.msh"
    p.write_text(MSH_2TET)
    return str(p)


def test_packaged_library_loads_and_is_seeded():
    from opensg_solid.io.materials_db import load_library, load_materials

    lib = load_library()
    assert lib["path"].replace("\\", "/").endswith(
        "opensg_solid/io/materials.yaml")
    mats = lib["materials"]
    # the seeded cards, case-insensitive, aliases included; blocks speak
    # the SG yamls' own material dialect (elastic E/G/nu triples)
    al = mats["al"]
    assert al["elastic"]["E"] == [7.0e10] * 3
    assert al["elastic"]["nu"] == [0.3] * 3 and al["density"] == 2700.0
    assert mats["aluminum"] is al and mats["alu"] is al
    ply = mats["hc_ply"]
    assert ply["elastic"]["E"] == [108000.0, 8000.0, 8000.0]
    assert ply["elastic"]["G"] == [4000.0, 4000.0, 3000.0]
    assert ply["elastic"]["nu"] == [0.32, 0.32, 0.30]
    al69 = mats["al69"]
    assert al69["elastic"]["E"] == [69000.0] * 3
    assert mats["hc_core"] is al69
    # layups are informational descriptors, parsed but not solver-bound
    assert lib["layups"]["pm45_sym"] == [("HC_ply", 45.0), ("HC_ply", -45.0),
                                         ("HC_ply", -45.0), ("HC_ply", 45.0)]
    assert load_materials() is not None       # the {name: block} face


def test_spec_parse_resolve_unknown_and_xml_refusal():
    from opensg_solid.io.materials_db import (find_library, load_library,
                                              parse_spec, resolve_spec,
                                              to_solid_material)

    assert parse_spec("Al") == ("Al", None)
    assert parse_spec("HC_ply:-45") == ("HC_ply", -45.0)
    assert parse_spec("hc_ply:22.5") == ("hc_ply", 22.5)
    lib = load_library()
    blk, ang = resolve_spec("hc_PLY:-45", lib)      # case-insensitive
    assert blk["name"] == "HC_ply" and ang == -45.0
    # to_solid_material is a near-passthrough of the yaml dialect
    m = to_solid_material(blk, name="skin", angle=ang)
    assert m["name"] == "skin" and m["E"][0] == 108000.0
    assert m["G"] == [4000.0, 4000.0, 3000.0] and m["angle"] == -45.0
    iso = to_solid_material(lib["materials"]["al"])
    assert abs(iso["G"][0] - 7.0e10 / 2.6) < 1.0e-3   # G = E/(2(1+nu))
    with pytest.raises(ValueError, match="unknown material 'unobtanium'"):
        resolve_spec("unobtanium", lib)
    # ... and the miss NAMES the available cards
    with pytest.raises(ValueError, match="Al"):
        resolve_spec("unobtanium", lib)
    # the XML schema is dropped: an explicit .xml path is refused clearly
    with pytest.raises(ValueError, match="YAML now"):
        find_library("old_library.xml")


def test_cli_full_coverage_emits_filled_yaml(hex_msh, capsys):
    from opensg.cli import main
    from opensg_solid.io.msh_to_yaml import check_filled
    from opensg_solid.io.sg_input import read_opensg_yaml
    from opensg_solid.sg_mesh import read_yaml_header

    rc = main(["msh_to_yaml", hex_msh, "--mat1", "Al69",
               "--mat2", "HC_ply:45", "--n_model", "3", "--refined", "0"])
    assert rc == 0
    out = capsys.readouterr().out
    # the terse two-part shape: ONE transformation line + one mat line
    # per tag; the packaged default library path is NOT printed
    assert "two_hex.msh -> two_hex.yaml   (hex8 linear;" in out
    assert "  mat 1 <- Al69" in out and "  mat 2 <- HC_ply:45" in out
    assert "library:" not in out and "NOT RUNNABLE" not in out
    yml = os.path.splitext(hex_msh)[0] + ".yaml"
    assert check_filled(yml) is None
    hdr = read_yaml_header(yml)               # mesh_order must be TOLERATED
    assert hdr["n_model"] == 3 and hdr["refined"] == 0
    assert hdr["mesh_order"] == "linear"
    sg = read_opensg_yaml(yml)
    assert len(sg["nodes"]) == 12 and len(sg["cells"]) == 2
    angles = sorted(float(b.get("angle", 0.0))
                    for b in sg["materials"].values())
    assert angles == [0.0, 45.0]              # the :45 suffix landed


def test_cli_partial_coverage_keeps_placeholders(hex_msh, capsys):
    from opensg.cli import main
    from opensg_solid.io.msh_to_yaml import FILL, check_filled

    rc = main(["msh_to_yaml", hex_msh, "--mat1", "Al"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "NOT RUNNABLE YET" in out and "skin" in out
    assert "n_model still a FILL_IN placeholder" in out
    yml = os.path.splitext(hex_msh)[0] + ".yaml"
    txt = open(yml).read()
    # covered tag filled, uncovered tag and n_model still placeholders
    assert "70000000000.000000" in txt        # Al's E landed on set `core`
    assert ("n_model: %s_N_MODEL" % FILL) in txt
    assert ("%s_E1" % FILL) in txt            # `skin` still a placeholder
    msg = check_filled(yml)
    assert msg is not None and "--mat<TAG>" in msg  # names the library fix


def test_cli_unknown_material_fails_cleanly(hex_msh, capsys):
    from opensg.cli import main

    assert main(["msh_to_yaml", hex_msh, "--mat1", "unobtanium",
                 "--mat2", "Al"]) == 1
    out = capsys.readouterr().out
    assert "FAILED" in out and "unknown material" in out and "HC_ply" in out


def test_cli_p_refine_elevates_then_converts(tet_msh, capsys):
    from opensg.cli import main
    from opensg_solid.io.msh_to_yaml import check_filled
    from opensg_solid.io.sg_input import read_opensg_yaml
    from opensg_solid.sg_mesh import read_yaml_header

    rc = main(["msh_to_yaml", tet_msh, "--p_refine", "--mat1", "Al",
               "--n_model", "3"])
    assert rc == 0
    out = capsys.readouterr().out
    # the chain rides INLINE on the one transformation line; the helper's
    # own progress prints stay silent on this path
    assert ("two_tet.msh -> two_tet_quad.msh -> two_tet_quad.yaml"
            "   (tet10 quadratic; 14 nodes / 2 elements)") in out
    assert "  mat 1 <- Al" in out
    assert "linear_msh_to_quad" not in out
    quad = os.path.splitext(tet_msh)[0] + "_quad.msh"
    yml = os.path.splitext(tet_msh)[0] + "_quad.yaml"
    assert os.path.exists(quad) and os.path.exists(yml)
    assert check_filled(yml) is None
    assert read_yaml_header(yml)["mesh_order"] == "quadratic"
    sg = read_opensg_yaml(yml)
    # 2 tet10 cells; 5 corners + 9 unique edges = 14 nodes
    assert len(sg["cells"]) == 2 and len(sg["cells"][0]) == 10
    assert len(sg["nodes"]) == 14
    # an already-quadratic input is refused (cubic not supported yet)
    assert main(["msh_to_yaml", quad, "--p_refine"]) == 1
    assert "ALREADY QUADRATIC" in capsys.readouterr().out
