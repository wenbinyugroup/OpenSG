"""Unit test: gmsh `.msh` -> solid SG yaml (io.msh_to_yaml), template mode.

The fixture is a synthetic two-hex gmsh 2.2 mesh with a $PhysicalNames
block, one physical tag per hex.  Without materials the converter must emit
a FILL_IN template whose sets are named from $PhysicalNames and whose
material entries carry the SAME names (the solid reader binds sets to
materials by name); check_filled must refuse the template and accept the
filled file; `opensg msh_to_yaml` is the CLI face of the same thing.

Run:  pytest tests/msg_k_reference -q      (env opensg_2_0)
"""
import os

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
