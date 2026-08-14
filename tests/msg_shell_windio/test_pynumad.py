"""Unit tests: the pyNuMAD dialect route (`opensg pynumad <yaml> <st-id>`).

The width-based TE-reinforcement placements must be IN the laminate (the
plain windio reader drops them), st-id must map 0-based indices to the
blade's own stations, and the st-id route must stay LEAN by default --
--xml opts into the PreVABS XML.  --view PNGs are exercised but not
asserted: tests/conftest.py no-ops Figure.savefig suite-wide.

Run:  pytest tests/msg_shell_windio -q      (env opensg_2_0)
"""
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
BLADE = os.path.join(ROOT, "examples", "pynumad", "IEA-15-240-RWT.yaml")


def test_width_placement_resolved():
    from opensg_shell.pynumad import load_blade_pynumad

    names = {L["name"]: L
             for L in load_blade_pynumad(BLADE).layers_at(0.3288)}
    for nm in ("TE_reinforcement_SS", "TE_reinforcement_PS"):
        assert nm in names, nm
        L = names[nm]
        assert 0.0 <= L["s"] < L["e"] <= 1.0


def test_station_id_resolution():
    from opensg_shell.pynumad import load_blade_pynumad, resolve_station

    b = load_blade_pynumad(BLADE)
    rs = b.stations()
    assert resolve_station(b, "0") == rs[0]
    assert resolve_station(b, "4") == pytest.approx(0.3288, abs=1e-4)
    assert resolve_station(b, "0.5") == 0.5
    with pytest.raises(SystemExit, match="out of range"):
        resolve_station(b, "99")


def test_pynumad_cli_lean_and_optins(tmp_path):
    from opensg_shell.cli import main

    # no prefix flag on this route: names derive from the blade file stem
    tag = "IEA-15-240-RWT_r1000"
    out = str(tmp_path / "st")
    rc = main(["pynumad", BLADE, "9", "--out", out])
    assert rc == 0
    # the pynumad station record is a VABS-layout .out (not .K), carrying
    # the elastic_properties_mb cross-check block
    kout = os.path.join(out, tag + ".out")
    assert os.path.exists(kout)
    assert "Cross-check vs the blade file" in open(kout).read()
    ypath = os.path.join(out, tag + "_shell.yaml")
    assert os.path.exists(ypath)
    # this dialect defaults the shell reference surface to the OML
    hdr = open(ypath).read(200)
    assert "reference: oml" in hdr
    # lean default: no xml/, no ABDG, no abd/
    assert not os.path.exists(os.path.join(out, "xml"))
    assert not os.path.exists(os.path.join(out, tag + "_shell_ABDG.out"))
    assert not os.path.exists(os.path.join(out, "abd"))

    out2 = str(tmp_path / "st_optin")
    rc = main(["pynumad", BLADE, "9", "--out", out2, "--xml", "--view"])
    assert rc == 0
    for f in (tag + ".xml", tag + ".dat", "materials.xml"):
        assert os.path.exists(os.path.join(out2, "xml", tag, f))
