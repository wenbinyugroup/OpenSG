"""Unit tests: the pyNuMAD dialect route (`opensg pynumad <yaml> <st-id>`).

The width-based TE-reinforcement placements must be IN the laminate (a
plain windIO-v1 reader drops them), st-id must map 0-based indices to the
blade's own stations, and the route must stay LEAN: the VABS-layout
<tag>.out record is the ONLY artifact; --xml and --view are accepted for
compatibility and IGNORED (no XML, no PNGs).

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


def test_pynumad_bare_defaults_to_all():
    from opensg_shell.cli import main

    # st-id omitted -> "all"; a missing blade file proves the parse accepted
    # the bare form and reached the existence check (never argparse usage)
    with pytest.raises(SystemExit, match="no such file"):
        main(["pynumad", "nope_missing.yaml"])


def test_pynumad_cli_lean_and_optins(tmp_path):
    from opensg_shell.cli import main

    # no prefix flag on this route: names derive from the blade file stem
    tag = "IEA-15-240-RWT_r1000"
    out = str(tmp_path / "st")
    rc = main(["pynumad", BLADE, "9", "--out", out])
    assert rc == 0
    # the in-memory route's ONLY artifact: the VABS-layout .out record,
    # carrying the elastic_properties_mb cross-check block and the OML
    # reference (the dialect default) in its banner
    kout = os.path.join(out, tag + ".out")
    assert os.path.exists(kout)
    txt = open(kout).read()
    assert "Cross-check vs the blade file" in txt
    assert "reference=oml" in txt
    # nothing else: no station yaml, no xml/, no ABDG, no abd/ (the 1-D
    # yaml emission was the deprecated windio machinery's job)
    assert not os.path.exists(os.path.join(out, tag + "_shell.yaml"))
    assert not os.path.exists(os.path.join(out, "xml"))
    assert not os.path.exists(os.path.join(out, tag + "_shell_ABDG.out"))
    assert not os.path.exists(os.path.join(out, "abd"))

    # --xml / --view are accepted for compatibility and IGNORED
    out2 = str(tmp_path / "st_optin")
    rc = main(["pynumad", BLADE, "9", "--out", out2, "--xml", "--view"])
    assert rc == 0
    assert os.path.exists(os.path.join(out2, tag + ".out"))
    assert not os.path.exists(os.path.join(out2, "xml"))
