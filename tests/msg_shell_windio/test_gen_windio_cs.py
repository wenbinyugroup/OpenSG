"""Unit test: the windIO cross-section generator behind `opensg gen_windio_cs`.

generate_cross_sections (sg_windio) at ONE station must reproduce the direct
emit_shell_yaml output byte for byte (the yaml carries no station tag, so the
CLI prefix cannot leak into the content), write the PreVABS XML byproduct by
default, and the CLI wrapper must parse
`gen_windio_cs <windio> --stations r --out DIR --prefix TAG --no-xml` and be
reachable through the unified `opensg` dispatcher.

Run:  pytest tests/msg_shell_windio -q      (env opensg_2_0)
"""
import filecmp
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
WDIR = os.path.join(ROOT, "examples", "OpenSG_shell", "windio")
WINDIO = os.path.join(WDIR, "IEA-22-280-RWT.yaml")


def test_generate_cross_sections_one_station(tmp_path):
    from opensg_shell.windio import (build_cross_section, emit_shell_yaml,
                                     generate_cross_sections, load_blade)

    out = str(tmp_path / "cs")
    R = generate_cross_sections(WINDIO, out_dir=out, stations=[0.1967],
                                plots=False, verbose=False)
    tag = "%s_r0197" % R["prefix"]
    ypath = os.path.join(out, tag + "_shell.yaml")
    assert R["yamls"] == [ypath] and os.path.exists(ypath)
    assert os.path.exists(R["dat"])
    # the PreVABS XML byproduct is ON by default
    xdir = os.path.join(out, "xml", tag)
    for f in (tag + ".xml", tag + ".dat", "materials.xml"):
        assert os.path.exists(os.path.join(xdir, f))
    # the emitted yaml equals the direct emit route byte for byte
    cs = build_cross_section(load_blade(WINDIO), 0.1967, mesh_size=0.01)
    ref = str(tmp_path / "ref_shell.yaml")
    emit_shell_yaml(cs, ref, reference="center")
    assert filecmp.cmp(ypath, ref, shallow=False)


def test_gen_windio_cs_cli(tmp_path):
    from opensg_shell.cli import main

    out = str(tmp_path / "cli_cs")
    rc = main(["gen_windio_cs", WINDIO, "--stations", "1.0",
               "--out", out, "--prefix", "iea22", "--no-xml"])
    assert rc == 0
    assert os.path.exists(os.path.join(out, "iea22_r1000_shell.yaml"))
    # PNG files are NOT asserted: tests/conftest.py no-ops Figure.savefig for
    # the whole suite (no plot files during pytest); the PNG route is covered
    # by the example/CLI runs outside pytest
    assert os.path.exists(os.path.join(out, "iea22_stations.dat"))
    assert not os.path.exists(os.path.join(out, "xml"))


def test_gen_windio_cs_dispatch_missing_file():
    from opensg.cli import main

    # the dispatcher must hand the token to the shell engine, whose own
    # missing-file message proves the route was taken
    with pytest.raises(SystemExit, match="no such file"):
        main(["gen_windio_cs", "nope_missing.yaml"])
