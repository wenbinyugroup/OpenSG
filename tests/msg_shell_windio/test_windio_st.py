"""Unit test: the one-shot windIO station bypass behind `opensg windio_st`.

station_timo (sg_props) fuses steps 1 + 2 for one station, so its .K must
reproduce the committed pipeline reference examples/windio/props/iea22_r0197.K
on the physical terms.  Gates: Timoshenko diagonals + mass matrix tight
(same mesh, same solver); full stiffness compared with an atol floor of
1e-3 * max|K| because the fast-ring dense-LU prints run-era noise of that
class in the PHYSICALLY ZERO couplings (documented trade-off).

Run:  pytest tests/msg_shell_windio -q      (env opensg_2_0)
"""
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
WINDIO = os.path.join(ROOT, "examples", "OpenSG_shell", "windio",
                      "IEA-22-280-RWT.yaml")
K_REF = os.path.join(ROOT, "examples", "windio", "props", "iea22_r0197.K")


def test_station_timo_matches_pipeline(tmp_path):
    from opensg_shell.windio import (blade_stations, load_blade, read_k_file,
                                     station_timo)

    # the committed reference was computed at the blade's OWN airfoil station
    # (r = 0.19670...), so use that exact value: reconstructing r = 0.1967
    # from the 4-decimal file tag is a DIFFERENT station and shifts every
    # interpolated quantity (chord, layup thickness) by ~1e-4 relative
    r = min(blade_stations(load_blade(WINDIO)), key=lambda v: abs(v - 0.197))
    P = station_timo(WINDIO, r, out_dir=str(tmp_path))
    assert P["tag"].endswith("_r0197")
    assert os.path.exists(P["yaml"]) and os.path.exists(P["k_file"])
    assert os.path.exists(os.path.splitext(P["yaml"])[0] + "_Timo.out")
    # terminal st-id contract: yaml + _Timo.out + .K ONLY
    assert not os.path.exists(os.path.splitext(P["yaml"])[0] + "_ABDG.out")
    assert not os.path.exists(os.path.join(str(tmp_path), "abd"))

    K, M = read_k_file(P["k_file"])
    Kr, Mr = read_k_file(K_REF)
    assert np.allclose(np.diag(K), np.diag(Kr), rtol=1e-6)
    assert np.allclose(M, Mr, rtol=1e-6, atol=1e-9 * np.abs(Mr).max())
    assert np.abs(K - Kr).max() < 1e-3 * np.abs(Kr).max()
    assert np.allclose(P["Timo"], P["Timo"].T)


def test_windio_st_cli(tmp_path):
    from opensg_shell.cli import main

    out = str(tmp_path / "st")
    rc = main(["windio_st", WINDIO, "1.0", "--out", out, "--prefix", "iea22"])
    assert rc == 0
    assert os.path.exists(os.path.join(out, "iea22_r1000_shell.yaml"))
    assert os.path.exists(os.path.join(out, "iea22_r1000.K"))
    assert not os.path.exists(os.path.join(out, "iea22_r1000_shell_ABDG.out"))
    assert not os.path.exists(os.path.join(out, "abd"))


def test_windio_st_bad_station():
    from opensg_shell.cli import main

    with pytest.raises(SystemExit, match=r"station r must be in \[0, 1\]"):
        main(["windio_st", WINDIO, "1.5"])


def test_windio_st_dispatch_missing_file():
    from opensg.cli import main

    with pytest.raises(SystemExit, match="no such file"):
        main(["windio_st", "nope_missing.yaml", "0.2"])
