"""Unit test: the blade sweep generator behind `opensg gen_windio_cs`.

generate_cross_sections (pynumad) runs every requested station fully IN
MEMORY: per station the VABS-layout <tag>.out record is the ONLY artifact
(no station yaml, no PreVABS XML), plus the three spanwise .dat tables at
the end.  One station must reproduce a direct station_timo call exactly
(one code path, deterministic solver), the .out must round-trip through
read_k_file, and the CLI wrapper must parse
`gen_windio_cs <windio> --stations r --out DIR --prefix TAG --no-xml`
(--no-xml accepted and ignored) and be reachable through the unified
`opensg` dispatcher.

Run:  pytest tests/msg_shell_windio -q      (env opensg_2_0)
"""
import glob
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
WDIR = os.path.join(ROOT, "examples", "OpenSG_shell", "windio")
WINDIO = os.path.join(WDIR, "IEA-22-280-RWT.yaml")


def test_generate_cross_sections_one_station(tmp_path):
    from opensg_shell.pynumad import (generate_cross_sections, read_k_file,
                                      station_timo)

    out = str(tmp_path / "cs")
    R = generate_cross_sections(WINDIO, out_dir=out, stations=[0.1967],
                                verbose=False)
    assert len(R) == 1
    P = R[0]
    stem = "IEA-22-280-RWT"
    assert P["tag"] == stem + "_r0197"
    assert P["k_file"] == os.path.join(out, P["tag"] + ".out")
    assert os.path.exists(P["k_file"])
    # in-memory contract: .out records + .dat tables and NOTHING else
    assert glob.glob(os.path.join(out, "*_shell.yaml")) == []
    assert not os.path.exists(os.path.join(out, "xml"))
    for t in ("_stations.dat", "_timo_by_r.dat", "_mass_by_r.dat"):
        assert os.path.exists(os.path.join(out, stem + t))
    # the .out record round-trips through the beam .K reader
    K, M = read_k_file(P["k_file"])
    assert np.allclose(K, P["Timo"], rtol=1e-8)
    assert np.allclose(M, P["Mass"], rtol=1e-8,
                       atol=1e-12 * np.abs(np.asarray(P["Mass"])).max())
    # the sweep equals the direct one-station route exactly (same code path)
    P2 = station_timo(WINDIO, "0.1967", out_dir=str(tmp_path / "one"))
    assert np.array_equal(np.asarray(P["Timo"]), np.asarray(P2["Timo"]))
    assert np.array_equal(np.asarray(P["Mass"]), np.asarray(P2["Mass"]))


def test_gen_windio_cs_cli(tmp_path):
    from opensg_shell.cli import main

    out = str(tmp_path / "cli_cs")
    rc = main(["gen_windio_cs", WINDIO, "--stations", "1.0",
               "--out", out, "--prefix", "iea22", "--no-xml"])
    assert rc == 0
    assert os.path.exists(os.path.join(out, "iea22_r1000.out"))
    assert not os.path.exists(os.path.join(out, "iea22_r1000_shell.yaml"))
    assert not os.path.exists(os.path.join(out, "xml"))
    for t in ("iea22_stations.dat", "iea22_timo_by_r.dat",
              "iea22_mass_by_r.dat"):
        assert os.path.exists(os.path.join(out, t))


def test_gen_windio_cs_dispatch_missing_file():
    from opensg.cli import main

    # the dispatcher must hand the token to the shell engine, whose own
    # missing-file message proves the route was taken
    with pytest.raises(SystemExit, match="no such file"):
        main(["gen_windio_cs", "nope_missing.yaml"])


def test_gen_windio_cs_reference_flag_removed():
    """The blade route is OML-only: --reference must be REJECTED (argparse
    usage error, exit 2), not silently accepted."""
    from opensg_shell.cli import main

    with pytest.raises(SystemExit) as e:
        main(["gen_windio_cs", WINDIO, "--reference", "oml"])
    assert e.value.code == 2
