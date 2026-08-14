"""Unit tests: the editable Blade object (pyNuMAD-style workflow).

Blade must load both dialects, its timo() must match the pynumad terminal
route digit-for-digit (same production path), and DEFINITION EDITS on the
object must propagate into the next homogenization without touching a yaml.

Run:  pytest tests/msg_shell_windio -q      (env opensg_2_0)
"""
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
BLADE = os.path.join(ROOT, "examples", "pynumad", "IEA-15-240-RWT.yaml")
WINDIO = os.path.join(ROOT, "examples", "OpenSG_shell", "windio",
                      "IEA-22-280-RWT.yaml")


def test_blade_load_and_views():
    from opensg_shell import Blade

    b = Blade(BLADE)
    assert b.dialect == "v1"
    assert len(b.stations) == 10
    for nm in ("Spar_Cap_SS", "TE_reinforcement_SS", "web0_filler"):
        assert nm in b.layers, nm
    assert "glass_triax" in b.materials or len(b.materials) == 11
    assert isinstance(b.chord, dict) and "values" in b.chord

    b22 = Blade(WINDIO)
    assert b22.dialect == "v2"
    assert len(b22.stations) == 16


def test_blade_timo_matches_pynumad_route(tmp_path):
    from opensg_shell import Blade
    from opensg_shell.pynumad import station_timo

    b = Blade(BLADE, workdir=str(tmp_path / "w"))
    R = b.timo(9)                                       # tip, fast
    P = station_timo(BLADE, "9", out_dir=str(tmp_path / "p"))
    assert np.allclose(np.diag(R["Timo"]), np.diag(P["Timo"]), rtol=1e-8)
    assert np.allclose(R["Mass"], P["Mass"], rtol=1e-8)
    assert R["k_file"].endswith(".out") and os.path.exists(R["k_file"])
    assert R["file_K"] is not None                      # cross-check present


def test_blade_edit_propagates(tmp_path):
    from opensg_shell import Blade

    b = Blade(BLADE, workdir=str(tmp_path / "w"))
    ea0 = float(b.timo(9)["Timo"][0, 0])
    # x4 so the pyNuMAD ply quantization cannot absorb the change at the
    # thin tip laminate (x2 of a sub-ply thickness rounds to the same count)
    b.scale_layer_thickness("Shell_skin", 4.0)
    b.update_blade()
    ea1 = float(b.timo(9)["Timo"][0, 0])
    assert ea1 > 1.2 * ea0                              # skin x4 -> EA up
    # material edit through the object: soften everything -> EA drops
    b2 = Blade(BLADE, workdir=str(tmp_path / "w2"))
    E = b2.materials["glass_triax"]["E"]
    b2.set_material("glass_triax", E=[0.5 * float(v) for v in E])
    ea2 = float(b2.timo(9)["Timo"][0, 0])
    assert ea2 < ea0


def test_opensg_blade_one_call(tmp_path):
    from opensg_shell import Blade, opensg_blade

    b = Blade(BLADE, workdir=str(tmp_path / "w"))
    K, M = opensg_blade(b, 9)
    assert K.shape == (6, 6) and M.shape == (6, 6)
    assert np.all(np.diag(K) > 0) and M[0, 0] > 0


def test_opensg_read_and_call(tmp_path):
    import opensg

    b = opensg.read(BLADE, workdir=str(tmp_path / "w"))
    K, M = b(9)                                         # blade(st) -> (K, M)
    assert K.shape == (6, 6) and M.shape == (6, 6)
    K2, _ = opensg.Blade(BLADE, workdir=str(tmp_path / "w2"))(9)
    assert np.allclose(K, K2, rtol=1e-12)


def test_section_mirror_edit_reset(tmp_path):
    from opensg_shell import Blade

    b = Blade(BLADE, workdir=str(tmp_path / "w"))
    K0, _ = b(4)                                        # caps are thick here
    S = b.section(4)
    Km, _ = b(4)                                        # override, UNEDITED
    assert np.allclose(K0, Km, rtol=1e-12, atol=0.0)    # exact mirror
    assert S.stacks["LP_SPAR"] and S.stacks["HP_SPAR"]
    S.scale_thickness(3.0, region="LP_SPAR")
    S.scale_thickness(3.0, region="HP_SPAR")
    K1, _ = b(4)
    assert K1[4, 4] > 1.15 * K0[4, 4]                   # EI2 tracks the caps
    assert abs(K1[5, 5] / K0[5, 5] - 1.0) < 0.30        # EI3 much less so
    assert b.section(4) is S                            # returns the live one
    S.reset()
    K2, _ = b(4)
    assert np.allclose(K0, K2, rtol=1e-12, atol=0.0)    # definition restored
