"""Unit tests: the editable Blade object (pyNuMAD-style workflow).

Coverage:
  - Blade loads both dialects; definition views are live references.
  - timo() matches the pynumad terminal route digit-for-digit.
  - Definition edits (layers, materials) propagate without touching a yaml.
  - opensg.read / callable blade / opensg_blade one-call interfaces.
  - Section: per-station override mirrors the baseline exactly, edits move
    the right stiffness channels, reset restores the definition.
  - pynumad.sg_homo (the standalone in-memory homogenizer): bit-parity
    with the file route, Section overrides honored, and the OWNERSHIP
    gates -- its homo-layer copies (ring_indep, laws, k22) must not drift
    from the core originals, and it must import only kernel layers.

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
    R = b.timo(9, artifacts=True)                       # tip; file route
    P = station_timo(BLADE, "9", out_dir=str(tmp_path / "p"))
    assert np.allclose(np.diag(R["Timo"]), np.diag(P["Timo"]), rtol=1e-8)
    assert np.allclose(R["Mass"], P["Mass"], rtol=1e-8)
    assert R["k_file"].endswith(".out") and os.path.exists(R["k_file"])
    assert R["file_K"] is not None                      # cross-check present
    # the DEFAULT route is in-memory: same numbers, no files, no bundle
    D = b.timo(9)
    assert D["k_file"] is None and D["yaml"] is None and D["bundle"] is None
    assert np.allclose(np.asarray(D["Timo"]), np.asarray(R["Timo"]),
                       rtol=1e-8, atol=1e-9 * np.abs(R["Timo"]).max())


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


def test_pynumad_sg_homo_bypass(tmp_path):
    """The standalone in-memory homogenizer must match the file route."""
    from opensg_shell import Blade
    from opensg_shell.pynumad.sg_homo import timo as timo_mem

    b = Blade(BLADE, workdir=str(tmp_path / "w"))
    K, M = timo_mem(b, 9)
    R = b.timo(9, artifacts=True)                       # yaml/file route
    Kf, Mf = np.asarray(R["Timo"]), np.asarray(R["Mass"])
    assert np.allclose(K, Kf, rtol=1e-8, atol=1e-9 * np.abs(Kf).max())
    assert np.allclose(M, Mf, rtol=1e-8, atol=1e-9 * np.abs(Mf).max())
    # no artifacts were produced by the in-memory call
    K4, M4, info = timo_mem(b, 4, full=True)
    assert info["mpus"] > 0 and K4[0, 0] > 0
    # honors a live Section override, same as the file route
    S = b.section(4)
    S.scale_thickness(2.0, region="LP_SPAR")
    S.scale_thickness(2.0, region="HP_SPAR")
    K4e, _ = timo_mem(b, 4)
    R4e = b.timo(4, artifacts=True)
    assert np.allclose(K4e, np.asarray(R4e["Timo"]), rtol=1e-8,
                       atol=1e-9 * np.abs(K4e).max())
    assert K4e[4, 4] > 1.1 * K4[4, 4]                   # the edit is in
    S.reset()


def test_pynumad_owns_homo_no_drift(tmp_path):
    """The pynumad-owned homo copies must equal the core originals on the
    SAME inputs -- the drift gate for the ownership split."""
    from opensg_shell import Blade
    from opensg_shell.pynumad import sg_homo as own
    from opensg_shell import sg_homo as core_homo
    from opensg_shell import sg_assembly as core_asm
    from opensg_shell import sg_materials as core_mat

    b = Blade(BLADE, workdir=str(tmp_path / "w"))
    r = b.resolve(9)
    A = own.station_arrays(b.cross_section(r), reference=b.reference)

    # laws: owned vs core, identical dicts
    D1, G1 = own._material_by_section(A["sections"], A["materials"],
                                      center_ref=False)
    D2, G2 = core_mat._material_by_section(A["sections"], A["materials"],
                                           center_ref=False)
    for si in range(len(A["sections"])):
        assert np.array_equal(np.asarray(D1[si]), np.asarray(D2[si]))
        assert np.array_equal(np.asarray(G1[si]), np.asarray(G2[si]))

    # k22: owned vs core, identical
    cen = A["rx"][A["cells"]].mean(1)
    k1 = own.compute_k22(cen, A["re2"], A["re3"], A["cells"])
    k2 = core_asm.compute_k22(cen, A["re2"], A["re3"], A["cells"])
    assert np.array_equal(k1, k2)

    # ring driver: owned vs core on the same arrays, identical 6x6
    D_by, G_by = own.ring_laws(A["sections"], A["materials"], b.reference)
    C1 = own.ring_indep(A["rx"], A["cells"], A["rsub"], A["re3"],
                        D_by, G_by, k1, 2, [0, 1])
    C2 = core_homo.ring_indep(A["rx"], A["cells"], A["rsub"], A["re3"],
                              D_by, G_by, k1, 2, [0, 1])
    d = np.abs(np.asarray(C1) - np.asarray(C2)).max()
    assert d <= 1e-12 * np.abs(C2).max()


def test_pynumad_sg_homo_kernel_only_imports():
    """Ownership gate: pynumad/sg_homo may import ONLY the kernel layers
    (sg_assembly element kernels, fe_jax msg_*, opensg_solid rm_plate_msg)
    -- never the core homo/materials/mesh drivers or windio."""
    import ast
    import opensg_shell.pynumad.sg_homo as m

    src = open(m.__file__.replace(".pyc", ".py")).read()
    banned = ("sg_homo", "sg_materials", "sg_mesh", "sg_dehom", "windio")
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ImportFrom) and node.module:
            mod = node.module
            if mod.startswith("opensg_shell.") or node.level > 0:
                leaf = mod.split(".")[-1] if mod else ""
                assert leaf not in banned and "windio" not in mod, \
                    "forbidden core import in pynumad.sg_homo: %s" % mod
        if isinstance(node, ast.Import):
            for a in node.names:
                assert "windio" not in a.name, a.name


def test_blade_and_pynumad_are_windio_free():
    """The Blade class and the whole pynumad package must not import the
    windio machinery (it is being deprecated)."""
    import opensg_shell.pynumad.sg_blade as blade_mod
    import opensg_shell.pynumad.sg_mesh as mesh_mod
    import opensg_shell.pynumad.sg_pynumad as pyn_mod
    import opensg_shell.pynumad.sg_homo as homo_mod

    for mod in (blade_mod, mesh_mod, pyn_mod, homo_mod):
        src = open(mod.__file__.replace(".pyc", ".py")).read()
        for line in src.splitlines():
            stripped = line.strip()
            if stripped.startswith(("from ", "import ")):
                assert ".windio" not in stripped and \
                    not stripped.startswith("from opensg_shell.windio"), \
                    "%s imports windio: %s" % (mod.__name__, stripped)


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
