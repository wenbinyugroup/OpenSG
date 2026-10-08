"""The laminate reference of an msg-shell SG is a RUN-TIME choice, never a yaml key.

    default            the contour is the OML, laminates stack inward
    --center / ref=    the contour is the laminate mid-surface: ABD shifted by t/2

Checked here, on an IEA-22 station emitted twice by the pynumad writer (once on
the OML, once center-offset):
  * the writer no longer emits a `reference:` key (a comment line instead);
  * `opensg <yaml>` runs at the OML and `opensg <yaml> --center` at the
    mid-surface, bit-identical to build_rm_bundle(ref=...);
  * --center changes ONLY the wall law (nodes, k22 untouched; B/D follow the
    parallel-axis formula), on the RM ring and on the classical route alike
    (the classical route no longer moves the nodes);
  * a leftover `reference:` key is ignored with a printed warning;
  * an unknown reference name is refused;
  * --mesh writes <base>_mesh.png + a gmsh-2.2 <base>.msh that parses back;
  * beam_props / dehom_station take ref= and record it.

Run:  pytest tests/msg_shell_windio/test_reference_cli.py -q   (env opensg_2_0)
"""
import os
import re
import shutil

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
WINDIO = os.path.join(ROOT, "examples", "OpenSG_shell", "windio", "IEA-22-280-RWT.yaml")
LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]


@pytest.fixture(scope="module")
def yamls(tmp_path_factory):
    from opensg_shell.pynumad import load_blade, build_cross_section, emit_shell_yaml
    tmp = tmp_path_factory.mktemp("refcli")
    blade = load_blade(WINDIO)
    cs = build_cross_section(blade, 0.5, mesh_size=0.02)
    yo = str(tmp / "st_oml_shell.yaml")
    yc = str(tmp / "st_center_shell.yaml")
    emit_shell_yaml(cs, yo, reference="oml")
    emit_shell_yaml(cs, yc, reference="center")
    return dict(tmp=tmp, oml=yo, center=yc)


def _copy(src, tmp, sub, name=None):
    d = tmp / sub
    d.mkdir(exist_ok=True)
    dst = str(d / (name or os.path.basename(src)))
    shutil.copy(src, dst)
    return dst


def _run(argv):
    from opensg_shell.cli import main
    rc = main(argv)
    assert rc in (0, None), rc


def _timo_out(path):
    """The 6x6 stiffness block of a SwiftComp-layout _Timo.out / _EB.out."""
    rows = []
    for ln in open(path):
        v = ln.split()
        try:
            f = [float(x) for x in v]
        except ValueError:
            continue
        if len(f) in (4, 6) and rows == [] or (rows and len(f) == len(rows[0])):
            rows.append(f)
        if rows and len(rows) == len(rows[0]):
            break
    return np.array(rows)


def test_writer_emits_no_reference_key(yamls):
    for key in ("oml", "center"):
        txt = open(yamls[key]).read()
        assert not re.search(r"^reference:", txt, re.M), key
        first = txt.splitlines()[0]
        assert first.startswith("# contour placed on the"), first
    assert "--center" in open(yamls["center"]).read().splitlines()[0]
    assert "outer mold line" in open(yamls["oml"]).read().splitlines()[0]
    from opensg_solid.sg_mesh import read_yaml_header
    assert read_yaml_header(yamls["center"]) == {"msg": "shell", "refined": 1}


def test_default_is_oml_and_center_is_a_flag(yamls):
    from opensg_shell.sg_homo import build_rm_bundle
    tmp = yamls["tmp"]
    y1 = _copy(yamls["center"], tmp, "cli_oml")
    _run([y1])
    K_cli_oml = _timo_out(y1[:-5] + "_Timo.out")
    y2 = _copy(yamls["center"], tmp, "cli_center")
    _run([y2, "--center"])
    K_cli_center = _timo_out(y2[:-5] + "_Timo.out")
    y3 = _copy(yamls["center"], tmp, "api_oml")
    K_api_oml = np.asarray(build_rm_bundle(y3)["Timo"])           # default = oml
    y4 = _copy(yamls["center"], tmp, "api_center")
    K_api_center = np.asarray(build_rm_bundle(y4, ref="center")["Timo"])
    assert np.allclose(K_cli_oml, K_api_oml, rtol=1e-7, atol=0)
    assert np.allclose(K_cli_center, K_api_center, rtol=1e-7, atol=0)
    # the two references are different laws on the same contour
    dGJ = abs(K_cli_center[3, 3] / K_cli_oml[3, 3] - 1.0)
    assert dGJ > 1e-3, "GJ center vs oml differs by only %.2e" % dGJ
    # the bundle records the choice
    assert build_rm_bundle(y4, ref="center")["ref"] == "center"
    assert build_rm_bundle(y3)["frac"] == 0.0


def test_center_shifts_the_wall_law_only(yamls):
    from opensg_shell.sg_mesh import load_ring_ref
    from opensg_shell.fe_jax.msg_materials import shift_abd_reference
    import yaml as _yaml
    Ro = load_ring_ref(yamls["center"], "oml")
    Rc = load_ring_ref(yamls["center"], "center")
    assert np.array_equal(Ro["rx"], Rc["rx"])                    # nodes untouched
    assert np.array_equal(np.asarray(Ro["k22"]), np.asarray(Rc["k22"]))
    d = _yaml.safe_load(open(yamls["center"]))
    for si, sec in enumerate(d["sections"]):
        h = sum(float(p[1]) for p in sec["layup"])
        Do = np.asarray(Ro["D_by"][si]); Dc = np.asarray(Rc["D_by"][si])
        A, B, D = Do[:3, :3], Do[:3, 3:], Do[3:, 3:]
        expl = np.block([[A, B - 0.5 * h * A],
                         [B - 0.5 * h * A, D - h * B + 0.25 * h * h * A]])
        assert np.allclose(Dc, shift_abd_reference(Do, 0.5 * h), rtol=1e-12, atol=0)
        assert np.allclose(Dc, expl, rtol=1e-9, atol=1e-9 * np.abs(Do).max())


def test_yaml_reference_key_is_ignored(yamls, capsys):
    tmp = yamls["tmp"]
    y_key = _copy(yamls["center"], tmp, "key")
    with open(y_key, "a") as f:
        f.write("reference: center\n")
    _run([y_key])
    out = capsys.readouterr().out
    assert "WARNING" in out and "IGNORED" in out and "--center" in out
    K_key = _timo_out(y_key[:-5] + "_Timo.out")
    y_nokey = _copy(yamls["center"], tmp, "nokey")
    _run([y_nokey])
    out = capsys.readouterr().out
    assert "WARNING" not in out
    K_nokey = _timo_out(y_nokey[:-5] + "_Timo.out")
    assert np.array_equal(K_key, K_nokey)                        # the key changed nothing
    # with --center the note is informational, not a warning
    y_key2 = _copy(yamls["center"], tmp, "key2")
    with open(y_key2, "a") as f:
        f.write("reference: center\n")
    _run([y_key2, "--center"])
    out = capsys.readouterr().out
    assert "note" in out and "WARNING" not in out


def test_unknown_reference_is_refused(yamls):
    from opensg_shell.sg_homo import build_rm_bundle
    from opensg_shell.sg_reference import norm_ref
    with pytest.raises(ValueError):
        norm_ref("mid")
    with pytest.raises(ValueError):
        build_rm_bundle(yamls["center"], ref="centre")
    assert norm_ref(None) == "oml" and norm_ref("CENTER") == "center"


def test_classical_route_never_moves_nodes(yamls):
    from opensg_shell.fe_jax.msg_hermite import solve_tw_from_yaml
    tmp = yamls["tmp"]
    a = solve_tw_from_yaml(yamls["center"], frac=0.0)
    b = solve_tw_from_yaml(yamls["center"], frac=0.5)
    assert np.array_equal(np.asarray(a["corners"]), np.asarray(b["corners"]))
    assert np.array_equal(np.asarray(a["k22"]), np.asarray(b["k22"]))
    assert not np.allclose(np.asarray(a["EB"]), np.asarray(b["EB"]), rtol=1e-3)
    # and through the CLI: refined 0 header, with and without --center
    txt = open(yamls["center"]).read().replace("refined: 1", "refined: 0", 1)
    y0 = str(tmp / "kl0" / "st_kl_shell.yaml"); os.makedirs(os.path.dirname(y0), exist_ok=True)
    y1 = str(tmp / "kl1" / "st_kl_shell.yaml"); os.makedirs(os.path.dirname(y1), exist_ok=True)
    open(y0, "w").write(txt); open(y1, "w").write(txt)
    _run([y0]); _run([y1, "--center"])
    E0 = _timo_out(y0[:-5] + "_EB.out"); E1 = _timo_out(y1[:-5] + "_EB.out")
    assert E0.shape == (4, 4) and E1.shape == (4, 4)
    sym = lambda M: 0.5 * (np.asarray(M) + np.asarray(M).T)   # the CLI writes the symmetrized law
    assert np.allclose(E0, sym(a["EB"]), rtol=1e-7, atol=0)
    assert np.allclose(E1, sym(b["EB"]), rtol=1e-7, atol=0)


def test_mesh_flag_writes_png_and_msh(yamls):
    import subprocess
    import sys
    import yaml as _yaml
    tmp = yamls["tmp"]
    y = _copy(yamls["center"], tmp, "mesh")
    _run([y, "--mesh", "--center"])
    base = y[:-5]
    listing = sorted(os.listdir(os.path.dirname(y)))
    assert os.path.exists(base + ".msh"), "msh missing; dir holds %s" % listing
    # tests/conftest.py turns savefig into a no-op for the whole pytest
    # session, so the png is checked in a fresh interpreter (the real CLI)
    y2 = _copy(yamls["center"], tmp, "mesh_png")
    out = subprocess.run([sys.executable, "-c",
                          "import sys; from opensg_shell.cli import main;"
                          " sys.exit(main(sys.argv[1:]) or 0)", y2, "--mesh", "--center"],
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    assert "mesh      : %s.msh + %s_mesh.png" % ((os.path.basename(y2)[:-5],) * 2) in out.stdout
    png = y2[:-5] + "_mesh.png"
    assert os.path.exists(png) and os.path.getsize(png) > 1000, sorted(os.listdir(os.path.dirname(y2)))
    d = _yaml.safe_load(open(y))
    lines = open(base + ".msh").read().splitlines()
    assert lines[:3] == ["$MeshFormat", "2.2 0 8", "$EndMeshFormat"]
    i = lines.index("$PhysicalNames")
    npn = int(lines[i + 1])
    names = [ln.split('"')[1] for ln in lines[i + 2:i + 2 + npn]]
    assert names == [s["elementSet"] for s in d["sections"]]
    i = lines.index("$Nodes"); assert int(lines[i + 1]) == len(d["nodes"])
    i = lines.index("$Elements"); ne = int(lines[i + 1])
    assert ne == len(d["elements"])
    rows = [ln.split() for ln in lines[i + 2:i + 2 + ne]]
    assert all(r[1] == "1" for r in rows)                        # 2-node lines
    # physical tag = the element's section (1-based), from sets
    sec_of = {}
    for k, grp in enumerate(d["sets"]["element"]):
        for lab in grp["labels"]:
            sec_of[int(lab)] = names.index(grp["name"]) + 1
    assert all(int(r[3]) == sec_of[int(r[0])] for r in rows)
    # the analysis still ran
    assert os.path.exists(base + "_Timo.out")


def test_beam_props_and_dehom_take_ref(yamls):
    from opensg_shell.pynumad import beam_props, dehom_station
    tmp = yamls["tmp"]
    y = _copy(yamls["center"], tmp, "props")
    P = beam_props(y, out_k=str(tmp / "props" / "st.K"), ref="center")
    assert P["bundle"]["ref"] == "center" and P["bundle"]["frac"] == 0.5
    assert "reference=center" in open(P["k_file"]).read()
    FF = np.array([5.0e6, 0.0, 0.0, 0.0, -6.0e7, 2.0e7])
    D = dehom_station(y, FF, n_depth=3, frame="plate", ref="center")
    assert D["bundle"]["frac"] == 0.5
    # recovery depths straddle the contour at the center reference
    assert D["z"].min() < 0.0 < D["z"].max()
    Po = beam_props(_copy(yamls["oml"], tmp, "props_oml"), out_k=str(tmp / "props_oml" / "st.K"))
    assert Po["bundle"]["ref"] == "oml"
    assert "reference=oml" in open(Po["k_file"]).read()
