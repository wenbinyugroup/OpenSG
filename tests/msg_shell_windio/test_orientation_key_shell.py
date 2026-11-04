"""T10, shell part: the one frame-block name `orientation` in opensg_shell.

Every msg-shell loader of the per-element frame block reads `orientation:`
and its OpenSG 1.x spelling `elementOrientations:` alike
(opensg_solid.io.sg_keys.normalize_orientation_key, called right after the
yaml load), refuses a file that carries both, and every shell writer writes
`orientation:` only.  The key name is a spelling, not data, so every gate
here is EXACT (==): the loader arrays, the boundary bundle, the written
boundary rings and the Timoshenko 6x6 of the two spellings are identical
bit for bit.

Inputs: the shipped IEA-22 station ring (examples/OpenSG_shell/
1_get_beam_props_from_shell_cross_section/iea_s10_shell.yaml, 310 line
elements), the [45/-45] tube segment examples/OpenSG_shell/
5_taper_segment_aperiodic_boundaries/meshes/seg_aniso_hR0.2.yaml (quads),
its end ring boundary_seg_iso_hR0.1_boundaries_L.yaml, a two-triangle gmsh
mesh written here, and the IEA-22 windIO blade at r = 0.5 for the pynumad
writer.  Every legacy or both-keys file is a tmp copy of a shipped yaml with
only its key line changed (or the block repeated under the legacy name).

Run:  pytest tests/msg_shell_windio/test_orientation_key_shell.py -q   (env opensg_2_0)
"""
import os
import shutil

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SHELL = os.path.join(ROOT, "examples", "OpenSG_shell")
RING = os.path.join(SHELL, "1_get_beam_props_from_shell_cross_section",
                    "iea_s10_shell.yaml")
RING_TIMO = os.path.join(SHELL, "1_get_beam_props_from_shell_cross_section",
                         "iea_s10_shell_Timo.out")
SEG = os.path.join(SHELL, "5_taper_segment_aperiodic_boundaries", "meshes",
                   "seg_aniso_hR0.2.yaml")
END_RING = os.path.join(SHELL, "5_taper_segment_aperiodic_boundaries", "meshes",
                        "boundary_seg_iso_hR0.1_boundaries_L.yaml")
WINDIO = os.path.join(SHELL, "windio", "IEA-22-280-RWT.yaml")
LEGACY = "elementOrientations"


def _lines(path):
    with open(path) as f:
        return f.read().split("\n")


def _write(path, lines):
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


def _legacy_copy(src, dst):
    """src with its `orientation:` key line renamed to the legacy spelling."""
    lines = _lines(src)
    assert lines.count("orientation:") == 1 and LEGACY + ":" not in lines, src
    lines[lines.index("orientation:")] = LEGACY + ":"
    return _write(dst, lines)


def _both_copy(src, dst):
    """src carrying its frame block twice: as `orientation:` and, appended at
    the end, under the legacy name (the rows continue while a line starts
    with '-' or ' ')."""
    lines = _lines(src)
    k = lines.index("orientation:")
    m = k + 1
    while m < len(lines) and lines[m][:1] in ("-", " "):
        m += 1
    tail = [] if lines[-1] else [lines.pop()]
    return _write(dst, lines + [LEGACY + ":"] + lines[k + 1:m] + tail)


def _same(a, b, where="."):
    """Recursive EXACT equality of loader outputs (dict / list / ndarray)."""
    if isinstance(a, dict):
        assert isinstance(b, dict) and list(a) == list(b), where
        for k in a:
            _same(a[k], b[k], "%s/%s" % (where, k))
    elif isinstance(a, (list, tuple)):
        assert isinstance(b, (list, tuple)) and len(a) == len(b), where
        for i, (x, y) in enumerate(zip(a, b)):
            _same(x, y, "%s[%d]" % (where, i))
    elif hasattr(a, "shape") or hasattr(b, "shape"):        # numpy or JAX array
        a, b = np.asarray(a), np.asarray(b)
        assert a.shape == b.shape and a.dtype == b.dtype, where
        assert np.array_equal(a, b), where
    else:
        assert a == b, where


def _ring_loaders():
    from opensg_shell.sg_mesh import load_ring, load_ring_ref
    from opensg_shell.sg_mesh_out import load_shell_mesh
    from opensg_shell.pynumad.sg_props import _ring_arrays
    from opensg_shell.kernels.msg_mesh import element_e3_from_yaml, load_yaml
    from opensg_shell.kernels.orient_plot import _load
    return {
        "load_ring": load_ring,
        "load_ring(center_ref)": lambda p: load_ring(p, center_ref=True),
        "load_ring_ref(oml)": lambda p: load_ring_ref(p, ref="oml"),
        "load_ring_ref(center)": lambda p: load_ring_ref(p, ref="center"),
        "load_shell_mesh": load_shell_mesh,
        "_ring_arrays": _ring_arrays,
        "element_e3_from_yaml": element_e3_from_yaml,
        "orient_plot._load": _load,
        # the refined: 0 (Hermite) route's loader: it reads no frames, but
        # checks the block's name as every other shell route does
        "msg_mesh.load_yaml": load_yaml,
    }


def test_shipped_shell_yamls_spell_orientation():
    """The 22 shipped msg-shell yamls carry `orientation:` once and the
    legacy key nowhere (only their key line changed)."""
    found = []
    for dp, _dn, fn in os.walk(SHELL):
        for f in fn:
            if not f.endswith(".yaml"):
                continue
            p = os.path.join(dp, f)
            with open(p) as fh:
                txt = fh.read()
            if "\nmsg: shell" not in "\n" + txt:
                continue
            found.append(p)
            assert LEGACY not in txt, p
            assert txt.count("\norientation:") == 1, p
    assert len(found) == 22, len(found)


@pytest.mark.parametrize("name", list(_ring_loaders()))
def test_ring_loaders_read_both_spellings(tmp_path, capsys, name):
    """Every ring loader returns identical arrays for `orientation:` and the
    legacy spelling; only the legacy read prints the one-line NOTE."""
    load = _ring_loaders()[name]
    old = _legacy_copy(RING, str(tmp_path / "iea_s10_legacy.yaml"))
    capsys.readouterr()
    new_out = load(RING)
    assert "NOTE" not in capsys.readouterr().out
    old_out = load(old)
    note = [ln for ln in capsys.readouterr().out.splitlines() if "NOTE" in ln]
    assert len(note) == 1 and LEGACY in note[0] and "orientation" in note[0], note
    _same(new_out, old_out, name)


@pytest.mark.parametrize("name", list(_ring_loaders()))
def test_ring_loaders_refuse_both_spellings(tmp_path, name):
    """A ring carrying the frame block under both names is refused by every
    ring loader (which block holds the frames would be a guess)."""
    both = _both_copy(RING, str(tmp_path / "iea_s10_both.yaml"))
    with pytest.raises(ValueError, match="twice"):
        _ring_loaders()[name](both)


def test_segment_loaders_refuse_both_spellings(tmp_path):
    """The 3-D shell loaders (segment yaml, 3-D shell SG) refuse both keys
    at load time, before any mesh check."""
    from opensg_shell.sg_mesh import load_segment_yaml, extract
    from opensg_shell.sg_homo import shell_sg3d
    both = _both_copy(SEG, str(tmp_path / "seg_both.yaml"))
    with pytest.raises(ValueError, match="twice"):
        load_segment_yaml(both)
    with pytest.raises(ValueError, match="twice"):
        extract(both, str(tmp_path / "seg_both.npz"), plot=False)
    with pytest.raises(ValueError, match="twice"):
        shell_sg3d(both)


def test_segment_extract_same_bundle_and_rings_both_spellings(tmp_path):
    """extract() of the segment in both spellings: the same .npz bundle and
    the same written end rings, which spell `orientation:` only."""
    from opensg_shell.sg_mesh import extract, load_segment_yaml, load_ring_ref
    new_dir, old_dir = tmp_path / "new", tmp_path / "old"
    new_dir.mkdir(); old_dir.mkdir()
    old = _legacy_copy(SEG, str(old_dir / "seg.yaml"))
    shutil.copy(SEG, str(new_dir / "seg.yaml"))
    _same(load_segment_yaml(str(new_dir / "seg.yaml")), load_segment_yaml(old),
          "load_segment_yaml")
    bn = extract(str(new_dir / "seg.yaml"), str(new_dir / "seg_b.npz"),
                 write_yaml=True, plot=False)
    bo = extract(old, str(old_dir / "seg_b.npz"), write_yaml=True, plot=False)
    with np.load(bn) as zn, np.load(bo) as zo:
        assert sorted(zn.files) == sorted(zo.files)
        for k in zn.files:
            _same(zn[k], zo[k], k)
    for side in ("L", "R"):
        f = "boundary_seg_b_%s.yaml" % side
        with open(str(new_dir / f)) as fh:
            tn = fh.read()
        with open(str(old_dir / f)) as fh:
            to = fh.read()
        assert tn == to, f
        assert "\norientation:" in tn and LEGACY not in tn, f
        R = load_ring_ref(str(new_dir / f))           # the written ring reads back
        assert R["cells"].shape[0] == R["re3"].shape[0] > 0


def test_msh_to_yaml_writes_orientation(tmp_path):
    """helper.msh_to_yaml: the yaml text and the returned dict name the frame
    block `orientation` only, and the file reads back to the same rows (to
    the writer's %.6f, i.e. within half a unit of the 6th decimal)."""
    from opensg_shell.helper.msh_to_yaml import convert
    from opensg_shell.sg_mesh import load_segment_yaml
    msh = tmp_path / "two_tri.msh"
    msh.write_text("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
                   "$Nodes\n4\n1 0 0 0\n2 1 0 0\n3 1 1 0\n4 0 1 0\n$EndNodes\n"
                   "$Elements\n2\n1 2 2 1 1 1 2 3\n2 2 2 1 1 1 3 4\n$EndElements\n")
    out = str(tmp_path / "two_tri.yaml")
    r = convert(str(msh), thickness=0.001, out_path=out)
    assert "orientation" in r and LEGACY not in r
    with open(out) as fh:
        txt = fh.read()
    assert "\norientation:" in txt and LEGACY not in txt
    d = load_segment_yaml(out)
    np.testing.assert_allclose(np.asarray(d["orientation"], float),
                               np.asarray(r["orientation"], float),
                               rtol=0.0, atol=5e-7)


def test_pynumad_shell_yaml_writes_orientation(tmp_path):
    """pynumad.emit_shell_yaml (the blade route's station ring) writes
    `orientation:` only, and the ring loader reads it back."""
    from opensg_shell.pynumad import load_blade, build_cross_section, emit_shell_yaml
    from opensg_shell.sg_mesh import load_ring_ref
    cs = build_cross_section(load_blade(WINDIO), 0.5, mesh_size=0.02)
    out = str(tmp_path / "st_shell.yaml")
    info = emit_shell_yaml(cs, out, reference="oml")
    with open(out) as fh:
        txt = fh.read()
    assert "\norientation:" in txt and LEGACY not in txt
    R = load_ring_ref(out)
    assert R["cells"].shape[0] == R["re3"].shape[0] == info["n_elems"]


def test_ring_law_identical_both_spellings(tmp_path):
    """The Timoshenko 6x6 of the renamed shipped ring equals its legacy-key
    twin bit for bit, and still equals the shipped _Timo.out (written with
    the legacy key, --center) to that file's print precision: 8 significant
    digits, |dK_ij| <= 5e-8 |K_ij| on every term above 1e-9 max|K| (the
    terms below are round-off of order 1e-15 max|K|)."""
    from opensg_shell.sg_homo import build_rm_bundle
    (tmp_path / "new").mkdir(); (tmp_path / "old").mkdir()
    new = str(tmp_path / "new" / "iea_s10_shell.yaml")      # tmp copies: the bundle
    shutil.copy(RING, new)                                  # writes its step-1 cache
    old = _legacy_copy(RING, str(tmp_path / "old" / "iea_s10_shell.yaml"))
    Kn = np.asarray(build_rm_bundle(new, ref="center", abd_out=False)["Timo"])
    Ko = np.asarray(build_rm_bundle(old, ref="center", abd_out=False)["Timo"])
    assert np.array_equal(Kn, Ko)
    with open(RING_TIMO) as fh:
        rows = [ln.split() for ln in fh.read().split("Compliance")[0].splitlines()]
    Ks = np.array([[float(v) for v in r] for r in rows
                   if len(r) == 6 and r[0][-4:-3] in ("+", "-")])   # 1.2E+010 rows
    assert Ks.shape == (6, 6)
    big = np.abs(Ks) > 1e-9 * np.abs(Ks).max()
    assert np.all(np.abs(Kn - Ks)[big] <= 5e-8 * np.abs(Ks)[big])


def _run_cli(path):
    from opensg.cli import main
    rc = main([path])
    assert rc in (0, None), rc
    base = os.path.splitext(path)[0]
    with open(base + "_Timo.out") as fh:
        return [ln for ln in fh if "Time taken" not in ln]


def test_cli_ring_both_spellings_and_dialect(tmp_path, capsys):
    """`opensg <ring>` runs the shell engine on the end ring in both
    spellings, and without its `msg:` line (the mesh dialect decides, in
    either spelling): one identical _Timo.out every time."""
    runs = {}
    for tag in ("new", "legacy", "new_nomsg", "legacy_nomsg"):
        d = tmp_path / tag
        d.mkdir()
        p = str(d / "ring.yaml")
        if tag.startswith("legacy"):
            _legacy_copy(END_RING, p)
        else:
            shutil.copy(END_RING, p)
        if tag.endswith("nomsg"):
            lines = [ln for ln in _lines(p) if not ln.startswith("msg:")]
            _write(p, lines)
        capsys.readouterr()
        runs[tag] = _run_cli(p)
        out = capsys.readouterr().out
        assert " msg       : shell" in out, (tag, out[-2000:])
    for tag in ("legacy", "new_nomsg", "legacy_nomsg"):
        assert runs[tag] == runs["new"], tag


@pytest.mark.parametrize("refined", [1, 0])
def test_cli_ring_refuses_both_spellings_in_one_line(tmp_path, refined):
    """`opensg <ring>` on a ring carrying both names ends with the one-line
    reason (SystemExit, no traceback) on the Timoshenko route and on the
    refined: 0 (Hermite) route alike."""
    from opensg.cli import main
    p = _both_copy(END_RING, str(tmp_path / "ring_both.yaml"))
    if refined == 0:
        lines = _lines(p)
        k = [i for i, ln in enumerate(lines) if ln.startswith("refined:")]
        assert len(k) == 1, p
        lines[k[0]] = "refined: 0"
        _write(p, lines)
    with pytest.raises(SystemExit,
                       match="^refused: .*carries the per-element frame block twice"):
        main([p])
