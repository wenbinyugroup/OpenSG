"""Per-element frames on the D route, and the D output files.

Before the fix the beam D route multiplied the INVERSE-frame strain by the
pre-frame stiffness and labelled it global (sg_dehom.py:546-547, :630 at
f31b196; a PreVABS NACA 0012 closed F1 0.920, M1 0.00157), --material
ignored per-element frames (it tested a key no reader sets), and every
--material run wrote the `_global` file names.  The gates: a framed section
equals its baked frameless twin (frames applied once, sigma = C_ess
Gamma), the ply-axis field satisfies S_ply = C0 G_ply with the material's
UNROTATED stiffness, and the file names follow the frame.
"""
import os
import struct

import numpy as np
import pytest

assert struct.calcsize("P") == 8

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import beam_sg_fixtures as fx  # noqa: E402
from opensg_solid.sg_dehom import (  # noqa: E402
    dehom_fields, export_gauss, material_frame_fields)

FF6 = np.array([1e5, 1e3, 2e3, 1e3, 1e3, 1e3])
SEED = 20261103


def _coupled_mesh():
    return fx.rect_quad9(0.4, 0.2, 16, 8, center=(-0.13, 0.07))


def _delta_F_cl(r, FF):
    Qm = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, -1.0], [1.0, 0.0]])
    A = np.asarray(r["C_eff_EB"], float)
    Ai = np.linalg.inv(A)
    P = np.eye(4) - Qm @ Qm.T
    return (-A @ P @ Ai @ np.asarray(r["B_tim"], float) @ Ai @ Qm
            @ FF[[1, 2]])


@pytest.fixture(scope="module")
def framed_pair(tmp_path_factory):
    """The T6 section (no block angle) with a seeded random proper frame
    per element, and its frameless twin baked from the constants."""
    d = str(tmp_path_factory.mktemp("framed"))
    nodes, cells = _coupled_mesh()
    E = len(cells)
    mats = {1: fx.ortho(**fx.GLASS)}
    mid = np.ones(E, int)
    M = fx.random_frames(E, SEED)
    rf = fx.homogenize(fx.write_yaml(os.path.join(d, "framed.yaml"), nodes,
                                     cells, mid, mats, frames=M))
    tw_m, tw_id = fx.baked_twin(mats, mid, M)
    rt = fx.homogenize(fx.write_yaml(os.path.join(d, "twin.yaml"), nodes,
                                     cells, tw_id, tw_m))
    return rf, rt, M


def test_beam_dehom_frames_equal_baked_twin(framed_pair):
    """T7: the framed section and its baked twin have the same 6x6 and the
    same SG-global Gamma and sigma, and the framed field obeys the T6
    resultant identities."""
    rf, rt, M = framed_pair
    Cf, Ct = np.asarray(rf["C_eff"]), np.asarray(rt["C_eff"])
    assert np.abs(Cf - Ct).max() <= 1e-12 * np.abs(Ct).max()
    Gf, Sf, _ = dehom_fields(rf, beam_force=FF6)
    Gt, St, _ = dehom_fields(rt, beam_force=FF6)
    assert np.abs(Sf - St).max() <= 1e-12 * np.abs(St).max()
    assert np.abs(Gf - Gt).max() <= 1e-12 * np.abs(Gt).max()
    R = fx.resultants(rf, Sf)
    dF = _delta_F_cl(rf, FF6)
    print("T7: Delta F_cl =", dF)
    assert np.abs(dF).max() > 1e-6 * np.abs(FF6).max()     # coupled case
    tol = 1e-10 * np.abs(FF6).max()
    assert np.abs(R[[1, 2]] - FF6[[1, 2]]).max() <= tol
    assert np.abs(R[[0, 3, 4, 5]] - (FF6[[0, 3, 4, 5]] + dF)).max() <= tol


def _ply_identities(Gg, Sg, Gm, Sm, M_e, th_e, C0_e):
    """S_ply = C0 G_ply and S_ply = T_sig(P(theta) M) sigma_g at every
    point; arrays flat (P, 6) with per-point frame / angle / C0."""
    PM = np.einsum("pij,pjk->pik",
                   np.stack([fx.ply_rotation(t) for t in th_e]), M_e)
    S_ref = np.einsum("pij,pj->pi", fx.voigt_T(PM), Sg)
    G_ref = np.einsum("pij,pj->pi", fx.strain_T(PM), Gg)
    smax = np.abs(Sm).max()
    assert np.abs(Sm - S_ref).max() <= 1e-12 * smax
    assert np.abs(Gm - G_ref).max() <= 1e-12 * np.abs(Gm).max()
    assert np.abs(Sm - np.einsum("pij,pj->pi", C0_e, Gm)).max() \
        <= 1e-12 * smax


def test_beam_dehom_material_frame(tmp_path):
    """T8a: random frames with block angles {0, 30, -45} on three
    materials; every eighth angle-0 element keeps the identity frame:
    there S_ply == sigma_g bit for bit, and on framed elements --material
    differs from --global."""
    nodes, cells = _coupled_mesh()
    E = len(cells)
    angs = (0.0, 30.0, -45.0)
    mats = {i + 1: fx.ortho(**fx.GLASS, angle=a) for i, a in enumerate(angs)}
    mid = 1 + (np.arange(E) % 3)
    M = fx.random_frames(E, SEED + 1)
    ident = (mid == 1) & (np.arange(E) % 8 == 0)
    M[ident] = np.eye(3)
    r = fx.homogenize(fx.write_yaml(str(tmp_path / "plies.yaml"), nodes,
                                    cells, mid, mats, frames=M))
    Gg, Sg, _ = dehom_fields(r, beam_force=FF6)
    Gm, Sm = material_frame_fields(Gg, Sg, r)
    Qn = Sg.shape[1]
    rep = lambda a: np.repeat(a, Qn, axis=0)              # noqa: E731
    C0 = np.stack([fx.stiffness(mats[int(m)]) for m in mid])
    _ply_identities(Gg.reshape(-1, 6), Sg.reshape(-1, 6), Gm.reshape(-1, 6),
                    Sm.reshape(-1, 6), rep(M), rep(np.array(
                        [angs[int(m) - 1] for m in mid])), rep(C0))
    assert ident.sum() > 0
    assert np.array_equal(Sm[ident], Sg[ident])
    assert np.array_equal(Gm[ident], Gg[ident])
    fr = ~ident
    assert np.abs(Sm[fr] - Sg[fr]).max() > 1e-3 * np.abs(Sg).max()


def _plate_mixed():
    """A 2-D plate cell [0, 1] x [-0.5, 0.5] (thickness last): two quad4
    in the lower half, four tri3 in the upper half."""
    n = lambda i, j: 3 * j + i                             # noqa: E731
    nodes = np.array([[0.5 * i, -0.5 + 0.5 * j, 0.0]
                      for j in range(3) for i in range(3)])
    cells = [[n(0, 0), n(1, 0), n(1, 1), n(0, 1)],
             [n(1, 0), n(2, 0), n(2, 1), n(1, 1)],
             [n(0, 1), n(1, 1), n(1, 2)], [n(0, 1), n(1, 2), n(0, 2)],
             [n(1, 1), n(2, 1), n(2, 2)], [n(1, 1), n(2, 2), n(1, 2)]]
    return nodes, cells


def _write_mixed(path, nodes, cells, mid, mats, frames):
    """write_yaml for ragged (mixed) cells."""
    fx.write_yaml(path, nodes, [[0]] * len(cells), mid, mats, n_model=2,
                  refined=0, frames=frames)
    txt = open(path).read().split("\n")
    i0 = txt.index("cells:")
    body = ["- [%s]" % ", ".join(str(v) for v in c) for c in cells]
    txt[i0 + 1:i0 + 1 + len(cells)] = body
    with open(path, "w") as f:
        f.write("\n".join(txt))
    return path


def test_plate_dehom_material_frame_mixed_and_api(tmp_path):
    """T8b/c: the flat (mixed tri3 + quad4) branch with file frames, and a
    single-batch plate whose frames come only through elem_rotation= of
    the API (stored in r, so --material rotates into them)."""
    nodes, cells = _plate_mixed()
    E = len(cells)
    angs = (0.0, 30.0)
    mats = {i + 1: fx.ortho(**fx.GLASS, angle=a) for i, a in enumerate(angs)}
    mid = 1 + (np.arange(E) % 2)
    M = fx.random_frames(E, SEED + 2)
    r = fx.homogenize(_write_mixed(str(tmp_path / "mixed.yaml"), nodes,
                                   cells, mid, mats, M))
    assert isinstance(r["x_end"], list)                   # the flat branch
    eps = np.array([1e-3, -2e-4, 5e-4, 1e-2, -3e-3, 2e-3])
    Gg, Sg, _ = dehom_fields(r, eps)
    Gm, Sm = material_frame_fields(Gg, Sg, r)
    ip = np.concatenate([np.repeat(np.asarray(idx), ph.shape[0])
                         for idx, ph in zip(r["batch_idx"], r["phi_qn"])])
    C0 = np.stack([fx.stiffness(mats[int(m)]) for m in mid])
    _ply_identities(Gg, Sg, Gm, Sm, M[ip],
                    np.array([angs[int(m) - 1] for m in mid])[ip], C0[ip])

    # (c) a frameless quad4 plate file, frames through the API only
    n = lambda i, j: 5 * j + i                             # noqa: E731
    nq = np.array([[0.25 * i, -0.5 + 0.25 * j, 0.0]
                   for j in range(5) for i in range(5)])
    cq = [[n(i, j), n(i + 1, j), n(i + 1, j + 1), n(i, j + 1)]
          for j in range(4) for i in range(4)]
    Eq = len(cq)
    Mq = fx.random_frames(Eq, SEED + 3)
    pq = fx.write_yaml(str(tmp_path / "plate_api.yaml"), nq, cq,
                       np.ones(Eq, int), {1: fx.ortho(**fx.GLASS)},
                       n_model=2, refined=0)
    er = Mq.reshape(Eq, 9)                                 # solver order
    rq = fx.homogenize(pq, elem_rotation=er)
    assert np.array_equal(np.asarray(rq["elem_rotation"]), er)
    Gg, Sg, _ = dehom_fields(rq, eps)
    Gm, Sm = material_frame_fields(Gg, Sg, rq)
    Qn = Sg.shape[1]
    C0 = np.repeat(fx.stiffness(fx.ortho(**fx.GLASS))[None], Eq * Qn, 0)
    _ply_identities(Gg.reshape(-1, 6), Sg.reshape(-1, 6), Gm.reshape(-1, 6),
                    Sm.reshape(-1, 6), np.repeat(Mq, Qn, 0),
                    np.zeros(Eq * Qn), C0)


def test_beam_dehom_refuses_non_rotation_frames(tmp_path):
    """T8d: one frame row scaled by 1.01 is not a rotation: homogenization
    keeps its law, the D route refuses (no output)."""
    nodes, cells = _coupled_mesh()
    E = len(cells)
    M = fx.random_frames(E, SEED + 4)
    M[5, 1] *= 1.01
    r = fx.homogenize(fx.write_yaml(str(tmp_path / "bad.yaml"), nodes, cells,
                                    np.ones(E, int),
                                    {1: fx.ortho(**fx.GLASS)}, frames=M))
    with pytest.raises(ValueError, match="not proper rotations"):
        dehom_fields(r, beam_force=FF6)
    G = np.zeros(r["dehomo_data"]["dV_q"].shape + (6,))
    with pytest.raises(ValueError, match="not proper rotations"):
        material_frame_fields(G, G, r)


def test_dehom_frames_written_to_6_decimals(tmp_path):
    """T8e: rows printed with 6 decimals (opensg_shell's msh_to_yaml
    writes %.6f) are a rotation off by |E_ij| <= 5e-7, so |M M^T - I| <=
    2 sqrt(3) 5e-7 + 3 (5e-7)^2: the D route accepts them; rows with 5
    decimals that exceed that bound are refused."""
    from opensg_solid.sg_dehom import _FRAME_TOL
    assert _FRAME_TOL == 2.0 * np.sqrt(3.0) * 5e-7 + 3.0 * 5e-7 ** 2
    nodes, cells = _coupled_mesh()
    E = len(cells)
    M = fx.random_frames(E, SEED + 5)
    dev = lambda A: np.abs(np.einsum("eij,ekj->eik", A, A)  # noqa: E731
                           - np.eye(3)).max()
    mats = {1: fx.ortho(**fx.GLASS)}
    for dec, ok in ((6, True), (5, False)):
        Md = np.round(M, dec)
        r = fx.homogenize(fx.write_yaml(str(tmp_path / ("f%d.yaml" % dec)),
                                        nodes, cells, np.ones(E, int), mats,
                                        frames=Md))
        print("T8e: %d decimals, max |M M^T - I| = %.4g" % (dec, dev(Md)))
        if ok:
            assert 1e-8 < dev(Md) <= _FRAME_TOL
            G, S, _ = dehom_fields(r, beam_force=FF6)
            material_frame_fields(G, S, r)
        else:
            assert dev(Md) > _FRAME_TOL
            with pytest.raises(ValueError, match="not proper rotations"):
                dehom_fields(r, beam_force=FF6)


def test_material_frame_uses_the_angles_argument(tmp_path):
    """T8f: plate_homo_2d(angles=...) rotates each material's C by the
    argument, so --material undoes that angle (it read the blocks' only):
    S_ply = C0 G_ply with the unrotated C0, as for a block angle.  A
    type-2 block without `angle` is named in ply_angles_as_given."""
    nodes, cells = _coupled_mesh()
    E = len(cells)
    mats = {1: fx.ortho(**fx.GLASS), 2: fx.ortho(**fx.GLASS)}   # no angle
    mid = 1 + (np.arange(E) % 2)
    r = fx.homogenize(fx.write_yaml(str(tmp_path / "ang.yaml"), nodes, cells,
                                    mid, mats), angles=[30.0, -45.0])
    assert r["ply_angles"] == {1: 30.0, 2: -45.0}
    assert r["ply_angles_as_given"] == []
    Gg, Sg, _ = dehom_fields(r, beam_force=FF6)
    Gm, Sm = material_frame_fields(Gg, Sg, r)
    Qn = Sg.shape[1]
    rep = lambda a: np.repeat(a, Qn, axis=0)              # noqa: E731
    C0 = np.stack([fx.stiffness(mats[int(m)]) for m in mid])
    _ply_identities(Gg.reshape(-1, 6), Sg.reshape(-1, 6), Gm.reshape(-1, 6),
                    Sm.reshape(-1, 6), rep(np.tile(np.eye(3), (E, 1, 1))),
                    rep(np.array([(30.0, -45.0)[int(m) - 1] for m in mid])),
                    rep(C0))
    t2 = {1: {"type": 2, "C": fx.stiffness(fx.ortho(**fx.GLASS)).tolist()}}
    r2 = fx.homogenize(fx.write_yaml(str(tmp_path / "t2.yaml"), nodes, cells,
                                     np.ones(E, int), t2))
    assert r2["ply_angles"] == {1: 0.0} and r2["ply_angles_as_given"] == [1]


# ---------------------------------------------------------------------- T9
def _read_txt(path):
    with open(path) as f:
        head = f.readline().rstrip("\n").split("\t")
    data = np.loadtxt(path, skiprows=1)
    return head, data


def test_beam_dehom_cli_files_and_symptom(tmp_path, capsys):
    """T9: `opensg <yaml> D` on the nu = 0 rectangle with `1: [0, 1000, 0,
    0, 0, 0]`: --global writes <base>_dehom_global.SM/.EM, --material the
    bare <base>_dehom.SM/.EM; the log names the recovery; max Sig_xy is
    1.5 V/A = 3000 Pa (v2.1.0: 2.96e-11 Pa) and the printed table equals
    the in-memory field to its %.6e print.  An API export without a frame
    is SG-global and is named so (example 8's call)."""
    from opensg.cli import main
    nodes, cells = fx.rect_quad9(1.0, 0.5, 40, 20)
    p = fx.write_yaml(str(tmp_path / "rect.yaml"), nodes, cells,
                      np.ones(len(cells), int), {1: fx.iso(200e9, 0.0)})
    with open(str(tmp_path / "rect.ff"), "w") as f:
        f.write("1: [0.0, 1000.0, 0.0, 0.0, 0.0, 0.0]\n")
    base = str(tmp_path / "rect_dehom")
    assert main([p, "D", "--global"]) == 0
    out = capsys.readouterr().out
    assert "generalized Timoshenko, F' = R1 F" in out
    assert "expected resultants [F1 F2 F3 M1 M2 M3] = [" in out
    assert "+ Delta F_cl on [F1 M1 M2 M3]" in out
    assert (" points -> %s.txt / .vtk / _global.SM / _global.EM / .U /"
            " _elemental_global.SM / _elemental.U (global frame)"
            % base) in out
    for ext in (".SM", ".EM"):
        assert os.path.exists(base + "_global" + ext)
        assert not os.path.exists(base + ext)
    head, tab = _read_txt(base + ".txt")
    assert main([p, "D", "--material"]) == 0
    capsys.readouterr()
    for ext in (".SM", ".EM"):
        assert os.path.exists(base + ext)
    sxy = tab[:, head.index("Sig_xy")]
    print("T9: max Sig_xy = %.6f Pa" % np.abs(sxy).max())
    assert abs(np.abs(sxy).max() / 3000.0 - 1.0) <= 1e-3

    r = fx.homogenize(p)
    G, S, U = dehom_fields(r, beam_force=np.array([0, 1e3, 0, 0, 0, 0.0]))
    sig = tab[:, head.index("Sig_xx"):head.index("Sig_xx") + 6]
    assert np.abs(sig - S.reshape(-1, 6)).max() \
        <= 1e-6 * np.abs(S).max()
    api = str(tmp_path / "api_dehom")
    export_gauss(r, G, S, api, U_eqd=U)
    assert os.path.exists(api + "_global.SM")
    assert not os.path.exists(api + ".SM")
    with pytest.raises(ValueError):
        export_gauss(r, G, S, api, frame="SG axes")
    with pytest.raises(ValueError):                   # names both frames
        export_gauss(r, G, S, api, frame="ply axes; U stays SG-global")
    # a 2.1.0-style free-text note naming ONE frame is read by its words,
    # with a note; the other frame's file next to the prefix is reported
    capsys.readouterr()
    export_gauss(r, G, S, api, frame="ply axes (material)")
    out = capsys.readouterr().out
    assert "read as material by its words" in out
    assert "%s_global.SM (the SG-global-frame name) is also next" % api in out
    assert os.path.exists(api + ".SM")
    export_gauss(r, G, S, api, frame="SG-global axes")
    assert "read as global by its words" in capsys.readouterr().out


def test_beam_dehom_cli_refusals_are_one_line(tmp_path):
    """The D-route refusals reach the command line as SystemExit with the
    reason (no traceback): a refined: 0 beam under F2, and a frame that is
    not a rotation; the refined: 0 macro-state line says what is used."""
    from opensg.cli import main
    nodes, cells = fx.rect_quad9(0.4, 0.2, 4, 2)
    E = len(cells)
    p0 = fx.write_yaml(str(tmp_path / "cl.yaml"), nodes, cells,
                       np.ones(E, int), {1: fx.iso(70e9, 0.3)}, refined=0)
    with open(str(tmp_path / "cl.ff"), "w") as f:
        f.write("1: [0.0, 1000.0, 0.0, 0.0, 0.0, 0.0]\n")
    with pytest.raises(SystemExit, match=r"^refused: refined: 0"):
        main([p0, "D"])
    M = fx.random_frames(E, SEED + 6)
    M[1, 0] *= 1.01
    pb = fx.write_yaml(str(tmp_path / "bad.yaml"), nodes, cells,
                       np.ones(E, int), {1: fx.ortho(**fx.GLASS)}, frames=M)
    with open(str(tmp_path / "bad.ff"), "w") as f:
        f.write("1: [0.0, 1000.0, 0.0, 0.0, 0.0, 0.0]\n")
    with pytest.raises(SystemExit, match="not proper rotations"):
        main([pb, "D"])
