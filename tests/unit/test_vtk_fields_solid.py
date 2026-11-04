"""The solid D route's .vtk carries the displacement (since 2.1.1).

`opensg <yaml> D` writes <base>_dehom.vtk with the strains Eps_*, the
stresses Sig_* and mat_id, and now also U1 U2 U3 (SCALARS) and VECTORS
U.  U is the per-element mean of the Gauss values, like every other
array, and stays in the SG axes on both frames.  The gates, on a 2-D
beam SG, a 2-D plate SG and a 3-D solid SG under --global and
--material: the strain and stress arrays equal the per-element mean of
the .txt Gauss table; U1-U3 equal the per-element mean of the .U and
the _elemental.U rows; VECTORS U holds them in the .vtk point axes
(and warps an isotropic section as nu says); every value is finite;
the titles say whether U holds the .ff rigid motion; meshio reads the
same arrays when it is importable; the point-cloud fallback writes U
per Gauss point.
"""
import os
import struct

import numpy as np
import pytest

assert struct.calcsize("P") == 8

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import beam_sg_fixtures as fx  # noqa: E402

OLD = (["Eps_%s" % c for c in ("xx", "yy", "zz", "yz", "xz", "xy")]
       + ["Sig_%s" % c for c in ("xx", "yy", "zz", "yz", "xz", "xy")])
MATS = {1: fx.iso(70e9, 0.33),
        2: fx.ortho(fx.GLASS["E"], fx.GLASS["G"], fx.GLASS["nu"],
                    angle=30.0)}
RIGID = "u: [1.0e-4, 2.0e-4, -1.0e-4]\ntheta: [0.01, -0.02, 0.005]\n"


# ------------------------------------------------------------------ inputs
def hex8_box(n, size=1.0):
    """Structured hex8 mesh of the cube [0, size]^3, n elements per side.

    Out: nodes (V, 3); cells (E, 8) 0-based, gmsh hex8 order (bottom face
         counter-clockwise, then the top face)."""
    g = np.linspace(0.0, size, n + 1)
    Z, Y, X = np.meshgrid(g, g, g, indexing="ij")
    nodes = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    nid = lambda i, j, k: (k * (n + 1) + j) * (n + 1) + i  # noqa: E731
    cells = []
    for k in range(n):
        for j in range(n):
            for i in range(n):
                cells.append([nid(i, j, k), nid(i + 1, j, k),
                              nid(i + 1, j + 1, k), nid(i, j + 1, k),
                              nid(i, j, k + 1), nid(i + 1, j, k + 1),
                              nid(i + 1, j + 1, k + 1),
                              nid(i, j + 1, k + 1)])
    return nodes, np.asarray(cells, int)


def two_layers(nodes, cells, col, cut):
    """mat_id 1 below `cut` (element centroid along column col), 2 above."""
    c = np.asarray(nodes)[np.asarray(cells)][:, :, col].mean(1)
    return np.where(c > cut, 2, 1)


def build(case, folder, rigid=True):
    """The yaml and its .ff of one case: 'beam' (2-D quad9 section,
    n_model 1, beam force), 'plate' (2-D quad9 plate SG, n_model 2) or
    'solid' (3-D hex8 cube, n_model 3), both strain driven.  rigid=False
    writes the state as `epsilon_bar:` in the yaml and no .ff (the
    fluctuation-only route).

    Out: yaml path."""
    os.makedirs(folder, exist_ok=True)
    if case == "beam":
        nodes, cells = fx.rect_quad9(0.4, 0.2, 8, 4)
        mid, kw = two_layers(nodes, cells, 1, 0.0), dict(n_model=1)
        ff, eps = "1: [1.0e5, 1.0e3, 2.0e3, 1.0e3, 1.0e3, 1.0e3]\n", \
            "[1.0e-3, 0.0, 0.0, 2.0e-3, 1.0e-3, -1.0e-3]"
    elif case == "plate":
        nodes, cells = fx.rect_quad9(1.0, 0.1, 6, 4, center=(0.5, 0.0))
        mid, kw = two_layers(nodes, cells, 1, 0.0), dict(n_model=2,
                                                         refined=0)
        ff = "0: [1.0e-3, -5.0e-4, 2.0e-4, 0.01, -0.02, 0.005]\n"
        eps = "[1.0e-3, -5.0e-4, 2.0e-4, 0.01, -0.02, 0.005]"
    else:
        nodes, cells = hex8_box(3)
        mid, kw = two_layers(nodes, cells, 2, 0.5), dict(n_model=3,
                                                         refined=0)
        ff = "0: [1.0e-3, -5.0e-4, 2.0e-4, 1.0e-4, -2.0e-4, 3.0e-4]\n"
        eps = "[1.0e-3, -5.0e-4, 2.0e-4, 1.0e-4, -2.0e-4, 3.0e-4]"
    p = fx.write_yaml(os.path.join(folder, case + ".yaml"), nodes, cells,
                      mid, MATS, **kw)
    if rigid:
        with open(p[:-5] + ".ff", "w") as f:
            f.write(ff + RIGID)
    else:
        txt = open(p).read().replace("nodes:\n",
                                     "epsilon_bar: %s\nnodes:\n" % eps, 1)
        open(p, "w").write(txt)
    return p


# ----------------------------------------------------------------- readers
def read_vtk(path):
    """A legacy ASCII .vtk parsed here: title line, the data section kind
    (CELL_DATA | POINT_DATA), its row count and {name: array} for every
    SCALARS (n,) and VECTORS (n, 3) block, in file order."""
    with open(path) as f:
        L = f.read().split("\n")
    kind = n = None
    data = {}
    i = 0
    while i < len(L):
        t = L[i].split()
        if t and t[0] in ("CELL_DATA", "POINT_DATA"):
            kind, n = t[0], int(t[1])
        elif t and t[0] == "SCALARS":
            assert L[i + 1].strip() == "LOOKUP_TABLE default"
            data[t[1]] = np.array([float(v) for v in L[i + 2:i + 2 + n]])
            i += 1 + n
        elif t and t[0] == "VECTORS":
            data[t[1]] = np.array([[float(v) for v in L[i + 1 + q].split()]
                                   for q in range(n)])
            i += n
        i += 1
    return L[1], kind, n, data


def rows(path):
    """The numeric rows of a .txt (one header line) or .SM/.EM/.U file."""
    if path.endswith(".txt"):
        return np.loadtxt(path, skiprows=1, ndmin=2)
    return np.loadtxt(path, comments="#", ndmin=2)


def point_axes(n_sg):
    """The SG axis of each .vtk point slot: the points hold the n_sg SG
    coordinates first, so VECTORS U = U[:, point_axes(n_sg)]."""
    return [(3 - n_sg + k) % 3 for k in range(3)]


def cell_centroids(path):
    """The corner centroid of every cell of a legacy ASCII .vtk."""
    with open(path) as f:
        L = f.read().split("\n")
    i = next(k for k, s in enumerate(L) if s.startswith("POINTS"))
    n = int(L[i].split()[1])
    P = np.array([[float(v) for v in s.split()] for s in L[i + 1:i + 1 + n]])
    j = next(k for k, s in enumerate(L) if s.startswith("CELLS"))
    E = int(L[j].split()[1])
    conn = [[int(v) for v in s.split()[1:]] for s in L[j + 1:j + 1 + E]]
    return np.array([P[c].mean(0) for c in conn])


def check_files(base, rigid):
    """The gates on one D run's <base>_dehom files; returns the .vtk data."""
    title, kind, E, data = read_vtk(base + ".vtk")
    assert kind == "CELL_DATA"
    assert list(data)[:12] == OLD
    assert [k for k in data if k.startswith("U")] == ["U1", "U2", "U3", "U"]
    assert title.startswith("OpenSG dehom, exploded SG mesh")
    assert title.endswith(")") and len(title) < 256
    tag = ("fluctuation + the .ff macro rigid motion" if rigid
           else "fluctuation only, no macro rigid motion")
    assert ("U1-U3 in the SG axes = " + tag) in title
    for k, v in data.items():
        assert np.isfinite(v).all(), k
    # the strain and stress arrays: the per-element mean of the .txt
    tab = rows(base + ".txt")
    Q = tab.shape[0] // E
    assert Q * E == tab.shape[0]
    d = tab.shape[1] - 13
    gauss = tab[:, 1 + d:].reshape(E, Q, 12).mean(1)
    for j, k in enumerate(OLD):
        scale = max(np.abs(gauss[:, j]).max(), 1e-300)
        assert np.abs(data[k] - gauss[:, j]).max() <= 2e-6 * scale, k
    # U1-U3: the per-element mean of the .U, the _elemental.U rows exactly
    U = np.column_stack([data["U1"], data["U2"], data["U3"]])
    ug = rows(base + ".U")[:, 3:6].reshape(E, Q, 3).mean(1)
    assert np.abs(U - ug).max() <= 2e-6 * np.abs(ug).max()
    assert np.array_equal(U, rows(base + "_elemental.U")[:, 3:6])
    # VECTORS U: the same values in the point axes (SG coordinates lead)
    assert np.array_equal(data["U"], U[:, point_axes(d)])
    assert np.abs(U).max() > 0.0
    with open(base + ".U") as f:
        head = f.readline()
    assert ("TOTAL displacement" in head) == rigid
    assert ("FLUCTUATION-ONLY" in head) == (not rigid)
    return data


@pytest.mark.parametrize("case", ["beam", "plate", "solid"])
def test_vtk_carries_u_on_both_frames(tmp_path, case):
    """Every case, --global then --material, the .ff with a rigid motion:
    the gates of check_files; U is the same on both frames (never
    rotated) while the material-frame stresses differ; the per-element
    strain file _elemental.EM labels its shears 2E12 2E13 2E23 and holds
    the mean of the .EM rows."""
    from opensg.cli import main
    p = build(case, str(tmp_path))
    base = p[:-5] + "_dehom"
    assert main([p, "D", "--global"]) == 0
    g = check_files(base, rigid=True)
    em = rows(base + "_elemental_global.EM")
    with open(base + "_elemental_global.EM") as f:
        hdr = f.read().split("\n")[2]
    assert hdr.split()[4:] == ["E11[-]", "E22[-]", "E33[-]", "2E12[-]",
                               "2E13[-]", "2E23[-]"]
    E = len(g["U1"])
    gm = rows(base + "_global.EM")[:, 3:].reshape(E, -1, 6).mean(1)
    assert np.abs(em[:, 3:] - gm).max() <= 2e-6 * np.abs(gm).max()
    assert main([p, "D", "--material"]) == 0
    m = check_files(base, rigid=True)
    for k in ("U1", "U2", "U3", "U"):
        assert np.array_equal(m[k], g[k]), k
    assert np.abs(m["Sig_xx"] - g["Sig_xx"]).max() > 0.0   # 30 deg plies


def test_vtk_u_without_ff_is_fluctuation(tmp_path):
    """No .ff (epsilon_bar in the yaml): nothing is added to U, and the
    titles say fluctuation only."""
    from opensg.cli import main
    p = build("solid", str(tmp_path), rigid=False)
    assert main([p, "D", "--global"]) == 0
    check_files(p[:-5] + "_dehom", rigid=False)


@pytest.mark.parametrize("case", ["beam", "plate", "solid"])
def test_vtk_rigid_part_is_what_the_ff_adds(tmp_path, case):
    """The .ff rigid motion: U(with u, theta) - U(without) is u + (C - I)
    y at every Gauss point, y the x y z columns of the .U itself (a beam
    section or a 2-D plate SG (0, mesh x, mesh y); 2.1.0 used (mesh x,
    mesh y, 0)), and its per-element mean in the .vtk U1 U2 U3."""
    from opensg.cli import main
    a = build(case, str(tmp_path / "a"))
    b = build(case, str(tmp_path / "b"))
    with open(a[:-5] + ".ff") as f:
        state = f.readline()
    with open(b[:-5] + ".ff", "w") as f:
        f.write(state)
    for p in (a, b):
        assert main([p, "D", "--global"]) == 0
    th = np.array([0.01, -0.02, 0.005])
    C = np.eye(3) + np.array([[0.0, -th[2], th[1]], [th[2], 0.0, -th[0]],
                              [-th[1], th[0], 0.0]])
    Ua, Ub = rows(a[:-5] + "_dehom.U"), rows(b[:-5] + "_dehom.U")
    assert np.array_equal(Ua[:, :3], Ub[:, :3])
    want = np.array([1.0e-4, 2.0e-4, -1.0e-4]) + Ua[:, :3] @ (C - np.eye(3)).T
    scale = np.abs(Ua[:, 3:]).max()
    assert np.abs((Ua[:, 3:] - Ub[:, 3:]) - want).max() <= 2e-6 * scale
    ua, ub = [np.column_stack([v["U%d" % i] for i in (1, 2, 3)]) for v in
              (read_vtk(q[:-5] + "_dehom.vtk")[3] for q in (a, b))]
    E = len(ua)
    assert np.abs((ua - ub) - want.reshape(E, -1, 3).mean(1)).max() \
        <= 2e-6 * scale


def test_vtk_vector_u_warps_in_the_point_axes(tmp_path):
    """VECTORS U is the displacement in the .vtk point axes.  An isotropic
    beam section under pure extension eps: x + U[:, 0] = x (1 - nu eps)
    along the point x (y2) and the point y (y3), up to a constant, and
    the third slot holds U1 (no warping)."""
    from opensg.cli import main
    nu, eps = 0.3, 1.0e-3
    nodes, cells = fx.rect_quad9(0.4, 0.2, 8, 4)
    p = fx.write_yaml(str(tmp_path / "iso.yaml"), nodes, cells,
                      np.ones(len(cells), int), {1: fx.iso(70e9, nu)},
                      n_model=1)
    txt = open(p).read().replace(
        "nodes:\n", "epsilon_bar: [%g, 0.0, 0.0, 0.0, 0.0, 0.0]\nnodes:\n"
        % eps, 1)
    open(p, "w").write(txt)
    assert main([p, "D", "--global"]) == 0
    vtk = p[:-5] + "_dehom.vtk"
    data = read_vtk(vtk)[3]
    c = cell_centroids(vtk)
    A = np.column_stack([np.ones(len(c)), c[:, 0], c[:, 1]])
    for k in (0, 1):                      # the point x and y: contraction
        coef = np.linalg.lstsq(A, data["U"][:, k], rcond=None)[0]
        assert abs(coef[1 + k] + nu * eps) <= 1e-4 * nu * eps, (k, coef)
        assert abs(coef[2 - k]) <= 1e-4 * nu * eps, (k, coef)
    assert np.array_equal(data["U"][:, 2], data["U1"])
    assert np.abs(data["U1"]).max() <= 1e-6 * np.abs(data["U"]).max()


def test_vtk_meshio_reads_the_arrays(tmp_path):
    """meshio (when importable) reads the 12 + 3 scalar arrays and the
    vector U as CELL_DATA of the same cells."""
    meshio = pytest.importorskip("meshio")
    from opensg.cli import main
    p = build("beam", str(tmp_path))
    assert main([p, "D", "--global"]) == 0
    vtk = p[:-5] + "_dehom.vtk"
    _t, _k, E, data = read_vtk(vtk)
    m = meshio.read(vtk)
    assert sum(len(c.data) for c in m.cells) == E
    for k in OLD + ["mat_id", "U1", "U2", "U3", "U"]:
        v = np.concatenate([np.asarray(b) for b in m.cell_data[k]])
        assert np.allclose(v.reshape(data[k].shape), data[k],
                           rtol=1e-6, atol=0.0), k


def test_cloud_fallback_writes_u_per_gauss_point(tmp_path, monkeypatch):
    """An element type without a VTK cell map writes the point cloud:
    POINT_DATA per Gauss point, U1-U3 the .U rows, VECTORS U the same
    values in the point axes."""
    from opensg_solid import sg_dehom
    from opensg_solid.sg_dehom import dehom_fields, export_gauss
    p = build("plate", str(tmp_path))
    r = fx.homogenize(p)
    G, S, U = dehom_fields(r, np.array([1e-3, -5e-4, 2e-4, 0.01, -0.02,
                                        0.005]))
    monkeypatch.setattr(sg_dehom, "_vtk_corner_cell", lambda n, N: None)
    pre = str(tmp_path / "cloud")
    export_gauss(r, G, S, pre, U_eqd=U)
    title, kind, n, data = read_vtk(pre + ".vtk")
    assert kind == "POINT_DATA" and n == np.asarray(U).reshape(-1, 3).shape[0]
    assert "fluctuation only" in title and title.endswith(")")
    uu = rows(pre + ".U")[:, 3:6]
    Uv = np.column_stack([data["U1"], data["U2"], data["U3"]])
    d = rows(pre + ".txt").shape[1] - 13          # the Coord_ columns
    assert np.array_equal(Uv[:, point_axes(d)], data["U"])
    assert np.abs(Uv - uu).max() <= 1e-6 * np.abs(uu).max()
    assert list(data)[:12] == OLD
