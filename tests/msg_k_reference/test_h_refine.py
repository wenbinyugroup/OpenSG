"""h-refinement anchors: every supported cell type, linear and
quadratic, each gated by helper.verify_refinement.

The verifier is the point of this file.  Counting children only proves
the templates fired; it says nothing about whether the pieces still
FIT.  verify_refinement checks the four things a real bug breaks --
measure conservation, no coincident nodes, positive orientation, and
facet conformity (nothing shared by more than two cells, boundary
measure preserved) -- so every case below asserts it, and the
deliberately-broken meshes at the end prove the gate can FAIL.
"""
import itertools
import math

import numpy as np
import pytest

from opensg_solid.helper.h_refine_msh import (
    cell_measures, mesh_measure, msh_h_refine, read_msh, verify_refinement)
from opensg_solid.helper.make_linear_msh_to_quad import linear_msh_to_quad

# gmsh type -> (nodes/cell, children per split)
CASES = {2: (3, 4), 3: (4, 4), 4: (4, 8), 5: (8, 8)}
NAME = {2: "tri3", 3: "quad4", 4: "tet4", 5: "hex8",
        9: "tri6", 10: "quad9", 11: "tet10"}


def write_msh(path, etype, nodes, cells, phys=1):
    """Minimal gmsh 2.2 writer for the fixtures."""
    with open(path, "w") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        f.write('$PhysicalNames\n1\n%d %d "m"\n$EndPhysicalNames\n'
                % (3 if etype in (4, 5) else 2, phys))
        f.write("$Nodes\n%d\n" % len(nodes))
        for k, p in enumerate(nodes):
            f.write("%d %r %r %r\n" % (k + 1, float(p[0]), float(p[1]),
                                       float(p[2])))
        f.write("$EndNodes\n$Elements\n%d\n" % len(cells))
        for k, c in enumerate(cells):
            f.write("%d %d 2 %d %d %s\n"
                    % (k + 1, etype, phys, phys,
                       " ".join(str(n + 1) for n in c)))
        f.write("$EndElements\n")


def _grid(nx, ny, nz, skew=0.0):
    """(nx+1)(ny+1)(nz+1) node lattice on a GRADED, optionally sheared
    box -- non-uniform on purpose so a template error cannot hide
    behind equal spacings."""
    xs = np.linspace(0.0, 2.0, nx + 1) ** 1.3
    ys = np.linspace(0.0, 1.5, ny + 1) ** 1.1
    zs = np.linspace(0.0, 1.0, nz + 1)
    idx, pts = {}, []
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            for k, z in enumerate(zs):
                idx[(i, j, k)] = len(pts)
                pts.append((x + skew * z, y, z))
    return np.array(pts), idx


def fixture(etype, tmp_path, name=None):
    """A small multi-cell mesh of `etype`; returns its path."""
    p = str(tmp_path / ("%s.msh" % (name or NAME[etype])))
    if etype == 2:                                   # tri3
        nodes, idx = _grid(2, 2, 0)
        cells = []
        for i in range(2):
            for j in range(2):
                a, b = idx[(i, j, 0)], idx[(i + 1, j, 0)]
                c, d = idx[(i + 1, j + 1, 0)], idx[(i, j + 1, 0)]
                cells += [[a, b, c], [a, c, d]]
    elif etype == 3:                                 # quad4
        nodes, idx = _grid(2, 2, 0)
        cells = [[idx[(i, j, 0)], idx[(i + 1, j, 0)],
                  idx[(i + 1, j + 1, 0)], idx[(i, j + 1, 0)]]
                 for i in range(2) for j in range(2)]
    elif etype == 5:                                 # hex8
        nodes, idx = _grid(2, 2, 2, skew=0.25)
        cells = [[idx[(i, j, k)], idx[(i + 1, j, k)],
                  idx[(i + 1, j + 1, k)], idx[(i, j + 1, k)],
                  idx[(i, j, k + 1)], idx[(i + 1, j, k + 1)],
                  idx[(i + 1, j + 1, k + 1)], idx[(i, j + 1, k + 1)]]
                 for i in range(2) for j in range(2) for k in range(2)]
    elif etype == 4:                                 # tet4 (6 per box)
        nodes, idx = _grid(2, 2, 2, skew=0.25)
        cells = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    v = [idx[(i, j, k)], idx[(i + 1, j, k)],
                         idx[(i + 1, j + 1, k)], idx[(i, j + 1, k)],
                         idx[(i, j, k + 1)], idx[(i + 1, j, k + 1)],
                         idx[(i + 1, j + 1, k + 1)], idx[(i, j + 1, k + 1)]]
                    # the standard Freudenthal 6-tet split of a cube:
                    # conforming across boxes because it follows the
                    # 0-6 body diagonal in every box
                    for t in ((0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
                              (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)):
                        cells.append([v[n] for n in t])
        # the Freudenthal templates are conforming but not all
        # positively oriented -- normalize, so the fixture itself is a
        # valid mesh and the child-vs-parent volume signs line up
        q = nodes[np.asarray(cells)]
        neg = np.einsum("ei,ei->e", q[:, 1] - q[:, 0],
                        np.cross(q[:, 2] - q[:, 0], q[:, 3] - q[:, 0])) < 0
        cells = [c if not n else [c[0], c[1], c[3], c[2]]
                 for c, n in zip(cells, neg)]
    else:
        raise ValueError(etype)
    write_msh(p, etype, nodes, cells)
    return p


# ------------------------------------------------------------- linear cases
@pytest.mark.parametrize("etype", [2, 3, 4, 5],
                         ids=["tri3", "quad4", "tet4", "hex8"])
def test_linear_one_level(etype, tmp_path):
    """One split: child count, halved h, and the full verifier."""
    src = fixture(etype, tmp_path)
    m0 = read_msh(src)
    n0 = sum(len(b["conn"]) for b in m0["blocks"])
    r = msh_h_refine(src, verbose=False)
    _, kids = CASES[etype]
    assert r["n_elems"] == n0 * kids
    assert r["msh"].endswith("_h2.msh")
    assert r["h_max"] == pytest.approx(
        _edge_max(m0) / 2.0, rel=1e-12)
    v = verify_refinement(src, r["msh"], verbose=False)
    assert v["ok"], v["problems"]


def _edge_max(m):
    from opensg_solid.helper.h_refine_msh import _edge_stats
    return _edge_stats(m["xyz"], m["blocks"])[2]


@pytest.mark.parametrize("etype", [2, 3, 4, 5],
                         ids=["tri3", "quad4", "tet4", "hex8"])
def test_linear_two_levels_chain(etype, tmp_path):
    """levels=2 in one call == two chained single-level calls, and the
    _h<k> name advances 1 -> 2 -> 4 either way."""
    src = fixture(etype, tmp_path)
    n0 = sum(len(b["conn"]) for b in read_msh(src)["blocks"])
    kids = CASES[etype][1]

    one = msh_h_refine(src, levels=2, verbose=False)
    assert one["msh"].endswith("_h4.msh")
    assert one["n_elems"] == n0 * kids * kids
    assert verify_refinement(src, one["msh"], verbose=False)["ok"]

    a = msh_h_refine(src, out=str(tmp_path / "step_h2.msh"), verbose=False)
    b = msh_h_refine(a["msh"], verbose=False)
    assert b["msh"].endswith("step_h4.msh")
    assert b["n_elems"] == one["n_elems"]
    assert b["n_nodes"] == one["n_nodes"]
    assert verify_refinement(a["msh"], b["msh"], verbose=False)["ok"]


def test_tet_children_all_positive_and_equal_volume(tmp_path):
    """The 8 tet children of one parent: total volume conserved, every
    child positively oriented, and the 4 corner tets exactly 1/8 of the
    parent (the octahedron four share the other half)."""
    src = fixture(4, tmp_path)
    m0, r = read_msh(src), msh_h_refine(fixture(4, tmp_path, "t2"),
                                        verbose=False)
    m1 = read_msh(r["msh"])
    v0 = cell_measures(m0["xyz"], m0["blocks"][0])
    v1 = cell_measures(m1["xyz"], m1["blocks"][0])
    assert (v1 > 0).all()
    # 1e-9 (the verifier's own convention), not 1e-12: summing 384
    # skewed child tets against one parent volume accumulates ~1e-11
    # relative roundoff through the triple products -- arithmetic, not
    # a template error.  The per-parent checks below stay tight, and
    # the corner tets are exact scaled copies, so they pin the geometry.
    assert v1.sum() == pytest.approx(v0.sum(), rel=1e-9)
    # children are emitted parent-major: 8 per parent, first 4 = corners
    per = v1.reshape(len(v0), 8)
    assert np.allclose(per.sum(axis=1), v0, rtol=1e-10)
    assert np.allclose(per[:, :4], (v0 / 8.0)[:, None], rtol=1e-12)


def test_hex_children_fill_the_parent(tmp_path):
    """8 hex children per parent, volumes summing to the parent's."""
    src = fixture(5, tmp_path)
    m0 = read_msh(src)
    r = msh_h_refine(src, verbose=False)
    m1 = read_msh(r["msh"])
    v0 = cell_measures(m0["xyz"], m0["blocks"][0])
    v1 = cell_measures(m1["xyz"], m1["blocks"][0])
    assert (v1 > 0).all()
    assert np.allclose(v1.reshape(len(v0), 8).sum(axis=1), v0, rtol=1e-12)


def test_hex_face_centre_is_shared_with_a_boundary_quad(tmp_path):
    """The mixed-dimension trap: a hex8 volume plus the quad4 surface
    markers covering it.  The quad's new centre node and the hex's face
    centre are the SAME point, so a coordinate-hash dedup would emit two
    coincident nodes; the corner-set key must fuse them."""
    nodes, idx = _grid(1, 1, 1)
    hexes = [[idx[(0, 0, 0)], idx[(1, 0, 0)], idx[(1, 1, 0)], idx[(0, 1, 0)],
              idx[(0, 0, 1)], idx[(1, 0, 1)], idx[(1, 1, 1)], idx[(0, 1, 1)]]]
    bottom = [idx[(0, 0, 0)], idx[(1, 0, 0)], idx[(1, 1, 0)], idx[(0, 1, 0)]]
    p = str(tmp_path / "mixed.msh")
    with open(p, "w") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        f.write("$Nodes\n%d\n" % len(nodes))
        for k, q in enumerate(nodes):
            f.write("%d %r %r %r\n" % (k + 1, *[float(x) for x in q]))
        f.write("$EndNodes\n$Elements\n2\n")
        f.write("1 3 2 7 7 %s\n" % " ".join(str(n + 1) for n in bottom))
        f.write("2 5 2 1 1 %s\n" % " ".join(str(n + 1) for n in hexes[0]))
        f.write("$EndElements\n")
    r = msh_h_refine(p, verbose=False)
    m = read_msh(r["msh"])
    assert sum(len(b["conn"]) for b in m["blocks"]) == 4 + 8
    # 27 lattice nodes total; a duplicated face centre would make 28
    assert len(m["xyz"]) == 27
    q = np.round(m["xyz"] * 1e9)
    assert len(np.unique(q, axis=0)) == len(m["xyz"])
    tags = {int(b["tags"][0, 0]) for b in m["blocks"]}
    assert tags == {1, 7}                       # markers survive the split


# ---------------------------------------------------------- quadratic cases
@pytest.mark.parametrize("etype", [2, 3, 4],
                         ids=["tri6", "quad9", "tet10"])
def test_quadratic_round_trip(etype, tmp_path):
    """A quadratic mesh refines through its linear skeleton and comes
    back quadratic, with the measure and the cell count right."""
    lin = fixture(etype, tmp_path)
    q = linear_msh_to_quad(lin, verbose=False)["msh"]
    mq = read_msh(q)
    qtype = {2: 9, 3: 10, 4: 11}[etype]
    assert {b["etype"] for b in mq["blocks"]} == {qtype}
    n0 = sum(len(b["conn"]) for b in mq["blocks"])

    r = msh_h_refine(q, verbose=False)
    mr = read_msh(r["msh"])
    assert {b["etype"] for b in mr["blocks"]} == {qtype}
    assert r["n_elems"] == n0 * CASES[etype][1]
    assert mesh_measure(mr["xyz"], mr["blocks"]) == pytest.approx(
        mesh_measure(mq["xyz"], mq["blocks"]), rel=1e-12)
    v = verify_refinement(q, r["msh"], verbose=False)
    assert v["ok"], v["problems"]


def test_curved_quadratic_is_refused(tmp_path):
    """A genuinely curved quadratic mesh must be REFUSED, not silently
    flattened: nudge one midside off the chord and the gate fires."""
    lin = fixture(2, tmp_path)
    q = linear_msh_to_quad(lin, verbose=False)["msh"]
    txt = open(q).read().split("\n")
    i = txt.index("$Nodes")
    n = int(txt[i + 1])
    v = txt[i + 2 + n - 1].split()           # the LAST node is a midside
    v[2] = repr(float(v[2]) + 0.3)           # far above the 1e-4 gate
    txt[i + 2 + n - 1] = " ".join(v)
    bent = str(tmp_path / "bent.msh")
    open(bent, "w").write("\n".join(txt))
    with pytest.raises(SystemExit, match="CURVED"):
        msh_h_refine(bent, verbose=False)


def test_mixed_grades_refused(tmp_path):
    """tri3 + tri6 in one file is a user error, not something to guess."""
    nodes, idx = _grid(1, 1, 0)
    p = str(tmp_path / "grades.msh")
    with open(p, "w") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        f.write("$Nodes\n%d\n" % len(nodes))
        for k, q in enumerate(nodes):
            f.write("%d %r %r %r\n" % (k + 1, *[float(x) for x in q]))
        f.write("$EndNodes\n$Elements\n2\n")
        f.write("1 2 2 1 1 1 2 3\n")
        f.write("2 9 2 1 1 1 2 3 4 1 2\n")
        f.write("$EndElements\n")
    with pytest.raises(SystemExit, match="mixes quadratic"):
        msh_h_refine(p, verbose=False)


def test_unsupported_type_named(tmp_path):
    """A prism (gmsh 6) is refused BY NAME, not with a KeyError."""
    nodes, _ = _grid(1, 1, 1)
    p = str(tmp_path / "prism.msh")
    with open(p, "w") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        f.write("$Nodes\n%d\n" % len(nodes))
        for k, q in enumerate(nodes):
            f.write("%d %r %r %r\n" % (k + 1, *[float(x) for x in q]))
        f.write("$EndNodes\n$Elements\n1\n1 6 2 1 1 1 2 3 4 5 6\n")
        f.write("$EndElements\n")
    with pytest.raises(SystemExit, match="unsupported"):
        msh_h_refine(p, verbose=False)


# ------------------------------------------------- the verifier itself fails
def test_verifier_catches_a_broken_refinement(tmp_path):
    """The gate must be able to FAIL, or it proves nothing.  Three
    sabotages of a good hex refinement, one per check."""
    src = fixture(5, tmp_path)
    good = msh_h_refine(src, verbose=False)["msh"]
    assert verify_refinement(src, good, verbose=False)["ok"]

    m = read_msh(good)

    # (1) measure: drop a cell
    bad = dict(m, blocks=[dict(b, conn=b["conn"][:-1],
                               tags=b["tags"][:-1]) for b in m["blocks"]])
    v = verify_refinement(src, bad, verbose=False)
    assert not v["ok"] and any("measure" in p for p in v["problems"])

    # (2) orientation: flip one hex's node order
    conn = m["blocks"][0]["conn"].copy()
    conn[0] = conn[0][[4, 5, 6, 7, 0, 1, 2, 3]]
    flipped = dict(m, blocks=[dict(m["blocks"][0], conn=conn)])
    v = verify_refinement(src, flipped, verbose=False)
    assert not v["ok"]

    # (3) duplicates: split a shared node into two coincident copies
    xyz = np.vstack([m["xyz"], m["xyz"][0:1]])
    conn = m["blocks"][0]["conn"].copy()
    conn[conn == 0] = len(m["xyz"])
    dup = dict(m, xyz=xyz, blocks=[dict(m["blocks"][0], conn=conn)])
    v = verify_refinement(src, dup, verbose=False)
    assert not v["ok"] and any("duplicate" in p for p in v["problems"])
