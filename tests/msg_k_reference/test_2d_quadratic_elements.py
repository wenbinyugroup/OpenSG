"""Anchors for the 2-D QUADRATIC elements (tri6 / quad9, 2026-08-30).

What they pin: sg_mesh._cell_basis (2,6)/(2,9), the sg_homo.
_to_basix_order tri6/quad9 gmsh->basix permutations, msh_to_yaml's
type-9/10 acceptance and helper.linear_msh_to_quad's quad4 -> quad9
upgrade.  A homogeneous isotropic plate meshed as a 2-D SG must
reproduce the closed-form membrane/bending law EXACTLY -- the A/D
warpings are linear/quadratic through the thickness, INSIDE the P2
space -- and the refined-G ladder must agree with the 1-D interval-P2
ladder at the same through-thickness resolution (cross-dimension
parity; G itself is mesh-dependent until the cubic warping converges).
A wrong midside permutation corrupts every element jacobian and every
anchor here, loudly.
"""
import os

import numpy as np
import pytest

from opensg_solid.sg_homo import plate_homo_2d

E0, NU, H, W = 1000.0, 0.3, 1.0, 3.0
G0 = E0 / (2.0 * (1.0 + NU))
MAT = {1: {"type": 0, "E": E0, "nu": NU}}


def _lattice(nx, nz):
    """The (2nx+1) x (2nz+1) half-step node lattice of an nx x nz cell
    grid over [0, W] x [-H/2, H/2], x-spacing NONUNIFORM (graded) so a
    permutation error cannot hide behind uniform jacobians.

    In:  nx, nz int cells
    Out: (nodes (N, 3), id[ix, iz] -> node index)."""
    xe = W * (np.linspace(0.0, 1.0, nx + 1)) ** 1.3
    xs = np.empty(2 * nx + 1)
    xs[0::2] = xe
    xs[1::2] = 0.5 * (xe[:-1] + xe[1:])
    zs = np.linspace(-H / 2, H / 2, 2 * nz + 1)
    idx = np.arange((2 * nx + 1) * (2 * nz + 1)).reshape(
        2 * nx + 1, 2 * nz + 1)
    nodes = np.zeros((idx.size, 3))
    nodes[:, 0] = np.repeat(xs, 2 * nz + 1)
    nodes[:, 1] = np.tile(zs, 2 * nx + 1)
    return nodes, idx


def sg_quad9(nx, nz):
    """quad9 SG in GMSH node order (corners cyclic, midsides
    (12)(23)(34)(41), center) -- _to_basix_order owns the hop.

    In:  nx, nz int cells
    Out: SG dict for plate_homo_2d."""
    nodes, idx = _lattice(nx, nz)
    cells = []
    for i in range(nx):
        for k in range(nz):
            a, c = 2 * i, 2 * k
            cells.append([idx[a, c], idx[a + 2, c], idx[a + 2, c + 2],
                         idx[a, c + 2], idx[a + 1, c], idx[a + 2, c + 1],
                         idx[a + 1, c + 2], idx[a, c + 1],
                         idx[a + 1, c + 1]])
    return {"dim": 2, "nodes": nodes, "cells": cells,
            "mat_id": np.ones(len(cells), int), "materials": MAT,
            "scale": 1.0}


def sg_tri6(nx, nz):
    """tri6 SG (each quad cell split along its diagonal) in GMSH node
    order (corners, midsides (12)(23)(31)).

    In:  nx, nz int cells
    Out: SG dict for plate_homo_2d."""
    nodes, idx = _lattice(nx, nz)
    cells = []
    for i in range(nx):
        for k in range(nz):
            a, c = 2 * i, 2 * k
            # (a,c) (a+2,c) (a+2,c+2) and (a,c) (a+2,c+2) (a,c+2);
            # the shared diagonal midside is the cell-center lattice node
            cells.append([idx[a, c], idx[a + 2, c], idx[a + 2, c + 2],
                          idx[a + 1, c], idx[a + 2, c + 1],
                          idx[a + 1, c + 1]])
            cells.append([idx[a, c], idx[a + 2, c + 2], idx[a, c + 2],
                          idx[a + 1, c + 1], idx[a + 1, c + 2],
                          idx[a, c + 1]])
    return {"dim": 2, "nodes": nodes, "cells": cells,
            "mat_id": np.ones(len(cells), int), "materials": MAT,
            "scale": 1.0}


def sg_1d_p2(nz):
    """1-D interval-P2 ladder twin (gmsh line3 order: ends, then mid).

    In:  nz int elements over the thickness
    Out: SG dict for plate_homo_2d."""
    ze = np.linspace(-H / 2, H / 2, nz + 1)
    zs = list(ze)
    cells = []
    for k in range(nz):
        zs.append(0.5 * (ze[k] + ze[k + 1]))
        cells.append([k, k + 1, len(zs) - 1])
    nodes = np.zeros((len(zs), 3))
    nodes[:, 0] = zs
    return {"dim": 1, "nodes": nodes, "cells": cells,
            "mat_id": np.ones(nz, int), "materials": MAT, "scale": 1.0}


def _abd(r):
    """The 6x6 ABD of a refined run result."""
    A = r.get("A6_ladder")
    return np.asarray(A if A is not None else r["C_eff"][:6, :6])


@pytest.mark.parametrize("maker", [sg_quad9, sg_tri6],
                         ids=["quad9", "tri6"])
def test_iso_plate_law_exact(maker):
    """Homogeneous iso plate: A/D closed forms are IN the P2 space."""
    r = plate_homo_2d(maker(3, 4), refined=1)
    A = _abd(r)
    A11 = E0 * H / (1.0 - NU ** 2)
    D11 = E0 * H ** 3 / 12.0 / (1.0 - NU ** 2)
    assert A[0, 0] == pytest.approx(A11, rel=1e-9)
    assert A[0, 1] == pytest.approx(NU * A11, rel=1e-9)
    assert A[2, 2] == pytest.approx(G0 * H, rel=1e-9)
    assert A[3, 3] == pytest.approx(D11, rel=1e-9)
    assert A[5, 5] == pytest.approx(G0 * H ** 3 / 12.0, rel=1e-9)
    # symmetric section: no membrane-bending coupling
    assert np.abs(A[:3, 3:]).max() <= 1e-8 * A11


def test_g_parity_quad9_with_1d_p2():
    """Refined G, quad9: the tensor-product warping space separates, so
    the 2-D quad9 ladder == the 1-D interval-P2 ladder at the SAME
    through-thickness resolution (G is not yet converged at nz = 4 --
    parity is the anchor, not 5/6)."""
    nz = 4
    g2 = np.asarray(plate_homo_2d(sg_quad9(3, nz), refined=1)["G_msg"])
    g1 = np.asarray(plate_homo_2d(sg_1d_p2(nz), refined=1)["G_msg"])
    assert g2[0, 0] == pytest.approx(g1[0, 0], rel=1e-7)
    assert g2[1, 1] == pytest.approx(g1[1, 1], rel=1e-7)


def test_g_convergence_tri6_to_1d_limit():
    """Refined G, tri6: the triangulated thickness has a slightly
    different discrete warping space (diagonal edges), so finite-nz
    parity with the interval ladder is NOT expected -- CONVERGENCE to
    the same limit is: measured rel gap 2.4e-5 at nz = 4, and it must
    shrink at least 4x per thickness doubling (order >= 2)."""
    gaps = []
    for nz in (4, 8):
        g2 = float(np.asarray(
            plate_homo_2d(sg_tri6(3, nz), refined=1)["G_msg"])[0, 0])
        g1 = float(np.asarray(
            plate_homo_2d(sg_1d_p2(nz), refined=1)["G_msg"])[0, 0])
        gaps.append(abs(g2 - g1) / abs(g1))
    assert gaps[0] < 1e-4
    assert gaps[1] < gaps[0] / 4.0


def _write_msh(path, etype, nodes, cells, phys):
    """Minimal gmsh 2.2 writer for the promoter round-trip tests."""
    with open(path, "w") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        f.write("$PhysicalNames\n1\n2 1 \"m\"\n$EndPhysicalNames\n")
        f.write("$Nodes\n%d\n" % len(nodes))
        for k, p in enumerate(nodes):
            f.write("%d %r %r 0.0\n" % (k + 1, float(p[0]), float(p[1])))
        f.write("$EndNodes\n$Elements\n%d\n" % len(cells))
        for k, c in enumerate(cells):
            f.write("%d %d 2 %d %d %s\n"
                    % (k + 1, etype, phys, phys,
                       " ".join(str(n + 1) for n in c)))
        f.write("$EndElements\n")


def test_promoter_quad4_to_quad9(tmp_path):
    """quad4 -> quad9: conforming shared midsides + per-element center
    at the corner mean; read back through msh_to_yaml.read_msh22."""
    from opensg_solid.helper.make_linear_msh_to_quad import (
        linear_msh_to_quad)
    from opensg_solid.io.msh_to_yaml import read_msh22

    nodes = [(0.0, 0.0), (1.0, 0.0), (2.2, 0.0),
             (0.0, 1.0), (1.0, 1.0), (2.2, 1.0)]
    cells = [[0, 1, 4, 3], [1, 2, 5, 4]]          # 2 quads, shared edge
    src = str(tmp_path / "two_quads.msh")
    _write_msh(src, 3, nodes, cells, 1)
    info = linear_msh_to_quad(src, verbose=False)
    # 6 corners + 7 unique edges + 2 centers
    assert info["n_nodes"] == 6 + 7 + 2
    assert info["n_midside"] == 7
    assert info["n_center"] == 2
    m = read_msh22(info["msh"])
    assert m["etype"] == 10 and m["npe"] == 9
    assert m["phys_names"] == {1: "m"}
    nd = m["nodes"]
    for c in np.asarray(m["cells"], int) - 1:      # 1-based -> 0-based
        corners, mids, ctr = nd[c[:4]], nd[c[4:8]], nd[c[8]]
        assert np.allclose(ctr, corners.mean(axis=0))
        for s, (p, q) in enumerate(((0, 1), (1, 2), (2, 3), (3, 0))):
            assert np.allclose(mids[s],
                               0.5 * (corners[p] + corners[q]))


def test_promoter_tri3_to_tri6(tmp_path):
    """tri3 -> tri6 (the HC ladder route) is accepted end to end."""
    from opensg_solid.helper.make_linear_msh_to_quad import (
        linear_msh_to_quad)
    from opensg_solid.io.msh_to_yaml import read_msh22

    nodes = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    cells = [[0, 1, 2], [0, 2, 3]]
    src = str(tmp_path / "two_tris.msh")
    _write_msh(src, 2, nodes, cells, 1)
    info = linear_msh_to_quad(src, verbose=False)
    assert info["n_midside"] == 5 and info.get("n_center", 0) == 0
    m = read_msh22(info["msh"])
    assert m["etype"] == 9 and m["npe"] == 6


def _area2(p):
    """Twice the shoelace area of one CCW polygon (n, 2)."""
    x, y = p[:, 0], p[:, 1]
    return float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def test_h_refine_quads(tmp_path):
    """quad4 h-refine: 4x elements, h halved, area/tags/names kept,
    the _h<k> naming chains, and h+p compose to a quad9 twin."""
    from opensg_solid.helper import linear_msh_to_quad, msh_h_refine
    from opensg_solid.io.msh_to_yaml import read_msh22

    nodes = [(0.0, 0.0), (1.0, 0.0), (2.2, 0.0),
             (0.0, 1.0), (1.0, 1.0), (2.2, 1.0)]
    cells = [[0, 1, 4, 3], [1, 2, 5, 4]]
    src = str(tmp_path / "two_quads.msh")
    _write_msh(src, 3, nodes, cells, 1)

    r1 = msh_h_refine(src, verbose=False)
    assert r1["msh"].endswith("two_quads_h2.msh")
    assert r1["levels"] == 1 and r1["n_elems"] == 8
    m = read_msh22(r1["msh"])
    assert m["etype"] == 3 and m["phys_names"] == {1: "m"}
    assert set(m["phys"]) == {1}
    a0 = sum(_area2(np.asarray(nodes)[c]) for c in cells)
    a1 = sum(_area2(m["nodes"][c - 1][:, :2]) for c in m["cells"])
    assert a1 == pytest.approx(a0, rel=1e-12)
    assert max(np.linalg.norm(np.diff(m["nodes"][c - 1][:, :2],
                                      axis=0), axis=1).max()
               for c in m["cells"]) <= 1.2 / 2 + 1e-12

    r2 = msh_h_refine(r1["msh"], verbose=False)      # the NEXT level
    assert r2["msh"].endswith("two_quads_h4.msh")
    assert r2["n_elems"] == 32

    q = linear_msh_to_quad(r1["msh"], verbose=False)  # h then p
    mq = read_msh22(q["msh"])
    assert mq["etype"] == 10 and mq["npe"] == 9


def test_h_refine_tris_target(tmp_path):
    """tri3 target-size mode: levels from the MAX edge; 4^L elements."""
    from opensg_solid.helper import msh_h_refine

    nodes = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    cells = [[0, 1, 2], [0, 2, 3]]
    src = str(tmp_path / "two_tris.msh")
    _write_msh(src, 2, nodes, cells, 1)
    # max edge = sqrt(2); target 0.4 -> ceil(log2(1.414/0.4)) = 2 levels
    r = msh_h_refine(src, target_h=0.4, verbose=False)
    assert r["levels"] == 2 and r["n_elems"] == 2 * 16
    assert r["msh"].endswith("two_tris_h4.msh")
    assert r["h_max"] <= 0.4 + 1e-12
    # already fine enough -> nothing written
    r0 = msh_h_refine(r["msh"], target_h=1.0, verbose=False)
    assert r0["levels"] == 0 and r0["msh"] == r["msh"]
