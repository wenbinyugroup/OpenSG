"""The GENERALIZED shear-refined plate ladder: dimensional parity and
default-machinery gates.

The doctrine (plate_shear_ladder's laminate-as-strip rule): ONE
homogeneous plate meshed as a 1-D interval stack, a 2-D quad strip and
a 3-D hex slab must agree on the 8x8 law and on the pressure-driven
recovery -- the 1-D column is the analytic-anchor dimension
(rm_plate_1D parity recorded at <= 2e-11), 3-D is the generalization
under test.  Hex parity is machine-tight because the trilinear slab
reproduces the interval stack exactly; tet4 is exercised at FE
tolerance elsewhere (the AR5 study).

Also pinned: a refined plate run stores the V2 chains and load columns
BY DEFAULT at every SG dimension, and the q_reaction AUTO default reads
the cell's in-plane fill (uniform for a full cell -- what the 1-D/2-D
closed-form anchors assume; tau for voided/heterogeneous cells).
"""
import numpy as np
import pytest

from opensg_solid.sg_homo import plate_homo_2d
from opensg_solid.sg_dehom import dehom_fields

E0, NU, H, NZ = 1000.0, 0.3, 1.0, 8
MAT = {1: {"type": 0, "E": E0, "nu": NU}}
Z = np.linspace(-H / 2, H / 2, NZ + 1)


def sg_1d():
    nodes = np.zeros((NZ + 1, 3))
    nodes[:, 0] = Z
    return {"dim": 1, "nodes": nodes,
            "cells": [[k, k + 1] for k in range(NZ)],
            "mat_id": np.ones(NZ, int), "materials": MAT, "scale": 1.0}


def sg_2d(nx=2, w=0.25):
    xs = np.linspace(0, w, nx + 1)
    nid = lambda i, k: k * (nx + 1) + i                 # noqa: E731
    nodes = np.zeros(((nx + 1) * (NZ + 1), 3))
    for k in range(NZ + 1):
        for i in range(nx + 1):
            nodes[nid(i, k), :2] = (xs[i], Z[k])
    cells = [[nid(i, k), nid(i + 1, k), nid(i + 1, k + 1),
              nid(i, k + 1)] for k in range(NZ) for i in range(nx)]
    return {"dim": 2, "nodes": nodes, "cells": cells,
            "mat_id": np.ones(len(cells), int), "materials": MAT,
            "scale": 1.0}


def sg_3d(nx=2, ny=2, w=0.25):
    xs = np.linspace(0, w, nx + 1)
    ys = np.linspace(0, w, ny + 1)
    nid = lambda i, j, k: (k * (ny + 1) + j) * (nx + 1) + i  # noqa: E731
    nodes = np.zeros(((nx + 1) * (ny + 1) * (NZ + 1), 3))
    for k in range(NZ + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                nodes[nid(i, j, k)] = (xs[i], ys[j], Z[k])
    cells = []
    for k in range(NZ):
        for j in range(ny):
            for i in range(nx):
                cells.append([nid(i, j, k), nid(i + 1, j, k),
                              nid(i + 1, j + 1, k), nid(i, j + 1, k),
                              nid(i, j, k + 1), nid(i + 1, j, k + 1),
                              nid(i + 1, j + 1, k + 1),
                              nid(i, j + 1, k + 1)])
    return {"dim": 3, "nodes": nodes, "cells": cells,
            "mat_id": np.ones(len(cells), int), "materials": MAT,
            "scale": 1.0}


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    import os
    os.chdir(tmp_path_factory.mktemp("ladder"))
    return {tag: plate_homo_2d(build(), refined=1)
            for tag, build in (("1d", sg_1d), ("2d", sg_2d),
                               ("3d", sg_3d))}


def test_default_banks_present_at_every_dimension(runs):
    """A refined plate run must store the V2 chains and the load
    columns BY DEFAULT -- 1-D, 2-D and 3-D alike (the old n_sg==2 gate
    is gone)."""
    for tag, r in runs.items():
        for k in ("V0_ladder", "V11bar", "V12bar", "V21", "V23t",
                  "V1Lt", "V2Lt", "V1Lb", "V2Lb"):
            assert r.get(k) is not None, "%s missing %s" % (tag, k)


def test_G_dimensional_parity(runs):
    G1 = np.asarray(runs["1d"]["G_msg"])
    for tag in ("2d", "3d"):
        G = np.asarray(runs[tag]["G_msg"])
        assert np.abs(G - G1).max() <= 1e-10 * np.abs(G1).max(), tag


def gauss_profile(r, col, **ff):
    _, Sig, _ = dehom_fields(r, np.zeros(6), **ff)
    xe = np.asarray(r["x_end"])
    zq = np.einsum("qn,en->eq", np.asarray(r["phi_qn"]),
                   xe[:, :, r["n_sg"] - 1])
    return zq.ravel(), np.asarray(Sig)[:, :, col].ravel()


def test_sigma33_pressure_profile_parity(runs):
    """sigma33 under a unit TOP pressure: the 3-D hex slab must ride
    the 1-D anchor to near machine precision, and the profile must run
    -q at the top face toward 0 at the bottom."""
    z1, s1 = gauss_profile(runs["1d"], 2,
                           qt6=np.array([1.0, 0, 0, 0, 0, 0]))
    o = np.argsort(z1)
    z1, s1 = z1[o], s1[o]
    assert s1[-1] < -0.9 and abs(s1[0]) < 0.1
    for tag in ("2d", "3d"):
        z, s = gauss_profile(runs[tag], 2,
                             qt6=np.array([1.0, 0, 0, 0, 0, 0]))
        v = np.interp(z1, np.sort(z), s[np.argsort(z)])
        assert np.abs(v - s1).max() <= 1e-8, tag


def test_q_reaction_auto_uniform_for_full_cell(runs, capsys):
    """The AUTO default must pick the uniform reaction for a fully
    dense cell (fill = 1) -- the gauge every closed-form anchor pins."""
    for tag, r in runs.items():
        assert r["q_reaction"] == "uniform", tag


def test_q_column_inplane_parity_sandwich(tmp_path):
    """The load column's IN-PLANE content: a flat [45/-45]s-skin
    sandwich under unit top pressure must recover the SAME
    sigma11/sigma22/sigma33 profile from a 1-D stack, a 3-D hex slab
    and a 3-D tet slab -- the in-plane skin stress of the q-column
    (~0.4 q on this section) is real construction physics and must be
    dimension-independent (the AR5 S11 debug's verifier)."""
    import os
    os.chdir(tmp_path)
    t, nzc, nply = 0.02, 6, 4
    ply = {"type": 1,
           "engineering": [108000.0, 8000.0, 8000.0, 4000.0, 4000.0,
                           3000.0, 0.32, 0.32, 0.3]}
    zc = np.linspace(-0.5, 0.5, nzc + 1)
    zt = 0.5 + t * np.arange(1, nply + 1)
    zl = np.r_[-zt[::-1], zc, zt]
    nz = len(zl) - 1
    lay = [2, 3, 3, 2][::-1] + [1] * nzc + [2, 3, 3, 2]
    mats = {1: {"type": 0, "E": 4000.0, "nu": 0.3},
            2: dict(ply, angle=-45.0), 3: dict(ply, angle=45.0)}

    def s1d():
        nodes = np.zeros((nz + 1, 3))
        nodes[:, 0] = zl
        return {"dim": 1, "nodes": nodes,
                "cells": [[k, k + 1] for k in range(nz)],
                "mat_id": np.array(lay, int), "materials": mats,
                "scale": 1.0}

    def slab(split):
        nx, w = 2, 0.25
        xs = np.linspace(0, w, nx + 1)
        nid = lambda i, j, k: (k * (nx + 1) + j) * (nx + 1) + i  # noqa
        nodes = np.zeros(((nx + 1) ** 2 * (nz + 1), 3))
        for k in range(nz + 1):
            for j in range(nx + 1):
                for i in range(nx + 1):
                    nodes[nid(i, j, k)] = (xs[i], xs[j], zl[k])
        hexes, mid = [], []
        for k in range(nz):
            for j in range(nx):
                for i in range(nx):
                    hexes.append([nid(i, j, k), nid(i + 1, j, k),
                                  nid(i + 1, j + 1, k),
                                  nid(i, j + 1, k), nid(i, j, k + 1),
                                  nid(i + 1, j, k + 1),
                                  nid(i + 1, j + 1, k + 1),
                                  nid(i, j + 1, k + 1)])
                    mid.append(lay[k])
        if split:
            T6 = [(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
                  (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)]
            cells = [[h[a], h[b], h[c], h[d]] for h in hexes
                     for a, b, c, d in T6]
            mid = list(np.repeat(mid, 6))
        else:
            cells = hexes
        return {"dim": 3, "nodes": nodes, "cells": cells,
                "mat_id": np.array(mid, int), "materials": mats,
                "scale": 1.0}

    def prof(sc):
        r = plate_homo_2d(sc, refined=1)
        _, Sig, _ = dehom_fields(r, np.zeros(6),
                                 qt6=np.array([1.0, 0, 0, 0, 0, 0]))
        xe = np.asarray(r["x_end"])
        zq = np.einsum("qn,en->eq", np.asarray(r["phi_qn"]),
                       xe[:, :, r["n_sg"] - 1]).ravel()
        S = np.asarray(Sig).reshape(-1, 6)
        k = np.clip(np.digitize(zq, zl) - 1, 0, nz - 1)
        return np.array([S[k == b].mean(axis=0) for b in range(nz)])

    ref = prof(s1d())
    assert np.abs(ref[-1, 0]) > 0.2          # real in-plane q content
    for split in (False, True):
        p = prof(slab(split))
        for j in (0, 1, 2):                  # S11, S22, S33
            assert np.abs(p[:, j] - ref[:, j]).max() <= 1e-8, \
                (split, j)


def test_q_reaction_auto_tau_for_voided_cell(tmp_path):
    """A cell with an in-plane void (fill < 1) must auto-select tau."""
    import os
    os.chdir(tmp_path)
    sc = sg_3d(nx=3, ny=3, w=0.75)
    cells = np.asarray(sc["cells"])
    nd = np.asarray(sc["nodes"])
    cen = nd[cells].mean(axis=1)
    # drill the centre column out (a through void away from the
    # periodic faces)
    keep = ~((np.abs(cen[:, 0] - 0.375) < 0.12)
             & (np.abs(cen[:, 1] - 0.375) < 0.12))
    sc["cells"] = [list(c) for c in cells[keep]]
    sc["mat_id"] = np.ones(int(keep.sum()), int)
    r = plate_homo_2d(sc, refined=1)
    assert r["q_reaction"] == "tau"
