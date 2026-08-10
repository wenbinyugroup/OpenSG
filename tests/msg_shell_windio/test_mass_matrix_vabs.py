"""Unit test: the msg-solid 6x6 mass matrix vs VABS on the SAME cross-section.

The vendored iea_s10.sg (IEA-22 eta = 0.2, VABS 51-station set: 30k nodes /
55k linear tris / 6 materials WITH densities) is parsed into the sc dict and
integrated by opensg_solid.sg_homo.mass_matrix_2d; the reference is the mass
block VABS itself printed in iea_s10.sg.K from the identical mesh and
densities -- so the gate is TIGHT (1e-5 relative), not a modeling tolerance.

(The SHELL mass model, mass_matrix_ring, is gated against the same VABS .K
in test_vabs_k_beam_props.py at shell-vs-solid modeling tolerances.)

Run:  pytest tests/msg_shell_windio -q      (env opensg_2_0)
"""
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
WDIR = os.path.join(ROOT, "examples", "OpenSG_shell", "windio")
SG = os.path.join(WDIR, "vabs_K", "iea_s10.sg")
KREF = os.path.join(WDIR, "vabs_K", "iea_s10.sg.K")


def _read_vabs_sg(path):
    """Parse a VABS .sg (format_flag nlayer / flags / nnode nelem nmat / nodes /
    elements / per-element layer / layer->material / orthotropic cards + density)
    into (sc dict, {mat_id: rho})."""
    lines = [ln for ln in open(path).read().splitlines() if ln.strip()]
    nlayer = int(lines[0].split()[1])
    nn, ne, nm = (int(v) for v in lines[3].split()[:3])
    k = 4
    nodes = np.zeros((nn, 3))
    for ln in lines[k:k + nn]:
        p = ln.split()
        nodes[int(p[0]) - 1, :2] = float(p[1]), float(p[2])
    k += nn
    cells = []
    for ln in lines[k:k + ne]:
        p = [int(v) for v in ln.split()]
        cells.append([v - 1 for v in p[1:] if v != 0])
    k += ne
    elem_layer = np.zeros(ne, int)
    for ln in lines[k:k + ne]:
        p = ln.split()
        elem_layer[int(p[0]) - 1] = int(p[1])
    k += ne
    layer_mat = {}
    for ln in lines[k:k + nlayer]:
        p = ln.split()
        layer_mat[int(p[0])] = int(p[1])
    k += nlayer
    rho = {}
    while k < len(lines) and len(rho) < nm:
        head = lines[k].split()
        mid, iso = int(head[0]), int(head[1])
        nprop = {0: 1, 1: 3, 2: 6}[iso]           # iso: E nu | ortho: 3 | aniso: 6
        rho[mid] = float(lines[k + 1 + nprop].split()[0])
        k += 2 + nprop
    mat_id = np.array([layer_mat[l] for l in elem_layer], int)
    sc = {"dim": 2, "nodes": nodes, "cells": cells, "mat_id": mat_id,
          "materials": {}, "scale": 1.0}
    return sc, rho


def test_mass_matrix_2d_vs_vabs():
    from opensg_solid.sg_homo import mass_matrix_2d
    from opensg_shell.windio import read_k_file

    sc, rho = _read_vabs_sg(SG)
    assert len(sc["cells"]) == 55434 and len(rho) == 6
    M, info = mass_matrix_2d(sc, rho)
    _, Mv = read_k_file(KREF)

    # same mesh + same densities -> integration parity, not a modeling gate
    for (i, j, lab) in [(0, 0, "m"), (3, 3, "i11"), (4, 4, "i22"), (5, 5, "i33"),
                        (0, 4, "S3"), (0, 5, "-S2"), (4, 5, "-I23")]:
        rel = abs(M[i, j] / Mv[i, j] - 1.0)
        assert rel < 1e-5, "%s: OpenSG %.8g vs VABS %.8g (rel %.2e)" % (
            lab, M[i, j], Mv[i, j], rel)
    # full-matrix agreement at the same tightness (scaled by the largest term)
    assert np.abs(M - Mv).max() / np.abs(Mv).max() < 1e-5
    # mass center against the .K's printed value
    assert np.allclose(info["mass_center"], [0.63867764, 0.06501045], atol=1e-5)
