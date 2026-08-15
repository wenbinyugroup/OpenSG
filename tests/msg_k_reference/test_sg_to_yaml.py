"""Unit test: PreVABS/VABS `.sg` -> 2-D solid OpenSG yaml (io.sg_to_yaml).

The fixture is the REAL VABS section behind the msg-shell benchmark,
examples/OpenSG_shell/windio/vabs_K/iea_s10.sg (IEA-22 at r/R = 0.2, the
same file whose .sg.K the shell route is gated against).  The port of the
validated OpenSG_io converter must reproduce its mesh exactly, compose
theta1 (per element) with theta3 (per layup group) into ORTHONORMAL
right-handed material frames, and emit a yaml the solid reader accepts.

Run:  pytest tests/msg_k_reference -q      (env opensg_2_0)
"""
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SG = os.path.join(ROOT, "examples", "OpenSG_shell", "windio", "vabs_K",
                  "iea_s10.sg")


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    from opensg_solid.io.sc_to_yaml import sg_to_yaml

    out = str(tmp_path_factory.mktemp("sg") / "iea_s10.yaml")
    return sg_to_yaml(SG, out_path=out)


def test_counts_match_the_sg_header(converted):
    """The header's node/element/material counts are what gets written."""
    head = [ln.split() for ln in open(SG) if ln.strip()][:4]
    n_layup = int(head[0][1])
    nnode, nelem, nmat = (int(v) for v in head[3])
    assert converted["n_nodes"] == nnode
    assert converted["n_elems"] == nelem
    assert converted["n_mats"] == nmat
    assert converted["n_layups"] == n_layup


def test_yaml_loads_and_frames_are_proper_rotations(converted):
    from opensg_solid.io.sg_input import read_opensg_yaml

    sg = read_opensg_yaml(converted["yaml"])
    assert sg["dim"] == 2
    assert len(sg["nodes"]) == converted["n_nodes"]
    assert len(sg["cells"]) == converted["n_elems"]
    # every VABS 2-D element is a tri3 or quad4 (zero padding dropped)
    assert set(len(c) for c in sg["cells"]) <= {3, 4}
    # one material frame per element, each a right-handed orthonormal basis
    R = np.asarray(sg["orientation"], float).reshape(-1, 3, 3)
    assert len(R) == converted["n_elems"]
    assert np.allclose(np.linalg.det(R), 1.0, atol=1e-10)
    assert np.abs(np.einsum("nij,nkj->nik", R, R) - np.eye(3)).max() < 1e-10


def test_theta3_is_in_the_frame_not_the_material(converted):
    """theta1 sets the ply plane, theta3 tilts the fiber inside it.  With a
    nonzero theta3 somewhere on the section, e1 must leave the beam axis --
    if theta3 were dropped (or baked into the materials) every e1 would sit
    exactly on z and the off-axis plies would vanish."""
    from opensg_solid.io.sc_to_yaml import _sg_frame
    from opensg_solid.io.sg_input import read_opensg_yaml

    sg = read_opensg_yaml(converted["yaml"])
    R = np.asarray(sg["orientation"], float).reshape(-1, 3, 3)
    e1 = R[:, 0, :]
    assert np.abs(e1[:, 2]).max() > 0.99          # some ply is on-axis
    # the reference frame construction, term by term
    F = np.asarray(_sg_frame(30.0, 45.0), float).reshape(3, 3)
    c1, s1 = np.cos(np.pi / 6), np.sin(np.pi / 6)
    c3, s3 = np.cos(np.pi / 4), np.sin(np.pi / 4)
    assert np.allclose(F[0], [s3 * c1, s3 * s1, c3])
    assert np.allclose(F[2], [-s1, c1, 0.0])      # ply normal in-plane
    assert np.allclose(np.cross(F[0], F[1]), F[2], atol=1e-12)


def test_material_names_come_from_the_mat_sidecar(converted, tmp_path):
    """PreVABS writes <stem>.sg.mat; without it the names fall back to
    Material_<id> rather than failing."""
    import shutil

    from opensg_solid.io.sc_to_yaml import sg_to_yaml

    assert len(converted["names"]) == converted["n_mats"]
    bare = str(tmp_path / "nosidecar.sg")
    shutil.copy(SG, bare)                      # deliberately no .sg.mat
    R = sg_to_yaml(bare)
    assert R["names"] == ["Material_%d" % (i + 1) for i in range(R["n_mats"])]
