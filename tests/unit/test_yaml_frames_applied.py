"""v2.1: the SG file's `elementOrientations` are applied by plate_homo_2d (and so by `opensg x.yaml`) without an
explicit elem_rotation= argument: a 45-degree frame on every element equals the block `angle: 45`, the global
triad equals no rotation, and a raw-identity row is a cyclic axis permutation (NOT a no-op)."""
import os
import struct

import numpy as np
import pytest

assert struct.calcsize("P") == 8

HEAD = """n_model: %d
refined: 0
msg: solid
nodes:
- [0.0, 0.0, 0.0]
- [1.0, 0.0, 0.0]
- [2.0, 0.0, 0.0]
- [0.0, 1.0, 0.0]
- [1.0, 1.0, 0.0]
- [2.0, 1.0, 0.0]
cells:
- [0, 1, 4, 3]
- [1, 2, 5, 4]
mat_id: [1, 1]
materials:
  1: {type: 1, engineering: [150.0e9, 10.0e9, 10.0e9, 5.0e9, 5.0e9, 3.5e9, 0.3, 0.3, 0.4]%s}
"""
c = s = np.sqrt(0.5)
ROW45 = "- [%.16g, 0.0, %.16g, %.16g, 0.0, %.16g, 0.0, 1.0, 0.0]" % (s, c, c, -s)   # 45 deg about the thickness (y)
NOOP = "- [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]"                          # the global triad
RAWID = "- [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]"                         # a cyclic axis permutation


def _law(tmp_path, name, n_model, angle_text="", rows=None):
    from opensg_solid.sg_homo import plate_homo_2d
    text = HEAD % (n_model, angle_text)
    if rows is not None:
        text += "elementOrientations:\n" + "\n".join([rows] * 2) + "\n"
    p = os.path.join(str(tmp_path), name + ".yaml")
    with open(p, "w") as f:
        f.write(text)
    r = plate_homo_2d(p, n_model=n_model, refined=0, plot=False, workdir=str(tmp_path), recovery=False)
    return np.asarray(r["law"], float)


@pytest.mark.parametrize("n_model", [2, 3])
def test_file_frames_are_applied(tmp_path, n_model):
    K0 = _law(tmp_path, "angle0", n_model)
    K45 = _law(tmp_path, "angle45", n_model, ", angle: 45.0")
    assert np.max(np.abs(K45 - K0)) > 1e-3 * np.abs(K0).max()            # the angle matters for this material
    Kf = _law(tmp_path, "frame45", n_model, rows=ROW45)
    assert np.allclose(Kf, K45, rtol=1e-7, atol=1e-7 * np.abs(K45).max())  # frames in the file == block angle
    Kn = _law(tmp_path, "noop", n_model, rows=NOOP)
    assert np.allclose(Kn, K0, rtol=1e-10, atol=1e-10 * np.abs(K0).max())   # the global triad == no rotation
    Kr = _law(tmp_path, "rawid", n_model, rows=RAWID)
    assert np.max(np.abs(Kr - K0)) > 1e-3 * np.abs(K0).max()              # raw identity is NOT the no-op
