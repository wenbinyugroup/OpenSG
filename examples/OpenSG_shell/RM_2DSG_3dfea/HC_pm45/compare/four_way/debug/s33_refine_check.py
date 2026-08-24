"""s33_refine_check.py -- the homogeneous-plate classical sigma33 under
pure kappa11 is nonzero at 1.9e-2 of sigma11 on a 16-layer mesh.  Bug
or discretization?  Exact answer is 0; linear elements cannot carry the
quadratic w3(z) the Poisson relief needs, leaving a residual that must
scale as 1/N_layers if it is interpolation error (and stay put if it
is a sign/term bug).  Sweep the layer count.
"""
import datetime
import os

import numpy as np

from opensg_solid.sg_dehom import dehom_fields
from opensg_solid.sg_homo import plate_homo_2d

HERE = os.path.dirname(os.path.abspath(__file__))
print("start : %s" % datetime.datetime.now().strftime("%H:%M:%S"))
print("%6s %14s %14s" % ("layers", "s33/s11", "x layers"))
for NZ in (8, 16, 32, 64):
    xs = np.linspace(-0.5, 0.5, 9)
    zs = np.linspace(-0.5, 0.5, NZ + 1)
    nid, nodes, cells = {}, [], []
    for i, x in enumerate(xs):
        for k, z in enumerate(zs):
            nid[(i, k)] = len(nodes)
            nodes.append((float(x), float(z)))
    for i in range(8):
        for k in range(NZ):
            cells.append([nid[(i, k)], nid[(i + 1, k)],
                          nid[(i + 1, k + 1)], nid[(i, k + 1)]])
    yml = os.path.join(HERE, "_iso_%d.yaml" % NZ)
    with open(yml, "w") as f:
        f.write("n_model: 2\nrefined: 1\nmsg: solid\n")
        f.write("nodes:\n")
        for p in nodes:
            f.write("- [%.9f, %.9f, 0.0]\n" % p)
        f.write("cells:\n")
        for c in cells:
            f.write("- [%d, %d, %d, %d]\n" % tuple(c))
        f.write("mat_id: [%s]\n" % ", ".join(["1"] * len(cells)))
        f.write("materials:\n  1:\n    type: 0\n    E: 70000.0\n"
                "    nu: 0.3\n    density: 0.0\n")
    r = plate_homo_2d(yml, refined=1, plot=False)
    _, S_, _ = dehom_fields(r, np.array([0.0, 0, 0, 1e-3, 0, 0]))
    S_ = np.asarray(S_)
    ratio = np.abs(S_[..., 2]).max() / np.abs(S_[..., 0]).max()
    print("%6d %14.3e %14.2f" % (NZ, ratio, ratio * NZ))
print("(a constant last column = pure 1/N interpolation error;"
      " a constant SECOND column would be a bug)")
print("end   : %s" % datetime.datetime.now().strftime("%H:%M:%S"))
