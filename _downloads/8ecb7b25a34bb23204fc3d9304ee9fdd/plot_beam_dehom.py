"""Draw one stress field of an OpenSG beam dehomogenization from its .vtk, and print the six beam forces that
the stress adds up to.

Run it in the folder of the run:  python plot_beam_dehom.py RHC_SW_2UC_45 S11
"""
import struct

assert struct.calcsize("P") == 8  # OpenSG runs on 64-bit Python

import sys

import matplotlib.pyplot as plt
import meshio
import numpy as np
import yaml

stem, field = sys.argv[1], sys.argv[2]                      # the base name of the run, and S11, S12 or S13
m = meshio.read(stem + "_dehom.vtk")
tri = m.cells_dict["triangle"]                              # one triangle per SG element
name = {"S11": "Sig_xx", "S12": "Sig_xy", "S13": "Sig_xz"}[field]
s = np.ravel(m.cell_data_dict[name]["triangle"])           # the element mean of the field
x, y = m.points[:, 0], m.points[:, 1]                       # the node coordinates y2 and y3

fig, ax = plt.subplots(figsize=(5.6, 6.0))
tpc = ax.tripcolor(x, y, tri, facecolors=s, cmap="jet", edgecolors="none")
ax.set_aspect("equal")
ax.set_xlabel("$y_2$ (mm)")
ax.set_ylabel("$y_3$ (mm)")
fig.colorbar(tpc, ax=ax, pad=0.04, fraction=0.06).set_label("$S_{%s}$ (MPa)" % field[1:])
fig.savefig("%s_dehom_%s.png" % (stem, field), dpi=200, bbox_inches="tight")
print("%s = %.1f to %.1f MPa" % (field, s.min(), s.max()))

# the six beam forces: the stress at the three Gauss points of each triangle, each weighted by a third of its area
sg = yaml.load(open(stem + ".yaml"), Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
p = np.asarray(sg["nodes"])[np.asarray(sg["cells"])][:, :, :2]   # the corners y2 y3 of every triangle
d1, d2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
w = np.repeat(0.5 * np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / 3, 3)
g = np.loadtxt(stem + "_dehom_global.SM")                  # x y z S11 S22 S33 S12 S13 S23 at every Gauss point
assert np.abs(g[:, 1:3].reshape(-1, 3, 2).mean(axis=1) - p.mean(axis=1)).max() < 1e-4  # 3 rows per triangle
y2, y3, s11, s12, s13 = g[:, 1], g[:, 2], g[:, 3], g[:, 6], g[:, 7]
F = [np.sum(w * s11), np.sum(w * s12), np.sum(w * s13),
     np.sum(w * (y2 * s13 - y3 * s12)), np.sum(w * y3 * s11), -np.sum(w * y2 * s11)]
for n, f, u in zip(("F1", "F2", "F3", "M1", "M2", "M3"), F, ("N", "N", "N", "N mm", "N mm", "N mm")):
    print("%s = %.8g %s" % (n, f, u))
