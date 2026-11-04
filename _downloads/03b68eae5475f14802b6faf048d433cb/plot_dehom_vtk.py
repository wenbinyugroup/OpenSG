"""Draw S11 of an OpenSG plate dehomogenization from its .vtk, and print the plate force it carries.

Run it in the folder of the run:  python plot_dehom_vtk.py
"""
import struct

assert struct.calcsize("P") == 8  # OpenSG runs on 64-bit Python

import re

import matplotlib.pyplot as plt
import meshio
import numpy as np

stem = "RHC_SW_2UC_45"
m = meshio.read(stem + "_dehom.vtk")
tri = m.cells_dict["triangle"]                              # one triangle per SG element
s11 = np.ravel(m.cell_data_dict["Sig_xx"]["triangle"])     # the element mean of S11
x, y = m.points[:, 0], m.points[:, 1]                       # the node coordinates y2 and y3

fig, ax = plt.subplots(figsize=(5.6, 6.0))
tpc = ax.tripcolor(x, y, tri, facecolors=s11, cmap="jet", edgecolors="none")
ax.set_aspect("equal")
ax.set_xlabel("$y_2$ (mm)")
ax.set_ylabel("$y_3$ (mm)")
fig.colorbar(tpc, ax=ax, pad=0.04, fraction=0.06).set_label("$S_{11}$ (MPa)")
fig.savefig(stem + "_dehom_S11.png", dpi=200, bbox_inches="tight")

# the force per unit width that S11 carries: N11 = (sum of element area x S11) / omega
p = m.points[tri][:, :, :2]
d1, d2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
area = 0.5 * np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
omega = float(re.search(r"omega\s+([0-9.eE+-]+)", open(stem + ".out").readline()).group(1))
print("omega = %.6f mm" % omega)
print("N11   = %.4f N/mm" % (np.sum(area * s11) / omega))
print("S11   = %.1f to %.1f MPa" % (s11.min(), s11.max()))
