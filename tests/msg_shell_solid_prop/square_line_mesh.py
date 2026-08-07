"""Square made of LINE (2-node) elements -- the shell/line mesh, and its image.

Run (from this folder):  python square_line_mesh.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

############### User Input #################################
a = 1.0                  # square side
nseg = 10                # line elements per side
############################################################

h = a/2
corners = [(-h, -h), (h, -h), (h, h), (-h, h)]
pts = []
for k in range(4):
    p0, p1 = np.array(corners[k]), np.array(corners[(k+1) % 4])
    for i in range(nseg):
        pts.append(p0 + (p1-p0)*i/nseg)
pts = np.array(pts)
m = len(pts)
cells = np.array([[i, (i+1) % m] for i in range(m)], int)   # 2-node line elems

print("square line mesh: %d nodes, %d line elements (%d per side)"
      % (m, len(cells), nseg))

fig, ax = plt.subplots(figsize=(5.5, 5.5))
for n1, n2 in cells:
    ax.plot(pts[[n1, n2], 0], pts[[n1, n2], 1], "-", color="C0", lw=1.6)
ax.plot(pts[:, 0], pts[:, 1], "o", color="k", ms=4)
ax.set_aspect("equal")
ax.set_xlabel("y2")
ax.set_ylabel("y3")
lim = h*1.15
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
fig.tight_layout()
fig.savefig("square_line_mesh.png", dpi=200)
print("wrote square_line_mesh.png")
