"""Square shell (line-element) mesh -> gmsh .msh -> DOLFIN .xml.

Writes, in this folder:
  square_shell.msh   gmsh 2.2 ASCII, 2-node line elements around the square
  square_shell.xml   legacy DOLFIN/FEniCS mesh XML (celltype="interval", dim=2)
  square_shell.png   the mesh image

Run (from this folder):  python make_square_shell_mesh.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

############### User Input #################################
a = 1.0                  # square side
nseg = 10                # line elements per side
name = "square_shell"
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
cells = np.array([[i, (i+1) % m] for i in range(m)], int)

# ---------------- gmsh 2.2 ASCII (.msh) -------------------------------------
with open(name + ".msh", "w") as f:
    f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
    f.write("$Nodes\n%d\n" % m)
    for i, (x, y) in enumerate(pts):
        f.write("%d %.12f %.12f 0.0\n" % (i+1, x, y))
    f.write("$EndNodes\n")
    f.write("$Elements\n%d\n" % len(cells))
    # elm-number elm-type(1=2-node line) n-tags phys geom n1 n2
    for e, (n1, n2) in enumerate(cells):
        f.write("%d 1 2 1 1 %d %d\n" % (e+1, n1+1, n2+1))
    f.write("$EndElements\n")
print("wrote %s.msh  (%d nodes, %d line elements)" % (name, m, len(cells)))

# ---------------- DOLFIN mesh XML -------------------------------------------
with open(name + ".xml", "w") as f:
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    f.write('<dolfin xmlns:dolfin="http://fenicsproject.org">\n')
    f.write('  <mesh celltype="interval" dim="2">\n')
    f.write('    <vertices size="%d">\n' % m)
    for i, (x, y) in enumerate(pts):
        f.write('      <vertex index="%d" x="%.12f" y="%.12f"/>\n' % (i, x, y))
    f.write('    </vertices>\n')
    f.write('    <cells size="%d">\n' % len(cells))
    for e, (n1, n2) in enumerate(cells):
        f.write('      <interval index="%d" v0="%d" v1="%d"/>\n' % (e, n1, n2))
    f.write('    </cells>\n')
    f.write('  </mesh>\n')
    f.write('</dolfin>\n')
print("wrote %s.xml  (DOLFIN mesh XML, celltype=interval, dim=2)" % name)

# ---------------- round-trip check via meshio -------------------------------
try:
    import meshio
    mm = meshio.read(name + ".msh")
    nl = sum(len(c.data) for c in mm.cells if c.type == "line")
    print("meshio re-read %s.msh: %d points, %d line cells"
          % (name, len(mm.points), nl))
except Exception as exc:
    print("meshio check skipped: %s" % exc)

fig, ax = plt.subplots(figsize=(5.5, 5.5))
for n1, n2 in cells:
    ax.plot(pts[[n1, n2], 0], pts[[n1, n2], 1], "-", color="C0", lw=1.6)
ax.plot(pts[:, 0], pts[:, 1], "o", color="k", ms=4)
ax.set_aspect("equal"); ax.set_xlabel("y2"); ax.set_ylabel("y3")
lim = h*1.15
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
fig.tight_layout(); fig.savefig(name + ".png", dpi=200)
print("wrote %s.png" % name)
