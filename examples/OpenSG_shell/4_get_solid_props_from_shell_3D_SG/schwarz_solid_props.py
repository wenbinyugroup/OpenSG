"""msg_shell 3-D shell SG: equivalent solid properties of the Schwarz-P TPMS
shell cell.  boundary="aperiodic" (default: boundary solution mapped onto
the bounding-box nodes as Dirichlet data) or boundary="periodic" (all three
directions tied) -- see compare_boundary_modes.py for the head-to-head.
Writes the timed .out (via shell_sg3d) and the mesh png.

Run (after make_schwarz_yaml.py):  python schwarz_solid_props.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml as _yaml

from opensg_shell.shell_sg3d import shell_sg3d

r = shell_sg3d("schwarz_p_3Dshell.yaml")   # omega = SG surface area (default)
C = r["C3D"]
print("junction edges: %d   ndof: %d   solve %.1f s"
      % (r["n_junction_edges"], r["ndof"], r["solve_time"]))
for i in range(6):
    print("  " + " ".join("%13.5e" % C[i, j] for j in range(6)))

d = _yaml.safe_load(open("schwarz_p_3Dshell.yaml"))
row = lambda r_: " ".join(str(x) for x in
                          (r_ if isinstance(r_, list) else [r_])).split()
nd = np.array([[float(v) for v in row(r_)][:3] for r_ in d["nodes"]])
el = np.array([[int(v) for v in row(r_)] for r_ in d["elements"]]) - 1
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(projection="3d")
ax.plot_trisurf(nd[:, 0], nd[:, 1], el, nd[:, 2], cmap="viridis",
                edgecolor="none", alpha=0.9)
ax.set_xlabel("y1")
ax.set_ylabel("y2")
ax.set_zlabel("y3")
ax.set_box_aspect((1, 1, 1))
plt.tight_layout()
plt.savefig("schwarz_p_mesh.png", dpi=150)
print("wrote schwarz_p_mesh.png")
