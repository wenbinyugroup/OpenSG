import numpy as np
SG = "/home/roger/a/bagla0/OpenSG-2.0/tests/08072026_square_shell_mesh/prevabs/square_tube.sg"
L = [l for l in open(SG).read().splitlines() if l.strip()]
nn, ne, nph = [int(v) for v in L[3].split()]
xy = np.array([[float(v) for v in L[4 + i].split()[1:3]] for i in range(nn)])

sel = (xy[:, 0] > 0.4) & (np.abs(xy[:, 1]) < 0.03)
rows = np.unique(np.round(xy[sel, 0], 6))
print("right-wall node columns (x) near y=0:", rows)
print("node rows through thickness =", len(rows), "-> element layers =", len(rows) - 1)

sel2 = (xy[:, 1] > 0.4) & (np.abs(xy[:, 0]) < 0.03)
print("top-wall node rows (y) near x=0:", np.unique(np.round(xy[sel2, 1], 6)))

ob = xy[np.abs(xy[:, 1] - 0.515) < 1e-9]
s = np.sort(ob[:, 0]); d = np.diff(s)
print("outer top-edge nodes=%d  spacing min/mean/max = %.5f %.5f %.5f"
      % (len(s), d.min(), d.mean(), d.max()))
