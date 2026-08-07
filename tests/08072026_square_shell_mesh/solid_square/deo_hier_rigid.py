"""Deo hierarchical square: rigid-block (rotational-spring) junction
correction test.  nseg=40 so the mesh resolves the block radius t/2.

Run (from this folder):  python deo_hier_rigid.py
"""
import time

import numpy as np

from opensg_shell import build_solid_bundle

exec(open("deo_hier_convergence.py").read().split("area = ")[0])  # make_yaml etc.

area = (2*R)**2
make_yaml(40, "_deo_r.yaml")
t0 = time.perf_counter()
b0 = build_solid_bundle("_deo_r.yaml", cell_area=area)
t_plain = time.perf_counter() - t0
t0 = time.perf_counter()
b1 = build_solid_bundle("_deo_r.yaml", cell_area=area, junction="rigid_block")
t_rig = time.perf_counter() - t0
print("junctions: %d, penalized node-pairs: %d   [plain %.1f s, rigid %.1f s]"
      % (b1["junction"]["n_junctions"], b1["junction"]["n_pen_nodes"],
         t_plain, t_rig))
print("  %-5s %10s %10s %10s %10s %9s %9s"
      % ("term", "plain", "rigid", "Deo TW", "Deo solid", "plain%", "rigid%"))
for k, (nm, (i, j)) in enumerate(zip(LBL, IJ)):
    v0 = b0["C3D"][i, j]*1e-6
    v1 = b1["C3D"][i, j]*1e-6
    so = DEO_SO[k]
    print("  %-5s %10.2f %10.2f %10.2f %10.2f %+9.2f %+9.2f"
          % (nm, v0, v1, DEO_TW[k], so, 100*(v0-so)/so, 100*(v1-so)/so))
