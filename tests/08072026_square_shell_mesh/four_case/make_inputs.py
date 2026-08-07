"""Build every input deck for the four cases: the 1-D shell yaml (route A), the
2-D solid yaml (route B) and the .sc it converts to (route C).

Run:  ~/miniconda3/envs/opensg_2_0/bin/python make_inputs.py
"""
import json
import os

import numpy as np

from crosscell import (ISO, M45, cross_shell_yaml, cross_solid_mesh,
                       cross_solid_yaml, mesh_sizing, wall_area_report)
from opensg_solid.rm_plate_1D.helper.yaml_to_sc import convert

L = 1.0
NSEG = 80                       # shell elements per wall (>= 60 required)
CASES = [("thick_iso", 0.10, ISO), ("thin_iso", 0.01, ISO),
         ("thick_m45", 0.10, M45), ("thin_m45", 0.01, M45)]

log, meta = [], {}
for name, t, mat in CASES:
    sy, nn_s, ne_s = cross_shell_yaml("%s_shell.yaml" % name, L, t, mat, NSEG)
    nt, na = mesh_sizing(L, t)
    nd, tri, frame, area = cross_solid_mesh(L, t, nt, na)
    cross_solid_yaml("%s_solid2d.yaml" % name, nd, tri, frame, mat)
    convert("%s_solid2d.yaml" % name, "%s.sc" % name, dim=2)
    got, exact, rel, shell_ct = wall_area_report(L, t, area)
    h_size = (0.5*L - 0.5*t)/na
    meta[name] = dict(t=t, material=mat["name"], angle=mat["angle"],
                      shell_nodes=nn_s, shell_elems=ne_s, nseg=NSEG,
                      solid_nodes=int(len(nd)), solid_elems=int(len(tri)),
                      nt=nt, na=na, h_thick=t/nt, h_along=h_size,
                      area_meshed=got, area_exact=exact, area_relerr=rel,
                      area_shell_counted=shell_ct, cell_area=L*L)
    log.append(
        "%-10s t/L=%-7.4f  shell %4d nodes /%4d elems (%d per wall)\n"
        "           solid %6d nodes /%6d tri   nt=%d na=%d  "
        "h_thick=%.5g h_along=%.5g (h_along/t=%.2f, aspect=%.2f)\n"
        "           wall area meshed %.10f  exact 2Lt-t^2 %.10f  rel err %+.3e\n"
        "           shell integrates each wall over its full length -> "
        "2Lt = %.10f (junction counted twice, +%.2f%%)"
        % (name, t/L, nn_s, ne_s, NSEG, len(nd), len(tri), nt, na, t/nt,
           h_size, h_size/t, h_size/(t/nt), got, exact, rel, shell_ct,
           100*(shell_ct-exact)/exact))

txt = "\n".join(log)
print(txt)
open("inputs.log", "w").write(txt + "\n")
json.dump(meta, open("inputs_meta.json", "w"), indent=1)
print("\nwrote inputs.log, inputs_meta.json and the 4x3 decks")
