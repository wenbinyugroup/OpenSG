"""Step 1 -- windIO blade -> 1-D shell SG yaml at every station (standalone, no wrapper).

Reads the windIO blade (v1 or v2), builds the cross-section at each station, and emits
one OpenSG 1-D shell SG yaml per station with the reference-axis origin and the chosen
reference surface recorded in the yaml.  Also writes the compulsory e1/e2/e3
orientation PNG and a layup-colored ring-mesh PNG per station, and a station table.

# ----------------------------------------------------------------------------
# ALL VARIABLES USED IN THIS SCRIPT
# ----------------------------------------------------------------------------
#   windio              windIO blade yaml (v1 outer_shape_bem or v2 outer_shape)
#   prefix              output tag; station files = <prefix>_rXXXX_shell.yaml,
#                       XXXX = round(r * 1000)
#   stations            "airfoil" = the blade's own airfoil positions (IEA-22: 16);
#                       an int N = N uniform stations r = i/(N-1) (51 matches the
#                       VABS 51-station set); or an explicit list of r
#   mesh_size           target element arc length (chord-normalised)
#   reference           "center" (mid-surface) | "oml" shell reference surface
#   cross_sections/     output folder: yamls + <tag>_mesh.png + <tag>_orient.png
#   <prefix>_stations.dat   r, chord, twist, n_nodes, n_elems, n_sets, n_webs
# ----------------------------------------------------------------------------
"""
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from opensg_shell import auto_emit
from opensg_shell.windio import load_blade, blade_stations, build_cross_section, \
    emit_shell_yaml, emit_prevabs_xml

############### User Input #################################
windio = "IEA-22-280-RWT.yaml"     # windIO blade file (v1 or v2)
prefix = "iea22"                   # station files = <prefix>_rXXXX_shell.yaml
stations = "airfoil"               # "airfoil" | int N (uniform, e.g. 51) | [0.2, 0.5, ...]
mesh_size = 0.01                   # element arc length / chord
reference = "center"               # shell reference surface: "center" | "oml"
write_xml = False                  # True: ALSO emit the PreVABS XML per station
                                   # (cross_sections/xml/<tag>/ -- the cross-check
                                   # input for the XML -> prevabs -> 2-D-solid path)
############################################################

out = "cross_sections"
os.makedirs(out, exist_ok=True)
blade = load_blade(windio)
if stations == "airfoil":
    rs = blade_stations(blade)                 # the blade's own airfoil positions
elif isinstance(stations, int):
    rs = list(np.linspace(0.0, 1.0, stations))  # uniform grid (51 = the VABS set)
else:
    rs = list(stations)
print("%s: %d stations, reference=%s" % (windio, len(rs), reference))

rows = []
for r in rs:
    tag = "%s_r%04d" % (prefix, round(r * 1000))
    cs = build_cross_section(blade, r, mesh_size=mesh_size)
    info = emit_shell_yaml(cs, os.path.join(out, tag + "_shell.yaml"), reference=reference)
    if write_xml:
        emit_prevabs_xml(cs, os.path.join(out, "xml", tag), name=tag)
    rows.append([r, cs["chord"], cs["twist"], info["n_nodes"], info["n_elems"],
                 info["n_sets"], info["n_webs"]])
    print("  %s  r=%.4f  chord=%7.3f m  %4d nodes  %4d elems  %d sets  %d webs"
          % (tag, r, cs["chord"], info["n_nodes"], info["n_elems"],
             info["n_sets"], info["n_webs"]))

    # ring mesh colored by layup (shell convention) + compulsory orientation PNG
    ypath = os.path.join(out, tag + "_shell.yaml")
    import yaml as _y
    d = _y.safe_load(open(ypath))
    nd = np.array([[float(v) for v in row[0].split()][:2] for row in d["nodes"]])
    el = np.array([[int(v) for v in row[0].split()] for row in d["elements"]]) - 1
    lam = np.zeros(len(el), int)
    for k, grp in enumerate(d["sets"]["element"]):
        for lab in grp["labels"]:
            lam[int(lab) - 1] = k
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    for k in range(lam.max() + 1):
        first = True
        for e in np.where(lam == k)[0]:
            seg = nd[el[e]]
            ax.plot(seg[:, 0], seg[:, 1], "-", color=cmap(k % 20), lw=2.0,
                    label="layup_%d" % k if first else None)
            first = False
    ax.plot(nd[:, 0], nd[:, 1], ".", color="0.25", ms=1.5)
    ax.set_aspect("equal"); ax.set_xlabel("y2 (m)"); ax.set_ylabel("y3 (m)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, tag + "_mesh.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    auto_emit(ypath, out_png=os.path.join(out, tag + "_orient.png"))

np.savetxt(prefix + "_stations.dat", np.array(rows),
           fmt="%.4f %10.4f %10.6f %6d %6d %3d %3d",
           header="r chord[m] twist[rad] n_nodes n_elems n_sets n_webs")
print("wrote %s/ (%d yamls + mesh/orient PNGs) + %s_stations.dat"
      % (out, len(rs), prefix))
