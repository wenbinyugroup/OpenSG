"""Sanity-check the PreVABS square-tube 2-D solid mesh and render it.

Reads   prevabs/square_tube.sg          (VABS-format 2-D mesh from PreVABS)
        square_tube_2dsolid.yaml        (OpenSG 2-D solid YAML w/ elementOrientations)
Writes  square_tube_mesh.png            (actual mesh, elements + edges)
        square_tube_orientation.png     (e1/e2/e3 material-frame arrows)
        square_tube_checks.txt          (the numbers)
"""
import os
import sys
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection

D = os.path.dirname(os.path.abspath(__file__))
SG = os.path.join(D, "prevabs", "square_tube.sg")
YML = os.path.join(D, "square_tube_2dsolid.yaml")

A_SIDE = 1.0      # midline square side
T_WALL = 0.03     # wall thickness

out = []


def say(s=""):
    print(s)
    out.append(s)


# ---------------------------------------------------------------- read .sg ---
lines = [l for l in open(SG).read().splitlines() if l.strip()]
ngrp = int(lines[0].split()[1])
nnode, nelem, nphase = [int(v) for v in lines[3].split()]

xy = np.array([[float(v) for v in lines[4 + i].split()[1:3]] for i in range(nnode)])
conn = []
for i in range(nnode + 4, nnode + 4 + nelem):
    nd = [int(v) for v in lines[i].split()[1:] if int(v) != 0]
    conn.append([n - 1 for n in nd])

p = 4 + nnode + nelem
egrp = np.array([int(lines[p + i].split()[1]) for i in range(nelem)])
theta1 = np.array([float(lines[p + i].split()[2]) for i in range(nelem)])
p = 4 + nnode + 2 * nelem
gmat = np.array([int(lines[p + g].split()[1]) for g in range(ngrp)])
gth3 = np.array([float(lines[p + g].split()[2]) for g in range(ngrp)])
theta3 = gth3[egrp - 1]

say("PreVABS .sg : %s" % SG)
say("  nodes            = %d" % nnode)
say("  elements         = %d" % nelem)
say("  material phases  = %d" % nphase)
say("  layup groups     = %d   (group -> material, theta3): %s"
    % (ngrp, ", ".join("g%d -> mat%d, %.1f deg" % (g + 1, gmat[g], gth3[g]) for g in range(ngrp))))
cnts = {}
for e in conn:
    cnts[len(e)] = cnts.get(len(e), 0) + 1
say("  element shapes   = %s   (3 = linear triangle, 4 = linear quad)"
    % ", ".join("%d-node: %d" % (k, v) for k, v in sorted(cnts.items())))
say("  theta1 (contour angle) unique values: %s"
    % np.array2string(np.unique(np.round(theta1, 4)), precision=3))
say("  theta3 (fibre angle)   unique values: %s deg" % np.unique(theta3))


# ---------------------------------------------------- geometry / area checks ---
def signed_area(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


sa = np.array([signed_area(xy[e]) for e in conn])
area = np.abs(sa).sum()
a_mid = 4.0 * A_SIDE * T_WALL                     # mitred corners, CENTER reference
a_oml = 4.0 * A_SIDE * T_WALL - 4.0 * T_WALL**2   # mitred corners, OML reference

say()
say("AREA")
say("  meshed wall area                     = %.9f" % area)
say("  analytical 4*a*t          (mitre, CENTER ref) = %.9f   rel err = %+.3e"
    % (a_mid, area / a_mid - 1.0))
say("  analytical 4*a*t - 4*t^2  (mitre, OML ref)    = %.9f   rel err = %+.3e"
    % (a_oml, area / a_oml - 1.0))

npos, nneg, nzero = int((sa > 0).sum()), int((sa < 0).sum()), int((sa == 0).sum())
say()
say("INVERTED / DEGENERATE ELEMENTS")
say("  signed areas: %d positive (CCW), %d negative (CW), %d zero" % (npos, nneg, nzero))
say("  consistent winding = %s  ->  inverted elements = %d"
    % (npos == 0 or nneg == 0, min(npos, nneg) + nzero))
say("  |area| min / max / mean = %.4e / %.4e / %.4e"
    % (np.abs(sa).min(), np.abs(sa).max(), np.abs(sa).mean()))

r = np.abs(xy).max(axis=1)   # Chebyshev radius = max(|x|,|y|) -> square "radius"
say()
say("GEOMETRY / LAYUP REFERENCE")
say("  bbox x = [%+.6f, %+.6f]   y = [%+.6f, %+.6f]"
    % (xy[:, 0].min(), xy[:, 0].max(), xy[:, 1].min(), xy[:, 1].max()))
say("  outer half-side max(|x|,|y|) = %.6f   (expected a/2 + t/2 = %.6f)"
    % (r.max(), A_SIDE / 2 + T_WALL / 2))
say("  inner half-side min(|x|,|y|) = %.6f   (expected a/2 - t/2 = %.6f)"
    % (r.min(), A_SIDE / 2 - T_WALL / 2))
say("  mid half-side (mean of the two) = %.6f   (expected a/2 = %.6f)  -> CENTER reference"
    % (0.5 * (r.max() + r.min()), A_SIDE / 2))

# ------------------------------------------------------------ closure check ---
from collections import Counter
edges = Counter()
for e in conn:
    n = len(e)
    for k in range(n):
        a, b = e[k], e[(k + 1) % n]
        edges[(min(a, b), max(a, b))] += 1
bnd = [e for e, c in edges.items() if c == 1]
bad = [e for e, c in edges.items() if c > 2]

adj = {}
for a, b in bnd:
    adj.setdefault(a, []).append(b)
    adj.setdefault(b, []).append(a)
seen, loops = set(), []
for s in adj:
    if s in seen:
        continue
    stack, comp = [s], []
    seen.add(s)
    while stack:
        v = stack.pop()
        comp.append(v)
        for w in adj[v]:
            if w not in seen:
                seen.add(w)
                stack.append(w)
    loops.append(comp)
loop_r = [(np.abs(xy[c]).max(axis=1).mean()) for c in loops]
say()
say("CONNECTIVITY / CLOSURE")
say("  interior edges (shared by 2 elements) = %d" % sum(1 for c in edges.values() if c == 2))
say("  boundary edges (shared by 1 element)  = %d" % len(bnd))
say("  non-manifold edges (>2 elements)      = %d" % len(bad))
say("  boundary loops = %d  (mean half-side: %s)"
    % (len(loops), ", ".join("%.4f" % v for v in sorted(loop_r))))
say("  every boundary node has exactly 2 boundary edges = %s"
    % all(len(v) == 2 for v in adj.values()))
say("  -> section is CLOSED (single closed cell: 1 outer + 1 inner loop) = %s"
    % (len(loops) == 2 and len(bad) == 0 and all(len(v) == 2 for v in adj.values())))

# through-thickness element count: how many elements does a radial ray cross?
cen = np.array([xy[e].mean(axis=0) for e in conn])


def in_tri(p, tri):
    (x1, y1), (x2, y2), (x3, y3) = tri
    d = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    if abs(d) < 1e-30:
        return False
    l1 = ((y2 - y3) * (p[0] - x3) + (x3 - x2) * (p[1] - y3)) / d
    l2 = ((y3 - y1) * (p[0] - x3) + (x1 - x3) * (p[1] - y3)) / d
    return l1 >= 0 and l2 >= 0 and (1 - l1 - l2) >= 0


counts = []
for xs in np.linspace(-0.42, 0.42, 15):        # rays crossing the TOP wall
    cand = np.where((np.abs(cen[:, 0] - xs) < 0.015) & (cen[:, 1] > 0.47))[0]
    hit = set()
    for yy in np.linspace(0.4851, 0.5149, 300):
        for k in cand:
            if in_tri((xs, yy), xy[conn[k]]):
                hit.add(k)
                break
    counts.append(len(hit))
counts = np.array(counts)
h = np.array([max(np.linalg.norm(xy[e[k]] - xy[e[(k + 1) % len(e)]])
                  for k in range(len(e))) for e in conn])
nouter = len(bnd) // 2
say()
say("MESH DENSITY (target ~4 elements through the wall)")
say("  element edge length  min/mean/max = %.5f / %.5f / %.5f" % (h.min(), h.mean(), h.max()))
say("  t / mean edge length                       = %.2f" % (T_WALL / h.mean()))
say("  element ROWS through thickness")
say("    = n_elem / (2 * outer-loop edges) = %d / (2*%d) = %.2f"
    % (nelem, nouter, nelem / (2.0 * nouter)))
say("  elements clipped by a straight through-thickness line, top wall (15 lines):")
say("    min %d, median %.1f, max %d   (an unstructured triangulation clips ~2x"
    % (counts.min(), np.median(counts), counts.max()))
say("     the row count because the triangles are not column-aligned)")

# --------------------------------------------------------------- read YAML ---
say()
if os.path.exists(YML):
    dat = yaml.safe_load(open(YML))
    eo = dat["elementOrientations"]
    say("OpenSG 2-D solid YAML : %s" % YML)
    say("  top-level keys      = %s" % list(dat.keys()))
    say("  nodes               = %d" % len(dat["nodes"]))
    say("  elements            = %d" % len(dat["elements"]))
    say("  elementOrientations = %d rows x %d components"
        % (len(eo), len(eo[0]) if eo else 0))
    say("  element sets        = %s"
        % [(s["name"], len(s["labels"])) for s in dat["sets"]["element"]])
    say("  materials           = %s" % [m["name"] for m in dat["materials"]])
    ok = (len(eo) == len(dat["elements"])) and all(len(rw) == 9 for rw in eo)
    say("  one 9-component row per element = %s" % ok)
    say()
    say("  first 6 elementOrientations rows  [e1x e1y e1z | e2x e2y e2z | e3x e3y e3z]"
        " (x,y = section plane, z = beam axis):")
    for i in range(6):
        c = cen[i]
        say("   elem %-3d (centroid %+7.4f %+7.4f, theta1=%7.2f, theta3=%6.1f)  "
            "e1=[%+7.4f %+7.4f %+7.4f]  e2=[%+7.4f %+7.4f %+7.4f]  e3=[%+7.4f %+7.4f %+7.4f]"
            % (i + 1, c[0], c[1], theta1[i], theta3[i], *[float(v) for v in eo[i]]))
    E = np.array(eo, float)
    e1, e2, e3 = E[:, 0:3], E[:, 3:6], E[:, 6:9]
    say()
    say("  frame orthonormality: max|e_i.e_i-1| = %.2e, max|e_i.e_j| = %.2e, "
        "max|e1 x e2 - e3| = %.2e"
        % (max(np.abs((e1 * e1).sum(1) - 1).max(), np.abs((e2 * e2).sum(1) - 1).max(),
               np.abs((e3 * e3).sum(1) - 1).max()),
           max(np.abs((e1 * e2).sum(1)).max(), np.abs((e1 * e3).sum(1)).max(),
               np.abs((e2 * e3).sum(1)).max()),
           np.abs(np.cross(e1, e2) - e3).max()))
    say("  e3 (ply normal) is purely in-plane: max|e3_z| = %.2e" % np.abs(e3[:, 2]).max())
    say("  e1 out-of-plane (axial) component  = %.4f (uniform: %s)"
        % (e1[:, 2].mean(), np.allclose(e1[:, 2], e1[0, 2])))
    say("  e2 out-of-plane (axial) component  = %.4f" % e2[:, 2].mean())
    # is e3 outward or inward?
    dotn = (e3[:, 0] * cen[:, 0] + e3[:, 1] * cen[:, 1])
    say("  e3 . (centroid) sign: %d outward, %d inward" % ((dotn > 0).sum(), (dotn < 0).sum()))
else:
    say("!! YAML not found: %s" % YML)
    e1 = e2 = e3 = None

# -------------------------------------------------------------------- plots ---
polys = [xy[e] for e in conn]

fig = plt.figure(figsize=(12.6, 6.2))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.25)
ax = fig.add_subplot(gs[0, 0])
ax.add_collection(PolyCollection(polys, facecolors="#9ec5e8",
                                 edgecolors="#1f4e79", linewidths=0.25))
ax.set_xlim(-0.545, 0.545)
ax.set_ylim(-0.545, 0.545)
ax.set_aspect("equal")
ax.set_xlabel(r"$y_2$")
ax.set_ylabel(r"$y_3$")

axi = fig.add_subplot(gs[0, 1])
axi.add_collection(PolyCollection(polys, facecolors="#9ec5e8",
                                  edgecolors="#1f4e79", linewidths=0.7))
axi.set_xlim(0.455, 0.525)
axi.set_ylim(0.455, 0.525)
axi.set_aspect("equal")
axi.set_xlabel(r"$y_2$")
axi.set_ylabel(r"$y_3$")
# mark the midline (layup reference) in the zoom
for a in (ax, axi):
    a.plot([-0.5, 0.5, 0.5, -0.5, -0.5], [0.5, 0.5, -0.5, -0.5, 0.5],
           "--", color="#c00000", lw=1.0, zorder=5)
axi.text(0.457, 0.503, "midline (layup reference)", color="#c00000", fontsize=8)
fig.savefig(os.path.join(D, "square_tube_mesh.png"), dpi=220, bbox_inches="tight")
plt.close(fig)
say()
say("wrote %s" % os.path.join(D, "square_tube_mesh.png"))

if e1 is not None:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(15.0, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.25, 0.95], wspace=0.30)

    vecs = ((e1, "#d62728", r"$\mathbf{e}_1$ (fibre)"),
            (e2, "#2ca02c", r"$\mathbf{e}_2$"),
            (e3, "#1f77b4", r"$\mathbf{e}_3$ (ply normal)"))

    # (a) whole section, sparse
    axa = fig.add_subplot(gs[0, 0])
    axa.add_collection(PolyCollection(polys, facecolors="#f4f4f4",
                                      edgecolors="#c8c8c8", linewidths=0.25))
    idx = np.arange(0, nelem, 55)
    for v, col, lab in vecs:
        axa.quiver(cen[idx, 0], cen[idx, 1], v[idx, 0], v[idx, 1], color=col,
                   angles="xy", scale_units="xy", scale=1 / 0.045,
                   width=0.007, label=lab)
    axa.set_xlim(-0.58, 0.58)
    axa.set_ylim(-0.58, 0.58)
    axa.set_aspect("equal")
    axa.set_xlabel(r"$y_2$")
    axa.set_ylabel(r"$y_3$")
    axa.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False)

    # (b) zoom: every element of a short piece of the TOP wall
    axb = fig.add_subplot(gs[0, 1])
    axb.add_collection(PolyCollection(polys, facecolors="#f4f4f4",
                                      edgecolors="#a8a8a8", linewidths=0.5))
    m = (np.abs(cen[:, 0]) < 0.075) & (cen[:, 1] > 0.47)
    idx = np.where(m)[0]
    for v, col, lab in vecs:
        axb.quiver(cen[idx, 0], cen[idx, 1], v[idx, 0], v[idx, 1], color=col,
                   angles="xy", scale_units="xy", scale=1 / 0.0075,
                   width=0.005)
    axb.plot([-0.08, 0.08], [0.5, 0.5], "--", color="#c00000", lw=1.0)
    axb.set_xlim(-0.08, 0.08)
    axb.set_ylim(0.478, 0.522)
    axb.set_aspect("equal")
    axb.set_xlabel(r"$y_2$")
    axb.set_ylabel(r"$y_3$")

    # (c) 3-D triad of one top-wall element (shows the out-of-plane / axial tilt)
    k = int(idx[len(idx) // 2])
    axc = fig.add_subplot(gs[0, 2], projection="3d")
    tri = xy[conn[k]]
    for j in range(3):
        p, q = tri[j], tri[(j + 1) % 3]
        axc.plot([p[0], q[0]], [p[1], q[1]], [0, 0], color="#808080", lw=1.0)
    c = cen[k]
    for v, col, lab in vecs:
        axc.quiver(c[0], c[1], 0.0, v[k, 0], v[k, 1], v[k, 2],
                   color=col, length=0.03, normalize=True, arrow_length_ratio=0.18)
    axc.quiver(c[0], c[1], 0.0, 0, 0, 1, color="k", length=0.03,
               normalize=True, arrow_length_ratio=0.18, linestyle=":")
    axc.text(c[0], c[1], 0.034, r"$y_1$ (beam axis)", fontsize=8)
    axc.text(c[0] + 0.023 * e1[k, 0], c[1] + 0.023 * e1[k, 1], 0.023 * e1[k, 2],
             r"$e_1$", color="#d62728", fontsize=10)
    axc.text(c[0] + 0.023 * e2[k, 0], c[1] + 0.023 * e2[k, 1], 0.023 * e2[k, 2],
             r"$e_2$", color="#2ca02c", fontsize=10)
    axc.text(c[0] + 0.023 * e3[k, 0], c[1] + 0.023 * e3[k, 1], 0.023 * e3[k, 2],
             r"$e_3$", color="#1f77b4", fontsize=10)
    axc.set_xlim(c[0] - 0.035, c[0] + 0.035)
    axc.set_ylim(c[1] - 0.035, c[1] + 0.035)
    axc.set_zlim(-0.035, 0.035)
    axc.set_box_aspect((1, 1, 1))
    axc.set_xlabel(r"$y_2$", labelpad=-4)
    axc.set_ylabel(r"$y_3$", labelpad=-4)
    axc.set_zlabel(r"$y_1$", labelpad=-4)
    axc.tick_params(labelsize=6, pad=-2)
    axc.view_init(elev=22, azim=-58)

    fig.savefig(os.path.join(D, "square_tube_orientation.png"), dpi=220,
                bbox_inches="tight")
    plt.close(fig)
    say("wrote %s" % os.path.join(D, "square_tube_orientation.png"))

open(os.path.join(D, "square_tube_checks.txt"), "w").write("\n".join(out) + "\n")
print("\nwrote", os.path.join(D, "square_tube_checks.txt"))
