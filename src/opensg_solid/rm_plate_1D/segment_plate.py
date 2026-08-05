"""segment_plate.py -- MESH-GENERATION HELPER for the through-thickness 1-D plate SG.

A laminate layup is all the geometry the MSG-RM plate law needs: the structure gene is a
1-D mesh through the wall thickness, one (or ``n_per_layer``) higher-order element per ply.
This module ONLY generates and reads that mesh -- it does no homogenization; feed the
result to ``msg_rm_plate.rm_plate_msg``:

    layup dict --plate_sg_yaml--> 1-D SG YAML --read_plate_sg_yaml--> rm_plate_msg(...)

``plate_sg_yaml`` draws the mesh PNG as it generates the SG (``png=False`` to skip),
from the document it has just built -- so the picture costs no file read, and
``read_plate_sg_yaml`` stays a pure reader.  ``plot_plate_sg`` is for a YAML you did
not just write.

Distinguish this from the 1-D SHELL SG YAML (e.g. ``examples/data/1d_yaml/st15_shell.yaml``):
that one is the CONTOUR of a cross-section -- line elements running around the airfoil, each
carrying a layup.  The file written here is the wall itself, discretised THROUGH the
thickness; its nodes are x3 coordinates, not points in the cross-section plane.

Mesh convention (identical to what ``msg_rm_plate`` builds internally, asserted in
``tests/test_segment_plate.py``):
  * ``elem_order = 4`` by default -> 5-NODED QUARTIC elements, the degree that represents
    the whole warping ladder exactly per ply (V0 quadratic, V1 cubic, V2 quartic)
  * ``n_per_layer = 1`` by default -> one element per ply
  * nodes are equispaced inside each element, elements share their end nodes, so
    n_node = elem_order * n_elem + 1
  * THE REFERENCE PLANE IS A PROPERTY OF THE MESH, not of the homogenization call.
    ``fraction`` is given ONCE here, at generation (0 = bottom/OML face, 0.5 =
    mid-surface, 1 = top/IML), and from then on it IS the node coordinates: x3 is
    measured from that plane, so the nodes run from -fraction*h to (1-fraction)*h and
    x3 = 0 is the reference.  A reader can therefore recover the reference from the
    coordinates alone; ``read_plate_sg_yaml`` does exactly that (it prefers the stored
    ``reference_fraction`` for bit-exactness but verifies it against node_x[0], so a
    file whose header and mesh disagree is rejected rather than silently trusted).

YAML layout::

    sg:        type/elem_order/n_per_layer/reference_fraction/thickness
    nodes:     [x3] per node, bottom to top
    elements:  1-based connectivity, (elem_order + 1) nodes each
    materials: name + elastic {E, G, nu} + density   (same block as the shell YAML)
    sets:      element sets, one per PLY (ply_1, ply_2, ...)
    sections:  per ply set: material, angle, thickness
"""
import os

import numpy as np
import yaml

from .msg_rm_plate import _node_grid


def plate_sg_mesh(thick, n_per_layer=1, elem_order=4, fraction=0.5):
    """The through-thickness 1-D mesh for a layup.

    Returns ``(node_x, elements, elem_ply)``:
      node_x    (n_node,)            x3 of each node, measured from the reference plane
      elements  (n_elem, p+1) int    0-based connectivity
      elem_ply  (n_elem,) int        which ply each element belongs to
    """
    thick = [float(t) for t in thick]
    p = int(elem_order)
    n_elem = len(thick) * int(n_per_layer)
    node_x = _node_grid(thick, int(n_per_layer), p, float(fraction) * sum(thick))
    elements = np.stack([np.arange(p * e, p * e + p + 1) for e in range(n_elem)])
    elem_ply = np.repeat(np.arange(len(thick)), int(n_per_layer))
    return node_x, elements, elem_ply


def plate_sg_dict(layup, material_db, n_per_layer=1, elem_order=4, fraction=0.5):
    """Build the 1-D plate SG as a plain dict (the YAML document, unwritten).

    ``layup`` is the layup dictionary used everywhere else in OpenSG::

        {"mat_names": [...], "thick": [...], "angles": [...]}      bottom ply first
    """
    mats = [str(m) for m in layup["mat_names"]]
    thick = [float(t) for t in layup["thick"]]
    angles = [float(a) for a in layup["angles"]]
    if not (len(mats) == len(thick) == len(angles)):
        raise ValueError("layup lists disagree: %d materials, %d thicknesses, %d angles"
                         % (len(mats), len(thick), len(angles)))
    if min(thick) <= 0:
        raise ValueError("ply thicknesses must be positive, got %s" % thick)
    missing = [m for m in mats if m not in material_db]
    if missing:
        raise KeyError("materials not in the database: %s" % ", ".join(sorted(set(missing))))

    node_x, elements, elem_ply = plate_sg_mesh(thick, n_per_layer, elem_order, fraction)
    n_per_layer = int(n_per_layer)

    sets, sections = [], []
    for k in range(len(thick)):
        labels = [int(e) + 1 for e in np.flatnonzero(elem_ply == k)]     # 1-based
        name = "ply_%d" % (k + 1)
        sets.append({"name": name, "labels": labels})
        sections.append({"elementSet": name, "material": mats[k],
                         "angle": angles[k], "thickness": thick[k]})

    used = [m for m in dict.fromkeys(mats)]
    materials = []
    for m in used:
        md = material_db[m]
        entry = {"name": m, "density": float(md.get("rho", 0.0)),
                 "elastic": {"E": [float(v) for v in md["E"]],
                             "G": [float(v) for v in md["G"]],
                             "nu": [float(v) for v in md["nu"]]}}
        # optional human-readable name: the key is a terse handle ("ge25"), this
        # is what a reader of the figure actually needs ("graphite/epoxy, E1/E2 = 25")
        if md.get("full_name"):
            entry["full_name"] = str(md["full_name"])
        materials.append(entry)

    return {"sg": {"type": "plate_1d", "elem_order": int(elem_order),
                   "n_per_layer": n_per_layer, "reference_fraction": float(fraction),
                   "thickness": float(sum(thick)), "n_ply": len(thick)},
            "nodes": [[float(x)] for x in node_x],
            "elements": [[int(n) + 1 for n in row] for row in elements],
            "materials": materials,
            "sets": {"element": sets},
            "sections": sections}


def plate_sg_yaml(path, layup, material_db, n_per_layer=1, elem_order=4, fraction=0.5,
                  png=True, png_path=None):
    """Write the through-thickness 1-D SG YAML for a layup.  Returns the written dict.

    The mesh PNG is drawn here too (``png=False`` to skip), straight from the document
    this call has just built -- so generating an SG costs ZERO reads: neither this
    function nor the caller has to open the file again just to picture it.  Use
    ``plot_plate_sg`` only for a YAML you did not just write.
    """
    doc = plate_sg_dict(layup, material_db, n_per_layer, elem_order, fraction)
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.isdir(d):
        os.makedirs(d)
    with open(path, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=None)
    if png:
        _draw_from_doc(doc, png_path or (os.path.splitext(path)[0] + ".png"),
                       material_db)
    return doc


def read_plate_sg_yaml(path, atol=1e-9):
    """Read a plate 1-D SG YAML back into the arguments ``rm_plate_msg`` takes.

    Returns a dict with keys thick / angles / mat_names / material_db / fraction /
    n_per_layer / elem_order / node_x.  A pure reader -- the mesh PNG is drawn by
    ``plate_sg_yaml`` when the SG is generated, so nothing here re-runs to plot.

    Ply thicknesses and the reference fraction are taken from the stored fields and then
    CROSS-CHECKED against the mesh (summed element spans, node_x[0]) to ``atol`` relative.
    Re-deriving them from the mesh instead costs exactness -- summing node differences
    turns a 0.004 ply into 0.004000000000000002, which propagates to ~1e-14 in the 8x8 --
    while checking keeps the round trip bit-exact AND still catches a corrupt or
    hand-edited file, where mesh and header would genuinely disagree.
    """
    with open(path, "r") as f:
        doc = yaml.safe_load(f)
    sg = doc.get("sg", {})
    if sg.get("type") not in (None, "plate_1d"):
        raise ValueError("%s is a %r SG, not a through-thickness plate_1d SG"
                         % (path, sg.get("type")))

    node_x = np.array([float(np.atleast_1d(nd)[0]) for nd in doc["nodes"]])
    elements = [[int(v) - 1 for v in el] for el in doc["elements"]]
    material_db = {m["name"]: {"E": [float(v) for v in m["elastic"]["E"]],
                               "G": [float(v) for v in m["elastic"]["G"]],
                               "nu": [float(v) for v in m["elastic"]["nu"]],
                               "rho": float(m.get("density", 0.0)),
                               "full_name": m.get("full_name") or m["name"]}
                   for m in doc["materials"]}
    elem_sets = {s["name"]: [int(v) - 1 for v in s["labels"]] for s in doc["sets"]["element"]}

    thick, angles, mat_names, counts = [], [], [], []
    for sec in doc["sections"]:
        eids = elem_sets[sec["elementSet"]]
        span = sum(abs(node_x[elements[e][-1]] - node_x[elements[e][0]]) for e in eids)
        t = float(sec["thickness"]) if "thickness" in sec else float(span)
        if abs(t - span) > atol * max(span, 1e-300):
            raise ValueError("%s: ply %r declares thickness %g but its elements span %g"
                             % (path, sec["elementSet"], t, span))
        thick.append(t)
        angles.append(float(sec["angle"]))
        mat_names.append(str(sec["material"]))
        counts.append(len(eids))
    if len(set(counts)) != 1:
        raise ValueError("%s: plies carry different element counts %s" % (path, counts))

    h = float(sum(thick))
    frac = float(sg["reference_fraction"]) if "reference_fraction" in sg \
        else (float(-node_x[0] / h) if h > 0 else 0.0)
    if h > 0 and abs(-frac * h - node_x[0]) > atol * h:
        raise ValueError("%s: reference_fraction %g puts the bottom face at %g, "
                         "but the first node is at %g" % (path, frac, -frac * h, node_x[0]))
    return {"thick": thick, "angles": angles, "mat_names": mat_names,
            "material_db": material_db, "n_per_layer": counts[0],
            "elem_order": len(elements[0]) - 1, "fraction": frac, "node_x": node_x}


def layup_label(angles, mat_names):
    """The stacking sequence in conventional laminate notation, as a mathtext string.

    Plies of the majority (face) material are named by their fibre angle, anything else
    by its material -- the sandwich convention that writes the core as a token rather
    than a meaningless 0 degrees.  A palindromic stack collapses onto its symmetric half
    with the ``s`` subscript; an odd stack keeps its mid-plane ply as the last token
    (Nayak's own form for this sandwich).  The classical overbar that would mark that
    ply is deliberately NOT drawn -- the figure shows the reference plane as a dotted
    line straight through the mesh, which says the same thing where you can see it::

        [0/90/0]                 -> $[0/90]_s$
        [0/90/0/90/core]s        -> $[0/90/0/90/\\mathrm{core}]_s$
        [0/45/-45/90] (no symm.) -> $[0/45/{-}45/90]$
    """
    face = max(set(mat_names), key=mat_names.count)
    # "{-}45", not "-45": braces make the minus an ordinary symbol, so mathtext
    # stops spacing it as the binary operator it would otherwise read after "/".
    tok = [("%g" % a).replace("-", "{-}") if m == face else ("\\mathrm{%s}" % m)
           for a, m in zip(angles, mat_names)]
    n = len(tok)
    if n > 1 and tok == tok[::-1]:
        return "$[%s]_s$" % "/".join(tok[:(n + 1) // 2])
    return "$[%s]$" % "/".join(tok)


def _draw_from_doc(doc, png_path, material_db=None):
    """Draw the mesh straight out of an SG document (the dict that IS the YAML).

    Used by ``plate_sg_yaml`` on the document it just built -- no file is reopened --
    and by ``plot_plate_sg`` on one it has just loaded.
    """
    node_x = np.array([float(np.atleast_1d(nd)[0]) for nd in doc["nodes"]])
    elements = [[int(v) - 1 for v in el] for el in doc["elements"]]
    elem_sets = {s["name"]: [int(v) - 1 for v in s["labels"]]
                 for s in doc["sets"]["element"]}
    section_sets = [s["elementSet"] for s in doc["sections"]]
    mat_names = [str(s["material"]) for s in doc["sections"]]
    angles = [float(s["angle"]) for s in doc["sections"]]
    if material_db is None:                       # names carried in the file itself
        material_db = {m["name"]: {"full_name": m.get("full_name") or m["name"]}
                       for m in doc["materials"]}
    return _draw_plate_sg(png_path, node_x, elements, elem_sets, section_sets,
                          mat_names, angles, material_db)


def _draw_plate_sg(png_path, node_x, elements, elem_sets, section_sets,
                   mat_names, angles, material_db=None):
    """Draw the 1-D through-thickness mesh from ALREADY-PARSED arrays and save it.

    Deliberately plain: each element is the line segment between its own end nodes,
    coloured BY MATERIAL, with the stacking sequence above a legend that names the
    materials against their colours.  No axes -- the picture is the layup, not a
    measurement -- and no legend entry for the nodes, which the small dots on the mesh
    speak for themselves.  Legend text is the material's ``full_name`` when the database
    carries one, since the dictionary key is a handle ("ge25") and not information.
    Returns the PNG path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ply_of = {e: k for k, nm in enumerate(section_sets) for e in elem_sets[nm]}
    mats = list(dict.fromkeys(mat_names))             # unique, in stacking order
    cmap = plt.get_cmap("tab10")
    colr = {m: cmap(i % 10) for i, m in enumerate(mats)}
    z = 1e3 * np.asarray(node_x)                      # mm

    fig, ax = plt.subplots(figsize=(2.1, 5.2))
    for e, el in enumerate(elements):                 # each element = one line segment
        ax.plot([0.0, 0.0], [z[el[0]], z[el[-1]]], "-",
                color=colr[mat_names[ply_of[e]]], lw=6.5,
                solid_capstyle="butt", zorder=1)
    ax.plot(np.zeros_like(z), z, ".", ms=2.5, color="0.15", zorder=3)
    for m in mats:
        label = (material_db or {}).get(m, {}).get("full_name") or m
        ax.plot([], [], "-", color=colr[m], lw=8, label=label)
    # x3 = 0 IS the reference plane by construction of the mesh -- draw it, rather
    # than encoding it as an overbar in the stacking sequence (no legend entry:
    # a dotted line straight through the mid-plane needs no caption)
    ax.axhline(0.0, ls=":", color="k", lw=1.3, zorder=4)

    ax.set_xlim(-0.05, 0.05)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False,
              fontsize=9, title=layup_label(angles, mat_names),
              title_fontsize=11)
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return png_path


def plot_plate_sg(path, png_path=None):
    """Draw the mesh of an EXISTING plate SG YAML.  Returns the PNG path.

    Only needed for a file you did not just write -- ``plate_sg_yaml`` already draws
    the PNG as it generates the SG, with no read at all.  One parse here, not the
    three this used to cost.
    """
    with open(path, "r") as f:
        doc = yaml.safe_load(f)
    return _draw_from_doc(doc, png_path or (os.path.splitext(path)[0] + ".png"))


