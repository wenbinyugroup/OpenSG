"""
boundary_from_yaml.py   [ pure numpy / JAX-native -- the PRIMARY boundary path ]
========================================================================
General TOPOLOGICAL boundary extraction for a shell segment (circle, airfoil,
airfoil-with-webs -- any cross-section).  No dolfinx / meshio / .msh.
(extract_boundaries_dolfinx.py is kept only as a reference prototype.)

Method: a mesh edge used by exactly ONE quad is a FREE edge; the connected
components of the free-edge graph are the segment's END CROSS-SECTIONS.  Web
junctions are simply degree>2 nodes -- no special treatment (RM is C0).

Each end is written as a 1-D cross-section YAML matching OpenSG's FEniCS
ShellSegmentMesh._create_1Dyaml:
  nodes  = [cross1, cross2, axial]     (the two in-plane coords + the axial coord)
  elements = 1-indexed line pairs
  sections/sets = all layups, edges grouped by subdomain
  elementOrientations = the PARENT quad's full 9-component orientation per edge
  materials
plus the e1/e2/e3 orientation PNG precheck and an .npz bundle (same schema the
solver consumes) with node2seg = the boundary node's own segment id (identity).
"""

import os
import sys
import json
import numpy as np
import yaml
from collections import Counter, defaultdict

from .orient_check import frame_report, orientation_png, orientation_png_ring


def load_segment_yaml(path):
    try:
        return yaml.load(open(path), Loader=yaml.CLoader)
    except AttributeError:
        return yaml.safe_load(open(path))


def subdomain_ids(seg):
    ne = len(seg["elements"])
    subdom = np.zeros(ne, dtype=np.int32)
    # OpenSG Shell_3D_Taper files use 0-INDEXED element ids in the sets (matching
    # their 0-indexed connectivity); our cylinder generator writes 1-indexed.
    # Detect: a label of 0 can only be 0-indexed.  (The old unconditional lab-1
    # shifted every label one element spanwise, corrupting exactly the LAST strip
    # -> R-boundary ring labels shifted one position along the hoop while L stayed
    # correct -- see ref_bar_urc_shell_spar_mislabel.)
    labs_min = min(min(es["labels"]) for es in seg["sets"]["element"] if es["labels"])
    off = 0 if labs_min == 0 else 1
    for i, es in enumerate(seg["sets"]["element"]):
        for lab in es["labels"]:
            subdom[lab - off] = i
    return subdom


def _free_edges(quads):
    """free edge = used by exactly one quad; returns (free list, edge->owner-quad)."""
    cnt = Counter(); owner = {}
    for qi, q in enumerate(quads):
        m = len(q)
        for a in range(m):
            e = tuple(sorted((int(q[a]), int(q[(a + 1) % m]))))
            cnt[e] += 1; owner.setdefault(e, qi)
    free = [e for e, c in cnt.items() if c == 1]
    return free, owner


def _components(free):
    """connected components (node lists) of the free-edge graph."""
    adj = defaultdict(list)
    for a, b in free:
        adj[a].append(b); adj[b].append(a)
    seen, comps = set(), []
    for n in adj:
        if n in seen:
            continue
        st, comp = [n], []
        while st:
            u = st.pop()
            if u in seen:
                continue
            seen.add(u); comp.append(u)
            st += [v for v in adj[u] if v not in seen]
        comps.append(comp)
    return comps, adj


def _write_boundary_yaml(comp, oedges, oq, nodes, ori, subdom, sections, ax_idx, materials, path):
    """Write one end cross-section as a 1-D YAML in the FEniCS _create_1Dyaml layout.
    `oedges` = oriented (tangent~e2) edges; `oq` = parent quad per edge."""
    loc = {n: i for i, n in enumerate(comp)}
    cross = [j for j in range(3) if j != ax_idx]
    d_nodes = [[float(nodes[n, cross[0]]), float(nodes[n, cross[1]]), float(nodes[n, ax_idx])] for n in comp]
    d_elems = [[loc[a] + 1, loc[b] + 1] for (a, b) in oedges]
    edge_sub = [int(subdom[q]) for q in oq]
    edge_ori = [[float(v) for v in ori[q]] for q in oq]                    # parent quad 9-comp
    # sets: 1-indexed edge labels grouped by layup name
    lay_names = [s["elementSet"] for s in sections]
    sets = []
    for si, name in enumerate(lay_names):
        labs = [i + 1 for i, sd in enumerate(edge_sub) if sd == si]
        sets.append({"name": name, "labels": labs})
    d = {
        "nodes": d_nodes,
        "elements": d_elems,
        "sets": {"element": sets},
        "sections": sections,
        "elementOrientations": edge_ori,
        "materials": materials,
    }
    with open(path, "w") as f:
        yaml.safe_dump(d, f, default_flow_style=None, sort_keys=False)
    return len(comp), len(oedges)


def extract(seg_yaml, out_npz, write_yaml=False, plot="auto"):
    """Extract the two end cross-sections into an .npz bundle (the solver runs the
    boundary Timoshenko IN-MEMORY from this bundle -- see solve_boundary_bundle).
    write_yaml=True additionally writes each end as a 1-D cross-section YAML
    (FEniCS _create_1Dyaml layout) -- only needed for inspection / FEniCS diff.
    plot: True = always draw the e1/e2/e3 orientation PNGs; 'auto' (default) = draw
    only the ones not already on disk (the matplotlib 3-D quivers cost ~1.5 s and
    dominated repeated solves); False = never."""
    seg = load_segment_yaml(seg_yaml)
    nodes = np.array(seg["nodes"], dtype=float)
    quads = [list(map(int, e)) for e in seg["elements"]]
    if min(min(q) for q in quads) == 1:                    # 1-indexed YAML (our cylinder) -> 0-indexed (OpenSG)
        quads = [[n - 1 for n in q] for q in quads]
    ori = np.array(seg["elementOrientations"], dtype=float)     # (Ne,9)
    e1s, e2s, e3s = ori[:, 0:3], ori[:, 3:6], ori[:, 6:9]
    subdom = subdomain_ids(seg)
    sections = seg["sections"]; materials = seg["materials"]
    out_dir = os.path.dirname(out_npz) or "."; os.makedirs(out_dir, exist_ok=True)
    tag = os.path.splitext(os.path.basename(out_npz))[0]

    ok, txt = frame_report(nodes, quads, e1s, e2s, e3s); print("segment " + txt)

    def _want(png):
        return plot is True or (plot == "auto" and not os.path.exists(png))

    png = os.path.join(out_dir, "orient_%s_segment.png" % tag)
    if _want(png):
        orientation_png(nodes, quads, e1s, e2s, e3s, png,
                        title="%s segment e1/e2/e3" % tag, step=max(1, len(quads) // 300))

    free, owner = _free_edges(quads)
    comps, adj = _components(free)
    comps = [c for c in comps if len(c) >= 3]
    print("free edges %d -> %d end cross-section(s)" % (len(free), len(comps)))
    if len(comps) < 2:
        raise RuntimeError("expected 2 end cross-sections, found %d" % len(comps))
    # axis = direction between the two extreme-position components
    cent = np.array([nodes[c].mean(0) for c in comps])
    pair = max(((i, j) for i in range(len(comps)) for j in range(i + 1, len(comps))),
               key=lambda ij: np.linalg.norm(cent[ij[0]] - cent[ij[1]]))
    axis_vec = cent[pair[1]] - cent[pair[0]]; ax_idx = int(np.argmax(np.abs(axis_vec)))
    # left = smaller axial coord, right = larger
    order = sorted(pair, key=lambda i: cent[i, ax_idx])
    ends = {"L": comps[order[0]], "R": comps[order[1]]}
    print("beam axis = %s ; %d/%d nodes on L/R cross-section"
          % ("xyz"[ax_idx], len(ends["L"]), len(ends["R"])))

    freeset = set(free)
    bundle = dict(seg_x=nodes, seg_cells=np.array([q for q in quads], dtype=np.int64), seg_subdom=subdom,
                  seg_e1=e1s, seg_e2=e2s, seg_e3=e3s,
                  materials=json.dumps(materials), sections=json.dumps(sections),
                  axis=ax_idx)
    for side in ("L", "R"):
        comp = ends[side]; cset = set(comp)
        loc = {n: i for i, n in enumerate(comp)}
        # ORIENT each free edge so its tangent aligns with the parent quad's e2
        # (hoop tangent) -- otherwise the sorted-tuple direction is arbitrary and the
        # curvature-coupling terms (k22, macro Rn) flip inconsistently per element.
        oedges, oq = [], []
        for e in free:
            if e[0] in cset and e[1] in cset:
                q = owner[e]
                tv = nodes[e[1]] - nodes[e[0]]
                oedges.append((e[1], e[0]) if float(np.dot(tv, e2s[q])) < 0 else (e[0], e[1]))
                oq.append(q)
        njunc = sum(len(adj[n]) > 2 for n in comp)
        print("  %s cross-section: %d nodes, %d edges, %d junctions(deg>2)" % (side, len(comp), len(oedges), njunc))
        if write_yaml:
            byaml = os.path.join(out_dir, "boundary_%s_%s.yaml" % (tag, side))
            _write_boundary_yaml(comp, oedges, oq, nodes, ori, subdom, sections, ax_idx, materials, byaml)
            print("    (+ wrote 1-D YAML %s)" % os.path.basename(byaml))
        rx = nodes[comp]                                      # (m,3) full 3-D ring coords
        rcells = np.array([[loc[a], loc[b]] for (a, b) in oedges], dtype=np.int64)
        re1 = np.array([e1s[q] for q in oq])                  # full 3-D parent frame (axis-agnostic;
        re2 = np.array([e2s[q] for q in oq])                  # for axis=x the x-comp of e2/e3 is ~0,
        re3 = np.array([e3s[q] for q in oq])                  # so this is unchanged for the cylinder)
        edge_sub = np.array([int(subdom[q]) for q in oq], np.int32)
        okr, txtr = frame_report(rx, rcells, re1, re2, re3); print("  %s-ring %s" % (side, txtr))
        png = os.path.join(out_dir, "orient_%s_ring_%s.png" % (tag, side))
        if _want(png):
            orientation_png_ring(rx, rcells, re2, re3, png,
                                 title="%s %s-ring e2/e3" % (tag, side))
        bundle["%s_x" % side] = rx
        bundle["%s_cells" % side] = rcells
        bundle["%s_subdom" % side] = edge_sub
        bundle["%s_e1" % side] = re1; bundle["%s_e2" % side] = re2; bundle["%s_e3" % side] = re3
        bundle["%s_node2seg" % side] = np.array(comp, dtype=np.int64)
    np.savez(out_npz, **bundle)
    print("wrote", out_npz)
    return out_npz


if __name__ == "__main__":
    # usage: boundary_from_yaml.py <segment.yaml> [out.npz] [--yaml]
    #   --yaml : also write the 1-D boundary YAML files (default: in-memory bundle only)
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    seg_yaml = args[0] if len(args) > 0 else "meshes/seg_iso_hR0.1.yaml"
    out_npz = args[1] if len(args) > 1 else os.path.join(
        "out", os.path.splitext(os.path.basename(seg_yaml))[0] + "_direct.npz")
    extract(seg_yaml, out_npz, write_yaml="--yaml" in sys.argv)
