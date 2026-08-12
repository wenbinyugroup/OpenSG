"""sg_mesh.py -- SG input loaders and mesh utilities: 1-D ring YAML loaders with
explicit laminate reference conventions, 3-D segment loading + topological
boundary extraction into solver .npz bundles, and orientation-frame numeric
reports and PNG prechecks.

Merged from: xsec_5v6_master.py, oml_ring.py, boundary_from_yaml.py, orient_check.py

xsec_5v6_master
---------------
Cross-section Timoshenko 6x6 for the RM cross-section paper.

Two RM cross-section (boundary-ring) formulations, BOTH tying only gamma_23:
  A = 6-DOF independent-omega3 element + element-wise LAGRANGE multiplier enforcing the
      drilling (omega3) in-plane constraint         -> sg_homo.ring_indep(shear="mitc4_g23")
  B = 5-DOF drilling-ELIMINATED MITC element        -> sg_assembly.ring_general(shear="mitc4_g23")
compared against the 2-D solid cross-section 6x6 (VABS-convention .txt), per Timo term.

All rings referenced at the contour CENTROID (center_ref=True), matching the centroidal
2-D solid reference.

Public entry points: load_ring, load_solid, ring_6dof, ring_5dof, pct, show.

oml_ring
--------
Ring loader with an explicit laminate reference convention.

'center'   : laminate centred on the contour line (mid-surface meshes: tubes, ellipse)
'oml'      : contour on the OML, laminate stacked INWARD from the line (airfoil yamls:
             IEA-22, st15/BAR-URC -- official OpenSG shell frac=0 convention)
'oml_flip' : reference shifted the full thickness the other way (diagnostic only)
'iml'      : contour on the INNER mold line (a fraction=1.0 yaml), laminate stacked
             OUTWARD from the line -- same full-thickness ABD shift as oml_flip

boundary_from_yaml   [ pure numpy / JAX-native -- the PRIMARY boundary path ]
-----------------------------------------------------------------------------
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

orient_check   [ pure numpy + matplotlib, no dolfinx ]
------------------------------------------------------
Mandatory ORIENTATION PRECHECK for the surface-quad / boundary meshes.

Because dolfinx renumbers nodes AND cells, the per-element material frame
(e1,e2,e3) must be verified AFTER any mesh handling.  These helpers:
  * `frame_report`   -- numeric check: orthonormality, right-handedness, and
                        that e1 is axial (x) and e3 is the inward radial normal
                        for the cylinder; returns a pass/fail summary string;
  * `orientation_png`-- ALWAYS-emitted arrow plot of e1/e2/e3 at element
                        centroids so the orientation can be eyeballed.

Convention (matches the project figure defaults): e2 blue, e3 black; e1 red.
"""
import os
import sys
import json
from collections import Counter, defaultdict

import numpy as np
import yaml

try:                                    # libyaml C loader: ~5x faster on big SG yamls
    from yaml import CSafeLoader as _YLoader
except ImportError:
    from yaml import SafeLoader as _YLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

from .sg_assembly import compute_k22
from .sg_assembly import ring_general
from .sg_assembly import require_quad_mesh
from .sg_materials import _material_by_section
from .fe_jax.msg_materials import shift_abd_reference

# NOTE: `from .sg_homo import ring_indep` would be circular at module import time
# (sg_homo imports sg_mesh); it is imported inside ring_6dof() and c6() instead.


# ----------------------------------------------------------------------
# from xsec_5v6_master.py
# ----------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
TWP = os.path.abspath(os.path.join(HERE, ".."))

LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
OUT = os.path.join(HERE, "results")


def _row(r):
    if isinstance(r, list):
        r = r[0] if (len(r) == 1 and isinstance(r[0], str)) else r
    if isinstance(r, str):
        return [float(v) for v in r.replace(",", " ").split()]
    return [float(v) for v in r]


def _norm_materials(mats):
    out = []
    for mm in mats:
        if "elastic" in mm:
            out.append(mm)
        else:
            out.append({"name": mm["name"], "elastic": {"E": mm["E"], "G": mm["G"], "nu": mm["nu"]}})
    return out


def load_ring(path, center_ref=True):
    """1-D contour shell YAML -> ring arrays (rx, cells, rsub, re3, D_by, G_by, k22, ax, cross).
    center_ref=True references the plate ABD to the laminate MID-surface (default, for a mid-surface
    contour); center_ref=False references it to the OML (use with an OML contour, fraction=0.0)."""
    d = yaml.safe_load(open(path))
    rx = np.array([_row(r)[:3] for r in d["nodes"]], dtype=float)
    cells = np.array([[int(v) for v in _row(e)] for e in d["elements"]], dtype=int)
    if cells.min() == 1:
        cells = cells - 1
    ori = np.array([_row(o) for o in d["elementOrientations"]], dtype=float)
    re2, re3 = ori[:, 3:6], ori[:, 6:9]
    sections = d["sections"]; materials = _norm_materials(d["materials"])
    setname_to_sec = {s["elementSet"]: i for i, s in enumerate(sections)}
    rsub = np.zeros(len(cells), dtype=int)
    for grp in d["sets"]["element"]:
        si = setname_to_sec[grp["name"]]
        for lab in grp["labels"]:
            rsub[int(lab) - 1] = si
    ax = 2; cross = [0, 1]
    D_by, G_by = _material_by_section(sections, materials, center_ref=center_ref)
    k22 = compute_k22(rx[cells].mean(1), re2, re3, cells)
    return dict(rx=rx, cells=cells, rsub=rsub, re3=re3, D_by=D_by, G_by=G_by, k22=k22,
                ax=ax, cross=cross, nweb=int((rsub > 0).sum()))


def load_solid(path):
    return 0.5 * (np.loadtxt(path) + np.loadtxt(path).T)


def ring_6dof(R):
    from .sg_homo import ring_indep
    return 0.5 * (lambda C: C + C.T)(ring_indep(R["rx"], R["cells"], R["rsub"], R["re3"],
                 R["D_by"], R["G_by"], R["k22"], R["ax"], R["cross"],
                 shear="mitc4_g23", lam_space="elem"))


def ring_5dof(R):
    C, _, _ = ring_general(R["rx"], R["cells"], R["rsub"], R["re3"], R["D_by"], R["G_by"],
                           R["k22"], R["ax"], R["cross"], shear="mitc4_g23")
    return 0.5 * (C + C.T)


def pct(S, ref):
    cut = np.abs(ref).max() / 1e3
    E = np.full((6, 6), np.nan)
    m = np.abs(ref) > cut
    E[m] = 100.0 * (S[m] - ref[m]) / ref[m]
    return E


def show(name, S, So, C6, C5):
    print("\n################ %s ################" % name)
    print("2-D solid diag :", "  ".join("%s=%.4g" % (LBL[i], So[i, i]) for i in range(6)))
    print("6-DOF (Lagrange, g23) diag:", "  ".join("%.4g" % C6[i, i] for i in range(6)))
    print("5-DOF (MITC,     g23) diag:", "  ".join("%.4g" % C5[i, i] for i in range(6)))
    for tagn, C in (("6-DOF Lagrange g23", C6), ("5-DOF MITC g23", C5)):
        e = [100.0 * (C[i, i] - So[i, i]) / So[i, i] for i in range(6)]
        print("  %-20s %%err diag: %s" % (tagn, "  ".join("%s %+6.2f" % (LBL[i], e[i]) for i in range(6))))


# ----------------------------------------------------------------------
# from oml_ring.py
# ----------------------------------------------------------------------

def load_ring_ref(path, ref="oml"):
    d = yaml.load(open(path), Loader=_YLoader)
    rx = np.array([_row(r)[:3] for r in d["nodes"]], dtype=float)
    cells = np.array([[int(v) for v in _row(e)] for e in d["elements"]], dtype=int)
    if cells.min() == 1:
        cells = cells - 1
    ori = np.array([_row(o) for o in d["elementOrientations"]], dtype=float)
    re3 = ori[:, 6:9]
    sections = d["sections"]; materials = _norm_materials(d["materials"])
    setname_to_sec = {s["elementSet"]: i for i, s in enumerate(sections)}
    rsub = np.zeros(len(cells), dtype=int)
    for grp in d["sets"]["element"]:
        si = setname_to_sec[grp["name"]]
        for lab in grp["labels"]:
            rsub[int(lab) - 1] = si
    if ref == "center":
        D_by, G_by = _material_by_section(sections, materials, center_ref=True)
    else:
        D_by, G_by = _material_by_section(sections, materials, center_ref=False)
        if ref in ("oml_flip", "iml"):
            for si, sec in enumerate(sections):
                t = sum(float(p[1]) for p in sec["layup"])
                D_by[si] = shift_abd_reference(np.asarray(D_by[si]), t)
    k22 = compute_k22(rx[cells].mean(1), ori[:, 3:6], re3, cells)
    return dict(rx=rx, cells=cells, rsub=rsub, re3=re3, D_by=D_by, G_by=G_by,
                k22=k22, ax=2, cross=[0, 1])


def c6(R):
    from .sg_homo import ring_indep
    C = ring_indep(R["rx"], R["cells"], R["rsub"], R["re3"], R["D_by"], R["G_by"],
                   R["k22"], R["ax"], R["cross"], shear="mitc4_g23", lam_space="elem")
    return 0.5 * (C + C.T)


def derr(C, So):
    return np.array([100.0 * (C[i, i] - So[i, i]) / So[i, i] for i in range(6)])


# ----------------------------------------------------------------------
# from boundary_from_yaml.py
# ----------------------------------------------------------------------

def load_segment_yaml(path):
    """Load a segment YAML file, preferring the fast C loader when available.

    In:
        path: str -- path to the segment YAML file.
    Out:
        dict -- parsed YAML (nodes, elements, sets, sections,
        elementOrientations, materials).
    """
    try:
        return yaml.load(open(path), Loader=yaml.CLoader)
    except AttributeError:
        return yaml.safe_load(open(path))


def subdomain_ids(seg):
    """Map every element to the index of the element set (layup) that contains it.

    In:
        seg: dict -- parsed segment YAML with 'elements' and 'sets'/'element'.
    Out:
        np.ndarray int32 (Ne,) -- element-set (subdomain) index per element.
    """
    ne = len(seg["elements"])
    subdom = np.zeros(ne, dtype=np.int32)
    # Element-set labels may be 0- or 1-indexed; auto-detect (a label of 0 can only be 0-indexed).
    labs_min = min(min(es["labels"]) for es in seg["sets"]["element"] if es["labels"])
    off = 0 if labs_min == 0 else 1
    for i, es in enumerate(seg["sets"]["element"]):
        for lab in es["labels"]:
            subdom[lab - off] = i
    return subdom


def _free_edges(quads):
    """Find the free edges of the mesh: edges used by exactly one quad.

    In:
        quads: list of list[int] -- 0-indexed node ids per element (any length).
    Out:
        free: list of (int, int) -- sorted node-id pairs of the free edges.
        owner: dict {(int, int): int} -- edge pair -> owning quad index.
    """
    cnt = Counter(); owner = {}
    for qi, q in enumerate(quads):
        m = len(q)
        for a in range(m):
            e = tuple(sorted((int(q[a]), int(q[(a + 1) % m]))))
            cnt[e] += 1; owner.setdefault(e, qi)
    free = [e for e, c in cnt.items() if c == 1]
    return free, owner


def _components(free):
    """Compute the connected components of the free-edge graph.

    In:
        free: list of (int, int) -- free-edge node pairs.
    Out:
        comps: list of list[int] -- node ids of each connected component.
        adj: dict {int: list[int]} -- node -> neighbouring node ids.
    """
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

    In:
        comp: list[int] -- node ids of this cross-section.
        oedges: list of (int, int) -- oriented edges (tangent aligned with parent e2).
        oq: list[int] -- parent quad index per edge.
        nodes: np.ndarray (N,3) -- global node coordinates.
        ori: np.ndarray (Ne,9) -- 9-component orientation per parent quad.
        subdom: np.ndarray int32 (Ne,) -- element-set index per quad.
        sections: list[dict] -- section (layup) definitions from the segment YAML.
        ax_idx: int -- beam-axis coordinate index (0/1/2).
        materials: list[dict] -- material definitions from the segment YAML.
        path: str -- output YAML path.
    Out:
        (int, int) -- (node count, edge count) written.
    """
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
        # msg-shell SG header: an end cross-section is a 1-D ring, so it is a
        # runnable SG in its own right -- `opensg_shell boundary_<tag>_L.yaml`
        # returns that ring's own Timoshenko 6x6 (the C6L/C6R the segment run
        # reports).  The ring IS periodic along the beam axis, so no
        # `aperiodic:` key here -- only the segment that owns it carries one.
        f.write("n_model: 1      # 1 = beam, 2 = plate (opensg_solid),"
                " 3 = solid -- the macro model this SG homogenizes to\n"
                "refined: 1      # 0 = classical (beam EB 4x4, KL wall);"
                " 1 = shear-refined (beam Timoshenko 6x6, RM wall)\n"
                "msg: shell      # the ENGINE this SG belongs to"
                " (opensg_shell); `opensg <yaml>` dispatches on it\n")
        yaml.safe_dump(d, f, default_flow_style=None, sort_keys=False)
    return len(comp), len(oedges)


def extract(seg_yaml, out_npz, write_yaml=False, plot="auto"):
    """Extract the two end cross-sections of a shell segment into an .npz bundle;
    the solver runs the boundary Timoshenko in-memory from it (solve_boundary_bundle).

    In:
        seg_yaml: str -- path to the segment YAML.
        out_npz: str -- output .npz bundle path.
        write_yaml: bool -- also write each end as a 1-D cross-section YAML
            (FEniCS _create_1Dyaml layout); only needed for inspection/diff.
        plot: True | 'auto' | False -- draw the e1/e2/e3 orientation PNGs
            always / only when not already on disk / never.
    Out:
        str -- out_npz path; bundle keys: seg_x/_cells/_subdom/_e1/_e2/_e3,
        {L,R}_x/_cells/_subdom/_e1/_e2/_e3/_node2seg, materials, sections, axis.
    """
    seg = load_segment_yaml(seg_yaml)
    nodes = np.array(seg["nodes"], dtype=float)
    quads = [list(map(int, e)) for e in seg["elements"]]
    # The segment solver behind this bundle is 4-node only, and seg_cells is ONE
    # rectangular array: say so here rather than let numpy raise "inhomogeneous
    # shape" (mixed mesh) or the assembler "cannot reshape ... into (ne,24)"
    # (all-triangle) several hundred lines downstream.
    require_quad_mesh(quads, "sg_mesh.extract")
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
        # Orient each free edge so its tangent aligns with the parent quad's e2 (hoop);
        # otherwise curvature-coupling terms (k22, macro Rn) flip inconsistently per element.
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


# ----------------------------------------------------------------------
# from orient_check.py
# ----------------------------------------------------------------------

E1_COL, E2_COL, E3_COL = "#e31a1c", "#1f78b4", "#000000"   # red / blue / black


def frame_report(nodes, cells, e1, e2, e3, tol=1e-6):
    """Numeric per-element frame check: unit norms, orthogonality, right-handedness.
    Expectations are CYLINDER-specific: e1 must be axial (+x) and e3 the inward
    radial normal (toward the axis in the y-z plane).

    In:
        nodes: (N,3) float array of mesh node coordinates.
        cells: iterable of per-element node-index sequences (C elements).
        e1, e2, e3: (C,3) per-element frame vectors.
        tol: float, accepted but unused (pass thresholds are hard-coded 1e-6/0.99/0.999).
    Out:
        ok: bool, True if all checks pass.
        txt: str, one-line summary ending in "OK" or "CHECK".
    """
    e1 = np.asarray(e1); e2 = np.asarray(e2); e3 = np.asarray(e3)
    cent = np.array([nodes[list(c)].mean(axis=0) for c in cells])
    n1 = np.linalg.norm(e1, axis=1); n2 = np.linalg.norm(e2, axis=1); n3 = np.linalg.norm(e3, axis=1)
    orth = np.max(np.abs([np.sum(e1 * e2, 1), np.sum(e2 * e3, 1), np.sum(e1 * e3, 1)]))
    rh = np.sum(np.cross(e1, e2) * e3, axis=1)                 # e1 x e2 . e3 (=+1 right-handed)
    axial = np.min(e1[:, 0] / np.clip(n1, 1e-30, None))        # e1 . x_hat  (expect ~1: e1 = axial)
    r_hat = cent[:, 1:] / np.clip(np.linalg.norm(cent[:, 1:], axis=1)[:, None], 1e-30, None)
    e3_dot_inward = np.mean(np.sum(e3[:, 1:] * (-r_hat), axis=1))   # e3 . (-r_hat) (expect ~1: inward)
    ok = (abs(n1.mean() - 1) < 1e-6 and abs(n2.mean() - 1) < 1e-6 and abs(n3.mean() - 1) < 1e-6
          and orth < 1e-6 and rh.min() > 0.99 and axial > 0.999)
    txt = ("frame: |e1|=%.4f |e2|=%.4f |e3|=%.4f  max|off-diag|=%.1e  "
           "RH(e1xe2.e3) in [%.3f,%.3f]  min(e1.x)=%.4f  mean(e3.-r)=%.4f  -> %s"
           % (n1.mean(), n2.mean(), n3.mean(), orth, rh.min(), rh.max(), axial,
              e3_dot_inward, "OK" if ok else "CHECK"))
    return ok, txt


def _set_equal_3d(ax, pts):
    """Set an equal-aspect 3-D box on the axes from the data extents.

    In:
        ax: matplotlib 3-D Axes.
        pts: (N,3) array; per-axis ranges become the box aspect (zero range -> 1).
    Out:
        None (mutates ax).
    """
    r = (pts.max(0) - pts.min(0)); r[r == 0] = 1.0
    ax.set_box_aspect(r)


def orientation_png(nodes, cells, e1, e2, e3, out_png, title="", step=1, scale=None):
    """3-D quiver plot of the per-cell (e1,e2,e3) frame at element centroids.

    In:
        nodes: (N,3) node coordinates.
        cells: iterable of per-element node-index sequences (C elements).
        e1, e2, e3: (C,3) per-element frame vectors (red/blue/black arrows).
        out_png: str, output PNG path.
        title: str, axes title.
        step: int, plot every step-th element.
        scale: float or None, arrow length; None -> 12% of the bbox diagonal.
    Out:
        str: out_png (file written at 130 dpi, figure closed).
    """
    nodes = np.asarray(nodes)
    cent = np.array([nodes[list(c)].mean(axis=0) for c in cells])
    if scale is None:
        scale = 0.12 * np.linalg.norm(nodes.max(0) - nodes.min(0))
    sel = slice(None, None, step)
    fig = plt.figure(figsize=(9, 7)); ax = fig.add_subplot(111, projection="3d")
    for vec, col, lab in [(e1, E1_COL, "e1 axial"), (e2, E2_COL, "e2 hoop"), (e3, E3_COL, "e3 normal")]:
        v = np.asarray(vec)[sel]
        ax.quiver(cent[sel, 0], cent[sel, 1], cent[sel, 2], v[:, 0], v[:, 1], v[:, 2],
                  length=scale, color=col, label=lab, normalize=True, linewidth=0.6)
    ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], s=1, c="#bbbbbb", alpha=0.35)
    ax.set_xlabel("x (axial)"); ax.set_ylabel("y"); ax.set_zlabel("z"); ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    _set_equal_3d(ax, nodes)
    fig.savefig(out_png, dpi=130, bbox_inches="tight"); plt.close(fig)
    return out_png


def orientation_png_ring(ring_nodes, ring_cells, e2, e3, out_png, title=""):
    """2-D (y,z) quiver of a boundary ring's in-plane e2/e3 frame at edge midpoints.

    In:
        ring_nodes: (N,3) ring node coordinates.
        ring_cells: iterable of per-edge node-index sequences (C edges).
        e2, e3: (C,3) per-edge frame vectors; only their y,z components are drawn.
        out_png: str, output PNG path.
        title: str, axes title.
    Out:
        str: out_png (file written at 130 dpi, figure closed).
    """
    rn = np.asarray(ring_nodes)
    mid = np.array([rn[list(c)].mean(axis=0) for c in ring_cells])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(rn[:, 1], rn[:, 2], ".", ms=2, c="#bbbbbb")
    e2 = np.asarray(e2); e3 = np.asarray(e3)
    ax.quiver(mid[:, 1], mid[:, 2], e2[:, 1], e2[:, 2], color=E2_COL, label="e2 hoop",
              scale=25, width=0.004)
    ax.quiver(mid[:, 1], mid[:, 2], e3[:, 1], e3[:, 2], color=E3_COL, label="e3 normal",
              scale=25, width=0.004)
    ax.set_aspect("equal"); ax.set_xlabel("y"); ax.set_ylabel("z"); ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(out_png, dpi=130, bbox_inches="tight"); plt.close(fig)
    return out_png
