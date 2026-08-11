"""msh_to_yaml.py -- gmsh .msh SURFACE mesh -> msg-shell SG yaml, the mesh-side
twin of helper.ff_to_yaml (which carries the macro state, not the geometry).

A gmsh shell mesh knows the geometry and nothing else: it has no layup, no
material and no macro model.  This converter writes the MESH side of the
msg-shell dialect completely --

    n_model / refined (/ omega)   the yaml header opensg_shell.cli reads
    sections:                     one `- type: shell` block per element set
    materials:                    one entry per material the layups name
    nodes:                        - [y1 y2 y3]                 (one per node)
    elements:                     - [n1 n2 n3(( n4))]          (1-based)
    elementOrientations:          - [e1(3), e2(3), e3(3)]      (per facet)
    sets: element:                one `layup_<k>` set per gmsh physical tag

The CONSTITUTIVE blocks are written FIRST, right under the header and ahead
of the mesh, because they are the part a human reads and edits: a TPMS cell
has ~1e4 nodes and ~1e5 mesh lines, and a `sections:` key buried at line
66262 of a 4 MB file looks missing.  A yaml mapping is unordered by
specification, so every reader in the stack is indifferent to this (the
opensg header/mesh scanners stop on the first top-level block key either
way) -- the ordering is purely for the person opening the file.

-- and leaves the CONSTITUTIVE side to the user.  The layup is a modelling
decision the mesh cannot supply, so it is never invented here:

    nothing given (default) ->  `sections:`/`materials:` come out as a marked
                              FILL_IN template; the file is deliberately NOT
                              runnable and opensg_shell rejects it by name
                              (see check_filled)
    thickness=<t>         ->  a complete, runnable yaml: ONE ply of `material`
                              (default ALUMINIUM, the Schwarz-P block) at
                              `angle`.  The wall thickness is the one number
                              a TPMS/lattice shell case really needs, and it
                              lands in the yaml -- never in the driver.
    layup=[[mat, t, angle], ...] + materials=[...]
                          ->  the general multi-ply form of the same thing

The per-facet frame is the one make_schwarz_yaml.py established for the
Schwarz-P TPMS cell and is reproduced here exactly: e3 is the facet normal,
e2 the first edge, e1 = e2 x e3 (right-handed, e1 x e2 = e3).  Only the
element types the shell loaders accept are converted -- 3-node triangles
(gmsh type 2) and 4-node quads (type 3); 0-/1-D entities that gmsh writes for
physical points and curves are skipped, anything else is an error.

Use:  from opensg_shell.helper.msh_to_yaml import convert
      convert("schwarz_p_D2_shell.msh")                    # template
      convert("schwarz_p_D2_shell.msh", thickness=0.036457)  # runnable, alu
      convert("schwarz_p_D2_shell.msh", thickness=0.036457,  # own material
              material={"name": "cfrp", "E": [1.4e11, 9.0e9, 9.0e9],
                        "G": [4.6e9, 4.6e9, 3.2e9], "nu": [0.3, 0.3, 0.4]})
      convert("SP_mesh_1_shell.msh", thickness=0.036457,     # unwelded mesher
              weld=True)                                     # output: repair it

A mesh whose nodes were never welded (several distinct ids at the SAME point,
which some marching-cubes / TPMS iso-surfacers emit) carries facets with a
zero-length edge, and facet_frames stops on the first one by name.  `weld=True`
is the opt-in repair -- see weld_nodes.  It is OFF by default: a zero-area
facet is a real defect and the converter says so rather than quietly deleting
elements behind the user's back.
"""
import os

import numpy as np

# the default wall material: the isotropic aluminium block of the Schwarz-P
# TPMS case (matches the SwiftComp .sc reference).  G closes from E and nu.
ALUMINIUM = {"name": "alu", "E": 69.0e9, "nu": 0.30, "density": 2700.0}

# the placeholder marker an unfilled template carries; check_filled (and, through
# it, opensg_shell.cli) refuses to run a yaml that still contains one
FILL = "FILL_IN"

# gmsh element type -> node count, for the types the shell loaders accept
_SHELL_TYPES = {2: 3, 3: 4}
# gmsh 0-/1-D entities: written for physical points/curves, not shell facets
_SKIP_TYPES = {15: 1, 1: 2, 8: 3}
_TYPE_NAME = {4: "4-node tetrahedron", 5: "8-node hexahedron",
              6: "6-node prism", 7: "5-node pyramid",
              9: "6-node triangle (2nd order)", 10: "9-node quad (2nd order)",
              16: "8-node quad (2nd order)"}


# --------------------------------------------------------------------------
# gmsh readers
# --------------------------------------------------------------------------
def _read_msh22(lines):
    """gmsh legacy ASCII 2.2: $Nodes / $Elements with per-element tags."""
    i = lines.index("$Nodes")
    nn = int(lines[i + 1])
    nd = np.array([[float(v) for v in lines[i + 2 + k].split()[1:4]]
                   for k in range(nn)], float)
    tag_of = {int(lines[i + 2 + k].split()[0]): k for k in range(nn)}
    i = lines.index("$Elements")
    ne = int(lines[i + 1])
    el, tg, skipped, bad = [], [], 0, {}
    for k in range(ne):
        p = lines[i + 2 + k].split()
        etype, ntag = int(p[1]), int(p[2])
        if etype in _SKIP_TYPES:
            skipped += 1
            continue
        if etype not in _SHELL_TYPES:
            bad[etype] = bad.get(etype, 0) + 1
            continue
        el.append([int(v) for v in p[3 + ntag:3 + ntag + _SHELL_TYPES[etype]]])
        tg.append(int(p[3]) if ntag else 0)      # first tag = physical group
    return nd, tag_of, el, tg, skipped, bad


def _read_msh41(lines):
    """gmsh ASCII 4.1: blocked $Nodes / $Elements, entity -> physical via
    $Entities (a 4.1 element block carries the ENTITY tag, not the physical
    one, so the entity->physical map has to be read first)."""
    phys = {}                                    # (dim, entity tag) -> physical
    if "$Entities" in lines:
        i = lines.index("$Entities")
        npt, ncu, nsu, nvo = (int(v) for v in lines[i + 1].split()[:4])
        j = i + 2
        for dim, cnt in ((0, npt), (1, ncu), (2, nsu), (3, nvo)):
            for _ in range(cnt):
                p = lines[j].split()
                # tag, (3 or 6) bbox reals, nphys, phys tags..., then bounding
                # entities (points have no bbox max -> 3 reals, others 6)
                off = 1 + (3 if dim == 0 else 6)
                npn = int(p[off])
                phys[(dim, int(p[0]))] = int(p[off + 1]) if npn else 0
                j += 1

    i = lines.index("$Nodes")
    nblk, nn = (int(v) for v in lines[i + 1].split()[:2])
    nd = np.zeros((nn, 3))
    tag_of, j, w = {}, i + 2, 0
    for _ in range(nblk):
        nb = int(lines[j].split()[3])
        for k in range(nb):
            tag_of[int(lines[j + 1 + k])] = w + k
        for k in range(nb):
            nd[w + k] = [float(v) for v in lines[j + 1 + nb + k].split()[:3]]
        w += nb
        j += 1 + 2*nb

    i = lines.index("$Elements")
    nblk = int(lines[i + 1].split()[0])
    el, tg, skipped, bad, j = [], [], 0, {}, i + 2
    for _ in range(nblk):
        p = lines[j].split()
        dim, ent, etype, nb = int(p[0]), int(p[1]), int(p[2]), int(p[3])
        if etype in _SKIP_TYPES:
            skipped += nb
        elif etype not in _SHELL_TYPES:
            bad[etype] = bad.get(etype, 0) + nb
        else:
            ph = phys.get((dim, ent), 0)
            for k in range(nb):
                el.append([int(v) for v in lines[j + 1 + k].split()[1:]])
                tg.append(ph)
        j += 1 + nb
    return nd, tag_of, el, tg, skipped, bad


def read_msh(path):
    """Nodes, shell facets and their physical tags from a gmsh .msh file.

    In:  path str -- an ASCII gmsh mesh, format 2.2 or 4.1
    Out: dict {nodes (n,3) float, elements list[list[int]] 1-based and
         CONSECUTIVE (gmsh node tags are renumbered), tags (ne,) int physical
         group per facet, n_skipped int 0-/1-D entities dropped, version str}.
    Raises ValueError on an element type the shell loaders cannot take."""
    with open(path) as f:
        lines = [ln.strip() for ln in f]
    if "$MeshFormat" not in lines:
        raise ValueError("%s: no $MeshFormat -- not an ASCII gmsh .msh" % path)
    ver = lines[lines.index("$MeshFormat") + 1].split()[0]
    if ver.startswith("2"):
        nd, tag_of, el, tg, skipped, bad = _read_msh22(lines)
    elif ver.startswith("4"):
        nd, tag_of, el, tg, skipped, bad = _read_msh41(lines)
    else:
        raise ValueError("%s: gmsh format %s is not supported (ASCII 2.2 or"
                         " 4.1)" % (path, ver))
    if bad:
        raise ValueError(
            "%s: the msg-shell SG takes 3-node triangles and 4-node quads"
            " only; this mesh has %s.\nRe-mesh the SURFACE with first-order"
            " triangular/quadrilateral elements (gmsh types 2 / 3)."
            % (path, ", ".join("%d x %s" % (n, _TYPE_NAME.get(t, "gmsh type %d"
                                                              % t))
                               for t, n in sorted(bad.items()))))
    if not el:
        raise ValueError("%s: no 3-node/4-node surface elements found" % path)
    # gmsh node tags need not be 1..n: renumber onto the node block order
    el = [[tag_of[v] + 1 for v in e] for e in el]
    return {"nodes": nd, "elements": el, "tags": np.asarray(tg, int),
            "n_skipped": skipped, "version": ver}


# --------------------------------------------------------------------------
# coincident-node welding: an OPT-IN connectivity repair
# --------------------------------------------------------------------------
def _blocks_by_size(elements):
    """(node count -> (row index array, (m, count) connectivity array)) for a
    ragged facet list, so tri and quad blocks can each be handled with numpy
    instead of a Python loop over millions of facets."""
    lens = np.fromiter((len(e) for e in elements), np.int64, len(elements))
    out = {}
    for n in np.unique(lens):
        idx = np.flatnonzero(lens == n)
        if len(idx) == len(elements):            # homogeneous: one C-level cast
            conn = np.asarray(elements, np.int64)
        else:
            conn = np.array([elements[i] for i in idx], np.int64)
        out[int(n)] = (idx, conn)
    return out


def weld_nodes(nodes, elements, tol=None, tags=None):
    """Merge nodes that sit at the SAME point, drop the facets that collapse.

    Some surface meshers -- marching-cubes / TPMS iso-surfacers above all --
    emit their nodes cell by cell and never WELD the shared ones, so the file
    carries several distinct node ids at bit-identical coordinates.  A facet
    that happens to reference two of them has a zero-LENGTH edge and therefore
    zero area, which facet_frames rejects by name: the geometry is right, only
    the connectivity is torn.  Welding is the repair, and it is conservative by
    construction -- the only facets it removes are the ones of exactly zero
    area, so the total surface area cannot move.

    The merge is a spatial hash, not a pairwise search: coordinates are rounded
    onto a grid of side `tol`, one lexsort over the rounded triples groups the
    coincident nodes, and the connectivity is remapped onto the group ids --
    O(n log n), so a 4e6-node mesh is a few seconds, not a quadratic blow-up.
    A facet whose ids are no longer distinct is then dropped (a quad left with
    exactly 3 distinct corners is COLLAPSED to a triangle instead of dropped,
    since it still carries real area), and the nodes nothing references any
    more are deleted and the rest renumbered.

    TOLERANCE.  `tol` defaults to 1e-12 x the bounding-box diagonal.  The
    duplicates this targets are bit-identical, so the merge needs no tolerance
    at all in principle and any value above the ASCII round-off of the .msh
    works; the default is chosen as the largest such value that is still
    unmistakably below every real feature of the mesh.  On the Schwarz-P ladder
    it is 1.7e-12 m against a shortest REAL edge of ~1e-5 m -- seven orders of
    margin -- so it cannot fuse two genuinely distinct nodes.  The returned
    `max_cluster_diameter` is the audit of that claim: it is the largest
    distance between any two nodes that were actually merged, and for
    bit-identical duplicates it comes out at exactly 0.

    In:  nodes (n,3) float; elements list[list[int]] 1-based facets;
         tol float | None -- weld radius (m), None = 1e-12 x bbox diagonal;
         tags (ne,) int | None -- per-facet physical tags, filtered alongside
    Out: dict {nodes (m,3) float, elements list[list[int]] 1-based, tags
         (m,) int | None, tol float, n_merged int nodes eliminated by the
         merge, n_dropped int facets removed as zero-area, n_collapsed int
         quads turned into triangles, n_orphans int unreferenced nodes
         deleted, n_nodes_in/out int, n_facets_in/out int,
         max_cluster_diameter float}."""
    nd = np.asarray(nodes, float)
    if nd.ndim != 2 or nd.shape[1] != 3:
        raise ValueError("weld_nodes: nodes must be (n,3), got %s"
                         % (nd.shape,))
    diag = float(np.linalg.norm(nd.max(0) - nd.min(0)))
    tol = 1e-12*diag if tol is None else float(tol)
    if not tol > 0.0:
        raise ValueError("weld_nodes: tol must be positive, got %r" % (tol,))

    # ---- group the coincident nodes (spatial hash + lexsort) --------------
    key = np.round(nd/tol)
    if np.abs(key).max(initial=0.0) >= 2.0**62:
        raise ValueError("weld_nodes: tol=%g is too small for coordinates of"
                         " size %g -- the grid index overflows int64"
                         % (tol, np.abs(nd).max()))
    key = key.astype(np.int64)
    order = np.lexsort((key[:, 2], key[:, 1], key[:, 0]))
    ks = key[order]
    new = np.empty(len(ks), bool)
    new[0] = True
    np.any(ks[1:] != ks[:-1], axis=1, out=new[1:])
    starts = np.flatnonzero(new)
    inv = np.empty(len(nd), np.int64)            # node -> group id
    inv[order] = np.cumsum(new) - 1
    rep = order[starts]                          # group -> a member node
    ng = len(starts)
    nds = nd[order]
    max_diam = float(np.linalg.norm(np.maximum.reduceat(nds, starts, axis=0)
                                    - np.minimum.reduceat(nds, starts, axis=0),
                                    axis=1).max(initial=0.0))
    del ks, nds, key, order

    # ---- remap the facets, drop / collapse the degenerate ones ------------
    kept_idx, kept_conn, n_drop, n_coll = [], [], 0, 0
    for n, (idx, conn) in sorted(_blocks_by_size(elements).items()):
        if n not in _SHELL_TYPES.values():
            raise ValueError("weld_nodes: every facet must have 3 or 4 nodes,"
                             " found one with %d" % n)
        mm = inv[conn - 1]
        ms = np.sort(mm, axis=1)
        nuniq = n - (ms[:, 1:] == ms[:, :-1]).sum(1)
        good = nuniq == n
        n_drop += int((nuniq < 3).sum())
        if good.any():
            kept_idx.append(idx[good])
            kept_conn.append(mm[good])
        if n == 4:                               # quad -> triangle, area kept
            half = np.flatnonzero(nuniq == 3)
            n_coll += len(half)
            for j in half:                       # rare: cyclic de-duplication
                r = mm[j]
                c = [int(r[0])]
                for a in r[1:]:
                    if a != c[-1]:
                        c.append(int(a))
                if len(c) > 1 and c[0] == c[-1]:
                    c.pop()
                kept_idx.append(np.array([idx[j]], np.int64))
                kept_conn.append(np.array([c], np.int64))

    # ---- delete the nodes nothing references any more, renumber -----------
    used = np.zeros(ng, bool)
    for c in kept_conn:
        used[c.ravel()] = True
    keep = np.flatnonzero(used)
    renum = np.full(ng, -1, np.int64)
    renum[keep] = np.arange(len(keep), dtype=np.int64)
    out_nodes = nd[rep[keep]]

    # original facet order is the element-set order: restore it
    flat = (np.concatenate(kept_idx) if kept_idx else np.zeros(0, np.int64))
    rows = []
    for c in kept_conn:
        rows.extend((renum[c] + 1).tolist())
    if len(kept_conn) > 1:                       # mixed tri/quad: re-sort
        out_el = [rows[i] for i in np.argsort(flat, kind="stable").tolist()]
        flat = np.sort(flat)
    else:
        out_el = rows
    out_tags = None if tags is None else np.asarray(tags, int)[flat]

    return {"nodes": out_nodes, "elements": out_el, "tags": out_tags,
            "tol": tol, "n_merged": int(len(nd) - ng),
            "n_dropped": int(n_drop), "n_collapsed": int(n_coll),
            "n_orphans": int(ng - len(keep)),
            "n_nodes_in": int(len(nd)), "n_nodes_out": int(len(keep)),
            "n_facets_in": int(len(elements)), "n_facets_out": int(len(flat)),
            "max_cluster_diameter": max_diam}


# --------------------------------------------------------------------------
# per-facet material frame
# --------------------------------------------------------------------------
def facet_frames(nodes, elements):
    """The (e1, e2, e3) triad and area of every shell facet.

    The frame make_schwarz_yaml.py established for the Schwarz-P cell,
    generalised to quads: e3 is the outward facet normal (right-hand rule on
    the node ordering), e2 the first edge, e1 = e2 x e3 -- a right-handed
    triad with e1 x e2 = e3.  A quad's normal comes from its DIAGONALS, which
    is the Newell normal for a planar quad and stays well-defined on a warped
    one; its e2 is then re-orthogonalised against e3 (for a triangle e2 is
    already exactly perpendicular to e3, so the reference formula is
    reproduced unchanged).

    In:  nodes (n,3) float; elements list[list[int]] 1-based facets
    Out: (ori (ne,9) float rows [e1 e2 e3], area (ne,) float facet areas)."""
    nd = np.asarray(nodes, float)
    ne = len(elements)
    ori = np.zeros((ne, 9))
    area = np.zeros(ne)
    idx3 = [i for i, e in enumerate(elements) if len(e) == 3]
    idx4 = [i for i, e in enumerate(elements) if len(e) == 4]
    if len(idx3) + len(idx4) != ne:
        raise ValueError("facet_frames: every element must have 3 or 4 nodes")

    for idx, quad in ((idx3, False), (idx4, True)):
        if not idx:
            continue
        e = np.array([elements[i] for i in idx], int) - 1
        p1, p2 = nd[e[:, 0]], nd[e[:, 1]]
        if quad:
            nrm = np.cross(nd[e[:, 2]] - p1, nd[e[:, 3]] - p2)
        else:
            nrm = np.cross(p2 - p1, nd[e[:, 2]] - p1)
        a = 0.5*np.linalg.norm(nrm, axis=1)
        if np.any(a <= 0.0):
            k = idx[int(np.argmin(a))]
            raise ValueError("facet_frames: element %d (nodes %s) is"
                             " degenerate (zero area)" % (k + 1, elements[k]))
        e3 = nrm/(2*a)[:, None]
        e2 = (p2 - p1)/np.linalg.norm(p2 - p1, axis=1)[:, None]
        if quad:                                  # warped quad: re-orthogonalise
            e2 = e2 - (e2*e3).sum(1)[:, None]*e3
            e2 = e2/np.linalg.norm(e2, axis=1)[:, None]
        e1 = np.cross(e2, e3)
        ori[idx] = np.column_stack([e1, e2, e3])
        area[idx] = a
    return ori, area


# --------------------------------------------------------------------------
# the constitutive side: user-supplied, never invented
# --------------------------------------------------------------------------
def _norm_material(m):
    """One `materials:` entry from either the yaml form or an isotropic
    shorthand {name, E, nu, [G], [density]} -- E/G/nu always end up as the
    3-vectors the shell material reader wants."""
    m = dict(m)
    name = m.get("name")
    if not name:
        raise ValueError("every material needs a `name`: %r" % (m,))
    if "elastic" in m:
        el = m["elastic"]
        miss = [k for k in ("E", "G", "nu") if k not in el]
        if miss:
            raise ValueError("material %r: elastic is missing %s"
                             % (name, ", ".join(miss)))
        out = {"E": [float(v) for v in np.broadcast_to(np.asarray(el["E"],
                                                                  float), 3)],
               "G": [float(v) for v in np.broadcast_to(np.asarray(el["G"],
                                                                  float), 3)],
               "nu": [float(v) for v in np.broadcast_to(np.asarray(el["nu"],
                                                                   float), 3)]}
    else:
        if "E" not in m or "nu" not in m:
            raise ValueError("material %r: give `elastic` {E, G, nu} or the"
                             " isotropic shorthand E and nu" % name)
        E = np.broadcast_to(np.asarray(m["E"], float), 3)
        nu = np.broadcast_to(np.asarray(m["nu"], float), 3)
        G = (np.broadcast_to(np.asarray(m["G"], float), 3) if "G" in m
             else E/(2.0*(1.0 + nu)))            # isotropic closure
        out = {"E": [float(v) for v in E], "G": [float(v) for v in G],
               "nu": [float(v) for v in nu]}
    return {"name": str(name), "density": float(m.get("density", 0.0)),
            "elastic": out}


def _norm_layup(layup, set_names):
    """{set name: [[mat, thickness, angle], ...]} from a single layup (applied
    to every element set) or a per-set dict."""
    if isinstance(layup, dict):
        miss = [s for s in set_names if s not in layup]
        if miss:
            raise ValueError("no layup given for element set(s) %s -- the mesh"
                             " has %s" % (", ".join(miss),
                                          ", ".join(set_names)))
        rows = {s: layup[s] for s in set_names}
    else:
        rows = {s: layup for s in set_names}
    out = {}
    for s, ply in rows.items():
        if not ply:
            raise ValueError("the layup of element set %s is empty" % s)
        out[s] = [[str(p[0]), float(p[1]), float(p[2])] for p in ply]
    return out


def _sections_block(layup_by_set, set_names, materials):
    """The `sections:` + `materials:` yaml lines -- the FILL_IN template when
    no layup was given, the real blocks when one was."""
    if layup_by_set is None:
        o = ["# " + "="*70,
             "# TEMPLATE -- the mesh blocks BELOW are complete, this LAYUP is"
             " NOT.",
             "# opensg_shell refuses to run this file until every %s field is"
             % FILL,
             "# replaced.  A layup row is [<material-name>, <thickness in m>,",
             "# <angle in deg>], listed OML -> IML; add one `- type: shell`"
             " block",
             "# per element set, and one entry under `materials:` per material",
             "# name the layups use (isotropic: repeat the value three times,",
             "# G = E/(2(1+nu))).",
             "# " + "="*70,
             "sections:"]
        for s in set_names:
            o += ["- type: shell", "  elementSet: %s" % s, "  layup:",
                  "  - - %s_MATERIAL_NAME" % FILL,
                  "    - %s_THICKNESS_M" % FILL,
                  "    - 0.0"]
        o += ["materials:",
              "- name: %s_MATERIAL_NAME" % FILL,
              "  density: %s_DENSITY_KG_M3" % FILL,
              "  elastic:",
              "    E: [{0}_E1, {0}_E2, {0}_E3]".format(FILL),
              "    G: [{0}_G12, {0}_G13, {0}_G23]".format(FILL),
              "    nu: [{0}_NU12, {0}_NU13, {0}_NU23]".format(FILL)]
        return o
    o = ["sections:"]
    for s in set_names:
        o += ["- type: shell", "  elementSet: %s" % s, "  layup:"]
        for mn, t, a in layup_by_set[s]:
            o += ["  - - %s" % mn, "    - %.6e" % t, "    - %s" % repr(a)]
    o.append("materials:")
    for m in materials:
        el = m["elastic"]
        o += ["- name: %s" % m["name"],
              "  density: %s" % repr(m["density"]),
              "  elastic:",
              "    E: [%.6e, %.6e, %.6e]" % tuple(el["E"]),
              "    G: [%.6e, %.6e, %.6e]" % tuple(el["G"]),
              "    nu: [%.6f, %.6f, %.6f]" % tuple(el["nu"])]
    return o


def check_filled(path, max_show=12):
    """Is this yaml a still-unfilled msh_to_yaml template?

    Cheap line scan (never a full parse of a multi-MB mesh), so the CLI can
    call it on every input.

    In:  path str -- an SG yaml; max_show int -- placeholders to list
    Out: None when the file is complete, else the error MESSAGE naming the
         fields the user still has to fill in."""
    hits = []
    try:
        with open(path) as f:
            for k, ln in enumerate(f, 1):
                if FILL in ln:
                    hits.append((k, ln.rstrip()))
    except OSError:
        return None
    hits = [h for h in hits if not h[1].lstrip().startswith("#")]
    if not hits:
        return None
    shown = "\n".join("  line %-6d %s" % h for h in hits[:max_show])
    more = ("\n  ... and %d more" % (len(hits) - max_show)
            if len(hits) > max_show else "")
    return ("%s is an UNFILLED msh_to_yaml TEMPLATE.\n"
            "The mesh blocks (nodes / elements / elementOrientations / sets)"
            " are complete,\nbut `sections:` and `materials:` are still"
            " placeholders -- opensg_shell will not\ninvent a layup or"
            " material properties.  Fill in these %d %s field(s):\n%s%s\n\n"
            "  a layup row is  [<material-name>, <thickness in m>,"
            " <angle in deg>]  (OML -> IML)\n"
            "  a material is   name, density, elastic {E:[E1,E2,E3],"
            " G:[G12,G13,G23], nu:[nu12,nu13,nu23]}\n"
            "                  (isotropic: repeat the value three times,"
            " G = E/(2(1+nu)))"
            % (os.path.basename(path), len(hits), FILL, shown, more))


# --------------------------------------------------------------------------
# the converter
# --------------------------------------------------------------------------
def convert(msh_path, thickness=None, material=None, angle=0.0, out_path=None,
            n_model=3, refined=0, layup=None, materials=None, omega=None,
            weld=False, weld_tol=None):
    """gmsh surface mesh -> msg-shell SG yaml.

    In:  msh_path str -- ASCII gmsh .msh (2.2 or 4.1) of a SHELL surface
         thickness float | None -- WALL thickness (m).  Given, it writes the
              single-ply layup [[material name, thickness, angle]] into
              `sections:` -- so the thickness lives in the yaml, not in the
              caller.  None (and no `layup`) writes the FILL_IN template.
         material dict | None -- the wall material for `thickness`, as a yaml
              `materials:` entry or the isotropic shorthand
              {name, E, nu, [G], [density]}; None takes ALUMINIUM
         angle float -- ply angle (deg) of that single ply
         out_path str | None -- output yaml; default <msh base>.yaml
         n_model int -- yaml header: 1 beam, 3 3-D solid macro model
         refined int -- yaml header: 0 classical (KL wall), 1 shear-refined
         layup list[[mat, t, angle]] | dict{set: rows} | None -- the general
              MULTI-ply form, mutually exclusive with `thickness`
         materials list[dict] | None -- yaml `materials:` entries, or the
              isotropic shorthand; required whenever `layup` is given
         omega float | None -- optional `omega:` header (user SG measure)
         weld bool -- OFF by default, so an unrepaired mesh still fails loudly
              in facet_frames.  True runs weld_nodes first: coincident nodes
              are merged, the facets that collapse to zero area are dropped
              and the orphaned nodes deleted.  Use it on an UNWELDED mesher
              output (the "element N is degenerate (zero area)" error), never
              to paper over a mesh whose geometry is actually wrong.
         weld_tol float | None -- weld radius (m) for `weld`; None takes
              weld_nodes' default of 1e-12 x the bounding-box diagonal
    Out: dict of the yaml blocks (nodes, elements, elementOrientations, sets,
         sections, materials, header keys) plus `out_path`, `surface_area`,
         `n_skipped` and `weld` (the weld_nodes COUNTS, None when off);
         writes out_path."""
    if thickness is not None:
        if layup is not None:
            raise ValueError("give EITHER thickness=<t> (one ply) OR"
                             " layup=[[mat, t, angle], ...] (the general"
                             " stack), not both")
        if materials is not None:
            raise ValueError("with thickness=<t> the wall material is"
                             " `material=` (one entry), not `materials=`")
        if float(thickness) <= 0.0:
            raise ValueError("thickness must be positive, got %r" % (thickness,))
        # _norm_material validates the name/E/nu and closes G = E/(2(1+nu));
        # it is idempotent, so the general path below re-normalizes harmlessly
        mat = _norm_material(ALUMINIUM if material is None else material)
        layup = [[mat["name"], float(thickness), float(angle)]]
        materials = [mat]
    elif material is not None:
        raise ValueError("material= is the companion of thickness=<t>; for a"
                         " multi-ply `layup` pass materials=[...]")
    m = read_msh(msh_path)
    nd, el = m["nodes"], m["elements"]
    wr = None
    if weld:
        wr = weld_nodes(nd, el, tol=weld_tol, tags=m["tags"])
        nd, el, m["tags"] = wr["nodes"], wr["elements"], wr["tags"]
    ori, area = facet_frames(nd, el)
    ne = len(el)

    # one element set per gmsh physical tag, in order of first appearance
    order, seen = [], set()
    for t in m["tags"]:
        if t not in seen:
            seen.add(t)
            order.append(int(t))
    set_names = ["layup_%d" % k for k in range(len(order))]
    labels = {n: [i + 1 for i in range(ne) if m["tags"][i] == t]
              for n, t in zip(set_names, order)}

    if layup is None:
        if materials is not None:
            raise ValueError("materials were given without a layup -- pass"
                             " layup=[[<material>, <thickness>, <angle>], ...]"
                             " too, or neither (which writes the template)")
        lay_by_set, mats = None, None
    else:
        if materials is None:
            raise ValueError("a layup needs its materials -- pass"
                             " materials=[{'name': ..., 'E': ..., 'nu': ...},"
                             " ...]")
        lay_by_set = _norm_layup(layup, set_names)
        if isinstance(materials, dict):
            materials = [dict(v, name=k) for k, v in materials.items()]
        mats = [_norm_material(x) for x in materials]
        have = {x["name"] for x in mats}
        need = {p[0] for rows in lay_by_set.values() for p in rows}
        if need - have:
            raise ValueError("the layup names material(s) %s that `materials`"
                             " does not define (it has %s)"
                             % (", ".join(sorted(need - have)),
                                ", ".join(sorted(have))))

    o = ["n_model: %d      # 1 = beam, 2 = plate (opensg_solid), 3 = solid"
         " -- the macro model this SG homogenizes to" % int(n_model),
         "refined: %d      # 0 = classical (beam EB 4x4, KL wall);"
         " 1 = shear-refined (beam Timoshenko 6x6, RM wall)" % int(refined),
         "msg: shell      # the ENGINE this SG belongs to (opensg_shell);"
         " `opensg <yaml>` dispatches on it"]
    if omega is not None:
        o.append("omega: %.12g      # user SG measure (overrides the measured"
                 " one)" % float(omega))
    # CONSTITUTIVE FIRST: `sections:`/`materials:` go directly under the header,
    # ahead of the ~1e5 mesh lines, so the two blocks a human edits are the two
    # at the top of the file.  A yaml mapping carries no order, so this is a
    # readability decision only -- see the module docstring.
    o += _sections_block(lay_by_set, set_names, mats)
    o.append("nodes:")
    o += ["- [%.12f %.12f %.12f]" % tuple(p) for p in nd]
    o.append("elements:")
    o += ["- [" + " ".join("%d" % v for v in e) + "]" for e in el]
    o.append("elementOrientations:")
    o += ["- [%.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f]"
          % tuple(r) for r in ori]
    o += ["sets:", "  element:"]
    for s in set_names:
        o += ["  - name: %s" % s, "    labels:"]
        o += ["    - %d" % i for i in labels[s]]

    if out_path is None:
        out_path = os.path.splitext(msh_path)[0] + ".yaml"
    with open(out_path, "w") as f:
        f.write("\n".join(o) + "\n")
    print("msh_to_yaml: %s (gmsh %s) -> %s\n  %d nodes, %d facets"
          " (%d tri, %d quad), %d set(s), surface area %.6g%s%s"
          % (os.path.basename(msh_path), m["version"],
             os.path.basename(out_path), len(nd), ne,
             sum(1 for e in el if len(e) == 3),
             sum(1 for e in el if len(e) == 4), len(set_names), area.sum(),
             "" if not m["n_skipped"] else
             "; %d 0-/1-D entities skipped" % m["n_skipped"],
             "\n  LAYUP NOT SET: fill in the %s fields before running"
             " opensg_shell" % FILL if lay_by_set is None else ""))
    if wr is not None:
        print("  welded at tol %.3g m: %d coincident node(s) merged (largest"
              " merged cluster %.3g m),\n    %d zero-area facet(s) dropped,"
              " %d quad(s) collapsed to triangles, %d orphan node(s) removed"
              "\n    %d -> %d nodes, %d -> %d facets"
              % (wr["tol"], wr["n_merged"], wr["max_cluster_diameter"],
                 wr["n_dropped"], wr["n_collapsed"], wr["n_orphans"],
                 wr["n_nodes_in"], wr["n_nodes_out"],
                 wr["n_facets_in"], wr["n_facets_out"]))

    d = {"n_model": int(n_model), "refined": int(refined),
         "nodes": [[float(v) for v in p] for p in nd],
         "elements": [list(e) for e in el],
         "elementOrientations": [[float(v) for v in r] for r in ori],
         "sets": {"element": [{"name": s, "labels": labels[s]}
                              for s in set_names]},
         "sections": ([{"type": "shell", "elementSet": s,
                        "layup": lay_by_set[s]} for s in set_names]
                      if lay_by_set is not None else None),
         "materials": mats, "out_path": out_path,
         "surface_area": float(area.sum()), "n_skipped": m["n_skipped"],
         # the COUNTS only: the repaired mesh itself is already the `nodes` /
         # `elements` above, and a caller that prints r["weld"] must not get a
         # 4e6-row array back in its face
         "weld": (None if wr is None else
                  {k: v for k, v in wr.items()
                   if k not in ("nodes", "elements", "tags")})}
    if omega is not None:
        d["omega"] = float(omega)
    return d
