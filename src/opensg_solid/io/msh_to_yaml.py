"""msh_to_yaml.py -- gmsh `.msh` 2-D/3-D SOLID mesh -> OpenSG solid SG yaml,
the solid-side twin of opensg_shell.helper.msh_to_yaml (which writes the
msg-shell surface dialect) and the mesh-side companion of io.sc_to_yaml
(which reads a SwiftComp `.sc` instead of a gmsh mesh).

A gmsh mesh knows the geometry and the physical tags and nothing else: it
carries no material and no macro model.  This converter writes the MESH side
of the OpenSG solid dialect completely --

    nodes:                 - [y1 y2 y3]              (one row per node)
    elements:              - [n1 n2 n3(( n4))]       (1-based connectivity)
    elementOrientations:   - [e1(3), e2(3), e3(3)]   (one row per element)
    sets: element:         one set per gmsh physical tag

-- and takes the CONSTITUTIVE side (`materials:`) from the caller, because a
material is a modelling decision the mesh cannot supply.  When the caller
gives none, `materials:` comes out as a marked FILL_IN template: the file is
deliberately NOT runnable and opensg_solid rejects it by name (see
check_filled), exactly as the shell twin does for a missing layup.  The
caller has TWO ways to supply materials: the classic `materials=[...]` list
(one dict per phase, fully runnable output), or a `materials={tag: mdict}`
DICT keyed by gmsh physical tag -- the library route `opensg msh_to_yaml
--mat<K> NAME[:ANGLE]` uses (names resolved in io.materials_db's
materials.yaml).  A dict may cover only SOME tags: covered sets come out
filled, uncovered sets keep their FILL_IN placeholders, and the header's
n_model stays a FILL_IN placeholder unless the caller pins it -- the
engine's silent default (2, plate) picking the wrong macro model is exactly
the footgun this refuses to reintroduce.  The
orientation is not in a .msh either -- gmsh 2.2 carries $Nodes, $Elements
and physical tags, no per-element frames -- so a constant frame is written
(the caller's `orientation=`, default e1 out of plane).  One `sets:
element:` entry is written per physical tag, named after $PhysicalNames
when the mesh carries the block (a template's material entries take the
SAME names, because the solid reader binds sets to materials by name --
io.sg_input._mat_id_from_sets).  This is the
`nodes`/`elements`/`elementOrientations`/`materials`/`sets` dialect the 2-D
SG drivers read (UDcomp_2D.yaml, square_tube_2Dsolid.yaml), NOT the
`dim`/`nodes`/`cells`/`mat_id`/`materials` dialect sc_to_yaml emits for
opensg_solid.sg_mesh.load_sg_input.

Accepted input is gmsh's LEGACY ASCII 2.2 format ($MeshFormat 2.x, file
type 0); msh 4.x (gmsh's default) and binary files are refused with the
re-save command.  The cells of the HIGHEST dimension present form the SG
and must be ONE of the element types the solid drivers accept -- 3-/6-node
triangles (gmsh types 2/9), 4-/9-node quads (types 3/10), 4-/10-node tets
(types 4/11) or 8-node hexes (type 5); the quadratic grades are what
helper.linear_msh_to_quad emits.  Every lower-dimensional cell (the faces,
curves and points gmsh writes for physical groups, with no physical groups
at all, or with Mesh.SaveAll) is dropped and counted; anything else is an
error, as are 2-D cells off the x-y plane (a 3-D mesh saved with Physical
Surfaces only) and cells written twice (one volume in two physical groups,
or a repeated element id).  Node ids are honoured as ids (any row order,
gaps allowed) and nodes no kept cell uses (orphans) are compacted away.  A
file without physical tags (no physical groups, or Mesh.SaveAll) takes its
sets from the elementary tags, while a tag-0 request (`--mat0`, a
one-material list) still covers the whole mesh.  The
mesh grade (linear | quadratic) is CONSOLE-ONLY information -- the
solver reads the arity off the cells, so the yaml carries no order key;
`refined:` is always written (default 0 = classical).

Use:  from opensg_solid.io.msh_to_yaml import convert
      convert("UDcomp_2D.msh", materials=[...], phases=[("matrix", 0),
                                                        ("fiber", 1)])
      convert("SP_solid.msh")                  # mesh-only FILL_IN template
      convert("SP_solid.msh", n_model=2, refined=1,
              materials={1: {"name": "-", "density": 2700.0, "E": 7e10,
                             "G": 2.6923e10, "nu": 0.3}})   # library route
"""
import os

import numpy as np

# the placeholder marker an unfilled template carries; check_filled (and,
# through it, opensg_solid.cli) refuses to run a yaml that still contains one
FILL = "FILL_IN"

# gmsh element type -> node count, for the types the solid loaders accept
_SOLID_TYPES = {2: 3, 3: 4, 4: 4, 5: 8, 9: 6, 10: 9, 11: 10}
# gmsh 0-/1-D entities: written for physical points/curves, never an SG
_SKIP_TYPES = {15: 1, 1: 2, 8: 3, 26: 4, 27: 5, 28: 6}
_TYPE_NAME = {15: "point", 1: "line2", 8: "line3", 26: "line4",
              27: "line5", 28: "line6", 20: "tri9", 21: "tri10",
              36: "quad16", 6: "6-node prism", 7: "5-node pyramid",
              18: "15-node prism", 13: "18-node prism",
              19: "13-node pyramid", 14: "14-node pyramid",
              17: "20-node hex (hex20)", 12: "27-node hex (hex27)",
              16: "8-node serendipity quad (unsupported -- the engine's"
                  " quadrilateral-P2 is the 9-node Lagrange quad)",
              29: "20-node tetrahedron (3rd order)"}
# gmsh element type -> topological dimension / node count, for every type
# named here; a code outside these tables is unknown and refused
_DIM = {15: 0, 1: 1, 8: 1, 26: 1, 27: 1, 28: 1,
        2: 2, 3: 2, 9: 2, 10: 2, 16: 2, 20: 2, 21: 2, 36: 2,
        4: 3, 5: 3, 6: 3, 7: 3, 11: 3, 12: 3, 13: 3, 14: 3, 17: 3, 18: 3,
        19: 3, 29: 3}
_NNODE = {**_SOLID_TYPES, **_SKIP_TYPES, 6: 6, 7: 5, 12: 27, 13: 18,
          14: 14, 16: 8, 17: 20, 18: 15, 19: 13, 20: 9, 21: 10, 29: 20,
          36: 16}
# max node id below _DENSE_IDS * n_nodes + _DENSE_SLACK -> a dense id -> row
# table, else a sorted map
_DENSE_IDS, _DENSE_SLACK = 8, 1024
# the words for an entity of the SG dimension (2-D, 3-D) in messages
_ENTITY = {2: ("surface", "Surface"), 3: ("volume", "Volume")}
# accepted type -> (the informational `mesh_order:` word, the cell name)
_ORDER_OF = {2: ("linear", "tri3"), 3: ("linear", "quad4"),
             4: ("linear", "tet4"), 5: ("linear", "hex8"),
             9: ("quadratic", "tri6"), 10: ("quadratic", "quad9"),
             11: ("quadratic", "tet10")}

# the default per-element frame of a 2-D cross-section SG: e1 along the beam
# axis (out of plane, +z), e2 = +x, e3 = +y -- the frame the 2-D solid SG
# examples use (Rules/orientation_e1_out_of_plane.md)
E1_OUT_OF_PLANE = (0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def _cell_word(t):
    """A gmsh element type code as the word every message prints.

    In:  t int -- gmsh element type
    Out: str -- tri3/tet10/... for an accepted type, the _TYPE_NAME word
         for the other named types, "gmsh type <t>" for an unknown code."""
    if t in _ORDER_OF:
        return _ORDER_OF[t][1]
    return _TYPE_NAME.get(t, "gmsh type %d" % t)


def _check_format(path, head):
    """Refuse anything but a gmsh legacy ASCII 2.x file, naming the re-save.

    In:  path str -- the .msh path (for the message); head bytes -- the
         first bytes of the file
    Out: None; ValueError when the $MeshFormat header is missing, the
         version is not 2.x or the file type is not 0 (ASCII)."""
    name = os.path.basename(path)
    fix = ("-- msh_to_yaml reads legacy ASCII 2.2: re-save with  gmsh %s"
           " -format msh22 -o %s_msh22.msh -save  (API: Mesh.MshFileVersion"
           " 2.2, Mesh.Binary 0)" % (name, os.path.splitext(name)[0]))
    ln = head.lstrip(b"\xef\xbb\xbf \t\r\n").split(b"\n", 2)
    if ln[0].strip() != b"$MeshFormat" or len(ln) < 2:
        raise ValueError("%s has no $MeshFormat header %s" % (name, fix))
    f = ln[1].split()
    ver = f[0].decode("ascii", "replace") if f else "?"
    kind = {b"0": "ascii", b"1": "binary"}.get(f[1] if len(f) > 1 else b"",
                                               "file-type ?")
    if not ver.startswith("2") or kind != "ascii":
        raise ValueError("%s is gmsh %s %s %s" % (name, ver, kind, fix))


def _section(raw, block, fname):
    """The count line and the body of one `$<block>` ... `$End<block>` block.

    In:  raw bytes -- the whole .msh file; block str -- the block name
         (Nodes | Elements | PhysicalNames); fname str -- the file name for
         messages
    Out: (n int | None, body bytes) -- the leading count and the lines after
         it up to the end marker (each ending in a newline); (None, b"")
         when the file has no such block."""
    tag = ("\n$" + block).encode()
    k = raw.find(tag)
    while k >= 0 and raw[k + len(tag):k + len(tag) + 1] not in (
            b"\n", b"\r", b" ", b"\t"):
        k = raw.find(tag, k + 1)
    if k < 0:
        return None, b""
    a = raw.index(b"\n", k + 1) + 1
    b = raw.index(b"\n", a) + 1
    e = raw.find(("\n$End" + block).encode(), a)
    if e < 0:
        raise ValueError("%s: $%s has no $End%s -- truncated mesh"
                         % (fname, block, block))
    return int(raw[a:b]), raw[b:e + 1]


def _element_runs(body, n_all, name):
    """The $Elements records as runs of equal (type, n_tags), walked token
    by token the way gmsh reads them (line breaks carry no meaning).

    In:  body bytes -- the $Elements lines; n_all int -- the declared
         element count; name str -- the file name for messages
    Out: list of (type int, n_tags int, rows (k, 3 + n_tags + nodes) int64
         -- `id type n_tags <tags> <nodes>` per record, file order)."""
    try:
        flat = np.fromstring(body, dtype=np.int64, sep=" ")
    except ValueError:
        raise ValueError("%s: $Elements carries a non-integer token" % name)
    bad_at = "%s: $Elements is truncated or malformed near token %d"
    runs, pos, T = [], 0, flat.size
    while pos < T:
        if pos + 3 > T:
            raise ValueError(bad_at % (name, pos))
        t, g = int(flat[pos + 1]), int(flat[pos + 2])
        if t not in _NNODE:
            raise ValueError("%s carries %s -- not a cell type msh_to_yaml"
                             " knows" % (name, _cell_word(t)))
        L = 3 + g + _NNODE[t]
        kmax = (T - pos) // L if g >= 0 else 0
        if kmax < 1:
            raise ValueError(bad_at % (name, pos))
        # gallop: the first record that changes (type, n_tags) ends the run
        k, step = 1, 1
        while k < kmax:
            hi = min(k + step, kmax)
            blk = flat[pos + k * L:pos + hi * L].reshape(-1, L)
            bad = np.flatnonzero((blk[:, 1] != t) | (blk[:, 2] != g))
            if bad.size:
                k += int(bad[0])
                break
            k, step = hi, 2 * step
        runs.append((t, g, flat[pos:pos + k * L].reshape(k, L)))
        pos += k * L
    if sum(len(r[2]) for r in runs) != n_all:
        raise ValueError("%s: $Elements declares %d elements but carries %d"
                         % (name, n_all, sum(len(r[2]) for r in runs)))
    return runs


def _same_cells(conn):
    """Two rows of a connectivity block that name the same nodes.

    In:  conn (k, npe) int64 -- cell rows
    Out: (i int, j int) row indices of one repeated cell, or None."""
    s = np.sort(conn, axis=1)
    o = np.lexsort(s.T[::-1])
    hit = np.flatnonzero((s[o[1:]] == s[o[:-1]]).all(axis=1))
    return (int(o[hit[0]]), int(o[hit[0] + 1])) if hit.size else None


def _refuse_repeats(name, top, eid, conn, phys, ent):
    """Refuse kept cells written twice: a repeated element id, or one
    elementary entity under two physical tags whose cell lists repeat (what
    gmsh 2.2 writes for a volume in two physical groups).

    In:  name str -- file name; top int -- SG dimension; eid (n,) int64
         element ids; conn (n, npe) int64; phys (n,) int64 physical tags;
         ent (n,) int64 elementary tags (-1 where the line has none)
    Out: None; ValueError on a repeat."""
    if eid.size > 1 and not (eid[1:] > eid[:-1]).all():
        s = np.sort(eid)
        d = s[1:][s[1:] == s[:-1]]
        if d.size:
            raise ValueError("%s: element id %d appears twice in $Elements"
                             % (name, int(d[0])))
    # (entity, tag) pairs at the starts of constant runs = every pair present
    st = np.flatnonzero(np.r_[True, (phys[1:] != phys[:-1])
                              | (ent[1:] != ent[:-1])])
    pairs = np.unique(np.stack([ent[st], phys[st]], 1), axis=0)
    pairs = pairs[pairs[:, 0] >= 0]
    e_u, n_u = np.unique(pairs[:, 0], return_counts=True)
    for e in e_u[n_u > 1]:
        sel = np.flatnonzero(ent == e)
        if _same_cells(conn[sel]) is not None:
            w, W = _ENTITY.get(top, ("entity", "group"))
            raise ValueError(
                "%s: elementary %s %d is in physical groups %s -- gmsh 2.2"
                " writes its cells once per group (stiffness counted twice):"
                " put each %s in ONE Physical %s"
                % (name, w, int(e), " and ".join(
                    str(int(p)) for p in pairs[pairs[:, 0] == e, 1]), w, W))


def read_msh22(path):
    """The node cloud, the top-dimension cells and their tags of a gmsh
    legacy ASCII 2.2 mesh (vectorised parse).

    In:  path str -- a gmsh 2.2 ASCII `.msh` ($MeshFormat 2.x 0 ..; each
         element line `id type n_tags <tags...> <conn...>`)
    Out: dict {nodes (n_nd, 3) float -- the nodes a kept cell uses, in file
         row order; cells (n_el, npe) int64 1-BASED rows of `nodes`; phys
         (n_el,) int64 set tag per cell (see tag_source); etype int gmsh
         element type; npe int nodes per element; phys_names {tag: name}
         from $PhysicalNames entries of the kept dimension ({} unless
         tag_source is "physical"); dropped {gmsh type: count} the
         lower-dimensional cells skipped; orphans int $Nodes rows no kept
         cell uses (compacted away); tag_source str "physical" (first tag),
         "elementary" (second tag; every physical tag is 0: no physical
         groups, or Mesh.SaveAll) or "none" (no usable tag: every cell tag
         1)}.  Node ids may come in any row order and with gaps; a cell
         naming an id absent from $Nodes, two kept cell types, an
         unsupported type of the kept dimension, 2-D cells spanning z, or a
         cell written twice is a ValueError."""
    name = os.path.basename(path)
    with open(path, "rb") as f:
        _check_format(path, f.read(512))
        f.seek(0)
        raw = f.read()

    n_nd, body = _section(raw, "Nodes", name)
    if n_nd is None:
        raise ValueError("%s has no $Nodes block" % name)
    try:
        xyz = np.fromstring(body, dtype=float, sep=" ")
    except ValueError:
        xyz = np.zeros(0)
    if xyz.size != 4 * n_nd:
        raise ValueError("%s: $Nodes declares %d nodes but does not carry"
                         " %d `id x y z` rows" % (name, n_nd, n_nd))
    xyz = xyz.reshape(n_nd, 4)
    ids = xyz[:, 0].astype(np.int64)

    n_all, body = _section(raw, "Elements", name)
    if n_all is None:
        raise ValueError("%s has no $Elements block" % name)
    runs = _element_runs(body, n_all, name)
    kinds = {}
    for t, _, rows in runs:
        kinds[t] = kinds.get(t, 0) + len(rows)
    if not kinds:
        raise ValueError("%s: $Elements is empty -- no cells" % name)
    big = [t for t in kinds if t not in _SKIP_TYPES]
    top = max((_DIM[t] for t in big), default=-1)
    acc = [t for t in big if t in _SOLID_TYPES and _DIM[t] == top]
    if not acc:
        raise ValueError(
            "%s carries no element type the solid loaders accept (3-/6-node"
            " tri, 4-/9-node quad, 4-/10-node tet, 8-node hex); it has %s"
            % (name, ", ".join(_cell_word(t) for t in sorted(big)
                               if _DIM[t] == top)
               or "only 0-/1-D cells (%s)" % ", ".join(
                   "%d %s" % (kinds[t], _cell_word(t))
                   for t in sorted(kinds))))
    etype = max(acc, key=lambda t: kinds[t])
    other = [t for t in big if t != etype and _DIM[t] == top]
    if other:
        raise ValueError(
            "%s mixes element types: %d cells of the dominant %s plus %s --"
            " one cell type per SG mesh"
            % (name, kinds[etype], _cell_word(etype),
               ", ".join(_cell_word(t) for t in other)))
    npe = _SOLID_TYPES[etype]
    kept = [(g, rows) for t, g, rows in runs if t == etype]
    conn = np.concatenate([rows[:, 3 + g:] for g, rows in kept])

    # node id -> $Nodes row: a dense table, or a sorted search for sparse ids
    if n_nd and ids.min() < 0:
        raise ValueError("%s: negative node id %d" % (name, int(ids.min())))
    mx = int(ids.max()) if n_nd else -1
    if mx < _DENSE_IDS * n_nd + _DENSE_SLACK:
        row = np.full(mx + 2, -1, np.int64)
        row[ids] = np.arange(n_nd)
        dup = ids[row[ids] != np.arange(n_nd)]
        inside = conn.size and conn.min() >= 0 and conn.max() <= mx
        r = row[conn if inside else np.clip(conn, -1, mx + 1)]
    else:
        srt = np.argsort(ids, kind="stable")
        sid = ids[srt]
        dup = sid[1:][sid[1:] == sid[:-1]]
        p = np.minimum(np.searchsorted(sid, conn), n_nd - 1)
        r = np.where(sid[p] == conn, srt[p], -1)
    if dup.size:
        raise ValueError("%s: node id %d appears twice in $Nodes"
                         % (name, int(dup[0])))
    if (r < 0).any():
        raise ValueError("%s: a %s references node id %d, which $Nodes does"
                         " not carry -- corrupt mesh"
                         % (name, _cell_word(etype), int(conn[r < 0][0])))

    # compact to the rows the kept cells use (identity for a clean mesh)
    used = np.zeros(n_nd, bool)
    used[r] = True
    keep = np.flatnonzero(used)
    if keep.size != n_nd:
        r = (np.cumsum(used) - 1)[r]
    nodes = xyz[keep, 1:4]
    if top == 2:
        span = nodes.max(axis=0) - nodes.min(axis=0)
        if span[2] > 1e-9 * max(float(span.max()), 1.0):
            raise ValueError(
                "%s has only 2-D cells (%d %s) spanning z: not a 2-D SG (x-y"
                " plane) -- a 3-D SG needs a Physical Volume (gmsh saves only"
                " physical-group cells); a shell surface goes through"
                " opensg_shell.helper.msh_to_yaml"
                % (name, len(conn), _cell_word(etype)))

    phys = np.concatenate([rows[:, 3] if g >= 1
                           else np.zeros(len(rows), np.int64)
                           for g, rows in kept])
    _refuse_repeats(name, top,
                    np.concatenate([rows[:, 0] for _, rows in kept]),
                    conn, phys,
                    np.concatenate([rows[:, 4] if g >= 2
                                    else np.full(len(rows), -1, np.int64)
                                    for g, rows in kept]))
    src = "physical"
    if not phys.any():
        if all(g >= 2 for g, _ in kept):
            phys = np.concatenate([rows[:, 4] for _, rows in kept])
            src = "elementary"
        else:
            phys, src = np.ones_like(phys), "none"

    phys_names = {}
    pn, body = _section(raw, "PhysicalNames", name)
    if pn and src == "physical":
        # each line: `<dim> <tag> "<name>"`; only the kept dimension names sets
        for ln in [s for s in body.decode("utf-8", "replace").splitlines()
                   if s.strip()][:pn]:
            p = ln.split(None, 2)
            if len(p) == 3 and int(p[0]) == top:
                phys_names[int(p[1])] = p[2].strip().strip('"')

    return {"nodes": nodes, "cells": r + 1, "phys": phys,
            "etype": etype, "npe": npe, "phys_names": phys_names,
            "dropped": {t: kinds[t] for t in kinds if _DIM[t] < top},
            "orphans": int(n_nd - keep.size), "tag_source": src}


def _material_block(m):
    """The yaml lines of ONE material of the solid dialect.

    In:  m dict -- {name, density, E, G, nu}; each elastic entry is a 3-vector
         or a single float (isotropic), which is broadcast to three.
         Optional `angle` [deg]: the ply rotation about y3 (the thickness
         axis), the same `angle:` slot the canonical dialect carries --
         io.sg_input._material_from_list_entry reads it back and the solver
         composes it with the element frame (block angle first).  0 is not
         written: it is the default and `0.0 skips the rotation`.
    Out: list[str] -- the `- name: ... elastic: E/G/nu` yaml lines."""
    def _three(v):
        return tuple(float(x) for x in
                     np.broadcast_to(np.asarray(v, float).ravel(), (3,)))
    out = ["- name: %s" % m["name"],
           "  density: %s" % float(m.get("density", 0.0)),
           "  elastic:",
           "    E: [%.6f, %.6f, %.6f]" % _three(m["E"]),
           "    G: [%.6f, %.6f, %.6f]" % _three(m["G"]),
           "    nu: [%.6f, %.6f, %.6f]" % _three(m["nu"])]
    if m.get("angle") is not None and float(m["angle"]) != 0.0:
        out.append("  angle: %s" % float(m["angle"]))
    return out


def _material_template_entry(set_name):
    """The placeholder yaml lines of ONE still-unfilled material.

    In:  set_name str -- the `sets: element:` name the entry must carry
         (the reader binds sets to materials by name)
    Out: list[str] -- the `- name: ...` FILL_IN entry lines."""
    return ["- name: %s" % set_name,
            "  density: %s_DENSITY_KG_M3" % FILL,
            "  elastic:",
            "    E: [{0}_E1, {0}_E2, {0}_E3]".format(FILL),
            "    G: [{0}_G12, {0}_G13, {0}_G23]".format(FILL),
            "    nu: [{0}_NU12, {0}_NU13, {0}_NU23]".format(FILL)]


def _template_banner():
    """The comment banner above a `materials:` block that still carries
    FILL_IN placeholders.

    In:  --
    Out: list[str] -- the comment lines."""
    return ["# " + "=" * 70,
         "# TEMPLATE -- the mesh blocks BELOW are complete, these MATERIALS"
         " are NOT:",
         "# a gmsh .msh carries geometry and physical tags, no constitutive"
         " data",
         "# (and no orientation -- every element got the constant default"
         " frame).",
         "# opensg refuses to run this file until every %s field is" % FILL,
         "# replaced.  The material names below already match the `sets:`"
         " names",
         "# (that is how the solid reader pairs them); each elastic entry is"
         " the",
         "# nine engineering constants (isotropic: repeat the value three"
         " times,",
         "# G = E/(2(1+nu))).  A ply material may ALSO carry `angle: <deg>`"
         " -- the",
         "# fiber rotation about y3 (the thickness axis); OpenSG angle a ="
         " Abaqus",
         "# *Orientation 3, a (2026-08-25 unified sign: fiber +a).  Omit it"
         " for isotropic/unrotated materials.",
         "# The `opensg msh_to_yaml` mat flags fill entries from the"
         " material",
         "# library instead: --mat<TAG> NAME[:ANGLE] per gmsh physical tag"
         " (see",
         "# io/materials.yaml and io/materials_db.py).",
         "# " + "=" * 70]


def check_filled(path, max_show=12):
    """Is this yaml a still-unfilled msh_to_yaml template?

    The solid-side twin of opensg_shell.helper.msh_to_yaml.check_filled.
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
            " are complete,\nbut `materials:` is still placeholders --"
            " opensg_solid will not invent material\nproperties.  Fill in"
            " these %d %s field(s):\n%s%s\n\n"
            "  a material is  name, density, elastic {E:[E1,E2,E3],"
            " G:[G12,G13,G23], nu:[nu12,nu13,nu23]}\n"
            "                 (isotropic: repeat the value three times,"
            " G = E/(2(1+nu)))\n"
            "  or re-emit FILLED from the material library:\n"
            "    opensg msh_to_yaml <mesh>.msh --mat<TAG> NAME[:ANGLE]"
            " --n_model {1,2,3} --force"
            % (os.path.basename(path), len(hits), FILL, shown, more))


def convert(msh_path, out_path=None, materials=None, phases=None,
            orientation=E1_OUT_OF_PLANE, n_model=None, refined=None):
    """gmsh `.msh` + the caller's materials -> an OpenSG solid SG yaml.

    The mesh supplies the nodes, the cells and the physical-tag split; the
    caller supplies the materials and which physical tag each occupies,
    because neither is in the file.  Given no materials, the yaml comes out
    a marked FILL_IN template (mesh complete, `materials:` placeholders,
    sets named after $PhysicalNames) that check_filled -- and through it the
    CLI -- refuses to run.  One `sets: element:` entry is written per phase,
    and the per-element frame is constant.

    In:  msh_path str -- gmsh legacy ASCII 2.2 mesh
         out_path str | None -- output yaml; None -> <msh stem>.yaml
         materials list[dict] | dict | None --
             list [{name, density, E, G, nu, angle?}, ...] (`angle` deg
             about y3, see _material_block): the classic fully-specified
             call, names the sets after the materials;
             dict {gmsh physical tag: mdict}: the LIBRARY route -- sets
             keep their $PhysicalNames / mat_<tag> names, each covered
             tag's entry is emitted filled (the mdict's `name` is
             overridden by the set name -- the reader binds by name; an
             optional mdict `note` lands as a comment on the name line),
             every uncovered tag keeps its FILL_IN placeholder entry;
             None writes the all-placeholder FILL_IN template.  On a file
             without physical tags (read_msh22 tag_source not "physical")
             a request for tag 0 alone ({0: m}, phases on tag 0) or a
             one-material list covers every cell, as set tag 0
         phases list[(str, int)] | None -- (set name, gmsh physical tag) pairs
             in the same order as `materials`; None -> one phase per distinct
             tag, ascending, named after the material at the same index
             (with a materials LIST) or after $PhysicalNames / "mat_<tag>"
         orientation (9,) floats -- the constant [e1(3) e2(3) e3(3)] frame
             written for every element (default: e1 out of plane, +z)
         n_model int | None -- 1 beam, 2 plate, 3 solid: the macro model,
             written into the header when given.  Never guessed from the
             mesh (sc_to_yaml's rule); None leaves the engine default (2)
             ONLY for the classic list call -- a template OR library-dict
             call writes a FILL_IN placeholder instead, so the engine's
             silent plate default can never pick the wrong macro model
             (the n_model-1-on-a-plate-SG footgun)
         refined int | None -- 0 classical, 1 shear-refined: written into
             the header when given, else left to the engine default
    Out: dict {path, n_nodes, n_elements, npe, order str (linear |
         quadratic), cell str (tet4/hex8/...), sets {name: count},
         missing [str] set names still carrying placeholders, filled
         bool (False = placeholders remain), dropped {gmsh type: count}
         lower-dimensional cells skipped, orphans int unused nodes
         compacted, tag_source str (physical | elementary | none, see
         read_msh22), note str -- the one-line console summary of
         dropped/orphans/tag fallback ("" when none applies)}; the yaml is
         written to `path`, node coordinates at full precision (%.17g)."""
    if n_model is not None and int(n_model) not in (1, 2, 3):
        raise ValueError("n_model must be 1 (beam), 2 (plate) or 3 (solid),"
                         " got %r" % (n_model,))
    if refined is not None and int(refined) not in (0, 1):
        raise ValueError("refined must be 0 (classical) or 1"
                         " (shear-refined), got %r" % (refined,))
    by_tag = materials if isinstance(materials, dict) else None
    mats = materials if (materials is not None and by_tag is None) else None
    M = read_msh22(msh_path)
    nd, cells, phys = M["nodes"], M["cells"], M["phys"]
    src = M["tag_source"]
    want = ({int(k) for k in by_tag} if by_tag is not None else
            {int(t) for _, t in phases} if phases is not None else None)
    whole = src != "physical" and (
        want == {0} or (want is None and mats is not None and len(mats) == 1))
    if whole:
        phys = np.zeros_like(phys)  # tag 0 = the whole untagged mesh
    tags = sorted(set(int(t) for t in phys))
    said = {"physical": "physical tags %s" % tags,
            "elementary": "elementary tags %s (no physical tags: no physical"
                          " groups, or Mesh.SaveAll 1)" % tags,
            "none": "no usable tags (one set, tag 1)"}[src]
    if by_tag is not None:
        stray = sorted(int(k) for k in by_tag if int(k) not in tags)
        if stray:
            raise ValueError(
                "%s has %s -- no tag matches material flag(s) for %s%s"
                % (os.path.basename(msh_path), said, stray,
                   "" if src == "physical"
                   else "; --mat0 alone covers the whole mesh"))
    if phases is None:
        if not mats:
            phases = [(M["phys_names"].get(t, "mat_%d" % t), t) for t in tags]
        elif len(tags) != len(mats):
            raise ValueError(
                "%s has %s but %d materials were given -- pass"
                " phases=[(set_name, tag), ...] explicitly"
                % (os.path.basename(msh_path), said, len(mats)))
        else:
            phases = [(m["name"], t) for m, t in zip(mats, tags)]
    missing = ([] if mats else
               [name for name, tag in phases
                if (by_tag or {}).get(int(tag)) is None])
    ori = ", ".join("%s" % float(v) for v in orientation)
    order, cell_name = _ORDER_OF[M["etype"]]

    out = ["msg: solid      # the ENGINE this SG belongs to (opensg_solid);"
           " `opensg <yaml>` dispatches on it"]
    if n_model is not None:
        out.append("n_model: %d      # 1 = beam, 2 = plate, 3 = solid -- the"
                   " macro model this SG homogenizes to" % int(n_model))
    elif mats is None:
        # a template (and the library route, which may be partial) defers
        # the macro model to the same fill-in step as the materials: the
        # mesh cannot say which model it serves, and the engine's silent
        # default (2, plate) is wrong for a 3-D SG
        out.append("n_model: %s_N_MODEL      # REPLACE with 1 = beam, 2 ="
                   " plate or 3 = solid -- the macro model this SG"
                   " homogenizes to (the mesh cannot say)" % FILL)
    # `refined:` is ALWAYS written (default 0 = classical) so the file
    # states the model it runs as; the mesh grade is NOT a yaml key --
    # the solver reads the arity off the cells (console-only info)
    out.append("refined: %d      # 0 = classical, 1 = shear-refined"
               % int(refined if refined is not None else 0))
    # CONSTITUTIVE FIRST: `materials:` goes directly under the header, ahead
    # of the mesh lines, so the one block a human edits is at the top of the
    # file (a yaml mapping carries no order -- readability only, the same
    # decision the shell twin documents)
    # a library-covered tag takes the MATERIAL's name for both the
    # material block and its element set (the reader binds material to
    # set by name): --mat1 Al emits `name: Al` + set Al, not the mesh's
    # Mat0.  Collisions (two tags, one material) uniquify with _<tag>.
    if by_tag:
        seen, renamed = set(), []
        for name, tag in phases:
            m = by_tag.get(int(tag))
            if m is not None and m.get("name"):
                nm = str(m["name"])
                if nm in seen:
                    nm = "%s_%d" % (nm, int(tag))
                seen.add(nm)
                renamed.append((nm, tag))
            else:
                renamed.append((name, tag))
        phases = renamed
    if mats:
        out.append("materials:")
        for m in mats:
            out += _material_block(m)
    else:
        if missing:
            out += _template_banner()
        out.append("materials:")
        for name, tag in phases:
            m = (by_tag or {}).get(int(tag))
            if m is None:
                out += _material_template_entry(name)
                continue
            mm = dict(m)
            mm["name"] = name
            blk = _material_block(mm)
            if m.get("note") and str(m["note"]) != str(name):
                blk[0] += "    # %s" % m["note"]
            out += blk
    out.append("nodes:")
    out += ["- [%.17g %.17g %.17g]" % (x, y, z) for x, y, z in nd]
    out.append("elements:")
    fmt = "- [" + " ".join(["%d"] * M["npe"]) + "]"
    out += [fmt % tuple(c) for c in cells]
    out.append("elementOrientations:")
    out += ["- [%s]" % ori] * len(cells)
    out += ["sets:", "  element:"]
    counts = {}
    for name, tag in phases:
        idx = np.where(phys == int(tag))[0]
        counts[name] = int(idx.size)
        out.append("  - name: %s" % name)
        out.append("    labels:")
        out += ["    - %d" % (e + 1) for e in idx]

    path = out_path or (os.path.splitext(msh_path)[0] + ".yaml")
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    note, drop = [], M["dropped"]
    if drop:
        note.append("dropped %s lower-dim cell%s" % (
            " + ".join("%d %s" % (drop[t], _cell_word(t))
                       for t in sorted(drop, key=lambda t: (-_DIM[t], t))),
            "" if sum(drop.values()) == 1 else "s"))
    if M["orphans"]:
        note.append("compacted %d orphan node%s"
                    % (M["orphans"], "" if M["orphans"] == 1 else "s"))
    if whole:
        note.append("no physical tags: one set (tag 0)")
    elif src == "elementary":
        note.append("no physical tags: sets from elementary tags %s" % tags)
    elif src == "none":
        note.append("no physical tags: one set (tag 1)")
    return {"path": path, "n_nodes": len(nd), "n_elements": len(cells),
            "npe": M["npe"], "order": order, "cell": cell_name,
            "sets": counts, "missing": missing,
            "filled": bool(mats) or (by_tag is not None and not missing),
            "dropped": M["dropped"], "orphans": M["orphans"],
            "tag_source": src, "note": "; ".join(note)}
