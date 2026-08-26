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

Only the element types the solid drivers accept are converted -- 3-node
triangles (gmsh type 2), 4-node quads (type 3), 4-node tets (type 4),
8-node hexes (type 5) and 10-node tets (type 11, the quadratic grade
helper.linear_msh_to_quad emits).  The 0-/1-D entities gmsh writes for
physical points and curves are skipped; anything else is an error.  The
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
_SOLID_TYPES = {2: 3, 3: 4, 4: 4, 5: 8, 11: 10}
# gmsh 0-/1-D entities: written for physical points/curves, not solid cells
_SKIP_TYPES = {15: 1, 1: 2, 8: 3}
_TYPE_NAME = {6: "6-node prism", 7: "5-node pyramid",
              9: "6-node triangle (2nd order)", 10: "9-node quad (2nd order)",
              16: "8-node quad (2nd order)",
              29: "20-node tetrahedron (3rd order)"}
# accepted type -> (the informational `mesh_order:` word, the cell name)
_ORDER_OF = {2: ("linear", "tri3"), 3: ("linear", "quad4"),
             4: ("linear", "tet4"), 5: ("linear", "hex8"),
             11: ("quadratic", "tet10")}

# the default per-element frame of a 2-D cross-section SG: e1 along the beam
# axis (out of plane, +z), e2 = +x, e3 = +y -- the frame the 2-D solid SG
# examples use (Rules/orientation_e1_out_of_plane.md)
E1_OUT_OF_PLANE = (0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def read_msh22(path):
    """The node cloud, the surface/volume cells and their physical tags of a
    gmsh legacy ASCII 2.2 mesh.

    In:  path str -- a gmsh 2.2 `.msh` file ($Nodes / $Elements blocks, each
         element line `id type n_tags <tags...> <conn...>`)
    Out: dict {nodes (n_nd, 3) float, cells (n_el, npe) int 1-BASED,
         phys (n_el,) int gmsh physical tag, etype int gmsh element type,
         npe int nodes per element, phys_names {tag: name} from the optional
         $PhysicalNames block ({} without one)}.  The dominant accepted cell
         type wins; any other non-skippable type is an error."""
    lines = open(path).read().split("\n")
    phys_names = {}
    if "$PhysicalNames" in lines:
        # each line: `<dim> <tag> "<name>"` after the count line
        h = lines.index("$PhysicalNames")
        for k in range(int(lines[h + 1])):
            p = lines[h + 2 + k].split(None, 2)
            phys_names[int(p[1])] = p[2].strip().strip('"')
    i = lines.index("$Nodes")
    n_nd = int(lines[i + 1])
    nodes = np.array([[float(v) for v in lines[i + 2 + k].split()[1:4]]
                      for k in range(n_nd)], float)
    j = lines.index("$Elements")
    n_all = int(lines[j + 1])
    kinds = {}
    for k in range(n_all):
        p = lines[j + 2 + k].split()
        kinds.setdefault(int(p[1]), []).append(p)
    solid = [t for t in kinds if t in _SOLID_TYPES]
    if not solid:
        bad = sorted(t for t in kinds if t not in _SKIP_TYPES)
        raise ValueError(
            "%s carries no element type the solid loaders accept (3-node tri,"
            " 4-node quad, 4-/10-node tet, 8-node hex); it has %s"
            % (os.path.basename(path),
               ", ".join(_TYPE_NAME.get(t, "gmsh type %d" % t) for t in bad)
               or "only 0-/1-D entities"))
    etype = max(solid, key=lambda t: len(kinds[t]))
    npe = _SOLID_TYPES[etype]
    rows = kinds[etype]
    other = [t for t in kinds if t != etype and t not in _SKIP_TYPES]
    if other:
        raise ValueError(
            "%s mixes element types: %d cells of the dominant %s plus %s --"
            " one cell type per SG mesh"
            % (os.path.basename(path), len(rows),
               _TYPE_NAME.get(etype, "gmsh type %d" % etype),
               ", ".join(_TYPE_NAME.get(t, "gmsh type %d" % t) for t in other)))
    cells = np.array([[int(v) for v in p[-npe:]] for p in rows], int)
    phys = np.array([int(p[3]) for p in rows], int)
    return {"nodes": nodes, "cells": cells, "phys": phys, "etype": etype,
            "npe": npe, "phys_names": phys_names}


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
         "# *Orientation 3, -a.  Omit it for isotropic/unrotated materials.",
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
             None writes the all-placeholder FILL_IN template
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
         bool (False = placeholders remain)}; the yaml is written to
         `path`."""
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
    tags = sorted(set(int(t) for t in phys))
    if by_tag is not None:
        stray = sorted(int(k) for k in by_tag if int(k) not in tags)
        if stray:
            raise ValueError(
                "%s has physical tags %s -- no tag matches material flag(s)"
                " for %s" % (os.path.basename(msh_path), tags, stray))
    if phases is None:
        if not mats:
            phases = [(M["phys_names"].get(t, "mat_%d" % t), t) for t in tags]
        elif len(tags) != len(mats):
            raise ValueError(
                "%s has %d physical tags %s but %d materials were given --"
                " pass phases=[(set_name, tag), ...] explicitly"
                % (os.path.basename(msh_path), len(tags), tags, len(mats)))
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
    out += ["- [%.12f %.12f %.12f]" % (x, y, z) for x, y, z in nd]
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
    return {"path": path, "n_nodes": len(nd), "n_elements": len(cells),
            "npe": M["npe"], "order": order, "cell": cell_name,
            "sets": counts, "missing": missing,
            "filled": bool(mats) or (by_tag is not None and not missing)}
