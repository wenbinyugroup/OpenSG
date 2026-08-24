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
triangles (gmsh type 2), 4-node quads (type 3), 4-node tets (type 4) and
8-node hexes (type 5).  The 0-/1-D entities gmsh writes for physical points
and curves are skipped; anything else is an error.

Use:  from opensg_solid.io.msh_to_yaml import convert
      convert("UDcomp_2D.msh", materials=[...], phases=[("matrix", 0),
                                                        ("fiber", 1)])
      convert("SP_solid.msh")                  # mesh-only FILL_IN template
"""
import os

import numpy as np

# the placeholder marker an unfilled template carries; check_filled (and,
# through it, opensg_solid.cli) refuses to run a yaml that still contains one
FILL = "FILL_IN"

# gmsh element type -> node count, for the types the solid loaders accept
_SOLID_TYPES = {2: 3, 3: 4, 4: 4, 5: 8}
# gmsh 0-/1-D entities: written for physical points/curves, not solid cells
_SKIP_TYPES = {15: 1, 1: 2, 8: 3}
_TYPE_NAME = {6: "6-node prism", 7: "5-node pyramid",
              9: "6-node triangle (2nd order)", 10: "9-node quad (2nd order)",
              11: "10-node tetrahedron (2nd order)",
              16: "8-node quad (2nd order)"}

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
            " 4-node quad, 4-node tet, 8-node hex); it has %s"
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


def _material_template_block(set_names):
    """The `materials:` yaml lines of the FILL_IN template -- one entry per
    element set, the NAMES already filled (the reader binds sets to
    materials by name), every number a placeholder.

    In:  set_names list[str] -- the `sets: element:` names, in order
    Out: list[str] -- the banner comment + the placeholder entries."""
    o = ["# " + "=" * 70,
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
         "# " + "=" * 70,
         "materials:"]
    for s in set_names:
        o += ["- name: %s" % s,
              "  density: %s_DENSITY_KG_M3" % FILL,
              "  elastic:",
              "    E: [{0}_E1, {0}_E2, {0}_E3]".format(FILL),
              "    G: [{0}_G12, {0}_G13, {0}_G23]".format(FILL),
              "    nu: [{0}_NU12, {0}_NU13, {0}_NU23]".format(FILL)]
    return o


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
            " G = E/(2(1+nu)))"
            % (os.path.basename(path), len(hits), FILL, shown, more))


def convert(msh_path, out_path=None, materials=None, phases=None,
            orientation=E1_OUT_OF_PLANE, n_model=None):
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
         materials list[dict] | None -- [{name, density, E, G, nu,
             angle?}, ...] (`angle` deg about y3, see _material_block);
             None writes the FILL_IN `materials:` template instead
         phases list[(str, int)] | None -- (set name, gmsh physical tag) pairs
             in the same order as `materials`; None -> one phase per distinct
             tag, ascending, named after the material at the same index
             (with materials) or after $PhysicalNames / "mat_<tag>" (without)
         orientation (9,) floats -- the constant [e1(3) e2(3) e3(3)] frame
             written for every element (default: e1 out of plane, +z)
         n_model int | None -- 1 beam, 2 plate, 3 solid: the macro model,
             written into the header when given.  Never guessed from the
             mesh (sc_to_yaml's rule); None leaves the engine default (2)
             for a runnable yaml, and a FILL_IN placeholder in a template
    Out: dict {path, n_nodes, n_elements, npe, sets {name: count}, filled
         bool (False = template)}; the yaml is written to `path`."""
    if n_model is not None and int(n_model) not in (1, 2, 3):
        raise ValueError("n_model must be 1 (beam), 2 (plate) or 3 (solid),"
                         " got %r" % (n_model,))
    M = read_msh22(msh_path)
    nd, cells, phys = M["nodes"], M["cells"], M["phys"]
    if phases is None:
        tags = sorted(set(int(t) for t in phys))
        if not materials:
            phases = [(M["phys_names"].get(t, "mat_%d" % t), t) for t in tags]
        elif len(tags) != len(materials):
            raise ValueError(
                "%s has %d physical tags %s but %d materials were given --"
                " pass phases=[(set_name, tag), ...] explicitly"
                % (os.path.basename(msh_path), len(tags), tags,
                   len(materials)))
        else:
            phases = [(m["name"], t) for m, t in zip(materials, tags)]
    ori = ", ".join("%s" % float(v) for v in orientation)

    out = ["msg: solid      # the ENGINE this SG belongs to (opensg_solid);"
           " `opensg <yaml>` dispatches on it"]
    if n_model is not None:
        out.append("n_model: %d      # 1 = beam, 2 = plate, 3 = solid -- the"
                   " macro model this SG homogenizes to" % int(n_model))
    elif not materials:
        # a template defers the macro model to the same fill-in step as the
        # materials: the mesh cannot say which model it serves, and the
        # engine's silent default (2, plate) is wrong for a 3-D SG
        out.append("n_model: %s_N_MODEL      # REPLACE with 1 = beam, 2 ="
                   " plate or 3 = solid -- the macro model this SG"
                   " homogenizes to (the mesh cannot say)" % FILL)
    # CONSTITUTIVE FIRST: `materials:` goes directly under the header, ahead
    # of the mesh lines, so the one block a human edits is at the top of the
    # file (a yaml mapping carries no order -- readability only, the same
    # decision the shell twin documents)
    if materials:
        out.append("materials:")
        for m in materials:
            out += _material_block(m)
    else:
        out += _material_template_block([name for name, _ in phases])
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
            "npe": M["npe"], "sets": counts, "filled": bool(materials)}
