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
material is a modelling decision the mesh cannot supply.  This is the
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
"""
import os

import numpy as np

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
         npe int nodes per element}.  The dominant accepted cell type wins;
         any other non-skippable type is an error."""
    lines = open(path).read().split("\n")
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
            "npe": npe}


def _material_block(m):
    """The yaml lines of ONE material of the solid dialect.

    In:  m dict -- {name, density, E, G, nu}; each elastic entry is a 3-vector
         or a single float (isotropic), which is broadcast to three
    Out: list[str] -- the `- name: ... elastic: E/G/nu` yaml lines."""
    def _three(v):
        return tuple(float(x) for x in
                     np.broadcast_to(np.asarray(v, float).ravel(), (3,)))
    return ["- name: %s" % m["name"],
            "  density: %s" % float(m.get("density", 0.0)),
            "  elastic:",
            "    E: [%.6f, %.6f, %.6f]" % _three(m["E"]),
            "    G: [%.6f, %.6f, %.6f]" % _three(m["G"]),
            "    nu: [%.6f, %.6f, %.6f]" % _three(m["nu"])]


def convert(msh_path, out_path=None, materials=None, phases=None,
            orientation=E1_OUT_OF_PLANE):
    """gmsh `.msh` + the caller's materials -> an OpenSG solid SG yaml.

    The mesh supplies the nodes, the cells and the physical-tag split; the
    caller supplies the materials and which physical tag each occupies,
    because neither is in the file.  One `sets: element:` entry is written per
    phase, named by the caller, and the per-element frame is constant.

    In:  msh_path str -- gmsh legacy ASCII 2.2 mesh
         out_path str | None -- output yaml; None -> <msh stem>.yaml
         materials list[dict] -- [{name, density, E, G, nu}, ...] (required)
         phases list[(str, int)] | None -- (set name, gmsh physical tag) pairs
             in the same order as `materials`; None -> one phase per distinct
             tag, ascending, named after the material at the same index
         orientation (9,) floats -- the constant [e1(3) e2(3) e3(3)] frame
             written for every element (default: e1 out of plane, +z)
    Out: dict {path, n_nodes, n_elements, npe, sets {name: count}}; the yaml
         is written to `path`."""
    if not materials:
        raise ValueError(
            "materials= is required: a gmsh mesh carries no constitutive data,"
            " so the phases cannot be named or given properties from the file"
            " alone")
    M = read_msh22(msh_path)
    nd, cells, phys = M["nodes"], M["cells"], M["phys"]
    if phases is None:
        tags = sorted(set(int(t) for t in phys))
        if len(tags) != len(materials):
            raise ValueError(
                "%s has %d physical tags %s but %d materials were given --"
                " pass phases=[(set_name, tag), ...] explicitly"
                % (os.path.basename(msh_path), len(tags), tags,
                   len(materials)))
        phases = [(m["name"], t) for m, t in zip(materials, tags)]
    ori = ", ".join("%s" % float(v) for v in orientation)

    out = ["msg: solid      # the ENGINE this SG belongs to (opensg_solid);"
           " `opensg <yaml>` dispatches on it",
           "nodes:"]
    out += ["- [%.12f %.12f %.12f]" % (x, y, z) for x, y, z in nd]
    out.append("elements:")
    fmt = "- [" + " ".join(["%d"] * M["npe"]) + "]"
    out += [fmt % tuple(c) for c in cells]
    out.append("elementOrientations:")
    out += ["- [%s]" % ori] * len(cells)
    out.append("materials:")
    for m in materials:
        out += _material_block(m)
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
            "npe": M["npe"], "sets": counts}
