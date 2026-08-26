"""inp_to_yaml.py -- Abaqus `.inp` mesh -> OpenSG solid SG yaml (+ gmsh
`.msh`), the Abaqus-side sibling of io.sc_to_yaml (SwiftComp) and
io.msh_to_yaml (gmsh).

WHAT IT READS from the deck: `*Node`, every `*Element` block (an .inp
may carry SEVERAL, one per type -- a MIXED mesh comes out as ragged
cells, which load_sg_input accepts and the batched homo assembles
type-by-type), `*Elset` (incl. `generate`), `*Solid Section` /
`*Shell Section` (the elset -> material binding when the deck has one),
`*Material` with `*Elastic` (isotropic or ENGINEERING CONSTANTS) and
`*Density`.  Steps, BCs, loads and output requests are ignored -- the
SG yaml describes the structure alone.

ELEMENT TYPES accepted (others raise):
    tri3   S3, S3R, STRI3, CPS3, CPE3, CPS3T, CPE3T
    quad4  S4, S4R, S4RS, CPS4, CPS4R, CPE4, CPE4R, CPE4I
    tet4   C3D4
    hex8   C3D8, C3D8R, C3D8I
    tet10  C3D10 -- Abaqus lists the midsides on edges (12, 23, 13, 14,
           24, 34), the .sc convention; the emitted yaml/.msh carry the
           GMSH order (12, 23, 13, 14, 34, 24), so the last two swap
           (identical to sc_to_yaml's tet10 rule; the gmsh->basix hop
           stays in sg_homo._to_basix_order).
Abaqus quad4/hex8 node order coincides with gmsh, so no other reorder.

COORDINATES.  The SG occupies the deck columns that actually VARY
(span > 1e-9 of the largest), kept in deck order and written into the
LEADING yaml columns -- a shell mesh drawn in the x-y plane at z = 0
becomes a 2-D SG with col0 = x, col1 = y (for a plate SG the LAST
varying column is the thickness direction, the engine's rule).  All
three varying -> a 3-D SG, verbatim.

MATERIAL ASSIGNMENT.  Section cards bind elsets to materials when the
deck has them.  Without sections (a mesh-only deck like
Core_2UC_SC.inp, which carries two `*Material` blocks and none), the
OPTIONAL `material=` / `elset_map=` arguments decide; given neither,
the converter emits ALL the deck's materials into the yaml, defaults
every element to material 1 (the deck's first `*Material`) and says so
LOUDLY -- the material set varies deck to deck, so the binding is the
user's edit in the yaml (`mat_id:` / `materials:`), not a CLI
requirement.

n_model is REQUIRED, never guessed (sc_to_yaml's rule): the mesh
cannot say which macro model it serves.

PLY ANGLES.  A binding value may be `MATERIAL@DEGREES`: plies that
share one deck material and differ ONLY by orientation (a [+45/-45]s
face sheet whose .inp carries no `*Orientation`) then become SEPARATE
SG materials carrying `angle:`, instead of collapsing into one id.

Use:  from opensg_solid.io.inp_to_yaml import convert
      convert("Core_2UC_SC.inp", n_model=2, refined=1)   # core only
      convert("HC_2UC_SW.inp", n_model=2, refined=1,
              elset_map={"P1": "Aluminum", "L1": "Laminate@45",
                         "L2": "Laminate@-45"})

# ----------------------------------------------------------------------------
# ALL VARIABLES USED IN THIS MODULE
# ----------------------------------------------------------------------------
# inputs
#   inp_path, out_base      the .inp; basename of the emitted sidecars
#   n_model, refined        the yaml analysis header (REQUIRED / 0)
#   material, elset_map     the constitutive binding when the deck has
#                           no section cards
# working
#   nodes {id: (3,)}        deck coordinates, 1-based ids
#   blocks [(type, rows)]   one entry per *Element block, rows 1-based
#   elsets {NAME: [eid]}    names UPPERCASED (Abaqus is case-blind)
#   sections [(elset, mat)] the deck's section bindings, in order
#   mats {NAME: dict}       type 0 (E, nu) | 1 (engineering), density
#   used_cols               deck columns with nonzero span (SG coords)
# outputs
#   sc                      {dim, nodes, cells, mat_id, materials} in
#                           the canonical load_sg_input dialect
#   <base>.yaml, <base>.msh via sc_to_yaml.write_yaml_file/write_msh
# ----------------------------------------------------------------------------
"""
import os

import numpy as np

from opensg_solid.io.sc_to_yaml import write_msh, write_yaml_file

_TRI3 = {"S3", "S3R", "STRI3", "CPS3", "CPE3", "CPS3T", "CPE3T"}
_QUAD4 = {"S4", "S4R", "S4RS", "CPS4", "CPS4R", "CPE4", "CPE4R", "CPE4I"}
_TET4 = {"C3D4"}
_HEX8 = {"C3D8", "C3D8R", "C3D8I"}
_TET10 = {"C3D10"}
_ACCEPT = _TRI3 | _QUAD4 | _TET4 | _HEX8 | _TET10


def read_inp(path):
    """Parse the mesh side of an Abaqus .inp.

    In:  path str -- the deck
    Out: dict {nodes {id: (x, y, z)}, blocks [(TYPE, [[eid, n...]])],
         elsets {NAME: [eid]}, sections [(ELSET, MATERIAL)],
         mats {NAME: {type, E/nu | engineering, density}}}."""
    nodes, blocks, elsets, sections, mats = {}, [], {}, [], {}
    mode = cur = mat = None
    pend = None
    for ln in open(path):
        s = ln.strip()
        if not s or s.startswith("**"):
            continue
        if s.startswith("*"):
            low = s.lower()
            key = {}
            for tok in s.split(",")[1:]:
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    key[k.strip().lower()] = v.strip()
            if low.startswith("*node") and "output" not in low:
                mode = "node"
            elif low.startswith("*element,") or low == "*element":
                et = key.get("type", "").upper()
                blocks.append((et, []))
                mode, pend = "elem", None
            elif low.startswith("*elset"):
                cur = key.get("elset", "").upper()
                elsets.setdefault(cur, [])
                mode = "elsetg" if "generate" in low else "elset"
            elif (low.startswith("*solid section")
                  or low.startswith("*shell section")):
                sections.append((key.get("elset", "").upper(),
                                 key.get("material", "")))
                mode = None
            elif low.startswith("*material"):
                mat = key.get("name", "")
                mats[mat] = {}
                mode = None
            elif low.startswith("*density"):
                mode = "dens"
            elif low.startswith("*elastic"):
                mode = ("eng" if "engineering" in low else "iso")
                mats[mat]["_rows"] = []
            else:
                mode = None
            continue
        p = [v for v in s.replace(",", " ").split()]
        if mode == "node":
            nodes[int(p[0])] = (float(p[1]), float(p[2]),
                                float(p[3]) if len(p) > 3 else 0.0)
        elif mode == "elem":
            frag = [int(v) for v in p]
            if pend is not None:
                pend.extend(frag)        # pend aliases the stored row
                row = pend
            else:
                blocks[-1][1].append(frag)
                row = frag
            # an Abaqus data line ending in "," continues on the next
            pend = row if s.endswith(",") else None
        elif mode == "elset":
            elsets[cur] += [int(v) for v in p]
        elif mode == "elsetg":
            a, b = int(p[0]), int(p[1])
            c = int(p[2]) if len(p) > 2 else 1
            elsets[cur] += list(range(a, b + 1, c))
        elif mode == "dens":
            mats[mat]["density"] = float(p[0])
            mode = None
        elif mode in ("iso", "eng"):
            mats[mat]["_rows"] += [float(v) for v in p]
            r = mats[mat]["_rows"]
            if mode == "iso" and len(r) >= 2:
                mats[mat].update(type=0, E=r[0], nu=r[1])
                del mats[mat]["_rows"]
                mode = None
            elif mode == "eng" and len(r) >= 9:
                # Abaqus order E1 E2 E3 nu12 nu13 nu23 G12 G13 G23 ->
                # the OpenSG type-1 order E1 E2 E3 G12 G13 G23 v12 v13 v23
                mats[mat].update(type=1, engineering=[
                    r[0], r[1], r[2], r[6], r[7], r[8], r[3], r[4], r[5]])
                del mats[mat]["_rows"]
                mode = None
    return {"nodes": nodes, "blocks": blocks, "elsets": elsets,
            "sections": sections, "mats": mats}


def _npe(et):
    """Nodes per element of an accepted Abaqus type (else raise)."""
    for group, n in ((_TRI3, 3), (_QUAD4, 4), (_TET4, 4), (_HEX8, 8),
                     (_TET10, 10)):
        if et in group:
            return n
    raise ValueError(
        "element type %s is not tri3/quad4/tet4/hex8/tet10 -- the SG "
        "engine cannot run it" % et)


def convert(inp_path, out_base=None, n_model=None, refined=0,
            material=None, elset_map=None):
    """Read an .inp, write <base>.yaml + <base>.msh, return the sc dict.

    In:  inp_path str -- the Abaqus deck; out_base str basename for the
         sidecars (None -> the .inp stem); n_model 1 beam | 2 plate |
         3 solid, REQUIRED (never guessed from the mesh);
         refined 0 classical | 1 shear-refined;
         material str | None -- ONE material name for every element
         (mesh-only decks); elset_map {ELSET: MATERIAL} | None --
         explicit binding (wins over the deck's section cards)
    Out: dict {dim, nodes, cells, mat_id, materials} (the canonical
         load_sg_input dialect; cells RAGGED for a mixed mesh)."""
    if n_model is None:
        raise ValueError("n_model is required (1 beam, 2 plate, 3 "
                         "solid) -- the mesh cannot say which macro "
                         "model it serves")
    d = read_inp(inp_path)
    if not d["nodes"] or not d["blocks"]:
        raise ValueError("%s carries no *Node/*Element mesh" % inp_path)

    ids = sorted(d["nodes"])
    remap = {nid: k for k, nid in enumerate(ids)}
    xyz = np.array([d["nodes"][nid] for nid in ids], float)
    span = xyz.max(axis=0) - xyz.min(axis=0)
    used = [k for k in range(3) if span[k] > 1e-9 * max(span.max(), 1.0)]
    dim = len(used)
    nd = np.zeros((len(ids), 3))
    nd[:, :dim] = xyz[:, used]

    cells, etypes, eids = [], [], []
    for et, rows in d["blocks"]:
        npe = _npe(et)
        for row in rows:
            if len(row) != npe + 1:
                raise ValueError(
                    "%s: element %d of type %s has %d nodes, expected "
                    "%d" % (inp_path, row[0], et, len(row) - 1, npe))
            conn = [remap[n] for n in row[1:]]
            if et in _TET10:
                # Abaqus midside order = the .sc order; gmsh swaps the
                # last two (edges 24 <-> 34) -- sc_to_yaml's tet10 rule
                conn = conn[:8] + [conn[9], conn[8]]
            cells.append(conn)
            etypes.append(et)
            eids.append(row[0])
    order = np.argsort(np.array(eids))
    cells = [cells[k] for k in order]
    eid_of = np.array(eids)[order]
    if len(set(eid_of.tolist())) != len(eid_of):
        raise ValueError("%s: duplicate element ids across *Element "
                         "blocks" % inp_path)

    # ---- material binding: explicit map > deck sections > `material=`
    bind = []
    if elset_map:
        bind = [(k.upper(), v) for k, v in elset_map.items()]
    elif d["sections"]:
        bind = d["sections"]
    name_of = {}                       # eid -> (material name, angle)
    for elset, spec in bind:
        # "MAT" or "MAT@ANGLE" -- plies that share a deck material and
        # differ ONLY by orientation (a [+45/-45]s face sheet whose .inp
        # carries no *Orientation) become SEPARATE SG materials
        matname, _, ang = str(spec).partition("@")
        matname = matname.strip()
        try:
            angle = float(ang) if ang.strip() else 0.0
        except ValueError:
            raise ValueError("%s: elset %s angle %r is not a number "
                             "(use MATERIAL@DEGREES)"
                             % (inp_path, elset, ang))
        if elset not in d["elsets"]:
            raise ValueError("%s: section/elset_map names elset %s, "
                             "not in the deck" % (inp_path, elset))
        if matname not in d["mats"]:
            raise ValueError("%s: elset %s bound to unknown material "
                             "%s" % (inp_path, elset, matname))
        for e in d["elsets"][elset]:
            name_of[e] = (matname, angle)
    defaulted = False
    if not name_of:
        if material is not None:
            mname, _, ang = str(material).partition("@")
            mname = mname.strip()
            if mname not in d["mats"]:
                raise ValueError("material %r not in the deck (has %s)"
                                 % (mname, sorted(d["mats"])))
            ang = float(ang) if ang.strip() else 0.0
            name_of = {e: (mname, ang) for e in eid_of}
        else:
            # no sections, no caller binding: EVERY deck material goes
            # into the yaml, every element defaults to the FIRST one --
            # the binding is the user's manual edit, never a CLI demand
            first = next(iter(d["mats"]))
            name_of = {e: (first, 0.0) for e in eid_of}
            defaulted = len(d["mats"]) > 1
    missing = [int(e) for e in eid_of if e not in name_of]
    if missing:
        raise ValueError("%s: %d elements have no material binding "
                         "(first: %s)" % (inp_path, len(missing),
                                          missing[:5]))

    if bind or material is not None:
        used = []                      # (name, angle) keys, deck order
        for nm in d["mats"]:
            for key in dict.fromkeys(name_of.values()):
                if key[0] == nm and key not in used:
                    used.append(key)
    else:
        used = [(nm, 0.0) for nm in d["mats"]]   # keep ALL: user reassigns
    mat_no = {key: k + 1 for k, key in enumerate(used)}
    mat_id = np.array([mat_no[name_of[e]] for e in eid_of], int)
    materials = {}
    for key in used:
        m = dict(d["mats"][key[0]])
        m.setdefault("aux", [0.0, m.pop("density", 0.0)])
        if key[1]:
            m["angle"] = float(key[1])
        materials[mat_no[key]] = m
    used_names = ["%s@%g" % key if key[1] else key[0] for key in used]

    sc = {"dim": dim, "nodes": nd, "cells": cells, "mat_id": mat_id,
          "materials": materials}
    if out_base is None:
        out_base = os.path.splitext(inp_path)[0]
    write_yaml_file(sc, out_base + ".yaml", n_model=int(n_model),
                    refined=int(refined))
    write_msh(sc, out_base + ".msh")
    from collections import Counter
    print("inp_to_yaml: %dD SG, %d nodes, %d cells (%s), %d materials"
          " (%s), n_model %d -> %s.yaml/.msh"
          % (dim, len(nd), len(cells),
             ", ".join("%s:%d" % kv for kv in
                       sorted(Counter(etypes).items())),
             len(materials), ", ".join(used_names), int(n_model),
             out_base))
    if defaulted:
        print("inp_to_yaml: NOTE -- %s has no *Solid/Shell Section "
              "cards; every element DEFAULTED to material 1 (%s). The "
              "deck also carries %s: edit `mat_id:`/`materials:` in "
              "the yaml (all deck materials are emitted), or re-run "
              "with --material NAME / --map ELSET=MAT."
              % (os.path.basename(inp_path), used_names[0],
                 ", ".join(used_names[1:])))
    return sc
