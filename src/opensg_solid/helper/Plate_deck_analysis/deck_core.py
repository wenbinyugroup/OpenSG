# vendored 2026-09-03 from Abaqus/skills/abaqus-inp-from-3dsg, the
# validated TPMS deck writer; upstream stays authoritative for the
# skill.  Only change: the plate --subdiv DEFAULT is 3 -- subdiv = 3 is
# the pipeline default, 2 has no pure second derivatives.
"""sg_plate_deck.py -- Abaqus decks for a plate tiled from a periodic
3-D Structure Gene, and the equivalent-plate S4R twin of its
homogenized 8x8 ABDG law, on one shared centred frame.

Frame: 1 = span, 2 = width, 3 = thickness.  The plate is centred on the
origin in-plane (x in [-a/2, a/2], y in [-b/2, b/2]) and the SG's own z
is PRESERVED VERBATIM.

Why z is preserved.  OpenSG builds the plate kinematic operator from the
raw x3 of each Gauss point, so the law is referred to z = 0 IN THE SG'S
OWN COORDINATES.  Preserving z makes the S4R nodal surface, the solid's
z = 0 and the law's reference surface the same physical surface by
construction -- no matter where that surface sits in the cell.  A
non-centred SG is therefore NOT an error: its law simply carries a
non-zero B block, `*Shell General Section` carries B explicitly, and the
deck is still right.  `gate` reports the offset and B as DIAGNOSTICS,
not as a pass/fail.

Subcommands
  gate    SG bbox, z mid-surface offset, cell solid volume and relative
          density, and (with --law) the law's diagonal and B norm
  solid   the tiled 3-D deck (C3D4 / C3D10 / C3D8I)
  plate   the one-S4R-per-cell equivalent plate deck

In:  an OpenSG solid SG yaml (canonical or mesh dialect); for `plate`
     also the 8x8 `refined: 1` .out of that same SG
Out: <out>/<stem>_3d_<nx>x<ny>_<etype>.inp   or
     <out>/<stem>_plate_<nx>x<ny>.inp
"""
import argparse
import os
import re
import shutil
import sys

import numpy as np

# Abaqus names each solid-element face P1..Pn, and the name is defined
# by which CORNER nodes bound it.  This is that table, 0-based local.
# Used only to find the pressure-loaded faces: a face is loaded when
# ALL its corner nodes lie on the loaded surface, which is a SET test --
# so the ordering here need not match the manual's outward-normal
# ordering, only the node sets, and those do match.  C3D10 shares the
# C3D4 corner definitions, so one table serves both.
_FACES = {4: [("P1", (0, 1, 2)), ("P2", (0, 3, 1)),      # tet
              ("P3", (1, 3, 2)), ("P4", (2, 3, 0))],
          8: [("P1", (0, 1, 2, 3)), ("P2", (4, 5, 6, 7)),  # hex
              ("P3", (0, 1, 5, 4)), ("P4", (1, 2, 6, 5)),
              ("P5", (2, 3, 7, 6)), ("P6", (3, 0, 4, 7))]}


# --------------------------------------------------------------- input
def read_sg(path):
    """The SG mesh and its material cards.

    In:  path str -- an OpenSG solid SG yaml (either dialect)
    Out: dict {nodes (N,3), cells (E,k) 0-based, mat_id (E,) 1-based,
         materials [block], span (3,) = the cell bbox extent
         (Px, Py, h) -- the tiling PERIOD in x and y and the thickness,
         V float = summed ELEMENT volume (not the bbox: for a lattice
         or TPMS the two differ by the relative density), lo, hi (3,)}."""
    try:
        from opensg_solid.sg_mesh import load_sg_input
    except ImportError:
        sys.exit("opensg_solid is not importable -- activate the OpenSG"
                 " env, or set PYTHONPATH to its src/")
    d = load_sg_input(path)
    nd = np.asarray(d["nodes"], float)[:, :3]
    cl = np.asarray(d["cells"], int)
    mid = np.asarray(d["mat_id"], int)
    lo, hi = nd.min(axis=0), nd.max(axis=0)
    if (hi[2] - lo[2]) <= 1e-12 * max(hi[0] - lo[0], hi[1] - lo[1], 1.0):
        # a 2-D CROSS-SECTION SG (all nodes at z = 0): hand back the
        # deck-frame remap so `gate` and `plate` serve it directly --
        # the tiling period is the width period in BOTH in-plane
        # directions (the section is prismatic along the span) and the
        # cell volume is the section solid area x one span period.
        # The 3-D deck of such an SG is `solid2d`, never `solid`.
        s2 = read_sg2d(path)
        return {"nodes": s2["nodes"], "cells": s2["cells"],
                "mat_id": s2["mat_id"], "materials": s2["materials"],
                "span": np.array([s2["Pw"], s2["Pw"], s2["h"]]),
                "V": s2["A"] * s2["Pw"], "lo": s2["lo"], "hi": s2["hi"],
                "dim2": True}
    p = nd[cl]
    if cl.shape[1] == 4:
        # tetrahedron volume by scalar triple product:
        #   V = |(p1-p0) . ((p2-p0) x (p3-p0))| / 6
        # einsum("ei,ei->e") is a row-wise dot -> one scalar per element
        v = np.abs(np.einsum("ei,ei->e", p[:, 1] - p[:, 0],
                             np.cross(p[:, 2] - p[:, 0],
                                      p[:, 3] - p[:, 0]))) / 6.0
        vol = float(v.sum())
    else:
        # hex8: exact for the structured cells this path serves
        vol = float(np.prod(hi - lo))
    return {"nodes": nd, "cells": cl, "mat_id": mid,
            "materials": _materials_of(path), "span": hi - lo,
            "V": vol, "lo": lo, "hi": hi}


def _materials_of(path):
    """The `materials:` block, read WITHOUT parsing the mesh blocks.

    These yamls reach 500 MB and a full yaml.safe_load costs minutes and
    gigabytes.  This accumulates lines until the first TOP-LEVEL key
    (column 0, not a list item or comment) that names a big mesh block,
    then parses only that header -- about forty lines.

    In:  path str -- the SG yaml
    Out: list of dicts in file order (1-based mat_id = index + 1)."""
    import yaml
    stop = ("nodes", "cells", "elements", "mat_id", "sets", "sections",
            "elementOrientations")
    head = []
    for ln in open(path):
        at_col0 = ln[:1] not in (" ", "-", "\t", "\n", "#")
        if at_col0 and ln.split(":")[0] in stop:
            break
        head.append(ln)
    m = yaml.safe_load("".join(head))["materials"]
    return m if isinstance(m, list) else [m[k] for k in sorted(m)]


def iso_of(blk):
    """Is this material block one ISOTROPIC solid?  If so its (E, nu).

    In:  blk dict -- a `materials:` entry
    Out: (E, nu) tuple | None."""
    el = blk.get("elastic", blk)
    if "E" not in el:
        return None

    def trip(v, n=3):
        v = v if isinstance(v, (list, tuple)) else [v] * n
        return [float(x) for x in v]

    E, nu = trip(el["E"]), trip(el["nu"])
    G = trip(el["G"]) if el.get("G") is not None \
        else [E[0] / (2.0 * (1.0 + nu[0]))] * 3
    flat = all(max(t) - min(t) <= 1e-9 * max(abs(t[0]), 1e-30)
               for t in (E, nu, G))
    if not flat:
        return None
    if abs(G[0] - E[0] / (2.0 * (1.0 + nu[0]))) > 1e-6 * G[0]:
        return None
    return E[0], nu[0]


def read_abdg(path):
    """The 8x8 shear-refined plate law of an OpenSG .out.

    In:  path str -- the .out written by the homogenization
    Out: (ABD (6,6) symmetrized, G (2,2) symmetrized)."""
    rows = []
    for ln in open(path):
        v = ln.split()
        if len(v) == 8 and re.match(r"^[-+]?\d", v[0]):
            try:
                rows.append([float(x) for x in v])
            except ValueError:
                pass
        if len(rows) == 8:
            break
    K = np.array(rows)
    if K.shape != (8, 8):
        sys.exit("%s is not an 8x8 shear-refined plate law" % path)
    return 0.5 * (K[:6, :6] + K[:6, :6].T), 0.5 * (K[6:, 6:] + K[6:, 6:].T)


# ---------------------------------------------------------------- gate
def gate(sg, law_path=None):
    """Report the cell measures that the deck actually depends on.

    The reference surface is fixed by the SG YAML's geometry -- where
    z = 0 falls in the mesh -- and the deck preserves it, so there is
    nothing here to verify.  In particular the B block is NOT checked:
    the .out carries the correct ABDG whatever the reference surface is,
    and `*Shell General Section` transports B verbatim.

    In:  sg dict; law_path str | None -- the homogenized .out
    Out: dict {rel_density, and the law diagonal if law_path given}."""
    lo, hi, sp = sg["lo"], sg["hi"], sg["span"]
    r = {"rel_density": float(sg["V"] / np.prod(sp))}
    print("SG bbox        x[%.6f %.6f]  y[%.6f %.6f]  z[%.6f %.6f]"
          % (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]))
    print("cell           Px %.6g  Py %.6g  h %.6g" % tuple(sp))
    print("solid volume   %.6e   relative density %.5f"
          % (sg["V"], r["rel_density"]))
    if law_path:
        ABD, G = read_abdg(law_path)
        r.update({"A11": ABD[0, 0], "D11": ABD[3, 3],
                  "G11": G[0, 0], "G22": G[1, 1]})
        print("law %-24s A11 %.5e  D11 %.5e  G11 %.5e"
              % (os.path.basename(law_path), ABD[0, 0], ABD[3, 3],
                 G[0, 0]))
    return r


# --------------------------------------------------------------- tiling
def tile(sg, nx, ny, tol=1e-6):
    """Tile the cell nx x ny, weld the interfaces, centre IN-PLANE only.

    z is preserved verbatim so the deck's z = 0 stays the law's
    reference surface -- see the module docstring.

    In:  sg dict; nx, ny int; tol float -- weld tolerance [length]
    Out: (nodes (N,3), cells (E,k) 0-based, mats (E,))."""
    nd, cl, mid = sg["nodes"], sg["cells"], sg["mat_id"]
    Px, Py = sg["span"][0], sg["span"][1]
    org = nd.copy()
    org[:, 0] -= sg["lo"][0]                 # cell corner to x = 0
    org[:, 1] -= sg["lo"][1]                 # z untouched
    nt = nx * ny
    offs = np.array([[i * Px, j * Py, 0.0]
                     for i in range(nx) for j in range(ny)])
    nodes = (org[None, :, :] + offs[:, None, :]).reshape(-1, 3)
    cells = (cl[None, :, :] + (np.arange(nt) * len(nd))[:, None, None]
             ).reshape(-1, cl.shape[1])
    mats = np.tile(mid, nt)
    key = np.round(nodes / tol).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True,
                              return_inverse=True)
    nodes, cells = nodes[first], inv[cells]
    nodes[:, 0] -= 0.5 * nx * Px             # centre the plate on 0
    nodes[:, 1] -= 0.5 * ny * Py
    print("tile: %d x %d cells -> %s nodes (welded from %s), %s elements"
          % (nx, ny, "{:,}".format(len(nodes)),
             "{:,}".format(nt * len(nd)), "{:,}".format(len(cells))))
    return nodes, cells, mats


def read_sg2d(path):
    """A 2-D CROSS-SECTION SG (width x thickness, prismatic along the
    span), remapped to the deck frame: SG x1 -> y (width), SG x2 -> z
    (thickness, VERBATIM -- z = 0 stays the law's reference surface),
    x = the span the section will be extruded along.

    In:  path str -- the 2-D SG yaml (quad4 cells)
    Out: dict {nodes (N,3) deck frame, cells (E,4) 0-based CCW,
         mat_id (E,) 1-based, materials [block], Pw float = the width
         period, h float = the thickness, A float = section SOLID area,
         lo, hi (3,)}."""
    try:
        from opensg_solid.sg_mesh import load_sg_input
    except ImportError:
        sys.exit("opensg_solid is not importable -- activate the OpenSG"
                 " env, or set PYTHONPATH to its src/")
    d = load_sg_input(path)
    cl = np.asarray(d["cells"], int)
    if cl.shape[1] != 4:
        sys.exit("read_sg2d: expected a quad4 section, got %d-node"
                 " cells" % cl.shape[1])
    x12 = np.asarray(d["nodes"], float)[:, :2]
    nd = np.column_stack([np.zeros(len(x12)), x12[:, 0], x12[:, 1]])
    p = x12[cl]
    a2 = np.zeros(len(cl))
    for j in range(4):
        q, r = p[:, j], p[:, (j + 1) % 4]
        a2 += q[:, 0] * r[:, 1] - r[:, 0] * q[:, 1]
    lo, hi = nd.min(axis=0), nd.max(axis=0)
    return {"nodes": nd, "cells": cl,
            "mat_id": np.asarray(d["mat_id"], int),
            "materials": _materials_of(path),
            "Pw": float(hi[1] - lo[1]), "h": float(hi[2] - lo[2]),
            "A": float(np.abs(a2).sum() / 2.0), "lo": lo, "hi": hi}


def extrude_section(sg2, n_w, span_len, n_layers, tol=1e-6):
    """Tile the section n_w times across the width (welding the shared
    boundary nodes), then extrude n_layers hex8 slices along the span.
    Centred in (x, y); z verbatim.

    The section quads are CCW in the (y, z) plane viewed from +x, so
    [quad at station k, same quad at station k+1] is a positively
    oriented C3D8 by construction.

    In:  sg2 dict (read_sg2d); n_w int; span_len float; n_layers int;
         tol float -- weld tolerance [length]
    Out: (nodes (N,3), cells (E,8) 0-based, mats (E,))."""
    nd, cl, mid = sg2["nodes"], sg2["cells"], sg2["mat_id"]
    Pw = sg2["Pw"]
    org = nd.copy()
    org[:, 1] -= sg2["lo"][1]                # width corner to y = 0
    offs = np.array([[0.0, i * Pw, 0.0] for i in range(n_w)])
    sec = (org[None, :, :] + offs[:, None, :]).reshape(-1, 3)
    scl = (cl[None, :, :] + (np.arange(n_w) * len(nd))[:, None, None]
           ).reshape(-1, 4)
    smat = np.tile(mid, n_w)
    key = np.round(sec / tol).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True,
                              return_inverse=True)
    sec, scl = sec[first], inv[scl]
    ns = len(sec)
    xs = np.linspace(0.0, span_len, n_layers + 1)
    nodes = np.vstack([sec + np.array([x, 0.0, 0.0]) for x in xs])
    lay = (scl[None, :, :]
           + (np.arange(n_layers) * ns)[:, None, None])
    cells = np.concatenate([lay, lay + ns], axis=2).reshape(-1, 8)
    mats = np.tile(smat, n_layers)
    nodes[:, 0] -= 0.5 * span_len
    nodes[:, 1] -= 0.5 * n_w * Pw
    print("extrude: %d-wide x %d layers -> %s nodes, %s hex8"
          % (n_w, n_layers, "{:,}".format(len(nodes)),
             "{:,}".format(len(cells))))
    return nodes, cells, mats


# Abaqus C3D20 midside order: bottom-face edges (9-12), top-face edges
# (13-16), then the verticals (17-20)
_HEX_EDGES = np.array([(0, 1), (1, 2), (2, 3), (3, 0),
                       (4, 5), (5, 6), (6, 7), (7, 4),
                       (0, 4), (1, 5), (2, 6), (3, 7)])


def to_c3d20(nodes, cells):
    """p-refine hex8 -> C3D20: one shared midside node per unique edge,
    APPENDED after the corners so corner ids and element ids are
    unchanged.

    In:  nodes (N,3); cells (E,8) 0-based
    Out: (nodes (N+M,3), cells (E,20))."""
    E = len(cells)
    pair = np.sort(cells[:, _HEX_EDGES], axis=2).reshape(-1, 2)
    key = pair[:, 0].astype(np.int64) * len(nodes) + pair[:, 1]
    _, first, inv = np.unique(key, return_index=True,
                              return_inverse=True)
    mid = 0.5 * (nodes[pair[first, 0]] + nodes[pair[first, 1]])
    out = np.hstack([cells, len(nodes) + inv.reshape(E, 12)])
    print("p-refine: %s midside nodes (unique edges)"
          % "{:,}".format(len(mid)))
    return np.vstack([nodes, mid]), out


def to_c3d10(nodes, cells):
    """p-refine tet4 -> tet10 in place: one shared midside node per
    unique edge, APPENDED after the corners so corner ids and ELEMENT
    ids are unchanged (element k stays the same parent tet).

    In:  nodes (N,3); cells (E,4) 0-based
    Out: (nodes (N+M,3), cells (E,10))."""
    E = len(cells)
    edge = np.array([[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3]])
    pair = np.sort(cells[:, edge], axis=2).reshape(-1, 2)
    key = pair[:, 0].astype(np.int64) * len(nodes) + pair[:, 1]
    _, first, inv = np.unique(key, return_index=True, return_inverse=True)
    mid = 0.5 * (nodes[pair[first, 0]] + nodes[pair[first, 1]])
    out = np.hstack([cells, len(nodes) + inv.reshape(E, 6)])
    print("p-refine: %s midside nodes (unique edges)"
          % "{:,}".format(len(mid)))
    return np.vstack([nodes, mid]), out


# ------------------------------------------------------------- writers
def _nset_lines(f, name, ids):
    """Write a *Nset, 16 ids per line.  In: f file; name str; ids array.
    Out: None."""
    f.write("*Nset, nset=%s\n" % name)
    for i in range(0, len(ids), 16):
        f.write(", ".join(str(v) for v in ids[i:i + 16]) + "\n")


def write_solid(sg, nodes, cells, mats, path, bc, load, q, grav, etype):
    """The tiled 3-D Abaqus deck.

    In:  sg dict; nodes (N,3); cells (E,k) 0-based; mats (E,); path str;
         bc 'ss1'|'clamped'; load 'gravity'|'pressure'; q float; grav
         float; etype str
    Out: dict {inp, n_nodes, n_elems, edgex, edgey, faces}."""
    E = cells.shape[0]
    k = 8 if cells.shape[1] in (8, 20) else 4    # corner count (C3D20
    lo, hi = nodes.min(axis=0), nodes.max(axis=0)  # shares hex corners)
    tol = 1e-6 * float(max(hi - lo))
    edx = np.nonzero((np.abs(nodes[:, 0] - lo[0]) < tol)
                     | (np.abs(nodes[:, 0] - hi[0]) < tol))[0] + 1
    edy = np.nonzero((np.abs(nodes[:, 1] - lo[1]) < tol)
                     | (np.abs(nodes[:, 1] - hi[1]) < tol))[0] + 1
    top = []
    if load == "pressure":
        on = np.abs(nodes[:, 2] - hi[2]) < tol
        for lab, loc in _FACES[k]:
            m = on[cells[:, list(loc)]].all(axis=1)
            top += [(e + 1, lab) for e in np.nonzero(m)[0]]
        if not top:
            sys.exit("no element face lies on z = zmax")
    print("solid: bbox x[%.4f %.4f] y[%.4f %.4f] z[%.4f %.4f]"
          % (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]))
    print("solid: %s nodes on the x faces, %s on the y faces%s"
          % ("{:,}".format(len(edx)), "{:,}".format(len(edy)),
             ", %s loaded faces" % "{:,}".format(len(top)) if top else ""))

    with open(path, "w", buffering=1 << 22) as f:
        f.write("*Heading\n** 3-D SG tiled plate, %s, bc=%s load=%s\n"
                "** frame 1=span 2=width 3=thickness; plate centred on"
                " the origin; z = 0 is the SG reference surface\n"
                % (etype, bc, load))
        f.write("*Preprint, echo=NO, model=NO, history=NO, contact=NO\n")
        f.write("*Node\n")
        np.savetxt(f, np.column_stack([np.arange(1, len(nodes) + 1),
                                       nodes]),
                   fmt=["%d", "%.8f", "%.8f", "%.8f"], delimiter=", ")
        f.write("*Element, type=%s\n" % etype)
        rows = np.column_stack([np.arange(1, E + 1), cells + 1])
        if rows.shape[1] <= 16:
            np.savetxt(f, rows, fmt="%d", delimiter=", ")
        else:
            # an Abaqus data line holds at most 16 tokens; C3D20 rows
            # (21) continue on a second line via the trailing comma
            for r in rows:
                f.write(", ".join(str(v) for v in r[:11]) + ",\n")
                f.write(", ".join(str(v) for v in r[11:]) + "\n")
        f.write("*Elset, elset=ALL, generate\n1, %d, 1\n" % E)
        for m in sorted(set(mats.tolist())):
            el = np.nonzero(mats == m)[0] + 1
            f.write("*Elset, elset=MAT%d\n" % m)
            for i in range(0, len(el), 16):
                f.write(", ".join(str(v) for v in el[i:i + 16]) + "\n")
            # a material block's `angle:` is the ply rotation about the
            # plate normal z (the unified fiber +a sign; OpenSG
            # `angle: a` == Abaqus `3, a`).  In THIS deck's frame the
            # local 3-axis of the default rectangular csys IS z, so the
            # card below rotates in the plate plane -- unlike the old
            # single-extruded deck whose global z was the SPAN, where a
            # `3, theta` card tilted fibres out of the sheet.  With an
            # *Orientation present Abaqus reports S in the MATERIAL
            # frame for those elements -- pair with material-frame
            # recovery output, never global.
            ang = float(sg["materials"][m - 1].get("angle", 0.0))
            if ang != 0.0:
                f.write("*Orientation, name=ORI%d\n"
                        "1., 0., 0., 0., 1., 0.\n3, %.6g\n" % (m, ang))
                f.write("*Solid Section, elset=MAT%d, orientation=ORI%d,"
                        " material=M%d\n,\n" % (m, m, m))
            else:
                f.write("*Solid Section, elset=MAT%d, material=M%d\n,\n"
                        % (m, m))
        for m in sorted(set(mats.tolist())):
            blk = sg["materials"][m - 1]
            f.write("*Material, name=M%d\n" % m)
            f.write("*Density\n%.8g,\n" % float(blk.get("density", 0.0)))
            iso = iso_of(blk)
            if iso:
                f.write("*Elastic\n%.8g, %.8g\n" % iso)
            else:
                el = blk.get("elastic", blk)
                e = ([float(v) for v in el["E"]]
                     + [float(v) for v in el["nu"]]
                     + [float(v) for v in el["G"]])
                f.write("*Elastic, type=ENGINEERING CONSTANTS\n"
                        "%g, %g, %g, %g, %g, %g, %g, %g,\n%g\n"
                        % (e[0], e[1], e[2], e[3], e[4], e[5],
                           e[6], e[7], e[8]))
        _nset_lines(f, "EDGEX", edx)
        _nset_lines(f, "EDGEY", edy)
        if bc == "ss1":
            # SS-1 (hard): w = 0 and the TANGENTIAL in-plane component
            # = 0 on each face; the face-normal component stays free, so
            # N_nn = M_nn = 0.  Applied to every node THROUGH the
            # thickness, because u_t = u0_t + x3*phi_t vanishing for all
            # x3 is what delivers both u0_t = 0 and phi_t = 0.  These
            # six alone remove all six rigid-body modes -- no pin.
            f.write("*Boundary\nEDGEX, 2, 2\nEDGEX, 3, 3\n"
                    "EDGEY, 1, 1\nEDGEY, 3, 3\n")
        else:
            f.write("*Boundary\nEDGEX, ENCASTRE\nEDGEY, ENCASTRE\n")
        f.write("*Step, name=STATIC, nlgeom=NO\n*Static\n")
        if load == "gravity":
            f.write("*Dload\nALL, GRAV, %g, 0., 0., -1.\n" % grav)
        else:
            f.write("*Dload\n")
            for e, lab in top:
                f.write("%d, %s, %.8g\n" % (e, lab, q))
        f.write("*Output, field\n*Node Output\nU\n")
        # CENTROIDAL stress, deliberately.  On a mesh this refined a
        # single elemental stress is the right quantity, and the OpenSG
        # side is matched to it by AVERAGING its 4 Gauss values per
        # element -- so the comparison is element-to-element rather
        # than trying to pair Gauss points across two codes.
        f.write("*Element Output, position=CENTROIDAL\nS\n")
        f.write("*End Step\n")
    print("solid: wrote %s" % path)
    return {"inp": path, "n_nodes": len(nodes), "n_elems": E,
            "edgex": len(edx), "edgey": len(edy), "faces": len(top)}


def _shell_grid(nex, ney, dx, dy, a, b, etype):
    """The nodal grid and connectivity of a structured shell mesh.

    S4R is 4-node; S8R is 8-node quadratic, whose midside nodes come
    from a (2nex+1) x (2ney+1) lattice with the element CENTRES dropped
    (S8R has no centre node).  Abaqus S8R order: corners 1-4 CCW, then
    5 = mid(1,2), 6 = mid(2,3), 7 = mid(3,4), 8 = mid(4,1).

    In:  nex, ney int; dx, dy float -- element size; a, b float --
         plate span/width (the grid is centred on the origin);
         etype 'S4R'|'S8R'
    Out: (nodes [(id, x, y)], elems [[id, n...]], edgex, edgey id
         lists)."""
    quad = etype.startswith("S8")
    s = 2 if quad else 1                      # lattice points per element
    NI, NJ = s * nex + 1, s * ney + 1
    hx, hy = dx / s, dy / s
    used = np.zeros((NI, NJ), bool)
    conn = []
    for j in range(ney):
        for i in range(nex):
            I, J = s * i, s * j
            if quad:
                c = [(I, J), (I + 2, J), (I + 2, J + 2), (I, J + 2),
                     (I + 1, J), (I + 2, J + 1), (I + 1, J + 2),
                     (I, J + 1)]
            else:
                c = [(I, J), (I + 1, J), (I + 1, J + 1), (I, J + 1)]
            for p in c:
                used[p] = True
            conn.append(c)
    nid = {}
    nodes = []
    for J in range(NJ):                       # row-major numbering
        for I in range(NI):
            if used[I, J]:
                nid[(I, J)] = len(nodes) + 1
                nodes.append((len(nodes) + 1, I * hx - 0.5 * a,
                              J * hy - 0.5 * b))
    elems = [[e + 1] + [nid[p] for p in c] for e, c in enumerate(conn)]
    edgex = sorted(nid[p] for p in nid if p[0] in (0, NI - 1))
    edgey = sorted(nid[p] for p in nid if p[1] in (0, NJ - 1))
    return nodes, elems, edgex, edgey


def write_plate(sg, law_path, nx, ny, subdiv, path, bc, load, q, grav,
                etype="S4R"):
    """The equivalent-plate deck on the same centred footprint.

    The shell nodes sit at z = 0, which the solid tiling preserved as
    the SG's own z = 0, so both decks share one reference surface.

    etype: 'S4R' (linear, 1 element per cell -- the exact-pairing rule)
    or 'S8R' (quadratic THICK shell, 6 dof/node).  S8R is the better
    choice for a thick, coarsely-tiled plate: at a/h = 5 with one
    element per cell the linear element is asked to bend across very
    few elements.  Do NOT substitute S8R5 or S9R5 -- those are 5-dof
    THIN shells that impose the Kirchhoff constraint, which would
    suppress exactly the transverse shear this deck exists to test.

    In:  sg dict; law_path str; nx, ny, subdiv int; path str; bc str;
         load str; q float; grav float; etype str
    Out: dict {inp, n_elems, n_nodes, areal}."""
    ABD, G = read_abdg(law_path)
    Px, Py = sg["span"][0], sg["span"][1]
    rho = float(sg["materials"][0].get("density", 0.0))
    # mass per unit area, from the SUMMED ELEMENT volume -- the bbox
    # would over-load a lattice/TPMS by 1/relative_density
    areal = rho * sg["V"] / (Px * Py)
    a, b = nx * Px, ny * Py
    nex, ney = nx * subdiv, ny * subdiv
    dx, dy = Px / subdiv, Py / subdiv
    nodes, elems, edx, edy = _shell_grid(nex, ney, dx, dy, a, b, etype)
    e = len(elems)

    print("plate: %s, %d elements, %d nodes; section DENSITY (mass per"
          " unit area) = %.6f" % (etype, e, len(nodes), areal))
    D = ["*Heading",
         "** equivalent RM plate of %s" % os.path.basename(law_path),
         "** %d x %d cells (%d x %d %s), centred on the origin,"
         " reference surface z = 0" % (nx, ny, nex, ney, etype),
         "** bc=%s load=%s, LINEAR" % (bc, load),
         "*Preprint, echo=NO, model=NO, history=NO, contact=NO",
         "*Node"]
    D += ["%d, %.8f, %.8f, 0." % n for n in nodes]
    D.append("*Element, type=%s" % etype)             # CCW -> +z normal
    D += [", ".join(str(v) for v in row) for row in elems]
    D += ["*Elset, elset=ALL, generate", "1, %d, 1" % e]
    for nm, ids in (("EDGEX", edx), ("EDGEY", edy)):
        D.append("*Nset, nset=%s" % nm)
        for k in range(0, len(ids), 16):
            D.append(", ".join(str(v) for v in ids[k:k + 16]))
    # Abaqus general-shell order: upper triangle read DOWN the columns,
    # with the 6x6 ordered [N11 N22 N12 M11 M22 M12] -- already OpenSG's
    # ordering, so K[:6,:6] transfers directly.  B is carried here.
    vals = [ABD[r, c] for c in range(6) for r in range(c + 1)]
    D.append("*Shell General Section, elset=ALL, density=%.8e" % areal)
    for k in range(0, 21, 8):
        D.append(", ".join("%.7e" % v for v in vals[k:k + 8]))
    D.append("*Transverse Shear Stiffness")
    D.append("%.7e, %.7e, %.7e" % (G[0, 0], G[1, 1], G[0, 1]))
    if bc == "ss1":
        # the plate-DOF spelling of the solid's face constraints:
        # UR1 = phi_2, UR2 = phi_1
        D += ["*Boundary",
              "EDGEX, 2, 2", "EDGEX, 3, 3", "EDGEX, 4, 4",
              "EDGEY, 1, 1", "EDGEY, 3, 3", "EDGEY, 5, 5"]
    else:
        D += ["*Boundary", "EDGEX, ENCASTRE", "EDGEY, ENCASTRE"]
    D += ["*Step, name=STATIC, nlgeom=NO", "*Static"]
    D += ["*Dload", "ALL, GRAV, %g, 0., 0., -1." % grav] \
        if load == "gravity" else ["*Dload", "ALL, P, %.8g" % (-abs(q))]
    # SE = section STRAINS (membrane + transverse shear) -- it does NOT
    # carry the curvatures.  Those are SK, a separate output, and they
    # are what the shear-refined (V2) recovery needs most.  Requesting
    # SE without SK silently loses kappa.
    D += ["*Output, field", "*Node Output", "U, UR, RF, RM",
          "*Element Output, directions=YES", "SF, SM, SE, SK",
          "*End Step", ""]
    open(path, "w").write("\n".join(D))
    print("plate: wrote %s  (%d %s, %g x %g, total weight %.6f N)"
          % (path, e, etype, a, b, areal * a * b * grav))

    # ---- the TWO-LEVEL element map ----------------------------------
    # With subdiv > 1 each unit CELL is meshed by subdiv^2 elements.
    # The dehomogenization still consumes the CELL -- one cell is one
    # 3-D SG -- but the sub-elements multiply the strain samples inside
    # it: subdiv^2 elements x 4 integration points.  At subdiv = 2 that
    # is 16 samples per cell instead of 1, which is what makes an
    # in-cell strain DERIVATIVE possible without breaking the
    # one-cell-one-SG correspondence.
    # Level 1 = where the cell is (its centre).  Level 2 = which
    # elements belong to it.
    cmap = os.path.splitext(path)[0] + "_cellmap.csv"
    with open(cmap, "w") as g:
        g.write("# two-level element map: one row per unit CELL\n")
        g.write("# cell = the dehom station (one 3-D SG); sub_* are the"
                " %d plate elements meshing it\n" % (subdiv * subdiv))
        g.write("cell,Ic,Jc,xc,yc,"
                + ",".join("sub%d" % (t + 1)
                           for t in range(subdiv * subdiv)) + "\n")
        cid = 0
        for J in range(ny):
            for I in range(nx):
                cid += 1
                subs = [(sj * nex + si + 1)
                        for sj in range(J * subdiv, (J + 1) * subdiv)
                        for si in range(I * subdiv, (I + 1) * subdiv)]
                g.write("%d,%d,%d,%.8f,%.8f,%s\n"
                        % (cid, I, J,
                           (I + 0.5) * Px - 0.5 * a,
                           (J + 0.5) * Py - 0.5 * b,
                           ",".join(str(s) for s in subs)))
    print("plate: wrote %s  (%d cells x %d elements, %d Gauss samples"
          " per cell)" % (os.path.basename(cmap), nx * ny,
                          subdiv * subdiv, 4 * subdiv * subdiv))
    return {"inp": path, "n_elems": e, "n_nodes": len(nodes),
            "areal": areal, "cellmap": cmap}


# ----------------------------------------------------------------- CLI
def main(argv=None):
    """Parse the subcommand and dispatch.

    In:  argv list | None -- None means sys.argv[1:]
    Out: int exit status."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    # solid2d: the 3-D deck of a 2-D CROSS-SECTION SG (width x
    # thickness, prismatic) -- tiled across the width and EXTRUDED
    # along the span.  The HC-sandwich analog of `solid`.
    s2 = sub.add_parser("solid2d")
    s2.add_argument("--sg", required=True,
                    help="the 2-D cross-section SG yaml (quad4)")
    s2.add_argument("--nw", type=int, required=True,
                    help="cells across the width")
    s2.add_argument("--nspan", type=int, required=True,
                    help="cells along the span (span = nspan * Pw)")
    s2.add_argument("--layers-per-cell", type=int, default=8,
                    dest="lpc",
                    help="extrusion layers per span cell")
    s2.add_argument("--order", type=int, default=2, choices=[1, 2],
                    help="1 = C3D8I, 2 = C3D20R")
    s2.add_argument("--bc", default="ss1", choices=["ss1", "clamped"])
    s2.add_argument("--load", default="pressure",
                    choices=["gravity", "pressure"])
    s2.add_argument("--q", type=float, default=1.0)
    s2.add_argument("--grav", type=float, default=9.81)
    s2.add_argument("--out", default=".")

    for nm in ("gate", "solid", "plate"):
        s = sub.add_parser(nm)
        s.add_argument("--sg", required=True,
                       help="the OpenSG solid SG yaml")
        s.add_argument("--law", default=None, required=(nm == "plate"),
                       help="the homogenized 8x8 .out of that SG")
        if nm != "gate":
            s.add_argument("--nx", type=int, help="cells along 1")
            s.add_argument("--ny", type=int, help="cells along 2")
            s.add_argument("--a", type=float,
                           help="span in length units (rounded to cells)")
            s.add_argument("--b", type=float, help="width, likewise")
            s.add_argument("--bc", default="ss1",
                           choices=["ss1", "clamped"])
            s.add_argument("--load", default="gravity",
                           choices=["gravity", "pressure"])
            s.add_argument("--q", type=float, default=1.0,
                           help="pressure magnitude, --load pressure")
            s.add_argument("--grav", type=float, default=9.81)
            s.add_argument("--out", default=".")
        if nm == "solid":
            s.add_argument("--order", type=int, default=1, choices=[1, 2],
                           help="2 = C3D10 p-refinement (tet4 SG only)")
        if nm == "plate":
            s.add_argument("--subdiv", type=int, default=3,
                           help="elements per cell edge; subdiv = 3 is"
                                " the pipeline default -- 2 has no"
                                " pure second derivatives; 1 = exact"
                                " pairing")
            s.add_argument("--etype", default="S4R",
                           choices=["S4R", "S8R"],
                           help="S8R = quadratic THICK shell, better"
                                " for a coarsely tiled thick plate."
                                " Never S8R5/S9R5: 5-dof THIN shells"
                                " that kill the transverse shear this"
                                " deck tests")
    A = p.parse_args(argv)

    if A.cmd == "solid2d":
        sg2 = read_sg2d(A.sg)
        print("section: Pw %.6g  h %.6g  solid area %.6g  (%s quads)"
              % (sg2["Pw"], sg2["h"], sg2["A"],
                 "{:,}".format(len(sg2["cells"]))))
        span_len = A.nspan * sg2["Pw"]
        n_layers = A.nspan * A.lpc
        n_el = A.nw * len(sg2["cells"]) * n_layers
        need = n_el * (60 if A.order == 1 else 190)
        free = shutil.disk_usage(A.out).free
        print("solid2d: %s elements, deck ~%.2f GB, %.2f GB free"
              % ("{:,}".format(n_el), need / 2 ** 30, free / 2 ** 30))
        if need > 0.8 * free:
            sys.exit("not enough room -- point --out at a local disk"
                     " with no quota")
        nodes, cells, mats = extrude_section(sg2, A.nw, span_len,
                                             n_layers)
        etype = "C3D8I" if A.order == 1 else "C3D20R"
        if A.order == 2:
            nodes, cells = to_c3d20(nodes, cells)
        os.makedirs(A.out, exist_ok=True)
        stem = os.path.splitext(os.path.basename(A.sg))[0]
        path = os.path.join(A.out, "%s_3d_%dx%d_%s.inp"
                            % (stem, A.nw, A.nspan, etype.lower()))
        write_solid(sg2, nodes, cells, mats, path, A.bc, A.load, A.q,
                    A.grav, etype)
        return 0

    sg = read_sg(A.sg)
    gate(sg, A.law)
    if A.cmd == "gate":
        return 0

    nx = A.nx or max(1, int(round(A.a / sg["span"][0])))
    ny = A.ny or max(1, int(round(A.b / sg["span"][1])))
    os.makedirs(A.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(A.sg))[0]

    if A.cmd == "plate":
        path = os.path.join(A.out, "%s_plate_%dx%d_%s.inp"
                            % (stem, nx, ny, A.etype.lower()))
        write_plate(sg, A.law, nx, ny, A.subdiv, path, A.bc, A.load,
                    A.q, A.grav, A.etype)
        return 0

    if sg.get("dim2"):
        sys.exit("this SG is a 2-D cross-section -- build its 3-D deck"
                 " with `solid2d` (tile across the width + extrude the"
                 " span), not `solid`")
    k = sg["cells"].shape[1]
    if A.order == 2 and k != 4:
        sys.exit("--order 2 (C3D10) needs a tet4 SG, got %d-node" % k)
    etype = {1: {4: "C3D4", 8: "C3D8I"}[k], 2: "C3D10"}[A.order]
    n_proj = nx * ny * len(sg["cells"])
    n_nd = n_proj / 4.5 + (1.6 * n_proj if A.order == 2 else 0.0)
    need = n_proj * (55 if A.order == 1 else 85) + n_nd * 52
    free = shutil.disk_usage(A.out).free
    print("solid: ~%s elements, deck ~%.2f GB, %.2f GB free in %s"
          % ("{:,}".format(n_proj), need / 2 ** 30, free / 2 ** 30,
             A.out))
    if need > 0.8 * free:
        sys.exit("not enough room -- point --out at a local disk with"
                 " no quota")

    nodes, cells, mats = tile(sg, nx, ny)
    if A.order == 2:
        nodes, cells = to_c3d10(nodes, cells)
    path = os.path.join(A.out, "%s_3d_%dx%d_%s.inp"
                        % (stem, nx, ny, etype.lower()))
    write_solid(sg, nodes, cells, mats, path, A.bc, A.load, A.q,
                A.grav, etype)
    return 0


if __name__ == "__main__":
    sys.exit(main())
