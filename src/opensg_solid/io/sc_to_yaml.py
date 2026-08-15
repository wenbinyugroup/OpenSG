"""sc_to_yaml.py -- SwiftComp `.sc` -> OpenSG solid SG yaml (+ gmsh `.msh`)
for ANY structure gene, so the SG examples never depend on a private
parser.  This module is the CANONICAL converter; the historical path
opensg_solid.rm_plate_1D.helper.sc_to_yaml re-exports it unchanged.

The .sc anatomy this reads (SwiftComp MSG-native input):

    <model header lines>            1-3 short lines (submodel flags etc.)
    dim  n_nodes  n_elems  n_mats  ...   <- the META line (>= 5 integers)
    <n_nodes>  id  x [y [z]]
    <n_elems>  id  mat_id  conn... (zero-padded; the zeros are NOT nodes)
    per material:
        mat_id  mat_type  n
        <aux line: temperature  density>
        mat_type 0: E  nu                     (isotropic)
        mat_type 1: E1 E2 E3 / G12 G13 G23 / v12 v13 v23  (orthotropic)
        mat_type 2: 21 constants, upper-triangular 6x6 C over SIX lines
                    (these are typically PRE-ROTATED ply stiffnesses)
    <trailing scale line>           (volume/thickness normalization)

WHAT THE EMITTED YAML CARRIES (the format opensg_solid reads today):

    n_model / refined   the analysis header, with the same trailing
                        comments the shipped example yamls use.  n_model
                        is guessed from the SG dimension -- a 3-D SG is a
                        solid (3), a 1-D/2-D SG a plate (2) -- and the
                        `n_model=` argument overrides the guess.
    omega               OPTIONAL, see the rule below.
    nodes/cells/mat_id/materials
                        the mesh, cells 0-based, mat_id 1-based.
                        Materials come out in the form build_material_C
                        accepts: `type` 0 (E/nu), 1 (nine `engineering`
                        constants) or 2 (stored 6x6 `C`), plus `angle`
                        (deg) and `density` when the .sc carries them.
                        The .sc `aux` line is NOT carried over: its only
                        payload is the density, which is emitted as the
                        `density:` key.

    NO `dim:`   -- load_sg_input infers the SG dimension from the mesh.
    NO `scale:` -- the SG measure omega is MEASURED from the mesh.

THE `omega:` RULE.  The solver measures omega itself (sg_assembly:
3-D SG -> node bounding-box VOLUME, i.e. the periodic unit cell; a
plate SG -> the in-plane span(s); see compute_homogenized_constants) and
the yaml header key `omega:` is consumed in ONE case only -- n_model 3,
where sg_homo rescales C_eff by omega_measured/omega_user.  The .sc
trailing line is therefore written out only when it can be both used and
trusted: n_model 3 AND a 3-D SG (the branch whose measure is exactly the
node bounding box) AND a positive finite value differing from that
bounding-box volume by more than 1e-6 relative.  In every other case the
trailing value is dropped -- it is not an SG measure the solver would
consume, and in the shipped 1-D/2-D files it is variously the in-plane
span (RHC_SW_2UC_45.sc: 35.248 = the measured omega already) or a
left-over material constant -- and convert() prints a one-line note when
the dropped value disagrees with the measurement.

# ----------------------------------------------------------------------------
# ALL VARIABLES USED IN THIS MODULE
# ----------------------------------------------------------------------------
# inputs
#   sc_path, out_base   the .sc input; basename of the emitted sidecars
#   n_model, refined    the yaml analysis header (n_model None = guess)
# working
#   lines, meta_idx     the whitespace-stripped .sc; index of the META line
#   dim, n_nodes, n_elems, n_mats   the META line integers
#   conn, mat_id        0-based element connectivity; 1-based material ids
#   aux                 the .sc per-material line [temperature, density]
#   meas                the bounding-box SG measure of the emitted n_model
# outputs
#   sc                  {dim, nodes, cells, mat_id, materials, scale}
#   <base>.yaml         header + nodes/cells/mat_id/materials
#   <base>.msh          gmsh v2.2, material id as physical/elementary tag
# ----------------------------------------------------------------------------
"""
import os

import numpy as np
import yaml

_HDR_MSG = ("msg: solid      # the ENGINE this SG belongs to (opensg_solid); "
            "`opensg <yaml>` dispatches on it\n")
_HDR_N_MODEL = ("n_model: %d      # 1 = beam, 2 = plate, 3 = solid -- the "
                "macro model this SG homogenizes to\n")
_HDR_REFINED = ("refined: %d      # 0 = classical (plate ABD / beam EB); "
                "1 = shear-refined (plate ABDG / beam Timoshenko)\n")
_HDR_OMEGA = ("omega: %.10g      # SG measure from the .sc trailing line "
              "(3-D solid; wins over the measured unit-cell volume)\n")

_OMEGA_RTOL = 1e-6


def read_sc(path):
    """Parse a .sc file.

    In:  path str -- the SwiftComp .sc
    Out: dict {dim int, nodes (n, 3) float, cells (list of 0-based
         connectivity lists), mat_id (n_elems,) int (1-based, as in the
         file), materials {id: {type, aux, E/nu | engineering | C}},
         scale float (the trailing normalization line, 1.0 when absent)}.
    """
    with open(path) as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    meta_idx = -1
    for i, ln in enumerate(lines):
        parts = ln.split()
        if len(parts) >= 5:
            meta_idx = i
            dim, n_nodes, n_elems, n_mats = (int(parts[0]), int(parts[1]),
                                             int(parts[2]), int(parts[3]))
            break
    if meta_idx == -1:
        raise ValueError("no META line (>=5 integers) in %s" % path)

    nodes = np.zeros((n_nodes, 3))
    for ln in lines[meta_idx + 1: meta_idx + 1 + n_nodes]:
        p = ln.split()
        i = int(p[0]) - 1
        for d in range(dim):
            nodes[i, d] = float(p[1 + d])

    cells, mat_id = [], []
    e0 = meta_idx + 1 + n_nodes
    for ln in lines[e0: e0 + n_elems]:
        p = ln.split()
        mat_id.append(int(p[1]))
        conn = [int(n) - 1 for n in p[2:] if n != "0"]
        if dim == 3 and len(conn) == 10:
            # the tet10 slot convention of fe_jax/sc_to_msh: 4 corners,
            # slot 5 = 0, then the 6 midsides in slots 7-12
            raw = p[2:]
            conn = [int(n) - 1 for n in (raw[:4] + raw[6:12])]
        elif dim == 3 and len(conn) not in (4, 8):
            # e.g. SW_2UC_45.sc: every record has 9 nonzero slots -- not a
            # tet4/hex8/tet10 under any known SwiftComp slicing.  Refuse
            # loudly instead of emitting phantom node 0 / -1 entries.
            raise ValueError(
                "%s: 3-D element %s has %d nonzero connectivity slots -- "
                "not tet4/hex8/tet10; confirm this file's SwiftComp slot "
                "convention before converting" % (path, p[0], len(conn)))
        cells.append(conn)

    materials, k = {}, e0 + n_elems
    for _ in range(n_mats):
        head = lines[k].split()
        mid, mtype = int(head[0]), int(head[1])
        aux = lines[k + 1].split()          # temperature  density
        m = {"type": mtype, "aux": [float(v) for v in aux]}
        if mtype == 0:
            p = lines[k + 2].replace("E+", "e+").replace("E-", "e-").split()
            m["E"], m["nu"] = float(p[0]), float(p[1])
            k += 3
        elif mtype == 1:
            vals = []
            j = k + 2
            while len(vals) < 9:
                vals += [float(v) for v in lines[j].split()]
                j += 1
            m["engineering"] = vals            # E1 E2 E3 G12 G13 G23 v12 v13 v23
            k = j
        else:                                   # type 2: upper-tri 6x6 C
            C = np.zeros((6, 6))
            j = k + 2
            for r in range(6):
                row = [float(v) for v in lines[j].split()]
                C[r, r:r + len(row)] = row
                C[r:r + len(row), r] = row
                j += 1
            m["C"] = C.tolist()
            k = j
        materials[mid] = m
    scale = float(lines[k].split()[0]) if k < len(lines) else 1.0

    return {"dim": dim, "nodes": nodes, "cells": cells,
            "mat_id": np.array(mat_id, int), "materials": materials,
            "scale": scale}


_GMSH_1D = {2: 1, 3: 8, 4: 26, 5: 27}
_GMSH_2D = {3: 2, 4: 3}
_GMSH_3D = {4: 4, 10: 11}


def write_msh(sc, path):
    """gmsh v2.2 with the material id as both physical/elementary tags
    (what the SG engine reads back as cell_domain_ids).

    In:  sc dict (read_sc); path str -- the .msh to write
    Out: None (writes path)."""
    dim, nodes = sc["dim"], sc["nodes"]
    with open(path, "w") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n%d\n"
                % len(nodes))
        for i, x in enumerate(nodes):
            f.write("%d %.8f %.8f %.8f\n" % (i + 1, x[0], x[1], x[2]))
        f.write("$EndNodes\n$Elements\n%d\n" % len(sc["cells"]))
        tmap = _GMSH_1D if dim == 1 else _GMSH_2D if dim == 2 else _GMSH_3D
        for e, (conn, mid) in enumerate(zip(sc["cells"], sc["mat_id"])):
            et = tmap.get(len(conn))
            if et is None:
                raise ValueError("element %d: %d nodes unsupported for %dD"
                                 % (e + 1, len(conn), dim))
            f.write("%d %d 2 %d %d %s\n"
                    % (e + 1, et, mid, mid,
                       " ".join(str(n + 1) for n in conn)))
        f.write("$EndElements\n")


def default_n_model(dim):
    """The macro model a bare .sc implies: a 3-D SG is a solid, a 1-D/2-D
    SG a plate.

    In:  dim int -- the SG spatial dimension
    Out: int n_model (2 plate | 3 solid)."""
    return 3 if int(dim) == 3 else 2


def bbox_measure(nodes, dim):
    """The node BOUNDING-BOX measure of an SG: the product of the
    coordinate spans over its own `dim` leading columns (a 3-D SG's cell
    volume -- what sg_assembly._solid_omega measures).

    In:  nodes (n, 3) float; dim int SG dimension
    Out: float measure."""
    nd = np.asarray(nodes, float)
    span = nd.max(axis=0) - nd.min(axis=0)
    return float(np.prod(span[:int(dim)]))


def header_omega(sc, n_model):
    """The `omega:` header value for the emitted yaml, or None.

    The .sc trailing normalization is kept ONLY where the solver both
    consumes it and measures the same quantity: n_model 3 on a 3-D SG,
    whose measure is exactly the node bounding-box volume, and only when
    it differs from that volume by more than 1e-6 relative.  See the
    module docstring for why the other cases are dropped.

    In:  sc dict (read_sc); n_model int
    Out: (omega float | None, note str | None) -- note is a one-line
         explanation when a disagreeing trailing value is DROPPED."""
    scale = float(sc.get("scale", 1.0))
    dim = int(sc["dim"])
    meas = bbox_measure(sc["nodes"], dim)
    ok = np.isfinite(scale) and scale > 0.0
    same = abs(scale - meas) <= _OMEGA_RTOL * max(abs(meas), abs(scale))
    if same or not ok:
        return None, None
    if int(n_model) == 3 and dim == 3:
        return scale, None
    return None, ("the .sc trailing value %.6g differs from the measured SG "
                  "measure %.6g -- dropped (omega: is read only by the 3-D "
                  "solid model on a 3-D SG)" % (scale, meas))


def yaml_materials(materials):
    """The `materials:` block in the form build_material_C accepts.

    Per material: `type` (0 isotropic E/nu, 1 the nine `engineering`
    constants, 2 the stored 6x6 `C`), then the optional `angle` (deg,
    when nonzero) and `density` (the .sc aux line's second entry, when
    positive).  The .sc `aux` pair itself is not carried over.

    In:  materials {id: {type, aux, ...}} (read_sc)
    Out: {int id: {ordered plain-python material dict}}."""
    out = {}
    for mid in sorted(materials):
        m = materials[mid]
        mtype = int(m["type"])
        d = {"type": mtype}
        if mtype == 0:
            d["E"] = float(m["E"])
            d["nu"] = float(m["nu"])
        elif mtype == 1:
            d["engineering"] = [float(v) for v in m["engineering"]]
        else:
            d["C"] = [[float(v) for v in row]
                      for row in np.asarray(m["C"], float)]
        angle = float(m.get("angle", 0.0))
        if angle != 0.0:
            d["angle"] = angle
        aux = list(m.get("aux", ()))
        rho = float(m.get("density", aux[1] if len(aux) > 1 else 0.0))
        if rho > 0.0:
            d["density"] = rho
        out[int(mid)] = d
    return out


def _num(v):
    """A yaml scalar for a number: round-tripping repr, ints stay ints."""
    return str(int(v)) if isinstance(v, (int, np.integer)) else repr(float(v))


def dump_materials(f, mats):
    """Write the `materials:` block in the shipped examples' spelling --
    block-style per material, the nine `engineering` constants on one
    flow line, the 6x6 `C` as six flow rows (yaml.dump would collapse a
    scalar-only material to `{type: 0, E: ..., nu: ...}`).

    In:  f open text file; mats {id: material dict} (yaml_materials)
    Out: None (writes to f)."""
    f.write("materials:\n")
    for mid in sorted(mats):
        f.write("  %d:\n" % int(mid))
        for k, v in mats[mid].items():
            if k == "C":
                f.write("    C:\n")
                for row in v:
                    f.write("    - [%s]\n" % ", ".join(_num(x) for x in row))
            elif isinstance(v, (list, tuple)):
                f.write("    %s: [%s]\n" % (k, ", ".join(_num(x) for x in v)))
            else:
                f.write("    %s: %s\n" % (k, _num(v)))


def write_yaml_file(sc, path, n_model=None, refined=0, omega=None):
    """Write the SG yaml: the analysis header, then the mesh blocks.

    In:  sc dict (read_sc); path str -- the .yaml to write;
         n_model 1 beam | 2 plate | 3 solid (None -> default_n_model);
         refined 0 classical | 1 shear-refined;
         omega float SG-measure override or None (see header_omega)
    Out: None (writes path)."""
    try:                                     # libyaml C dumper: the pure-python
        from yaml import CSafeDumper as _YD  # emitter costs ~4x on a 3-D SG
    except ImportError:
        from yaml import SafeDumper as _YD
    n_model = default_n_model(sc["dim"]) if n_model is None else int(n_model)
    mesh = {"nodes": [[float(v) for v in row] for row in sc["nodes"]],
            "cells": [[int(n) for n in c] for c in sc["cells"]],
            "mat_id": [int(m) for m in sc["mat_id"]]}
    with open(path, "w") as f:
        f.write(_HDR_N_MODEL % n_model)
        f.write(_HDR_REFINED % int(refined))
        f.write(_HDR_MSG)
        if omega is not None:
            f.write(_HDR_OMEGA % float(omega))
        yaml.dump(mesh, f, Dumper=_YD, sort_keys=False,
                  default_flow_style=None)
        dump_materials(f, yaml_materials(sc["materials"]))


def convert(sc_path, out_base=None, n_model=None, refined=0):
    """Read a .sc, write <base>.yaml + <base>.msh, return the parsed dict.

    Up-to-date sidecars short-circuit the conversion: when <base>.yaml and
    <base>.msh are both newer than the .sc, the yaml is loaded back
    instead (through load_sg_input's npz cache, ~0.5 s) -- the .sc
    re-parse plus the yaml/msh rewrite of a 3-D SG costs ~40 s otherwise.
    An existing yaml therefore WINS over `n_model`/`refined`: the file may
    be hand-edited, and this converter never overwrites it.  Delete
    <base>.yaml to re-emit the header from the .sc.

    In:  sc_path str -- the .sc; out_base str basename for the sidecars
         (None -> the .sc stem); n_model 1 beam | 2 plate | 3 solid
         (None -> default_n_model, i.e. the SG dimension's default);
         refined 0 classical | 1 shear-refined -- both go into the
         emitted yaml header
    Out: dict {dim, nodes, cells, mat_id, materials, scale}."""
    if out_base is None:
        out_base = os.path.splitext(sc_path)[0]
    yml, msh = out_base + ".yaml", out_base + ".msh"
    if (os.path.exists(yml) and os.path.exists(msh)
            and os.path.getmtime(yml) >= os.path.getmtime(sc_path)
            and os.path.getmtime(msh) >= os.path.getmtime(sc_path)):
        from opensg_solid.sg_mesh import load_sg_input   # call-time: no cycle
        return load_sg_input(yml)
    sc = read_sc(sc_path)
    n_model = default_n_model(sc["dim"]) if n_model is None else int(n_model)
    omega, note = header_omega(sc, n_model)
    write_yaml_file(sc, yml, n_model=n_model, refined=refined, omega=omega)
    write_msh(sc, msh)
    print("sc_to_yaml: %dD SG, %d nodes, %d cells, %d materials, n_model %d"
          " -> %s.yaml/.msh"
          % (sc["dim"], len(sc["nodes"]), len(sc["cells"]),
             len(sc["materials"]), n_model, out_base))
    if note is not None:
        print("sc_to_yaml: %s" % note)
    return sc


SIGN3 = +1.0   # sign of the theta3 fiber rotation about the ply normal
               # (validated against the mh104 .sg.K off-diagonals)


def _sg_frame(t1, t3):
    """Material frame [e1(fiber), e2, e3(ply normal)] as a 9-list.

    theta1 sets the ply-plane in the x-y cross-section (e3 = normal,
    e2 = tangent, e1 = beam axis Z); theta3 then rotates the fiber about
    the ply normal e3, tilting e1 from Z toward the tangent e2."""
    c1, s1 = np.cos(np.deg2rad(t1)), np.sin(np.deg2rad(t1))
    c3, s3 = np.cos(np.deg2rad(SIGN3 * t3)), np.sin(np.deg2rad(SIGN3 * t3))
    e1 = [s3 * c1, s3 * s1, c3]
    e2 = [c3 * c1, c3 * s1, -s3]
    e3 = [-s1, c1, 0.0]
    return [float(v) for v in e1 + e2 + e3]


def sg_to_yaml(sg_path, out_path=None):
    """PreVABS/VABS `.sg` -> the 2-D solid mesh-dialect OpenSG yaml.

    The port of the VALIDATED OpenSG_io converter
    (scripts/convert_sg_to_yaml.py -- mh104: solid vs VABS to ~1e-5): the
    .sg PreVABS writes (e.g. from the pynumad `--xml` route) becomes the
    nodes/elements/sets/elementOrientations/materials yaml the solid
    engine reads.  Per-element theta1 (contour angle) and per-layup-group
    theta3 (fiber angle) compose into the elementOrientations frames --
    theta3 is NOT baked into the materials.  Material names come from the
    `<stem>.sg.mat` sidecar when PreVABS wrote one.

    The emitted yaml is the MESH dialect (node/element rows are
    space-separated strings, `materials` a LIST), so it is read back with
    io.sg_input.read_opensg_yaml -- NOT sg_mesh.load_sg_input, which takes
    the canonical dialect sc_to_yaml.convert writes.

    In:  sg_path str -- the VABS `.sg` (format_flag 1, curve/oblique off,
         orthotropic materials -- exactly what PreVABS 2.x writes);
         out_path str | None -- the yaml (None = `<stem>.yaml` beside it).
    Out: dict {yaml, n_nodes, n_elems, n_layups, n_mats, names}.
    """
    with open(sg_path) as f:
        data = [ln.rstrip("\n") for ln in f if ln.strip()]
    grp = int(data[0].split()[1])
    flags = [int(v) for v in data[2].split()]
    if len(flags) != 4 or flags[0] or flags[1]:
        raise ValueError(
            "%s: expected the PreVABS header (curve_flag oblique_flag "
            "trapeze_flag Vlasov_flag with the first two 0), got %r"
            % (sg_path, data[2]))
    nnode, nelem, nphases = [int(v) for v in data[3].split()]

    points = []                                    # (x2, x3)
    for i in range(nnode):
        d = data[4 + i].split()
        points.append((float(d[1]), float(d[2])))
    cells = []                                     # 0-based, zero-padding dropped
    for i in range(nelem):
        d = data[4 + nnode + i].split()
        cells.append([int(x) - 1 for x in d[1:] if int(x) != 0])
    p = 4 + nnode + 2 * nelem                      # layup group -> (mat, theta3)
    gmat = [int(data[p + i].split()[1]) - 1 for i in range(grp)]
    gth3 = [float(data[p + i].split()[2]) for i in range(grp)]
    p = 4 + nnode + nelem                          # element -> (group, theta1)
    sub, th1, th3 = [], [], []
    for e in range(nelem):
        _eid, g, t1 = data[p + e].split()
        sub.append(gmat[int(g) - 1])
        th1.append(float(t1))
        th3.append(gth3[int(g) - 1])
    p = 4 + nnode + 2 * nelem + grp                # materials: 5 rows/phase
    mat_par, density = [], []
    for m in range(nphases):
        head = data[p + 5 * m].split()
        if int(head[1]) != 1:
            raise ValueError(
                "%s: material %s is orth %s -- this converter handles the "
                "orthotropic (orth 1) blocks PreVABS writes" %
                (sg_path, head[0], head[1]))
        rows = [data[p + 5 * m + k].split() for k in (1, 2, 3)]
        density.append(float(data[p + 5 * m + 4].split()[0]))
        mat_par.append(np.array(rows, float).flatten().tolist())

    names = {}
    side = sg_path + ".mat"
    if os.path.exists(side):
        for ln in open(side):
            t = ln.split()
            if len(t) >= 2 and t[0].isdigit():
                names[int(t[0])] = t[1]
    name_of = lambda i: names.get(i + 1, "Material_%d" % (i + 1))  # noqa: E731

    class _Flow(list):
        pass

    class _D(yaml.SafeDumper):
        pass

    _D.add_representer(_Flow, lambda d, v: d.represent_sequence(
        "tag:yaml.org,2002:seq", v, flow_style=True))

    seg = {"nodes": [_Flow(["%.6f %.6f %.6f" % (x2, x3, 0.0)])
                     for (x2, x3) in points],
           "elements": [_Flow([" ".join(str(n + 1) for n in c)])
                        for c in cells],
           "sets": {"element": [
               {"name": name_of(mi),
                "labels": _Flow([e + 1 for e, s in enumerate(sub)
                                 if s == mi])}
               for mi in sorted(set(sub))]},
           "elementOrientations": [_Flow(_sg_frame(a, b))
                                   for a, b in zip(th1, th3)],
           "materials": [{"name": name_of(m),
                          "E": _Flow(mat_par[m][0:3]),
                          "G": _Flow(mat_par[m][3:6]),
                          "nu": _Flow(mat_par[m][6:9]),
                          "rho": float(density[m])}
                         for m in range(nphases)]}
    out_path = out_path or os.path.splitext(sg_path)[0] + ".yaml"
    with open(out_path, "w") as f:
        yaml.dump(seg, f, Dumper=_D, sort_keys=False,
                  default_flow_style=False)
    print("sg_to_yaml: %d nodes, %d elems, %d layup groups, %d materials"
          " -> %s" % (nnode, nelem, grp, nphases, out_path))
    return dict(yaml=out_path, n_nodes=nnode, n_elems=nelem, n_layups=grp,
                n_mats=nphases, names=[name_of(m) for m in range(nphases)])
