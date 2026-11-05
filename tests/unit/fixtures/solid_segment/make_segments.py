"""Small solid beam segments for tests/unit/test_solid_segment.py.

Run it once to rewrite the yamls of this folder:

    python tests/unit/fixtures/solid_segment/make_segments.py

Every segment is a structured box of hex8 elements, x in [x0, x0 + W],
y in [y0, y0 + H], z in [0, L], mesh z = the beam axis.  A tapered
segment scales x and y about the origin, linearly from 1 at z = 0 to
`taper` at z = L.  tet4 segments split every hex into six tets on its
main diagonal, and the mixed segments split the hexes of the first half
of the x columns.  tet10 segments add a straight midside node on every
edge of the tet4 split.  The composite section is a -45 ply row at the bottom,
a +45 ply row at the top (frames rotated about x3) and an isotropic core.
The frame rows use the 2-D beam SG spelling: mesh components (x, y, z).
"""
import os
import struct

import numpy as np

assert struct.calcsize("P") == 8

HERE = os.path.dirname(os.path.abspath(__file__))

CARBON = dict(name="carbon", E=[140000.0, 10000.0, 10000.0],
              G=[5000.0, 5000.0, 10000.0 / 2.8], nu=[0.3, 0.3, 0.4],
              density=1.6e-9)
CORE = dict(name="core", E=[1000.0] * 3, G=[1000.0 / 2.6] * 3,
            nu=[0.3] * 3, density=1.0e-10)
STEEL = dict(name="steel", E=[210000.0] * 3, G=[210000.0 / 2.6] * 3,
             nu=[0.3] * 3, density=7.85e-9)

IDENTITY = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
_TETS = [(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6),
         (0, 4, 5, 6), (0, 5, 1, 6)]


def frame_row(t_deg):
    """The frame of a ply whose fiber is at t from x1 toward x2 (a
    rotation about x3), in the 2-D beam SG spelling (x2, x3, x1)."""
    t = np.deg2rad(t_deg)
    c, s = np.cos(t), np.sin(t)
    return [s, 0.0, c, c, 0.0, -s, 0.0, 1.0, 0.0]


def box(nx, ny, nz, W, H, L, x0=0.0, y0=0.0, taper=1.0):
    """Nodes (N, 3) and hex8 connectivity (gmsh order), with the row j
    and column i of every hex."""
    xs = x0 + W * np.arange(nx + 1) / nx
    ys = y0 + H * np.arange(ny + 1) / ny
    zs = L * np.arange(nz + 1) / nz
    nid = np.arange((nx + 1) * (ny + 1) * (nz + 1)).reshape(
        nz + 1, ny + 1, nx + 1)
    nodes = []
    for z in zs:
        s = 1.0 + (taper - 1.0) * z / L
        for y in ys:
            for x in xs:
                nodes.append((s * x, s * y, z))
    hexes, rows, cols = [], [], []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                a = [nid[k, j, i], nid[k, j, i + 1], nid[k, j + 1, i + 1],
                     nid[k, j + 1, i]]
                b = [nid[k + 1, j, i], nid[k + 1, j, i + 1],
                     nid[k + 1, j + 1, i + 1], nid[k + 1, j + 1, i]]
                hexes.append(a + b)
                rows.append(j)
                cols.append(i)
    return np.array(nodes), hexes, np.array(rows), np.array(cols)


def split(hexes, mask):
    """Split the masked hexes into six tets each, keeping the others."""
    out, src = [], []
    for e, h in enumerate(hexes):
        if mask[e]:
            for t in _TETS:
                out.append([h[q] for q in t])
                src.append(e)
        else:
            out.append(list(h))
            src.append(e)
    return out, np.array(src)


_MIDS10 = [(0, 1), (1, 2), (0, 2), (0, 3), (2, 3), (1, 3)]   # gmsh tet10


def to_tet10(nodes, elems):
    """tet4 cells -> tet10 cells (gmsh node order) with a straight midside
    node on every edge, shared by the cells around it."""
    nodes = [tuple(p) for p in nodes]
    mid = {}
    out = []
    for c in elems:
        row = list(c)
        for a, b in _MIDS10:
            key = tuple(sorted((c[a], c[b])))
            if key not in mid:
                mid[key] = len(nodes)
                nodes.append(tuple(0.5 * (np.asarray(nodes[c[a]])
                                          + np.asarray(nodes[c[b]]))))
            row.append(mid[key])
        out.append(row)
    return np.array(nodes), out


def write_yaml(path, nodes, elems, mat_names, frames, materials,
               frame_key="orientation", header=None):
    """One mesh-dialect segment yaml: nodes, elements (1-based), sets by
    material name, one frame row per element, materials with density."""
    hdr = header or {"msg": "solid", "n_model": 1, "refined": 1,
                     "aperiodic": 1}
    lines = ["%s: %s" % (k, v) for k, v in hdr.items()]
    lines.append("nodes:")
    lines += ["- [%.15g %.15g %.15g]" % tuple(p) for p in nodes]
    lines.append("elements:")
    lines += ["- [%s]" % " ".join(str(int(n) + 1) for n in c) for c in elems]
    lines.append("sets:")
    lines.append("  element:")
    for m in materials:
        lab = [k + 1 for k, nm in enumerate(mat_names) if nm == m["name"]]
        if lab:
            lines.append("  - name: %s" % m["name"])
            lines.append("    labels: [%s]" % ", ".join(str(v) for v in lab))
    lines.append("%s:" % frame_key)
    lines += ["- [%s]" % ", ".join("%.16g" % v for v in fr) for fr in frames]
    lines.append("materials:")
    for m in materials:
        if not any(nm == m["name"] for nm in mat_names):
            continue
        lines.append("- name: %s" % m["name"])
        lines.append("  density: %.16g" % m["density"])
        if m.get("angle") is not None:
            lines.append("  angle: %.16g" % m["angle"])
        lines.append("  elastic:")
        for key in ("E", "G", "nu"):
            lines.append("    %s: [%s]" % (key, ", ".join("%.16g" % v
                                                         for v in m[key])))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def composite(nx=4, ny=4, nz=3, W=2.0, H=1.0, L=2.0, x0=-0.6, y0=-0.3,
              taper=1.0, cut=None, block_angles=False):
    """The composite segment: nodes, elements, material names, frames,
    materials.  cut None = hex8, "all" = tet4, "half" = mixed,
    "tet10" = tet10."""
    nodes, hexes, rows, cols = box(nx, ny, nz, W, H, L, x0, y0, taper)
    mask = {None: np.zeros(len(hexes), bool),
            "all": np.ones(len(hexes), bool),
            "tet10": np.ones(len(hexes), bool),
            "half": cols < nx // 2}[cut]
    elems, src = split(hexes, mask)
    if cut == "tet10":
        nodes, elems = to_tet10(nodes, elems)
    names, frames = [], []
    for e in src:
        j = rows[e]
        if j == 0 or j == ny - 1:
            t = -45.0 if j == 0 else 45.0
            if block_angles:
                names.append("carbon_m45" if j == 0 else "carbon_p45")
                frames.append(IDENTITY)
            else:
                names.append("carbon")
                frames.append(frame_row(t))
        else:
            names.append("core")
            frames.append(IDENTITY)
    if block_angles:
        mats = [dict(CARBON, name="carbon_m45", angle=-45.0),
                dict(CARBON, name="carbon_p45", angle=45.0), CORE]
    else:
        mats = [CARBON, CORE]
    return nodes, elems, names, frames, mats


def isotropic(nx=4, ny=2, nz=3, W=2.0, H=1.0, L=2.0, x0=-0.5, y0=-0.3,
              taper=1.0, cut=None):
    nodes, hexes, rows, cols = box(nx, ny, nz, W, H, L, x0, y0, taper)
    mask = {None: np.zeros(len(hexes), bool),
            "all": np.ones(len(hexes), bool),
            "tet10": np.ones(len(hexes), bool),
            "half": cols < nx // 2}[cut]
    elems, src = split(hexes, mask)
    if cut == "tet10":
        nodes, elems = to_tet10(nodes, elems)
    return (nodes, elems, ["steel"] * len(elems), [IDENTITY] * len(elems),
            [STEEL])


# the stored meshes: name -> (builder, keyword arguments)
CASES = {
    "iso_rect_hex8": (isotropic, {}),
    "comp_rect_hex8": (composite, {}),
    "comp_rect_tet4": (composite, {"cut": "all"}),
    "comp_rect_mixed": (composite, {"cut": "half"}),
    "comp_taper_hex8": (composite, {"taper": 0.6}),
    "comp_taper_tet4": (composite, {"taper": 0.6, "cut": "all"}),
    "comp_taper_mixed": (composite, {"taper": 0.6, "cut": "half"}),
    "comp_rect_tet10": (composite, {"nx": 2, "ny": 4, "nz": 2,
                                    "cut": "tet10"}),
    "comp_taper_tet10": (composite, {"nx": 2, "ny": 4, "nz": 2,
                                     "taper": 0.6, "cut": "tet10"}),
    "iso_rect_tet10": (isotropic, {"nx": 2, "ny": 2, "nz": 2,
                                   "cut": "tet10"}),
}


def write_case(name, folder=HERE, frame_key="orientation", **kw):
    builder, base_kw = CASES[name]
    args = dict(base_kw)
    args.update(kw)
    nodes, elems, names, frames, mats = builder(**args)
    return write_yaml(os.path.join(folder, name + ".yaml"), nodes, elems,
                      names, frames, mats, frame_key=frame_key)


if __name__ == "__main__":
    for nm in CASES:
        print(write_case(nm))
