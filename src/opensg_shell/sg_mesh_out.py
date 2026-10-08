"""sg_mesh_out.py -- the ``--mesh`` outputs of an msg-shell run: <base>_mesh.png + <base>.msh.

png : elements coloured by section (layup), nodes as dots.  For a 1-D ring SG
      the laminate BAND of every element is drawn about the contour at the
      ACTIVE reference (oml: [0, t] inward along e3; center: [-t/2, +t/2]) --
      the picture is the visual check that --center matches how the yaml was
      generated.  A 3-D shell SG is drawn as its coloured facets.
msh : gmsh 2.2 ASCII, one physical group per section named after the
      elementSet (2-node lines for a ring SG, 3/4-node facets for a 3-D shell
      SG).  Nodes are written 1..N in yaml order and elements in yaml order, so
      the ids match the yaml, <base>_ABDG.out and the dehom .txt element column.

matplotlib is imported lazily: a plain run never pays for it.
"""
import os

import numpy as np
import yaml

try:                                    # libyaml C loader: ~5x faster on big SG yamls
    from yaml import CSafeLoader as _YL
except ImportError:
    from yaml import SafeLoader as _YL

# nodes per element -> gmsh 2.2 element type: 2-node line, 3-node triangle, 4-node quad
_GMSH_TYPE = {2: 1, 3: 2, 4: 3}


def _row(r):
    """Yaml row -> list of floats (rows are single space-separated strings or lists)."""
    if isinstance(r, (list, tuple)) and len(r) == 1 and isinstance(r[0], str):
        r = r[0]
    if isinstance(r, str):
        return [float(x) for x in r.split()]
    return [float(x) for x in r]


def load_shell_mesh(yaml_path):
    """The geometry an msg-shell yaml describes, as plain arrays.

    In:  yaml_path str
    Out: dict -- nodes (N,3); elems list of 0-based node lists; sec (E,) section
         index per element (-1 = in no set); names list of elementSet names;
         thick (S,) laminate thickness per section; ori (E,9) elementOrientations
         or None."""
    d = yaml.load(open(yaml_path), Loader=_YL)
    nodes = np.array([_row(r)[:3] for r in d["nodes"]], float)
    elems = [[int(v) for v in _row(e)] for e in d["elements"]]
    if min(min(e) for e in elems) == 1:
        elems = [[v - 1 for v in e] for e in elems]
    sections = d.get("sections", [])
    names = [str(s["elementSet"]) for s in sections]
    idx = {n: i for i, n in enumerate(names)}
    sec = np.full(len(elems), -1, int)
    for grp in d.get("sets", {}).get("element", []):
        si = idx.get(str(grp["name"]))
        if si is None:
            continue
        for lab in grp["labels"]:
            sec[int(lab) - 1] = si
    thick = np.array([sum(float(p[1]) for p in s["layup"]) for s in sections], float)
    ori = (np.array([_row(o) for o in d["elementOrientations"]], float)
           if d.get("elementOrientations") else None)
    return dict(nodes=nodes, elems=elems, sec=sec, names=names, thick=thick, ori=ori)


def write_shell_msh(mesh, msh_path):
    """gmsh 2.2 ASCII of the shell mesh, one physical group per section.

    In:  mesh dict (load_shell_mesh); msh_path str
    Out: str msh_path (written)."""
    nodes, elems, sec, names = mesh["nodes"], mesh["elems"], mesh["sec"], mesh["names"]
    dim = 1 if all(len(e) == 2 for e in elems) else 2
    with open(msh_path, "w") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        if names:
            f.write("$PhysicalNames\n%d\n" % len(names))
            for i, n in enumerate(names):
                f.write('%d %d "%s"\n' % (dim, i + 1, n))
            f.write("$EndPhysicalNames\n")
        f.write("$Nodes\n%d\n" % len(nodes))
        for i, x in enumerate(nodes):
            f.write("%d %.10f %.10f %.10f\n" % (i + 1, x[0], x[1], x[2]))
        f.write("$EndNodes\n$Elements\n%d\n" % len(elems))
        for e, conn in enumerate(elems):
            et = _GMSH_TYPE.get(len(conn))
            if et is None:
                raise ValueError("element %d has %d nodes; an msg-shell SG holds"
                                 " 2-node lines or 3/4-node facets" % (e + 1, len(conn)))
            tag = int(sec[e]) + 1                  # physical = elementary = section id (1-based)
            f.write("%d %d 2 %d %d %s\n"
                    % (e + 1, et, tag, tag, " ".join(str(n + 1) for n in conn)))
        f.write("$EndElements\n")
    return msh_path


def plot_shell_mesh(mesh, png_path, ref="oml", title=""):
    """<base>_mesh.png: elements by section, nodes, and (ring SG) the laminate band
    at the active reference.

    In:  mesh dict (load_shell_mesh); png_path str; ref str (sg_reference);
         title str
    Out: str png_path (written)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection
    from .sg_reference import describe, frac_of, norm_ref

    ref = norm_ref(ref)
    frac = frac_of(ref)
    nodes, elems, sec = mesh["nodes"], mesh["elems"], mesh["sec"]
    names, thick, ori = mesh["names"], mesh["thick"], mesh["ori"]
    n_sec = max(len(names), 1)
    cmap = plt.get_cmap("tab20" if n_sec > 10 else "tab10")
    col = {i: cmap(i % cmap.N) for i in range(n_sec)}
    col[-1] = (0.6, 0.6, 0.6, 1.0)
    counts = np.bincount(sec[sec >= 0], minlength=n_sec) if len(sec) else np.zeros(n_sec, int)
    ring = all(len(e) == 2 for e in elems)

    if ring:
        # the cross-section plane = the two coordinates that vary (the ring
        # loaders' cross = [0, 1], ax = 2 for a station yaml with z = 0)
        ptp = nodes.max(0) - nodes.min(0)
        ax_idx = int(np.argmin(ptp))
        cross = [j for j in range(3) if j != ax_idx]
        xy = nodes[:, cross]
        fig, ax = plt.subplots(figsize=(11, 6))
        bands, bcols, segs, scols = [], [], [], []
        for e, (a, b) in enumerate(elems):
            p0, p1 = xy[a], xy[b]
            t = float(thick[sec[e]]) if sec[e] >= 0 else 0.0
            if ori is not None:
                n = ori[e, 6:9][cross]             # e3 in-plane components (inward)
            else:
                tv = p1 - p0
                n = np.array([-tv[1], tv[0]])
            nn = float(np.linalg.norm(n))
            n = n / nn if nn > 0 else n
            z0, z1 = -frac * t, (1.0 - frac) * t    # laminate band about the contour along +e3
            bands.append([p0 + z0 * n, p1 + z0 * n, p1 + z1 * n, p0 + z1 * n])
            bcols.append(col[int(sec[e])])
            segs.append([p0, p1])
            scols.append(col[int(sec[e])])
        ax.add_collection(PolyCollection(bands, facecolors=bcols, edgecolors="none", alpha=0.35))
        ax.add_collection(LineCollection(segs, colors=scols, linewidths=1.6))
        ax.plot(xy[:, 0], xy[:, 1], "k.", ms=2.5)
        ax.set_aspect("equal")
        ax.autoscale()
        ax.set_xlabel("$y_%d$" % (cross[0] + 1))
        ax.set_ylabel("$y_%d$" % (cross[1] + 1))
        band_note = "band = laminate at the %s reference" % ref
    else:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        polys = [nodes[np.asarray(e, int), :3] for e in elems]
        cols = [col[int(s)] for s in sec]
        ax.add_collection3d(Poly3DCollection(polys, facecolors=cols, edgecolors="k",
                                             linewidths=0.15))
        lo, hi = nodes.min(0), nodes.max(0)
        c0, r = 0.5 * (lo + hi), 0.5 * float((hi - lo).max()) or 1.0
        ax.set_xlim(c0[0] - r, c0[0] + r)
        ax.set_ylim(c0[1] - r, c0[1] + r)
        ax.set_zlim(c0[2] - r, c0[2] + r)
        ax.set_xlabel("$y_1$"); ax.set_ylabel("$y_2$"); ax.set_zlabel("$y_3$")
        band_note = "reference: %s" % ref

    handles = [plt.Line2D([0], [0], color=col[i], lw=4,
                          label="%s  (%d el, t = %.2f mm)" % (names[i], counts[i], 1e3 * thick[i]))
               for i in range(len(names))]
    if (sec < 0).any():
        handles.append(plt.Line2D([0], [0], color=col[-1], lw=4,
                                  label="no section (%d el)" % int((sec < 0).sum())))
    if handles:
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
                  fontsize=8, frameon=False)
    ax.set_title("%s -- %d nodes, %d elements\n%s" % (title, len(nodes), len(elems),
                                                     describe(ref) if ring else band_note),
                 fontsize=9)
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return png_path


def write_mesh_files(yaml_path, out_base, ref="oml"):
    """The ``--mesh`` pair of an msg-shell run.

    In:  yaml_path str; out_base str (path stem next to the other outputs);
         ref str -- the reference the run uses (draws the laminate band there)
    Out: (msh_path, png_path)."""
    mesh = load_shell_mesh(yaml_path)
    msh = write_shell_msh(mesh, out_base + ".msh")
    png = plot_shell_mesh(mesh, out_base + "_mesh.png", ref=ref,
                          title=os.path.basename(yaml_path))
    for p in (msh, png):          # never report a file that is not there
        if not os.path.exists(p):
            import matplotlib
            raise RuntimeError("%s was not written (matplotlib backend %s)"
                               % (p, matplotlib.get_backend()))
    return msh, png
