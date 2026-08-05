"""Canonical e1/e2/e3 orientation plotter for OpenSG shell/solid SG YAMLs.

COMPULSORY DELIVERABLE: an orientation PNG is emitted on EVERY homogenization run (auto-emitted by
strip_RM.rm_timoshenko_6x6 with orient=True, and by gradient_junction_kirchhoff). Two panels are drawn,
the "Shell SG (1-D)" line mesh and the "Solid SG (2-D)" area mesh, with the per-element material axes:
  e2 = blue   (in-plane ply-flow direction)
  e3 = green  (wall normal; the OML->IML sign convention is documented on the RM theory page)
  e1 = beam axis, out-of-plane.

API:
  plot_orient(shell_yaml, solid_yaml=None, out_png=None) -> out_png   (1 or 2 stacked panels)
  auto_emit(shell_yaml, solid_yaml=None, out_png=None)               (cached once per path per process,
                                                                       never raises -- safe in compute path)
"""
import os
import numpy as np
import yaml

_EMITTED = set()   # process cache of (shell_abs, solid_abs) already plotted -> emit once per mesh


def _load(path):
    d = yaml.safe_load(open(path))
    nodes = np.array([[float(v) for v in str(r[0]).split()][:2] for r in d["nodes"]])
    elems = [[int(v) - 1 for v in str(r[0]).split()] for r in d["elements"]]
    oris = np.array([[float(v) for v in (r if isinstance(r, (list, tuple)) else [r])]
                     for r in d["elementOrientations"]])
    return nodes, elems, oris


def _panel(ax, nodes, elems, oris, is_solid, title, stride):
    """Draw one orientation panel: light mesh + per-element e2 (blue, in-plane) and e3 (green, wall normal)."""
    if is_solid:
        tris = []
        for e in elems:                          # split quads into 2 triangles for the background mesh
            tris.append([e[0], e[1], e[2]])
            if len(e) >= 4:
                tris.append([e[0], e[2], e[3]])
        ax.triplot(nodes[:, 0], nodes[:, 1], np.array(tris), color="0.88", lw=0.12)
    else:
        for e in elems:
            ax.plot(nodes[e[:2], 0], nodes[e[:2], 1], color="0.6", lw=0.6)
    for k in range(0, len(elems), stride):
        cen = nodes[elems[k]].mean(axis=0) if is_solid else nodes[elems[k][:2]].mean(axis=0)
        e2, e3 = oris[k, 3:5], oris[k, 6:8]
        ax.quiver(cen[0], cen[1], e3[0], e3[1], color="tab:green", scale=44, width=0.0030)
        ax.quiver(cen[0], cen[1], e2[0], e2[1], color="tab:blue", scale=40, width=0.0040)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(r"$y_2$ (m)")
    ax.set_ylabel(r"$y_3$ (m)")


def plot_orient(shell_yaml, solid_yaml=None, out_png=None, side_by_side=False):
    """Emit the e1/e2/e3 orientation PNG (shell panel always; solid panel if solid_yaml given).

    side_by_side=True lays the Shell SG and Solid SG panels in one row (1x2) instead of the
    default stacked column (2x1); ignored when no solid panel is drawn."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if out_png is None:
        base = os.path.splitext(os.path.abspath(shell_yaml))[0]
        out_png = base + "_orient_e1e2e3.png"
    sn, se, so = _load(shell_yaml)
    if solid_yaml and side_by_side:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    else:
        rows = 2 if solid_yaml else 1
        fig, axes = plt.subplots(rows, 1, figsize=(12, 4.6 * rows))
        if rows == 1:
            axes = [axes]
    _panel(axes[0], sn, se, so, False, "Shell SG (1-D)", 1)
    if solid_yaml:
        qn, qe, qo = _load(solid_yaml)
        stride = max(1, len(qe) // 450)
        _panel(axes[1], qn, qe, qo, True, "Solid SG (2-D)", stride)
    fig.suptitle("Material orientation    e2 = blue (in-plane ply direction)    e3 = green (wall normal)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    try:                                   # show a relocatable path, never the machine-absolute one
        shown = os.path.relpath(out_png).replace(os.sep, "/")
    except ValueError:                     # out_png on a different drive than cwd (Windows)
        shown = os.path.basename(out_png)
    print("[orient_plot] wrote", shown)
    return out_png


def auto_emit(shell_yaml, solid_yaml=None, out_png=None):
    """Emit once per (shell, solid) path per process; never raises (safe inside the compute path)."""
    key = (os.path.abspath(shell_yaml), os.path.abspath(solid_yaml) if solid_yaml else None)
    if key in _EMITTED:
        return None
    _EMITTED.add(key)
    try:
        return plot_orient(shell_yaml, solid_yaml, out_png)
    except Exception as e:                       # plotting must NEVER break a homogenization run
        print("[orient_plot] skipped (%s)" % e)
        return None
