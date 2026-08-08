"""
orient_check.py   [ pure numpy + matplotlib, no dolfinx ]
========================================================================
Mandatory ORIENTATION PRECHECK for the surface-quad / boundary meshes.

Because dolfinx renumbers nodes AND cells, the per-element material frame
(e1,e2,e3) must be verified AFTER any mesh handling.  These helpers:
  * `frame_report`   -- numeric check: orthonormality, right-handedness, and
                        that e1 is axial (x) and e3 is the inward radial normal
                        for the cylinder; returns a pass/fail summary string;
  * `orientation_png`-- ALWAYS-emitted arrow plot of e1/e2/e3 at element
                        centroids so the orientation can be eyeballed.

Convention (matches the project figure defaults): e2 blue, e3 black; e1 red.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

E1_COL, E2_COL, E3_COL = "#e31a1c", "#1f78b4", "#000000"   # red / blue / black


def frame_report(nodes, cells, e1, e2, e3, tol=1e-6):
    """Numeric orientation-consistency check. Returns (ok, text)."""
    e1 = np.asarray(e1); e2 = np.asarray(e2); e3 = np.asarray(e3)
    cent = np.array([nodes[list(c)].mean(axis=0) for c in cells])
    n1 = np.linalg.norm(e1, axis=1); n2 = np.linalg.norm(e2, axis=1); n3 = np.linalg.norm(e3, axis=1)
    orth = np.max(np.abs([np.sum(e1 * e2, 1), np.sum(e2 * e3, 1), np.sum(e1 * e3, 1)]))
    rh = np.sum(np.cross(e1, e2) * e3, axis=1)                 # e1 x e2 . e3 (=+1 right-handed)
    axial = np.min(e1[:, 0] / np.clip(n1, 1e-30, None))        # e1 . x_hat  (expect ~1: e1 = axial)
    r_hat = cent[:, 1:] / np.clip(np.linalg.norm(cent[:, 1:], axis=1)[:, None], 1e-30, None)
    e3_dot_inward = np.mean(np.sum(e3[:, 1:] * (-r_hat), axis=1))   # e3 . (-r_hat) (expect ~1: inward)
    ok = (abs(n1.mean() - 1) < 1e-6 and abs(n2.mean() - 1) < 1e-6 and abs(n3.mean() - 1) < 1e-6
          and orth < 1e-6 and rh.min() > 0.99 and axial > 0.999)
    txt = ("frame: |e1|=%.4f |e2|=%.4f |e3|=%.4f  max|off-diag|=%.1e  "
           "RH(e1xe2.e3) in [%.3f,%.3f]  min(e1.x)=%.4f  mean(e3.-r)=%.4f  -> %s"
           % (n1.mean(), n2.mean(), n3.mean(), orth, rh.min(), rh.max(), axial,
              e3_dot_inward, "OK" if ok else "CHECK"))
    return ok, txt


def _set_equal_3d(ax, pts):
    r = (pts.max(0) - pts.min(0)); r[r == 0] = 1.0
    ax.set_box_aspect(r)


def orientation_png(nodes, cells, e1, e2, e3, out_png, title="", step=1, scale=None):
    """3-D quiver of the per-cell frame at element centroids (precheck byproduct)."""
    nodes = np.asarray(nodes)
    cent = np.array([nodes[list(c)].mean(axis=0) for c in cells])
    if scale is None:
        scale = 0.12 * np.linalg.norm(nodes.max(0) - nodes.min(0))
    sel = slice(None, None, step)
    fig = plt.figure(figsize=(9, 7)); ax = fig.add_subplot(111, projection="3d")
    for vec, col, lab in [(e1, E1_COL, "e1 axial"), (e2, E2_COL, "e2 hoop"), (e3, E3_COL, "e3 normal")]:
        v = np.asarray(vec)[sel]
        ax.quiver(cent[sel, 0], cent[sel, 1], cent[sel, 2], v[:, 0], v[:, 1], v[:, 2],
                  length=scale, color=col, label=lab, normalize=True, linewidth=0.6)
    ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], s=1, c="#bbbbbb", alpha=0.35)
    ax.set_xlabel("x (axial)"); ax.set_ylabel("y"); ax.set_zlabel("z"); ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    _set_equal_3d(ax, nodes)
    fig.savefig(out_png, dpi=130, bbox_inches="tight"); plt.close(fig)
    return out_png


def orientation_png_ring(ring_nodes, ring_cells, e2, e3, out_png, title=""):
    """2-D (y,z) quiver of a boundary ring's in-plane frame at edge midpoints."""
    rn = np.asarray(ring_nodes)
    mid = np.array([rn[list(c)].mean(axis=0) for c in ring_cells])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(rn[:, 1], rn[:, 2], ".", ms=2, c="#bbbbbb")
    e2 = np.asarray(e2); e3 = np.asarray(e3)
    ax.quiver(mid[:, 1], mid[:, 2], e2[:, 1], e2[:, 2], color=E2_COL, label="e2 hoop",
              scale=25, width=0.004)
    ax.quiver(mid[:, 1], mid[:, 2], e3[:, 1], e3[:, 2], color=E3_COL, label="e3 normal",
              scale=25, width=0.004)
    ax.set_aspect("equal"); ax.set_xlabel("y"); ax.set_ylabel("z"); ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(out_png, dpi=130, bbox_inches="tight"); plt.close(fig)
    return out_png
