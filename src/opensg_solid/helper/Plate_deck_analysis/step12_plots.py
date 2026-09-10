"""step12_plots.py -- the 3-curve comparison figures of the plate benchmark:
3-D FEA vs shear-refined vs classical dehomogenization along one path.

    python -m opensg_solid.helper.Plate_deck_analysis.step12_plots \
           --fea <dat> --refined <dat> --classical <dat> --path-label 1 \
           --out-dir plots/ [--npt 25] [--window "0 1"] [--title TEXT] \
           [--labels FEA REFINED CLASSICAL]

WHY NPT POINTS PER CURVE.  The three .dat files sample the same physical
path on three different grids; NPT evenly spaced in-window points PER CURVE
keep the markers legible and make grid-to-grid interpolation unnecessary --
the 2026-09-03 nm25 convention of the TPMS pressure_case study
(fea_driven/scripts/plot_nm25.py), whose look these figures follow.

All three inputs use the unified path layout (written by step06/step10):

    s x y z S11 S22 S33 S12 S13 S23 U1 U2 U3      (# comment lines allowed)

stress in Pa (plotted in kPa), displacement in m, s non-dimensional in
[0, 1].  Curves: 3-D FEA thick tab:blue "o"; shear-refined tab:orange "s"
open; classical tab:green "^" open.  No dotted lines.  NO TITLE by default
(house rule: the caption is the title; since 2026-09-05 -- the earlier
"Path N" heading comes back with --title "Path N").  The legend labels
default to the generic three and are overridden per deck with --labels,
e.g. to name the 3-D element type ("3-D FEA (Abaqus C3D10)").

In:  the three unified path .dat files, the path label (file names only),
     the s-window, optional title, optional three legend labels
Out: <out-dir>/path<N>_<comp>.png for S11 S22 S33 S12 S13 S23 U1 U2 U3
     (dpi 200, bbox tight, legend outside right)"""
import argparse
import os
import sys

import numpy as np

NCOL = 13
COMPS = [
    ("S11", r"$\sigma_{11}$  [kPa]", 4, 1e-3),
    ("S22", r"$\sigma_{22}$  [kPa]", 5, 1e-3),
    ("S33", r"$\sigma_{33}$  [kPa]", 6, 1e-3),
    ("S12", r"$\sigma_{12}$  [kPa]", 7, 1e-3),
    ("S13", r"$\sigma_{13}$  [kPa]", 8, 1e-3),
    ("S23", r"$\sigma_{23}$  [kPa]", 9, 1e-3),
    ("U1", r"$u_1$  [m]", 10, 1.0),
    ("U2", r"$u_2$  [m]", 11, 1.0),
    ("U3", r"$u_3$  [m]", 12, 1.0),
]
CURVES = [
    ("fea", "3-D FEA (Abaqus)",
     dict(color="tab:blue", lw=2.8, ls="-", marker="o", ms=5)),
    ("refined", "Shear-refined plate model (OpenSG)",
     dict(color="tab:orange", lw=1.3, ls="-", marker="s", ms=5,
          mfc="none")),
    ("classical", "Classical plate model (OpenSG)",
     dict(color="tab:green", lw=1.3, ls="-", marker="^", ms=5,
          mfc="none")),
]


def load_path_dat(path):
    """Load and validate one unified path table.

    In:  path str -- whitespace table s x y z S11..S23 U1..U3,
         # comment lines allowed
    Out: (n, 13) float array, rows sorted by s."""
    A = np.atleast_2d(np.loadtxt(path))
    if A.ndim != 2 or A.shape[1] != NCOL:
        raise SystemExit("%s: %d columns, expected %d (unified path"
                         " layout)" % (path, A.shape[-1], NCOL))
    return A[np.argsort(A[:, 0], kind="stable")]


def pick(s, lo, hi, npt):
    """NPT evenly spaced in-window sample indices of one s column.

    In:  s (n,) array; lo, hi float s-window; npt int
    Out: int64 index array (all in-window rows when fewer than npt)."""
    m = np.nonzero((s >= lo) & (s <= hi))[0].astype(np.int64)
    if m.size == 0:
        raise SystemExit("no path points inside the window [%g, %g]"
                         % (lo, hi))
    if m.size <= npt:
        return m
    return m[np.round(np.linspace(0, m.size - 1, npt)).astype(np.int64)]


def make_figures(fea, refined, classical, path_label, out_dir,
                 npt=25, window=(0.0, 1.0), title=None, labels=None):
    """Write the nine comparison figures of one path.

    In:  fea, refined, classical str -- unified path .dat files;
         path_label str | int (file names only); out_dir str; npt int;
         window (lo, hi); title str | None -- axes title (None = none);
         labels [fea, refined, classical] | None -- legend labels
         (None = the CURVES defaults)
    Out: list of PNG paths written."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    files = {"fea": fea, "refined": refined, "classical": classical}
    data = {}
    for key, _lbl, _sty in CURVES:
        A = load_path_dat(files[key])
        data[key] = (A, pick(A[:, 0], window[0], window[1], npt))
    os.makedirs(out_dir, exist_ok=True)
    if labels is not None and len(labels) != len(CURVES):
        raise SystemExit("labels: expected %d, got %d"
                         % (len(CURVES), len(labels)))
    names = dict(zip([k for k, _l, _s in CURVES], labels or []))
    out = []
    for name, ylab, col, sc in COMPS:
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        for key, lbl, sty in CURVES:
            A, idx = data[key]
            ax.plot(A[idx, 0], sc * A[idx, col], label=names.get(key, lbl),
                    **sty)
        if title:
            ax.set_title(title, fontsize=11)
        ax.set_xlabel(r"$s$")
        ax.set_ylabel(ylab)
        ax.grid(True, ls="-", lw=0.4, alpha=0.35)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
                  frameon=False, fontsize=8)
        fig.tight_layout()
        png = os.path.join(out_dir,
                           "path%s_%s.png" % (path_label, name))
        fig.savefig(png, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("wrote %s" % png)
        out.append(png)
    return out


def main(argv=None):
    """CLI: the nine figures of one path in one call.

    In:  argv list | None
    Out: int exit status."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--fea", required=True)
    p.add_argument("--refined", required=True)
    p.add_argument("--classical", required=True)
    p.add_argument("--path-label", required=True,
                   help="N of the path<N>_<comp>.png file names")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--npt", type=int, default=25)
    p.add_argument("--window", default="0 1",
                   help='s-window "lo hi" (default the full 0..1)')
    p.add_argument("--title", default=None,
                   help="axes title (default none; e.g. \"Path 1\")")
    p.add_argument("--labels", nargs=3, default=None,
                   metavar=("FEA", "REFINED", "CLASSICAL"),
                   help="legend labels of the three curves (default:"
                        " the generic ones)")
    a = p.parse_args(argv)
    w = [float(v) for v in a.window.split()]
    if len(w) != 2 or w[0] >= w[1]:
        raise SystemExit('--window must be "lo hi" with lo < hi')
    make_figures(a.fea, a.refined, a.classical, a.path_label, a.out_dir,
                 npt=a.npt, window=(w[0], w[1]), title=a.title,
                 labels=a.labels)
    return 0


if __name__ == "__main__":
    sys.exit(main())
