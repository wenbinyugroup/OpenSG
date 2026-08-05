"""xsec_5v6_master.py -- cross-section Timoshenko 6x6 for the RM cross-section paper.

Two RM cross-section (boundary-ring) formulations, BOTH tying only gamma_23:
  A = 6-DOF independent-omega3 element + element-wise LAGRANGE multiplier enforcing the
      drilling (omega3) in-plane constraint         -> run_ring_indep.ring_indep(shear="mitc4_g23")
  B = 5-DOF drilling-ELIMINATED MITC element        -> segment_element_general.ring_general(shear="mitc4_g23")
compared against the 2-D solid cross-section 6x6 (VABS-convention .txt), per Timo term.

Cases: single-cell tube, two-cell (webbed) tube, IEA-22 r=0.2 and r=0.3 boundary rings.
All rings referenced at the contour CENTROID (center_ref=True), matching the centroidal
2-D solid reference.  Prints full 6x6 %err for both formulations + saves an npz per case.

    python xsec_5v6_master.py
"""
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
TWP = os.path.abspath(os.path.join(HERE, ".."))

from .segment_element import compute_k22
from .solve_segment_jax import _material_by_section
from .run_ring_indep import ring_indep
from .segment_element_general import ring_general

LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
OUT = os.path.join(HERE, "results")


def _row(r):
    if isinstance(r, list):
        r = r[0] if (len(r) == 1 and isinstance(r[0], str)) else r
    if isinstance(r, str):
        return [float(v) for v in r.replace(",", " ").split()]
    return [float(v) for v in r]


def _norm_materials(mats):
    out = []
    for mm in mats:
        if "elastic" in mm:
            out.append(mm)
        else:
            out.append({"name": mm["name"], "elastic": {"E": mm["E"], "G": mm["G"], "nu": mm["nu"]}})
    return out


def load_ring(path, center_ref=True):
    """1-D contour shell YAML -> ring arrays (rx, cells, rsub, re3, D_by, G_by, k22, ax, cross).
    center_ref=True references the plate ABD to the laminate MID-surface (default, for a mid-surface
    contour); center_ref=False references it to the OML (use with an OML contour, fraction=0.0)."""
    d = yaml.safe_load(open(path))
    rx = np.array([_row(r)[:3] for r in d["nodes"]], dtype=float)
    cells = np.array([[int(v) for v in _row(e)] for e in d["elements"]], dtype=int)
    if cells.min() == 1:
        cells = cells - 1
    ori = np.array([_row(o) for o in d["elementOrientations"]], dtype=float)
    re2, re3 = ori[:, 3:6], ori[:, 6:9]
    sections = d["sections"]; materials = _norm_materials(d["materials"])
    setname_to_sec = {s["elementSet"]: i for i, s in enumerate(sections)}
    rsub = np.zeros(len(cells), dtype=int)
    for grp in d["sets"]["element"]:
        si = setname_to_sec[grp["name"]]
        for lab in grp["labels"]:
            rsub[int(lab) - 1] = si
    ax = 2; cross = [0, 1]
    D_by, G_by = _material_by_section(sections, materials, center_ref=center_ref)
    k22 = compute_k22(rx[cells].mean(1), re2, re3, cells)
    return dict(rx=rx, cells=cells, rsub=rsub, re3=re3, D_by=D_by, G_by=G_by, k22=k22,
                ax=ax, cross=cross, nweb=int((rsub > 0).sum()))


def load_solid(path):
    return 0.5 * (np.loadtxt(path) + np.loadtxt(path).T)


def ring_6dof(R):
    return 0.5 * (lambda C: C + C.T)(ring_indep(R["rx"], R["cells"], R["rsub"], R["re3"],
                 R["D_by"], R["G_by"], R["k22"], R["ax"], R["cross"],
                 shear="mitc4_g23", lam_space="elem"))


def ring_5dof(R):
    C, _, _ = ring_general(R["rx"], R["cells"], R["rsub"], R["re3"], R["D_by"], R["G_by"],
                           R["k22"], R["ax"], R["cross"], shear="mitc4_g23")
    return 0.5 * (C + C.T)


def pct(S, ref):
    cut = np.abs(ref).max() / 1e3
    E = np.full((6, 6), np.nan)
    m = np.abs(ref) > cut
    E[m] = 100.0 * (S[m] - ref[m]) / ref[m]
    return E


def show(name, S, So, C6, C5):
    print("\n################ %s ################" % name)
    print("2-D solid diag :", "  ".join("%s=%.4g" % (LBL[i], So[i, i]) for i in range(6)))
    print("6-DOF (Lagrange, g23) diag:", "  ".join("%.4g" % C6[i, i] for i in range(6)))
    print("5-DOF (MITC,     g23) diag:", "  ".join("%.4g" % C5[i, i] for i in range(6)))
    for tagn, C in (("6-DOF Lagrange g23", C6), ("5-DOF MITC g23", C5)):
        e = [100.0 * (C[i, i] - So[i, i]) / So[i, i] for i in range(6)]
        print("  %-20s %%err diag: %s" % (tagn, "  ".join("%s %+6.2f" % (LBL[i], e[i]) for i in range(6))))


if __name__ == "__main__":          # demo/validation study -- runs ONLY when this file is executed
    #                                 directly; importing load_ring/ring_6dof no longer triggers it.
    IB = os.path.join(TWP, "iea22_blade", "data")
    SC = os.path.join(TWP, "single_cell_tube", "data")
    TC = os.path.join(TWP, "two_cell_tube", "data")

    CASES = [
        ("single_tube", os.path.join(SC, "shell_center.yaml"), os.path.join(SC, "C6_solid_314.txt")),
        ("two_cell_tube", os.path.join(TC, "tube2cell_thin.yaml"), os.path.join(TC, "C6_solid_tube2cell_thin.txt")),
        ("iea_r020", os.path.join(IB, "shell_r020.yaml"), os.path.join(IB, "C6_solid_r020.txt")),
        ("iea_r030", os.path.join(IB, "shell_r030.yaml"), os.path.join(IB, "C6_solid_r030.txt")),
    ]

    os.makedirs(OUT, exist_ok=True)
    summary = {}
    for name, shell, solid in CASES:
        if not (os.path.exists(shell) and os.path.exists(solid)):
            print("SKIP %s (missing %s or %s)" % (name, shell, solid))
            continue
        R = load_ring(shell)
        So = load_solid(solid)
        C6 = ring_6dof(R)
        C5 = ring_5dof(R)
        print("\n=== %s : %d nodes, %d cells, %d web cells ===" % (name, len(R["rx"]), len(R["cells"]), R["nweb"]))
        show(name, R, So, C6, C5)
        np.savez(os.path.join(OUT, "%s.npz" % name), solid=So, c6dof=C6, c5dof=C5,
                 err6=pct(C6, So), err5=pct(C5, So))
        summary[name] = (So, C6, C5)

    print("\n\n=================== SUMMARY: diagonal %err (6-DOF Lagrange | 5-DOF MITC) ===================")
    print("%-14s %s" % ("case", "  ".join("%-14s" % L for L in LBL)))
    for name, (So, C6, C5) in summary.items():
        e6 = [100.0 * (C6[i, i] - So[i, i]) / So[i, i] for i in range(6)]
        e5 = [100.0 * (C5[i, i] - So[i, i]) / So[i, i] for i in range(6)]
        print("%-14s %s" % (name, "  ".join("%+5.1f|%+5.1f  " % (e6[i], e5[i]) for i in range(6))))
    print("\nwrote npz per case ->", OUT)
