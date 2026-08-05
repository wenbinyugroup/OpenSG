"""oml_ring.py -- ring loader with an explicit laminate reference convention.

'center'   : laminate centred on the contour line (mid-surface meshes: tubes, ellipse)
'oml'      : contour on the OML, laminate stacked INWARD from the line (airfoil yamls:
             IEA-22, st15/BAR-URC -- official OpenSG shell frac=0 convention)
'oml_flip' : reference shifted the full thickness the other way (diagnostic only)
'iml'      : contour on the INNER mold line (a fraction=1.0 yaml), laminate stacked
             OUTWARD from the line -- same full-thickness ABD shift as oml_flip
"""
import os
import sys

import numpy as np
import yaml

from .segment_element import compute_k22
from .solve_segment_jax import _material_by_section
from .fe_jax.msg_materials import shift_abd_reference
from .run_ring_indep import ring_indep
from .xsec_5v6_master import _row, _norm_materials


def load_ring_ref(path, ref="oml"):
    d = yaml.safe_load(open(path))
    rx = np.array([_row(r)[:3] for r in d["nodes"]], dtype=float)
    cells = np.array([[int(v) for v in _row(e)] for e in d["elements"]], dtype=int)
    if cells.min() == 1:
        cells = cells - 1
    ori = np.array([_row(o) for o in d["elementOrientations"]], dtype=float)
    re3 = ori[:, 6:9]
    sections = d["sections"]; materials = _norm_materials(d["materials"])
    setname_to_sec = {s["elementSet"]: i for i, s in enumerate(sections)}
    rsub = np.zeros(len(cells), dtype=int)
    for grp in d["sets"]["element"]:
        si = setname_to_sec[grp["name"]]
        for lab in grp["labels"]:
            rsub[int(lab) - 1] = si
    if ref == "center":
        D_by, G_by = _material_by_section(sections, materials, center_ref=True)
    else:
        D_by, G_by = _material_by_section(sections, materials, center_ref=False)
        if ref in ("oml_flip", "iml"):
            for si, sec in enumerate(sections):
                t = sum(float(p[1]) for p in sec["layup"])
                D_by[si] = shift_abd_reference(np.asarray(D_by[si]), t)
    k22 = compute_k22(rx[cells].mean(1), ori[:, 3:6], re3, cells)
    return dict(rx=rx, cells=cells, rsub=rsub, re3=re3, D_by=D_by, G_by=G_by,
                k22=k22, ax=2, cross=[0, 1])


def c6(R):
    C = ring_indep(R["rx"], R["cells"], R["rsub"], R["re3"], R["D_by"], R["G_by"],
                   R["k22"], R["ax"], R["cross"], shear="mitc4_g23", lam_space="elem")
    return 0.5 * (C + C.T)


def derr(C, So):
    return np.array([100.0 * (C[i, i] - So[i, i]) / So[i, i] for i in range(6)])
