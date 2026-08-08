"""Aperiodic-boundary segment: prismatic-identity validation.

The 3-D shell yaml of a PRISMATIC circular tube (R = 1, t/R = 0.1, L = 1.5;
isotropic E = 70 GPa nu = 0.3, and a +-45 anisotropic layup) is homogenized
with the aperiodic pipeline: the two end cross-sections are extracted from
the 3-D yaml as separate 1-D boundary yamls, each boundary ring is solved on
its own (V0, V1), and the ring fields are mapped onto the segment's boundary
nodes as Dirichlet data replacing axial periodicity.

For a prismatic segment the taper terms vanish, so the segment Timoshenko
6x6 MUST equal the boundary ring's own 6x6 -- this identity is the
acceptance test of the whole boundary extraction + V0/V1 mapping.

Variables
---------
CASES : mesh tags produced by make_cylinder_segment.py (meshes/seg_<tag>.yaml)
r     : dict(S6, C6L, C6R, L, solve_time) from segment_timo_from_3dyaml;
        also writes meshes/seg_<tag>_Timo.out (timed SwiftComp layout),
        the two boundary 1-D yamls and the orientation-frame PNGs
S, Rg : segment 6x6 and boundary-ring 6x6 (diag: EA GA2 GA3 GJ EI2 EI3)
E     : per-term % error 6x6 (insignificant ring terms reported as 0)

Run (from this folder):
    python make_cylinder_segment.py
    python run_taper_segment.py
"""
import numpy as np
from opensg_shell.segment_taper import segment_timo_from_3dyaml

CASES = ["iso_hR0.1", "aniso_hR0.1"]

rows = ["# prismatic identity: segment (aperiodic Dirichlet ends) vs"
        " boundary ring",
        "# diag order: EA GA2 GA3 GJ EI2 EI3"]
for tg in CASES:
    r = segment_timo_from_3dyaml("meshes/seg_%s.yaml" % tg)
    S, Rg = r["S6"], r["C6L"]
    sig = np.abs(Rg) > 1e-6*np.abs(Rg).max()
    E = 100*(S - Rg)/np.where(sig, Rg, np.inf)
    rows.append("")
    rows.append("## %s   L=%g   solve %.1f s" % (tg, r["L"], r["solve_time"]))
    for nm, Mx, fmt in (("segment S6", S, "%14.6e"),
                        ("boundary ring C6", Rg, "%14.6e"),
                        ("% error", E, "%14.4f")):
        rows.append("# " + nm)
        for i in range(6):
            rows.append(" ".join(fmt % v for v in Mx[i]))
open("seg_ring_identity.dat", "w").write("\n".join(rows) + "\n")
print("\n".join(rows))
