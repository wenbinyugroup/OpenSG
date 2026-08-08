"""Tapered demonstration: BAR-URC 100 m blade spanwise shell segment.

A real tapered blade segment (two DIFFERENT end cross-sections) run through
the aperiodic pipeline.  Here no identity holds: each boundary ring has its
own 6x6, and the segment result is the true length-averaged Timoshenko
stiffness of the tapered piece -- its terms fall between the two ring
values (the diag ratio columns make that visible).

Variables
---------
YAML   : 3-D shell yaml of the tapered segment (from the BAR-URC dataset)
r      : dict(S6, C6L, C6R, L, solve_time); writes <yaml>_Timo.out (timed),
         boundary 1-D yamls and orientation PNGs alongside
S, L, R: segment 6x6 and the left/right boundary-ring 6x6s
ratio  : (S_ii - R_ii)/(L_ii - R_ii); a value in [0, 1] means the segment
         diagonal lies between its two boundary rings

Run (from this folder):  python run_bar_urc_segment.py
"""
import numpy as np
from opensg_shell.segment_taper import segment_timo_from_3dyaml

YAML = "BAR_URC_numEl_52_segment_12.yaml"
LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]

r = segment_timo_from_3dyaml(YAML)
S, L6, R6 = r["S6"], r["C6L"], r["C6R"]
rows = ["# BAR-URC tapered shell segment 12: aperiodic-boundary Timoshenko"
        " 6x6",
        "# L=%.6g   solve %.1f s" % (r["L"], r["solve_time"])]
for nm, Mx in (("segment S6", S), ("boundary ring L C6", L6),
               ("boundary ring R C6", R6)):
    rows.append("# " + nm)
    for i in range(6):
        rows.append(" ".join("%14.6e" % v for v in Mx[i]))
rows.append("# diag: segment between the two rings?"
            "  ratio = (S-R)/(L-R)")
for i in range(6):
    den = L6[i, i] - R6[i, i]
    ratio = (S[i, i] - R6[i, i])/den if abs(den) > 0 else float("nan")
    rows.append("#  %-4s S %13.6e   ringL %13.6e   ringR %13.6e   ratio %7.3f"
                % (LBL[i], S[i, i], L6[i, i], R6[i, i], ratio))
open("bar_urc_segment12.dat", "w").write("\n".join(rows) + "\n")
print("\n".join(rows))
