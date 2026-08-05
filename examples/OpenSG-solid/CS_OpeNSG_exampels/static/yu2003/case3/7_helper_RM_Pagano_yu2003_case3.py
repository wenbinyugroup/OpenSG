"""Yu-2003 sec. 6.1, case 3 -- symmetric nearly cross ply [0.5/90.5/90.5/0.5],
L/h = 4.

Their Figs. 15-20 configuration: Pagano-ratio material (psi), split face load
s3 = b3 = (p0/2) sin(px), stresses normalized by p0.  Yu perturbed the cross-ply
angles by 0.5 deg only because Sutyrin's Mathematica exact code could not handle
cross-ply; the state-space exact solver here has no such restriction, but the
SAME angles are kept to reproduce his curves.  Three standalone chains -- exact
3-D (reference only), MSG-RM (8x8 -> harmonic RM plate solve -> Eq.-63 recovery
-> s33 by thickness equilibrium), Whitney-1973 FSDT baseline.  The 0.5-deg
off-axis makes sigma_23 tiny but nonzero (its exact scale is printed in the
.dat -- a relative error on it is fragile by construction).  Writes yu_case3.dat,
yu_case3.png, rm_8x8.out here.

Variables: the sys.path line puts examples/yu2003/ on the path; run_case does
everything (see yu_bench.run_case for the full variable glossary).

Run:  python examples/yu2003/case3/7_helper_RM_Pagano_yu2003_case3.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from yu_bench import run_case

run_case("case3")
