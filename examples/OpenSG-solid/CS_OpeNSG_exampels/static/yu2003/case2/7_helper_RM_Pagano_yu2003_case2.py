"""Yu-2003 sec. 6.1, case 2 -- symmetric angle ply [30/-30/-30/30], L/h = 4.

Their Figs. 9-14 configuration: Pagano-ratio material (psi), split face load
s3 = b3 = (p0/2) sin(px), stresses normalized by p0.  Three standalone chains --
exact 3-D (shear-coupled Pagano, reference only), MSG-RM (8x8 -> harmonic RM
plate solve -> Eq.-63 recovery -> s33 by thickness equilibrium), Whitney-1973
FSDT baseline.  Symmetric, so B = 0, but D16 bending-twist coupling still makes
M12/Q2 and sigma_23 nonzero.  Writes yu_case2.dat, yu_case2.png, rm_8x8.out here.

Variables: the sys.path line puts examples/yu2003/ on the path; run_case does
everything (see yu_bench.run_case for the full variable glossary).

Run:  python examples/yu2003/case2/7_helper_RM_Pagano_yu2003_case2.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from yu_bench import run_case

run_case("case2")
