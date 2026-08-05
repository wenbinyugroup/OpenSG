"""Yu-2003 sec. 6.1, case 1 -- antisymmetric angle ply [15/-15], L/h = 4.

Their Figs. 3-8 configuration: Pagano-ratio material (psi), split face load
s3 = b3 = (p0/2) sin(px), stresses normalized by p0.  Three standalone chains --
exact 3-D (shear-coupled Pagano, reference only), MSG-RM (8x8 -> harmonic RM
plate solve -> Eq.-63 recovery -> s33 by thickness equilibrium), Whitney-1973
FSDT baseline.  The antisymmetric B-coupling makes M12/Q2 nonzero: the harmonic
solve supplies them (see yu_bench.rm_cyl_bend).  Writes yu_case1.dat,
yu_case1.png, rm_8x8.out here.

Variables: the sys.path line puts examples/yu2003/ on the path; run_case does
everything (see yu_bench.run_case for the full variable glossary).

Run:  python examples/yu2003/case1/7_helper_RM_Pagano_yu2003_case1.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from yu_bench import run_case

run_case("case1")
