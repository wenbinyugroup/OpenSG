"""Benchmark 3 -- caseC sandwich [0/core/0] (faces 0.1h, core 0.8h), S = a/h in
{4, 10, 50}.

Three STANDALONE chains at the stations x = 0 (sigma_13) and x = a/2 (sigma_33):
exact 3-D (Pagano, reference curves only), MSG-RM (statics Q1 = q0/p -> 8x8
inversion -> Eq.-63 recovery -> sigma_33 by thickness equilibrium), and FSDT with
the Whitney-1973 Eq.-(7) k1^2.  The sandwich is where the FSDT staircase collapses
(the soft core carries almost no constitutive shear).  S = 4 is the thick
Yu-2003-style check, 10/50 the plate-model regime.  Writes pagano_S<S>.dat,
pagano_S<S>.png, rm_8x8.out here.

Variables: the two sys.path lines put examples/garg/ on the path;
run_benchmark(case, S_list, tag) does everything (see pagano_bench.run_case for
the full variable glossary).

Run:  python examples/garg/caseC/7_helper_RM_Pagano_benchmark3.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pagano_bench import run_benchmark

run_benchmark("caseC", (4, 10, 50), tag="3")
