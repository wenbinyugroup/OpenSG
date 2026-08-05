"""Benchmark 1 -- caseA [0/90/0] Pagano graphite/epoxy, S = a/h in {4, 10, 50}.

Three STANDALONE chains at the stations x = 0 (sigma_13) and x = a/2 (sigma_33):
exact 3-D (Pagano, reference curves only), MSG-RM (statics Q1 = q0/p -> 8x8
inversion -> Eq.-63 recovery -> sigma_33 by thickness equilibrium), and FSDT with
the Whitney-1973 Eq.-(7) k1^2.  S = 4 is the thick Yu-2003-style check, 10/50 the
plate-model regime.  Writes pagano_S<S>.dat, pagano_S<S>.png, rm_8x8.out here.

Variables: the two sys.path lines put examples/garg/ on the path;
run_benchmark(case, S_list, tag) does everything (see pagano_bench.run_case for
the full variable glossary).

Run:  python examples/garg/caseA/7_helper_RM_Pagano_benchmark1.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pagano_bench import run_benchmark

run_benchmark("caseA", (4, 10, 50), tag="1")
