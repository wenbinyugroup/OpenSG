"""gpu_vs_direct.py -- the OpenSG solver benchmark: every SG yaml given
on the command line is homogenized with each requested solver, and the
laws are compared entry by entry against the FIRST solver's result (the
digits contract: direct is exact, iter's error enters the stiffness at
second order of the CG tolerance).

Runs anywhere the package imports -- the Purdue server (CPU: direct
pardiso / superlu) and a Colab GPU runtime (jax-cuda: iter cg on the
device).  The same file is the server baseline and the GPU comparison,
so the two tables are produced by identical code.

    python gpu_vs_direct.py case1.yaml case2.yaml --solvers direct,cg
    python gpu_vs_direct.py *.yaml --solvers cg --repeat 2

In:  yaml paths (positional); --solvers comma list of
     direct|pardiso|superlu|cg|iter1|stream|iter3 (default direct,cg);
     --repeat N timings per case (default 1, min reported); --out PATH
     for the .dat table (default ./solver_benchmark.dat)
Out: the printed table + the .dat file; per case it also states the
     device jax is using, wall time, peak host RSS and the law diff.
"""
import argparse
import datetime
import os
import resource
import time

import numpy as np


def law8(r):
    """The homogenization result dict -> the 8x8 (or 6x6) law array."""
    for k in ("ABDG", "law_matrix", "C_eff"):
        v = r.get(k)
        if v is not None:
            return np.asarray(v, float)
    raise SystemExit("no law found in the result dict")


p = argparse.ArgumentParser()
p.add_argument("yaml", nargs="+")
p.add_argument("--solvers", default="direct,cg")
p.add_argument("--repeat", type=int, default=1)
p.add_argument("--out", default="solver_benchmark.dat")
a = p.parse_args()
solvers = [s.strip() for s in a.solvers.split(",") if s.strip()]

import jax                                            # noqa: E402
from opensg_solid.sg_homo import plate_homo_2d        # noqa: E402

dev = jax.devices()
print("start : %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("jax devices: %s   (x64 %s)"
      % (", ".join(str(d) for d in dev), jax.config.jax_enable_x64))

rows = []
for y in a.yaml:
    base = {}
    print("\n=== %s" % os.path.basename(y))
    for s in solvers:
        best, r = None, None
        for _ in range(max(1, a.repeat)):
            t0 = time.perf_counter()
            try:
                r = plate_homo_2d(y, solver=s, plot=False, recovery=False)
            except Exception as exc:                  # OOM / unwired
                print("  %-8s FAILED: %s" % (s, str(exc)[:90]))
                r = None
                break
            dt = time.perf_counter() - t0
            best = dt if best is None else min(best, dt)
        if r is None:
            rows.append((os.path.basename(y), s, np.nan, np.nan, np.nan))
            continue
        L = law8(r)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        if not base:
            base["L"], base["s"] = L, s
            d = 0.0
        else:
            d = float(np.abs(L - base["L"]).max()
                      / max(np.abs(base["L"]).max(), 1e-30))
        print("  %-8s %8.1f s   peak %5.1f GB   rel diff vs %s: %.2e"
              % (s, best, rss, base["s"], d))
        rows.append((os.path.basename(y), s, best, rss, d))

with open(a.out, "w") as f:
    f.write("# OpenSG solver benchmark -- %s\n"
            % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    f.write("# jax devices: %s\n" % ", ".join(str(d) for d in dev))
    f.write("# case  solver  wall_s  peak_GB  rel_law_diff_vs_first\n")
    for c, s, t, m, d in rows:
        f.write("%-36s %-8s %10.2f %8.2f %12.3e\n" % (c, s, t, m, d))
print("\nwrote %s" % a.out)
print("end   : %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
