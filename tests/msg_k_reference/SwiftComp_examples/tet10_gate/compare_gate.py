"""compare_gate.py -- score the six SwiftComp `.sc.k` results against the
OpenSG `.out` twins and the exact one-material C.  Run after the laptop
SwiftComp runs have landed the `.k` files here."""
import os
import re
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
E1, NU1, RHO1 = 70000.0, 0.30, 2700.0

print("start", time.strftime("%Y-%m-%d %H:%M:%S"))


def six_by_six(path, title):
    L = open(path).read().splitlines()
    i = next(k for k, ln in enumerate(L) if title in ln)
    rows = [ln for ln in L[i + 1: i + 12] if re.match(r"^\s*[-+0-9.]", ln)]
    return np.array([[float(v) for v in ln.split()] for ln in rows[:6]])


def density(path):
    for ln in open(path):
        if "density" in ln.lower():
            nums = re.findall(r"[-+0-9.]+E[-+]\d+|[-+]?\d+\.\d+", ln)
            if nums:
                return float(nums[-1]), ln.strip()
    return None, None


lam, mu = E1 * NU1 / ((1 + NU1) * (1 - 2 * NU1)), E1 / (2 * (1 + NU1))
Cexact = np.array([[lam + 2 * mu, lam, lam, 0, 0, 0],
                   [lam, lam + 2 * mu, lam, 0, 0, 0],
                   [lam, lam, lam + 2 * mu, 0, 0, 0],
                   [0, 0, 0, mu, 0, 0], [0, 0, 0, 0, mu, 0],
                   [0, 0, 0, 0, 0, mu]])

for case in ("iso", "bilayer"):
    ref = six_by_six(os.path.join(HERE, "cube_%s.out" % case),
                     "Effective Cauchy Continuum Stiffness Matrix")
    print("\n=== %s ===  OpenSG diag: %s" % (case, np.array2string(
        np.diag(ref), precision=6)))
    if case == "iso":
        print("  OpenSG vs exact  max rel dev %.2e"
              % (np.abs(ref - Cexact).max() / np.abs(Cexact).max()))
    for L in "ABC":
        k = os.path.join(HERE, "cube_%s_%s.sc.k" % (case, L))
        if not os.path.exists(k) or os.path.getsize(k) == 0:
            print("  layout %s: NO .k (SwiftComp produced nothing)" % L)
            continue
        try:
            sc = six_by_six(k, "Effective Stiffness Matrix")
        except Exception as e:                       # noqa: BLE001
            print("  layout %s: .k unreadable (%s)" % (L, e))
            continue
        dev = np.abs(sc - ref).max() / np.abs(ref).max()
        line = "  layout %s: SwiftComp vs OpenSG max rel dev %.2e" % (L, dev)
        if case == "iso":
            line += " | vs exact %.2e" % (np.abs(sc - Cexact).max()
                                          / np.abs(Cexact).max())
            rho, txt = density(k)
            line += " | density line: %r" % txt
        print(line)
        print("     diag:", np.array2string(np.diag(sc), precision=6))

print("\nend", time.strftime("%Y-%m-%d %H:%M:%S"))
