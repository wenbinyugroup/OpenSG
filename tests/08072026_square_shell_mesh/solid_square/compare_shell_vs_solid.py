"""Compare the equivalent 3-D solid properties from the two SG routes.

  msg_shell : square_solid_msg_shell.out   (1-D shell SG, periodic)
  msg_solid : square_solid_newarch.out     (2-D solid SG, n_model = 3, periodic,
                                            per-element frames with the -45 ply)

Both are periodic and both are normalized by the wall material area 0.12, so
the entries are directly comparable.  Writes square_solid_shell_vs_solid.dat.

Run (from this folder):  python compare_shell_vs_solid.py
"""
import numpy as np

############### User Input #################################
SHELL = "square_solid_msg_shell.out"
SOLID = "square_solid_newarch.out"
OUT = "square_solid_shell_vs_solid.dat"
RTOL = 1e-6                    # |C| below RTOL*max|C| counts as unresolved
############################################################

KEYS = ("E1", "E2", "E3", "G23", "G13", "G12", "nu12", "nu13", "nu23")


def read_out(path):
    C, cons, mode = [], {}, None
    for ln in open(path):
        if ln.startswith("# ---- C"):
            mode = "C"; continue
        if ln.startswith("# ---- 9"):
            mode = "k"; continue
        if ln.startswith("#"):
            continue
        p = ln.split()
        if not p:
            continue
        if mode == "C" and len(p) == 6:
            C.append([float(v) for v in p])
        elif mode == "k" and len(p) == 2:
            cons[p[0]] = float(p[1])
    return np.array(C), cons


Csh, ksh = read_out(SHELL)
Cso, kso = read_out(SOLID)
thr = RTOL * max(np.max(np.abs(Csh)), np.max(np.abs(Cso)))

lines = []
for i in range(6):
    for j in range(i, 6):
        a, b = Csh[i, j], Cso[i, j]
        if abs(a) > thr or abs(b) > thr:
            pct = 100.0*(a-b)/b if abs(b) > thr else float("inf")
            lines.append("  C%d%d   %16.8e %16.8e %+10.3f" % (i+1, j+1, a, b, pct))

krows = []
for k in KEYS:
    a, b = ksh[k], kso[k]
    pct = 100.0*(a-b)/b if abs(b) > 1e-30 else float("inf")
    krows.append("  %-5s %16.8e %16.8e %+10.3f" % (k, a, b, pct))

with open(OUT, "w") as f:
    f.write("# square thin-walled section -- equivalent 3-D solid properties\n")
    f.write("# SHELL = %s  (1-D shell SG, msg_shell build_solid_bundle)\n" % SHELL)
    f.write("# SOLID = %s  (2-D solid SG, opensg_solid n_model=3)\n" % SOLID)
    f.write("# both PERIODIC, both normalized by the wall material area 0.12\n")
    f.write("# pct = 100*(shell-solid)/solid; order [e11 e22 e33 2e23 2e13 2e12]\n")
    f.write("# resolved terms only (|C| > %.0e * max|C|)\n" % RTOL)
    f.write("# ---- stiffness terms (Pa) ----\n")
    f.write("# %-5s %16s %16s %10s\n" % ("term", "shell", "solid", "pct"))
    f.write("\n".join(lines) + "\n")
    f.write("# ---- 9 effective constants ----\n")
    f.write("# %-5s %16s %16s %10s\n" % ("const", "shell", "solid", "pct"))
    f.write("\n".join(krows) + "\n")

print("resolved stiffness terms, shell vs solid:")
print("  %-5s %16s %16s %10s" % ("term", "shell", "solid", "pct"))
print("\n".join(lines))
print("9 effective constants:")
print("  %-5s %16s %16s %10s" % ("const", "shell", "solid", "pct"))
print("\n".join(krows))
print("\nwrote %s" % OUT)
