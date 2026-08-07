"""Compare the two .out results -> square_solid_compare.dat.

Reads square_solid_new_architecture.out and square_solid_old_architecture.out
and writes a term-by-term table: the resolved 6x6 entries (near-zero entries
are dropped) and the 9 engineering constants, each with the percent difference
of the new architecture against the old.

Run (from this folder):  python compare_architectures.py
"""
import numpy as np

############### User Input #################################
NEW = "square_solid_new_architecture.out"
OLD = "square_solid_old_architecture.out"
OUT = "square_solid_compare.dat"
RTOL = 1e-6                    # |C| below RTOL*max|C| counts as unresolved
                               # (the zero entries here sit at ~1e-7 of the peak)
############################################################


def read_out(path):
    C, cons, mode = [], {}, None
    for ln in open(path):
        if ln.startswith("# ---- C_eff"):
            mode = "C"; continue
        if ln.startswith("# ---- 9"):
            mode = "k"; continue
        if ln.startswith("#"):
            continue
        p = ln.split()
        if not p:
            continue
        if mode == "C":
            C.append([float(v) for v in p])
        elif mode == "k":
            cons[p[0]] = float(p[1])
    return np.array(C), cons


Cn, kn = read_out(NEW)
Co, ko = read_out(OLD)
thr = RTOL * max(np.max(np.abs(Cn)), np.max(np.abs(Co)))

lines = []
for i in range(6):
    for j in range(i, 6):
        a, b = Cn[i, j], Co[i, j]
        if abs(a) > thr or abs(b) > thr:
            pct = 100.0*(a-b)/b if abs(b) > thr else float("inf")
            lines.append("  C%d%d   %16.8e %16.8e %+10.4f" % (i+1, j+1, a, b, pct))

krows = []
for k in ("E1", "E2", "E3", "G23", "G13", "G12", "nu12", "nu13", "nu23"):
    a, b = kn[k], ko[k]
    krows.append("  %-5s %16.8e %16.8e %+10.4f" % (k, a, b, 100.0*(a-b)/b))

with open(OUT, "w") as f:
    f.write("# square thin-walled section, equivalent 3-D solid properties\n")
    f.write("# NEW = %s   (opensg_solid.sg_homo, n_model=3)\n" % NEW)
    f.write("# OLD = %s   (JAX_BICGoptimize single script, n_model=3)\n" % OLD)
    f.write("# same mesh, same material, unrotated ply; pct = 100*(new-old)/old\n")
    f.write("# order [e11 e22 e33 2e23 2e13 2e12]; resolved terms only"
            " (|C| > %.0e * max|C|)\n" % RTOL)
    f.write("# ---- stiffness terms (Pa) ----\n")
    f.write("# %-5s %16s %16s %10s\n" % ("term", "new", "old", "pct"))
    f.write("\n".join(lines) + "\n")
    f.write("# ---- 9 effective constants ----\n")
    f.write("# %-5s %16s %16s %10s\n" % ("const", "new", "old", "pct"))
    f.write("\n".join(krows) + "\n")

print("resolved stiffness terms, new vs old:")
print("  %-5s %16s %16s %10s" % ("term", "new", "old", "pct"))
print("\n".join(lines))
print("9 effective constants:")
print("  %-5s %16s %16s %10s" % ("const", "new", "old", "pct"))
print("\n".join(krows))
print("\nwrote %s" % OUT)
