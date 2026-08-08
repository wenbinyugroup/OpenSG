"""Four-way TPMS table: the Schwarz-P shell SG (at each measured thickness),
the previous-try solid, and Sample_1 / Sample_2 — geometry measured from each
mesh plus the resulting per-unit-cell stiffness.

Thickness convention: t = V / A_mid with A_mid the TRUE midsurface area.
(t = 2V/S_free over-reads on thick sheets: on a minimal surface both offset
faces have less area than the midsurface.)

Run (from this folder):  python tpms_all_compare.py"""
import numpy as np

A_SHELL = 2.3533              # Schwarz-P shell mesh midsurface area
SH = "../../OpenSG_shell/4_get_solid_props_from_shell_3D_SG"

# name, path, per-cell scale, rho, t, A_mid, surface
CASES = [
    ("Sample_1 solid (Schwarz-P)", "sample_1/Sample_1_msg_solid.out",
     1.0, 0.3000, 0.12748, 2.3533, "Schwarz-P"),
    ("Schwarz-P shell  t=0.1293", SH + "/schwarz_t0.129330.out",
     1.0, 0.3000, 0.12933, 2.3533, "Schwarz-P"),
    ("Sample_2 solid (Schwarz-P)", "sample_2/Sample_2_msg_solid.out",
     1.0, 0.0857, 0.03641, 2.3533, "Schwarz-P"),
    ("Schwarz-P shell  t=0.0364", SH + "/schwarz_t0.036414.out",
     1.0, 0.0857, 0.03641, 2.3533, "Schwarz-P"),
    ("previous try solid", "previous_try/preovios_try_msg_solid.out",
     1.0, 0.1000, 0.03139, 3.1859, "Neovius-class"),
]


def read_C(path):
    C = []
    for ln in open(path):
        try:
            row = [float(v) for v in ln.split()]
        except ValueError:
            continue
        if len(row) == 6:
            C.append(row)
        if len(C) == 6:
            break
    return np.array(C)


rows = ["# TPMS equivalent 3-D solid properties (alu E=69 GPa nu=0.3,"
        " periodic in all 3 directions)",
        "# thickness measured from each mesh: t = V / A_mid;"
        " per-cell C in MPa",
        "# %-27s %-13s %6s %8s %7s %9s %9s %9s"
        % ("case", "surface", "rho", "t", "A_mid", "C11", "C12", "C44")]
C = {}
for nm, p, scl, rho, t, am, surf in CASES:
    c = read_C(p)*1e-6*scl
    C[nm] = c
    rows.append("  %-27s %-13s %6.4f %8.5f %7.4f %9.1f %9.1f %9.1f"
                % (nm, surf, rho, t, am, c[0, 0], c[0, 1], c[3, 3]))

rows.append("")
rows.append("# shell vs solid at MATCHED thickness (same Schwarz-P surface)")
rows.append("# Kbar = -10.237 -> R = 1/sqrt|Kbar| = 0.3126;"
            "  t/R = 0.117 (thin) and 0.414 (thick)")
rows.append("# %-6s %11s %11s %8s | %11s %11s %8s"
            % ("term", "sh t=.0364", "solid S2", "%", "sh t=.1293",
               "solid S1", "%"))
IJ = [("C11", 0, 0), ("C22", 1, 1), ("C33", 2, 2), ("C12", 0, 1),
      ("C13", 0, 2), ("C23", 1, 2), ("C44", 3, 3), ("C55", 4, 4),
      ("C66", 5, 5)]
for nm, i, j in IJ:
    a = C["Schwarz-P shell  t=0.0364"][i, j]
    b = C["Sample_2 solid (Schwarz-P)"][i, j]
    c2 = C["Schwarz-P shell  t=0.1293"][i, j]
    d = C["Sample_1 solid (Schwarz-P)"][i, j]
    rows.append("  %-6s %11.1f %11.1f %+8.2f | %11.1f %11.1f %+8.2f"
                % (nm, a, b, 100*(a-b)/b, c2, d, 100*(c2-d)/d))

open("tpms_all_compare.dat", "w").write("\n".join(rows) + "\n")
print("\n".join(rows))
