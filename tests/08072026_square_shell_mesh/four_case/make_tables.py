"""Assemble the four stiffness tables and the four 9-constant tables from the
.out files.

Columns: msg_shell (A) | msg_solid new (B) | msg_solid old (C) | analytical
(ISO only) | +45 probe (M45 only), with percent differences of every column
against msg_solid new (B), which is the reference.

Run:  ~/miniconda3/envs/opensg_2_0/bin/python make_tables.py
"""
import json
import os

import numpy as np

L, CELL = 1.0, 1.0
E_ISO, NU_ISO = 70.0e9, 0.30
EP = E_ISO/(1-NU_ISO**2)                       # E' = E/(1-nu^2)
CASES = ["thick_iso", "thin_iso", "thick_m45", "thin_m45"]
NAMES = {"thick_iso": "THICK / ISO  (t/L = 0.10, E = 70 GPa, nu = 0.30)",
         "thin_iso": "THIN / ISO  (t/L = 0.01, E = 70 GPa, nu = 0.30)",
         "thick_m45": "THICK / M45  (t/L = 0.10, orthotropic, single ply -45 deg)",
         "thin_m45": "THIN / M45  (t/L = 0.01, orthotropic, single ply -45 deg)"}
WANT = [("C11", 0, 0), ("C12", 0, 1), ("C13", 0, 2), ("C22", 1, 1),
        ("C23", 1, 2), ("C33", 2, 2), ("C44", 3, 3), ("C55", 4, 4),
        ("C66", 5, 5)]
KEYS = ("E1", "E2", "E3", "G23", "G13", "G12", "nu12", "nu13", "nu23")
meta = json.load(open("inputs_meta.json"))


def readC(path):
    if not os.path.exists(path):
        return None, None
    rows, cons = [], {}
    for ln in open(path):
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        p = s.split()
        if len(p) == 6:
            rows.append([float(v) for v in p])
        elif len(p) == 2:
            cons[p[0]] = float(p[1])
    C = np.array(rows[:6], float)
    return 0.5*(C + C.T), cons


def pct(x, ref):
    if x is None or ref is None or ref == 0.0:
        return "     --   "
    return "%+9.3f%%" % (100.0*(x-ref)/abs(ref))


def fmt(x):
    return "     n/a       " if x is None else "%16.8e" % x


out_all = []
for case in CASES:
    m = meta[case]
    t, iso = m["t"], m["material"] == "iso"
    rho = (2*L*t - t*t)/(L*L)
    A, consA = readC("%s_shell.out" % case)
    B, consB = readC("%s_solidnew.out" % case)
    C, consC = readC("%s_solidold.out" % case)
    P, consP = readC("%s_solidnew_ang+45.out" % case)      # M45 probe only

    # ---- analytical anchors (ISO only) ---------------------------------
    an = {}
    if iso:
        an["C11"] = rho*E_ISO                 # rho_bar * E
        an["C44"] = 0.5*EP*(t/L)**3           # slender limit
    extra_col = ("analytical" if iso else "msg_solid new +45")
    X, consX = ((None, None) if iso else (P, consP))

    thr = 1e-6*max(np.max(np.abs(A)), np.max(np.abs(B)))

    o = []
    o.append("# " + "="*100)
    o.append("# EQUIVALENT 3-D SOLID STIFFNESS -- square-lattice CROSS cell, %s"
             % NAMES[case])
    o.append("# " + "="*100)
    o.append("# geometry : unit cell L = %.1f, one wall of each family running"
             " THROUGH the cell," % L)
    o.append("#            thickness t = %.4f (t/L = %.4f); wall material area"
             " 2Lt - t^2 = %.6f" % (t, t/L, m["area_exact"]))
    o.append("# NORMALIZATION : every column is per CELL area L^2 = %.4f"
             " (the void included)" % (L*L))
    o.append("# order : [e11 e22 e33 2e23 2e13 2e12], axis 1 = the prismatic"
             " (out-of-plane) direction")
    o.append("# routes : A msg_shell  1-D shell SG (opensg_shell.build_solid_bundle,"
             " periodic, %d elem/wall)" % m["nseg"])
    o.append("#          B msg_solid NEW (opensg_solid.sg_homo.plate_homo_2d,"
             " n_model=3) -- the REFERENCE")
    o.append("#          C msg_solid OLD (JAX_BICGoptimize single script,"
             " n_model=3), reads the .sc")
    o.append("# solid mesh : %d nodes / %d linear triangles, nt=%d through t,"
             " na=%d per arm (aspect 1.5)"
             % (m["solid_nodes"], m["solid_elems"], m["nt"], m["na"]))
    if iso:
        o.append("# analytical anchors (ISOTROPIC ONLY):")
        o.append("#   C11 = rho_bar * E            = %.8e   with rho_bar ="
                 " (2Lt-t^2)/L^2 = %.6f" % (an["C11"], rho))
        o.append("#   C44 = 0.5 * E/(1-nu^2) * (t/L)^3 = %.8e   (slender limit)"
                 % an["C44"])
        o.append("#   CAVEAT: rho_bar*E is the UNIAXIAL-STRESS anchor -- it is"
                 " the effective E1, not the")
        o.append("#   fully-constrained C11.  A wall in a shell/plane-stress"
                 " state carries E/(1-nu^2), so")
        o.append("#   the constrained C11 of this cell is ~ 2Lt/L^2 * E/(1-nu^2)"
                 " = %.4e.  The E1 row of the" % (2*L*t/(L*L)*EP))
        o.append("#   companion constants table is where rho_bar*E belongs, and"
                 " it is matched there.")
    else:
        o.append("# COLUMN C IS n/a FOR THIS CASE.  A .sc carries ONE material id"
                 " per element and the old")
        o.append("# driver applies ONE fibre angle per material, so the"
                 " per-element material FRAMES this")
        o.append("# cell needs (the two wall families differ by a 90 deg rotation"
                 " ABOUT the beam axis,")
        o.append("# which no in-plane fibre angle can produce) cannot be"
                 " represented.  It is left n/a")
        o.append("# rather than faked with a single global angle.")
        o.append("# The last column is route B re-run at angles = +45: the shell"
                 " layup angle and the solid")
        o.append("# rotate_C_matrix use OPPOSITE senses, so -45 in the two codes"
                 " is not the same laminate.")
        o.append("# It flips the sign of every extension-shear coupling and"
                 " leaves the diagonal alone.")
    o.append("#")
    o.append("%-6s %16s %16s %16s %16s   %10s %10s %10s"
             % ("term", "A msg_shell", "B msg_solid new", "C msg_solid old",
                extra_col, "A vs B", "C vs B", extra_col[:10] + " vs B"))
    o.append("#" + "-"*116)
    for k, i, j in WANT:
        a = float(A[i, j]) if A is not None else None
        b = float(B[i, j]) if B is not None else None
        c = float(C[i, j]) if C is not None else None
        x = an.get(k) if iso else (float(X[i, j]) if X is not None else None)
        if max(abs(v) for v in (a or 0.0, b or 0.0)) <= thr:
            continue                                   # numerical zero, dropped
        o.append("%-6s %s %s %s %s   %s %s %s"
                 % (k, fmt(a), fmt(b), fmt(c), fmt(x),
                    pct(a, b), pct(c, b), pct(x, b)))
    o.append("#" + "-"*116)
    o.append("# dropped as numerical zeros: |C_ij| <= 1e-6*max|C| = %.3e" % thr)
    ex = [(i+1, j+1) for i in range(6) for j in range(i, 6)
          if abs(B[i, j]) > thr and (i, j) not in [(w[1], w[2]) for w in WANT]]
    if ex:
        o.append("# ADDITIONAL above-threshold terms present in this cell"
                 " (not in the 9-term list):")
        for i, j in ex:
            a = float(A[i-1, j-1]); b = float(B[i-1, j-1])
            x = float(X[i-1, j-1]) if X is not None else None
            o.append("C%d%d    %s %s %s %s   %s %s %s"
                     % (i, j, fmt(a), fmt(b), fmt(None), fmt(x),
                        pct(a, b), "     --   ", pct(x, b)))
    txt = "\n".join(o) + "\n"
    open("table_stiffness_%s.dat" % case, "w").write(txt)
    out_all.append(txt)

    # ------------------------------------------------- 9 effective constants --
    g = []
    g.append("# " + "="*100)
    g.append("# 9 EFFECTIVE CONSTANTS -- square-lattice CROSS cell, %s"
             % NAMES[case])
    g.append("# " + "="*100)
    g.append("# from the 6x6 above (per CELL area L^2 = %.4f): S = C^-1,"
             " E_i = 1/S_ii, nu_ij = -S_ij/S_ii" % (L*L))
    if iso:
        g.append("# analytical anchor: E1 = rho_bar * E = %.8e"
                 " (rho_bar = %.6f)" % (rho*E_ISO, rho))
        g.append("#                    G23 = 0.5*E/(1-nu^2)*(t/L)^3 = %.8e"
                 " (slender limit; = C44 here since C44 is uncoupled)"
                 % (0.5*EP*(t/L)**3))
    else:
        g.append("# column C n/a (per-element frames not representable in a .sc)"
                 " -- see the stiffness table")
    g.append("#")
    g.append("%-6s %16s %16s %16s %16s   %10s %10s %10s"
             % ("const", "A msg_shell", "B msg_solid new", "C msg_solid old",
                extra_col, "A vs B", "C vs B", extra_col[:10] + " vs B"))
    g.append("#" + "-"*116)
    anc = {"E1": rho*E_ISO, "G23": 0.5*EP*(t/L)**3} if iso else {}
    for k in KEYS:
        a = consA.get(k) if consA else None
        b = consB.get(k) if consB else None
        c = consC.get(k) if consC else None
        x = anc.get(k) if iso else (consP.get(k) if consP else None)
        g.append("%-6s %s %s %s %s   %s %s %s"
                 % (k, fmt(a), fmt(b), fmt(c), fmt(x),
                    pct(a, b), pct(c, b), pct(x, b)))
    txt = "\n".join(g) + "\n"
    open("table_constants_%s.dat" % case, "w").write(txt)
    out_all.append(txt)

print("\n".join(out_all))
print("wrote table_stiffness_<case>.dat and table_constants_<case>.dat"
      " for %s" % ", ".join(CASES))
