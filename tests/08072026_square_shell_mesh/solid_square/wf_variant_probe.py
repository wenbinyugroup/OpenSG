"""wf_variant_probe.py -- Task C decisive experiment (scratch, wf_).

Scales the Gamma_e transverse-shear rows (VAR_GE_SHEAR) and/or the Gamma_h
transverse-shear rows (VAR_GH_SHEAR) in opensg_shell.wf_solid_props_var and
recomputes the iso CROSS-cell C3D for each variant.

Exact targets (iso, E' = E/(1-nu^2), G = E/(2(1+nu)), per cell area L^2):
    C11 = 2*E'*t/L        (two walls of length L carry eps11 = G11)
    C22 = C33 = E'*t/L    (one wall in-line, the crossing wall relaxes)
    C12 = C13 = nu*E'*t/L
    C44 = 0.5*E'*(t/L)^3  (relaxed in-plane shear -- the channel under test)
    C55 = C66 = G*t/L
The (1,1) baseline is KNOWN exact (ratio 1.0000) on every channel except C44
(3.98-4.00x).  If a closed form above disagrees with the baseline on a
non-shear channel, the baseline numerics are the reference there; hence every
variant row is printed BOTH as ratio-to-exact and ratio-to-baseline.

Variants (VAR_GE_SHEAR, VAR_GH_SHEAR): (1,1), (0.5,1), (1,2), (0.5,2).
Thicknesses: t = 0.0125 (thin) and t = 0.05 (moderately thick).

Run (from this folder):  python wf_variant_probe.py
"""
import numpy as np

import opensg_shell.wf_solid_props_var as spv
from opensg_shell.periodic_multiscale import mesh_to_periodic_sparse_assembly_map

############### User Input #################################
L = 1.0
E_iso, nu_iso = 70.0e9, 0.30
T_SWEEP = [0.0125, 0.05]
nseg = 40                       # shell elems/wall
############################################################

G_iso = E_iso/(2*(1+nu_iso))
Ep = E_iso/(1-nu_iso**2)
c = L/2


def shell_C(t_w):
    # verbatim from crosscell_thin_sweep.py, except: yaml filename is
    # wf_cross_var.yaml and build_solid_bundle comes from the VARIANT module.
    xs = np.linspace(0.0, L, nseg+1)
    h_n = np.stack([xs, np.full(nseg+1, c)], 1)
    v_n = np.stack([np.full(nseg+1, c), xs], 1)
    ic = nseg//2
    keep = [j for j in range(nseg+1) if j != ic]
    pts = np.vstack([h_n, v_n[keep]])
    vid, k = {}, len(h_n)
    for j in range(nseg+1):
        if j == ic:
            vid[j] = ic
        else:
            vid[j] = k; k += 1
    cells = ([[i, i+1] for i in range(nseg)]
             + [[vid[j], vid[j+1]] for j in range(nseg)])
    ori = ([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]*nseg
           + [[0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0]]*nseg)
    o = ["nodes:"]
    for x, y in pts:
        o.append("- [%.12f %.12f 0.00000000]" % (x, y))
    o.append("elements:")
    for a_, b_ in cells:
        o.append("- [%d %d]" % (a_+1, b_+1))
    o.append("elementOrientations:")
    for r in ori:
        o.append("- [" + ", ".join("%.1f" % v for v in r) + "]")
    o += ["sections:", "- type: shell", "  elementSet: layup_0", "  layup:",
          "  - - iso", "    - %.6e" % t_w, "    - 0.0",
          "materials:", "- name: iso", "  density: 2700.0", "  elastic:",
          "    E: [%.6e, %.6e, %.6e]" % (E_iso, E_iso, E_iso),
          "    G: [%.6e, %.6e, %.6e]" % (G_iso, G_iso, G_iso),
          "    nu: [%.6f, %.6f, %.6f]" % (nu_iso, nu_iso, nu_iso),
          "sets:", "  element:", "  - name: layup_0", "    labels:"]
    o += ["    - %d" % (e+1) for e in range(len(cells))]
    o.append("reference: center")
    open("wf_cross_var.yaml", "w").write("\n".join(o) + "\n")
    return np.asarray(spv.build_solid_bundle("wf_cross_var.yaml",
                                             cell_area=L*L)["C3D"])


CH = [("C11", 0, 0), ("C22", 1, 1), ("C33", 2, 2), ("C12", 0, 1),
      ("C13", 0, 2), ("C44", 3, 3), ("C55", 4, 4), ("C66", 5, 5)]
VARIANTS = [(1.0, 1.0), (0.5, 1.0), (1.0, 2.0), (0.5, 2.0)]


def exact_forms(t_w):
    return {"C11": 2*Ep*t_w/L, "C22": Ep*t_w/L, "C33": Ep*t_w/L,
            "C12": nu_iso*Ep*t_w/L, "C13": nu_iso*Ep*t_w/L,
            "C44": 0.5*Ep*(t_w/L)**3, "C55": G_iso*t_w/L, "C66": G_iso*t_w/L}


for t_w in T_SWEEP:
    ex = exact_forms(t_w)
    print("=" * 110)
    print("t = %.4f  (t/L = %.4f)   C44_exact = %.6e   [E=%.3e nu=%.2f "
          "nseg=%d L=%g]" % (t_w, t_w/L, ex["C44"], E_iso, nu_iso, nseg, L))
    print("-" * 110)
    hdr = ("  %-10s %-9s" % ("(GE,GH)", "ref")) + \
        "".join("%11s" % n for n, _, _ in CH)
    print(hdr)
    base = None
    for ge, gh in VARIANTS:
        spv.VAR_GE_SHEAR = ge
        spv.VAR_GH_SHEAR = gh
        C = shell_C(t_w)
        vals = np.array([C[i, j] for _, i, j in CH])
        if base is None:
            base = vals.copy()
        r_ex = vals / np.array([ex[n] for n, _, _ in CH])
        r_ba = vals / base
        print(("  (%.1f,%.1f)  %-9s" % (ge, gh, "exact"))
              + "".join("%11.4f" % v for v in r_ex))
        print(("  (%.1f,%.1f)  %-9s" % (ge, gh, "baseline"))
              + "".join("%11.4f" % v for v in r_ba))
    print()
print("done.")
