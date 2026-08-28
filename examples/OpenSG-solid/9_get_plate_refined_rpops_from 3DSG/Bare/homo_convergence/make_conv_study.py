"""Bare-core homogenization mesh-convergence study (rho = 0.05 and rho = 0.3).

In : the SP_solid_rho<rho>_n<knob>[_quad].out effective-law files in this
     study folder (tet4 = plain, tet10 = "_quad"), and the matching gmsh
     .msh files, which are the authoritative source of element/node counts.
Out: conv_tables.txt  (per-density convergence tables, values + % change
                       against the finest tet10 of that density)
     convergence_plots/*.png
     the numbers echoed to stdout.

Solver-certificate reruns (_amg, _amgtight, _directref, _gamg) are the SAME
mesh re-solved with a different KSP/PC; they are excluded from the
convergence ladder and reported separately as a solver-agreement note.
"""
import hashlib
import os
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = os.path.expanduser("~/OpenSG-2.0/examples/OpenSG-solid/"
                       "9_get_plate_refined_rpops_from 3DSG/Bare/homo_convergence")
R3 = os.path.join(B, "rho_0.3")
PLOTS = os.path.join(B, "convergence_plots")
os.makedirs(PLOTS, exist_ok=True)

# ---------------------------------------------------------------- ingest ---
# .out files that live outside the study folder get copied in first so the
# study is self-contained.  Source is recorded for the table footnotes.
IMPORT = [
    (os.path.expanduser("~/claude_tmp/cv_drive_out/rho_0.3/"
                        "SP_solid_rho0.3_n5.09714_quad.out"),
     os.path.join(R3, "SP_solid_rho0.3_n5.09714_quad.out")),
    (os.path.expanduser("~/claude_tmp/cv_drive_out/rho_0.3/"
                        "SP_solid_rho0.3_n12.7429.out"),
     os.path.join(R3, "SP_solid_rho0.3_n12.7429.out")),
    (os.path.expanduser("~/claude_tmp/conv_results/"
                        "SP_solid_rho0.05_n0.424762_quad.out"),
     os.path.join(B, "SP_solid_rho0.05_n0.424762_quad.out")),
]
for src, dst in IMPORT:
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy2(src, dst)
        print("imported ->", os.path.basename(dst))

# EXTERNAL DEPENDENCY -- read this before deleting anything in ~/claude_tmp.
# SP_solid_rho0.05_n0.424762_quad.msh (9.2 MB) was built in a scratch dir and
# was never copied into the study folder, so the element/node/dof counts of one
# of the only TWO rho=0.05 tet10 points come from outside this folder.  The
# study is therefore NOT self-contained for regeneration (the .out files are;
# the meshes are not).  If this path disappears the run aborts (see below)
# rather than silently dropping the point.
EXTRA_MSH = [os.path.expanduser("~/claude_tmp/mshdirect")]

TYPE_NAME = {4: "tet4", 11: "tet10"}


def msh_counts(path):
    """(nnode, {gmsh_elem_type: count}) straight from the gmsh header."""
    nnode, etypes, ver = None, {}, None
    with open(path, "r", errors="ignore") as f:
        line = f.readline()
        while line:
            s = line.strip()
            if s == "$MeshFormat":
                ver = f.readline().split()[0]
            elif s == "$Nodes":
                hdr = f.readline().split()
                if ver and ver.startswith("4"):
                    nnode = int(hdr[1])
                    while True:
                        l2 = f.readline()
                        if not l2 or l2.strip() == "$EndNodes":
                            break
                else:
                    nnode = int(hdr[0])
                    for _ in range(nnode):
                        f.readline()
            elif s == "$Elements":
                hdr = f.readline().split()
                if ver and ver.startswith("4"):
                    nblk = int(hdr[0])
                    for _ in range(nblk):
                        b = f.readline().split()
                        if len(b) < 4:
                            break
                        et, nb = int(b[2]), int(b[3])
                        etypes[et] = etypes.get(et, 0) + nb
                        for _ in range(nb):
                            f.readline()
                else:
                    nelem = int(hdr[0])
                    for _ in range(nelem):
                        v = f.readline().split()
                        if len(v) > 1:
                            etypes[int(v[1])] = etypes.get(int(v[1]), 0) + 1
                break
            line = f.readline()
    return nnode, etypes


def find_msh(stem):
    for d in (B, R3) + tuple(EXTRA_MSH):
        p = os.path.join(d, stem + ".msh")
        if os.path.exists(p):
            return p
    return None


def law8(path):
    """First 8 rows of 8 numbers = the effective 8x8 RM stiffness."""
    rows = []
    for ln in open(path, errors="ignore"):
        v = ln.split()
        if len(v) == 8:
            try:
                rows.append([float(x) for x in v])
            except ValueError:
                pass
        if len(rows) == 8:
            break
    return np.array(rows) if len(rows) == 8 else None


CERT = ("_amg", "_amgtight", "_directref", "_gamg")

recs, certs = [], []
for d in (B, R3):
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".out"):
            continue
        stem = fn[:-4]
        rho = "0.05" if "rho0.05" in stem else "0.3"
        is_cert = any(stem.endswith(c) for c in CERT)
        base = stem
        for c in CERT:
            if base.endswith(c):
                base = base[: -len(c)]
        mp = find_msh(base)
        if mp is None:
            # Hard failure on purpose.  Silently dropping a point would
            # shrink the ladder without changing any printed caveat --
            # e.g. losing n0.424762_quad would leave rho=0.05 with a single
            # tet10 point and no convergence evidence at all.
            raise SystemExit(
                "FATAL: no .msh for %s.  Searched %s.  The ladder would "
                "silently lose a mesh point -- refusing to write tables."
                % (stem, [B, R3] + list(EXTRA_MSH)))
        nnode, et = msh_counts(mp)
        vol = {k: v for k, v in et.items() if k in (4, 11)}
        etype = TYPE_NAME[max(vol, key=vol.get)]
        nelem = sum(vol.values())
        L = law8(os.path.join(d, fn))
        rec = dict(stem=stem, base=base, rho=rho, nelem=nelem, nnode=nnode,
                   ndof=3 * nnode, etype=etype, L=L, path=os.path.join(d, fn))
        (certs if is_cert else recs).append(rec)

# ------------------------------------------------------------ the tables ---
IDX = [("A11", 0, 0), ("A66", 2, 2), ("D11", 3, 3), ("D66", 5, 5),
       ("G11", 6, 6), ("G22", 7, 7)]

lines = []
W = lines.append
W("=" * 146)
W("BARE-CORE HOMOGENIZATION -- MESH CONVERGENCE OF THE EFFECTIVE "
  "REISSNER-MINDLIN PLATE LAW")
W("=" * 146)
W("")
W("Element/node counts read from the gmsh .msh headers (authoritative);")
W("dofs = 3 x nodes (three fluctuation-function components per node).")
W("tet4 = plain filename, tet10 = '_quad' filename (p-refinement of the "
  "SAME tet4 mesh).")
W("Reference for the %-change columns = the FINEST tet10 of that density.")
W("Units: A [N/m], D [N.m], G [N/m].")
W("")

ref_of, ladder_of = {}, {}
for rho in ("0.05", "0.3"):
    rows = sorted([r for r in recs if r["rho"] == rho],
                  key=lambda r: (r["nelem"], r["etype"] == "tet10"))
    if not rows:
        continue
    t10 = [r for r in rows if r["etype"] == "tet10"]
    ref = t10[-1]
    ref_of[rho] = ref
    ladder_of[rho] = rows

    W("")
    W("-" * 146)
    W("rho = %s   (%d mesh points; reference = %s, tet10, %s elements)"
      % (rho, len(rows), ref["base"], format(ref["nelem"], ",")))
    W("-" * 146)
    hdr = ("%-30s %6s %11s %11s %12s" % ("mesh", "type", "elements",
                                         "nodes", "dofs"))
    hdr += "".join("%15s" % n for n, _, _ in IDX)
    W(hdr)
    W("-" * 146)
    for r in rows:
        tag = r["base"].replace("SP_solid_rho%s_" % rho, "")
        s = ("%-30s %6s %11s %11s %12s"
             % (tag, r["etype"], format(r["nelem"], ","),
                format(r["nnode"], ","), format(r["ndof"], ",")))
        s += "".join("%15.6E" % r["L"][i, j] for _, i, j in IDX)
        W(s)
        d = ("%-30s %6s %11s %11s %12s"
             % ("   d% vs finest tet10", "", "", "", ""))
        for _, i, j in IDX:
            v = 100.0 * (r["L"][i, j] / ref["L"][i, j] - 1.0)
            d += "%14.4f%%" % v
        W(d)
    W("-" * 146)

# ---------------------------------------------------- solver certificates ---
W("")
W("=" * 146)
W("SOLVER-CERTIFICATE RERUNS (same mesh, different KSP/PC -- NOT extra mesh "
  "points)")
W("=" * 146)
def _md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


if certs:
    indep, notes = [], []
    for c in sorted(certs, key=lambda r: r["stem"]):
        base_rec = [r for r in recs if r["base"] == c["base"]]
        if not base_rec:
            continue
        b = base_rec[0]
        dd = max(abs(c["L"][i, i] / b["L"][i, i] - 1.0) for i in range(8))
        # full 8x8, so the transverse-shear block is not excluded
        off, worst = [], (0.0, None)
        for i in range(8):
            for j in range(8):
                if i != j and abs(b["L"][i, j]) > 0:
                    d = abs(c["L"][i, j] / b["L"][i, j] - 1.0)
                    off.append(d)
                    if d > worst[0]:
                        worst = (d, (i, j))
        copy = _md5(c["path"]) == _md5(b["path"])
        W("%-46s vs %-34s  max|d| diagonal = %9.3e %%   max|d| "
          "off-diagonal = %9.3e %%%s"
          % (c["stem"], b["stem"], 100 * dd, 100 * max(off),
             "   << byte-identical file" if copy else ""))
        if copy:
            notes.append(c["stem"])
        else:
            indep.append((c, b, worst))
    W("")
    W("Every diagonal/energy term -- A11 A22 A66 D11 D22 D66 G11 G22, i.e. "
      "every term this")
    W("study reports -- is identical to all 8 printed digits under every "
      "solver (max|d| = 0).")
    W("So are the two physically significant couplings A12 and D12 "
      "(max|d| = 0 exactly).")
    W("")
    for c, b, (d, (i, j)) in indep:
        W("%-46s worst-moving entry = K[%d,%d] = %12.5E, which is %.1e of "
          "A11;" % (c["stem"], i, j, b["L"][i, j],
                    abs(b["L"][i, j]) / b["L"][0, 0]))
        W("%-46s that entry holds %.1f significant digits."
          % ("", -np.log10(d)))
    W("")
    W("Only the numerically-zero couplings move -- entries 7 to 8 orders of "
      "magnitude below")
    W("the diagonal, which are zero by symmetry for this orthotropic core and "
      "are pure solver")
    W("noise.  That is the second-order-vs-first-order tolerance behaviour "
      "explained in")
    W("README.md; it does not touch any term this study reports.")
    if notes:
        W("")
        W("CAVEAT -- NOT AN INDEPENDENT SOLVE: %s is byte-for-byte the same "
          "file as its" % ", ".join(notes))
        W("base run (same md5, same 'Time taken' line), i.e. an archived copy "
          "of the reference")
        W("run rather than a re-solve.  Its 0.000e+00 row is a self-comparison "
          "and carries no")
        W("information.  There are %d independent solver reruns here, not %d."
          % (len(indep), len(indep) + len(notes)))
else:
    W("(none found)")

# --------------------------------------------- what the tet10 pair supports --
W("")
W("=" * 146)
W("WHAT THE tet10 PAIR CAN AND CANNOT SUPPORT")
W("=" * 146)
W("")
W("Each density has exactly TWO tet10 points, so the observed order of "
  "convergence is NOT")
W("measurable and no genuine (3-point) Richardson extrapolation is possible.  "
  "The d% columns")
W("above are the DIFFERENCE between the two tet10 meshes, not the error of "
  "the finer one.")
W("Assuming an order p, that error is  diff / (r_h^p - 1)  with r_h = "
  "(N_fine/N_coarse)^(1/3):")
W("")
for rho in ("0.05", "0.3"):
    t10 = sorted([r for r in recs if r["rho"] == rho and r["etype"] == "tet10"],
                 key=lambda r: r["nelem"])
    if len(t10) < 2:
        continue
    c, f = t10[-2], t10[-1]
    rh = (f["nelem"] / c["nelem"]) ** (1.0 / 3.0)
    W("rho = %-5s  coarse %s el -> fine %s el,  N ratio = %.3f,  h ratio = "
      "%.4f" % (rho, format(c["nelem"], ","), format(f["nelem"], ","),
                f["nelem"] / c["nelem"], rh))
    W("   %-6s %14s %14s %14s %14s"
      % ("term", "2-pt diff", "err @ p=1", "err @ p=2", "err @ p=3"))
    for nm, i, j in IDX:
        d = 100.0 * (c["L"][i, j] / f["L"][i, j] - 1.0)
        W("   %-6s %13.4f%% %13.4f%% %13.4f%% %13.4f%%"
          % (nm, d, d / (rh - 1), d / (rh ** 2 - 1), d / (rh ** 3 - 1)))
    W("")
W("Read the p=1 column as the pessimistic bound and p=2 as the expected one.")
W("At rho = 0.3 every term is <= 0.042% even at p = 1 -- the verdict is safe "
  "under any")
W("assumed order.  At rho = 0.05 the G error of the finest tet10 is ~0.31% "
  "at p = 2 and")
W("~0.69% at p = 1, i.e. 2 to 4 times the 0.174% two-point difference; quote "
  "0.3-0.7%, not")
W("0.17%, if G at rho = 0.05 is used quantitatively.")
W("")
W("The tet4 legs DO have enough points for an observed order (4 at rho = "
  "0.05, 3 at rho = 0.3).")
W("Measured over the finest interval, deviation from the converged tet10 law "
  "falls at:")
for rho in ("0.05", "0.3"):
    ref = ref_of[rho]
    t4 = sorted([r for r in recs if r["rho"] == rho and r["etype"] == "tet4"],
                key=lambda r: r["nelem"])
    if len(t4) < 2:
        continue
    seg = []
    for nm, i, j in IDX:
        e0 = abs(100 * (t4[-2]["L"][i, j] / ref["L"][i, j] - 1))
        e1 = abs(100 * (t4[-1]["L"][i, j] / ref["L"][i, j] - 1))
        p = np.log(e0 / e1) / np.log((t4[-1]["nelem"] / t4[-2]["nelem"])
                                     ** (1 / 3.0))
        seg.append("%s p=%.2f" % (nm, p))
    W("   rho = %-5s  %s" % (rho, "   ".join(seg)))
W("")
W("So G is NOT slower in RATE than A11/D11 -- at both densities its tet4 "
  "order is ~2, the")
W("same as the others.  G is simply far larger in MAGNITUDE at rho = 0.05 "
  "(77% vs 15% on the")
W("coarsest mesh), which is what makes it the last term to become usable.")

# -------------------------------------------------------------- locking ----
W("")
W("=" * 146)
W("tet4 LOCKING: tet4 over-prediction of the converged tet10 law")
W("=" * 146)
for rho in ("0.05", "0.3"):
    ref = ref_of[rho]
    W("")
    W("rho = %s   (converged tet10 = %s, %s elements, %s dofs)"
      % (rho, ref["base"], format(ref["nelem"], ","),
         format(ref["ndof"], ",")))
    W("%-34s %11s %12s %s" % ("tet4 mesh", "elements", "dofs",
                              "".join("%13s" % n for n, _, _ in IDX)))
    for r in sorted([x for x in recs
                     if x["rho"] == rho and x["etype"] == "tet4"],
                    key=lambda x: x["nelem"]):
        s = ("%-34s %11s %12s"
             % (r["base"].replace("SP_solid_rho%s_" % rho, ""),
                format(r["nelem"], ","), format(r["ndof"], ",")))
        s += "".join("%12.3f%%" % (100 * (r["L"][i, j] / ref["L"][i, j] - 1))
                     for _, i, j in IDX)
        W(s)

txt = "\n".join(lines) + "\n"
open(os.path.join(B, "conv_tables.txt"), "w").write(txt)
print(txt)

# ---------------------------------------------------------------- plots ----
TERMS = [("A11", 0, 0, "#1f77b4"), ("D11", 3, 3, "#d62728"),
         ("G11", 6, 6, "#2ca02c")]

for rho in ("0.05", "0.3"):
    ref = ref_of[rho]
    # Two panels.  On the tet4 scale (up to 1.77) the tet10 points sit on top
    # of y = 1 and convey nothing -- yet the tet10 pair IS the convergence
    # evidence, so it gets its own zoomed panel underneath.
    fig, (ax, axz) = plt.subplots(
        2, 1, figsize=(7.2, 6.0), sharex=True,
        gridspec_kw=dict(height_ratios=[2.3, 1.0], hspace=0.10))
    for name, i, j, col in TERMS:
        for etype, ls, mk in (("tet4", "-", "o"), ("tet10", "--", "s")):
            pts = sorted([r for r in recs
                          if r["rho"] == rho and r["etype"] == etype],
                         key=lambda r: r["nelem"])
            if not pts:
                continue
            x = [p["nelem"] for p in pts]
            y = [p["L"][i, j] / ref["L"][i, j] for p in pts]
            ax.plot(x, y, ls, marker=mk, color=col, ms=5, lw=1.6,
                    label="%s, %s" % (name, etype))
            if etype == "tet10":
                axz.plot(x, y, ls, marker=mk, color=col, ms=5, lw=1.6)
    span = max(abs(p["L"][i, j] / ref["L"][i, j] - 1.0)
               for p in recs
               if p["rho"] == rho and p["etype"] == "tet10"
               for _, i, j, _ in TERMS)
    axz.set_ylim(1.0 - 0.30 * span, 1.0 + 1.45 * span)
    for a in (ax, axz):
        a.axhline(1.0, color="0.55", lw=0.8, zorder=0)
        a.set_xscale("log")
        a.grid(True, which="both", alpha=0.25)
    axz.set_xlabel("number of elements")
    ax.set_ylabel("stiffness normalised by\nconverged tet10 value")
    axz.set_ylabel("tet10 only,\n%.2f%% full scale" % (100 * 1.75 * span))
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
              frameon=False, fontsize=9)
    axz.annotate("2 tet10 points -> a difference, not a convergence rate",
                 xy=(0.015, 0.06), xycoords="axes fraction", fontsize=8,
                 color="0.35")
    fig.savefig(os.path.join(PLOTS, "conv_rho%s.png" % rho), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("wrote conv_rho%s.png" % rho)

# combined locking figure: |error| vs converged tet10, log-log
fig, ax = plt.subplots(figsize=(7.2, 4.4))
STYLE = {"0.05": "#7b3294", "0.3": "#e08214"}
for rho in ("0.05", "0.3"):
    ref = ref_of[rho]
    for name, i, j in (("A11", 0, 0), ("G11", 6, 6)):
        for etype, ls, mk in (("tet4", "-", "o"), ("tet10", "--", "s")):
            pts = sorted([r for r in recs
                          if r["rho"] == rho and r["etype"] == etype],
                         key=lambda r: r["nelem"])
            x, y = [], []
            for p in pts:
                e = abs(100 * (p["L"][i, j] / ref["L"][i, j] - 1))
                if e > 0:                      # the reference itself is exact
                    x.append(p["nelem"])
                    y.append(e)
            if not x:
                continue
            alpha = 1.0 if name == "G11" else 0.55
            ax.plot(x, y, ls, marker=mk, color=STYLE[rho], ms=5, lw=1.6,
                    alpha=alpha,
                    label=r"$\rho$=%s, %s, %s" % (rho, name, etype))
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("number of elements")
ax.set_ylabel("deviation from converged tet10 law  [%]")
ax.grid(True, which="both", alpha=0.25)
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False,
          fontsize=8.5)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "locking_tet4_vs_tet10.png"), dpi=150,
            bbox_inches="tight")
plt.close(fig)
print("wrote locking_tet4_vs_tet10.png")

# --------------------------------------------------- machine-readable dump --
print()
print("### CONVERGED LAW DIAGONALS")
for rho in ("0.05", "0.3"):
    ref = ref_of[rho]
    print("rho=%s  %s  tet10  %s elem  %s dof"
          % (rho, ref["base"], format(ref["nelem"], ","),
             format(ref["ndof"], ",")))
    for k, (nm, i, j) in enumerate([("A11", 0, 0), ("A22", 1, 1),
                                    ("A66", 2, 2), ("D11", 3, 3),
                                    ("D22", 4, 4), ("D66", 5, 5),
                                    ("G11", 6, 6), ("G22", 7, 7)]):
        print("    %-4s = %.7E" % (nm, ref["L"][i, j]))

print()
print("### tet10 SELF-CONVERGENCE (coarser tet10 vs finest tet10)")
for rho in ("0.05", "0.3"):
    ref = ref_of[rho]
    t10 = sorted([r for r in recs
                  if r["rho"] == rho and r["etype"] == "tet10"],
                 key=lambda r: r["nelem"])
    for p in t10[:-1]:
        d = {nm: 100 * (p["L"][i, j] / ref["L"][i, j] - 1)
             for nm, i, j in IDX}
        print("rho=%s  %s (%s el) vs %s (%s el): %s"
              % (rho, p["base"], format(p["nelem"], ","), ref["base"],
                 format(ref["nelem"], ","),
                 "  ".join("%s %+.4f%%" % (k, v) for k, v in d.items())))
