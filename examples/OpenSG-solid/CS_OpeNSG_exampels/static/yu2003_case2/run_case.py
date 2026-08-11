"""run_case.py -- this case, END TO END on the ex5 pipeline:

  1. HOMO      core rm_homo on layup_db.yaml -> 1dsg.yaml + <name>_plate_homo.out
  2. ANALYSIS  the validated benchmark engine (pagano_bench / yu_bench),
               DRIVEN BY THIS FOLDER'S layup_db.yaml: its module globals
               (MATERIAL_DB, LAYUPS, H, ...) are overridden from the YAML, so
               the engine keeps its physics (statics / harmonic RM solve ->
               Eq.-63 recovery -> sigma33 by thickness equilibrium, Pagano
               exact 3-D as reference curves only) while every INPUT comes
               from the pipeline's single user file
  3. PLOTS     the engine .dat is re-plotted as TWO-CURVE figures:
               Pagano exact 3-D vs OpenSG-RM, one figure per stress component
               per aspect ratio (no FSDT, no titles, legend outside)

Run:  python run_case.py

# ----------------------------------------------------------------------------
# ALL VARIABLES USED IN THIS SCRIPT
# ----------------------------------------------------------------------------
#   FAMILY, CASE    which engine ("garg" / "yu") and its LAYUPS key
#   HERE, ROOT      this folder; the repo root
#   d               the parsed layup_db (core loader)
#   eng             the engine module, globals overridden from layup_db
#   S_LIST          bench.S aspect ratios
#   COLS            per-family .dat column map {quantity: (msg, exact)}
#   UNIT            the stress axis label suffix
#   dat, z, h       one engine .dat, its z column, the thickness
#   fig, ax, q, cm, ce, S, stem   the re-plot loop working set
# ----------------------------------------------------------------------------
"""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FAMILY = "yu"                 # "garg" or "yu"
CASE = "case2"                  # the engine's LAYUPS key

HERE = os.path.dirname(os.path.abspath(__file__))
try:
    import opensg_solid                      # pip install -e . -- nothing to do
except ImportError:                          # fall back to the in-repo source tree
    ROOT = HERE
    while not os.path.isdir(os.path.join(ROOT, "src", "opensg_solid")):
        parent = os.path.dirname(ROOT)
        if parent == ROOT:                   # hit the filesystem root
            raise ImportError(
                "opensg_solid not installed and no src/ found above " + HERE)
        ROOT = parent
    sys.path.insert(0, os.path.join(ROOT, "src"))

import time as _t
print("start: " + _t.strftime("%Y-%m-%d %H:%M:%S"))
sys.path.insert(0, os.path.join(ROOT, "examples", "OpenSG-solid",
                                "CS_OpeNSG_exampels", "static",
                                "garg" if FAMILY == "garg" else "yu2003"))

from opensg_solid.rm_plate_1D.rm_homo import load_layup_db, homogenize_layup_db

# ---- 1. HOMO through the core (the pipeline artefacts of this folder) -------
DB = os.path.join(HERE, "layup_db.yaml")
d = load_layup_db(DB)
homogenize_layup_db(DB)
print("homo: 1dsg.yaml + %s_plate_homo.out written"
      % d["db"].get("name", "layup_db"))

# ---- 2. the engine, driven by THIS layup_db ---------------------------------
if FAMILY == "garg":
    import pagano_bench as eng
else:
    import yu_bench as eng

eng.HERE = HERE                            # outputs -> HERE/<CASE>/
os.makedirs(os.path.join(HERE, CASE), exist_ok=True)
eng.MATERIAL_DB.clear()
eng.MATERIAL_DB.update(d["material_db"])
eng.LAYUPS = {CASE: d["layup"]}
eng.H = float(sum(d["layup"]["thick"]))
p = d["db"]["plate"]
if FAMILY == "garg":
    eng.q0 = float(p["q0"])
    eng.a = float(p["a"])
    eng.p = np.pi / eng.a
else:
    eng.L_SPAN = float(p["a"])
    eng.P0 = float(p["q0"])

S_LIST = tuple(d["db"].get("bench", {}).get("S", [4]))
if FAMILY == "garg":
    eng.run_benchmark(CASE, S_LIST, tag="ex5-pipeline")
else:
    eng.run_case(CASE)

# ---- 3. the two-curve re-plots ---------------------------------------------
if FAMILY == "garg":
    COLS = {"s13": (1, 2), "s33": (4, 5)}
    UNIT = "[Pa]"
    dats = [(os.path.join(HERE, CASE, "pagano_S%g.dat" % S), "S%g" % S)
            for S in S_LIST]
else:
    COLS = {"s13": (1, 2), "s23": (4, 5), "s33": (6, 7)}
    UNIT = r"$/p_0$"
    dats = [(os.path.join(HERE, CASE, "yu_%s.dat" % CASE), "S4")]

LBL = {"s13": r"$\sigma_{13}$", "s23": r"$\sigma_{23}$",
       "s33": r"$\sigma_{33}$"}
for dat, stem in dats:
    D = np.loadtxt(dat)
    z = D[:, 0]
    h = z.max() - z.min()
    for q, (cm, ce) in COLS.items():
        fig, ax = plt.subplots(figsize=(5.2, 5.6))
        ax.plot(D[:, ce], z / h, ls="--", marker="o", color="k", lw=1.4,
                ms=5, mfc="none", mew=1.1, markevery=6,
                label="Pagano exact 3-D")
        ax.plot(D[:, cm], z / h, ls="-", marker="s", color="#ff7f0e",
                lw=1.3, ms=4, mfc="none", mew=1.0, markevery=(3, 6),
                label="OpenSG-RM")
        if q == "s13":
            # the FSDT third curve: the Whitney-1973 k^2 staircase -- the
            # same construction the ABAQUS community-FSDT composite-S4
            # process reproduces (statics fixes Q1 identically; process
            # check in the folder README)
            ax.plot(D[:, 3], z / h, ls="-.", marker="^", color="#2ca02c",
                    lw=1.2, ms=5, mfc="none", mew=1.0, markevery=(1, 6),
                    label="FSDT (Abaqus composite-S4" "\nprocess, Whitney k^2)")
        ax.set_xlabel("%s %s" % (LBL[q], UNIT), fontsize=11)
        ax.set_ylabel(r"$z/h$", fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                  frameon=False)
        fig.tight_layout()
        out = os.path.join(HERE, "%s_%s.png" % (q, stem))
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("wrote", os.path.basename(out))
print("done:", CASE)
print("end:   " + _t.strftime("%Y-%m-%d %H:%M:%S"))
