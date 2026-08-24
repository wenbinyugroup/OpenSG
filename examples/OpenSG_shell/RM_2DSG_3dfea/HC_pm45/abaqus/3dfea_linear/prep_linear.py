"""prep_linear.py -- build the LINEAR Abaqus rerun package in
abaqus/3dfea_linear/: the nlgeom=NO deck (byte-identical to the run
deck otherwise), the matching dump script and .bat (CRLF), a README,
and refresh_linear.py for the post-run path extraction."""
import os

HC = os.path.join(os.path.expanduser("~"), "OpenSG-2.0", "examples",
                  "OpenSG_shell", "RM_2DSG_3dfea", "HC_pm45")
SRC = os.path.join(HC, "abaqus", "3dfea_output")
OUT = os.path.join(HC, "abaqus", "3dfea_linear")
os.makedirs(OUT, exist_ok=True)

# ---- 1. the deck: nlgeom=YES -> NO, everything else byte-identical
raw = open(os.path.join(SRC, "45pm_L_min_pm45_run.inp"), "rb").read()
n_yes = raw.count(b"nlgeom=YES")
n_all = raw.count(b"nlgeom")
print("deck: %d bytes, nlgeom occurrences %d (YES: %d)"
      % (len(raw), n_all, n_yes))
assert n_yes == 1 and n_all == 1, "unexpected nlgeom cards -- inspect"
open(os.path.join(OUT, "45pm_L_min_pm45_linear.inp"), "wb").write(
    raw.replace(b"nlgeom=YES", b"nlgeom=NO"))
print("wrote 45pm_L_min_pm45_linear.inp (nlgeom=NO)")

# ---- 2. the dump script: same extractor, distinct file names
d = open(os.path.join(SRC, "abq_dump_45pm.py")).read()
for a, b in (("45pm_L_min_pm45_run.odb", "45pm_L_min_pm45_linear.odb"),
             ("45pm_S_global.csv", "45pm_S_global_linear.csv"),
             ("45pm_S_material.csv", "45pm_S_material_linear.csv"),
             ("45pm_U.csv", "45pm_U_linear.csv"),
             ("abq_dump_45pm.py", "abq_dump_45pm_linear.py")):
    d = d.replace(a, b)
open(os.path.join(OUT, "abq_dump_45pm_linear.py"), "w").write(d)
print("wrote abq_dump_45pm_linear.py")

# ---- 3. the .bat (CRLF; SH points at THIS folder)
SH = (r"\\roger.ecn.purdue.edu\bagla0\OpenSG-2.0\examples\OpenSG_shell"
      r"\RM_2DSG_3dfea\HC_pm45\abaqus\3dfea_linear")
bat = "\r\n".join([
    "@echo off",
    "REM run_45pm_linear.bat -- the SAME pm45 L_min deck with nlgeom=NO",
    "REM (geometrically linear), for the apples-to-apples comparison",
    "REM with the linear OpenSG/SwiftComp chain.  Outputs carry the",
    "REM _linear suffix so the nonlinear reference is never overwritten.",
    "set SH=" + SH,
    "set JD=C:\\Temp\\PM45LIN",
    "if exist %JD% rmdir /s /q %JD%",
    "mkdir %JD%",
    "cd /d %JD%",
    "echo === FREE SPACE ===",
    "wmic logicaldisk where \"DeviceID='C:'\" get FreeSpace"
    " /format:list",
    "xcopy \"%SH%\\45pm_L_min_pm45_linear.inp\" %JD%\\ /Y",
    "xcopy \"%SH%\\abq_dump_45pm_linear.py\" %JD%\\ /Y",
    "call abaqus job=45pm_L_min_pm45_linear cpus=4 interactive",
    "echo ===JOB DONE===",
    "call abaqus python abq_dump_45pm_linear.py",
    "echo ===DUMP DONE===",
    "xcopy %JD%\\*.sta \"%SH%\\\" /Y",
    "xcopy %JD%\\*.dat \"%SH%\\\" /Y",
    "xcopy %JD%\\*.msg \"%SH%\\\" /Y",
    "xcopy %JD%\\45pm_S_global_linear.csv \"%SH%\\\" /Y",
    "xcopy %JD%\\45pm_S_material_linear.csv \"%SH%\\\" /Y",
    "xcopy %JD%\\45pm_U_linear.csv \"%SH%\\\" /Y",
    "echo ===ALL DONE===",
    ""])
open(os.path.join(OUT, "run_45pm_linear.bat"), "w", newline="").write(bat)
print("wrote run_45pm_linear.bat")

# ---- 4. refresh_linear.py (post-run: csv -> path dats + delta report)
R = '''"""refresh_linear.py -- AFTER the linear Abaqus run: extract the
path_1/path_2 references from 45pm_S_material_linear.csv (the same
Richardson-onto-midspan + element pairing as csv_to_dat.py) and print
what the nlgeom=NO rerun changed against the nonlinear reference.

Writes ../station_path_stress/path_N_abaqus_linear.dat -- the
nonlinear path_N_abaqus.dat is NOT touched; promote by hand once the
numbers are reviewed.
"""
import datetime
import os

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
STP = os.path.join(HERE, "..", "station_path_stress")
COMP = ["S11", "S22", "S33", "S12", "S13", "S23"]
YCELL, HALF = 30.10244, 3.762805

print("start : %s" % datetime.datetime.now().strftime("%H:%M:%S"))
S = np.genfromtxt(os.path.join(HERE, "45pm_S_material_linear.csv"),
                  delimiter=",", names=True)
XC = 0.5 * (S["x"].min() + S["x"].max())
lay = np.unique(np.round(S["x"] - XC, 4))
d1 = np.sort(np.abs(lay[np.abs(lay) > 1e-6]))[0]
print("linear dump: %d rows; midspan x = %.6f" % (len(S), XC))


def layer(o):
    k = np.abs(np.round(S["x"] - XC, 4) - round(o, 4)) < 1e-4
    k &= (S["y"] >= YCELL - HALF) & (S["y"] <= YCELL + HALF)
    r = S[k]
    return r[np.lexsort((np.round(r["z"], 5), np.round(r["y"], 5)))]


avg = {}
for tag, dd in (("in", d1), ("out", 3.0 * d1)):
    m, p = layer(-dd), layer(+dd)
    if len(m) != len(p):
        raise SystemExit("layer pair %s does not align" % tag)
    avg[tag] = {c: 0.5 * (m[c] + p[c]) for c in COMP}
    avg[tag]["y"], avg[tag]["z"] = m["y"], m["z"]
F = np.stack([(9.0 * avg["in"][c] - avg["out"][c]) / 8.0
              for c in COMP], axis=-1)
tree = cKDTree(np.column_stack([avg["in"]["y"] - YCELL,
                                avg["in"]["z"]]))

for n in (1, 2):
    cf = os.path.join(HERE, "..", "..", "opensg", "path_%d" % n,
                      "path_%d.coords" % n)
    C = np.loadtxt(cf)
    dist, idx = tree.query(C[:, 1:3])
    print("path %d: pairing %.2e %s" % (n, dist.max(),
                                        "OK" if dist.max() < 1e-4
                                        else "FAIL"))
    if dist.max() >= 1e-4:
        raise SystemExit("pairing failed")
    out = os.path.join(STP, "path_%d_abaqus_linear.dat" % n)
    with open(out, "w") as f:
        f.write("# Abaqus 3-D FEA stress along path_%d -- MATERIAL"
                " frame, LINEAR (nlgeom=NO) rerun, Richardson onto"
                " midspan x = %.6f\\n" % (n, XC))
        f.write("# %10s %12s %12s %4s" % ("s", "x1", "x2", "mat"))
        for c in COMP:
            f.write(" %13s" % c)
        f.write("\\n")
        for i in range(len(C)):
            f.write("%12.6f %12.6f %12.6f %4d"
                    % (C[i, 0], C[i, 1], C[i, 2], int(C[i, 3])))
            for j in range(6):
                f.write(" %13.6e" % F[idx[i], j])
            f.write("\\n")
    print("wrote %s" % os.path.basename(out))
    old = np.loadtxt(os.path.join(STP, "path_%d_abaqus.dat" % n))
    new = np.loadtxt(out)
    print("  linear vs NONLINEAR reference (RMS %% of nonlinear max):")
    for j, c in enumerate(COMP):
        den = max(np.abs(old[:, 4 + j]).max(), 1e-30)
        print("    %-4s %6.2f %%" % (c, 100 * np.sqrt(np.mean(
            (new[:, 4 + j] - old[:, 4 + j]) ** 2)) / den))
print("end   : %s" % datetime.datetime.now().strftime("%H:%M:%S"))
'''
open(os.path.join(OUT, "refresh_linear.py"), "w").write(R)
print("wrote refresh_linear.py")

# ---- 5. README
open(os.path.join(OUT, "README.md"), "w").write("""\
# 3dfea_linear -- the nlgeom=NO rerun (apples-to-apples with OpenSG)

The shipped 3-D reference (../3dfea_output) ran `*Step, nlgeom=YES`
while every model curve (plate, RM, classical, SwiftComp) is linear;
the measured consequences are the midspan membrane resultants
(N11 = -0.26, N22 = -0.17 N/mm) and part of the face-sheet mean
offset.  This folder is the SAME deck with nlgeom=NO, outputs suffixed
_linear so the nonlinear reference stays untouched.

## Run (Remote Desktop, like run_45pm.bat)

1. double-click run_45pm_linear.bat  (deck + dump copied to
   C:\\Temp\\PM45LIN, solved with cpus=4, CSVs copied back here)
2. back on the server:  python refresh_linear.py
   -> ../station_path_stress/path_N_abaqus_linear.dat + a printed
   linear-vs-nonlinear delta per component
3. review the deltas; to promote the linear reference into the 4-way
   comparison, point compare/four_way at the _linear dats (or copy
   them over path_N_abaqus.dat) and rerun make_path_plots_4way.py.

Provenance: deck generated by prep_linear.py -- byte-identical to
45pm_L_min_pm45_run.inp except the single nlgeom card.
""")
print("wrote README.md")
print("\nfolder contents:")
for f in sorted(os.listdir(OUT)):
    print("  %-34s %d bytes"
          % (f, os.path.getsize(os.path.join(OUT, f))))
