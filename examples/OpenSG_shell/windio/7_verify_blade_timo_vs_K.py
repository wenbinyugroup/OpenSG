"""Verifier: in-memory Blade Timoshenko + mass vs the committed VABS .K files.

The demo, the verifier and the unit test in one script: the IEA-22 blade is
read into the editable Blade object and every station in props/ is
recomputed with the standalone in-memory homogenizer
(opensg_shell.pynumad.sg_homo.timo -- no yaml, no .out); the beam
STIFFNESS and the MASS matrix are compared against the committed
iea22_rXXXX.K files, one PASS/FAIL line per station printed immediately.
Any station outside tolerance fails the final assertion, so a nonzero
exit code = verification failure (tests/msg_shell_windio/test_blade_vs_k.py
runs this script as the unit test).

The .K files were produced by the file-based pipeline (steps 1-2) at
reference=center, so the Blade is built with the same reference.

# ----------------------------------------------------------------------------
# ALL VARIABLES USED IN THIS SCRIPT
# ----------------------------------------------------------------------------
#   BLADE_YAML       the IEA-22-280-RWT windIO (v2) blade file
#   K_FOLDER, RTOL   committed .K folder; max-entry-relative tolerance
#   blade            editable Blade at the .K files' reference (center)
#   k_path, span_r   one committed .K file and its span (tag rXXXX / 1000)
#   stiffness_file,  the 6x6 pair parsed from the .K file
#   mass_file
#   stiffness_mem,   the 6x6 pair recomputed fully in memory
#   mass_mem
#   diff_K, diff_M   max |difference| / max |file entry| per matrix
#   failed_stations  stations outside RTOL (must stay empty)
# ----------------------------------------------------------------------------
"""
import glob
import os
import re

import numpy as np

import opensg
from opensg_shell.pynumad.sg_homo import timo

############### User Input #################################
BLADE_YAML = "IEA-22-280-RWT.yaml"
K_FOLDER = "props"
RTOL = 1e-8                        # max-entry-relative pass tolerance
############################################################


def read_vabs_k(path):
    """Parse a VABS .K-layout file into its two 6x6 blocks.

    In:  path str -- .K file (OpenSG write_vabs_k or VABS layout).
    Out: (stiffness (6,6), mass (6,6)) -- the 'Timoshenko Stiffness
         Matrix' block and the FIRST 'The 6X6 Mass Matrix' block (the
         section-origin frame, not the mass-center one)."""
    lines = open(path).read().splitlines()

    def block(header):
        for i, ln in enumerate(lines):
            if header in ln:
                rows, j = [], i + 1
                while len(rows) < 6 and j < len(lines):
                    try:
                        row = [float(v) for v in lines[j].split()]
                    except ValueError:
                        row = []
                    j += 1
                    if len(row) == 6:
                        rows.append(row)
                return np.array(rows)
        raise ValueError("header %r not found in %s" % (header, path))

    mass = block("The 6X6 Mass Matrix")
    stiffness = block("Timoshenko Stiffness Matrix")
    return stiffness, mass


blade = opensg.read(BLADE_YAML, reference="center")   # the .K files' reference
k_files = sorted(glob.glob(os.path.join(K_FOLDER, "iea22_r*.K")))
assert k_files, "no committed iea22_r*.K files under %s/" % K_FOLDER
# the file tag rXXXX is round(r*1000): map each .K back to the blade's TRUE
# airfoil station (solving at the rounded span would mesh a slightly
# different section and read as a false mismatch)
tag_to_station = {round(r * 1000): r for r in blade.stations}
print("IEA-22 blade, %d committed .K stations; tolerance %.0e"
      % (len(k_files), RTOL), flush=True)

failed_stations = []
for k_path in k_files:
    tag = int(re.search(r"_r(\d{4})\.K$", k_path).group(1))
    span_r = tag_to_station[tag]
    stiffness_file, mass_file = read_vabs_k(k_path)
    stiffness_mem, mass_mem = timo(blade, float(span_r))
    diff_K = (np.abs(stiffness_mem - stiffness_file).max()
              / np.abs(stiffness_file).max())
    diff_M = np.abs(mass_mem - mass_file).max() / np.abs(mass_file).max()
    ok = diff_K < RTOL and diff_M < RTOL
    print("r = %.3f   max|dK|rel = %.2e   max|dM|rel = %.2e   %s"
          % (span_r, diff_K, diff_M, "PASS" if ok else "FAIL"), flush=True)
    if not ok:
        failed_stations.append((span_r, diff_K, diff_M))

print("-" * 60, flush=True)
print("%d/%d stations verified against props/*.K"
      % (len(k_files) - len(failed_stations), len(k_files)), flush=True)
assert not failed_stations, "stations outside tolerance: %s" % failed_stations
