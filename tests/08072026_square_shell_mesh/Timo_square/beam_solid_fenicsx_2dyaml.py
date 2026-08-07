"""Timoshenko 6x6 from the 2-D SOLID SG (OpenSG-FEniCSx boundary engine).

Reads  : square_tube_2Dsolid.yaml -- the OpenSG 2-D solid cross-section yaml for the
         PreVABS square thin-walled tube (midline a=1.0, t=0.03, 3228 nodes, 5384
         LINEAR TRIANGLES, one orthotropic Material_1, single -45 deg ply).  The fibre
         is already baked into `elementOrientations` (e1 = [-0.7071, 0, 0.7071]), which
         is what the FEniCSx solid route expects -- no extra ply angle is applied here.
Writes : square_tube_beam_fenicsx.out -- the 6x6 in
         [eps11 gam12 gam13 kappa1 kappa2 kappa3] = [ext sh2 sh3 twist bend2 bend3],
         diagonal = [EA GA2 GA3 GJ EI2 EI3].

Pipeline: SolidBounMesh(yaml) -> compute_timo_boun(mat_param, meshdata), which returns
the 3-tuple (Deff_srt 6x6, V0, V1s).  Being a boundary (cross-section) solve there is no
division by a segment length L.

Runs in WSL, conda env opensg_env_v8 (dolfinx 0.8.0 + gmsh).  PKG below is prepended to
sys.path so the training-data opensg-FEniCS copy SHADOWS the installed opensg: only that
copy picks the gmsh element type from the connectivity (3 nodes -> type 2 triangle),
whereas site-packages hard-codes type 3 (4-node quad) and cannot read this triangle mesh.

Run:  wsl -e /home/bagla0/miniconda3/envs/opensg_env_v8/bin/python /tmp/Timo_square/beam_solid_fenicsx_2dyaml.py
"""
import os
import subprocess
import sys

import numpy as np

############### User Input #################################
# Cross-section yaml on the server / Y: drive (WSL cannot see Y:, staged below).
YAML_WIN = r"Y:\OpenSG-2.0\tests\08072026_square_shell_mesh\square_tube_2Dsolid.yaml"
OUT_WIN = r"Y:\OpenSG-2.0\tests\08072026_square_shell_mesh\square_tube_beam_fenicsx.out"
WORK = "/tmp/Timo_square"
# Material_1, SI (Pa), flat in the order SolidBounMesh.material_database emits:
# E1 E2 E3  G12 G13 G23  nu12 nu13 nu23
mat_param = [np.array((142.0e9, 9.8e9, 9.8e9, 6.0e9, 6.0e9, 4.8e9, 0.30, 0.30, 0.42))]
# The training-data opensg-FEniCS copy -- the one that handles linear triangles.
PKG = ("/mnt/c/Users/bagla0/OneDrive - purdue.edu/2026_195/Claude_code/"
       "tests/research/training data/opensg-FEniCS")
############################################################

sys.path.insert(0, PKG)
os.makedirs(WORK, exist_ok=True)
os.chdir(WORK)
YAML = os.path.join(WORK, os.path.basename(YAML_WIN))
OUT = os.path.join(WORK, os.path.basename(OUT_WIN))

# Stage the yaml off Y: into WSL by shelling back out to Windows PowerShell (WSL has no Y:).
subprocess.run(["powershell.exe", "-NoProfile", "-Command", "Copy-Item -LiteralPath '%s' -Destination '%s'"
                % (YAML_WIN, subprocess.run(["wslpath", "-w", YAML], capture_output=True, text=True).stdout.strip())], check=True)

from opensg.mesh.segment import SolidBounMesh
from opensg.core.solid import compute_timo_boun

sm = SolidBounMesh(YAML)
C6 = np.asarray(compute_timo_boun(mat_param, sm.meshdata)[0], dtype=float)

LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
print("FEniCSx 2-D solid SG, beam 6x6  [eps11 gam12 gam13 kappa1 kappa2 kappa3]  (Timoshenko):")
for i in range(6):
    print("  " + " ".join("%15.6e" % C6[i, j] for j in range(6)))
print("diagonal: " + "  ".join("%s=%.6e" % (LBL[i], C6[i, i]) for i in range(6)))

np.savetxt(OUT, C6, fmt="%18.8e",
           header="Timoshenko 6x6 beam stiffness (SI, Pa-based) from the 2-D SOLID SG\n"
                  "source yaml  : %s\n"
                  "engine       : OpenSG-FEniCSx SolidBounMesh + core.solid.compute_timo_boun\n"
                  "               (dolfinx 0.8.0, conda env opensg_env_v8)\n"
                  "row/col order: [eps11  gam12  gam13  kappa1  kappa2  kappa3]\n"
                  "             = [ ext    sh2    sh3    twist   bend2   bend3 ]\n"
                  "diagonal     = [ EA     GA2    GA3    GJ      EI2     EI3   ]"
                  % os.path.basename(YAML_WIN))
# Push the result back out to Y: / the server home.
subprocess.run(["powershell.exe", "-NoProfile", "-Command", "Copy-Item -LiteralPath '%s' -Destination '%s' -Force"
                % (subprocess.run(["wslpath", "-w", OUT], capture_output=True, text=True).stdout.strip(), OUT_WIN)], check=True)
print("wrote %s" % OUT_WIN)
