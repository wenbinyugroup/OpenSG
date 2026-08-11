"""Convert the Schwarz-P TPMS shell mesh (gmsh 2.2 triangles) to the OpenSG
3-D shell SG yaml.  Aluminum matching the SwiftComp .sc (E = 69 GPa,
nu = 0.3); shell thickness T below (relative density = area x T, reported).

The mesh -> yaml work is opensg_shell.helper.msh_to_yaml.convert (nodes,
elements, the per-facet e1/e2/e3 frame and the element sets); this driver
only owns the PATHS, the wall THICKNESS and the MATERIAL -- the modelling
choices the mesh cannot supply.  convert writes them as a real
`sections:` layup [[alu, T, 0.0]], so the thickness ends up in the yaml and
not in code; `material=` is spelled out here rather than left to convert's
aluminium default only because this case's block is worth reading.
Everything the analysis needs then lives in the yaml, so
`opensg_shell schwarz_p_3Dshell.yaml` runs with nothing passed in code.

Run (from this folder):  python make_schwarz_yaml.py"""
from opensg_shell.helper.msh_to_yaml import convert

MSH = ("../../../tests/08072026_square_shell_mesh/TPMS_SC_files/"
       "schwarz_p_D2_shell.msh")
YAML = "schwarz_p_3Dshell.yaml"
T = 0.036457           # Sample_2, volume-consistent
E, nu = 69.0e9, 0.30

d = convert(MSH, thickness=T, out_path=YAML, n_model=3, refined=0,
            material={"name": "alu", "E": E, "nu": nu, "density": 2700.0})
print("t = %.6f -> relative density %.4f"
      % (T, d["surface_area"]*T))
