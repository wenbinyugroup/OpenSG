"""Schwarz-P shell SG convergence study: the .msh ladder -> SG yamls.

Three refinements of the SAME Schwarz-P surface at the SAME wall
thickness, so the only thing that varies is the mesh.  Each yaml is
written next to its mesh and is then a complete, self-contained input:

    opensg SP_mesh_1_shell.yaml

# ----------------------------------------------------------------------------
# ALL VARIABLES USED IN THIS SCRIPT
# ----------------------------------------------------------------------------
#   MESHES              the .msh ladder, coarse -> fine
#   THICKNESS           wall thickness (m) -- the SAME for every refinement,
#                       so the study measures discretization error alone
#   MATERIAL            isotropic wall material (helper.msh_to_yaml.ALUMINIUM
#                       shorthand: name, E, nu, density)
#   N_MODEL, REFINED    yaml header: 3 = equivalent 3-D solid, 0 = classical
#   WELD                merge coincident nodes before writing the yaml
#   r                   msh_to_yaml.convert result dict per mesh
# ----------------------------------------------------------------------------

WELDING.  The iso-surfacer that produced this ladder never welded its nodes:
meshes 1 and 3 carry several distinct node ids at the same point, and any
facet using two of them has a zero-length edge, so msh_to_yaml stops on
"element N is degenerate (zero area)".  WELD=True turns on the opt-in repair
(helper.msh_to_yaml.weld_nodes): coincident nodes are merged at 1e-12 x the
bounding-box diagonal -- 1.7e-12 m, against a shortest REAL edge of 5.0e-5 m
in mesh 1 and 8.9e-7 m in mesh 2 -- the collapsed facets are dropped and the
orphans deleted.  It is conservative: on mesh 1 the 2888 facets it removes
carry 2.3e-15 of the 2.352828 total area, so the surface area is unchanged to
1.9e-16 relative and the three refinements stay comparable.
"""
import os

from opensg_shell.helper.msh_to_yaml import convert

############### User Input #################################
MESHES = ["SP_mesh_1_shell.msh",       # coarse
          "SP_mesh_2_shell.msh",       # medium
          "SP_mesh_3_shell.msh"]       # fine
THICKNESS = 0.036457                   # m, the committed Schwarz-P wall
MATERIAL = {"name": "alu", "E": 69.0e9, "nu": 0.30, "density": 2700.0}
N_MODEL = 3                            # equivalent 3-D solid
REFINED = 0                            # classical wall law
WELD = True                            # meshes 1 and 3 ship UNWELDED: distinct
                                       # node ids at identical coordinates make
                                       # 124 / 1760 zero-area facets.  Weld the
                                       # coincident nodes and drop those facets.
############################################################

HERE = os.path.dirname(os.path.abspath(__file__))

for msh in MESHES:
    path = os.path.join(HERE, msh)
    if not os.path.exists(path):
        print("skip %s (not found)" % msh)
        continue
    r = convert(path, thickness=THICKNESS, material=MATERIAL,
                n_model=N_MODEL, refined=REFINED, weld=WELD)
    w = r.get("weld")
    print("%-22s -> %-22s  %7d nodes  %8d elements  area %.6f%s"
          % (msh, os.path.basename(r["out_path"]), len(r["nodes"]),
             len(r["elements"]), r["surface_area"],
             "" if not w else
             "   [welded: %d node(s) merged, %d facet(s) dropped,"
             " %d orphan(s) removed]"
             % (w["n_merged"], w["n_dropped"], w["n_orphans"])))

print("\nrun each with:  opensg <name>.yaml")
