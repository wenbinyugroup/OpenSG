"""Plate properties of the 2-D honeycomb SG.  The yaml IS the whole
problem: its header says in words what to run (`model: plate`,
`refined: 0`, `analysis: H`), and its materials block defines each
material either by the 6x6 elastic stiffness (type 2, pre-rotated) or
by the 9 orthotropic engineering constants plus the ply angle (type 1;
this example ships that form).  The terminal route is one command,
one argument:
    opensg_solid RHC_SW_2UC_45.yaml
One-time SwiftComp conversion (the helper defines no __main__ -- import
its convert):
    from opensg_solid.helper.sc_to_yaml import convert
    convert("RHC_SW_2UC_45.sc")     # writes the .yaml + .msh

# ----------------------------------------------------------------------------
# ALL VARIABLES USED IN THIS SCRIPT
# ----------------------------------------------------------------------------
#   name                reads <name>.yaml (analysis + materials live in it)
#   r                   sg_homo.plate_homo_2d result dict; r["law"] is the
#                       macro law the yaml header selected, r["law_title"]
#                       its strain order; the timed <name>.out and
#                       <name>_mesh.png are written by the solver
# ----------------------------------------------------------------------------
"""
from opensg_solid.sg_homo import plate_homo_2d

############### User Input #################################
name = "RHC_SW_2UC_45"         # reads <name>.yaml
############################################################

r = plate_homo_2d(name + ".yaml")
print(r["law_title"] + ":")
print(r["law"])
print("Homogenization stored in %s.out" % name)
print("Time taken: %.2f sec" % float(r["solve_time"]))
