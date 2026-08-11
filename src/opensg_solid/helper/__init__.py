"""Format conversions of the general SG engine (the opensg_shell.helper
analog on the solid side): sc_to_yaml turns a SwiftComp `.sc` structure
gene into the OpenSG solid SG yaml (+ a gmsh `.msh` sidecar), and
msh_to_yaml goes the other way round the loop -- a gmsh `.msh` mesh plus
the caller's materials into the `nodes`/`elements`/`elementOrientations`
solid SG yaml dialect the 2-D SG drivers read.
"""
