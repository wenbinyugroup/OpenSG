from opensg import helper
helper.linear_msh_to_quad("core_L2.msh")            # -> core_L2_quad.msh
helper.linear_msh_to_quad("core_L2.msh", out="core_L2_tet10.msh")