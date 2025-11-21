import time
import opensg.utils as utils
from opensg.mesh.segment import SolidSegmentMesh
from opensg.core.solid import compute_stiffness
import opensg.core.stress_recov as stress_recov
tic = time.time()

file_name='data/Solid_3DSG/bar_urc_npl_1_ar_5-segment_' 
beam_out=utils.beamdyn_trans.beam_reaction(file_name)
segid=0 # user defined
# Read 3D yaml
mesh_yaml=file_name+ str(segid) +'.yaml' 
segment_mesh = SolidSegmentMesh(mesh_yaml)

# Extract material parameters and mesh data
material_parameters, density = segment_mesh.material_database
meshdata = segment_mesh.meshdata
l_submesh, r_submesh = segment_mesh.left_submesh, segment_mesh.right_submesh

# Compute stiffness and mass matrices 
timo = compute_stiffness(material_parameters, segment_mesh.meshdata, 
                         l_submesh, r_submesh, Taper=True)
print('\n Taper Stiffness \n', timo[0])

# Local strain and local displacement 
strain_m,u_loc=stress_recov.local_strain(timo,beam_out,segid,meshdata) 

# Local stress (quadrature points) and local stress(elemental nodes)
stress_m_quad, coord_quad, stress_m_node, coord_node=stress_recov.stress_eval(material_parameters, meshdata, strain_m)


# Eigenvalue
eigen= stress_recov.eigen_solve(material_parameters,
                                    meshdata,
                                    l_submesh,
                                    r_submesh,
                                    strain_m,
                                    u_loc)    
print('Computed Eigenvalue:', eigen)
print('\n Time Taken: 2D yaml',str(time.time()-tic))