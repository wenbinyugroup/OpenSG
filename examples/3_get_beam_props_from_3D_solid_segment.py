import time
import opensg.utils.solid as utils
from opensg.mesh.segment import SolidSegmentMesh
from opensg.core.solid import compute_stiffness
tic = time.time()

# Read 3D yaml
mesh_yaml='data/Solid_3DSG/bar_urc_npl_1_ar_5-segment_2.yaml'
segment_mesh = SolidSegmentMesh(mesh_yaml)

# Extract material parameters and mesh data
material_parameters, density = segment_mesh.material_database
l_submesh, r_submesh = segment_mesh.left_submesh, segment_mesh.right_submesh

# Compute stiffness and mass matrices 
timo = compute_stiffness(material_parameters, segment_mesh.meshdata, 
                         l_submesh, r_submesh, Taper=True)
taper_mass=utils.get_mass_solid(segment_mesh.meshdata,density,Taper=True)

print('Tapered Segment Origin:', segment_mesh.meshdata["origin"])
print('\n Taper Mass \n',taper_mass)
print('\n Taper Stiffness \n', timo[0])
print('\n Time Taken',str(time.time()-tic))

