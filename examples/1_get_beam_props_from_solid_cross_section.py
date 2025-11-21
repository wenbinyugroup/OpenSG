
import time
from opensg.mesh.segment import  SolidBounMesh
from opensg.core.solid import  compute_timo_boun
import opensg.utils.solid as utils
tic = time.time()

# Read 2D yaml 
mesh_yaml='data/Solid_2DSG/2Dbar_urc_npl_1_ar_5-segment_0.yaml'
segment_mesh = SolidBounMesh(mesh_yaml)
material_parameters, density = segment_mesh.material_database
meshdata = segment_mesh.meshdata

# Timoshenko stiffness and mass matrix
timo=compute_timo_boun(material_parameters, meshdata)
mass=utils.get_mass_solid(meshdata, density)

print('Boundary Origin:', meshdata['origin'],'\n')
print('2D Mass matrix \n',mass)
print('\n 2D Timo Stiffness\n', timo[0])
print('\n Time Taken: 2D yaml',str(time.time()-tic))

