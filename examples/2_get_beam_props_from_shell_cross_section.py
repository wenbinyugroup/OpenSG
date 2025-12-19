import time
from opensg.mesh.segment import  ShellBounMesh
import opensg.utils.shell as utils

tic = time.time()

mesh_yaml = 'data/Shell_1DSG/1Dshell_28.yaml'
segment_mesh = ShellBounMesh(mesh_yaml)
meshdata=segment_mesh.meshdata

ABD, mass = segment_mesh.compute_ABD() # MSG based ABD matrix
print('ABD time',str(time.time()-tic))

# Timoshenko stiffness and mass amtrix
Deff_srt= segment_mesh.compute_timo(ABD)[0]
mass=utils.get_mass_shell(meshdata, mass)

print('Cross-Section', 'Origin:', meshdata['origin'],'\n')
print('1D mass \n', mass)
print('1D Timo \n ', Deff_srt)
print('\n Time Taken: 1D yaml',str(time.time()-tic))
