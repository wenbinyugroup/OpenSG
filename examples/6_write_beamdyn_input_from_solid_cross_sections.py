import numpy as np
import time
from opensg.mesh.segment import  SolidBounMesh
from opensg.core.solid import  compute_timo_boun
import opensg.utils as utils


left_timo, left_mass, left_origin=[],[],[]
blade_length = 100

for segment in np.linspace(0,28,29): # np.linspace(start segment, end segment, (end_segment-start segment+1))
    file_name='data/Solid_2DSG/2Dbar_urc_npl_1_ar_5-segment_'
    segid=int(segment)
    tic = time.time()
 #   print('Boundary:',str(segid)," \n")

    # Read 2D yaml 
    mesh_yaml = file_name+ str(segid) +'.yaml' 
    segment_mesh = SolidBounMesh(mesh_yaml)

    material_parameters, density = segment_mesh.material_database
    meshdata = segment_mesh.meshdata

    timo=compute_timo_boun(material_parameters, meshdata)
    left_mass.append(utils.solid.get_mass_solid(meshdata, density))

    left_origin.append(float(meshdata["origin"])/blade_length)   
    left_timo.append(timo[0]) # Right (x_max) boun: timo[0][1])
    print('2D Mass matrix \n',left_mass[segid])
    print('\n 2D Timo Stiffness\n', timo[0])
    print('\nTime Taken:',str(time.time()-tic))
    
    print('Origin', left_origin[segid])
left_timo=np.array(left_timo)
left_mass=np.array(left_mass)
utils.shared.write_beamdyn_files(np.array(left_timo), np.array(left_mass), left_origin,'2Dyaml_npl1ar5_20251121')


