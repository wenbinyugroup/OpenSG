import numpy as np
import time
from opensg.mesh.segment import  SolidBounMesh
from opensg.core.solid import  compute_timo_boun
import opensg
import opensg.utils.solid as utils


np.set_printoptions(precision=4)
left_timo=[]
left_mass=[]
blade_length = 100
left_origin=[]
eig=[]
time_seg=[]

# Process segments (adjust range based on desired segments)
for segment in np.linspace(0,5,6): # np.linspace(start segment, end segment, (end_segment-start segment+1))
    file_name='2Dbar_urc_npl_1_ar_5-segment_'
    segid=int(segment)
    tic = time.time()
    print('Boundary:',str(segid)," \n")

    # Read 2D yaml 
    mesh_yaml = file_name+ str(int(segment)) +'.yaml' 
    segment_mesh = SolidBounMesh(mesh_yaml)
    

    material_parameters, density = segment_mesh.material_database
    meshdata = segment_mesh.meshdata

    timo=compute_timo_boun(material_parameters, meshdata)
    print('2D yaml Timo Stiffness\n', timo[0])

    left_mass.append(utils.get_mass_solid(meshdata, density))
    print('2D yaml Mass matrix \n',left_mass[int(segment)])

    left_origin.append(float(meshdata["origin"])/blade_length)   
    left_timo.append(timo[0])

    print('\n Time Taken: 2D yaml',str(time.time()-tic))


