import numpy as np
import time
from opensg.mesh.segment import  ShellBounMesh
import opensg.utils as utils

left_timo, left_mass, left_origin=[],[],[]
blade_length = 100

for segment in np.linspace(0,28,29): # np.linspace(start segment, end segment, (end_segment-start segment+1))
    file_name='data/Shell_1DSG/1D_v8/1Dshell_'
    segid=int(segment)
    tic = time.time()
    print('Boundary:',str(segid)," \n")

    # Read 1D yaml 
    mesh_yaml = file_name+ str(int(segment)) +'.yaml' 
    segment_mesh = ShellBounMesh(mesh_yaml)
    meshdata=segment_mesh.meshdata
    
    ABD, mass = segment_mesh.compute_ABD()
    Deff_srt,_,_= segment_mesh.compute_timo(ABD)
    print('1D Timo \n ', Deff_srt)
    left_mass.append(utils.shell.get_mass_shell(meshdata, mass))
    print('1D mass \n', left_mass[int(segment)])

    left_origin.append(float(meshdata["origin"])/blade_length)  

    print('\n Time Taken: 1D yaml',str(time.time()-tic))
    

utils.shared.write_beamdyn_files(np.array(left_timo), np.array(left_mass), left_origin,'1Dyaml_npl1ar5_20251120')



