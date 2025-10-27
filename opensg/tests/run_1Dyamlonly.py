# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 20:42:56 2025

@author: bagla0
"""

import numpy as np
import time
from opensg.mesh.segment import  ShellBounMesh
import opensg
import opensg.utils.shell as utils


np.set_printoptions(precision=4)
left_timo=[]
left_mass=[]
blade_length = 100
left_origin=[]
eig=[]
time_seg=[]


# Process segments (adjust range based on desired segments)
for segment in np.linspace(0,5,6): # np.linspace(start segment, end segment, (end_segment-start segment+1))
    file_name='1D_boun_BAR_URC_numEl_52_segment_'
    segid=int(segment)
    tic = time.time()
    print('Boundary:',str(segid)," \n")

    # Read 2D yaml 
    mesh_yaml = file_name+ str(int(segment)) +'.yaml' 
    segment_mesh = ShellBounMesh(mesh_yaml)
    meshdata=segment_mesh.meshdata
    
    ABD, mass = segment_mesh.compute_ABD()
    print('ABD time',str(time.time()-tic))
    Deff_srt,_,_= segment_mesh.compute_timo(ABD)
    print('1D Timo \n ', Deff_srt)
    left_mass.append(utils.get_mass_shell(meshdata, mass))
    print('1D mass \n', left_mass[int(segment)])

    left_origin.append(float(meshdata["origin"])/blade_length)  

    print('\n Time Taken: 1D yaml',str(time.time()-tic))
    
print('Total Time for Wind Blade (in sec)', np.sum(time_seg))  
print('\nleft_origin')
print(left_origin)

left_timo=np.array(left_timo)
left_mass=np.array(left_mass)


print(left_origin)

opensg.utils.shared.write_beamdyn_files(left_timo, left_mass, left_origin,'2Dyaml_npl1ar5_20251014')

