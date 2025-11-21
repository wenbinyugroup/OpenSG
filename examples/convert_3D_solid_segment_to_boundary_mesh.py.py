import numpy as np
import time
from opensg.mesh.segment import SolidSegmentMesh


for segment in np.linspace(0,27,28): # np.linspace(start segment, end segment, (end_segment-start segment+1))
    file_name='../data/Solid_3DSG/bar_urc_npl_1_ar_5-segment_'
    segid=int(segment)
    tic = time.time()
    print('Segment:',str(segid)," \n")

    mesh_yaml = file_name+ str(int(segment)) +'.yaml' 
    
    end=False
    if segment==28:
        end=True
    segment_mesh = SolidSegmentMesh(mesh_yaml, create_2D=True, end=end)

    print('2D yaml created')
    print('\n Time Taken: 2D yaml',str(time.time()-tic))