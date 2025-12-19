import numpy as np
import time
from opensg.mesh.segment import ShellSegmentMesh

# Process segments (adjust range based on desired segments)
for segment in np.linspace(0,28,29): # np.linspace(start segment, end segment, (end_segment-start segment+1))
    file_name='data/Shell_3D_Taper/BAR_URC_numEl_52_segment_'
    segid=int(segment)
    tic = time.time()
    print('Segment:',str(segid)," \n")

    # Generate mesh for current segment using StandaloneSolidSegmentMesh
    mesh_yaml = file_name+ str(int(segment)) +'.yaml' 
    end=False
    if segment==28:
        end=True
    segment_mesh = ShellSegmentMesh(mesh_yaml, create_1D=True, end=end)

    print('1D yaml created')
    print('\n Time Taken:',str(time.time()-tic))
    
    
  
    
