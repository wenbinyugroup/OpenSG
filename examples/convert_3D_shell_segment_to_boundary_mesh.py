import numpy as np
import time
from pathlib import Path
from opensg.mesh.segment import ShellSegmentMesh

data_dir = Path('data') / 'Shell_3D_Taper'
output_dir = Path('outputs')
output_dir.mkdir(exist_ok=True)

# Process segments (adjust range based on desired segments)
for segment in np.linspace(0, 28, 29): # np.linspace(start segment, end segment, (end_segment-start segment+1))
    segid = int(segment)
    tic = time.time()
    print('Segment:', str(segid), ' \n')

    # Generate mesh for current segment using StandaloneSolidSegmentMesh
    mesh_yaml = data_dir / f'BAR_URC_numEl_52_segment_{segid}.yaml'
    end = segment == 28
    segment_mesh = ShellSegmentMesh(mesh_yaml, create_1D=True, end=end, output_dir=str(output_dir))

    print('1D yaml created')
    print('\n Time Taken:', str(time.time() - tic))
