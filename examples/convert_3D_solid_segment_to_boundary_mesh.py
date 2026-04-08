import numpy as np
import time
from pathlib import Path
from opensg.mesh.segment import SolidSegmentMesh

data_dir = Path('data') / 'Solid_3DSG'
output_dir = Path('outputs')
output_dir.mkdir(exist_ok=True)

for segment in np.linspace(0, 27, 28): # np.linspace(start segment, end segment, (end_segment-start segment+1))
    segid = int(segment)
    tic = time.time()
    print('-' * 50)
    print('Segment:', str(segid), ' \n')

    mesh_yaml = data_dir / f'bar_urc_npl_1_ar_5-segment_{segid}.yaml'
    end = segment == 27
    segment_mesh = SolidSegmentMesh(mesh_yaml, create_2D=True, end=end, output_dir=str(output_dir))

    print('2D yaml created')
    print('\n Time Taken: 2D yaml', str(time.time() - tic), 'sec \n')
