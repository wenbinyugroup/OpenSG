import time
from pathlib import Path
from opensg.mesh.segment import SolidBounMesh
from opensg.core.solid import compute_timo_boun
import opensg.core.stress_recov as stress_recov
import opensg.utils as utils
from dolfinx import io

tic = time.time()

# Read 2D yaml
data_dir = Path('data')
segid = 5
mesh_yaml = data_dir / 'Solid_2DSG' / f'2Dsolid_{segid}.yaml'
beam_out = utils.beamdyn_trans.beam_reaction(data_dir / 'bd_bar_urc.out')

segment_mesh = SolidBounMesh(mesh_yaml)
material_parameters, density = segment_mesh.material_database
meshdata = segment_mesh.meshdata

# Timoshenko stiffness and mass matrix
timo = compute_timo_boun(material_parameters, meshdata)

# Local strain and local displacement
strain_m, u_loc = stress_recov.local_strain(timo, beam_out, segid, meshdata)

# Local stress (quadrature points) and local stress(elemental nodes)
stress_m_quad, coord_quad, stress_m, coord_node = stress_recov.stress_eval(material_parameters, meshdata, strain_m)

# To do- resolve xmdf multiblock (color in paraview)
output_dir = Path('outputs')
output_dir.mkdir(exist_ok=True)
filename = str(output_dir / 'SG_mesh.xdmf')
stress_m.name = "Stress" # Set a name for Paraview
#strain_m.name = "Strain" # Set a name for Paraview
#u_loc.name = "Displacement" # Set a name for Paraview

with io.XDMFFile(meshdata['mesh'].comm, filename, "a") as xdmf:
    xdmf.write_function(stress_m, 0.0)
    #  xdmf.write_function(strain_m, 0.0)

print('\n Time Taken:', str(time.time() - tic))
print('xdmf file written- stress distribution at nodal points')
