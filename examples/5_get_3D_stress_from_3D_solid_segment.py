import time
from pathlib import Path
import opensg.utils as utils
from opensg.mesh.segment import SolidSegmentMesh
from opensg.core.solid import compute_stiffness
import opensg.core.stress_recov as stress_recov
from dolfinx import io

tic = time.time()

data_dir = Path('data')
segid = 0 # user defined
# Read 3D yaml
mesh_yaml = data_dir / 'Solid_3DSG' / f'bar_urc_npl_1_ar_5-segment_{segid}.yaml'
beam_out = utils.beamdyn_trans.beam_reaction(data_dir / 'bd_bar_urc.out')

# Extract material parameters and mesh data
segment_mesh = SolidSegmentMesh(mesh_yaml)
material_parameters, density = segment_mesh.material_database
meshdata = segment_mesh.meshdata
l_submesh, r_submesh = segment_mesh.left_submesh, segment_mesh.right_submesh

# Compute stiffness and mass matrices
timo = compute_stiffness(material_parameters, segment_mesh.meshdata,
                         l_submesh, r_submesh, Taper=True)
print('\n Taper Stiffness \n', timo[0])

# Local strain and local displacement
strain_m, u_loc = stress_recov.local_strain(timo, beam_out, segid, meshdata)

# Local stress (quadrature points) and local stress(elemental nodes)
stress_m_quad, coord_quad, stress_m, coord_node = stress_recov.stress_eval(material_parameters, meshdata, strain_m)

# Eigenvalue (Buckling load factor)
eigen = stress_recov.eigen_solve(material_parameters, meshdata, l_submesh, r_submesh, strain_m, u_loc)
print('Computed Eigenvalue:', eigen)
print('\n Time Taken: 3D yaml', str(time.time() - tic))

output_dir = Path('outputs')
output_dir.mkdir(exist_ok=True)
filename = str(output_dir / 'SG_mesh.xdmf')
stress_m.name = "Stress" #
#strain_m.name = "Strain" # To obtain local strain
#u_loc.name = "Displacement" # To obtain local displacement

with io.XDMFFile(meshdata['mesh'].comm, filename, "a") as xdmf:
    xdmf.write_function(stress_m, 0.0)
    #  xdmf.write_function(strain_m, 0.0)

print('xdmf file written- stress distribution at nodal points')
