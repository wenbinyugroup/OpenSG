import numpy as np
import time
import opensg.utils as utils
from opensg.mesh.segment import SolidSegmentMesh
from opensg.core.solid import compute_stiffness
import opensg.core.stress_recov as stress_recov
tic = time.time()

taper_mass, taper_origin, taper_timo=[], [], []
file_name='data/Solid_3DSG/bar_urc_npl_1_ar_5-segment_' 
beam_out=utils.beamdyn_trans.beam_reaction(file_name)

for segment in np.linspace(0,28,29): # np.linspace(start segment, end segment, (end_segment-start segment+1))
    segid=int(segment)
    # Read 3D yaml
    mesh_yaml=file_name+ str(segid) +'.yaml' 
    segment_mesh = SolidSegmentMesh(mesh_yaml)
    print('\nSegment Origin:', segment_mesh.meshdata["origin"])
    # Extract material parameters and mesh data
    material_parameters, density = segment_mesh.material_database
    meshdata = segment_mesh.meshdata
    l_submesh, r_submesh = segment_mesh.left_submesh, segment_mesh.right_submesh
    
    # Compute stiffness and mass matrices 
    timo = compute_stiffness(material_parameters, segment_mesh.meshdata, 
                             l_submesh, r_submesh, Taper=True)
    print('\n Taper Stiffness \n', timo[0])
    taper_origin.append(segment_mesh.meshdata["origin"])

    taper_mass.append(utils.solid.get_mass_solid(segment_mesh.meshdata,density,Taper=True))
    
    taper_timo.append(timo[0])
    
    if segid==0:
        timo_left= timo[3]
        origin_left= l_submesh['origin']
        mass_left= utils.solid.get_mass_solid(l_submesh, density)
    if segid==28:
        timo_right=timo[4]
        origin_right= r_submesh['origin']
        mass_right=utils.solid.get_mass_solid(r_submesh, density)
        
    # Local strain and local displacement 
    strain_m,u_loc=stress_recov.local_strain(timo,beam_out,segid,meshdata) 
    
    # Local stress (quadrature points) and local stress(elemental nodes)
    stress_m_quad, coord_quad, stress_m_node, coord_node=stress_recov.stress_eval(material_parameters, meshdata, strain_m)
    
    # Eigenvalue
    eigen= stress_recov.eigen_solve(material_parameters,
                                        meshdata,
                                        l_submesh,
                                        r_submesh,
                                        strain_m,
                                        u_loc)    
    print('\nComputed Eigenvalue:', eigen)
    print('\n Time Taken:',str(time.time()-tic),'sec')

# #Prepend root to segment values
taper_timo.insert(0,timo_left)
taper_origin.insert(0,origin_left)
taper_mass.insert(0,mass_left)

#Append tip to segment values
taper_timo.append(timo_right)
taper_origin.append(origin_right)
taper_mass.append(mass_right)

print('\ntaper_origin')
print(taper_origin)

utils.shared.write_beamdyn_files(np.array(taper_timo), np.array(taper_mass), taper_origin,'bar_urc_segment_npl1ar5_20251121')
