import numpy as np
import time
import opensg.core.stress_recov as stress_recov
from opensg.mesh.segment import SolidSegmentMesh
from opensg.core.solid import compute_stiffness
import opensg
import opensg.utils.solid as utils


np.set_printoptions(precision=4)
left_timo,right_timo, taper_timo=[],[],[]
left_mass,right_mass, taper_mass=[],[],[]
blade_length = 100
left_origin,right_origin, taper_origin=[],[],[] 
out_file_name='bar_urc_npl_1_ar_5-segment_'
beam_out=opensg.utils.beamdyn_trans.beam_reaction(out_file_name)
eig=[]
time_seg=[]

# Process segments (adjust range based on desired segments)
for segment in np.linspace(0,28,29): # np.linspace(start segment, end segment, (end_segment-start segment+1))
    file_name='bar_urc_npl_1_ar_5-segment_'
    segid=int(segment)
    tic = time.time()
    print('Segment:',str(segid)," \n")
    
    
    # Generate mesh for current segment using StandaloneSolidSegmentMesh
    mesh_yaml = file_name+ str(int(segment)) +'.yaml' 
    segment_mesh = SolidSegmentMesh(mesh_yaml)

    # The solid segment mesh provides the necessary data for stiffness computation
    
    # Extract material parameters and mesh data
    material_parameters, density = segment_mesh.material_database
    meshdata = segment_mesh.meshdata
    l_submesh = segment_mesh.left_submesh
    r_submesh = segment_mesh.right_submesh
    
    # Compute stiffness matrices using the solid analysis function
    timo = compute_stiffness(
        material_parameters,
        meshdata,
        l_submesh,
        r_submesh,
        Taper=True)
    
    left_origin.append(segment_mesh.left_submesh["origin"])
    right_origin.append(segment_mesh.right_submesh["origin"])
    taper_origin.append(segment_mesh.meshdata["origin"])


    # Dehomogenization
    strain_mm,_,u_loc=stress_recov.local_strain(timo,beam_out,segment,meshdata) # Local Strain
    
    # Eigen Solver
    eigen= opensg.core.stress_recov.eigen_solve(material_parameters,
                                    meshdata,
                                    l_submesh,
                                    r_submesh,
                                    strain_mm,
                                    u_loc)    
    eig.append(eigen)

    # Mass Matrix
    
    left_mass.append(utils.get_mass_solid(segment_mesh.left_submesh, density))
    
    right_mass.append(utils.get_mass_solid(segment_mesh.right_submesh,density))
    taper_mass.append(utils.get_mass_solid(segment_mesh.meshdata,density,Taper=True))
    
    taper_timo.append(timo[0])
    left_timo.append(timo[1])
    right_timo.append(timo[2]) 
    
    # Store results

    time_seg.append(time.time()-tic)
  
    print('\n Left mass \n',left_mass[int(segment)])
  #  print('\n Right mass \n',right_mass[int(segment)])
    print('\n Taper mass \n',taper_mass[int(segment)])
    
    print('\n Left timo \n',left_timo[int(segment)])
  #  print('\n Right timo \n',right_timo[int(segment)])
    print('\n Taper timo \n',taper_timo[int(segment)])
    
    print('\n Time Taken',str(time.time()-tic))
  
    
  
print('Total Time for Wind Blade (in sec)', np.sum(time_seg))  
print('\nleft_origin')
print(left_origin)

print('\nright_origin')
print(right_origin)

print('\ntaper_origin')
print(taper_origin)
#Append tip to left boundary values
left_timo.append(right_timo[-1])
left_origin.append(right_origin[-1])
left_mass.append(right_mass[-1])
print('\n\nleft_origin ###############')
print(left_origin)

#Prepend root to right boundary values
right_timo.insert(0,left_timo[0])
right_origin.insert(0,left_origin[0])
right_mass.insert(0,left_mass[0])
print('\n\nright_origin ###############')
print(right_origin)

# #Prepend root to segment values
taper_timo.insert(0,left_timo[0])
taper_origin.insert(0,left_origin[0])
taper_mass.insert(0,left_mass[0])
print('\n\ntaper_origin ###############')
print(taper_origin)

#Append tip to segment values
taper_timo.append(right_timo[-1])
taper_origin.append(right_origin[-1])
taper_mass.append(right_mass[-1])
#print('\n\ntaper_origin ###############')
#print(taper_origin)

left_timo=np.array(left_timo)
right_timo=np.array(right_timo)
taper_timo=np.array(taper_timo)

left_mass=np.array(left_mass)
right_mass=np.array(right_mass)
taper_mass=np.array(taper_mass)

print(taper_origin)
opensg.utils.shared.write_beamdyn_files(taper_timo, taper_mass, taper_origin,'bar_urc_segment_npl1ar5_20251003')
opensg.utils.shared.write_beamdyn_files(right_timo, right_mass, right_origin,'bar_urc_right_npl1ar5_20251003')
opensg.utils.shared.write_beamdyn_files(left_timo, left_mass, left_origin,'bar_urc_left_npl1ar5_20251003')



    
