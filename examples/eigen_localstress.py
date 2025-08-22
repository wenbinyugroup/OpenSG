import opensg
import numpy as np
import time

import opensg.mesh.blade


out_file_name='bar_urc_npl_2_ar_10-segment_'
beam_force=opensg.core.stress_recov.beam_reaction(out_file_name)

for segment in np.linspace(1,1,1): # np.linspace(start segment, end segment, (end_segment-start segment+1))
    file_name='bar_urc_npl_2_ar_10/bar_urc_npl_2_ar_10-segment_'
    tic = time.time()
    print("\n Computing Segment:",str(int(segment))," \n")

    # read yaml mesh
    mesh_yaml = file_name+ str(int(segment)) +'.yaml'  ## the name of the yaml file containing the whole blade mesh
    mesh_data = opensg.io.load_yaml(mesh_yaml)
  
   # Blade Segment Mesh 
    blade_mesh = opensg.mesh.blade.SolidBladeMesh(mesh_data)  
    segment_mesh=blade_mesh.generate_segment_mesh(segment_index=0, filename="section.msh")
    print('\n Mesh Time of:'+str(blade_mesh.num_elements),"elements segment is ", time.time()-tic)
  
    # Homogenization
    timo=opensg.core.solid.compute_stiffness(segment_mesh.material_database[0],
                                    segment_mesh.meshdata,
                                    segment_mesh.left_submesh,
                                    segment_mesh.right_submesh)
    
    # Dehomogenization
    strain_3D=opensg.core.stress_recov.recover_local_strain(timo,beam_force,segment,segment_mesh.meshdata) # Local 3D Strain
    
    # Eigen Solver
    eigen= opensg.core.stress_recov.eigen_stiffness_matrix(segment_mesh.material_database[0],segment_mesh,strain_3D, 1)
    
    # Note:segment_mesh.material_database[1] contains densities- to be used for mass matrix computaion (found in compute_utils).
    
    # Local Stress Path (Optional)
    file_name='solid.lp_sparcap_center_thickness_001' 
    points=np.loadtxt(file_name, skiprows=0, dtype=float) # Load path coordinates
    eval_data=opensg.core.stress_recov.local_stress(segment_mesh.material_database[0],segment_mesh, strain_3D,points)
    for idx,pt in enumerate(points):
        print(' \n Point:',[float(i) for i in pt],' \n   Stress Vector:', eval_data[idx])
        
    # If it shows error in points that means, no such coordinate points found in the current segment. 
    # Suppress Local Stress Path- it is under working and needs validation
    
    print('\n Total Time for Segment',str(int(segment)),' :',time.time()-tic)
