
OpenSG requires user to input the mesh data in .msh format with
 
 1.The defined physical surfaces (2D)/ volumes (3D) for seperating material phases in connectivity matrix (like in gmsh),
	the material-ID numbering should starts from 0.
 2.If PreVABS is used to generate the 2D cross section mesh, the user needs to define cross-section in xy plane. 
   (* OpenSG_PreVABS_modify.m is provided to modify .msh file for direct use in OpenSG*)


Examples are provided for 2D and 3D SG (.msh files) which can directly be used in beam, plate or 3D model OpenSG code. 

