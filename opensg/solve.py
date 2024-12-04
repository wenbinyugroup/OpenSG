# Initialization of libraries
from mpi4py import MPI
import numpy as np
import dolfinx
import basix
from dolfinx.fem import form, petsc, Function, functionspace, apply_lifting, set_bc, locate_dofs_topological
from ufl import TrialFunction, TestFunction, inner, lhs, rhs, as_tensor, dot
from scipy.sparse import csr_matrix
import petsc4py.PETSc
from dolfinx.fem.petsc import assemble_matrix
import opensg
import scipy

from mpi4py import MPI
import ufl


### ABD matrix computation
# @profile
# NOTE can pass in thick[ii], nlay[ii], etc instead of the dictionaries
def compute_ABD_matrix(thick, nlay, angle, mat_names, material_database):
    """Compute the ABD matrix for a composite layup structure

    Constructs a local stiffness matrix for a composite laminate
    by assembling the contributions from multiple layers.

    Parameters
    ----------
    ii : _type_
        _description_
    thick : _type_
        _description_
    nlay : _type_
        _description_
    material_parameters : _type_
        _description_
    matid : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """

    ## 1D mesh generation
    # initialization
    deg = 2
    cell = ufl.Cell("interval")
    elem = basix.ufl.element("Lagrange", "interval", 1, shape = (3,))
    domain = ufl.Mesh(elem)

    # nodes (1D SG)
    th, s = [0], 0  # Reference starting point
    for k in thick:
        s = s - k  # Add the thickness of each layer
        th.append(s)
    points = np.array(th)
    
    # elements
    cell = []
    for k in range(nlay):
        cell.append([k, k + 1])
    cellss = np.array(cell)
    
    # Create mesh object
    dom = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cellss, points, domain)
    num_cells = dom.topology.index_map(dom.topology.dim).size_local
    cells = np.arange(num_cells, dtype = np.int32)
    
    #
    # Note: For a layup like layup1: containing 3 layers:
    # we defined 1D mesh with 3 elements and make each element a seperate subdomain(where integration is computed by *dx(i)
    # for subdomain i) using cells-[0,1,2]
    
    # assigning each element as subdomain
    subdomain = dolfinx.mesh.meshtags(dom, dom.topology.dim, cells, cells)
    x, dx = ufl.SpatialCoordinate(dom), ufl.Measure("dx")(domain = dom, subdomain_data = subdomain)
    gamma_e = opensg.compute_utils.create_gamma_e(x)

    nphases = len(cells)

    # Creating FE function Space
    V = functionspace(dom, basix.ufl.element("CG", "interval", deg, shape = (3,)))
    u, v = TrialFunction(V), TestFunction(V)
    
    # Weak form of energy
    F2 = 0
    for j in range(nphases):
        mat_name = mat_names[j]
        material_props = material_database[mat_name]
        theta = angle[j]
        sigma_val = opensg.compute_utils.sigma(u, material_props, theta, Eps = gamma_e[:,0])[0]
        inner_val = inner(sigma_val, opensg.compute_utils.eps(v)[0])
        F2 += inner_val * dx(j)
    
    # lhs gives left hand side of weak form : coeff matrix here
    A = petsc.assemble_matrix(form(lhs(F2)))  
    A.assemble()
    null = opensg.compute_utils.compute_nullspace(V)
    A.setNullSpace(null)  # Set the nullspace
    xx = 3 * V.dofmap.index_map.local_range[1]  # total dofs
    
    # Initialization
    V0, Dhe, D_ee = np.zeros((xx, 6)), np.zeros((xx, 6)), np.zeros((6, 6))

    # Assembly
    for p in range(6):
        Eps = gamma_e[:, p]
        
        # weak form
        F2 = 0
        for j in range(nphases):
            mat_name = mat_names[j]
            material_props = material_database[mat_name]
            theta = angle[j]
            sigma_val = opensg.compute_utils.sigma(u, material_props, theta, Eps)[0]
            inner_val = inner(sigma_val, opensg.compute_utils.eps(v)[0])
            F2 += inner_val * dx(j)
            
        # F2 = sum([inner(sigma(u, i, Eps)[0], eps(v)[0]) * dx(i) for i in range(nphases)])
        
        # rhs is used for getting right hand side of Aw = F; (which is F here)
        F = petsc.assemble_vector(form(rhs(F2)))
        F.ghostUpdate(addv = petsc4py.PETSc.InsertMode.ADD, mode = petsc4py.PETSc.ScatterMode.REVERSE)
        null.remove(F)  # Orthogonalize F to the null space of A^T
        w = opensg.compute_utils.solve_ksp(A, F, V)
        Dhe[:, p] = F[:]  # Dhe matrix formation
        V0[:, p] = w.vector[:]  # V0 matrix formation
    
    # NOTE: fixed..
    D1 = np.matmul(V0.T, -Dhe)  # Additional information matrix

    # Scalar assembly for each term of D_ee matrix
    for s in range(6):  
        for k in range(6):
            # f=dolfinx.fem.form(sum([opensg.compute_utils.Dee(x, u, material_props, theta, Eps)[s,k]*dx(i) for i in range(nphases)])) # Scalar assembly
            f = 0
            for j in range(nphases):
                mat_name = mat_names[j]
                material_props = material_database[mat_name]
                theta = angle[j]
                dee_val = opensg.compute_utils.Dee(x, u, material_props, theta, Eps)[s, k]
                f += dee_val * dx(j)
            f = dolfinx.fem.form(f)
            D_ee[s, k] = dolfinx.fem.assemble_scalar(f)
            
    D_eff = D_ee + D1
    
    return D_eff


### solve stiffness via EB method
# @profile
def compute_stiffness_EB_blade_segment(
    ABD, # array
    mesh, # 
    frame, # 
    subdomains, 
    l_submesh, # dictionary with mesh data for l boundary
    r_submesh # dictionary with mesh data for r boundary
    ):
    """Computes the ABD stiffness matrix using the EB method for a blade segment

    Parameters
    ----------
    ABD : list[array]
        list of ABD matrices for each phase
    mesh : _type_
        _description_
    frame : _type_
        _description_
    subdomains : _type_
        _description_
    l_submesh : _type_
        _description_
    r_submesh : _type_
        _description_

    Returns
    -------
    array
        Stiffness matrix
    """
    tdim = mesh.topology.dim
    fdim = tdim - 1
    nphases = max(subdomains.values[:]) + 1
    assert nphases == len(ABD)
    pp = mesh.geometry.x # point data
    x_min, x_max=min(pp[:,0]), max(pp[:,0])
    
    # Initialize nullspaces
    V_l = dolfinx.fem.functionspace(mesh, basix.ufl.element(
        "S", mesh.topology.cell_name(), 2, shape = (3, )))
        
    V_r = dolfinx.fem.functionspace(mesh, basix.ufl.element(
        "S", mesh.topology.cell_name(), 2, shape = (3, )))
    
    # Compute boundaries
    # NOTE: not the stiffness matrix
    
    V0_l = compute_eb_blade_segment_boundary(ABD, l_submesh)
    V0_r = compute_eb_blade_segment_boundary(ABD, r_submesh)
        
    # The local_frame_l(submesh["mesh"]) can be replaced with frame_l, if we want to use mapped orientation from given direction cosine matrix (orien mesh data-yaml)

    # Quad mesh
    e, V, dv, v, x, dx = opensg.local_boun(mesh, frame, subdomains)
    V0, Dle, Dhe, Dhd, Dld, D_ed, D_dd, D_ee, V1s = opensg.initialize_array(V)
    mesh.topology.create_connectivity(1, 2)
    l_submesh["mesh"].topology.create_connectivity(1, 1)
    r_submesh["mesh"].topology.create_connectivity(1, 1)

    # Obtaining coefficient matrix AA and BB with and without bc applied.
    # Note: bc is applied at boundary dofs. We define v2a containing all dofs of entire wind blade.
    # NOTE: does order of right and left submesh matter here?
    boundary_dofs = locate_dofs_topological(V, fdim, np.concatenate((r_submesh["entity_map"], l_submesh["entity_map"]), axis=0))
    F2=sum([dot(dot(as_tensor(ABD[i]), opensg.gamma_h(e,x,dv)), opensg.gamma_h(e,x,v))*dx(i) for i in range(nphases)])  
    v2a=Function(V) # b default, v2a has zero value for all. 
    bc = dolfinx.fem.dirichletbc(v2a, boundary_dofs) # This shows only boundary_dofs are taken for v2a under bc, which are zero (known) as input.
    a = form(F2)
    
    B = assemble_matrix(a)   # Obtain coefficient matrix without BC applied: BB
    B.assemble()
    ai, aj, av=B.getValuesCSR()
    BB = csr_matrix((av, aj, ai))
    BB = BB.toarray()  
    
    A = assemble_matrix(a,[bc])  # Obtain coefficient matrix with BC applied: AA
    A.assemble()
    ai, aj, av=A.getValuesCSR()
    AA = csr_matrix((av, aj, ai))
    AA=AA.toarray()
    avg=np.trace(AA)/len(AA)     

    # averaging is done so that all terms are of same order. Note after appliying bc at [A=assemble_matrix(a,[bc])], the dofs of
    # coefficientmatrix has only 1 replaced at that dofs. 
    for i,xx in enumerate(av):
        if xx==1:
            av[i]=avg        
                            
    AA_csr = csr_matrix((av, aj, ai))
    AAA = AA_csr.toarray() 
    AA = scipy.sparse.csr_matrix(AAA) 

    # Assembly
    # Running for 4 different F vector. However, F has bc applied to it where, stored known values of v2a is provided for each loop (from boun solve).
    for p in range(4): # 4 load cases meaning 
        # Boundary 
        v2a = Function(V)
        # NOTE: V0 first dimension is degrees of freedom and second dimension is load cases
        v2a = opensg.dof_mapping_quad(V, v2a, V_l, V0_l[:,p], l_submesh["facets"], l_submesh["entity_map"])
        # NOTE: does this second call overwrite previous, or is the data combined? -klb
        v2a = opensg.dof_mapping_quad(V, v2a,V_r,V0_r[:,p], r_submesh["facets"], r_submesh["entity_map"])  
        
        # quad mesh
        F2=sum([dot(dot(as_tensor(ABD[i]),opensg.construct_gamma_e(e,x)[:,p]), opensg.gamma_h(e,x,v))*dx(i) for i in range(nphases)])  
        bc = dolfinx.fem.dirichletbc(v2a, boundary_dofs)
        F = petsc.assemble_vector(form(rhs(F2)))
        apply_lifting(F, [a], [[bc]]) # apply bc to rhs vector (Dhe)
        set_bc(F, [bc])
        with F.localForm() as local_F:
            for i in boundary_dofs:
                for k in range(3):
                    # F[3*i+k]=avg*F[3*i+k] # normalize small terms
                    local_index = 3 * i + k
                    local_F[local_index] = avg * local_F[local_index]
                
        V0[:,p]=  scipy.sparse.linalg.spsolve(AA, F, permc_spec=None, use_umfpack=True) # obtain sol: E* V1s = b*
        Dhe[:,p]= scipy.sparse.csr_array(BB).dot(V0[:,p])
        
    D1 = np.matmul(V0.T,-Dhe) 
    gamma_e = opensg.construct_gamma_e(e,x)
    q1 = gamma_e.T
    q3 = gamma_e
    
    for s in range(4):
        for k in range(4):
            # f = dolfinx.fem.form(sum([dot(dot(opensg.construct_gamma_e(e,x).T,as_tensor(ABD[i])),opensg.construct_gamma_e(e,x))[s,k]*dx(i) for i in range(nphases)])) 
            form_list = []
            for i in range(nphases):
                q2 = as_tensor(ABD[i])
                dot1 = dot(q1, q2)
                dot2 = dot(dot1, q3)
                q4 = dot2[s,k]
                form_list.append(q4*dx(i))
            f = dolfinx.fem.form(sum(form_list))
            
            D_ee[s,k]=dolfinx.fem.assemble_scalar(f)
    L = (x_max - x_min)
    D_eff= D_ee + D1
    D_eff=D_eff/L  # L is divided because of 3D shell mesh and corresponding beam length need to divided.
    #--------------------------------------Printing Output Data---------------------------------------
    print('  ')  
    print('Stiffness Matrix')
    np.set_printoptions(precision=4)
    print(np.around(D_eff))
    
    return D_eff

def compute_eb_blade_segment_boundary(
    ABD, # array
    submeshdata, # dictionary with mesh data for l boundary
    # nphases
    # r_submesh # dictionary with mesh data for r boundary
    ):
    
    # Initialize terms
    # e, V, dv, v, x, dx = opensg.local_boun(
    #     submeshdata["mesh"], submeshdata["frame"], submeshdata["subdomains"])
    V = dolfinx.fem.functionspace(submeshdata["mesh"], basix.ufl.element(
        "S", submeshdata["mesh"].topology.cell_name(), 2, shape = (3, )))
    
    submeshdata["nullspace"] = opensg.compute_nullspace(V)
    
    V0 = opensg.solve_eb_boundary(ABD, submeshdata)
    
    return V0
    

def compute_timo_boun(ABD, mesh, subdomains, frame, nullspace, sub_nullspace, nphases):

    mesh = opensg.local_frame_1D(mesh)
    e, V_l, dv, v_, x, dx = opensg.compute_utils.local_boun(mesh,frame,subdomains)          
    mesh.topology.create_connectivity(1, 1)
    V0,Dle,Dhe,Dhd,Dld,D_ed,D_dd,D_ee,V1s = opensg.compute_utils.initialize_array(V_l)
    A_l = opensg.A_mat(e,x,dx,nullspace(V_l),v_,dv, nphases)
    
    for p in range(4):
        Eps = opensg.compute_utils.gamma_e(e,x)[:,p]
        F2 = sum([dot(dot(as_tensor(ABD[i]),Eps), opensg.compute_utils.gamma_h(e,x,v_))*dx(i) for i in range(nphases)])   
        r_he = form(rhs(F2))
        F_l = petsc.assemble_vector(r_he)
        F_l.ghostUpdate(addv = petsc4py.PETSc.InsertMode.ADD, mode = petsc4py.PETSc.ScatterMode.REVERSE)
        sub_nullspace.remove(F_l)
        w_l = opensg.compute_utils.solve_ksp(A_l,F_l,V_l)
        Dhe[:,p]=  F_l[:]
        V0[:,p]= w_l.vector[:]
    D1 = np.matmul(V0.T,-Dhe)   
    for s in range(4):
        for k in range(4): 
            f = dolfinx.fem.form(sum([dot(dot(opensg.compute_utils.gamma_e(e,x).T,as_tensor(ABD[i])),opensg.compute_utils.gamma_e(e,x))[s,k]*dx(i) for i in range(nphases)]))
            D_ee[s,k] = dolfinx.fem.assemble_scalar(f)
        
    D_eff= D_ee + D1 # Effective Stiffness Matrix (EB)
    print('EB Computed')
    F1 = sum([dot(dot(as_tensor(ABD[i]),opensg.compute_utils.gamma_l(e,x,v_)),opensg.compute_utils.gamma_l(e,x,dv))*dx(i) for i in range(nphases)])
    a1 = form(F1)
    Dll = assemble_matrix(a1)
    Dll.assemble()
    ai, aj, av = Dll.getValuesCSR()
    Dll=  csr_matrix((av, aj, ai)).toarray()
    
    for p in range(4):
            Eps = opensg.compute_utils.gamma_e(e,x)[:,p] 
            F1 = sum([dot(dot(as_tensor(ABD[i]),Eps),opensg.compute_utils.gamma_l(e,x,v_))*dx(i) for i in range(nphases)])
            Dle[:,p]= petsc.assemble_vector(form(F1))[:]
        
            Eps = opensg.compute_utils.gamma_d(e,x)[:,p] 
            F1 = sum([dot(dot(as_tensor(ABD[i]),Eps),opensg.compute_utils.gamma_h(e,x,v_))*dx(i) for i in range(nphases)])      
            Dhd[:,p]= petsc.assemble_vector(form(F1))[:]
        
            Eps = opensg.compute_utils.gamma_d(e,x)[:,p] 
            F1 = sum([dot(dot(as_tensor(ABD[i]),Eps),opensg.compute_utils.gamma_l(e,x,v_))*dx(i) for i in range(nphases)])      
            Dld[:,p]= petsc.assemble_vector(form(F1))[:]
        
    F_dhl = sum([dot(dot(as_tensor(ABD[i]),opensg.compute_utils.gamma_h(e,x,dv)),opensg.compute_utils.gamma_l(e,x,v_))*dx(i) for i in range(nphases)]) 
    a3 = form(F_dhl)
    Dhl = assemble_matrix(a3)
    Dhl.assemble()
    ai, aj, av = Dhl.getValuesCSR()
    Dhl = csr_matrix((av, aj, ai)).toarray()
    for s in range(4):
        for k in range(4):    
            f = dolfinx.fem.form(sum([dot(dot(opensg.compute_utils.gamma_e(e,x).T,as_tensor(ABD[i])),opensg.compute_utils.gamma_d(e,x))[s,k]*dx(i) for i in range(nphases)]))
            D_ed[s,k] = dolfinx.fem.assemble_scalar(f)
    
            f = dolfinx.fem.form(sum([dot(dot(opensg.compute_utils.gamma_d(e,x).T,as_tensor(ABD[i])),opensg.compute_utils.gamma_d(e,x))[s,k]*dx(i) for i in range(nphases)]))
            D_dd[s,k] = dolfinx.fem.assemble_scalar(f) 
        
    #DhlTV0
    DhlV0 = np.matmul(Dhl.T,V0)
    
    #DhlTV0Dle
    DhlTV0Dle = np.matmul(Dhl,V0)+Dle
    
    #V0DllV0
    V0DllV0 = np.matmul(np.matmul(V0.T,Dll),V0)
    
    # V1s
    b = DhlTV0Dle-DhlV0-Dhd
    
    for p in range(4):
        F = petsc4py.PETSc.Vec().createWithArray(b[:,p],comm = MPI.COMM_WORLD)
        F.ghostUpdate(addv = petsc4py.PETSc.InsertMode.ADD, mode = petsc4py.PETSc.ScatterMode.REVERSE)
        sub_nullspace.remove(F)
        w_l = opensg.compute_utils.solve_ksp(A_l,F,V_l)
        V1s[:,p]= w_l.vector[:]  
        
    # Ainv
    Ainv = np.linalg.inv(D_eff)
    
    # B_tim
    B_tim= np.matmul(DhlTV0Dle.T,V0)+ D_ed+ np.matmul(V0.T,Dhd)
    
    # C_tim
    C_tim= V0DllV0 + np.matmul(V1s.T,DhlV0 + DhlTV0Dle)+2*np.matmul(V0.T,Dld)+ D_dd+np.matmul(V1s.T,Dhd)
    C_tim = 0.5*(C_tim+C_tim.T)
    
    # Ginv
    Q_tim = np.matmul(Ainv,np.array([(0,0),(0,0),(0,-1),(1,0)]))
    Ginv= np.matmul(np.matmul(Q_tim.T,(C_tim-np.matmul(np.matmul(B_tim.T,Ainv),B_tim))),Q_tim)
    G_tim = np.linalg.inv(Ginv)
    Y_tim= np.matmul(np.matmul(B_tim.T,Q_tim),G_tim)
    A_tim= D_eff + np.matmul(np.matmul(Y_tim,Ginv),Y_tim.T)
    
    # Deff_srt
    D = np.zeros((6,6))
    
    D[4:6,4:6] = G_tim
    D[0:4,4:6] = Y_tim
    D[4:6,0:4] = Y_tim.T
    D[0:4,0:4] = A_tim
    
    Deff_srt = np.zeros((6,6))
    Deff_srt[0,3:6] = A_tim[0,1:4]
    Deff_srt[0,1:3] = Y_tim[0,:]
    Deff_srt[0,0] = A_tim[0,0]
    
    Deff_srt[3:6,3:6] = A_tim[1:4,1:4]
    Deff_srt[3:6,1:3] = Y_tim[1:4,:]
    Deff_srt[3:6,0] = A_tim[1:4,0]
    
    Deff_srt[1:3,1:3] = G_tim
    Deff_srt[1:3,3:6] = Y_tim.T[:,1:4]
    Deff_srt[1:3,0] = Y_tim.T[:,0]
    
    return np.around(D_eff), np.around(Deff_srt)