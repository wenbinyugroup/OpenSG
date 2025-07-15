# Initialization of libraries
from mpi4py import MPI
import numpy as np
import dolfinx
import basix
from dolfinx.fem import (
    form,
    petsc,
    Function,
    functionspace,
    apply_lifting,
    set_bc,
    locate_dofs_topological,
)
from ufl import TrialFunction, TestFunction, inner, lhs, rhs, as_tensor, dot, SpatialCoordinate, Measure, as_vector
from scipy.sparse import csr_matrix
import petsc4py.PETSc
from dolfinx.fem.petsc import assemble_matrix
from ..utils import compute_utils as utils
import scipy

from mpi4py import MPI
import ufl


### ABD matrix computation
# @profile
# NOTE can pass in thick[ii], nlay[ii], etc instead of the dictionaries
def compute_ABD_matrix_old(thick, nlay, angle, mat_names, material_database):
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
    elem = basix.ufl.element("Lagrange", "interval", 1, shape=(3,))
    domain = ufl.Mesh(elem)

    # nodes (1D SG)
    th, s = [0], 0  # Reference starting point
    for k in thick:
        s = s + k  # Add the thickness of each layer
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
    cells = np.arange(num_cells, dtype=np.int32)

    #
    # Note: For a layup like layup1: containing 3 layers:
    # we defined 1D mesh with 3 elements and make each element a seperate subdomain(where integration is computed by *dx(i)
    # for subdomain i) using cells-[0,1,2]

    # assigning each element as subdomain
    subdomain = dolfinx.mesh.meshtags(dom, dom.topology.dim, cells, cells)
    x, dx = ufl.SpatialCoordinate(dom), ufl.Measure("dx")(
        domain=dom, subdomain_data=subdomain
    )
    gamma_e = utils.create_gamma_e(x)

    nphases = len(cells)

    # Creating FE function Space
    V = functionspace(dom, basix.ufl.element("CG", "interval", deg, shape=(3,)))
    u, v = TrialFunction(V), TestFunction(V)

    # Weak form of energy
    F2 = 0
    for j in range(nphases):
        mat_name = mat_names[j]
        material_props = material_database[mat_name]
        theta = angle[j]
        sigma_val = utils.sigma(
            u, material_props, theta, Eps=gamma_e[:, 0]
        )[0]
        inner_val = inner(sigma_val, utils.eps(v)[0])
        F2 += inner_val * dx(j)

    # lhs gives left hand side of weak form : coeff matrix here
    A = petsc.assemble_matrix(form(lhs(F2)))
    A.assemble()
    null = utils.compute_nullspace(V)
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
            sigma_val = utils.sigma(u, material_props, theta, Eps)[0]
            inner_val = inner(sigma_val, utils.eps(v)[0])
            F2 += inner_val * dx(j)

        # F2 = sum([inner(sigma(u, i, Eps)[0], eps(v)[0]) * dx(i) for i in range(nphases)])

        # rhs is used for getting right hand side of Aw = F; (which is F here)
        F = petsc.assemble_vector(form(rhs(F2)))
        F.ghostUpdate(
            addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE
        )
        null.remove(F)  # Orthogonalize F to the null space of A^T
        w = utils.solve_ksp(A, F, V)
        Dhe[:, p] = F[:]  # Dhe matrix formation
        V0[:, p] = w.vector[:]  # V0 matrix formation

    # NOTE: fixed..
    D1 = np.matmul(V0.T, -Dhe)  # Additional information matrix

    # Scalar assembly for each term of D_ee matrix
    for s in range(6):
        for k in range(6):
            # f=dolfinx.fem.form(sum([utils.Dee(x, u, material_props, theta, Eps)[s,k]*dx(i) for i in range(nphases)])) # Scalar assembly
            f = 0
            for j in range(nphases):
                mat_name = mat_names[j]
                material_props = material_database[mat_name]
                theta = angle[j]
                dee_val = utils.Dee(x, u, material_props, theta, Eps)[
                    s, k
                ]
                f += dee_val * dx(j)
            f = dolfinx.fem.form(f)
            D_ee[s, k] = dolfinx.fem.assemble_scalar(f)

    D_eff = D_ee + D1

    return D_eff


# Generate ABD matrix (Plate model)
def compute_ABD_matrix(thick, nlay, angle, mat_names, material_database):
    """
    MSG based Kirchoff plate stiffness matrix of composite laminates
    
    Parameters:
        ii: Layup id

    Returns:
        ABD: [6,6] ! Plate Stiffness Matrix

    """  
    deg = 2
    cell = ufl.Cell("interval")
    elem= basix.ufl.element("Lagrange", "interval", 1, shape=(3, ))
    domain = ufl.Mesh(elem)

    # Nodes (1D SG)
    th,s=[0],0 # Reference-------- 0
    for k in thick:
        s = s + k           # Inward normal in orien provided by yaml file
        th.append(s)
    points = np.array(th) 
    # Elements  
    cell=[]
    for k in range(nlay):
        cell.append([k,k+1])   
    cellss = np.array(cell)
    
    # Create 1D SG mesh
    dom = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cellss, points, domain)
    num_cells= dom.topology.index_map(dom.topology.dim).size_local 
    cells = np.arange(num_cells, dtype=np.int32)
    subdomain = dolfinx.mesh.meshtags(dom, dom.topology.dim, cells, cells) # assigning each element as subdomain
    x,dx = SpatialCoordinate(dom), Measure('dx')(domain=dom, subdomain_data=subdomain, metadata={"quadrature_degree": 4})

    gamma_e=as_tensor([(1,0,0,x[0],0,0),       # Gamma_e matrix
                      (0,1,0,0,x[0],0),
                      (0,0,0,0,0,0),
                      (0,0,0,0,0,0),
                      (0,0,0,0,0,0),
                      (0,0,1,0,0,x[0])])  

    nphases = len(cells)
    
    def gamma_h(v): # (Gamma_h * w)
        E1= as_vector([0,0,v[2].dx(0),(v[1].dx(0)),(v[0].dx(0)),0])
        return E1

    def R_sig(C,t): # Rotation matrix
        """
        Performs rotation from local material frame to global frame
        
        Parameters:
            C: [6,6] numpy array  ! Stiffness matrix
            t: constant           ! rotation angle
            
        Returns:
            C': Rotated Stiffness matrix
        """
        th=np.deg2rad(t)
        c,s,cs=np.cos(th),np.sin(th),np.cos(th)*np.sin(th)
        R_Sig= np.array([(c**2, s**2, 0,0,0,-2*cs),
                   (s**2, c**2, 0,0,0,2*cs),
                   (0,0,1,0,0,0),
                   (0,0,0,c,s,0),
                   (0,0,0,-s,c,0),
                   (cs,-cs,0,0,0,c**2-s**2)])
        return np.matmul(np.matmul(R_Sig,C),R_Sig.transpose())

    def Stiff_mat(material_parameters, theta):     
        """
        Compute the [6,6] Stiffness matrix using material elastic constants
        
        Parameters:
            i: Material parameters id/ matid
            
        Returns:
            C: [6,6] Stiffness Matrix
        """
        # E1,E2,E3,G12,G13,G23,v12,v13,v23= material_parameters[matid[ii][i]]
        E1, E2, E3 = material_parameters["E"]
        G12, G13, G23 = material_parameters["G"]
        v12, v13, v23 = material_parameters["nu"]
        S=np.zeros((6,6))
        S[0,0], S[1,1], S[2,2]=1/E1, 1/E2, 1/E3
        S[0,1], S[0,2]= -v12/E1, -v13/E1
        S[1,0], S[1,2]= -v12/E1, -v23/E2
        S[2,0], S[2,1]= -v13/E1, -v23/E2
        S[3,3], S[4,4], S[5,5]= 1/G23, 1/G13, 1/G12    
        C=np.linalg.inv(S)
        # theta=angle[ii][i] # ii denotes the layup id
        C= as_tensor(R_sig(C,theta)) 
        return  C
        
    # Creating FE function Space
    V = functionspace(dom, basix.ufl.element("CG", "interval", deg, shape=(3, )))
    dv,v_ = TrialFunction(V), TestFunction(V)
    F2 = sum([dot(dot(Stiff_mat(material_database[mat_names[i]], angle[i]),gamma_h(dv)),gamma_h(v_))*dx(i) for i in range(nphases)]) # Weak form of energy(load vec)
    A=  petsc.assemble_matrix(form(F2))
    A.assemble()
    null = utils.compute_nullspace(V)
    A.setNullSpace(null)      # Set the nullspace
    ndofs = 3*V.dofmap.index_map.local_range[1] # total dofs
    # Initialization
    V0, Dhe, D_ee = np.zeros((ndofs,6)), np.zeros((ndofs,6)), np.zeros((6,6))

    # Assembly ! 6 Load Cases
    for p in range(6):
        # Right Hand Side vector in weak form (F2(v_))
        # < gamma_e[:,p].T Stiffness_Matrix gamma_h>
        F2= -sum([dot(dot(Stiff_mat(material_database[mat_names[i]], angle[i]),gamma_e[:,p]),gamma_h(v_))*dx(i) for i in range(nphases)]) 
        F = petsc.assemble_vector(form(F2))
        F.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)
        null.remove(F)                        # Orthogonalize F to the null space of A^T
        Dhe[:,p]= F[:]                        # Dhe matrix formation
        w = utils.solve_ksp(A,F,V)
        V0[:,p]= w.vector[:]          # Solved Fluctuating Functions
    D1=np.matmul(V0.T,-Dhe)
    
    def Dee_abd(x, material_parameters, theta):  
        """
        Performs < gamma_e.T Stiffness_matrix gamma_e > and give simplified form
        
        Parameters:
            i: matid 
     
        Returns:
            Dee: [6,6] ufl tensor 
        """
        C = Stiff_mat(material_parameters, theta)
        x0 = x[0]
        return as_tensor([(C[0,0],C[0,1],C[0,5],x0*C[0,0],x0*C[0,1],x0*C[0,5]),
                      (C[1,0],C[1,1],C[1,5],x0*C[1,0],x0*C[1,1],x0*C[1,5]),
                      (C[5,0],C[5,1],C[5,5],x0*C[5,0],x0*C[5,1],x0*C[5,5]),
                      (x0*C[0,0],x0*C[0,1],x0*C[0,5],x0*x0*C[0,0],x0*x0*C[0,1],x0*x0*C[0,5]),
                      (x0*C[1,0],x0*C[1,1],x0*C[1,5],x0*x0*C[1,0],x0*x0*C[1,1],x0*x0*C[1,5]),
                     (x0*C[5,0],x0*C[5,1],x0*C[5,5],x0*x0*C[5,0],x0*x0*C[5,1],x0*x0*C[5,5])])
    
    # Scalar assembly
    for s in range(6):
        for k in range(6):
            f=dolfinx.fem.form(sum([
                Dee_abd(x, material_database[mat_names[i]], angle[i])[s,k]*dx(i) for i in range(nphases)])) 
            D_ee[s,k]=dolfinx.fem.assemble_scalar(f)   # D_ee [6,6]  

    D_eff = D_ee + D1 
    return(D_eff)


# def compute_timo_boun(ABD, mesh, subdomains, frame, nullspace, sub_nullspace, nphases):
def compute_timo_boun(ABD, boundary_submeshdata, nh):
    """
    Solve EB and Timo model for boundary mesh.
    The output flcutuating functions V0 and V1s are used dirichilet boundary constraints

    Parameters:
        mesh_l: Left Boundary mesh
        subdomains_l:  Left Boundary subdomains (layup information)
        frame_l: local orientation frame (DG space)
    Returns:
        D_eff (np array): [4,4] !Boundary EB Stiffness matrix
        Deff_srt (np array):[6,6] !Boundary Timoshenko Stiffness matrix
        V0 (np array): [ndofs_leftmesh,4] !Boundary fluctuating function solutions after solving load cases (useful in WB segment EB)
        V1s (np array):[ndofs_leftmesh,4] !Boundary fluctuating function solutions after solving load cases (useful in WB segment Timoshenko Stiffness)
    """
    boundary_mesh = boundary_submeshdata["mesh"]
    boundary_subdomains = boundary_submeshdata["subdomains"]
    boundary_frame = boundary_submeshdata["frame"]
    # frame override:
    boundary_frame = utils.local_frame_1D(boundary_mesh)

    nphases = len(ABD)

    e, V_l, dv, v_, x, dx = utils.local_boun(
        boundary_mesh, boundary_frame, boundary_subdomains
    )
    boundary_mesh.topology.create_connectivity(1, 1)
    V0, Dle, Dhe, D_ee, V1s = utils.initialize_array(V_l)
    nullspace_l = utils.compute_nullspace(V_l)
    #   A_l=A_mat(e,x,dx,nullspace(V_l),v_,dv,mesh_l)
    # Compute A_mat
    F2 = sum(
        [
            dot(
                dot(as_tensor(ABD[i]), utils.gamma_h(e, x, dv)),
                utils.gamma_h(e, x, v_),
            )
            * dx(i)
            for i in range(nphases)
        ]
    )
    ff = utils.deri_constraint(dv, v_, boundary_mesh, nh)
    A_l = assemble_matrix(form(F2 + ff))
    A_l.assemble()
    A_l.setNullSpace(nullspace_l)

    gamma_e = utils.gamma_e(e, x)
    for p in range(4):
        Eps = gamma_e[:, p]
        F2 = sum(
            [
                dot(dot(as_tensor(ABD[i]), Eps), utils.gamma_h(e, x, v_)) * dx(i)
                for i in range(nphases)
            ]
        )
        r_he = form(rhs(F2))
        F_l = petsc.assemble_vector(r_he)
        F_l.ghostUpdate(
            addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE
        )
        nullspace_l.remove(F_l)
        Dhe[:, p] = petsc.assemble_vector(r_he)[:]
        w = utils.solve_ksp(A_l, F_l, V_l)
        V0[:, p] = w.vector[:]

    D1 = np.matmul(V0.T, -Dhe)
    for s in range(4):
        for k in range(4):
            f = dolfinx.fem.form(
                sum(
                    [
                        dot(dot(gamma_e.T, as_tensor(ABD[i])), gamma_e)[s, k] * dx(i)
                        for i in range(nphases)
                    ]
                )
            )
            D_ee[s, k] = dolfinx.fem.assemble_scalar(f)

    D_eff = D_ee + D1  # Effective Stiffness Matrix (EB)
    F1 = sum(
        [
            dot(
                dot(as_tensor(ABD[i]), utils.gamma_l(e, x, v_)),
                utils.gamma_l(e, x, dv),
            )
            * dx(i)
            for i in range(nphases)
        ]
    )
    a1 = form(F1)
    Dll = assemble_matrix(a1)
    Dll.assemble()
    ai, aj, av = Dll.getValuesCSR()
    Dll = csr_matrix((av, aj, ai)).toarray()

    for p in range(4):
        Eps = gamma_e[:, p]
        F1 = sum(
            [
                dot(dot(as_tensor(ABD[i]), Eps), utils.gamma_l(e, x, v_)) * dx(i)
                for i in range(nphases)
            ]
        )
        Dle[:, p] = petsc.assemble_vector(form(F1))[:]

    F_dhl = sum(
        [
            dot(
                dot(as_tensor(ABD[i]), utils.gamma_h(e, x, dv)),
                utils.gamma_l(e, x, v_),
            )
            * dx(i)
            for i in range(nphases)
        ]
    )
    ff = utils.deri_constraint(dv, v_, boundary_mesh, nh)
    a3 = form(F_dhl + ff)
    Dhl = assemble_matrix(a3)
    Dhl.assemble()
    ai, aj, av = Dhl.getValuesCSR()
    Dhl = csr_matrix((av, aj, ai)).toarray()

    # DhlTV0
    DhlV0 = np.matmul(Dhl.T, V0)

    # DhlTV0Dle
    DhlTV0Dle = np.matmul(Dhl, V0) + Dle

    # V0DllV0
    V0DllV0 = np.matmul(np.matmul(V0.T, Dll), V0)

    # V1s  ****Updated from previous version as for solving boundary V1s, we can directly use (A_l V1s=b),  and solve for V1s
    b = DhlTV0Dle - DhlV0
    ai, aj, av = A_l.getValuesCSR()
    A_l = csr_matrix((av, aj, ai))
    V1s = scipy.sparse.linalg.spsolve(A_l, b, permc_spec=None, use_umfpack=True)

    # Ainv
    Ainv = np.linalg.inv(D_eff)

    # B_tim
    B_tim = np.matmul(DhlTV0Dle.T, V0)

    # C_tim
    C_tim = V0DllV0 + np.matmul(V1s.T, DhlV0 + DhlTV0Dle)
    C_tim = 0.5 * (C_tim + C_tim.T)

    # Ginv
    Q_tim = np.matmul(Ainv, np.array([(0, 0), (0, 0), (0, -1), (1, 0)]))
    Ginv = np.matmul(
        np.matmul(Q_tim.T, (C_tim - np.matmul(np.matmul(B_tim.T, Ainv), B_tim))), Q_tim
    )
    G_tim = np.linalg.inv(Ginv)
    Y_tim = np.matmul(np.matmul(B_tim.T, Q_tim), G_tim)
    A_tim = D_eff + np.matmul(np.matmul(Y_tim, Ginv), Y_tim.T)

    # Deff_srt
    D = np.zeros((6, 6))

    D[4:6, 4:6] = G_tim
    D[0:4, 4:6] = Y_tim
    D[4:6, 0:4] = Y_tim.T
    D[0:4, 0:4] = A_tim

    Deff_srt = np.zeros((6, 6))
    Deff_srt[0, 3:6] = A_tim[0, 1:4]
    Deff_srt[0, 1:3] = Y_tim[0, :]
    Deff_srt[0, 0] = A_tim[0, 0]

    Deff_srt[3:6, 3:6] = A_tim[1:4, 1:4]
    Deff_srt[3:6, 1:3] = Y_tim[1:4, :]
    Deff_srt[3:6, 0] = A_tim[1:4, 0]

    Deff_srt[1:3, 1:3] = G_tim
    Deff_srt[1:3, 3:6] = Y_tim.T[:, 1:4]
    Deff_srt[1:3, 0] = Y_tim.T[:, 0]

    return np.around(D_eff), np.around(Deff_srt), V0, V1s


def compute_stiffness(
    ABD,
    mesh,
    subdomains,
    l_submesh,
    r_submesh):
    """_summary_

    Parameters
    ----------
    ABD : _type_
        _description_
    mesh : _type_
        _description_
    subdomains : _type_
        _description_
    l_submesh : _type_
        _description_
    r_submesh : _type_
        _description_

    Returns
    -------
    tuple(np.array)
        segment_timo_stiffness, segment_eb_stiffness, l_timo_stiffness, r_timo_stiffness
    """

    nphases = len(ABD)
    tdim=mesh.topology.dim
    fdim = tdim - 1
    
    pp=mesh.geometry.x
    x_min, x_max=min(pp[:,0]), max(pp[:,0])

    # Case flags
    all_facets = False  # Approximate all facet vectors if True, subset if False
    tangent_flag = True  # Approximate normals if False, approximate tangents if True
    interior = True  # Set to True if the facets are internal (e.g. an interface between two domains)
    # Set to False if the facets are on the mesh boundary

    dim = 3  # Spatial dimension of mesh

    DEFAULT = 2
    SUBSET = 3

    # Mark the interior facets lying on an interface inside the square/cube.
    # NOTE: this is a pretty ad-hoc way of marking an interface.
    def locator(x):
        """Marker function that returns True if the x-coordinate is between xmin and xmax."""
        return np.logical_and(x[0] > x_min, x[0] > x_min)

    # Create necessary topological entities of the mesh
    mesh.topology.create_entities(mesh.topology.dim - 1)
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)

    # Create an array facet_marker which contains the facet marker value for all facets in the mesh.
    facet_dim = mesh.topology.dim - 1  # Topological dimension of facets
    num_facets = (
        mesh.topology.index_map(facet_dim).size_local
        + mesh.topology.index_map(facet_dim).num_ghosts
    )  # Total number of facets in mesh
    facet_marker = np.full(
        num_facets, DEFAULT, dtype=np.int32
    )  # Default facet marker value is 2

    subset = np.linspace(1, num_facets, num_facets, dtype=int) - 1
    n1 = np.setdiff1d(subset, l_submesh["entity_map"])
    subset_facets = np.setdiff1d(n1, r_submesh["entity_map"])
    # subset_facets = dolfinx.mesh.locate_entities(mesh, facet_dim, locator) # Get subset of facets to be marked

    facet_marker[subset_facets] = (
        SUBSET  # Fill facet marker array with the value SUBSET
    )
    facet_tags = dolfinx.mesh.meshtags(
        mesh, facet_dim, np.arange(num_facets, dtype=np.int32), facet_marker
    )  # Create facet meshtags
    ft_id = (
        SUBSET  # Set the meshtags id for which we will approximate the facet vectors
    )

    # Create a DG1 space for the facet vectors to be approximated.
    DG1 = basix.ufl.element(
        family="Lagrange",
        cell=mesh.basix_cell(),
        degree=1,
        discontinuous=True,
        shape=(mesh.geometry.dim,),
    )
    space = dolfinx.fem.functionspace(mesh=mesh, element=DG1)

    # Compute the facet vector approximation (Tangent)
    mesh.topology.create_connectivity(tdim, 0)
    nh = utils.facet_vector_approximation(
        V=space, mt=facet_tags, mt_id=ft_id, interior=interior, tangent=tangent_flag
    )
    
    # Initialize terms
    # NOTE: why do we need the frame from local_frame_1D instead of the already computed frames
    e_l, V_l, dvl, v_l, x_l, dx_l = utils.local_boun(l_submesh["mesh"], utils.local_frame_1D(l_submesh["mesh"]),l_submesh["subdomains"])
    e_r, V_r, dvr, v_r, x_r, dx_r = utils.local_boun(r_submesh["mesh"], utils.local_frame_1D(r_submesh["mesh"]) ,r_submesh["subdomains"])


    # V0_l,V0_r=solve_boun(mesh_l,local_frame_1D(mesh_l),subdomains_l),solve_boun(mesh_r,local_frame_1D(mesh_l),subdomains_r)
    D_effEB_l, Deff_l, V0_l, V1_l = core.compute_timo_boun(ABD, l_submesh, nh)
        # mesh_l, subdomains_l, local_frame_1D(mesh_l)
    # )
    D_effEB_r, Deff_r, V0_r, V1_r = core.compute_timo_boun(ABD, r_submesh, nh)
        # mesh_r, subdomains_r, local_frame_1D(mesh_r)
    # )

    # ***************** Wb Segment (surface mesh) computation begins************************
    e, V, dv, v_, x, dx = utils.local_boun(mesh, utils.local_frame(mesh), subdomains)
    V0, Dle, Dhe, D_ee, V1s = utils.initialize_array(V)

    mesh.topology.create_connectivity(1, 2)
    l_submesh["mesh"].topology.create_connectivity(1, 1)
    r_submesh["mesh"].topology.create_connectivity(1, 1)

    # Obtaining coefficient matrix AA and BB with and without bc applied.
    # Note: bc is applied at boundary dofs. We define v2a containing all dofs of entire wind blade.

    F2 = sum(
        [
            dot(dot(as_tensor(ABD[i]), utils.gamma_h(e, x, dv)), utils.gamma_h(e, x, v_)) * dx(i)
            for i in range(nphases)
        ]
    )
    ff = utils.deri_constraint(dv, v_, mesh, nh)
    a = form(F2 + ff)
    B = assemble_matrix(form(F2))  # Obtain coefficient matrix without BC applied: BB
    B.assemble()
    ai, aj, av = B.getValuesCSR()
    BB = csr_matrix((av, aj, ai))

    # bc applied
    boundary_dofs = locate_dofs_topological(
        V, fdim, np.concatenate((r_submesh["entity_map"], l_submesh["entity_map"]), axis=0)
    )
    v2a = Function(V)
    bc = dolfinx.fem.dirichletbc(v2a, boundary_dofs)
    A = assemble_matrix(a, [bc])  # Obtain coefficient matrix with BC applied: AA
    A.assemble()

    # Assembly
    # Running for 4 different F vector. However, F has bc applied to it where, stored known values of v2a is provided for each loop (from boun solve).
    # Assembly
    for p in range(4):  # 4 load cases meaning
        # Boundary
        v2a = Function(V)
        
        v2a = utils.dof_mapping_quad(V, v2a, V_l, V0_l[:, p], l_submesh["facets"], l_submesh["entity_map"])
        v2a = utils.dof_mapping_quad(V, v2a, V_r, V0_r[:, p], r_submesh["facets"], r_submesh["entity_map"])

        F2 = -sum(
            [
                dot(dot(as_tensor(ABD[i]), utils.gamma_e(e, x)[:, p]), utils.gamma_h(e, x, v_)) * dx(i)
                for i in range(nphases)
            ]
        )
        bc = [dolfinx.fem.dirichletbc(v2a, boundary_dofs)]
        F = petsc.assemble_vector(form(F2))
        apply_lifting(
            F, [a], [bc]
        )  # apply bc to rhs vector (Dhe) based on known fluc solutions
        set_bc(F, bc)
        v = utils.solve_ksp(A, F, V)
        V0[:, p] = v.vector[:]
        #  Dhe[:,p]= petsc.assemble_vector(form(F2))
        Dhe[:, p] = scipy.sparse.csr_array(BB).dot(V0[:, p])

    D1 = np.matmul(V0.T, -Dhe)
    for s in range(4):
        for k in range(4):
            f = dolfinx.fem.form(
                sum(
                    [
                        dot(dot(utils.gamma_e(e, x).T, as_tensor(ABD[i])), utils.gamma_e(e, x))[s, k]
                        * dx(i)
                        for i in range(nphases)
                    ]
                )
            )
            D_ee[s, k] = dolfinx.fem.assemble_scalar(f)
    L = x_max - x_min
    D_eff = (D_ee + D1) / L

    print("\n EB Tapered Stiffness \n")
    np.set_printoptions(precision=4)
    print(np.around(D_eff))

    ##################Timoshenko Stiffness Matrix for WB segment begins###################################
    # Process is similar to Timoshenko boundary implemented over WB segment mesh
    F1 = sum(
        [
            dot(dot(as_tensor(ABD[i]), utils.gamma_l(e, x, v_)), utils.gamma_l(e, x, dv)) * dx(i)
            for i in range(nphases)
        ]
    )
    a1 = form(F1)
    Dll = assemble_matrix(a1)
    Dll.assemble()
    ai, aj, av = Dll.getValuesCSR()
    Dll = csr_matrix((av, aj, ai)).toarray()

    # Dhl
    F_dhl = sum(
        [
            dot(dot(as_tensor(ABD[i]), utils.gamma_h(e, x, dv)), utils.gamma_l(e, x, v_)) * dx(i)
            for i in range(nphases)
        ]
    )
    a3 = form(F_dhl)
    Dhl = assemble_matrix(a3)
    Dhl.assemble()
    ai, aj, av = Dhl.getValuesCSR()
    Dhl = csr_matrix((av, aj, ai)).toarray()

    for p in range(4):
        F1 = sum(
            [
                dot(dot(as_tensor(ABD[i]), utils.gamma_e(e, x)[:, p]), utils.gamma_l(e, x, v_)) * dx(i)
                for i in range(nphases)
            ]
        )
        Dle[:, p] = petsc.assemble_vector(form(F1))[:]

    # DhlTV0
    DhlV0 = np.matmul(Dhl.T, V0)

    # DhlTV0Dle
    DhlTV0Dle = np.matmul(Dhl, V0) + Dle

    # V0DllV0
    V0DllV0 = np.matmul(np.matmul(V0.T, Dll), V0)

    # V1s
    b = DhlTV0Dle - DhlV0

    # B_tim
    B_tim = np.matmul(DhlTV0Dle.T, V0)
    B_tim = B_tim / L

    # Assembly
    # For Wb mesh (surface elements, for solving (A V1s = b), directly cannot be computed as we require dirichilet bc)
    for p in range(4):  # 4 load cases meaning
        # Boundary
        v2a = Function(V)
        v2a = utils.dof_mapping_quad(V, v2a, V_l, V1_l[:, p], l_submesh["facets"], l_submesh["entity_map"])
        v2a = utils.dof_mapping_quad(V, v2a, V_r, V1_r[:, p], r_submesh["facets"], r_submesh["entity_map"])
        bc = [dolfinx.fem.dirichletbc(v2a, boundary_dofs)]

        # quad mesh
        F = petsc4py.PETSc.Vec().createWithArray(
            b[:, p], comm=MPI.COMM_WORLD
        )  # Converting b[:,p] numpy vector to petsc array
        F.ghostUpdate(
            addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE
        )
        apply_lifting(F, [a], [bc])
        set_bc(F, bc)
        w = utils.solve_ksp(A, F, V)
        V1s[:, p] = w.vector[:]

    # C_tim
    C_tim = V0DllV0 + np.matmul(V1s.T, DhlV0 + DhlTV0Dle)
    C_tim = 0.5 * (C_tim + C_tim.T)
    C_tim = C_tim / L
    # Ainv
    Ainv = np.linalg.inv(D_eff)

    # Ginv
    Q_tim = np.matmul(Ainv, np.array([(0, 0), (0, 0), (0, -1), (1, 0)]))
    Ginv = np.matmul(
        np.matmul(Q_tim.T, (C_tim - np.matmul(np.matmul(B_tim.T, Ainv), B_tim))), Q_tim
    )
    G_tim = np.linalg.inv(Ginv)
    Y_tim = np.matmul(np.matmul(B_tim.T, Q_tim), G_tim)
    A_tim = D_eff + np.matmul(np.matmul(Y_tim, Ginv), Y_tim.T)

    # Deff_srt
    D = np.zeros((6, 6))

    D[4:6, 4:6] = G_tim
    D[0:4, 4:6] = Y_tim
    D[4:6, 0:4] = Y_tim.T
    D[0:4, 0:4] = A_tim

    Deff_srt = np.zeros((6, 6))
    Deff_srt[0, 3:6] = A_tim[0, 1:4]
    Deff_srt[0, 1:3] = Y_tim[0, :]
    Deff_srt[0, 0] = A_tim[0, 0]

    Deff_srt[3:6, 3:6] = A_tim[1:4, 1:4]
    Deff_srt[3:6, 1:3] = Y_tim[1:4, :]
    Deff_srt[3:6, 0] = A_tim[1:4, 0]

    Deff_srt[1:3, 1:3] = G_tim
    Deff_srt[1:3, 3:6] = Y_tim.T[:, 1:4]
    Deff_srt[1:3, 0] = Y_tim.T[:, 0]

    print("\n Timo Stiffness Matrix for WB Segment \n")
    np.set_printoptions(precision=4)
    print(np.around(Deff_srt))

    print("\n Timoshenko Stiffness (Left Boundary) \n")
    np.set_printoptions(precision=4)
    print(np.around(Deff_l))

    # Compare Unique ABD matrices
    for ii, AB in enumerate(ABD):
        np.set_printoptions(precision=4)
        print("\n", ii, "\n")
        print(np.around(AB))
        
    return Deff_srt, D_eff, Deff_l, Deff_r

