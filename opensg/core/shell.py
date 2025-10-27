# Initialization of libraries
from mpi4py import MPI
import numpy as np
import dolfinx
import basix
import scipy

from dolfinx.fem import (
    form,
    petsc,
    Function,
    functionspace,
    apply_lifting,
    set_bc,
    locate_dofs_topological,
    assemble_scalar,
)


from ufl import (
    TrialFunction,
    TestFunction,
    inner,
    lhs,
    rhs,
    as_tensor,
    dot,
    SpatialCoordinate,
    Measure,
    as_vector,
)

from scipy.sparse import csr_matrix
import petsc4py.PETSc
from dolfinx.fem.petsc import assemble_matrix

import ufl
import opensg.utils.shared as shared_utils
import opensg.utils.shell as utils
import opensg.core.shell as core



def compute_ABD_CLT(thick, nlay, angle, mat_names, material_database):
    """Compute the ABD matrix for composite laminates.

    This function implements the CLT-based Kirchhoff plate stiffness matrix computation
    for composite laminates. It creates a 1D through-thickness mesh and computes
    the homogenized stiffness properties for the laminate.

    Parameters
    ----------
    thick : list[float]
        List of layer thicknesses for each layer in the laminate [length units]
    nlay : int
        Number of layers in the composite laminate
    angle : list[float]
        List of fiber angles (in radians) for each layer
    mat_names : list[str]
        List of material names corresponding to each layer
    material_database : dict
        Database containing material properties for each material name.
        Each material should have orthotropic properties: E1, E2, G12, nu12, etc.

    Returns
    -------
    numpy.ndarray
        6x6 ABD matrix representing the laminate stiffness matrix.
        Structure: [[A, B], [B, D]] where:
        - A: membrane stiffness (3x3)
        - B: coupling stiffness (3x3) 
        - D: bending stiffness (3x3)
        Relates generalized forces to generalized strains: {N, M} = ABD * {ε₀, κ}
    """
    # Nodes (1D SG)
    th, s = [0], 0  # Reference-------- 0
    for k in thick:
        s = s + k  # Inward normal in orien provided by yaml file
        th.append(s)
  #  points = np.array(th)
    
    def R_sig(Q, t):  # Rotation matrix
        """
        Performs rotation from local material frame to global frame

        Parameters
        ----------
        Q : numpy.ndarray
            [3,3] numpy array - Plane stress- reduced Stiffness matrix
        t : float
            rotation angle

        Returns
        -------
        numpy.ndarray
            Q': Rotated Stiffness matrix
        """
        th = np.deg2rad(t)
        c, s, cs = np.cos(th), np.sin(th), np.cos(th) * np.sin(th)
        R_Sig = np.array(
            [
                (c**2, s**2,  -2 * cs),
                (s**2, c**2,   2 * cs),
                (cs, -cs,     c**2 - s**2),
            ]
        )
        return np.matmul(np.matmul(R_Sig, Q), R_Sig.transpose())    
    
    def Q_mat(material_parameters, theta):
        """
        Compute the [3,3] Stiffness matrix using material elastic constants

        Parameters
        ----------
        material_parameters : dict
            Material parameters
        theta : float
            Rotation angle

        Returns
        -------
        numpy.ndarray
            C: [3,3] Stiffness Matrix
        """
        # E1,E2,E3,G12,G13,G23,v12,v13,v23= material_parameters[matid[ii][i]]
        E1, E2, E3 = material_parameters["E"]
        G12, G13, G23 = material_parameters["G"]
        v12, v13, v23 = material_parameters["nu"]
        Q11=E1/(1-v12**2)
        Q22=E2/(1-v12**2)
        Q12=(v12*E2)/(1-v12**2)
        Q66=G12
        Q=np.array([(Q11, Q12,  0),
                    (Q12, Q22,  0),
                    (0,     0,  Q66)])
        # theta=angle[ii][i] # ii denotes the layup id
        Q = R_sig(Q, theta)
        return Q   
    A,B,D=np.zeros((3,3)), np.zeros((3,3)), np.zeros((3,3))
        
    # Compute ABD
    for i in range(nlay): 
        Q= Q_mat(material_database[mat_names[i]], angle[i])
        A=A+Q*(th[i+1]-th[i])
        B=B+Q*(th[i+1]**2-th[i]**2)  
        D=D+Q*(th[i+1]**3-th[i]**3)    
    ABD= np.block([[A, B],
              [B, D]])

 
    return ABD
        

# Generate ABD matrix (Plate model)
def compute_ABD_matrix(thick, nlay, angle, mat_names, material_database):
    """Compute the ABD matrix for composite laminates.

    This function implements the MSG-based Kirchhoff plate stiffness matrix computation
    for composite laminates. It creates a 1D through-thickness mesh and computes
    the homogenized stiffness properties for the laminate.

    Parameters
    ----------
    thick : list[float]
        List of layer thicknesses for each layer in the laminate [length units]
    nlay : int
        Number of layers in the composite laminate
    angle : list[float]
        List of fiber angles (in radians) for each layer
    mat_names : list[str]
        List of material names corresponding to each layer
    material_database : dict
        Database containing material properties for each material name.
        Each material should have orthotropic properties: E1, E2, G12, nu12, etc.

    Returns
    -------
    numpy.ndarray
        6x6 ABD matrix representing the laminate stiffness matrix.
        Structure: [[A, B], [B, D]] where:
        - A: membrane stiffness (3x3)
        - B: coupling stiffness (3x3) 
        - D: bending stiffness (3x3)
        Relates generalized forces to generalized strains: {N, M} = ABD * {ε₀, κ}
    """
    deg = 2
    cell = ufl.Cell("interval")
    elem = basix.ufl.element("Lagrange", "interval", 1, shape=(3,))
    domain = ufl.Mesh(elem)

    # Nodes (1D SG)
    th = [[0.0, 0.0, 0.0]]  # Start with the node at the origin
    s = 0  # Reference-------- 0
    for k in thick:
        s = s + k
        th.append([s, 0.0, 0.0])  # Append the 3D coordinate [x, 0, 0]
    points = np.array(th, dtype=np.float64)
    # Elements
    cell = []
    for k in range(nlay):
        cell.append([k, k + 1])
    cellss = np.array(cell, dtype=np.int64)

    # Create 1D SG mesh
    dom = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cellss, domain, points)
    num_cells = dom.topology.index_map(dom.topology.dim).size_local
    cells = np.arange(num_cells, dtype=np.int32)
    subdomain = dolfinx.mesh.meshtags(
        dom, dom.topology.dim, cells, cells
    )  # assigning each element as subdomain
    x, dx = (
        SpatialCoordinate(dom),
        Measure("dx")(
            domain=dom, subdomain_data=subdomain, metadata={"quadrature_degree": 4}
        ),
    )

    gamma_e = as_tensor(
        [
            (1, 0, 0, x[0], 0, 0),  # Gamma_e matrix
            (0, 1, 0, 0, x[0], 0),
            (0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0),
            (0, 0, 1, 0, 0, x[0]),
        ]
    )

    nphases = len(cells)

    def gamma_h(v):  # (Gamma_h * w)
        E1 = as_vector([0, 0, v[2].dx(0), (v[1].dx(0)), (v[0].dx(0)), 0])
        return E1

    def R_sig(C, t):  # Rotation matrix
        """
        Performs rotation from local material frame to global frame

        Parameters
        ----------
        C : numpy.ndarray
            [6,6] numpy array - Stiffness matrix
        t : float
            rotation angle

        Returns
        -------
        numpy.ndarray
            C': Rotated Stiffness matrix
        """
        th = np.deg2rad(t)
        c, s, cs = np.cos(th), np.sin(th), np.cos(th) * np.sin(th)
        R_Sig = np.array(
            [
                (c**2, s**2, 0, 0, 0, -2 * cs),
                (s**2, c**2, 0, 0, 0, 2 * cs),
                (0, 0, 1, 0, 0, 0),
                (0, 0, 0, c, s, 0),
                (0, 0, 0, -s, c, 0),
                (cs, -cs, 0, 0, 0, c**2 - s**2),
            ]
        )
        return np.matmul(np.matmul(R_Sig, C), R_Sig.transpose())

    def Stiff_mat(material_parameters, theta):
        """
        Compute the [6,6] Stiffness matrix using material elastic constants

        Parameters
        ----------
        material_parameters : dict
            Material parameters
        theta : float
            Rotation angle

        Returns
        -------
        numpy.ndarray
            C: [6,6] Stiffness Matrix
        """
        # E1,E2,E3,G12,G13,G23,v12,v13,v23= material_parameters[matid[ii][i]]
        E1, E2, E3 = material_parameters["E"]
        G12, G13, G23 = material_parameters["G"]
        v12, v13, v23 = material_parameters["nu"]
        S = np.zeros((6, 6))
        S[0, 0], S[1, 1], S[2, 2] = 1 / E1, 1 / E2, 1 / E3
        S[0, 1], S[0, 2] = -v12 / E1, -v13 / E1
        S[1, 0], S[1, 2] = -v12 / E1, -v23 / E2
        S[2, 0], S[2, 1] = -v13 / E1, -v23 / E2
        S[3, 3], S[4, 4], S[5, 5] = 1 / G23, 1 / G13, 1 / G12
        C = np.linalg.inv(S)
       # theta=angle[ii][i] # ii denotes the layup id
        C = as_tensor(R_sig(C, theta))
        return C


    # Creating FE function Space
    V = functionspace(dom, basix.ufl.element("CG", "interval", deg, shape=(3,)))
    dv, v_ = TrialFunction(V), TestFunction(V)
    F2 = sum(
        [
            dot(
                dot(Stiff_mat(material_database[mat_names[i]], angle[i]), gamma_h(dv)),
                gamma_h(v_),
            )
            * dx(i)
            for i in range(nphases)
        ]
    )  # Weak form of energy(load vec)
    A = petsc.assemble_matrix(form(F2))
    A.assemble()
    _,null = shared_utils.compute_nullspace(V, ABD=True)
    A.setNullSpace(null)  # Set the nullspace
    ndofs = 3 * V.dofmap.index_map.local_range[1]  # total dofs
    # Initialization
    V0, Dhe, D_ee = np.zeros((ndofs, 6)), np.zeros((ndofs, 6)), np.zeros((6, 6))

    # Assembly ! 6 Load Cases
    for p in range(6):
        # Right Hand Side vector in weak form (F2(v_))
        # < gamma_e[:,p].T Stiffness_Matrix gamma_h>
        F2 = -sum(
            [
                dot(
                    dot(
                        Stiff_mat(material_database[mat_names[i]], angle[i]),
                        gamma_e[:, p],
                    ),
                    gamma_h(v_),
                )
                * dx(i)
                for i in range(nphases)
            ]
        )
        F = petsc.assemble_vector(form(F2))
        F.ghostUpdate(
            addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE
        )
        null.remove(F)  # Orthogonalize F to the null space of A^T
        Dhe[:, p] = F[:]  # Dhe matrix formation
        w = shared_utils.solve_ksp(A, F, V)
        V0[:, p] = w.x.array[:]  # Solved Fluctuating Functions
    D1 = np.matmul(V0.T, -Dhe)
    x0 = x[0]
    def Dee_abd(x, material_parameters, theta):
        """
        Performs < gamma_e.T Stiffness_matrix gamma_e > and give simplified form

        Parameters
        ----------
        x : array-like
            Coordinate point
        material_parameters : dict
            Material parameters
        theta : float
            Rotation angle

        Returns
        -------
        ufl.Tensor
            Dee: [6,6] ufl tensor
        """
        C = Stiff_mat(material_parameters, theta)

        return as_tensor(
            [
                (C[0, 0], C[0, 1], C[0, 5], x0 * C[0, 0], x0 * C[0, 1], x0 * C[0, 5]),
                (C[1, 0], C[1, 1], C[1, 5], x0 * C[1, 0], x0 * C[1, 1], x0 * C[1, 5]),
                (C[5, 0], C[5, 1], C[5, 5], x0 * C[5, 0], x0 * C[5, 1], x0 * C[5, 5]),
                (
                    x0 * C[0, 0],
                    x0 * C[0, 1],
                    x0 * C[0, 5],
                    x0 * x0 * C[0, 0],
                    x0 * x0 * C[0, 1],
                    x0 * x0 * C[0, 5],
                ),
                (
                    x0 * C[1, 0],
                    x0 * C[1, 1],
                    x0 * C[1, 5],
                    x0 * x0 * C[1, 0],
                    x0 * x0 * C[1, 1],
                    x0 * x0 * C[1, 5],
                ),
                (
                    x0 * C[5, 0],
                    x0 * C[5, 1],
                    x0 * C[5, 5],
                    x0 * x0 * C[5, 0],
                    x0 * x0 * C[5, 1],
                    x0 * x0 * C[5, 5],
                ),
            ]
        )

    # Scalar assembly
    for s in range(6):
        for k in range(6):
            f = dolfinx.fem.form(
                sum(
                    [
                        Dee_abd(x, material_database[mat_names[i]], angle[i])[s, k]
                        * dx(i)
                        for i in range(nphases)
                    ]
                )
            )
            D_ee[s, k] = dolfinx.fem.assemble_scalar(f)  # D_ee [6,6]

    D_eff = D_ee + D1

    mu=assemble_scalar(form(sum([material_database[mat_names[i]]['rho']*dx(i) for i in range(nphases)]))) # mass per unit area
    xm3=(1/mu)*assemble_scalar(form(sum([x0*material_database[mat_names[i]]['rho']*dx(i) for i in range(nphases)])))
    i22=assemble_scalar(form(sum([x0**2*material_database[mat_names[i]]['rho']*dx(i) for i in range(nphases)])))
    
  #  D_eff=compute_ABD_CLT(thick, nlay, angle, mat_names, material_database)

    return D_eff ,  [mu,mu*xm3,i22]


# def compute_timo_boun(ABD, mesh, subdomains, frame, nullspace, sub_nullspace, nphases):

def compute_timo_boun(ABD, boundary_submeshdata):
    """Compute boundary stiffness matrices for Euler-Bernoulli and Timoshenko beam theories.

    This function solves both Euler-Bernoulli (EB) and Timoshenko beam models on the 
    boundary mesh to compute effective stiffness matrices and fluctuating functions 
    used for Dirichlet boundary constraints in segment analysis.

    Parameters
    ----------
    ABD : list[numpy.ndarray]
        List of 6x6 ABD matrices for each layup in the boundary mesh
    boundary_submeshdata : dict
        Dictionary containing boundary mesh data with keys:
        - 'mesh': Boundary mesh object
        - 'subdomains': Subdomain tags for different layups
        - 'frame': Local orientation frame functions (optional, computed if not provided)
    nh : int
        Number of hierarchical constraint equations for mesh consistency

    Returns
    -------
    tuple
        Contains (D_eff, Deff_srt, V0, V1s):
        
        - D_eff : numpy.ndarray
            4x4 boundary Euler-Bernoulli stiffness matrix relating boundary forces
            to boundary displacements and rotations
        - Deff_srt : numpy.ndarray  
            6x6 boundary Timoshenko stiffness matrix including shear deformation effects
        - V0 : numpy.ndarray
            Boundary fluctuating function solutions [ndofs_boundary, 4] for EB model.
            Used for imposing boundary constraints in segment analysis
        - V1s : numpy.ndarray
            Boundary fluctuating function solutions [ndofs_boundary, 4] for Timoshenko model.
            Used for imposing boundary constraints with shear effects

    """
    boundary_mesh = boundary_submeshdata["mesh"]
    boundary_subdomains = boundary_submeshdata["subdomains"]
    # boundary_frame = boundary_submeshdata["frame"]
    # Use pre-computed frame if available, otherwise compute it
    
    #if "frame" in boundary_submeshdata:
 #   boundary_frame = boundary_submeshdata["frame"]
   # else:
    boundary_frame = utils.local_frame_1D(boundary_mesh)

    nphases = len(ABD)

    e, V_l, dv, v_, x, dx = utils.local_boun(
        boundary_mesh, boundary_frame, boundary_subdomains
    )
    
    boundary_mesh.topology.create_connectivity(1, 1)
    V0, Dle, Dhe, D_ee, V1s = utils.initialize_array(V_l)
    nullspace_basis, null = shared_utils.compute_nullspace(V_l, ABD=True)
    
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
    
    penalty=1e9  # known for now
    
    ff = utils.deri_constraint(dv,v_,x,V_l,boundary_mesh,penalty)
    
    A_l = assemble_matrix(form(F2 + ff))
    A_l.assemble()
    A_l.setNullSpace(null) 
  #  A_l=utils.apply_null_A(A_ll,nullspace_basis) 

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
        null.remove(F_l)
        Dhe[:,p]=F_l[:]  #F_array

        w = shared_utils.solve_ksp(A_l, F_l, V_l)
        V0[:, p] = w.x.array[:]

    V0_csr=csr_matrix(V0) 
    D1=V0_csr.T.dot(csr_matrix(-Dhe)) 
    
    
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
                dot(as_tensor(ABD[i]), utils.gamma_l(e, x, dv)),
                utils.gamma_l(e, x, v_)
            )
            * dx(i)
            for i in range(nphases)
        ]
    )
    a1 = form(F1)
    Dll = assemble_matrix(a1)
    Dll.assemble()
    ai, aj, av = Dll.getValuesCSR()
    Dll = csr_matrix((av, aj, ai))

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
                utils.gamma_l(e, x, v_)
            )
            * dx(i)
            for i in range(nphases)
        ]
    )
  #  ff = shared_utils.deri_constraint(dv, v_, boundary_mesh, nh)
    a3 = form(F_dhl)
    Dhl = assemble_matrix(a3)
    Dhl.assemble()
    ai, aj, av = Dhl.getValuesCSR()
    Dhl = csr_matrix((av, aj, ai))

    # DhlTV0
    DhlV0 = Dhl.T.dot(V0_csr) 

    # DhlTV0Dle
    DhlTV0Dle = Dhl.dot(V0_csr)+ csr_matrix(Dle)

    # V0DllV0
    V0DllV0 = (V0_csr.T.dot(Dll)).dot(V0_csr)

    # V1s  ****Updated from previous version as for solving boundary V1s, we can directly use (A_l V1s=b),  and solve for V1s
    b = (DhlTV0Dle-DhlV0).toarray()
  #  for i in range(4):
     #   F_array = b[:,i]

     #   n_array = nullspace_basis[i].array
     #   F_array -= np.dot(F_array, n_array) * n_array
        
      #  F=petsc4py.PETSc.Vec().createWithArray(F_array,comm=MPI.COMM_WORLD)
      #  F.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE) 
     #   w = shared_utils.solve_ksp(A_l, F, V_l)
     #   V1s[:, i] = w.x.array[:]
    
    # V1s
    ai, aj, av=A_l.getValuesCSR()  
    A_l=csr_matrix((av, aj, ai))
    V1s=scipy.sparse.linalg.spsolve(A_l, b, permc_spec=None, use_umfpack=True)
    V1s_csr=csr_matrix(V1s)   

    # Ainv
    Ainv=np.linalg.inv(D_eff).astype(np.float64)

    # B_tim
    B_tim=DhlTV0Dle.T.dot(V0_csr)
    B_tim=B_tim.toarray().astype(np.float64)


    # C_tim
    C_tim= V0DllV0 + V1s_csr.T.dot(DhlV0 + DhlTV0Dle) 
    C_tim=0.5*(C_tim+C_tim.T)
    C_tim=C_tim.toarray().astype(np.float64)

    # Ginv
    Q_tim = np.matmul(Ainv,np.array([(0,0),(0,0),(0,-1),(1,0)])).astype(np.float64)
    Ginv = np.matmul(
        np.matmul(Q_tim.T, (C_tim - np.matmul(np.matmul(B_tim.T, Ainv), B_tim))), Q_tim
    ).astype(np.float64)
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
    Deff_srt[3:6, 0] = A_tim[1:4, 0].flatten()

    Deff_srt[1:3, 1:3] = G_tim
    Deff_srt[1:3, 3:6] = Y_tim.T[:, 1:4]
    Deff_srt[1:3, 0] = Y_tim.T[:, 0].flatten()

    return np.around(Deff_srt), V0, V1s

def compute_stiffness(ABD, mesh, subdomains, l_submesh, r_submesh, boun=False):
    """Compute stiffness matrices for shell segments.

    Parameters
    ----------
    ABD : numpy.ndarray
        ABD matrix
    mesh : dolfinx.mesh.Mesh
        Mesh object
    subdomains : dict
        Subdomain data
    l_submesh : dict
        Left submesh data
    r_submesh : dict
        Right submesh data

    Returns
    -------
    tuple
        segment_timo_stiffness, segment_eb_stiffness, l_timo_stiffness, r_timo_stiffness
    """

     # Use pre-computed frames if available, otherwise compute them
    # NOTE: why do we need the frame from local_frame_1D instead of the already computed frames
    
    if "frame" in l_submesh:
        l_frame = l_submesh["frame"]
    else:
        l_frame = utils.local_frame_1D(l_submesh["mesh"])

    if "frame" in r_submesh:
        r_frame = r_submesh["frame"]
    else:
        r_frame = utils.local_frame_1D(r_submesh["mesh"])
          
    
    e_l, V_l, dvl, v_l, x_l, dx_l = utils.local_boun(
        l_submesh["mesh"], l_frame, l_submesh["subdomains"]
    )

    e_r, V_r, dvr, v_r, x_r, dx_r = utils.local_boun(
        r_submesh["mesh"], r_frame, r_submesh["subdomains"]
    )

    Deff_l, V0_l, V1_l = core.compute_timo_boun(
        ABD, l_submesh
    )

    Deff_r, V0_r, V1_r = core.compute_timo_boun(
        ABD, r_submesh
    )
    if boun:
        return None, Deff_l, Deff_r
        
    
    # ***************** Wb Segment (surface mesh) computation begins************************
    nphases = len(ABD)
    fdim = mesh.topology.dim - 1

    pp = mesh.geometry.x
    x_min, x_max = min(pp[:, 0]), max(pp[:, 0])
    
    e, V, dv, v_, x, dx = utils.local_boun(mesh, utils.local_frame(mesh), subdomains)
    V0, Dle, Dhe, D_ee, V1s = utils.initialize_array(V)

    mesh.topology.create_connectivity(1, 2)
    l_submesh["mesh"].topology.create_connectivity(1, 1)
    r_submesh["mesh"].topology.create_connectivity(1, 1)

    # Obtaining coefficient matrix AA and BB with and without bc applied.
    # Note: bc is applied at boundary dofs. We define v2a containing all dofs of entire wind blade.

    F2 = sum(
        [
            dot(
                dot(as_tensor(ABD[i]), utils.gamma_h(e, x, dv)), utils.gamma_h(e, x, v_)
            )
            * dx(i)
            for i in range(nphases)
        ]
    )
    penalty=1e12
    ff = utils.deri_constraint(dv, v_, x, V, mesh,penalty)
    a = form(F2+ff)
    
    # bc applied
    boundary_dofs = locate_dofs_topological(
        V,
        fdim,
        np.concatenate((r_submesh["entity_map"], l_submesh["entity_map"]), axis=0),
    )
    v2a = Function(V)
    bc = dolfinx.fem.dirichletbc(v2a, boundary_dofs)
    A = assemble_matrix(a, [bc])  # Obtain coefficient matrix with BC applied: AA
    A.assemble()

    # Assembly
    # Running for 4 different F vector. However, F has bc applied to it where, stored known values of v2a is provided for each loop (from boun solve).

    for p in range(4):  # 4 load cases meaning
        # Boundary
        v2a = Function(V)

        v2a = utils.dof_mapping_quad(
            V, v2a, V_l, V0_l[:, p], l_submesh["facets"], l_submesh["entity_map"]
        )
        v2a = utils.dof_mapping_quad(
            V, v2a, V_r, V0_r[:, p], r_submesh["facets"], r_submesh["entity_map"]
        )

        F2 = -sum(
            [
                dot(
                    dot(as_tensor(ABD[i]), utils.gamma_e(e, x)[:, p]),
                    utils.gamma_h(e, x, v_),
                )
                * dx(i)
                for i in range(nphases)
            ]
        )
        bc = [dolfinx.fem.dirichletbc(v2a, boundary_dofs)]
        F = petsc.assemble_vector(form(F2))
        Dhe[:, p] = F[:]
        apply_lifting(
            F, [a], [bc]
        )  # apply bc to rhs vector (Dhe) based on known fluc solutions
        set_bc(F, bc)
        v = shared_utils.solve_ksp(A, F, V)
        V0[:, p] = v.x.array[:]



    V0_csr=csr_matrix(V0)    
    D1=V0_csr.T.dot(csr_matrix(-Dhe))
    for s in range(4):
        for k in range(4):
            f = dolfinx.fem.form(
                sum(
                    [
                        dot(
                            dot(utils.gamma_e(e, x).T, as_tensor(ABD[i])),
                            utils.gamma_e(e, x),
                        )[s, k]
                        * dx(i)
                        for i in range(nphases)
                    ]
                )
            )
            D_ee[s, k] = dolfinx.fem.assemble_scalar(f)
            
    L = x_max - x_min
    D_eff = (D_ee + D1) / L
    D_eff=0.5*(D_eff+D_eff.T)  
    

    ##################Timoshenko Stiffness Matrix for WB segment begins###################################
    # Process is similar to Timoshenko boundary implemented over WB segment mesh
    F1 = sum(
        [
            dot(
                dot(as_tensor(ABD[i]), utils.gamma_l(e, x, dv)), utils.gamma_l(e, x, v_)
            )
            * dx(i)
            for i in range(nphases)
        ]
    )
    a1 = form(F1)
    Dll = assemble_matrix(a1)
    Dll.assemble()
    ai, aj, av = Dll.getValuesCSR()
    Dll = csr_matrix((av, aj, ai))

    # Dhl
    F_dhl = sum(
        [
            dot(
                dot(as_tensor(ABD[i]), utils.gamma_h(e, x, dv)), utils.gamma_l(e, x, v_)
            )
            * dx(i)
            for i in range(nphases)
        ]
    )
    a3 = form(F_dhl)
    Dhl = assemble_matrix(a3)
    Dhl.assemble()
    ai, aj, av = Dhl.getValuesCSR()
    Dhl = csr_matrix((av, aj, ai))

    for p in range(4):
        F1 = sum(
            [
                dot(
                    dot(as_tensor(ABD[i]), utils.gamma_e(e, x)[:, p]),
                    utils.gamma_l(e, x, v_),
                )
                * dx(i)
                for i in range(nphases)
            ]
        )
        Dle[:, p] = petsc.assemble_vector(form(F1))[:]

    # DhlTV0
    DhlV0 = Dhl.T.dot(V0_csr) 

    # DhlTV0Dle
    DhlTV0Dle = Dhl.dot(V0_csr)+ csr_matrix(Dle)

    # V0DllV0
    V0DllV0 = (V0_csr.T.dot(Dll)).dot(V0_csr)

    # V1s
    b = (DhlTV0Dle-DhlV0).toarray()

    # B_tim
    B_tim = DhlTV0Dle.T.dot(V0_csr)
    B_tim = (B_tim/L).toarray()

    # Assembly
    # For Wb mesh (surface elements, for solving (A V1s = b), directly cannot be computed as we require dirichilet bc)
    for p in range(4):  # 4 load cases meaning
        # Boundary
        v2a = Function(V)
        v2a = utils.dof_mapping_quad(
            V, v2a, V_l, V1_l[:, p], l_submesh["facets"], l_submesh["entity_map"]
        )
        v2a = utils.dof_mapping_quad(
            V, v2a, V_r, V1_r[:, p], r_submesh["facets"], r_submesh["entity_map"]
        )
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
        w = shared_utils.solve_ksp(A, F, V)
        V1s[:, p] = w.x.array[:]

    V1s_csr=csr_matrix(V1s)
    # C_tim
    C_tim = V0DllV0 + V1s_csr.T.dot(DhlV0 + DhlTV0Dle)
    C_tim = 0.5 * (C_tim + C_tim.T)
    C_tim = C_tim.toarray()/L
    
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
    Deff_srt[3:6, 0] = A_tim[1:4, 0].flatten()

    Deff_srt[1:3, 1:3] = G_tim
    Deff_srt[1:3, 3:6] = Y_tim.T[:, 1:4]
    Deff_srt[1:3, 0] = Y_tim.T[:, 0].flatten()

    print("\n Timo Stiffness Matrix for WB Segment \n")
    np.set_printoptions(precision=4)
    print(np.around(Deff_srt))

    # Compare Unique ABD matrices
   # for ii, AB in enumerate(ABD):
    #    np.set_printoptions(precision=4)
     #   print("\n", ii, "\n")
     #   print(np.around(AB))

    return Deff_srt, Deff_l, Deff_r
