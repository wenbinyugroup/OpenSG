from ufl import (
    CellDiameter,
    FacetNormal,
    Jacobian,
    Measure,
    as_vector,
    avg,
    cross,
    div,
    dot,
    grad,
    inner,
    jump,
    sqrt,
)
import dolfinx
import petsc4py
from dolfinx.fem import Function
import mpi4py.MPI as MPI
import numpy as np

def xmdf_convert(mesh, subdomains):
    # Set the names for reading later

    mesh.name = "Grid"
    subdomains.name = "Grid_Cells"
    
    # Create the connectivity for facets
    fdim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(fdim, mesh.topology.dim)
    
    # --- Write the mesh to XDMF in PARALLEL ---
    # Each process will now write its OWN PIECE to the file.
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "SG_solid.xdmf", "w") as xdmf:
        # This is now a parallel write operation
        xdmf.write_mesh(mesh)
        
        # These are also parallel write operations
        
        xdmf.write_meshtags(subdomains, mesh.geometry)
    
  #  print("Parallel mesh conversion complete.")
    return 



def compute_nullspace(V, ABD=False):

    """Compute nullspace to restrict rigid body motions in finite element analysis.

    Constructs a translational null space for the vector-valued function space V
    and ensures that it is properly orthonormalized. This is essential for solving 
    systems with potential rigid body motions that need to be constrained.

    Parameters
    ----------
    V : dolfinx.fem.FunctionSpace
        Vector-valued function space (typically with 3 components for 3D problems)
        representing displacement degrees of freedom
    ABD : bool, optional
        Flag indicating if this is for ABD matrix computation (default: False).
        Currently only affects internal dimension settings.

    Returns
    -------
    petsc4py.PETSc.NullSpace
        PETSc nullspace object containing orthonormalized basis vectors for
        translational rigid body modes. Used to constrain the linear system
        and ensure uniqueness of the solution.

    Notes
    -----
    The function creates a nullspace with 3 translational modes (one for each 
    coordinate direction). The basis vectors are orthonormalized using the 
    DOLFINx linear algebra utilities to ensure numerical stability.
    """
# Get geometric dim

    gdim = 3

    # Set dimension of nullspace
    dim=4
    if ABD:
         dim = 3

    # Create list of vectors for null space
    nullspace_basis = [
        dolfinx.la.vector(
            V.dofmap.index_map,
            bs=V.dofmap.index_map_bs,
            dtype=petsc4py.PETSc.ScalarType,
        )  # type: ignore
        for i in range(dim)
    ]

    basis = [b.array for b in nullspace_basis]
    dofs = [V.sub(i).dofmap.list.reshape(-1) for i in range(gdim)]

    # Build translational null space basis
    for i in range(gdim):
        basis[i][dofs[i]] = 1.0
    # Build rotational null space basis
    if not ABD:
         xx = V.tabulate_dof_coordinates()
         dofs_block = V.dofmap.list.reshape(-1)
         x2, x3 = xx[dofs_block, 1], xx[dofs_block, 2]
         basis[3][dofs[1]] = -x3
         basis[3][dofs[2]] = x2
         for b in nullspace_basis:
             b.scatter_forward()

    dolfinx.la.orthonormalize(nullspace_basis)
    local_size = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    basis_petsc = [
        petsc4py.PETSc.Vec().createWithArray(
            x[:local_size], bsize=gdim, comm=V.mesh.comm
        )  # type: ignore
        for x in basis
    ]

    return nullspace_basis, petsc4py.PETSc.NullSpace().create(comm=V.mesh.comm, vectors=basis_petsc)  # type: ignore



def solve_ksp(A, F, V):
    """Solve linear system Aw = F using PETSc Krylov Subspace solver.

    This function sets up and solves a linear system using PETSc's KSP
    (Krylov Subspace) solver with MUMPS direct factorization for robust
    solution of finite element systems.

    Parameters
    ----------
    A : petsc4py.PETSc.Mat
        Assembled stiffness matrix from finite element discretization
    F : petsc4py.PETSc.Vec
        Right-hand side vector (load vector or force vector)
    V : dolfinx.fem.FunctionSpace
        Function space corresponding to the degrees of freedom in the system

    Returns
    -------
    dolfinx.fem.Function
        Solution function containing the displacement field that satisfies Aw = F

    Notes
    -----
    The solver is configured with:
    - MUMPS direct solver for robustness
    - Automatic null pivot detection
    - Optimized memory management for large systems
    - Monitoring capabilities for debugging (optional)
    
    This is the primary linear solver used throughout OpenSG for solving
    finite element systems arising from ABD matrix computations and
    boundary value problems.
    """


    w = Function(V)
    ksp = petsc4py.PETSc.KSP()
    ksp.create(comm=MPI.COMM_WORLD)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    ksp.getPC().setFactorSetUpSolverType()
    mat = ksp.getPC().getFactorMatrix()
    mat.setMumpsIcntl(icntl=24, ival=1)  # detect null pivots
    mat.setMumpsIcntl(icntl=25, ival=0)  # do not compute null space again
    mat.setMumpsIcntl(icntl=4, ival=1)
    mat.setMumpsIcntl(icntl=14, ival=80)
    mat.setMumpsIcntl(icntl=7, ival=7)
    mat.setMumpsIcntl(icntl=1, ival=1e-6)
    # Enable solver monitoring

    #     PETSc.Options().setValue("ksp_monitor", "")  # Equivalent to "ksp_monitor": None in petsc_options
    ksp.setFromOptions()
    ksp.solve(F, w.x.petsc_vec)  # Solve scaled system

    w.x.petsc_vec.ghostUpdate(
        addv=petsc4py.PETSc.InsertMode.INSERT, mode=petsc4py.PETSc.ScatterMode.FORWARD
    )
    ksp.destroy()
    return w

def local_frame_1D(mesh):
    """Compute local orthonormal frame for 1D curved elements.

    This function computes an orthonormal frame (e1, e2, e3) at each point
    of a 1D curved mesh, where:
    - e1 is aligned with the global x-axis
    - e2 is the tangent vector
    - e3 is computed to complete the right-handed system

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
        1D mesh object

    Returns
    -------
    tuple
        Three ufl.Vector objects (e1, e2, e3) forming the local frame
    """
    t = Jacobian(mesh)
    t1 = as_vector([t[0, 0], t[1, 0], t[2, 0]])
    e2 = t1 / sqrt(dot(t1, t1))
    e1 = as_vector([1, 0, 0])  # Right Lay up
    e3 = cross(e1, e2)
    e3 = e3 / sqrt(dot(e3, e3))
    return e1, e2, e3

def write_beamdyn_files(beam_stiff, beam_inertia, radial_stations,file_name_prepend):
    
    # extension='K'
    # if station_list is None or len(station_list) == 0:
    #     station_list = list(range(len(geometry.ispan)))

    # radial_stations=geometry.ispan/geometry.ispan[-1]
    # radial_stations=radial_stations[station_list]

    if round(radial_stations[-1],2) ==1.0:
        radial_stations[-1]=1.0
    else:
        raise ValueError('The last radial station should be 1.0')
    if round(radial_stations[0],2) ==0.0:
        radial_stations[0]=0.0
    else:
        raise ValueError('The first radial station should be 0.0')
    
    if len(beam_stiff) != len(beam_inertia) and len(beam_stiff) != len(radial_stations):
        raise ValueError(f'\nThere are {len(beam_stiff)} stiffnesses\nThere are {len(beam_inertia)} inertias\nThere are {len(radial_stations)} radial stations \nThese need to be equal.')

    mu=[0.00257593, 0.0017469,  0.0017469,  0.0017469,  0.00257593, 0.0017469]

    beam_stiff,beam_inertia=transformMatrixToBeamDyn(beam_stiff,beam_inertia)
    _=write_beamdyn_prop('.', file_name_prepend, radial_stations, beam_stiff, beam_inertia, mu)
    
    return 


def write_beamdyn_prop(folder, wt_name, radial_stations, beam_stiff, beam_inertia, mu):
    n_pts = len(radial_stations)

        
    propFileName= 'bd_props_'+wt_name + '.inp'
    
    
    file = open(folder +'/'+propFileName, 'w')
    file.write(' ------- BEAMDYN V1.00.* INDIVIDUAL BLADE INPUT FILE --------------------------\n')
    file.write(' Test Format 1\n')
    file.write(' ---------------------- BLADE PARAMETERS --------------------------------------\n')
    file.write('%u   station_total    - Number of blade input stations (-)\n' % (n_pts))
    file.write(' 1   damp_type        - Damping type: 0: no damping; 1: damped\n')
    file.write('  ---------------------- DAMPING COEFFICIENT------------------------------------\n')
    file.write('   mu1        mu2        mu3        mu4        mu5        mu6\n')
    file.write('   (-)        (-)        (-)        (-)        (-)        (-)\n')
    file.write('\t %.5e \t %.5e \t %.5e \t %.5e \t %.5e \t %.5e\n' % (mu[0], mu[1], mu[2], mu[3], mu[4], mu[5])) 
    file.write(' ---------------------- DISTRIBUTED PROPERTIES---------------------------------\n')
    
    for i in range(n_pts):
        file.write('\t %.6f \n' % (radial_stations[i]))
        # write stiffness matrices
        for j in range(6):
            file.write('\t %.16e \t %.16e \t %.16e \t %.16e \t %.16e \t %.16e\n' % (
            beam_stiff[i, j, 0], beam_stiff[i, j, 1], beam_stiff[i, j, 2], beam_stiff[i, j, 3], beam_stiff[i, j, 4],
            beam_stiff[i, j, 5]))
        file.write('\n')

        # write inertia properties
        for j in range(6):
            file.write('\t %.16e \t %.16e \t %.16e \t %.16e \t %.16e \t %.16e\n' % (
            beam_inertia[i, j, 0], beam_inertia[i, j, 1], beam_inertia[i, j, 2], beam_inertia[i, j, 3],
            beam_inertia[i, j, 4], beam_inertia[i, j, 5]))
        file.write('\n')
        # ToDO: check correct translation of stiffness and mass matrices from VABS and anbax !!!
    file.close()

    print('Finished writing BeamDyn_Blade File')

    return propFileName

def transformMatrixToBeamDyn(beam_stiff,beam_inertia):
  #  beamDynData={}

    B = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])  # NEW transformation matrix
    T = np.dot(np.identity(3), np.linalg.inv(B))
    
    nStations, _,_=np.shape(beam_stiff)

    for i_station in range(nStations):
        beam_stiff[i_station,:,:]=trsf_sixbysix(beam_stiff[i_station,:,:], T)
        beam_inertia[i_station,:,:]=trsf_sixbysix(beam_inertia[i_station,:,:], T)
   
    return(beam_stiff,beam_inertia)

def trsf_sixbysix(M, T):
    """
    Transform six-by-six compliance/stiffness matrix. 
    change of reference frame in engineering (or Voigt) notation.
    
    Parameters
    ----------
    M : np.ndarray
        6x6 Siffness or Mass Matrix
    T : np.ndarray
        Transformation Matrix
        
    Returns
    ----------
    res : np.ndarray
        Transformed 6x6 matrix
    """

    TS_1 = np.dot(np.dot(T.T, M[0:3, 0:3]), T)
    TS_2 = np.dot(np.dot(T.T, M[3:6, 0:3]), T)
    TS_3 = np.dot(np.dot(T.T, M[0:3, 3:6]), T)
    TS_4 = np.dot(np.dot(T.T, M[3:6, 3:6]), T)

    tmp_1 = np.vstack((TS_1, TS_2))
    tmp_2 = np.vstack((TS_3, TS_4))
    res = np.hstack((tmp_1, tmp_2))
    return res

def transform_beam_matrices(beam_stiff, beam_inertia):
    """
    Transform 6x6 beam stiffness and inertia matrices to a new reference frame.

    Parameters:
    ----------
    beam_stiff : np.ndarray
        6x6 stiffness matrix
    beam_inertia : np.ndarray
        6x6 inertia matrix

    Returns:
    ----------
    beam_stiff_tr : np.ndarray
        Transformed 6x6 stiffness matrix
    beam_inertia_tr : np.ndarray
        Transformed 6x6 inertia matrix
    """
    # Example transformation matrix (customize as needed)
    B = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]]) 
    T = np.dot(np.identity(3), np.linalg.inv(B)) 


    beam_stiff_tr = trsf_sixbysix(beam_stiff, T)
    beam_inertia_tr = trsf_sixbysix(beam_inertia, T)
    return beam_stiff_tr, beam_inertia_tr

def tension_center(flex):
    ff=flex[2,2]*flex[3,3]-flex[2,3]*flex[2,3]
    return [(flex[2,2]*flex[0,3]-flex[2,3]*flex[0,2])/ff, (flex[2,3]*flex[0,3]-flex[3,3]*flex[0,2])/ff]

def beam_reaction(file_name):
    """Parse beam reaction forces from output file.

    Reads and parses beam reaction forces from a simulation output file,
    extracting force and moment components for each blade segment.

    Parameters
    ----------
    file_name : str
        Base name of the output file (without .out extension)

    Returns
    -------
    list
        List of beam force data for each segment, where each segment contains
        6 components (3 forces, 3 moments) with labels and values.
    """
    data = np.loadtxt(file_name + ".out", delimiter=",", skiprows=0, dtype=str)
    index = data[1].split()
    last_data = data[-1].split()
    pp = 7
    # beam_f=[[[index[pp+k],float(last_data[pp+k])] for k in range(6)]] ;if root also needed
    beam_force, beam_disp = [], []
    num_segment = int((len(index) - 13) / 15)
    pp = 13

    for seg in range(num_segment):
        beam_seg_reac = []
        for f in range(6):
            sc = pp + 30 * (f) + seg
            if f > 2:
                sc = pp + num_segment * (3 + f) + seg
            beam_seg_reac.append([index[sc], float(last_data[sc])])
        #  print(f,index[sc])
        beam_force.append(beam_seg_reac)
    
        beam_seg_disp=[]
        for f in range(9,15):
            sc=pp+num_segment*(f)+seg
            beam_seg_disp.append([index[sc],float(last_data[sc])])
        beam_disp.append(beam_seg_disp)
    return beam_force, beam_disp