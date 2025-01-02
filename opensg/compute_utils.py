from mpi4py import MPI
import numpy as np
import dolfinx
import basix
from dolfinx.fem import form, petsc, Function, locate_dofs_topological
from ufl import TrialFunction, TestFunction, rhs, as_tensor, dot, SpatialCoordinate, Measure, \
    inner, grad, jump, CellDiameter, FacetNormal, avg, div, as_vector

from petsc4py import PETSc
from dolfinx.fem.petsc import assemble_matrix

from mpi4py import MPI
import ufl
from contextlib import ExitStack


def generate_boundary_markers(xmin, xmax):
    def is_left_boundary(x):
        return np.isclose(x[0], xmin)
    def is_right_boundary(x):
        return np.isclose(x[0], xmax)
    return is_left_boundary, is_right_boundary


def solve_ksp(A, F, V):
    """Krylov Subspace Solver for Aw = F

    Parameters
    ----------
    A : array
        stiffness matrix
    F : array
        Load or force vector
    V : function space
        _description_

    Returns
    -------
    array
        solution vector (displacement field)
    """
    w = Function(V)
    ksp = PETSc.KSP()
    ksp.create(comm = MPI.COMM_WORLD)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    ksp.getPC().setFactorSetUpSolverType()
    ksp.getPC().getFactorMatrix().setMumpsIcntl(icntl = 24, ival = 1)  # detect null pivots
    ksp.getPC().getFactorMatrix().setMumpsIcntl(icntl = 25, ival = 0)  # do not compute null space again
    ksp.setFromOptions()
    ksp.solve(F, w.vector)
    w.vector.ghostUpdate(
        addv = PETSc.InsertMode.INSERT, mode = PETSc.ScatterMode.FORWARD
    )
    ksp.destroy()
    return w
    # return w.vector[:],w

def compute_nullspace(V):
    """Compute nullspace to restrict Rigid body motions

    Constructs a translational null space for the vector-valued function space V
    and ensures that it is properly orthonormalized.

    Parameters
    ----------
    V : functionspace
        _description_

    Returns
    -------
    NullSpace
        Nullspace of V
    """
    # extract the Index Map from the Function Space
    index_map = V.dofmap.index_map

    # initialize nullspace basis with petsc vectors
    nullspace_basis = [
        dolfinx.la.create_petsc_vector(index_map, V.dofmap.index_map_bs)
        for _ in range(4)
    ]

    with ExitStack() as stack:
        vec_local = [stack.enter_context(xx.localForm()) for xx in nullspace_basis]
        basis = [np.asarray(xx) for xx in vec_local]

    # identify the degrees of freedom indices for each subspace (x, y, z)
    dofs = [V.sub(i).dofmap.list for i in range(3)]

    # Build translational null space basis
    for i in range(3):
        basis[i][dofs[i]] = 1.0

    # Create vector space basis and orthogonalize
    dolfinx.la.orthonormalize(nullspace_basis)
    
    ret_val = PETSc.NullSpace().create(nullspace_basis, comm = MPI.COMM_WORLD)

    return ret_val

def create_gamma_e(x):
    """_summary_

    Parameters
    ----------
    x : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    gamma_e = as_tensor([
        (1, 0, 0, x[0], 0, 0),
        (0, 1, 0, 0, x[0], 0),
        (0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0),
        (0, 0, 1, 0, 0, x[0]),
    ])
    
    return gamma_e

def R_sig(C, t):
    """
    Performs rotation from local material frame to global frame
    
    Parameters:
        C: [6,6] numpy array  ! Stiffness matrix
        t: constant           ! rotation angle
        
    Returns:
        C': Rotated Stiffness matrix
    """
    th = np.deg2rad(t)
    c, s, cs = np.cos(th), np.sin(th), np.cos(th) * np.sin(th)
    R_Sig = np.array([
        (c**2, s**2, 0, 0, 0, -2 * cs),
        (s**2, c**2, 0, 0, 0, 2 * cs),
        (0, 0, 1, 0, 0, 0),
        (0, 0, 0, c, s, 0),
        (0, 0, 0, -s, c, 0),
        (cs, -cs, 0, 0, 0, c**2 - s**2),
        ])
    ret_val = np.matmul(np.matmul(R_Sig, C), R_Sig.transpose())
    return ret_val

def Dee(x, u, material_props, theta, Eps):
    """ Simplifying gamma_e.T*C*gamma_e

    Parameters
    ----------
    i : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    C = stiffness_matrix(u, material_props, theta, Eps)[1]
    x0 = x[0]
    dee = as_tensor([
        (C[0, 0], C[0, 1], C[0, 5], x0 * C[0, 0], x0 * C[0, 1], x0 * C[0, 5]),
        (C[1, 0], C[1, 1], C[1, 5], x0 * C[1, 0], x0 * C[1, 1], x0 * C[1, 5]),
        (C[5, 0], C[5, 1], C[5, 5], x0 * C[5, 0], x0 * C[5, 1], x0 * C[5, 5]),
        (x0 * C[0, 0], x0 * C[0, 1], x0 * C[0, 5], x0 * x0 * C[0, 0], x0 * x0 * C[0, 1], x0 * x0 * C[0, 5]),
        (x0 * C[1, 0], x0 * C[1, 1], x0 * C[1, 5], x0 * x0 * C[1, 0], x0 * x0 * C[1, 1], x0 * x0 * C[1, 5]),
        (x0 * C[5, 0], x0 * C[5, 1], x0 * C[5, 5], x0 * x0 * C[5, 0], x0 * x0 * C[5, 1], x0 * x0 * C[5, 5]),
    ])
    return dee

def stiffness_matrix(vector, material_parameters, theta, Eps):
    """
    Compute the [6,6] Stiffness matrix using material elastic constants
    
    Parameters:
        i: Material parameters id/ matid
        
    Returns:
        C: [6,6] Stiffness Matrix
    """
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
    C = R_sig(C, theta)
    s1 = ufl.dot(as_tensor(C), eps(vector)[1] + Eps)
    return as_tensor([(s1[0], s1[5], s1[4]), (s1[5], s1[1], s1[3]), (s1[4], s1[3], s1[2])]), C
    
def eps(vector):  # (Gamma_h * w)
    E1 = ufl.as_vector([0, 0, vector[2].dx(0), (vector[1].dx(0)), (vector[0].dx(0)), 0])
    ans = as_tensor([
        (E1[0], 0.5 * E1[5], 0.5 * E1[4]),
        (0.5 * E1[5], E1[1], 0.5 * E1[3]),
        (0.5 * E1[4], 0.5 * E1[3], E1[2]),
        ])
    return ans, E1

####################################Finite Element Implementation ##########################################
# Local Frame for Boundary (should be same as quad mesh local frame)
# Local frame from OpenSG  (wind blade quad mesh)

### local reference frames
def local_frame(mesh): 
    t = ufl.Jacobian(mesh)
    t1 = ufl.as_vector([t[0, 0], t[1, 0], t[2, 0]]) # tangential curvilinear direction -1
    t2 = ufl.as_vector([t[0, 1], t[1, 1], t[2, 1]]) # tangential curvilinear direction -2
    e3 = ufl.cross(t1, t2) # normal direction 
    e3 /= ufl.sqrt(ufl.dot(e3, e3))
    e1=  t2/ ufl.sqrt(ufl.dot(t2, t2)) # 1- direction
    e2 = ufl.cross(e3, e1)
    e2 /= ufl.sqrt(ufl.dot(e2, e2)) # 2- direction
    return e1, e2, e3
    # Note: we don't require local_frame when we are using input orientation stored in EE1, EE2,N.
# On comparison of local_frame(mesh) and given dc matrix (EE1,EE2,N), the difference is former is clockwise about beam axis, while later is ccw about beam axis
# for circumferential tangent direction. local_frame(mesh)-[e1,e2,e3] can be interpolated to DG (deg 0) with shape 3 to compare both data.
# manually generated local frame of 1D curved mesh
def local_frame_1D_manual(mesh_l, num_cells_l):
    coord = mesh_l.geometry.x
    V_l = dolfinx.fem.functionspace(mesh_l, basix.ufl.element(
        "DG", mesh_l.topology.cell_name(), 0, shape = (3, )))   
    c_to_v = mesh_l.topology.connectivity(mesh_l.topology.dim, 0)  # cell_to_vertex connectivity
    E2 = Function(V_l) 
    for cell in range(num_cells_l):
        vertices = c_to_v.links(cell)
        dofs = V_l.dofmap.cell_dofs(cell) 
        n = coord[vertices[1]]-coord[vertices[0]] #          [x2-x1,y2-y1,z2-z1]
        E2.vector[3*cell:3*cell+3] = n/np.linalg.norm(n) 
    e1= ufl.as_vector([1,0,0]) # Global x-axis 
    e3 = ufl.cross(e1,E2) 
    return e1,E2,e3
    
def local_frame_1D(mesh): 
    t = ufl.Jacobian(mesh)
    t1 = ufl.as_vector([t[0, 0], t[1, 0], t[2, 0]]) # tangent vector of 1D curved mesh
    e2 =  t1/ ufl.sqrt(ufl.dot(t1, t1))
    e1 =  ufl.as_vector([1,0,0]) # Right Lay up
    e3 = ufl.cross(e1,e2)
    e3 =  e3/ ufl.sqrt(ufl.dot(e3, e3)) 
    return e1, e2, e3

# Note grad(e2) will give derivatives of e2 in (x,yz,) global mesh frame. But, to obtain derivatives along local curvoilinear coordinates, 
# we use directional derivatives.
def deri(e): 
    # derivatives of local frame (Refer: Prof: Yu thesis)
    #a3,1
    e1,e2,e3 = e[0],e[1],e[2]
    a1_1 = ufl.dot(e1,ufl.grad(e1))
    a1_2 = ufl.dot(e2,ufl.grad(e1)) # directional derivative of e2 along e1 direction.
    a2_1 = ufl.dot(e1,ufl.grad(e2))
    a2_2 = ufl.dot(e2,ufl.grad(e2))
    a3_1 = ufl.dot(e1,ufl.grad(e3))
    a3_2 = ufl.dot(e2,ufl.grad(e3))
    
    # Initial Curvatures (Shell Element)
    k11= ufl.dot(a3_1,e1)
    k12= ufl.dot(a3_1,e2)
    k21= ufl.dot(a3_2,e1)
    k22= ufl.dot(a3_2,e2)
    k13= ufl.dot(a1_1,e2)
    k23= ufl.dot(a1_2,e2)
    return k11, k12, k21, k22, k13, k23

def local_grad(ee,q):
    return ufl.dot(ee, ufl.grad(q))

def ddot(w,d1):
    return (d1[0] * w[0] + d1[1] * w[1] + d1[2] * w[2])

############################################################################
# We need four matrices for our energy form/weak form 
# EB requires only (Gamma_h and Gamma_e) matrix
# Timo needs (Gamma_l and Gamma_d) matrix. Note in paper, Gamma_l mentioend, but that does not contribute to Timoshenko like model.
# For input, local frame (e), spatial coordinate x and dx is required as Gamma_h for left boun/right boun/main qaud mesh would be different.
# Gamma_h*w column matrix
def gamma_h(e,x,w):    
    k11,k12,k21,k22,k13,k23= deri(e) # extracting initial curvatures
    x11,x21,x31= local_grad(e[0],x[0]), local_grad(e[0],x[1]), local_grad(e[0],x[2])
    x12,x22,x32= local_grad(e[1],x[0]), local_grad(e[1],x[1]), local_grad(e[1],x[2])
    y1,y2,y3= local_grad(e[2],x[0]), local_grad(e[2],x[1]), local_grad(e[2],x[2])
    d1 = [x11,x21,x31]
    d2 = [x12,x22,x32]
    d3 = [y1,y2,y3]

    d11 = ufl.as_vector([-k11*d3[ii]+k13*d2[ii] for ii in range(3)])
    d22 = ufl.as_vector([-k22*d3[ii]-k23*d1[ii] for ii in range(3)])
    d12 = ufl.as_vector([-k21*d3[ii]+k23*d2[ii] for ii in range(3)])
    d21 = ufl.as_vector([-k12*d3[ii]-k13*d1[ii] for ii in range(3)])
    
    w_d1 = [local_grad(e[0],w[i]) for i in range(3)]
    w_d2 = [local_grad(e[1],w[i]) for i in range(3)]
    w_d11 = [local_grad(e[0],w_d1[i]) for i in range(3)]
    w_d22 = [local_grad(e[1],w_d2[i]) for i in range(3)]

    w_d12 = [local_grad(e[1],w_d1[ii]) for ii in range(3)]
    w_d21 = [local_grad(e[0],w_d2[ii]) for ii in range(3)]
    w_11 = [local_grad(d11,w[ii]) for ii in range(3)]
    w_22 = [local_grad(d22,w[ii]) for ii in range(3)]
    w_12 = [local_grad(d12,w[ii]) for ii in range(3)]
    w_21 = [local_grad(d21,w[ii]) for ii in range(3)]

    G1 = ddot(w_d1,d1)
    G2 = ddot(w_d2,d2)
    G3 = ddot(w_d1,d2)+ddot(w_d2,d1)
    G4 = -k11*G1-k12*0.5*G3-ddot(w_d11,d3)+k13*ddot(w_d2,d3)-ddot(w_11,d3)
    G5 = -k22*G2-k21*0.5*G3-ddot(w_d22,d3)-k23*ddot(w_d1,d3)-ddot(w_22,d3)
    G6 = -(k11+k22)*0.5*G3-k12*G2-k21*G1+k23*ddot(w_d2,d3)-k13*ddot(w_d1,d3) \
       -ddot(w_d12,d3)-ddot(w_d21,d3)-ddot(w_12,d3)-ddot(w_21,d3)

    ret_val = as_tensor([G1,G2,G3,G4,G5,G6])
    return ret_val


def gamma_h_new(v): # (Gamma_h * w)
    E1= as_vector([0,0,v[2].dx(0),(v[1].dx(0)),(v[0].dx(0)),0])
    return E1

def gamma_l(e,x,w): 
    # e,x required as element can be of left/right boundary or quad mesh
    y1,y2,y3 = x[2],x[0],x[1]  # In MSG-Shell formulations, y1 should be the beam axis & (y2,y3) as cross-sectional coordinates)
    
    # In mesh data, z coordinates are
    k11,k12,k21,k22,k13,k23= deri(e)
    x11,x21,x31= local_grad(e[0],x[0]), local_grad(e[0],x[1]), local_grad(e[0],x[2])
    x12,x22,x32= local_grad(e[1],x[0]), local_grad(e[1],x[1]), local_grad(e[1],x[2])
    y1,y2,y3= local_grad(e[2],x[0]), local_grad(e[2],x[1]), local_grad(e[2],x[2])
    d1 = [x11,x21,x31]
    d2 = [x12,x22,x32]
    d3 = [y1,y2,y3]
    dd1,dd2,dd3 = ufl.as_vector(d1), ufl.as_vector(d2), ufl.as_vector(d3)
    # wd = [w[i].dx(0) for i in range(3)] # dx(0) is for w'
    
    # Gamma_l*w' column matrix     (w' defined by same function space as test/trail function
    #                                are basis functions which are same for both w and w')
    w_d1 = ufl.as_vector([local_grad(dd1,w[ii]) for ii in range(3)])
    w_d2= ufl.as_vector([local_grad(dd2,w[ii]) for ii in range(3)])

    L1,L2 = x11*(ddot(d1,w)),x12*(ddot(d2,w))
    L3 = ddot(x11*dd2+x12*dd1,w)
    L4 = -2*x11*ddot(w_d1,d3)+ddot(k11*(y1*dd3-x11*dd1)-0.5*k12*(x12*dd1+x11*dd2),w)
    L5 = -2*x12*ddot(w_d2,d3)+ddot(k22*(y1*dd3-x12*dd2)-0.5*k21*(x12*dd1+x11*dd2),w)
    L6 = -2*ddot(x11*w_d2+x12*w_d1,d3)+ddot(k12*(y1*dd3-x12*dd2)+k21*(y1*dd3-x11*dd1)-0.5*(k11+k22)*(x12*dd1+x11*dd2),w)
    ret_val = as_tensor([L1,L2,L3,L4,L5,L6])
    return ret_val
    
# Gamma_e matrix   
def gamma_e(e,x):
    k11,k12,k21,k22,k13,k23= deri(e)
    x11,x21,x31= local_grad(e[0],x[0]), local_grad(e[0],x[1]), local_grad(e[0],x[2])
    x12,x22,x32= local_grad(e[1],x[0]), local_grad(e[1],x[1]), local_grad(e[1],x[2])
    y1,y2,y3= local_grad(e[2],x[0]), local_grad(e[2],x[1]), local_grad(e[2],x[2])
    
    d1 = [x11,x21,x31]
    d2 = [x12,x22,x32]
    d3 = [y1,y2,y3]
    Rn = x[1]*(x11*x32+x12*x31)-x[2]*(x11*x22+x12*x21)

    E41 = -k11*(x11**2-y1**2)-k12*x11*x12
    E51 = -k21*x11*x12-k22*(x12**2-y1**2)
    E61 = -k11*x11*x12-k12*(x12**2-y1**2)-k22*x11*x12-k21*(x11**2-y1**2)

    E42 = k11*(y1*(x[1]*y3-x[2]*y2)+x11*(x[2]*x21-x[1]*x31))+k12*0.5*(-Rn)+2*x11*(y2*x31-y3*x21)
    E52 = k22*(y1*(x[1]*y3-x[2]*y2)+x12*(x[2]*x22-x[1]*x32))+k21*0.5*(-Rn)+2*x12*(y2*x32-y3*x22)     

    E62 = (k11+k22)*0.5*(-Rn)+(k12+k21)*(y1*(x[1]*y3-x[2]*y2))+k12*x12*(x[2]*x22-x[1]*x32)\
        +2*(y2*(x12*x31+x11*x32)-y3*(x12*x21+x11*x22))+k21*x11*(x[2]*x21-x[1]*x31)

    E43= k11*x[2]*(y1**2-x11**2)-k12*x11*x12*x[2]+x11*(y3*x11-2*y1*x31)
    E53= k22*x[2]*(y1**2-x12**2)-k21*x11*x12*x[2]+x12*(y3*x12-2*y1*x32)
    E63= -(k11+k22)*x[2]*x11*x12+x[2]*(k12*(y1**2-x12**2)+k21*(y1**2-x11**2))-2*y1*(x12*x31+x11*x32)+2*y3*x11*x12

    E44= k11*x[1]*(-y1**2+x11**2)+k12*x11*x12*x[1]+x11*(-y2*x11+2*y1*x21)
    E54= k22*x[1]*(-y1**2+x12**2)+k21*x11*x12*x[1]+x12*(-y2*x12+2*y1*x22)
    E64= (k11+k22)*x11*x12*x[1]+x[1]*(k12(-y1**2+x12**2)+k21*(-y1**2+x11**2))+2*y1*(x12*x21+x11*x22)-2*y2*x11*x12

    ret_val = as_tensor([
        (x11**2, x11*(x[1]*x31-x[2]*x21), x[2]*x11**2, -x[1]*x11**2),
        (x12**2, x12*(x[1]*x32-x[2]*x22), x[2]*x12**2, -x[1]*x12**2),
        (2*x11*x12,Rn, 2*x11*x12*x[2], -2*x11*x12*x[1]),
        (E41,E42,E43,E44),
        (E51,E52,E53,E54), 
        (E61,E62,E63,E64)])
    return ret_val

def gamma_d(e,x):
    x11, x21, x31 = local_grad(e[0],x[0]), local_grad(e[0],x[1]), local_grad(e[0],x[2])
    x12, x22, x32 = local_grad(e[1],x[0]), local_grad(e[1],x[1]), local_grad(e[1],x[2])
    y1, y2, y3= local_grad(e[2],x[0]), local_grad(e[2],x[1]), local_grad(e[2],x[2])
    O = ufl.as_vector((0,0,0,0))
    R = ufl.as_vector((-y1,-y3*x[1]+y2*x[2],-x[2]*y1,x[1]*y1))
    ret_val = as_tensor([O,O,O,x11*x11*R,x12*x12*R,2*x11*x12*R])
    return ret_val


def local_boun(mesh, frame, subdomains):
    # Obtaining preprocessing mesh terms for boundary as well as main wb (quad mesh).
    # not to confuse with l here, it applies for right boundary as well by giving appropriate arguments
    V = dolfinx.fem.functionspace(mesh, basix.ufl.element(
        "S", mesh.topology.cell_name(), 2, shape = (3, )))
    le1, le2, le3 = frame
    e1l, e2l, e3l = Function(V), Function(V), Function(V)
    
    fexpr1 = dolfinx.fem.Expression(le1,V.element.interpolation_points(), comm = MPI.COMM_WORLD)
    e1l.interpolate(fexpr1) 
    
    fexpr2 = dolfinx.fem.Expression(le2,V.element.interpolation_points(), comm = MPI.COMM_WORLD)
    e2l.interpolate(fexpr2) 
    
    fexpr3 = dolfinx.fem.Expression(le3,V.element.interpolation_points(), comm = MPI.COMM_WORLD)
    e3l.interpolate(fexpr3) 
    
    frame = [e1l,e2l,e3l]
    dv = TrialFunction(V)
    v = TestFunction(V)
    x = SpatialCoordinate(mesh)
    dx = Measure('dx')(domain=mesh, subdomain_data=subdomains)
        
    return frame, V, dv, v, x, dx
          
def A_mat(ABD, e_l, x_l, dx_l, nullspace_l, v_l, dvl, nphases):
    """Assembly matrix

    Parameters
    ----------
    ABD : _type_
        _description_
    e_l : _type_
        _description_
    x_l : _type_
        _description_
    dx_l : _type_
        _description_
    nullspace_l : _type_
        _description_
    v_l : _type_
        _description_
    dvl : _type_
        _description_
    nphases : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    F2 = sum([dot(dot(as_tensor(ABD[i]),gamma_h(e_l,x_l,dvl)), gamma_h(e_l,x_l,v_l))*dx_l(i) for i in range(nphases)])   
    A_l = assemble_matrix(form(F2))
    A_l.assemble()
    A_l.setNullSpace(nullspace_l) 
    return A_l

def solve_eb_boundary(ABD, meshdata):

    """_summary_

    Parameters
    ----------
    ABD : List of ABD matrices for each phase
        _description_
    meshdata : _type_
        _description_
    nphases : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """

    # assert len(ABD) == nphases
    nphases = len(ABD)

    mesh = meshdata["mesh"]
    # frame = meshdata["frame"]
    frame = local_frame_1D(mesh)
    subdomains = meshdata["subdomains"]
    nullspace = meshdata["nullspace"]
    # For applying bc, we only require solved fluctuating functions (V0) as input to bc.
    e, V, dv, v_, x, dx = local_boun(mesh,frame,subdomains)          
    mesh.topology.create_connectivity(1, 1)
    V0 = initialize_array(V)[0]
    A = A_mat(ABD, e,x,dx,compute_nullspace(V),v_,dv, nphases)
    for p in range(4):
        Eps = gamma_e(e,x)[:,p]
        F2 = sum([dot(dot(as_tensor(ABD[i]),Eps), gamma_h(e,x,v_))*dx(i) for i in range(nphases)])   
        r_he = form(rhs(F2))
        F = petsc.assemble_vector(r_he)
        F.ghostUpdate(addv = PETSc.InsertMode.ADD, mode = PETSc.ScatterMode.REVERSE)
        nullspace.remove(F)
        w = solve_ksp(A,F,V)
        V0[:,p]= w.vector[:] 
    return V0

# NOTE: different definition between timo and eb scripts
# def initialize_array(V_l):
#     xxx = 3*len(np.arange(*V_l.dofmap.index_map.local_range))  # total dofs 
#     V0 = np.zeros((xxx,4))
#     Dle = np.zeros((xxx,4))
#     Dhe = np.zeros((xxx,4))
#     Dhd = np.zeros((xxx,4))
#     Dld = np.zeros((xxx,4))
#     D_ed = np.zeros((4,4))
#     D_dd = np.zeros((4,4))
#     D_ee = np.zeros((4,4))  
#     V1s = np.zeros((xxx,4))
#     return V0,Dle,Dhe,Dhd,Dld,D_ed,D_dd,D_ee,V1s

def initialize_array(V):
    """
    Output initialized numpy arrays
    From Timo_Taper_ABDrearranged_debugging.py
    
    Parameters:
        V: functionspace
        
    Returns:
        V0,Dle,Dhe,D_ee,V1s: numpy arrays
    """
    ndofs=3*len(np.arange(*V.dofmap.index_map.local_range))  # total dofs of mesh
    V0 = np.zeros((ndofs,4))
    Dle=np.zeros((ndofs,4))
    Dhe=np.zeros((ndofs,4))
    D_ee=np.zeros((4,4))  
    V1s=np.zeros((ndofs,4)) 
    return V0,Dle,Dhe,D_ee,V1s

def dof_mapping_quad(V, v2a, V_boundary, w_ll, boundary_facets, entity_mapl):
    """dof mapping makes solved unknown value w_l(Function(V_l)) assigned to v2a (Function(V)). 
    The boundary of wind blade mesh is a 1D curve. The facet/edge number is obtained from cell to edge connectivity (conn3) showed in subdomain subroutine.
    The same facet/edge number of extracted mesh_l (submesh) is obtaine din entity_mapl (gloabl mesh number). refer how submesh was generated.
    Therefore, once identifying the edge number being same for global(mesh)&boundary mesh(mesh_l), we equate the dofs and store w_l to v2a.
    The dofs can be verified by comparing the coordinates of local and global dofs if required. 

    Parameters
    ----------
    V : _type_
        _description_
    v2a : _type_
        ufl function to store boundary solutions
    V_boundary : _type_
        Boundary functionspace
    w_ll : 1D array (len(4))
        Dims=[ndofs_leftmesh,1]. numpy column vector containing columns of V0_l (i.e. V0_l[:,p]).
    boundary_facets : array
        Dims=[num_facets_left,1]. Boundary facets id (numbering in boundary mesh).
    entity_mapl : array
        Dim=[num_facets_left,1].  Boundary facets id (numbering in Wb Segment mesh)


    Returns
    -------
    _type_
        _description_
    """
    dof_S2L = []
    deg = 2 # hard coded for now, could change to optional kwarg.
    for i,xx in enumerate(entity_mapl):
        dofs = locate_dofs_topological(V, 1, np.array([xx]))
        dofs_left= locate_dofs_topological(V_boundary, 1, np.array([boundary_facets[i]]))
        for k in range(deg+1):
            if dofs[k] not in dof_S2L:
                dof_S2L.append(dofs[k])
                for j in range(3):
                    v2a.vector[3*dofs[k]+j] = w_ll[3*dofs_left[k]+j] # store boundary solution of fluctuating functions
    return v2a


def tangential_projection(u: ufl.Coefficient, n: ufl.FacetNormal) -> ufl.Coefficient:
    """
    See for instance:
    https://link.springer.com/content/pdf/10.1023/A:1022235512626.pdf
    """
    return (ufl.Identity(u.ufl_shape[0]) - ufl.outer(n, n)) * u


def facet_vector_approximation(
    V: dolfinx.fem.FunctionSpace,
    mt: dolfinx.mesh.MeshTags | None = None,
    mt_id: int | None = None,
    tangent: bool = False,
    interior: bool = False,
    jit_options: dict | None = None,
    form_compiler_options: dict | None = None) -> dolfinx.fem.Function:
 
    jit_options = jit_options if jit_options is not None else {}
    form_compiler_options = form_compiler_options if form_compiler_options is not None else {}

    comm  = V.mesh.comm # MPI Communicator
    n     = ufl.FacetNormal(V.mesh) # UFL representation of mesh facet normal
    u, v  = ufl.TrialFunction(V), ufl.TestFunction(V) # Trial and test functions

    # Create interior facet integral measure
    dS = ufl.dS(domain=V.mesh) if mt is None else ufl.dS(domain=V.mesh, subdomain_data=mt, subdomain_id=mt_id)
    
    # If tangent==True, the right-hand side of the problem should be a tangential projection of the facet normal vector.


    c = dolfinx.fem.Constant(V.mesh, (1.0, 1.0, 1.0)) # Vector to tangentially project the facet normal vectors on

    a = (ufl.inner(u('+'), v('+')) + ufl.inner(u('-'), v('-'))) * dS
    L = ufl.inner(tangential_projection(c, n('+')), v('+')) * dS \
        + ufl.inner(tangential_projection(c, n('-')), v('-')) * dS
    # If tangent==false the right-hand side is simply the facet normal vector.
    a = (ufl.inner(u('+'), v('+')) + ufl.inner(u('-'), v('-'))) * dS
    L = (ufl.inner(n('+'), v('+')) + ufl.inner(n('-'), v('-'))) * dS

    # Find all boundary dofs, which are the dofs where we want to solve for the facet vector approximation.
    # Start by assembling test functions integrated over the boundary integral measure.
    ones = dolfinx.fem.Constant(V.mesh, dolfinx.default_scalar_type((1,) * V.mesh.geometry.dim)) # A vector of ones

    local_val = dolfinx.fem.form((ufl.dot(ones, v('+')) + ufl.dot(ones, v('-')))*dS)
    local_vec = dolfinx.fem.assemble_vector(local_val)

    # For the dofs that do not lie on the boundary of the mesh the assembled vector has value zero.
    # Extract these dofs and use them to deactivate the corresponding block in the linear system we will solve.
    bdry_dofs_zero_val  = np.flatnonzero(np.isclose(local_vec.array, 0))
    deac_blocks = np.unique(bdry_dofs_zero_val // V.dofmap.bs).astype(np.int32)

    # Create sparsity pattern by manipulating the blocks to be deactivated and set
    # a zero Dirichlet boundary condition for these dofs.
    bilinear_form = dolfinx.fem.form(a, jit_options=jit_options,
                                 form_compiler_options=form_compiler_options)
    pattern = dolfinx.fem.create_sparsity_pattern(bilinear_form)
    pattern.insert_diagonal(deac_blocks)
    pattern.finalize()
    u_0 = dolfinx.fem.Function(V)
    u_0.vector.set(0)
    bc_deac = dolfinx.fem.dirichletbc(u_0, deac_blocks)

    # Create the matrix
    A = dolfinx.cpp.la.petsc.create_matrix(comm, pattern)
    A.zeroEntries()

    # Assemble the matrix with all entries
    form_coeffs = dolfinx.cpp.fem.pack_coefficients(bilinear_form._cpp_object)
    form_consts = dolfinx.cpp.fem.pack_constants(bilinear_form._cpp_object)
    dolfinx.fem.petsc.assemble_matrix(A, bilinear_form, constants=form_consts, coeffs=form_coeffs, bcs=[bc_deac])

    # Insert the diagonal with the deactivated blocks.
    if bilinear_form.function_spaces[0] is bilinear_form.function_spaces[1]:
        A.assemblyBegin(PETSc.Mat.AssemblyType.FLUSH)
        A.assemblyEnd(PETSc.Mat.AssemblyType.FLUSH)
        dolfinx.cpp.fem.petsc.insert_diagonal(A=A, V=bilinear_form.function_spaces[0], bcs=[bc_deac._cpp_object], diagonal=1.0)
    A.assemble()

    # Assemble the linear form and the right-hand side vector.
    linear_form = dolfinx.fem.form(L, jit_options=jit_options,
                               form_compiler_options=form_compiler_options)
    b = dolfinx.fem.petsc.assemble_vector(linear_form)


    # Apply lifting to the right-hand side vector and set boundary conditions.
    dolfinx.fem.petsc.apply_lifting(b, [bilinear_form], [[bc_deac]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.petsc.set_bc(b, [bc_deac])

    # Setup a linear solver using the Conjugate Gradient method.
    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setType("cg")
    solver.rtol = 1e-8
    solver.setOperators(A)

    # Solve the linear system and perform ghost update.
    nh    = dolfinx.fem.Function(V)     # Function for the facet vector approximation
    solver.solve(b, nh.vector)
    nh.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    # Normalize the vectors to get the unit facet normal/tangent vector.
    nh_norm = ufl.sqrt(ufl.inner(nh, nh)) # Norm of facet vector
    cond_norm = ufl.conditional(ufl.gt(nh_norm, 1e-10), nh_norm, 1.0) # Avoid division by zero
    nh_norm_vec = ufl.as_vector((nh[0]/cond_norm, nh[1]/cond_norm, nh[2]/cond_norm))

    nh_normalized = dolfinx.fem.Expression(nh_norm_vec, V.element.interpolation_points())
    
    n_out = dolfinx.fem.Function(V)
    n_out.interpolate(nh_normalized)

    return n_out


def deri_constraint(dvl, v_l, mesh, nh):
    """
    Generate C0-Interior Penalty term for bilinear weak form to constrain normal and tangent derivative along the interface
    Boundary mesh only requires normal derivative constraint, and WB mesh requires both tangent and normal derivative constraint
    Parameters:
        dv,v_: Trail and test function (either WB segment/boundary)
        mesh: Input mesh (either WB segment/boundary)
        nh: TODO
        
    Returns:
        nn: bilinear normal derivative constraint terms (boundary mesh)
        nn+tt: bilinear constraint terms for both normal and tangent boundary (WB surface mesh).
    """
    h = CellDiameter(mesh)
    n = FacetNormal(mesh)
    h_avg = (h('+') + h('-')) / 2.0
    dS=Measure('dS')(domain=mesh)
    if mesh.topology.dim==1:
        alpha=1e10
        nn= - inner(avg(div(grad(dvl))), jump(grad(v_l), n))*dS \
            - inner(jump(grad(dvl), n), avg(div(grad(v_l))))*dS \
            + (alpha/h_avg**2)*inner(jump(grad(dvl),n), jump(grad(v_l),n))*dS
        return nn
    elif mesh.topology.dim==2: 
        alpha=1e10
        beta=4e5
        nn= - inner(avg(div(grad(dvl))), jump(grad(v_l), n))*dS \
            - inner(jump(grad(dvl), n), avg(div(grad(v_l))))*dS \
            + (alpha/h_avg**2)*inner(jump(grad(dvl),n), jump(grad(v_l),n))*dS
        tt=- inner(avg(div(grad(dvl))), jump(grad(v_l), nh))*dS \
           - inner(jump(grad(dvl), nh), avg(div(grad(v_l))))*dS \
           + (beta/h_avg**2)*inner(jump(grad(dvl),nh), jump(grad(v_l),nh))*dS
        return nn+tt


# def Stiff_mat(i, angle, material_parameters):     
#     """
#     Compute the [6,6] Stiffness matrix using material elastic constants
    
#     Parameters:
#         i: Material parameters id/ matid
        
#     Returns:
#         C: [6,6] Stiffness Matrix
#     """
#     E1,E2,E3,G12,G13,G23,v12,v13,v23= material_parameters[matid[ii][i]]
#     S = np.zeros((6,6))
#     S[0,0], S[1,1], S[2,2]=1/E1, 1/E2, 1/E3
#     S[0,1], S[0,2]= -v12/E1, -v13/E1
#     S[1,0], S[1,2]= -v12/E1, -v23/E2
#     S[2,0], S[2,1]= -v13/E1, -v23/E2
#     S[3,3], S[4,4], S[5,5]= 1/G23, 1/G13, 1/G12    
#     C = np.linalg.inv(S)
#     theta = angle[ii][i] # ii denotes the layup id
#     C = as_tensor(R_sig(C,theta)) 
#     return  C


# def Dee(i):  
#     """
#     Performs < gamma_e.T Stiffness_matrix gamma_e > and give simplified form
    
#     Parameters:
#         i: matid 
    
#     Returns:
#         Dee: [6,6] ufl tensor 
#     """
#     C = Stiff_mat(i)
#     x0 = x[0]
#     return as_tensor([(C[0,0],C[0,1],C[0,5],x0*C[0,0],x0*C[0,1],x0*C[0,5]),
#                     (C[1,0],C[1,1],C[1,5],x0*C[1,0],x0*C[1,1],x0*C[1,5]),
#                     (C[5,0],C[5,1],C[5,5],x0*C[5,0],x0*C[5,1],x0*C[5,5]),
#                     (x0*C[0,0],x0*C[0,1],x0*C[0,5],x0*x0*C[0,0],x0*x0*C[0,1],x0*x0*C[0,5]),
#                     (x0*C[1,0],x0*C[1,1],x0*C[1,5],x0*x0*C[1,0],x0*x0*C[1,1],x0*x0*C[1,5]),
#                     (x0*C[5,0],x0*C[5,1],x0*C[5,5],x0*x0*C[5,0],x0*x0*C[5,1],x0*x0*C[5,5])])