
# Initialization of libraries
from mpi4py import MPI
import numpy as np
import dolfinx
from dolfinx.fem import (
    form,
    petsc,
    Function,
    apply_lifting,
    set_bc,
    locate_dofs_topological,
)
from ufl import rhs, as_tensor, dot
from scipy.sparse import csr_matrix
import petsc4py.PETSc
from dolfinx.fem.petsc import assemble_matrix
import opensg
import opensg.utils.solid as utils
import opensg.utils.shared as shared_utils
import scipy


### ABD matrix computation
# @profile
# NOTE can pass in thick[ii], nlay[ii], etc instead of the dictionaries

# Generate ABD matrix (Plate model)
# def compute_timo_boun(ABD, mesh, subdomains, frame, nullspace, sub_nullspace, nphases):
    
def compute_timo_boun(ABD, boundary_submeshdata, nh):
    """
    Solve EB and Timo model for boundary mesh.
    The output fluctuating functions V0 and V1s are used dirichilet boundary constraints

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
    boundary_frame = shared_utils.local_frame_1D(boundary_mesh)

    nphases = len(ABD)

    e, V_l, dv, v_, x, dx = utils.local_boun(
        boundary_mesh, boundary_frame, boundary_subdomains
    )
    boundary_mesh.topology.create_connectivity(1, 1)
    V0, Dle, Dhe, D_ee, V1s = utils.initialize_array(V_l)
    nullspace_l = shared_utils.compute_nullspace(V_l)
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
    ff = shared_utils.deri_constraint(dv, v_, boundary_mesh, nh)
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
        w = shared_utils.solve_ksp(A_l, F_l, V_l)
        V0[:, p] = w.x.array[:]

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
    ff = shared_utils.deri_constraint(dv, v_, boundary_mesh, nh)
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

def compute_timo_boun(mat_param, boundary_submeshdata):
    """
    Solve EB and Timo model for boundary mesh.
    The output fluctuating functions V0 and V1s are used dirichilet boundary constraints

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
    
    nphases = len(mat_param)

    e, V_l, dv, v_, x, dx = utils.local_boun(
        boundary_mesh, boundary_frame, boundary_subdomains
    )
    boundary_null = shared_utils.compute_nullspace(V_l)
    boundary_mesh.topology.create_connectivity(2,2)
    V0, Dle, Dhe, D_ee, V1s = utils.initialize_array(V_l)

    F2 = sum(
        [
            dot(
                dot(utils.C(i,boundary_frame,mat_param),utils.gamma_h(dx,dv,dim=2)),
                utils.gamma_h(dx, v_,dim=2)
            )
            * dx(i)
            for i in range(nphases)
        ]
    )
    A_l = assemble_matrix(form(F2))
    A_l.assemble()
    A_l.setNullSpace(boundary_null)

    gamma_e = utils.gamma_e(x)
    for p in range(4):
        F2 = sum(
            [
            dot(dot(utils.C(ii,boundary_frame,mat_param),gamma_e[:,p]),utils.gamma_h(dx,v_,dim=2))
             *dx(ii) for ii in range(nphases)
            ]
        )
        r_he = form(rhs(F2))
        F_l = petsc.assemble_vector(r_he)
        F_l.ghostUpdate(
            addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE
        )
        boundary_null.remove(F_l)
        Dhe[:, p] = petsc.assemble_vector(r_he)[:]
        w = shared_utils.solve_ksp(A_l, F_l, V_l)
        V0[:,p]= w.x.array[:] 
        
    V0_csr=csr_matrix(V0) 
    D1=V0_csr.T.dot(csr_matrix(-Dhe))
    for s in range(4):
        for k in range(4):
            f = dolfinx.fem.form(
                sum(
                    [
                        dot(dot(gamma_e.T, utils.C(i,boundary_frame,mat_param)), gamma_e)[s, k] * dx(i)
                        for i in range(nphases)
                    ]
                )
            )
            D_ee[s, k] = dolfinx.fem.assemble_scalar(f)

    D_eff = D_ee + D1  # Effective Stiffness Matrix (EB)

    F1 = sum(
        [
            dot(
                dot(utils.C(i,boundary_frame,mat_param), utils.gamma_l(dv)),
                utils.gamma_l(v_)
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
        F1 = sum(
            [
                dot(dot(utils.C(i,boundary_frame,mat_param), gamma_e[:, p]), utils.gamma_l(v_)) * dx(i)
                for i in range(nphases)
            ]
        )
        Dle[:, p] = petsc.assemble_vector(form(F1))[:]

    F_dhl = sum(
        [
            dot(
                dot(utils.C(i,boundary_frame,mat_param), utils.gamma_h(dx, dv,dim=2)),
                utils.gamma_l(v_)
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
    
    #DhlV0
    DhlV0=Dhl.T.dot(V0_csr) 
    
    #DhlTV0Dle
    DhlTV0Dle=Dhl.dot(V0_csr)+csr_matrix(Dle)
    
    #V0DllV0
    V0DllV0=(V0_csr.T.dot(Dll)).dot(V0_csr)

    # V1s  ****Updated from previous version as for solving boundary V1s, we can directly use (A_l V1s=b),  and solve for V1s
    b = (DhlTV0Dle - DhlV0).toarray()
    for p in range(4):
        F=petsc4py.PETSc.Vec().createWithArray(b[:,p],comm=MPI.COMM_WORLD)
        F.ghostUpdate(
            addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE
        )
        boundary_null.remove(F)
        w = shared_utils.solve_ksp(A_l, F, V_l)
        V1s[:,p]= w.x.array[:] 

    # Ainv
    Ainv=np.linalg.inv(D_eff).astype(np.float64)
    
    # B_tim
    B_tim=DhlTV0Dle.T.dot(V0_csr)
    B_tim=B_tim.toarray().astype(np.float64)

    # C_tim
    C_tim= V0DllV0 + csr_matrix(V1s).T.dot(DhlV0 + DhlTV0Dle) 
    C_tim=0.5*(C_tim+C_tim.T)
    C_tim=C_tim.toarray().astype(np.float64)

    # Ginv
    Q_tim = np.matmul(Ainv, np.array([(0, 0), (0, 0), (0, -1), (1, 0)])).astype(np.float64)
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

    return np.around(D_eff), np.around(Deff_srt), V0, V1s
    

def compute_stiffness(
    mat_param,
    meshdata,
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
    tdim=meshdata["mesh"].topology.dim
    fdim = tdim - 1
    nphases=len(mat_param)
    # Initialize terms
    # NOTE: why do we need the frame from local_frame_1D instead of the already computed frames
    e_l, V_l, dvl, v_l, x_l, dx_l = utils.local_boun(l_submesh["mesh"], l_submesh["frame"],l_submesh["subdomains"])
    e_r, V_r, dvr, v_r, x_r, dx_r = utils.local_boun(r_submesh["mesh"], r_submesh["frame"] ,r_submesh["subdomains"])

    # V0_l,V0_r=solve_boun(mesh_l,local_frame_1D(mesh_l),subdomains_l),solve_boun(mesh_r,local_frame_1D(mesh_l),subdomains_r)
    D_effEB_l, Deff_l, V0_l, V1_l = opensg.core.solid.compute_timo_boun(mat_param, l_submesh)
    D_effEB_r, Deff_r, V0_r, V1_r = opensg.core.solid.compute_timo_boun(mat_param, r_submesh)

    # ***************** Wb Segment (surface mesh) computation begins************************
    e, V, dv, v_, x, dx = utils.local_boun(meshdata["mesh"], meshdata["frame"], meshdata["subdomains"])
    V0, Dle, Dhe, D_ee, V1s = utils.initialize_array(V)
    gamma_e = utils.gamma_e(x)
    meshdata["mesh"].topology.create_connectivity(2, 3)
    l_submesh["mesh"].topology.create_connectivity(2, 2)
    r_submesh["mesh"].topology.create_connectivity(2, 2)

    F2 = sum(
        [
            dot(dot(utils.C(i, meshdata["frame"],mat_param), utils.gamma_h(dx, dv,dim=3)), utils.gamma_h(dx, v_,dim=3)) * dx(i)
            for i in range(nphases)
        ]
    )

    # bc applied
    boundary_dofs = locate_dofs_topological(
        V, fdim, np.concatenate((r_submesh["entity_map"], l_submesh["entity_map"]), axis=0)
    )
    a= form(F2)
    v2a = Function(V)
    bc = dolfinx.fem.dirichletbc(v2a, boundary_dofs)
    A = assemble_matrix(a, [bc])  # Obtain coefficient matrix with BC applied: AA
    A.assemble()

    # Assembly
    # Running for 4 different F vector. However, F has bc applied to it where, stored known values of v2a is provided for each loop (from boun solve).

    for p in range(4):  # 4 load cases meaning
        # Boundary
        v2a = Function(V)
        
        v2a = utils.dof_mapping_quad(V, v2a, V_l, V0_l[:, p], l_submesh["facets"], l_submesh["entity_map"])
        v2a = utils.dof_mapping_quad(V, v2a, V_r, V0_r[:, p], r_submesh["facets"], r_submesh["entity_map"])

        F2 = -sum(
            [
                dot(dot(utils.C(i,meshdata["frame"],mat_param), gamma_e[:, p]), utils.gamma_h(dx, v_,dim=3)) * dx(i)
                for i in range(nphases)
            ]
        )
        bc = [dolfinx.fem.dirichletbc(v2a, boundary_dofs)]
        F = petsc.assemble_vector(form(F2))
        Dhe[:,p]=F

        apply_lifting(
            F, [a], [bc]
        )  # apply bc to rhs vector (Dhe) based on known fluc solutions  C(i,frame,mat_param)
        set_bc(F, bc)

        w = shared_utils.solve_ksp(A, F, V)
        V0[:, p] = w.x.array[:]

    V0_csr=csr_matrix(V0)    
    D1=V0_csr.T.dot(csr_matrix(-Dhe)).astype(np.float64)

    for s in range(4):
        for k in range(4):
            f = dolfinx.fem.form(
                sum(
                    [
                        dot(dot(gamma_e.T, utils.C(i,meshdata["frame"],mat_param)), gamma_e)[s, k] * dx(i)
                        for i in range(nphases)
                    ]
                )
            )
            D_ee[s, k] = dolfinx.fem.assemble_scalar(f)
    L = max(meshdata["mesh"].geometry.x[:,0]) - min(meshdata["mesh"].geometry.x[:,0])
    D_eff = (D_ee.astype(np.float64) + D1)/L 
    D_eff=0.5*(D_eff+D_eff.T)

    ##################Timoshenko Stiffness Matrix for WB segment begins###################################
    # Process is similar to Timoshenko boundary implemented over WB segment mesh
    F1 = sum(
        [
            dot(dot(utils.C(i,meshdata["frame"],mat_param), utils.gamma_l(v_)), utils.gamma_l(dv)) * dx(i)
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
            dot(dot(utils.C(i,meshdata["frame"],mat_param), utils.gamma_h(x, dv,dim=3)), utils.gamma_l(v_)) * dx(i)
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
                dot(dot(utils.C(i,meshdata["frame"],mat_param), utils.gamma_e(x)[:, p]), utils.gamma_l(v_)) * dx(i)
                for i in range(nphases)
            ]
        )
        Dle[:, p] = petsc.assemble_vector(form(F1))[:]

    #DhlV0
    DhlV0=Dhl.T.dot(V0_csr) 
    
    #DhlTV0Dle
    DhlTV0Dle=Dhl.dot(V0_csr)+csr_matrix(Dle)
    
    #V0DllV0
    V0DllV0=(V0_csr.T.dot(Dll)).dot(V0_csr)

    # V1s
    b = (DhlTV0Dle - DhlV0).toarray()

    # Assembly
    # For WB mesh (surface elements, for solving (A V1s = b), directly cannot be computed as we require dirichilet bc)
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
        w = shared_utils.solve_ksp(A, F, V)
        V1s[:, p] = w.x.array[:]

    # Ainv
    Ainv=np.linalg.inv(D_eff).astype(np.float64)
    
    # B_tim
    B_tim=DhlTV0Dle.T.dot(V0_csr)
    B_tim=B_tim.toarray().astype(np.float64)/L

    # C_tim
    C_tim= V0DllV0 + csr_matrix(V1s).T.dot(DhlV0 + DhlTV0Dle) 
    C_tim=0.5*(C_tim+C_tim.T)
    C_tim=C_tim.toarray().astype(np.float64)/L

    # Ginv
    Q_tim = np.matmul(Ainv, np.array([(0, 0), (0, 0), (0, -1), (1, 0)])).astype(np.float64)
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
    print('Tapered Timo Stiffness: VABS Convention')    
    np.set_printoptions(precision=4) 
    print(np.around(Deff_srt)) 
    return Deff_srt, V0, V1s 

