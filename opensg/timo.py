
import ufl
import opensg


def local_frame_1D(mesh):
    t = ufl.Jacobian(mesh)
    t1 = ufl.ufl.as_vector(
        [t[0, 0], t[1, 0], t[2, 0]]
    )  # [x2-x1,y2-y1,z2-z1] tangent vector for each element.
    # Direction of tangenet vector about axis (1,0,0) depends on how we defined the mesh elements.
    # For given example, final node-initial node --> clockwise direction about (1,0,0) axis.

    e2 = t1 / ufl.sqrt(ufl.dot(t1, t1))  # converting to unit vector
    e1 = ufl.ufl.as_vector([1, 0, 0])  # Beam axis in x direction (global)
    e3 = ufl.cross(e1, e2)  # normal vector
    e3 = e3 / ufl.sqrt(ufl.dot(e3, e3))
    return e1, e2, e3



def directional_derivative(e): 
    """Obtain the curvature of curved elements
    
    Directional derivative of e1 w.r.t. e2  --> d(e1)/d(e2)
    ufl.grad (.)- is 3X1 vector --> [d(.)/dx ,d(.)/dy,d(.)/dz]
    e2_1 shows derivative of e2 vector w.r.t. e1 vector
    
    Parameters
    ----------
    e : _type_
        local element frame

    Returns
    -------
    _type_
        _description_
    """
    # a3,1

    e1, e2, e3 = e[0], e[1], e[2]
    e1_1 = ufl.dot(e1, ufl.grad(e1))
    e1_2 = ufl.dot(e2, ufl.grad(e1))
    e2_1 = ufl.dot(e1, ufl.grad(e2))
    e2_2 = ufl.dot(e2, ufl.grad(e2))
    e3_1 = ufl.dot(e1, ufl.grad(e3))
    e3_2 = ufl.dot(e2, ufl.grad(e3))

    # Initial Curvatures     #
    k11 = ufl.dot(e3_1, e1)
    k12 = ufl.dot(e3_1, e2)
    k21 = ufl.dot(e3_2, e1)
    k22 = ufl.dot(e3_2, e2)
    k13 = ufl.dot(e1_1, e2)
    k23 = ufl.dot(e1_2, e2)
    return k11, k12, k21, k22, k13, k23


def local_grad(ee, q):
    return ufl.dot(ee, ufl.grad(q))


def ddot(w, d1):
    return d1[0] * w[0] + d1[1] * w[1] + d1[2] * w[2]


def gamma_h(e, x, w):
    d11 = ufl.as_vector([-k11 * d3[ii] + k13 * d2[ii] for ii in range(3)])
    d22 = ufl.as_vector([-k22 * d3[ii] - k23 * d1[ii] for ii in range(3)])
    d12 = ufl.as_vector([-k21 * d3[ii] + k23 * d2[ii] for ii in range(3)])
    d21 = ufl.as_vector([-k12 * d3[ii] - k13 * d1[ii] for ii in range(3)])

    w_d1 = [opensg.local_grad(e[0], w[i]) for i in range(3)]
    w_d2 = [opensg.local_grad(e[1], w[i]) for i in range(3)]
    w_d11 = [opensg.local_grad(e[0], w_d1[i]) for i in range(3)]
    w_d22 = [opensg.local_grad(e[1], w_d2[i]) for i in range(3)]

    w_d12 = [opensg.local_grad(e[1], w_d1[ii]) for ii in range(3)]
    w_d21 = [opensg.local_grad(e[0], w_d2[ii]) for ii in range(3)]
    w_11 = [opensg.local_grad(d11, w[ii]) for ii in range(3)]
    w_22 = [opensg.local_grad(d22, w[ii]) for ii in range(3)]
    w_12 = [opensg.local_grad(d12, w[ii]) for ii in range(3)]
    w_21 = [opensg.local_grad(d21, w[ii]) for ii in range(3)]

    G1 = opensg.ddot(w_d1, d1)
    G2 = opensg.ddot(w_d2, d2)
    G3 = opensg.ddot(w_d1, d2) + opensg.ddot(w_d2, d1)
    G4 = -k11 * G1 - k12 * 0.5 * G3 - opensg.ddot(w_d11, d3) + k13 * opensg.ddot(w_d2, d3)
    G5 = -k22 * G2 - k21 * 0.5 * G3 - opensg.ddot(w_d22, d3) - k23 * opensg.ddot(w_d1, d3)
    G6 = (
        -(k11 + k22) * 0.5 * G3
        - k12 * G2
        - k21 * G1
        + k23 * opensg.ddot(w_d2, d3)
        - k13 * opensg.ddot(w_d1, d3)
        - opensg.ddot(w_d12, d3)
        - opensg.ddot(w_d21, d3)
    )

    E1 = as_tensor([G1, G2, G3, G4, G5, G6])
    return E1