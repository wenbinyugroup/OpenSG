"""5-variable (omega3-ELIMINATED) numerical solution of the circle solid SG.

Analytical1 program: eliminate omega3 through the drilling relation
    omega3 = -w1'/(2 C33) - (C32/C33) omega2        (C33 = yd2, C32 = -yd3)
and minimize the shell strain energy over the FIVE fields
    w1, w2, w3, omega1, omega2
by calculus of variations.  Numerically: Fourier series in theta for each field
(derivatives exact, so omega3' needs no C1 elements), energy by fine-grid
quadrature, 1/C33 floored at |C33| >= 1e-3 at the two singular points
(theta = 0, pi), constraints <w_i> = 0, <omega1> = 0.  Isotropic wall.

Compares C3D diagonal with the 6-DOF multiplier route (solid_props.ring_solid
via build_solid_bundle) and the analytical diag(2Et/R, 0, ..., 0).

Run (from this folder):  python circle_5dof_eliminated.py
"""
import numpy as np

from opensg_shell import build_solid_bundle, GBAR_ORDER

############### User Input #################################
R, t = 1.0, 0.03
E, nu = 70.0e9, 0.30
NH = 12                  # Fourier harmonics per field
NG = 720                 # quadrature grid points
############################################################

A_cell = np.pi * R**2
A11 = E * t / (1 - nu**2); A12 = nu * A11; A66 = E * t / (2 * (1 + nu))
D11 = E * t**3 / (12 * (1 - nu**2)); D66 = E * t**3 / (24 * (1 + nu))
Gs = (5.0 / 6.0) * E * t / (2 * (1 + nu))

th = (np.arange(NG) + 0.5) * 2 * np.pi / NG
dth = 2 * np.pi / NG
yd2, yd3 = np.sin(th), np.cos(th)                 # tangent (Analytical1 conv.)
c2, c3 = yd2, yd3
y2n, y3n = -yd3, yd2                              # normals; C33 = y3n, C32 = y2n
C33f = np.where(np.abs(y3n) > 1e-3, y3n, np.sign(y3n + 1e-30) * 1e-3)

# Fourier basis: [1, cos th, sin th, ..., cos NH th, sin NH th] per field
nb = 1 + 2 * NH
B = np.zeros((NG, nb)); Bp = np.zeros((NG, nb))   # value and d/ds
B[:, 0] = 1.0
for n in range(1, NH + 1):
    B[:, 2*n-1] = np.cos(n * th); B[:, 2*n] = np.sin(n * th)
    Bp[:, 2*n-1] = -n * np.sin(n * th) / R; Bp[:, 2*n] = n * np.cos(n * th) / R

nf = 5 * nb                                        # w1 w2 w3 om1 om2
sl = [slice(k * nb, (k + 1) * nb) for k in range(5)]


def field_rows(q):
    """8 strain rows on the grid for coefficient vector q (per unit macro=0)."""
    w1, w2, w3 = B @ q[sl[0]], B @ q[sl[1]], B @ q[sl[2]]
    o1, o2 = B @ q[sl[3]], B @ q[sl[4]]
    w1p, w2p, w3p = Bp @ q[sl[0]], Bp @ q[sl[1]], Bp @ q[sl[2]]
    o1p, o2p = Bp @ q[sl[3]], Bp @ q[sl[4]]
    o3 = -w1p / (2 * C33f) - (y2n / C33f) * o2
    # o3' exactly from the basis: differentiate the expression numerically via
    # spectral derivative of o3 (o3 is smooth off the poles; floor regularizes)
    o3h = np.fft.rfft(o3) / NG
    kk = np.arange(len(o3h))
    o3p = np.fft.irfft(1j * kk * o3h * NG, NG) / R
    e22 = c2 * w2p + c3 * w3p
    e12 = w1p
    g13 = c2 * o2 + c3 * o3
    g23 = y2n * w2p + y3n * w3p - o1
    k22 = -o1p
    k12 = c2 * o2p + c3 * o3p
    return e22, e12, g13, g23, k22, k12


MACRO = {                                          # macro parts on the grid
    'ZERO': dict(),
    'G11':  dict(e11=np.ones(NG)),
    'G22':  dict(e22=c2**2, g23=y2n*c2),
    'G33':  dict(e22=c3**2, g23=y3n*c3),
    '2G23': dict(e22=c2*c3, g23=0.5*(c3*y2n + c2*y3n)),
    '2G13': dict(e12=c3, g13=0.5*y3n),
    '2G12': dict(e12=c2, g13=0.5*y2n),
}
KEYS = ['G11', 'G22', 'G33', '2G23', '2G13', '2G12']


def energy_q(q, mk):
    m = MACRO[mk]
    e22, e12, g13, g23, k22, k12 = field_rows(q)
    e11 = m.get('e11', 0)
    e22 = e22 + m.get('e22', 0)
    e12 = e12 + m.get('e12', 0)
    g13 = g13 + m.get('g13', 0)
    g23 = g23 + m.get('g23', 0)
    U = (A11*e11**2 + 2*A12*e11*e22 + A11*e22**2 + A66*e12**2
         + D11*k22**2 + D66*k12**2 + Gs*(g13**2 + g23**2))
    return np.sum(U) * R * dth


# quadratic form by basis probing (functional is exactly quadratic)
print("assembling 5-field Fourier system (%d dof)..." % nf)
K = np.zeros((nf, nf)); F = {k: np.zeros(nf) for k in KEYS}
c0 = {k: energy_q(np.zeros(nf), k) for k in list(KEYS) + ['ZERO']}
eye = np.eye(nf)
Q_i = np.array([energy_q(eye[i], 'ZERO') for i in range(nf)])   # pure quadratic
for i in range(nf):
    for j in range(i, nf):
        val = energy_q(eye[i] + eye[j], 'ZERO')
        K[i, j] = K[j, i] = 0.5 * (val - Q_i[i] - Q_i[j])
for k in KEYS:
    for i in range(nf):
        F[k][i] = 0.5 * (energy_q(eye[i], k) - c0[k] - Q_i[i])  # linear term

# constraints: <w_i> = 0 and <omega1> = 0  -> zero the constant coefficients
con = [sl[0].start, sl[1].start, sl[2].start, sl[3].start]
Cc = np.zeros((len(con), nf))
for r, idx in enumerate(con):
    Cc[r, idx] = 1.0
A = np.zeros((nf + len(con), nf + len(con)))
A[:nf, :nf] = 2 * K; A[:nf, nf:] = Cc.T; A[nf:, :nf] = Cc

Cd = np.zeros(6)
for a, k in enumerate(KEYS):
    rhs = np.zeros(nf + len(con)); rhs[:nf] = -2 * F[k]
    q = np.linalg.solve(A, rhs)[:nf]
    Cd[a] = (c0[k] + 2 * F[k] @ q + q @ K @ q) / A_cell

print("\ncircle iso, order %s" % GBAR_ORDER)
print("%-6s %15s %15s %15s" % ("mode", "5-dof eliminated", "6-dof multiplier",
                               "analytical"))
Bnd = build_solid_bundle("circle_iso_shell.yaml", cell_area=A_cell)
C6d = np.asarray(Bnd["C3D"])
ana = [2 * E * t / R, 0, 0, 0, 0, 0]
for a, k in enumerate(KEYS):
    print("%-6s %15.6e %15.6e %15.6e" % (k, Cd[a], C6d[a, a], ana[a]))
