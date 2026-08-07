"""
wf_ode_d44.py -- exact 1-D relaxation model of the periodic iso "cross" cell
(one horizontal + one vertical wall through an L x L cell, wall thickness t)
under macro in-plane shear 2*Gbar23 = g, built from the CODE'S row definitions
(src/opensg_shell/solid_props.py):

  Gamma_e 2g23 row (l.173-175): col-4 (2G23) coeff = (X32*C32 + X22*C33)
      horizontal wall  a2=(0,1,0), n=(0,0,1): X22=1,X32=0,C32=0,C33=1 -> +1
      vertical  wall   a2=(0,0,1), n=(0,-1,0): X22=0,X32=1,C32=-1,C33=0 -> -1
      => drive = d_w * g with d_w = +1 (H), -1 (V).  FULL Bond pair sum.
  Gamma_h 2g23 row (l.105-106):  C_3i w_i,2 - X_i1 om_i = d(n.u)/ds - om1
  Gamma_h K22  row (l.98):      -X_i1 om_i,2            = -d(om1)/ds

  NOTE (l.150 comment vs entries): substituting w -> G.y into the Gamma_h
  shear row gives n'Ga2 = HALF the pair sum (mid-surface slope part g/2).
  The other g/2 is the affine normal-fiber TILT, i.e. an affine rotation
  om_aff = -(g/2)*d_w -- DIFFERENT per wall.  The code uses the full pair sum
  algebraically while the fluctuation rotation nodes are single-valued at
  junctions, so the fluctuation cannot supply the inter-wall tilt jump.

Wall model (arc s in [0,L] per wall):
  e_shear(s) = a*g*d_w + b*(dw_n/ds - om1)        (hypotheses on a,b)
  U_wall = int (1/2)*Gt*e_shear^2 + (1/2)*EI*(om1')^2 ds
Three junction/periodicity formulations:
  P  (code space)   : w_n periodic fluctuation w(0)=w(L); om1(0)=om1(L)=thJ
                      shared single-valued joint rotation.  == what the FE
                      assembly actually solves.
  P2 (jump-allowed) : same as P but om1 end values = thJ - om_aff
                      = thJ + (g/2)*d_w  (fluct. rotation carries the affine
                      tilt jump).  Tests the root-cause hypothesis.
  T  (task-stated)  : w_n TOTAL, sway = n.[u_aff(L)-u_aff(0)],
                      u_aff = Gbar.y + rho*e1 x y  (rho free lattice rotation)
                      H: w_n=+u3 -> sway = (g/2+rho)L
                      V: w_n=-u2 -> sway = -(g/2-rho)L = (rho-g/2)L
                      om ends = thJ; gauge thJ=0 (exact zero mode: om+=d,
                      w+=d*s, rho+=d, thJ+=d is energy+constraint neutral).
FE: 2-node linear Timoshenko elements, midpoint (reduced) shear integration,
exact bending integration -- locking-free.
"""
import numpy as np

E, nu, L, tL = 70e9, 0.3, 1.0, 0.0125
t  = tL * L
Ep = E / (1 - nu**2)                    # E' plane-strain-ish wall modulus
G  = E / (2*(1 + nu))
EI = Ep * t**3 / 12.0
Gt = G * t                              # kappa = 1 (thin-limit insensitive)
g  = 1.0
D44_exact = 6.0 * EI / L                # = 0.5*E'*t^3/L = C44*L^2 (established)

NEL = 200
h   = L / NEL
NN  = NEL + 1

# dof layout: [wH(0..N) | omH(0..N) | wV(0..N) | omV(0..N) | thJ | rho]
iwH = np.arange(0, NN);        ioH = np.arange(NN, 2*NN)
iwV = np.arange(2*NN, 3*NN);   ioV = np.arange(3*NN, 4*NN)
ithJ, irho = 4*NN, 4*NN + 1
ndof = 4*NN + 2


def assemble(a, b):
    """Energy = 1/2 x'Kx + g f'x + 1/2 c g^2 over both walls."""
    K = np.zeros((ndof, ndof)); f = np.zeros(ndof); c = 0.0
    r = b * np.array([-1/h, 1/h, -0.5, -0.5])      # shear op at midpoint
    q = np.array([-1/h, 1/h])                      # curvature op
    for iw, io, d in ((iwH, ioH, +1.0), (iwV, ioV, -1.0)):
        for e in range(NEL):
            idx = [iw[e], iw[e+1], io[e], io[e+1]]
            K[np.ix_(idx, idx)] += Gt * h * np.outer(r, r)
            f[np.array(idx)]    += Gt * h * a * d * r
            c                   += Gt * h * a * a
            K[np.ix_(idx[2:], idx[2:])] += EI * h * np.outer(q, q)
    return K, f, c


def solve(K, f, c, C, dvec):
    """min 1/2 x'Kx + g f'x + 1/2 c g^2  s.t.  C x = dvec*g   (g=1)."""
    m = C.shape[0]
    A = np.block([[K, C.T], [C, np.zeros((m, m))]])
    sol = np.linalg.solve(A, np.concatenate([-f*g, dvec*g]))
    x = sol[:ndof]
    return 0.5*x@K@x + g*f@x + 0.5*c*g*g


def rows(spec):
    C = np.zeros((len(spec), ndof)); dv = np.zeros(len(spec))
    for k, (cols, vals, rhs) in enumerate(spec):
        C[k, cols] = vals; dv[k] = rhs
    return C, dv


om_ends = [([ioH[0], ithJ], [1, -1]), ([ioH[-1], ithJ], [1, -1]),
           ([ioV[0], ithJ], [1, -1]), ([ioV[-1], ithJ], [1, -1])]
# P: periodic fluctuation, single-valued joint rotation
CP, dP = rows([([iwH[0]], [1], 0), ([iwH[-1]], [1], 0),
               ([iwV[0]], [1], 0), ([iwV[-1]], [1], 0)]
              + [(c_, v_, 0.0) for c_, v_ in om_ends]
              + [([irho], [1], 0)])
# P2: same but om_fluct ends = thJ + (g/2)*d_w  (affine tilt jump allowed)
CP2, dP2 = rows([([iwH[0]], [1], 0), ([iwH[-1]], [1], 0),
                 ([iwV[0]], [1], 0), ([iwV[-1]], [1], 0)]
                + [(om_ends[0][0], om_ends[0][1], +0.5),
                   (om_ends[1][0], om_ends[1][1], +0.5),
                   (om_ends[2][0], om_ends[2][1], -0.5),
                   (om_ends[3][0], om_ends[3][1], -0.5)]
                + [([irho], [1], 0)])
# T: total displacement with affine junction sway, rho free, gauge thJ=0
CT, dT = rows([([iwH[0]], [1], 0), ([iwH[-1], irho], [1, -L], +0.5*L),
               ([iwV[0]], [1], 0), ([iwV[-1], irho], [1, -L], -0.5*L)]
              + [(c_, v_, 0.0) for c_, v_ in om_ends]
              + [([ithJ], [1], 0)])


def analytic_P(a, b):
    """closed form (formulation P): e_shear const/wall, om quadratic, thJ=0:
       D44 = 2*Gt*L*a^2 / (1 + Gt*b^2*L^2/(12*EI));
       thin limit -> 24*EI*(a/b)^2/L = 4*(a/b)^2 * D44_exact."""
    return 2*Gt*L*a*a / (1 + Gt*b*b*L*L/(12*EI))


print(f"L={L} t/L={tL} E'={Ep:.4e} EI={EI:.4e} Gt={Gt:.4e} NEL={NEL}/wall")
print(f"D44_exact = 6*EI/L = {D44_exact:.6e}")
print()
print("ratio D44/D44_exact  (P=code space | P2=tilt-jump | T=task sway)")
print(f"{'(a,b)':>9} | {'P (FE)':>9} {'P analytic':>10} | {'P2 (FE)':>9} | "
      f"{'T (FE)':>9}")
print("-" * 64)
cases = [((1.0, 1.0), "current"), ((0.5, 1.0), "empirical fix"),
         ((1.0, 2.0), "Gh doubled"), ((0.5, 2.0), "both"),
         ((0.0, 1.0), "no drive")]
for (a, b), tag in cases:
    K, f, c = assemble(a, b)
    rP  = 2*solve(K, f, c, CP,  dP)  / g**2 / D44_exact
    rP2 = 2*solve(K, f, c, CP2, dP2) / g**2 / D44_exact
    rT  = 2*solve(K, f, c, CT,  dT)  / g**2 / D44_exact
    rA  = analytic_P(a, b) / D44_exact
    print(f"({a:3.1f},{b:3.1f}) | {rP:9.5f} {rA:10.5f} | {rP2:9.5f} | "
          f"{rT:9.5f}   {tag}")

print()
print("thin-limit closed forms: P: 4(a/b)^2 | P2: 4(a/b-1/2)^2 | "
      "T: 4(a/b+1/2)^2")
