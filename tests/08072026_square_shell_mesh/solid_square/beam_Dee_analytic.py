"""Analytical Dee for the msg_shell EB BEAM block (4x4) over the four edges of
the isotropic square tube, vs the numerical shell assembly and the solid.

Beam macro columns [eps11, kap1, kap2, kap3].  Per wall at angle phi, position
(y2, y3), the shell Gamma_e-beam rows reduce (prismatic identities) to

  eps11 row : [1, 0, y3, -y2]                      -> A11
  2eps12 row: [0, Rn2, 0, 0],  Rn2 = y2 X32 - y3 X22 = -d  (wall offset)
                                                    -> A66  (torsion shear flow)
  K11 row   : [0, 0, X22, X32]                      -> D11d (wall bending)
  K12+K21   : [0, -1, 0, 0]                         -> D66d (wall twist)
  2g13 row  : [0, swept, 0, 0], swept = y2 C33 - y3 C32 = u (along-wall coord)
                                                    -> Gm   (transverse shear)

Closed forms, square side a (walls at distance d = a/2, u in [-a/2, a/2]):

  D(1,1) = 4a A11
  D(2,2) = a^3 A66 + (a^3/3) Gm + 4a D66d
  D(3,3) = D(4,4) = (2/3) a^3 A11 + 2a D11d
  all off-diagonals = 0 by the double symmetry of the square

Solid (exact area integrals of the mitred annulus, outer a+t / inner a-t):
  A_mat = 4at,  I = ((a+t)^4 - (a-t)^4)/12,  J2 = 2I
  D(1,1) = (lam+2mu) A_mat ;  D(2,2) = mu J2 ;  D(3,3) = D(4,4) = (lam+2mu) I

Run (from this folder):  python beam_Dee_analytic.py
"""
import numpy as np
import yaml as _yaml

from opensg_shell.sg_mesh import load_ring_ref
from opensg_shell.sg_assembly import assemble_segment_indep
from opensg_shell.sg_materials import material_db_from_yaml
from opensg_solid.rm_plate_1D.msg_rm_plate import rm_plate_msg

############### User Input #################################
a, t = 1.0, 0.03
E, nu = 70.0e9, 0.30
SHELL_YAML = "square_tube_1Dshell_iso.yaml"
############################################################

G = E/(2*(1+nu))
Ep = E/(1-nu**2)
A11, A66 = Ep*t, G*t
D11d, D66d = Ep*t**3/12, G*t**3/12
mat = {"name": "iso", "density": 2700.0,
       "elastic": {"E": [E]*3, "G": [G]*3, "nu": [nu]*3}}
Gm = float(np.asarray(rm_plate_msg([t], [0.0], ["iso"],
                                   material_db_from_yaml([mat]),
                                   fraction=0.5)["G_msg"])[0, 0])

Dee_an = np.zeros((4, 4))
Dee_an[0, 0] = 4*a*A11
Dee_an[1, 1] = a**3*A66 + (a**3/3)*Gm + 4*a*D66d
Dee_an[2, 2] = (2.0/3.0)*a**3*A11 + 2*a*D11d
Dee_an[3, 3] = Dee_an[2, 2]

# ---- numerical shell beam Dee ----------------------------------------------
R = load_ring_ref(SHELL_YAML, "center")
d_sh = _yaml.safe_load(open(SHELL_YAML))
G_by = list(R["G_by"])
G_by[0] = np.asarray(rm_plate_msg([t], [0.0], ["iso"],
                                  material_db_from_yaml(d_sh["materials"]),
                                  fraction=0.5)["G_msg"])
rx, rcells = R["rx"], R["cells"]
m = len(rx)
h_st = float(np.mean(np.linalg.norm(rx[rcells[:, 1]] - rx[rcells[:, 0]], axis=1)))
ez = np.zeros(3); ez[R["ax"]] = 1.0
nodes_st = np.vstack([rx, rx + h_st*ez])
dof_map = np.concatenate([np.arange(m), np.arange(m)])
quads = np.array([[q1, q2, m+q2, m+q1] for q1, q2 in rcells], int)
_, _, Dee_sh, _, _, _ = assemble_segment_indep(
    nodes_st, quads, R["rsub"], np.asarray(R["re3"]), R["D_by"], G_by,
    np.asarray(R["k22"]), R["cross"], R["ax"], kg_e=None,
    dof_map=dof_map, shear="mitc4_g23")
Dee_sh = np.asarray(Dee_sh)/h_st

# ---- solid beam Dee (exact annulus integrals) ------------------------------
lam = E*nu/((1+nu)*(1-2*nu)); mu = G
A_mat = (a+t)**2 - (a-t)**2
I_sec = ((a+t)**4 - (a-t)**4)/12.0
Dee_so = np.diag([(lam+2*mu)*A_mat, mu*2*I_sec,
                  (lam+2*mu)*I_sec, (lam+2*mu)*I_sec])

LBL = ["eps11", "kap1", "kap2", "kap3"]
print("iso square tube a=%.2f t=%.3f;  A11=%.5e A66=%.5e Gm=%.5e"
      " D11=%.5e D66=%.5e" % (a, t, A11, A66, Gm, D11d, D66d))
print("A_mat=%.6f  I=%.8f\n" % (A_mat, I_sec))
print("  %-12s %14s %14s %9s %14s %9s"
      % ("term", "analytical", "shell num", "an/sh", "solid", "an/so"))
for i in range(4):
    an, sh, so = Dee_an[i, i], Dee_sh[i, i], Dee_so[i, i]
    print("  D_%-9s %14.6e %14.6e %9.6f %14.6e %9.4f"
          % (LBL[i], an, sh, an/sh, so, an/so))
off = max(abs(Dee_sh[i, j]) for i in range(4) for j in range(4) if i != j)
print("max |off-diagonal| shell num = %.3e  (analytical: 0 by symmetry)" % off)
