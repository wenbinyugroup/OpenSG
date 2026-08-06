"""UD-composite 2-D SG (fiber+matrix square unit cell) -- MSG solid model=3.

Reproduces the ASC-23 OpenSG-Solid paper, Table II Model-3 (1536 linear triangles)
and the wenbinyugroup/OpenSG dev 3DModel.ipynb (FEniCS) result:

    E1 = 167.2553 GPa, E2 = E3 = 11.4136, G12 = G13 = 6.8017, G23 = 3.0955 GPa,
    nu12 = nu13 = 0.31178, nu23 = 0.57574

Formulation (identical to the FEniCS reference): generalized-plane-strain 2-D SG,
strains e = Eps_macro + eps(v), PERIODIC boundary conditions on the square cell
faces (right->left, top->bottom; the essential ingredient the free-SG cases lack),
rigid translations pinned, 6 unit macro strain solves,
D_eff = (Dee + V0^T Dhe)/omega, omega = cell area.

Materials (MPa, from the notebook / paper; 1 = fiber = axial uniform direction):
  matrix (phys 0):  E = 4760, nu = 0.37, isotropic
  fiber  (phys 1):  E1 = 276e3, E2 = E3 = 19.5e3, G12 = G13 = 70e3, G23 = 5735,
                    nu12 = nu13 = 0.28, nu23 = 0.70
Voigt order [e11 e22 e33 2e23 2e13 2e12].

Run (from this folder):  python udcomp2d_solid.py
"""
import numpy as np

############### User Input #################################
MSH = "UDcomp_2D.msh"
matrix = {"E": [4760.0]*3, "G": [4760.0/2/1.37]*3, "nu": [0.37]*3}
fiber = {"E": [276e3, 19.5e3, 19.5e3], "G": [70e3, 70e3, 5735.0],
         "nu": [0.28, 0.28, 0.70]}
############################################################


def C_from_eng(m):
    """6x6 stiffness from engineering constants; G order [G12,G13,G23],
    nu order [nu12,nu13,nu23]; Voigt [11,22,33,23,13,12]."""
    E1, E2, E3 = m["E"]; G12, G13, G23 = m["G"]; n12, n13, n23 = m["nu"]
    S = np.zeros((6, 6))
    S[0, 0], S[1, 1], S[2, 2] = 1/E1, 1/E2, 1/E3
    S[0, 1] = S[1, 0] = -n12/E1
    S[0, 2] = S[2, 0] = -n13/E1
    S[1, 2] = S[2, 1] = -n23/E2
    S[3, 3], S[4, 4], S[5, 5] = 1/G23, 1/G13, 1/G12
    return np.linalg.inv(S)


# ---------------- parse gmsh 2.2 --------------------------------------------
lines = open(MSH).read().split("\n")
i = lines.index("$Nodes"); nn = int(lines[i+1])
nd = np.array([[float(v) for v in lines[i+2+k].split()[1:3]] for k in range(nn)])
i = lines.index("$Elements"); ne_all = int(lines[i+1])
tris, phys = [], []
for k in range(ne_all):
    p = lines[i+2+k].split()
    if int(p[1]) == 2:                          # tri3
        phys.append(int(p[3]))
        tris.append([int(p[-3])-1, int(p[-2])-1, int(p[-1])-1])
tri = np.array(tris, int); phys = np.array(phys, int)
ne = len(tri)
print("mesh: %d nodes, %d tris (phys counts: %s)"
      % (nn, ne, {t: int((phys == t).sum()) for t in np.unique(phys)}))

Cmat = {0: C_from_eng(matrix), 1: C_from_eng(fiber)}
Ce = np.stack([Cmat[t] for t in phys])          # (ne,6,6)

# ---------------- CST assembly ----------------------------------------------
xy = nd
p1, p2, p3 = xy[tri[:, 0]], xy[tri[:, 1]], xy[tri[:, 2]]
det = (p2[:, 0]-p1[:, 0])*(p3[:, 1]-p1[:, 1]) - (p3[:, 0]-p1[:, 0])*(p2[:, 1]-p1[:, 1])
area = 0.5*np.abs(det)
omega = float(area.sum())
b = np.stack([p2[:, 1]-p3[:, 1], p3[:, 1]-p1[:, 1], p1[:, 1]-p2[:, 1]], 1)/det[:, None]
c = np.stack([p3[:, 0]-p2[:, 0], p1[:, 0]-p3[:, 0], p2[:, 0]-p1[:, 0]], 1)/det[:, None]
B = np.zeros((ne, 6, 9))
for a in range(3):
    B[:, 1, 3*a+1] = b[:, a]                    # e22 = w2,x
    B[:, 2, 3*a+2] = c[:, a]                    # e33 = w3,y
    B[:, 3, 3*a+1] = c[:, a]; B[:, 3, 3*a+2] = b[:, a]
    B[:, 4, 3*a] = c[:, a]                      # 2e13 = w1,y
    B[:, 5, 3*a] = b[:, a]                      # 2e12 = w1,x
Ke = np.einsum('e,eia,eij,ejb->eab', area, B, Ce, B)
Fe = np.einsum('e,eia,eij->eaj', area, B, Ce)
Dee = np.einsum('e,eij->ij', area, Ce)

ndof = 3*nn
gdof = (3*tri[:, :, None] + np.arange(3)[None, None, :]).reshape(ne, 9)
K = np.zeros((ndof, ndof)); Dhe = np.zeros((ndof, 6))
np.add.at(K, (gdof[:, :, None].repeat(9, 2), gdof[:, None, :].repeat(9, 1)), Ke)
np.add.at(Dhe, gdof.ravel(), Fe.reshape(-1, 6))

# ---------------- periodic master-slave map ---------------------------------
tol = 1e-6
xlo, xhi = nd[:, 0].min(), nd[:, 0].max()
ylo, yhi = nd[:, 1].min(), nd[:, 1].max()
master = np.arange(nn)


def pair(face_s, face_m, coord):
    """map each slave-face node to the master-face node with matching coord."""
    for s in face_s:
        d = np.abs(nd[face_m, coord] - nd[s, coord])
        m = face_m[np.argmin(d)]
        if d.min() > 1e-5:
            raise RuntimeError("no periodic partner for node %d (%.2e)" % (s, d.min()))
        master[s] = m


right = np.where(np.abs(nd[:, 0]-xhi) < tol)[0]
left = np.where(np.abs(nd[:, 0]-xlo) < tol)[0]
top = np.where(np.abs(nd[:, 1]-yhi) < tol)[0]
bot = np.where(np.abs(nd[:, 1]-ylo) < tol)[0]
pair(right, left, 1)                            # right -> left (match y)
pair(top, bot, 0)                               # top -> bottom (match x)
# corners: chain right-top corner -> left-top -> left-bottom
for _ in range(3):
    master = master[master]
print("periodic pairs: %d right, %d top; %d independent nodes"
      % (len(right), len(top), len(np.unique(master))))

# reduction operator: full dof -> reduced dof
uniq, inv = np.unique(master, return_inverse=True)
nred = 3*len(uniq)
T = np.zeros((ndof, nred))
for n in range(nn):
    for k in range(3):
        T[3*n+k, 3*inv[n]+k] = 1.0
Kr = T.T @ K @ T
Fr = T.T @ Dhe

# pin the three rigid translations (area-weighted null average)
wA = np.zeros(nn)
np.add.at(wA, tri.ravel(), np.repeat(area/3.0, 3))
wAr = np.zeros(len(uniq))
np.add.at(wAr, inv, wA)
Cc = np.zeros((3, nred))
Cc[0, 0::3] = wAr; Cc[1, 1::3] = wAr; Cc[2, 2::3] = wAr

A = np.zeros((nred+3, nred+3))
A[:nred, :nred] = Kr; A[:nred, nred:] = Cc.T; A[nred:, :nred] = Cc
Rhs = np.zeros((nred+3, 6)); Rhs[:nred] = -Fr
V0r = np.linalg.solve(A, Rhs)[:nred]
V0 = T @ V0r
Deff = (Dee + V0.T @ Dhe) / omega
Deff = 0.5*(Deff + Deff.T)

S = np.linalg.inv(Deff)
E1, E2, E3 = 1/S[0, 0], 1/S[1, 1], 1/S[2, 2]
G23, G13, G12 = 1/S[3, 3], 1/S[4, 4], 1/S[5, 5]
n12, n13, n23 = -S[0, 1]/S[0, 0], -S[0, 2]/S[0, 0], -S[1, 2]/S[1, 1]

print("\nomega (cell area) = %.6f" % omega)
print("D_eff (6x6, MPa), order [e11 e22 e33 2e23 2e13 2e12]:")
for i in range(6):
    print("  " + " ".join("%12.4e" % Deff[i, j] for j in range(6)))

ref = {"E1": 167.2553, "E2": 11.4136, "G12": 6.8017, "G23": 3.0955,
       "nu12": 0.31178, "nu23": 0.57574}
print("\n9 effective constants vs ASC-23 Table II Model-3 (FEniCS):")
print("  %-10s %12s %12s %9s" % ("const", "this code", "paper", "%diff"))
rows = [("E1 (GPa)", E1/1e3, ref["E1"]), ("E2 (GPa)", E2/1e3, ref["E2"]),
        ("E3 (GPa)", E3/1e3, ref["E2"]), ("G23 (GPa)", G23/1e3, ref["G23"]),
        ("G13 (GPa)", G13/1e3, ref["G12"]), ("G12 (GPa)", G12/1e3, ref["G12"]),
        ("nu12", n12, ref["nu12"]), ("nu13", n13, ref["nu12"]),
        ("nu23", n23, ref["nu23"])]
with open("udcomp2d_constants.dat", "w") as f:
    f.write("# UD composite 2-D SG, model=3; vs ASC-23 Table II Model-3\n")
    f.write("# %-10s %14s %14s %9s\n" % ("const", "this_code", "paper", "pct"))
    for nm, a, r in rows:
        pd = 100*(a-r)/r
        print("  %-10s %12.4f %12.4f %+9.3f" % (nm, a, r, pd))
        f.write("  %-10s %14.6f %14.6f %+9.4f\n" % (nm, a, r, pd))
np.savetxt("udcomp2d_Deff.out", Deff, fmt="%16.8e",
           header="D_eff (MPa), order [e11 e22 e33 2e23 2e13 2e12]; omega=%.8f"
                  % omega)
print("\nwrote udcomp2d_Deff.out, udcomp2d_constants.dat")
