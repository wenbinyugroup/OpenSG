"""3-D solid properties (6x6 C_eff + 9 elastic constants) of a 2-D SG.

UD fiber/matrix square unit cell (UDcomp_2D.yaml, 801 nodes / 1536 tris),
model = 3 (3-D elastic).  Generalized plane strain, PERIODIC boundary
conditions on the cell faces (right->left, top->bottom), rigid translations
pinned, 6 unit macro-strain solves, D_eff = (Dee + V0^T Dhe)/omega.

Reproduces the ASC-23 OpenSG-Solid paper Table II Model-3 and the
wenbinyugroup/OpenSG dev 3DModel.ipynb (FEniCS) 9 constants:
    E1 = 167.2553 GPa, E2 = E3 = 11.4136, G12 = G13 = 6.8017,
    G23 = 3.0955 GPa, nu12 = nu13 = 0.31178, nu23 = 0.57574

# ----------------------------------------------------------------------------
# ALL VARIABLES USED IN THIS SCRIPT
# ----------------------------------------------------------------------------
#   n_model, name       3 = 3D elastic; reads <name>.yaml
#   nd, tri, mat_id     nodes (x,y), tri3 connectivity, per-element material
#   Ce                  (ne,6,6) element stiffness, Voigt [11 22 33 23 13 12]
#   omega               SG area (= cell area, material fills the cell)
#   master, T           periodic master-slave node map and dof reduction
#   V0                  (ndof,6) fluctuation solutions for 6 unit strains
#   Deff                (6,6) effective solid stiffness, MPa
#   <name>_Deff.out, <name>_constants_vs_paper.dat, <name>_mesh.png  outputs
# ----------------------------------------------------------------------------
"""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_t0 = time.perf_counter()

############### User Input #################################
n_model = 3                    # 1: Beam; 2: Plate; 3: 3D elastic
name = "UDcomp_2D"             # reads <name>.yaml
paper = {"E1": 167.2553, "E2": 11.4136, "G12": 6.8017, "G23": 3.0955,
         "nu12": 0.31178, "nu23": 0.57574}      # ASC-23 Table II Model-3
############################################################


def C_from_eng(E, G, nu):
    """6x6 stiffness from engineering constants; G = [G12,G13,G23],
    nu = [nu12,nu13,nu23]; Voigt [11,22,33,23,13,12]."""
    S = np.zeros((6, 6))
    S[0, 0], S[1, 1], S[2, 2] = 1/E[0], 1/E[1], 1/E[2]
    S[0, 1] = S[1, 0] = -nu[0]/E[0]
    S[0, 2] = S[2, 0] = -nu[1]/E[0]
    S[1, 2] = S[2, 1] = -nu[2]/E[1]
    S[3, 3], S[4, 4], S[5, 5] = 1/G[2], 1/G[1], 1/G[0]
    return np.linalg.inv(S)


# ---------------- load yaml -------------------------------------------------
import yaml
d = yaml.safe_load(open(name + ".yaml"))
row = lambda r: [float(v) for v in
                 " ".join(str(x) for x in (r if isinstance(r, list) else [r])).split()]
nd = np.array([row(r) for r in d["nodes"]], float)[:, :2]
tri = np.array([[int(v) for v in row(r)] for r in d["elements"]], int) - 1
nn, ne = len(nd), len(tri)
mats = {m["name"]: C_from_eng(m["elastic"]["E"], m["elastic"]["G"],
                              m["elastic"]["nu"]) for m in d["materials"]}
mat_id = np.zeros(ne, int)
Ce = np.zeros((ne, 6, 6))
for k, s in enumerate(d["sets"]["element"]):
    idx = np.array(s["labels"], int) - 1
    mat_id[idx] = k
    Ce[idx] = mats[s["name"]]
print("mesh: %d nodes, %d tris  (%s)" % (nn, ne,
      ", ".join("%s %d" % (s["name"], len(s["labels"]))
                for s in d["sets"]["element"])))

# ---------------- CST assembly (strains [e11 e22 e33 2e23 2e13 2e12]) -------
p1, p2, p3 = nd[tri[:, 0]], nd[tri[:, 1]], nd[tri[:, 2]]
det = (p2[:, 0]-p1[:, 0])*(p3[:, 1]-p1[:, 1]) - (p3[:, 0]-p1[:, 0])*(p2[:, 1]-p1[:, 1])
area = 0.5*np.abs(det)
omega = float(area.sum())
b = np.stack([p2[:, 1]-p3[:, 1], p3[:, 1]-p1[:, 1], p1[:, 1]-p2[:, 1]], 1)/det[:, None]
c = np.stack([p3[:, 0]-p2[:, 0], p1[:, 0]-p3[:, 0], p2[:, 0]-p1[:, 0]], 1)/det[:, None]
B = np.zeros((ne, 6, 9))
for a in range(3):
    B[:, 1, 3*a+1] = b[:, a]                    # e22 = w2,x
    B[:, 2, 3*a+2] = c[:, a]                    # e33 = w3,y
    B[:, 3, 3*a+1] = c[:, a]; B[:, 3, 3*a+2] = b[:, a]     # 2e23
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
    for s in face_s:
        dd = np.abs(nd[face_m, coord] - nd[s, coord])
        if dd.min() > 1e-5:
            raise RuntimeError("no periodic partner for node %d" % s)
        master[s] = face_m[np.argmin(dd)]


pair(np.where(np.abs(nd[:, 0]-xhi) < tol)[0],
     np.where(np.abs(nd[:, 0]-xlo) < tol)[0], 1)   # right -> left (match y)
pair(np.where(np.abs(nd[:, 1]-yhi) < tol)[0],
     np.where(np.abs(nd[:, 1]-ylo) < tol)[0], 0)   # top -> bottom (match x)
for _ in range(3):                                  # corner chains
    master = master[master]

uniq, inv = np.unique(master, return_inverse=True)
nred = 3*len(uniq)
T = np.zeros((ndof, nred))
for n in range(nn):
    for k in range(3):
        T[3*n+k, 3*inv[n]+k] = 1.0
Kr = T.T @ K @ T
Fr = T.T @ Dhe

# pin the three rigid translations (area-weighted zero average)
wA = np.zeros(nn)
np.add.at(wA, tri.ravel(), np.repeat(area/3.0, 3))
wAr = np.zeros(len(uniq))
np.add.at(wAr, inv, wA)
Cc = np.zeros((3, nred))
Cc[0, 0::3] = wAr; Cc[1, 1::3] = wAr; Cc[2, 2::3] = wAr

A = np.zeros((nred+3, nred+3))
A[:nred, :nred] = Kr; A[:nred, nred:] = Cc.T; A[nred:, :nred] = Cc
Rhs = np.zeros((nred+3, 6)); Rhs[:nred] = -Fr
V0 = T @ np.linalg.solve(A, Rhs)[:nred]
Deff = (Dee + V0.T @ Dhe) / omega
Deff = 0.5*(Deff + Deff.T)

# ---------------- outputs ---------------------------------------------------
S = np.linalg.inv(Deff)
E1, E2, E3 = 1/S[0, 0], 1/S[1, 1], 1/S[2, 2]
G23, G13, G12 = 1/S[3, 3], 1/S[4, 4], 1/S[5, 5]
n12, n13, n23 = -S[0, 1]/S[0, 0], -S[0, 2]/S[0, 0], -S[1, 2]/S[1, 1]

print("\nomega (cell area) = %.6f" % omega)
print("D_eff (6x6, MPa), order [e11 e22 e33 2e23 2e13 2e12]:")
for i in range(6):
    print("  " + " ".join("%12.4e" % Deff[i, j] for j in range(6)))
np.savetxt(name + "_Deff.out", Deff, fmt="%16.8e",
           header="D_eff (MPa), order [e11 e22 e33 2e23 2e13 2e12]; omega=%.8f"
                  % omega)

rows = [("E1 (GPa)", E1/1e3, paper["E1"]), ("E2 (GPa)", E2/1e3, paper["E2"]),
        ("E3 (GPa)", E3/1e3, paper["E2"]), ("G23 (GPa)", G23/1e3, paper["G23"]),
        ("G13 (GPa)", G13/1e3, paper["G12"]), ("G12 (GPa)", G12/1e3, paper["G12"]),
        ("nu12", n12, paper["nu12"]), ("nu13", n13, paper["nu12"]),
        ("nu23", n23, paper["nu23"])]
print("\n9 effective constants vs ASC-23 Table II Model-3 (FEniCS):")
print("  %-10s %12s %12s %9s" % ("const", "this code", "paper", "%diff"))
with open(name + "_constants_vs_paper.dat", "w") as f:
    f.write("# UD composite 2-D SG, model=3 (periodic); vs ASC-23 Table II"
            " Model-3 / FEniCS 3DModel.ipynb\n")
    f.write("# %-10s %14s %14s %9s\n" % ("const", "this_code", "paper", "pct"))
    for nm, a, r in rows:
        pd = 100*(a-r)/r
        print("  %-10s %12.4f %12.4f %+9.3f" % (nm, a, r, pd))
        f.write("  %-10s %14.6f %14.6f %+9.4f\n" % (nm, a, r, pd))

fig, ax = plt.subplots(figsize=(5, 5))
ax.tripcolor(nd[:, 0], nd[:, 1], tri, facecolors=mat_id.astype(float),
             cmap="coolwarm", edgecolors="k", linewidth=0.15)
ax.set_aspect("equal"); ax.set_xlabel("y2"); ax.set_ylabel("y3")
fig.tight_layout(); fig.savefig(name + "_mesh.png", dpi=200)
print("\nHomogenization stored in %s_Deff.out (+ %s_constants_vs_paper.dat)"
      % (name, name))
print("Time taken: %.2f sec" % (time.perf_counter() - _t0))
