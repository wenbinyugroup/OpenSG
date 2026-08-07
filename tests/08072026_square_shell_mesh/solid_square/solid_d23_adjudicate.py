"""Adjudicate the m45 ring D23 dispute: independent periodic CST solve on the
ACTUAL PreVABS mesh (square_tube_2Dsolid.yaml, fiber frames BAKED into
elementOrientations -- no elem_rotation path, no ply-angle path).

  candidate 1 (plate_homo_2d benchmark) : D23 = 1.328e8
  candidate 2 (ply-resolved microcell)  : D23 = 3.3e7

Same mesh + same material as candidate 1, same solver class as candidate 2.

Run (from this folder):  python solid_d23_adjudicate.py
"""
import numpy as np
import yaml as _yaml

from opensg_shell.solid_props import _C_from_eng
from opensg_shell.junction_micro import _voigt_rotate
from opensg_shell.periodic_multiscale import mesh_to_periodic_sparse_assembly_map

############### User Input #################################
YAML = "../square_tube_2Dsolid.yaml"     # fiber-baked orientations
E1, E2, G12, G23n, nu12, nu23 = 142.0e9, 9.8e9, 6.0e9, 4.8e9, 0.30, 0.42
############################################################

d = _yaml.safe_load(open(YAML))
row = lambda r: [float(v) for v in
                 " ".join(str(x) for x in (r if isinstance(r, list) else [r])).split()]
nd = np.array([row(r)[:2] for r in d["nodes"]], float)
tri = np.array([[int(v) for v in row(r)] for r in d["elements"]], int) - 1
ori = np.array(d["elementOrientations"], float)
nn, ne = len(nd), len(tri)
print("mesh: %d nodes, %d tris" % (nn, ne))

Cply = _C_from_eng([E1, E2, E2], [G12, G12, G23n], [nu12, nu12, nu23])
cyc = [2, 0, 1]                          # yaml (y2,y3,axial) -> (axial,y2,y3)
Ce = np.empty((ne, 6, 6))
for e in range(ne):
    Q = np.stack([ori[e, 0:3][cyc], ori[e, 3:6][cyc], ori[e, 6:9][cyc]], 1)
    Ce[e] = _voigt_rotate(Cply, Q)

p1, p2, p3 = nd[tri[:, 0]], nd[tri[:, 1]], nd[tri[:, 2]]
det = ((p2[:, 0]-p1[:, 0])*(p3[:, 1]-p1[:, 1])
       - (p3[:, 0]-p1[:, 0])*(p2[:, 1]-p1[:, 1]))
area = 0.5*np.abs(det)
bb = np.stack([p2[:, 1]-p3[:, 1], p3[:, 1]-p1[:, 1], p1[:, 1]-p2[:, 1]],
              1)/det[:, None]
cc = np.stack([p3[:, 0]-p2[:, 0], p1[:, 0]-p3[:, 0], p2[:, 0]-p1[:, 0]],
              1)/det[:, None]
B = np.zeros((ne, 6, 9))
for k in range(3):
    B[:, 1, 3*k+1] = bb[:, k]
    B[:, 2, 3*k+2] = cc[:, k]
    B[:, 3, 3*k+1] = cc[:, k]; B[:, 3, 3*k+2] = bb[:, k]
    B[:, 4, 3*k] = cc[:, k]
    B[:, 5, 3*k] = bb[:, k]
Ke = np.einsum('e,eia,eij,ejb->eab', area, B, Ce, B)
Fe = np.einsum('e,eia,eij->eaj', area, B, Ce)
Dee = np.einsum('e,eij->ij', area, Ce)
ndof = 3*nn
gd = (3*tri[:, :, None] + np.arange(3)[None, None, :]).reshape(ne, 9)
K = np.zeros((ndof, ndof))
Dhe = np.zeros((ndof, 6))
np.add.at(K, (gd[:, :, None].repeat(9, 2), gd[:, None, :].repeat(9, 1)), Ke)
np.add.at(Dhe, gd.ravel(), Fe.reshape(-1, 6))
rc, _ = mesh_to_periodic_sparse_assembly_map(nn, np.arange(nn)[:, None], nd,
                                             3, 3)
master = np.asarray(rc, int).ravel()
uniq, inv = np.unique(master, return_inverse=True)
npair = nn - len(uniq)
print("periodic pairs tied: %d nodes" % npair)
nred = 3*len(uniq)
T = np.zeros((ndof, nred))
for n_ in range(nn):
    for k in range(3):
        T[3*n_+k, 3*inv[n_]+k] = 1.0
wA = np.zeros(nn)
np.add.at(wA, tri.ravel(), np.repeat(area/3.0, 3))
wAr = np.zeros(len(uniq))
np.add.at(wAr, inv, wA)
Cc = np.zeros((3, nred))
Cc[0, 0::3] = wAr; Cc[1, 1::3] = wAr; Cc[2, 2::3] = wAr
A_ = np.zeros((nred+3, nred+3))
A_[:nred, :nred] = T.T @ K @ T
A_[:nred, nred:] = Cc.T
A_[nred:, :nred] = Cc
Rr = np.zeros((nred+3, 6))
Rr[:nred] = -(T.T @ Dhe)
V0 = T @ np.linalg.lstsq(A_, Rr, rcond=None)[0][:nred]
Deff = Dee + V0.T @ Dhe
Deff = 0.5*(Deff + Deff.T)

print("independent periodic CST on the PreVABS mesh, D_eff (order"
      " [e11 e22 e33 2e23 2e13 2e12]):")
for i in range(6):
    print("  " + " ".join("%13.5e" % Deff[i, j] for j in range(6)))
print("\nD23 = %.5e   (benchmark plate_homo_2d: 1.328e8;"
      " microcell full-cell: 3.3e7)" % Deff[1, 2])
print("D22 = %.5e   D33 = %.5e   D44 = %.5e"
      % (Deff[1, 1], Deff[2, 2], Deff[3, 3]))
