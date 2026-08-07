"""Estimate the sheet thickness of the Sample_2 solid TPMS mesh:
t = material volume / midsurface area, with the midsurface area taken as
half the free boundary area of the tet mesh (thin-sheet geometry).
Run (from this folder):  python est_thickness_sample2.py"""
from collections import Counter

import numpy as np

toks = [ln.split() for ln in open("Sample_2.sc") if ln.strip()]
k = 2
nn, ne = int(toks[k][1]), int(toks[k][2])
nd = np.array([[float(v) for v in toks[k+1+i][1:4]] for i in range(nn)])
el = np.array([[int(v) for v in toks[k+1+nn+i][2:6]] for i in range(ne)]) - 1

p = nd[el]
V = np.abs(np.einsum('ij,ij->i', np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0]),
                     p[:, 3]-p[:, 0])).sum()/6.0
faces = Counter()
F = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
for e in el:
    for a, b, c in F:
        faces[tuple(sorted((e[a], e[b], e[c])))] += 1
bnd = np.array([f for f, n in faces.items() if n == 1])
q = nd[bnd]
S = 0.5*np.linalg.norm(np.cross(q[:, 1]-q[:, 0], q[:, 2]-q[:, 0]),
                       axis=1).sum()
# exclude the cell-boundary cut strips: faces lying ON the unit-cube faces
on_cube = np.zeros(len(bnd), bool)
for ax in range(3):
    for v in (nd[:, ax].min(), nd[:, ax].max()):
        on_cube |= np.all(np.abs(nd[bnd][:, :, ax] - v) < 1e-9, axis=1)
qf = nd[bnd[~on_cube]]
S_free = 0.5*np.linalg.norm(np.cross(qf[:, 1]-qf[:, 0], qf[:, 2]-qf[:, 0]),
                            axis=1).sum()
t = 2.0*V/S_free
print("V = %.6f   S_bnd = %.4f  S_free(faces) = %.4f" % (V, S, S_free))
print("A_mid ~ S_free/2 = %.4f   ->  t = 2V/S_free = %.6f" % (S_free/2, t))
