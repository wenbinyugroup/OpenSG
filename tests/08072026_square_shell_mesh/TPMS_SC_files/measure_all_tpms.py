"""Measure sheet thickness of every TPMS solid mesh from the mesh itself:
    A_mid ~ S_free/2 ,  t = 2V/S_free
for Sample_1.sc, Sample_2.sc (SwiftComp tets) and preovios_try_solid.msh
(gmsh).  Also reports the shell mesh area for reference.

Run (from this folder):  python measure_all_tpms.py"""
from collections import Counter

import numpy as np

F = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]


def metrics(nd, el, name):
    p = nd[el]
    V = np.abs(np.einsum('ij,ij->i',
                         np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0]),
                         p[:, 3]-p[:, 0])).sum()/6.0
    faces = Counter()
    for e in el:
        for a, b, c in F:
            faces[tuple(sorted((e[a], e[b], e[c])))] += 1
    bnd = np.array([f for f, n in faces.items() if n == 1])
    on_cube = np.zeros(len(bnd), bool)
    for ax in range(3):
        for v in (nd[:, ax].min(), nd[:, ax].max()):
            on_cube |= np.all(np.abs(nd[bnd][:, :, ax] - v) < 1e-9, axis=1)
    q = nd[bnd[~on_cube]]
    S = 0.5*np.linalg.norm(np.cross(q[:, 1]-q[:, 0], q[:, 2]-q[:, 0]),
                           axis=1).sum()
    bb = nd.max(0) - nd.min(0)
    print("%-22s %7d nodes %8d tets  bbox %s" % (name, len(nd), len(el),
                                                 np.round(bb, 3)))
    print("%-22s V=%.6f  S_free=%.4f  A_mid=%.4f  t=2V/S_free=%.6f  "
          "rho=%.4f" % ("", V, S, S/2, 2*V/S, V/np.prod(bb)))
    return V, S/2, 2*V/S


def read_sc(path):
    toks = [ln.split() for ln in open(path) if ln.strip()]
    k = 2
    nn, ne = int(toks[k][1]), int(toks[k][2])
    nd = np.array([[float(v) for v in toks[k+1+i][1:4]] for i in range(nn)])
    el = np.array([[int(v) for v in toks[k+1+nn+i][2:6]]
                   for i in range(ne)]) - 1
    return nd, el


def read_msh_tets(path):
    ln = open(path).read().split("\n")
    i = ln.index("$Nodes")
    nn = int(ln[i+1])
    nd = np.array([[float(v) for v in ln[i+2+k].split()[1:4]]
                   for k in range(nn)])
    j = ln.index("$Elements")
    ne = int(ln[j+1])
    tet, other = [], Counter()
    for k in range(ne):
        p = ln[j+2+k].split()
        etype, ntag = int(p[1]), int(p[2])
        other[etype] += 1
        if etype == 4:                      # 4-node tet
            tet.append([int(v) for v in p[3+ntag:3+ntag+4]])
    print("  gmsh element types present:", dict(other))
    return nd, np.array(tet, int) - 1


A_SHELL = 2.3533          # schwarz_p_D2_shell.msh midsurface area

print("=== TPMS solid meshes ===")
for f, nm in (("Sample_1.sc", "Sample_1"), ("Sample_2.sc", "Sample_2")):
    V, Am, t = metrics(*read_sc(f), nm)
    print("%-22s t from the TRUE midsurface (V/A_shell) = %.6f"
          % ("", V/A_SHELL))
nd, el = read_msh_tets("preovios_try_solid.msh")
if len(el):
    metrics(nd, el, "preovios_try_solid")
else:
    print("preovios_try_solid: no 4-node tets found")

print("\n=== shell surface mesh (reference) ===")
ln = open("schwarz_p_D2_shell.msh").read().split("\n")
i = ln.index("$Nodes")
nn = int(ln[i+1])
snd = np.array([[float(v) for v in ln[i+2+k].split()[1:4]]
                for k in range(nn)])
j = ln.index("$Elements")
ne = int(ln[j+1])
tri = []
for k in range(ne):
    p = ln[j+2+k].split()
    ntag = int(p[2])
    tri.append([int(v) for v in p[3+ntag:]])
tri = np.array(tri, int) - 1
p1, p2, p3 = snd[tri[:, 0]], snd[tri[:, 1]], snd[tri[:, 2]]
A = 0.5*np.linalg.norm(np.cross(p2-p1, p3-p1), axis=1).sum()
print("schwarz_p_D2_shell     %7d nodes %8d tris  area=%.4f"
      % (len(snd), len(tri), A))
print("  (exact Schwarz-P minimal surface area in a unit cube = 2.3451)")
