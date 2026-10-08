"""build_gate.py -- pin the SwiftComp tet10 slot layout (and the aux-pair
`T rho` order) with a LIVE SwiftComp run, against OpenSG and closed form.

Builds two tet10 cubes (unit cube, 2x2x2 hex cells, Kuhn 6-tet split,
midside nodes on the half-lattice):

  iso      one material everywhere      -> exact C is the material's own;
                                           the `.sc.k` effective density
                                           must read back rho = 2700
  bilayer  z < 0.5 material 1, z >= 0.5 material 2 -> compare SwiftComp
                                           with OpenSG's own tet10 solve

and for each writes three candidate `.sc` decks that differ ONLY in the
element record:

  A  corners 1-4, slot 5 = 0, midsides 6-11 on edges (12,23,13,14,24,34)
     -- the layout read_sc measured on BCC/CoreUC.sc + Plate_coarse.sc
  B  same slots, midsides in GMSH order (12,23,13,14,34,24)
  C  contiguous fill 1-10 (corners then Abaqus-order midsides), slot 5 /= 0

Run here (server, opensg env):  python build_gate.py
Then on the laptop:  for d in *.sc; do SwiftComp $d 3D H; done
Then here:           python compare_gate.py
"""
import os
import time

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
N = 2                                           # hex cells per direction
E1, NU1, RHO1 = 70000.0, 0.30, 2700.0           # Al-like  [MPa, -, kg/m3]
E2, NU2, RHO2 = 7000.0, 0.35, 1200.0            # soft layer

print("start", time.strftime("%Y-%m-%d %H:%M:%S"))

# --------------------------------------------------- lattice + tet4 corners
M = 2 * N + 1                                   # half-lattice points/dir
h = 1.0 / (2 * N)


def lid(i, j, k):
    return (i * M + j) * M + k


nodes = np.array([[i * h, j * h, k * h]
                  for i in range(M) for j in range(M) for k in range(M)])

# Kuhn split of a cube into 6 tets, corner (0..1)^3 offsets, all positive
# volume (checked below)
KUHN = [((0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)),
        ((0, 0, 0), (1, 1, 0), (0, 1, 0), (1, 1, 1)),
        ((0, 0, 0), (0, 1, 0), (0, 1, 1), (1, 1, 1)),
        ((0, 0, 0), (0, 1, 1), (0, 0, 1), (1, 1, 1)),
        ((0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1)),
        ((0, 0, 0), (1, 0, 1), (1, 0, 0), (1, 1, 1))]

tets = []                                       # 4 corner lattice triples
for ci in range(N):
    for cj in range(N):
        for ck in range(N):
            for t in KUHN:
                tets.append([(2 * (ci + a), 2 * (cj + b), 2 * (ck + c))
                             for a, b, c in t])


def vol(c):
    p = nodes[[lid(*q) for q in c]]
    return np.linalg.det(np.array([p[1] - p[0], p[2] - p[0],
                                   p[3] - p[0]])) / 6.0


vols = np.array([vol(c) for c in tets])
assert (vols > 0).all(), vols.min()
assert abs(vols.sum() - 1.0) < 1e-12, vols.sum()

# edge lists (0-based corner positions) in the two candidate orders
ABQ = [(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)]    # 12 23 13 14 24 34
GMSH = [(0, 1), (1, 2), (0, 2), (0, 3), (2, 3), (1, 3)]   # 12 23 13 14 34 24


def mid(c, e):
    a, b = c[e[0]], c[e[1]]
    return ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2, (a[2] + b[2]) // 2)


tet10_gmsh = [[lid(*q) for q in c] + [lid(*mid(c, e)) for e in GMSH]
              for c in tets]
tet10_abq = [[lid(*q) for q in c] + [lid(*mid(c, e)) for e in ABQ]
             for c in tets]
cent_z = np.array([nodes[[lid(*q) for q in c]][:, 2].mean() for c in tets])

mats = {"iso": {1: (E1, NU1, RHO1)},
        "bilayer": {1: (E1, NU1, RHO1), 2: (E2, NU2, RHO2)}}
mat_id = {"iso": np.ones(len(tets), int),
          "bilayer": np.where(cent_z < 0.5, 1, 2)}
assert (mat_id["bilayer"] == 1).sum() == (mat_id["bilayer"] == 2).sum()


def fe(v):
    return "%.15e" % v


# ------------------------------------------------------------ OpenSG yaml
def write_yaml(case):
    m = {}
    for k, (E, nu, rho) in mats[case].items():
        G = E / (2 * (1 + nu))
        m[k] = {"type": 1, "density": rho,
                "engineering": [E, E, E, G, G, G, nu, nu, nu]}
    doc = {"n_model": 3, "refined": 0, "msg": "solid",
           "nodes": [[float(v) for v in x] for x in nodes],
           "cells": [[int(v) for v in c] for c in tet10_gmsh],
           "mat_id": [int(v) for v in mat_id[case]],
           "materials": m}
    p = os.path.join(HERE, "cube_%s.yaml" % case)
    with open(p, "w") as f:
        yaml.safe_dump(doc, f, default_flow_style=None, sort_keys=False)
    return p


# ------------------------------------------------------------ .sc decks
def write_sc(case, layout):
    p = os.path.join(HERE, "cube_%s_%s.sc" % (case, layout))
    with open(p, "w") as f:
        f.write("0\n")                          # submodel (3-D macro model)
        f.write("0 0 0 0\n\n")                  # analysis elem trans temp
        f.write("3 %d %d %d 0 0\n\n" % (len(nodes), len(tets),
                                        len(mats[case])))
        for i, x in enumerate(nodes):
            f.write("%d %s %s %s\n" % (i + 1, fe(x[0]), fe(x[1]), fe(x[2])))
        f.write("\n")
        for e in range(len(tets)):
            c4 = [v + 1 for v in tet10_abq[e][:4]]
            if layout == "A":
                row = c4 + [0] + [v + 1 for v in tet10_abq[e][4:]]
            elif layout == "B":
                row = c4 + [0] + [v + 1 for v in tet10_gmsh[e][4:]]
            else:
                row = c4 + [v + 1 for v in tet10_abq[e][4:]]
            row = row + [0] * (20 - len(row))
            f.write("%d %d %s\n" % (e + 1, mat_id[case][e],
                                    " ".join(str(v) for v in row)))
        f.write("\n")
        for k, (E, nu, rho) in sorted(mats[case].items()):
            f.write("%d 0 1\n" % k)
            f.write("%s %s\n" % (fe(0.0), fe(rho)))        # T rho
            f.write("%s %s\n\n" % (fe(E), fe(nu)))
        f.write("%s\n" % fe(1.0))                          # omega = volume
    return p


for case in ("iso", "bilayer"):
    y = write_yaml(case)
    for L in "ABC":
        write_sc(case, L)
    print("wrote", y, "+ decks A/B/C")

# ---------------------------------------------------- OpenSG reference
for case in ("iso", "bilayer"):
    y = os.path.join(HERE, "cube_%s.yaml" % case)
    t0 = time.strftime("%H:%M:%S")
    rc = os.system("cd %s && opensg_solid cube_%s.yaml > cube_%s.log 2>&1"
                   % (HERE, case, case))
    print("opensg_solid cube_%s.yaml rc=%d (%s -> %s)"
          % (case, rc, t0, time.strftime("%H:%M:%S")))

print("end", time.strftime("%Y-%m-%d %H:%M:%S"))
