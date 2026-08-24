"""refine_sg_test.py -- is the path-1 in-plane amplitude deficit the
SG's ELEMENT TECHNOLOGY (bilinear Q4 locking in the bending-dominated
pressure cell solve), not the RM theory?

Test: split every SG quad n x n (n = 1, 2, 3 -- bilinear subdivision,
exact for these straight-sided quads), rerun the FULL refined
homogenization + dehom chains on each mesh, average the children of
each original top-row element (child p*n^2..(p+1)*n^2-1 of parent p),
and watch the pressure-chain in-plane amplitude converge.  The 3-D
reference is C3D20R (quadratic) on the SAME in-plane layout -- if the
amplitude marches from 0.77x toward 1x of the 3-D as n grows, the gap
is discretization (locking), not a missing term of the recovery.

In:  ../sg_2d/45pm_singleextrude_bd.yaml, ../dehomo/45pm_singleextrude_bd.ff,
     ../path_1/path_1.coords, ../../abaqus/station_path_stress/path_1_abaqus.dat
Out: ./sg_refined_n2.yaml / _n3.yaml, ./refine_test_path1.dat,
     ./path_1_refine_S11.png / _S22.png / _S12.png, the printed table
"""
import datetime
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                            # noqa: E402
import numpy as np                                         # noqa: E402
import yaml                                                # noqa: E402
from scipy.spatial import cKDTree                          # noqa: E402

from opensg_solid.cli import read_ff_state                 # noqa: E402
from opensg_solid.sg_dehom import (SGIDX, ORDER,           # noqa: E402
                                   dehom_fields, material_frame_fields)
from opensg_solid.sg_homo import plate_homo_2d             # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
YAML = os.path.join(HERE, "..", "sg_2d", "45pm_singleextrude_bd.yaml")
FF = os.path.join(HERE, "..", "dehomo", "45pm_singleextrude_bd.ff")
COORDS = os.path.join(HERE, "..", "path_1", "path_1.coords")
ABQ = os.path.join(HERE, "..", "..", "abaqus", "station_path_stress",
                   "path_1_abaqus.dat")
PIDX = [SGIDX[c] for c in ORDER]
JP = {"11": 0, "22": 1, "12": 3}       # printed-order in-plane slots

print("start : %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
d0 = yaml.safe_load(open(YAML))
nd0 = np.asarray(d0["nodes"], float)[:, :2]
cl0 = np.asarray(d0["cells"], int)
mid0 = [int(m) for m in d0["mat_id"]]
E0 = len(cl0)
cen0 = nd0[cl0].mean(axis=1)


def refine(n):
    """n x n bilinear split of every quad; nodes merged on rounded
    coordinates.  Children of parent p are cells p*n^2..(p+1)*n^2-1.
    Out: (nodes (V, 2), cells (E0*n^2, 4), mat_id list)."""
    key2id, nodes, cells, mats = {}, [], [], []

    def nid(p):
        k = (round(float(p[0]), 8), round(float(p[1]), 8))
        if k not in key2id:
            key2id[k] = len(nodes)
            nodes.append([float(p[0]), float(p[1])])
        return key2id[k]

    for e in range(E0):
        c = nd0[cl0[e]]                       # corners, cyclic
        # bilinear lattice (n+1)^2 on the parametric square
        g = np.linspace(0.0, 1.0, n + 1)
        P = {}
        for i in range(n + 1):
            for j in range(n + 1):
                u, v = g[i], g[j]
                p = ((1-u)*(1-v)*c[0] + u*(1-v)*c[1]
                     + u*v*c[2] + (1-u)*v*c[3])
                P[(i, j)] = nid(p)
        for j in range(n):
            for i in range(n):
                cells.append([P[(i, j)], P[(i+1, j)],
                              P[(i+1, j+1)], P[(i, j+1)]])
                mats.append(mid0[e])
    return np.asarray(nodes), np.asarray(cells, int), mats


def write_yaml(path, nodes, cells, mats):
    with open(path, "w") as f:
        f.write("n_model: 2\nrefined: 1\nmsg: solid\n")
        f.write("nodes:\n")
        for p in nodes:
            f.write("- [%.9f, %.9f, 0.0]\n" % (p[0], p[1]))
        f.write("cells:\n")
        for c in cells:
            f.write("- [%d, %d, %d, %d]\n" % tuple(c))
        f.write("mat_id: [%s]\n" % ", ".join(str(m) for m in mats))
        yaml.safe_dump({"materials": d0["materials"]}, f,
                       default_flow_style=None)


state = read_ff_state(FF)
C = np.loadtxt(COORDS)
dist, idx = cKDTree(cen0).query(C[:, 1:3])
if dist.max() >= 1e-4:
    raise SystemExit("coords do not sit on the original centroids")
A = np.loadtxt(ABQ)
abq = A[:, 4:10]
s = C[:, 0]
fl = lambda v: v - v.mean()                            # noqa: E731

kw_grad = dict(dE1=state["dE1"], dE2=state["dE2"],
               dE11=state["dE11"], dE12=state["dE12"],
               dE22=state["dE22"])
res, z6 = {}, np.zeros(6)
for n in (1, 2, 3):
    if n == 1:
        yml = YAML
    else:
        yml = os.path.join(HERE, "sg_refined_n%d.yaml" % n)
        nodes, cells, mats = refine(n)
        write_yaml(yml, nodes, cells, mats)
        cen_c = nodes[cells].mean(axis=1)
        par = cen_c.reshape(E0, n * n, 2).mean(axis=1)
        gate = np.abs(par - cen0).max()
        print("n=%d: %d nodes, %d cells; child-centroid gate %.1e %s"
              % (n, len(nodes), len(cells), gate,
                 "OK" if gate < 1e-7 else "FAIL"))
    r = plate_homo_2d(yml, refined=1, plot=False)
    eps = np.linalg.solve(np.asarray(r["C_eff"], float), state["FF"])
    Gf, Sf, _ = dehom_fields(r, eps, qt6=state["qt6"], **kw_grad)
    Gb, Sb, _ = dehom_fields(r, eps, **kw_grad)
    Gq, Sq, _ = dehom_fields(r, z6, qt6=state["qt6"])
    piece = {}
    for tag, (G_, S_) in (("full", (Gf, Sf)), ("beps", (Gb, Sb)),
                          ("q", (Gq, Sq))):
        _, Sm = material_frame_fields(G_, S_, r)
        Sm = np.asarray(Sm)[:, :, PIDX]
        el = Sm.mean(axis=1)                     # (E, 6) element means
        parent = el.reshape(E0, n * n, 6).mean(axis=1)
        piece[tag] = parent[idx]                 # along path 1
    res[n] = {"piece": piece,
              "D11": float(np.asarray(r["A6_ladder"])[3, 3]),
              "G11": (float(np.asarray(r["G_msg"])[0, 0])
                      if r.get("G_msg") is not None else np.nan)}
    print("n=%d done: D11 %.6e  G11 %.6e"
          % (n, res[n]["D11"], res[n]["G11"]))

print("\n---- path 1 convergence with SG h-refinement (element means"
      " on the ORIGINAL elements)")
print("%-3s %12s %10s | %s" % ("n", "D11", "G11", " | ".join(
    "S%s: flq  ratio3D  rms%%" % c for c in JP)))
for n in (1, 2, 3):
    p = res[n]["piece"]
    cols = []
    for c, j in JP.items():
        fq = np.std(fl(p["q"][:, j]))
        f3 = np.std(fl(abq[:, j] - p["beps"][:, j]))
        rms = (100 * np.sqrt(np.mean((p["full"][:, j]
                                      - abq[:, j]) ** 2))
               / np.abs(abq[:, j]).max())
        cols.append("%7.4f %7.3f %6.2f" % (fq, fq / max(f3, 1e-30),
                                           rms))
    print("%-3d %12.5e %10.4e | %s"
          % (n, res[n]["D11"], res[n]["G11"], " | ".join(cols)))

print("\nchain means along path 1 (S11): "
      + ", ".join("n=%d: beps %.4f q %.4f full %.4f"
                  % (n, res[n]["piece"]["beps"][:, 0].mean(),
                     res[n]["piece"]["q"][:, 0].mean(),
                     res[n]["piece"]["full"][:, 0].mean())
                  for n in (1, 2, 3)))
print("3-D mean S11: %.4f" % abq[:, 0].mean())

out = os.path.join(HERE, "refine_test_path1.dat")
with open(out, "w") as f:
    f.write("# SG h-refinement test along path_1 -- MATERIAL frame,"
            " parent-element means\n")
    f.write("# %10s %4s" % ("s", "mat"))
    for c in JP:
        f.write(" %13s" % ("S%s_abq" % c))
        for n in (1, 2, 3):
            f.write(" %13s %13s" % ("S%s_full_n%d" % (c, n),
                                    "S%s_q_n%d" % (c, n)))
    f.write("\n")
    for i in range(len(s)):
        f.write("%12.6f %4d" % (s[i], int(C[i, 3])))
        for c, j in JP.items():
            f.write(" %13.6e" % abq[i, j])
            for n in (1, 2, 3):
                f.write(" %13.6e %13.6e"
                        % (res[n]["piece"]["full"][i, j],
                           res[n]["piece"]["q"][i, j]))
        f.write("\n")
print("\nwrote %s" % os.path.basename(out))

for c, j in JP.items():
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(s, abq[:, j], "-", lw=2.0, color="#c0392b",
            label="Abaqus 3-D FEA (C3D20R)")
    st = {1: (":", "#7f8c8d"), 2: ("-.", "#2471a3"), 3: ("--", "k")}
    for n in (1, 2, 3):
        ax.plot(s, res[n]["piece"]["full"][:, j], st[n][0], lw=1.6,
                color=st[n][1], label="MSG-RM recovery, SG %dx%d"
                % (n, n) if n > 1 else "MSG-RM recovery, SG mesh")
    ax.set_xlabel(r"$x_1$  (mm)")
    ax.set_ylabel(r"$\sigma_{%s}$  (MPa)" % c)
    ax.grid(alpha=0.3)
    ax.axhline(0.0, color="k", lw=0.6, alpha=0.4)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    png = os.path.join(HERE, "path_1_refine_S%s.png" % c)
    fig.savefig(png, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote %s" % os.path.basename(png))
print("end   : %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
