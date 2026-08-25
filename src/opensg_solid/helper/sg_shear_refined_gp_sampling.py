"""sg_shear_refined_gp_sampling.py -- EXACT element-by-element pairing
of a tiled 3-D FEA reference against the OpenSG shear-refined recovery
(the honeycomb pairing doctrine on plate SGs).

THE DOCTRINE.  One shell element = one SG cell, and the 3-D plate mesh
is TILED from the SG -- so a cell's tets ARE the SG's tets translated
by the cell offset.  Both sides therefore evaluate the SAME elements:

  Abaqus  C3D4 carries ONE integration point; dumped with
          position=CENTROIDAL that is one value per element, matched
          here by centroid after the cell offset (gate `pair_tol`).
  OpenSG  the same tet4 mesh; a dehom .SM carries the engine's Gauss
          rows per element -- reduced to the per-element MEAN, with an
          element-major order gate against the yaml centroids (1e-6).

Use:  from opensg import helper
      cen = helper.sg_centroids("preovios_try.yaml")
      S   = helper.elem_mean_sm("preovios_try_dehom.SM", cen)
      j, ok = helper.pair_cell(cen, elems, ix, iy, tree3d)

In:  the SG yaml (either solid dialect), dehom .SM files, the 3-D
     odb-dump csv (elem,x,y,z,S11..S23 -- abq_dump_3d.py)
Out: per-element arrays aligned across the two models
"""
import os

import numpy as np
from scipy.spatial import cKDTree


def sg_centroids(yaml_path):
    """The SG's element centroids (E, 3), SG frame -- the pairing key.

    In:  yaml_path str -- the SG yaml (any solid dialect)
    Out: (cen (E, 3) float, cells (E, N) int, nodes (V, 3) float)."""
    from opensg_solid.sg_mesh import load_sg_input
    d = load_sg_input(yaml_path)
    nd = np.asarray(d["nodes"], float)[:, :3]
    cl = np.asarray(d["cells"], int)
    return nd[cl].mean(axis=1), cl, nd


def elem_mean_sm(sm_path, cen, gate=1e-6):
    """A dehom .SM -> per-element Gauss-point MEANS (E, 6), with the
    element-major order gate against the SG centroids.

    In:  sm_path str; cen (E, 3) from sg_centroids; gate float
    Out: (E, 6) float rows [S11 S22 S33 S12 S13 S23]."""
    a = np.loadtxt(sm_path)
    E = len(cen)
    ngp = len(a) // E
    if ngp * E != len(a):
        raise ValueError("%s: %d Gauss rows is not a multiple of %d"
                         " elements" % (os.path.basename(sm_path),
                                        len(a), E))
    xyz = a[:, :3].reshape(E, ngp, 3).mean(axis=1)
    off = np.abs(xyz - cen).max()
    if off >= gate:
        raise ValueError("%s is not in yaml element order (%.2e)"
                         % (os.path.basename(sm_path), off))
    return a[:, 3:9].reshape(E, ngp, 6).mean(axis=1)


def elem_mean_u(u_path, cen, gate=1e-6):
    """A dehom .U -> per-element Gauss means (E, 3), same gates."""
    a = np.loadtxt(u_path)
    E = len(cen)
    ngp = len(a) // E
    xyz = a[:, :3].reshape(E, ngp, 3).mean(axis=1)
    if np.abs(xyz - cen).max() >= gate:
        raise ValueError("%s is not in yaml element order"
                         % os.path.basename(u_path))
    return a[:, 3:6].reshape(E, ngp, 3).mean(axis=1)


def load_3d_csv(csv_path):
    """An abq_dump_3d stress csv -> (data (n, 10), cKDTree on the
    centroid columns) for pair_cell.

    In:  csv_path str -- elem,x,y,z,S11,S22,S33,S12,S13,S23
    Out: (data, tree)."""
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    return data, cKDTree(data[:, 1:4])


def pair_cell(cen, elems, ix, iy, tree, pair_tol=2e-4):
    """The 3-D-model rows of SG elements `elems` in cell (ix, iy):
    centroid-matched after the cell offset (the tiling identity).

    In:  cen (E, 3); elems (n,) int; ix, iy int cell indices; tree
         from load_3d_csv; pair_tol float
    Out: (rows (n,) int indices into the csv data, ok (n,) bool)."""
    tgt = cen[elems] + np.array([ix + 0.5, iy + 0.5, 0.0])
    dist, j = tree.query(tgt)
    return j, dist < pair_tol


def ply_region(z, h_core=1.0, t_ply=0.02):
    """Material band of a through-thickness coordinate: 1 = core,
    2/3 = the [45/-45]s ply pattern outward."""
    az = abs(z)
    if az <= h_core / 2:
        return 1
    k = min(int((az - h_core / 2) / t_ply), 3)
    return [2, 3, 3, 2][k]


def write_path_dat(path, rows, note):
    """One paired path table: rows [s x y z reg S11..S23]."""
    with open(path, "w") as f:
        f.write("# %s -- MATERIAL frame, EXACT element pairing (one"
                " C3D4 IP vs the same tet's Gauss mean)\n" % note)
        f.write("# %10s %12s %12s %12s %4s" % ("s", "x", "y", "z",
                                               "reg"))
        for c in ("S11", "S22", "S33", "S12", "S13", "S23"):
            f.write(" %13s" % c)
        f.write("\n")
        for r in rows:
            f.write("%12.6f %12.6f %12.6f %12.6f %4d" % tuple(r[:5]))
            for v in r[5:]:
                f.write(" %13.6e" % v)
            f.write("\n")
