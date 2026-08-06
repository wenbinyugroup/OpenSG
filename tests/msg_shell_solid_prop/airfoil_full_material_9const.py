"""IEA-22 r/R = 0.2 airfoil 2-D solid SG, FULL layup materials -- model=3 solid props.

Same solver as the UD-composite example (generalized plane strain, 6 unit macro
strains, D_eff = (Dee + V0^T Dhe)/omega) applied to the real r=10 cross-section
mesh with its 6 materials and per-element fiber orientations:

    ~/OpenSG_io/examples/mesh_out/iea_prevabs_refined/r020_solid_boundary.yaml
    (43k nodes, 80k CST triangles, axial = global z)

Being an ISOLATED cross-section (free lateral surfaces, no in-plane cell
periodicity -- the ingredient the UD square cell has), the free-SG solve
relaxes the in-plane and axial-shear columns; the resolved content is the
axial column (E1).  The script reports the full 6x6, all 9 constants from
the compliance, and flags which are membrane-resolved vs relaxed.

Voigt order [e11 e22 e33 2e23 2e13 2e12], 1 = beam axis (global z).

Run (from this folder):  python airfoil_full_material_9const.py
"""
import os
import time

import numpy as np
import yaml
from scipy.sparse import coo_matrix, csr_matrix, bmat
from scipy.spatial import ConvexHull

############### User Input #################################
SOLID_YAML = os.path.expanduser(
    "~/OpenSG_io/examples/mesh_out/iea_prevabs_refined/r020_solid_boundary.yaml")
############################################################

VI = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]


def C_ply(E, G, nu):
    """Orthotropic 6x6 stiffness in ply axes; OpenSG dialect E=[E1,E2,E3],
    G=[G12,G13,G23], nu=[nu12,nu13,nu23]; Voigt [11,22,33,23,13,12]."""
    S = np.zeros((6, 6))
    S[0, 0], S[1, 1], S[2, 2] = 1/E[0], 1/E[1], 1/E[2]
    S[0, 1] = S[1, 0] = -nu[0]/E[0]
    S[0, 2] = S[2, 0] = -nu[1]/E[0]
    S[1, 2] = S[2, 1] = -nu[2]/E[1]
    S[3, 3], S[4, 4], S[5, 5] = 1/G[2], 1/G[1], 1/G[0]
    return np.linalg.inv(S)


def bond_batch(a):
    """Batched 6x6 Bond stress-rotation M from (ne,3,3) a with sig' = a sig a^T;
    then C' = M C M^T (engineering Voigt [11,22,33,23,13,12])."""
    M = np.empty((len(a), 6, 6))
    for r, (i, j) in enumerate(VI):
        for cidx, (k, l) in enumerate(VI):
            if k == l:
                M[:, r, cidx] = a[:, i, k]*a[:, j, k]
            else:
                M[:, r, cidx] = a[:, i, k]*a[:, j, l] + a[:, i, l]*a[:, j, k]
    return M


def rotate_C_tensor(C6, Q):
    """Reference full 4th-order rotation (rows of Q = new axes in old comps)."""
    C4 = np.zeros((3, 3, 3, 3))
    for aa, (i, j) in enumerate(VI):
        for bb, (k, l) in enumerate(VI):
            C4[i, j, k, l] = C4[j, i, k, l] = C4[i, j, l, k] = C4[j, i, l, k] = C6[aa, bb]
    C4r = np.einsum('pi,qj,rk,sl,ijkl->pqrs', Q, Q, Q, Q, C4)
    out = np.zeros((6, 6))
    for aa, (i, j) in enumerate(VI):
        for bb, (k, l) in enumerate(VI):
            out[aa, bb] = C4r[i, j, k, l]
    return out


t0 = time.perf_counter()

# ---------------- load mesh + materials + orientations ----------------------
d = yaml.safe_load(open(SOLID_YAML))
row = lambda r: [float(v) for v in
                 " ".join(str(x) for x in (r if isinstance(r, list) else [r])).split()]
nd = np.array([row(r) for r in d["nodes"]], float)
tri = np.array([[int(v) for v in row(r)] for r in d["elements"]], int) - 1
xy = nd[:, :2]
nn, ne = len(xy), len(tri)
orient = np.array(d["elementOrientations"], float).reshape(ne, 3, 3)  # rows e1,e2,e3 (x,y,z)
Cply = {m["name"]: C_ply(m["E"], m["G"], m["nu"]) for m in d["materials"]}
mat_of = np.empty(ne, dtype=object)
for s in d["sets"]["element"]:
    mat_of[np.array(s["labels"], int) - 1] = s["name"]
print("mesh: %d nodes, %d tris, %d materials  [%.1f s]"
      % (nn, ne, len(Cply), time.perf_counter()-t0))

hull = ConvexHull(xy); A_cell = float(hull.volume)

# ---------------- per-element global stiffness ------------------------------
# solver global frame: axis 1 = beam axial = global z, axis 2 = x, axis 3 = y.
# triad rows are (x,y,z) comps -> reorder each vector to (z,x,y).
Q = orient[:, :, [2, 0, 1]]              # rows = ply axes in solver-global comps
a = np.transpose(Q, (0, 2, 1))           # v_glob = a v_ply
M = bond_batch(a)
Cp = np.stack([Cply[m] for m in mat_of])
Ce = np.einsum('eij,ejk,elk->eil', M, Cp, M)
# spot-check the batched Bond route against the full tensor rotation
for e in np.random.default_rng(0).choice(ne, 5, replace=False):
    ref = rotate_C_tensor(Cply[mat_of[e]], Q[e].T)
    assert np.allclose(Ce[e], ref, rtol=1e-9, atol=1e-3), "Bond mismatch elem %d" % e
print("per-element rotated C done  [%.1f s]" % (time.perf_counter()-t0))

# ---------------- vectorized CST assembly -----------------------------------
p1, p2, p3 = xy[tri[:, 0]], xy[tri[:, 1]], xy[tri[:, 2]]
det = (p2[:, 0]-p1[:, 0])*(p3[:, 1]-p1[:, 1]) - (p3[:, 0]-p1[:, 0])*(p2[:, 1]-p1[:, 1])
area = 0.5*np.abs(det)
A_mesh = float(area.sum())
b = np.stack([p2[:, 1]-p3[:, 1], p3[:, 1]-p1[:, 1], p1[:, 1]-p2[:, 1]], 1) / det[:, None]
c = np.stack([p3[:, 0]-p2[:, 0], p1[:, 0]-p3[:, 0], p2[:, 0]-p1[:, 0]], 1) / det[:, None]
B = np.zeros((ne, 6, 9))
for k in range(3):
    B[:, 1, 3*k+1] = b[:, k]
    B[:, 2, 3*k+2] = c[:, k]
    B[:, 3, 3*k+1] = c[:, k]; B[:, 3, 3*k+2] = b[:, k]
    B[:, 4, 3*k] = c[:, k]
    B[:, 5, 3*k] = b[:, k]
Ke = np.einsum('e,eia,eij,ejb->eab', area, B, Ce, B)
Fe = np.einsum('e,eia,eij->eaj', area, B, Ce)
Dee = np.einsum('e,eij->ij', area, Ce)

gdof = (3*tri[:, :, None] + np.arange(3)[None, None, :]).reshape(ne, 9)
rr = np.broadcast_to(gdof[:, :, None], (ne, 9, 9)).ravel()
cc = np.broadcast_to(gdof[:, None, :], (ne, 9, 9)).ravel()
ndof = 3*nn
Dhh = coo_matrix((Ke.ravel(), (rr, cc)), shape=(ndof, ndof)).tocsr()
Dhe = np.zeros((ndof, 6))
np.add.at(Dhe, gdof.ravel(), Fe.reshape(-1, 6))
print("assembled  A_mesh=%.4f  A_cell(hull)=%.4f  [%.1f s]"
      % (A_mesh, A_cell, time.perf_counter()-t0))

# ---------------- constraints + KKT solve -----------------------------------
wA = np.zeros(nn)
np.add.at(wA, tri.ravel(), np.repeat(area/3.0, 3))
Cc = np.zeros((4, ndof))
Cc[0, 0::3] = wA; Cc[1, 1::3] = wA; Cc[2, 2::3] = wA
Cc[3, 1::3] = -wA*xy[:, 1]; Cc[3, 2::3] = wA*xy[:, 0]
Csp = csr_matrix(Cc)
A_kkt = bmat([[Dhh, Csp.T], [Csp, None]], format="csc")
Rhs = np.zeros((ndof+4, 6)); Rhs[:ndof] = -Dhe
try:
    from pypardiso import spsolve
    V0 = np.column_stack([spsolve(A_kkt, Rhs[:, k]) for k in range(6)])[:ndof]
except Exception:
    from scipy.sparse.linalg import spsolve as ssp
    V0 = np.column_stack([ssp(A_kkt, Rhs[:, k]) for k in range(6)])[:ndof]
C3D = (Dee + V0.T @ Dhe) / A_cell
C3D = 0.5*(C3D + C3D.T)
print("solve done  [%.1f s]" % (time.perf_counter()-t0))

# ---------------- outputs ---------------------------------------------------
S = np.linalg.inv(C3D)
E1, E2, E3 = 1/S[0, 0], 1/S[1, 1], 1/S[2, 2]
G23, G13, G12 = 1/S[3, 3], 1/S[4, 4], 1/S[5, 5]
n12, n13, n23 = -S[0, 1]/S[0, 0], -S[0, 2]/S[0, 0], -S[1, 2]/S[1, 1]

thr = 1e-8*np.max(np.abs(C3D))
print("\nC3D (6x6, Pa), order [e11 e22 e33 2e23 2e13 2e12]:")
for i in range(6):
    print("  " + " ".join("%12.4e" % C3D[i, j] for j in range(6)))
print("resolved diagonals (|Cii| > %.2e): %s"
      % (thr, [i+1 for i in range(6) if abs(C3D[i, i]) > thr]))

rows = [("E1 (GPa)", E1/1e9), ("E2 (GPa)", E2/1e9), ("E3 (GPa)", E3/1e9),
        ("G23 (GPa)", G23/1e9), ("G13 (GPa)", G13/1e9), ("G12 (GPa)", G12/1e9),
        ("nu12", n12), ("nu13", n13), ("nu23", n23)]
flags = ["membrane-resolved" if abs(C3D[i, i]) > thr else "relaxed (free SG)"
         for i in [0, 1, 2, 3, 4, 5]]
fmap = {"E1 (GPa)": flags[0], "E2 (GPa)": flags[1], "E3 (GPa)": flags[2],
        "G23 (GPa)": flags[3], "G13 (GPa)": flags[4], "G12 (GPa)": flags[5],
        "nu12": flags[1], "nu13": flags[2], "nu23": flags[2]}
print("\n9 effective constants (free cross-section SG):")
with open("airfoil_full_9const.dat", "w") as f:
    f.write("# IEA-22 r/R=0.2 airfoil solid 2-D SG, FULL materials, model=3\n")
    f.write("# free SG (no in-plane cell periodicity) -> only the axial column\n")
    f.write("# is membrane-resolved; relaxed constants are bending-scale residuals\n")
    f.write("# %-10s %14s   %s\n" % ("const", "value", "status"))
    for (nm, v), key in zip(rows, [r[0] for r in rows]):
        print("  %-10s %14.6g   %s" % (nm, v, fmap[key]))
        f.write("  %-10s %14.6g   %s\n" % (nm, v, fmap[key]))

with open("airfoil_full_solid.out", "w") as f:
    f.write("# IEA-22 r/R=0.2 airfoil, FULL layup materials, solid 2-D SG model=3\n")
    f.write("# order [e11 e22 e33 2e23 2e13 2e12]; A_cell(hull)=%.6f A_mesh=%.6f\n"
            % (A_cell, A_mesh))
    f.write("# ---- C3D stiffness (6x6, Pa) ----\n")
    for i in range(6):
        f.write(" ".join("%16.8e" % C3D[i, j] for j in range(6)) + "\n")
    f.write("# ---- compliance inv(C3D) ----\n")
    for i in range(6):
        f.write(" ".join("%16.8e" % S[i, j] for j in range(6)) + "\n")
print("\nwrote airfoil_full_solid.out, airfoil_full_9const.dat")
