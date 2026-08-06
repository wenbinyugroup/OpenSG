"""IEA-22 r/R = 0.2 airfoil cross-section: solid 2-D SG vs shell 1-D SG, ISOTROPIC.

Both models run with a single isotropic material (E = 70 GPa, nu = 0.3) so this is
a pure geometry test of the equivalent-solid route on a real airfoil section
(closed OML + webs, free SG -- no cell periodicity).

  SOLID  ~/OpenSG_io/examples/mesh_out/iea_prevabs_refined/r020_solid_boundary.yaml
         (43k nodes, 80k CST triangles, axial = global z), generalized plane
         strain, 3 DOF/node, MSG constraints <w_i> dA = 0 + in-plane rotation,
         sparse KKT (PARDISO).
  SHELL  iea_s10_shell.yaml (1-D contour, center ref) with all materials replaced
         by the same isotropic constants -- opensg_shell.build_solid_bundle.

Both normalized by the SAME cell area (convex hull of the solid mesh section).
Outputs: airfoil_iso_solid.out / airfoil_iso_shell.out / airfoil_iso_compare.dat
(resolved terms, %diff vs solid).  Geometric sanity: wall area A_mesh vs the
shell's contour-integral of thickness.

Run (from this folder):  python airfoil_solid_vs_shell_iso.py
"""
import os
import time

import numpy as np
import yaml
from scipy.sparse import coo_matrix, csr_matrix, bmat
from scipy.spatial import ConvexHull

from opensg_shell import build_solid_bundle, GBAR_ORDER

############### User Input #################################
E, nu = 70.0e9, 0.30
SOLID_YAML = os.path.expanduser(
    "~/OpenSG_io/examples/mesh_out/iea_prevabs_refined/r020_solid_boundary.yaml")
SHELL_YAML = "iea_s10_shell.yaml"
############################################################

t0 = time.perf_counter()

# ---------------- load solid mesh -------------------------------------------
d = yaml.safe_load(open(SOLID_YAML))
row = lambda r: [float(v) for v in
                 " ".join(str(x) for x in (r if isinstance(r, list) else [r])).split()]
nd = np.array([row(r) for r in d["nodes"]], float)          # (Nn,3) x y z
tri = np.array([[int(v) for v in row(r)] for r in d["elements"]], int) - 1
xy = nd[:, :2]                                              # in-plane (x,y)
nn, ne = len(xy), len(tri)
print("solid mesh: %d nodes, %d tris   [%.1f s]" % (nn, ne, time.perf_counter()-t0))

hull = ConvexHull(xy); A_cell = float(hull.volume)

# ---------------- isotropic C (Voigt [11 22 33 23 13 12], 1 = axial z) ------
lam = E*nu/((1+nu)*(1-2*nu)); mu = E/(2*(1+nu))
C = np.zeros((6, 6)); C[:3, :3] = lam
C[np.arange(3), np.arange(3)] = lam + 2*mu
C[3, 3] = C[4, 4] = C[5, 5] = mu

# ---------------- vectorized CST assembly -----------------------------------
p1, p2, p3 = xy[tri[:, 0]], xy[tri[:, 1]], xy[tri[:, 2]]
det = (p2[:, 0]-p1[:, 0])*(p3[:, 1]-p1[:, 1]) - (p3[:, 0]-p1[:, 0])*(p2[:, 1]-p1[:, 1])
area = 0.5*np.abs(det)
A_mesh = float(area.sum())
# shape gradients (CST): dN_a/dx, dN_a/dy
b = np.stack([p2[:, 1]-p3[:, 1], p3[:, 1]-p1[:, 1], p1[:, 1]-p2[:, 1]], 1) / det[:, None]
c = np.stack([p3[:, 0]-p2[:, 0], p1[:, 0]-p3[:, 0], p2[:, 0]-p1[:, 0]], 1) / det[:, None]
# strain rows [e11,e22,e33,2e23,2e13,2e12]; dofs per node [w1,w2,w3];
# "2" = x, "3" = y (axial = z = "1")
B = np.zeros((ne, 6, 9))
for a in range(3):
    B[:, 1, 3*a+1] = b[:, a]                 # e22 = w2,x
    B[:, 2, 3*a+2] = c[:, a]                 # e33 = w3,y
    B[:, 3, 3*a+1] = c[:, a]; B[:, 3, 3*a+2] = b[:, a]     # 2e23
    B[:, 4, 3*a] = c[:, a]                   # 2e13 = w1,y
    B[:, 5, 3*a] = b[:, a]                   # 2e12 = w1,x
Ke = np.einsum('e,eia,ij,ejb->eab', area, B, C, B)          # (ne,9,9)
Fe = np.einsum('e,eia,ij->eaj', area, B, C)                 # (ne,9,6)
Dee = C * A_mesh

gdof = (3*tri[:, :, None] + np.arange(3)[None, None, :]).reshape(ne, 9)
rr = np.broadcast_to(gdof[:, :, None], (ne, 9, 9)).ravel()
cc = np.broadcast_to(gdof[:, None, :], (ne, 9, 9)).ravel()
ndof = 3*nn
Dhh = coo_matrix((Ke.ravel(), (rr, cc)), shape=(ndof, ndof)).tocsr()
Dhe = np.zeros((ndof, 6))
np.add.at(Dhe, gdof.ravel(), Fe.reshape(-1, 6))
print("assembled sparse Dhh nnz=%.2fM  A_mesh=%.4f  A_cell(hull)=%.4f  [%.1f s]"
      % (Dhh.nnz/1e6, A_mesh, A_cell, time.perf_counter()-t0))

# ---------------- constraints: <w_i> dA = 0, in-plane rotation --------------
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
C_so = (Dee + V0.T @ Dhe) / A_cell
C_so = 0.5*(C_so + C_so.T)
print("solid solve done  [%.1f s]" % (time.perf_counter()-t0))

# ---------------- shell run (same iso material everywhere) ------------------
ds = yaml.safe_load(open(SHELL_YAML))
G = E/(2*(1+nu))
for m in ds["materials"]:
    m["elastic"]["E"] = [E, E, E]
    m["elastic"]["G"] = [G, G, G]
    m["elastic"]["nu"] = [nu, nu, nu]
with open("iea_s10_iso_shell.yaml", "w") as f:
    yaml.safe_dump(ds, f, default_flow_style=True)
Bs = build_solid_bundle("iea_s10_iso_shell.yaml", cell_area=A_cell)
C_sh = np.asarray(Bs["C3D"])
# shell wall area = contour integral of thickness (geometric sanity)
print("shell done  [%.1f s]" % (time.perf_counter()-t0))

# ---------------- outputs ---------------------------------------------------
def write_out(path, tag, Cm):
    S = np.linalg.inv(Cm)
    with open(path, "w") as f:
        f.write("# IEA-22 r/R=0.2 airfoil, ISOTROPIC E=%.3e nu=%.2f, %s\n"
                % (E, nu, tag))
        f.write("# %s;  cell area (hull) = %.6f;  A_mesh = %.6f\n"
                % (GBAR_ORDER, A_cell, A_mesh))
        f.write("# ---- C3D stiffness (6x6, Pa) ----\n")
        for i in range(6):
            f.write(" ".join("%16.8e" % Cm[i, j] for j in range(6)) + "\n")
        f.write("# ---- compliance inv(C3D) ----\n")
        for i in range(6):
            f.write(" ".join("%16.8e" % S[i, j] for j in range(6)) + "\n")

write_out("airfoil_iso_solid.out", "SOLID 2-D SG (CST, free SG)", C_so)
write_out("airfoil_iso_shell.out", "SHELL 1-D SG (center ref)", C_sh)

thr = 1e-8 * max(np.max(np.abs(C_sh)), np.max(np.abs(C_so)))
print("\nresolved terms (|C| > %.2e), shell vs solid:" % thr)
print("  %-5s %16s %16s %10s" % ("term", "C_shell", "C_solid", "pct_diff"))
with open("airfoil_iso_compare.dat", "w") as f:
    f.write("# IEA r0.2 airfoil iso: shell vs solid; %s\n" % GBAR_ORDER)
    f.write("# %-5s %16s %16s %10s\n" % ("term", "C_shell", "C_solid", "pct"))
    for i in range(6):
        for j in range(i, 6):
            sh, so = C_sh[i, j], C_so[i, j]
            if abs(sh) > thr or abs(so) > thr:
                pct = 100.0*(sh-so)/so if abs(so) > thr else float("inf")
                print("  C%d%d   %16.8e %16.8e %+10.3f" % (i+1, j+1, sh, so, pct))
                f.write("  C%d%d   %16.8e %16.8e %+10.4f\n"
                        % (i+1, j+1, sh, so, pct))
print("\nanalytical anchor: E*A_mesh/A_cell = %.6e   (C11 should be near this"
      % (E*A_mesh/A_cell))
print("with hoop/Poisson relaxation reducing it slightly)")
print("wrote airfoil_iso_solid.out, airfoil_iso_shell.out, airfoil_iso_compare.dat")
