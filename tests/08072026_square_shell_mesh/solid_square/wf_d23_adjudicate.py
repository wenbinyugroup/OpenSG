"""TASK D -- D23 adjudication: is shell D23 = 0 (iso 0/90 lattice) the correct
8-strain Reissner-Mindlin answer (junction physics outside the model), or a
Gamma_e/Gamma_h bug?

Three parts:
  1. EXACT WALL TEST: single flat iso wall (normal y3, thickness t, free faces,
     periodic along-wall) under macro ebar22 = 1: analytic plane stress says
     sigma22 = E' = E/(1-nu^2), sigma11 = nu*E', sigma33 = 0.  A tiny 2-D CST
     strip SG (same machinery as single_wall_c44.py) confirms avg sigma33 = 0
     to machine precision -> an interior WALL contributes ZERO to D23.
  2. JUNCTION LAW FIT: read ibeam_thick_sweep.dat and test
     D23_solid ~= lambda * A_j with A_j = 2 t^2 (two web-flange junctions of
     t x t), lambda = E nu/((1+nu)(1-2nu)).  Bonus: same census on the iso
     square ring (4 corners, A_j = 4 t^2) and the 0-deg orthotropic square
     (lambda -> ply 3-D C23).
  3. STRUCTURAL-ZERO CHECK: run the ACTUAL build_solid_bundle (current, clean
     code) on
       (a) iso 0/90 cross cell      -> D23 (expect machine zero) + C44 state
       (b) iso X-cell (+-45 walls)  -> D23 (expect LARGE ~ E' t L/sqrt(2))
       (c) fresh m45 square ring    -> D23 (untainted by the reverted e_nn)
       (d) mixed 45/0 square ring   -> can A16/A26 warping give D23 on 0/90?

Run (from this folder):  python wf_d23_adjudicate.py
"""
import time

import numpy as np

from opensg_shell import build_solid_bundle
from opensg_shell.sg_periodicity import mesh_to_periodic_sparse_assembly_map

E, nu = 70.0e9, 0.30
G_iso = E/(2*(1 + nu))
mu = G_iso
lam3d = E*nu/((1 + nu)*(1 - 2*nu))
Ep = E/(1 - nu**2)

print("="*72)
print("PART 1 -- EXACT WALL TEST (wall normal y3, free faces, ebar22 = 1)")
print("="*72)
print("analytic (one line): e22=1, e11=0, sigma33=0 pointwise (free faces) ->")
print("  sigma22 = E' = %.6e   sigma11 = nu*E' = %.6e   sigma33 = 0" % (Ep, nu*Ep))

# ---- 2-D CST strip: x = y2 (along wall, periodic), y = y3 (thickness, free)
L_w, t_w, nt = 0.10, 0.03, 8
na = max(8, int(round(L_w/(t_w/nt))))
xg = np.linspace(0, L_w, na + 1)
yg = np.linspace(-t_w/2, t_w/2, nt + 1)
nx, ny = len(xg), len(yg)
nd = np.array([[xg[i], yg[j]] for i in range(nx) for j in range(ny)])
nid = lambda i, j: i*ny + j
tri = []
for i in range(nx - 1):
    for j in range(ny - 1):
        n00, n10 = nid(i, j), nid(i + 1, j)
        n01, n11 = nid(i, j + 1), nid(i + 1, j + 1)
        if (i + j) % 2 == 0:
            tri += [[n00, n10, n11], [n00, n11, n01]]
        else:
            tri += [[n00, n10, n01], [n10, n11, n01]]
tri = np.array(tri, int)
nn, ne = len(nd), len(tri)

C0 = np.zeros((6, 6)); C0[:3, :3] = lam3d
C0[np.arange(3), np.arange(3)] = lam3d + 2*mu
C0[3, 3] = C0[4, 4] = C0[5, 5] = mu

p1, p2, p3 = nd[tri[:, 0]], nd[tri[:, 1]], nd[tri[:, 2]]
det = ((p2[:, 0]-p1[:, 0])*(p3[:, 1]-p1[:, 1])
       - (p3[:, 0]-p1[:, 0])*(p2[:, 1]-p1[:, 1]))
area = 0.5*np.abs(det)
b = np.stack([p2[:, 1]-p3[:, 1], p3[:, 1]-p1[:, 1], p1[:, 1]-p2[:, 1]], 1)/det[:, None]
cs = np.stack([p3[:, 0]-p2[:, 0], p1[:, 0]-p3[:, 0], p2[:, 0]-p1[:, 0]], 1)/det[:, None]
B = np.zeros((ne, 6, 9))
for k in range(3):
    B[:, 1, 3*k+1] = b[:, k]
    B[:, 2, 3*k+2] = cs[:, k]
    B[:, 3, 3*k+1] = cs[:, k]; B[:, 3, 3*k+2] = b[:, k]
    B[:, 4, 3*k] = cs[:, k]
    B[:, 5, 3*k] = b[:, k]
Ke = np.einsum('e,eia,ij,ejb->eab', area, B, C0, B)
Fe = np.einsum('e,eia,ij->eaj', area, B, C0)
Dee = C0*float(area.sum())
ndof = 3*nn
gd = (3*tri[:, :, None] + np.arange(3)[None, None, :]).reshape(ne, 9)
K = np.zeros((ndof, ndof)); Dhe = np.zeros((ndof, 6))
np.add.at(K, (gd[:, :, None].repeat(9, 2), gd[:, None, :].repeat(9, 1)), Ke)
np.add.at(Dhe, gd.ravel(), Fe.reshape(-1, 6))

rc, _ = mesh_to_periodic_sparse_assembly_map(nn, np.arange(nn)[:, None],
                                             nd, 2, 3)      # tie x (along-wall)
master = np.asarray(rc, int).ravel()
uniq, inv = np.unique(master, return_inverse=True)
nred = 3*len(uniq)
T = np.zeros((ndof, nred))
for n in range(nn):
    for k in range(3):
        T[3*n+k, 3*inv[n]+k] = 1.0
wA = np.zeros(nn)
np.add.at(wA, tri.ravel(), np.repeat(area/3.0, 3))
wAr = np.zeros(len(uniq))
np.add.at(wAr, inv, wA)
Cc = np.zeros((3, nred))
Cc[0, 0::3] = wAr; Cc[1, 1::3] = wAr; Cc[2, 2::3] = wAr
A = np.zeros((nred+3, nred+3))
A[:nred, :nred] = T.T @ K @ T; A[:nred, nred:] = Cc.T; A[nred:, :nred] = Cc
Rhs = np.zeros((nred+3, 6)); Rhs[:nred] = -(T.T @ Dhe)
V0 = T @ np.linalg.solve(A, Rhs)[:nred]
C_strip = (Dee + V0.T @ Dhe)/float(area.sum())

sig = C_strip[:, 1]                      # avg stress under ebar22 = 1
print("FEM strip (%d x %d grid, %d CSTs, free faces, periodic ends):"
      % (na, nt, ne))
print("  avg sigma11 = %.6e   (nu*E' = %.6e, ratio %.6f)"
      % (sig[0], nu*Ep, sig[0]/(nu*Ep)))
print("  avg sigma22 = %.6e   (E'    = %.6e, ratio %.6f)"
      % (sig[1], Ep, sig[1]/Ep))
print("  avg sigma33 = %.6e   -> |sigma33|/E' = %.2e   (MACHINE ZERO?)"
      % (sig[2], abs(sig[2])/Ep))
print("CONCLUSION 1: an interior wall carries ZERO average sigma33 under")
print("  transverse stretch -- walls cannot contribute to D23 at all; any")
print("  nonzero solid D23 must come from NON-WALL (junction) material.")

print()
print("="*72)
print("PART 2 -- JUNCTION-CENSUS LAW  D23 ~= lambda * A_j")
print("="*72)
print("lambda = E nu/((1+nu)(1-2nu)) = %.6e Pa" % lam3d)
print("I-beam (b=0.5 h=1.0): 2 web-flange junctions of t x t -> A_j = 2 t^2")
rows = []
cur_t = None
for ln in open("ibeam_thick_sweep.dat"):
    if ln.startswith("==== t="):
        cur_t = float(ln.split("t=")[1].split()[0])
    p = ln.split()
    if p and p[0] == "D23" and cur_t is not None:
        rows.append((cur_t, float(p[3])))          # msg_solid column
        cur_t = None
print("  %-7s %14s %14s %10s" % ("t", "lambda*2t^2", "D23_solid", "law/solid"))
for t, d23 in rows:
    law = lam3d*2*t**2
    print("  %-7.3f %14.5e %14.5e %10.4f" % (t, law, d23, law/d23))

# bonus: iso square ring, 4 corners -> A_j = 4 t^2  (a=1, t=0.03, area 0.12)
for ln in open("square_solid_iso_compare.dat"):
    p = ln.split()
    if p and p[0] == "C23":
        d23_sq = float(p[2])*0.12
        law = lam3d*4*0.03**2
        print("iso square ring (4 corners, A_j=4t^2, t=0.03):"
              "  law %.5e / solid %.5e = %.4f" % (law, d23_sq, law/d23_sq))
        break

# bonus: 0-deg orthotropic square -- lambda -> ply 3-D C23 (material frame)
E1, E2, E3 = 142.0e9, 9.8e9, 9.8e9
nu12, nu13, nu23 = 0.30, 0.30, 0.42
S = np.array([[1/E1, -nu12/E1, -nu13/E1],
              [-nu12/E1, 1/E2, -nu23/E2],
              [-nu13/E1, -nu23/E2, 1/E3]])
C3 = np.linalg.inv(S)
for ln in open("square_solid_three_way.dat"):
    p = ln.split()
    if p and p[0] == "C23":
        d23_o = float(p[3])*0.12                   # old-solid benchmark column
        law = C3[1, 2]*4*0.03**2
        print("0-deg ORTHO square (ply C23_3D=%.4e):"
              "      law %.5e / solid %.5e = %.4f"
              % (C3[1, 2], law, d23_o, law/d23_o))
        break

print()
print("="*72)
print("PART 3 -- IS D23 STRUCTURALLY ZEROED IN THE SHELL FORMULATION?")
print("="*72)


def write_ring_yaml(fname, walls, sections_txt, mats_txt, set_lists):
    """walls: list of (p0, p1, nseg); returns nothing, writes yaml."""
    pts, cells, tans, idx = [], [], [], {}

    def add_node(p):
        key = (round(p[0], 12), round(p[1], 12))
        if key not in idx:
            idx[key] = len(pts)
            pts.append(np.array(p, float))
        return idx[key]

    eids_per_wall = []
    for p0, p1, n in walls:
        p0, p1 = np.array(p0, float), np.array(p1, float)
        tv = (p1 - p0)/np.linalg.norm(p1 - p0)
        ids = [add_node(p0 + (p1 - p0)*k/n) for k in range(n + 1)]
        wall_eids = []
        for k in range(n):
            wall_eids.append(len(cells))
            cells.append([ids[k], ids[k + 1]])
            tans.append(tv)
        eids_per_wall.append(wall_eids)

    o = ["nodes:"]
    for x, y in np.array(pts):
        o.append("- [%.12f %.12f 0.00000000]" % (x, y))
    o.append("elements:")
    for c1, c2 in cells:
        o.append("- [%d %d]" % (c1 + 1, c2 + 1))
    o.append("elementOrientations:")
    for tv in tans:
        e3 = np.cross([0, 0, 1.0], [tv[0], tv[1], 0])
        o.append("- [0.0, 0.0, 1.0, %.9f, %.9f, 0.0, %.9f, %.9f, 0.0]"
                 % (tv[0], tv[1], e3[0], e3[1]))
    o += sections_txt + mats_txt
    o += ["sets:", "  element:"]
    for sname, wlist in set_lists:
        o.append("  - name: %s" % sname)
        o.append("    labels:")
        for w in wlist:
            o += ["    - %d" % (e + 1) for e in eids_per_wall[w]]
    o.append("reference: center")
    open(fname, "w").write("\n".join(o) + "\n")


iso_mat = ["materials:", "- name: iso", "  density: 2700.0", "  elastic:",
           "    E: [%.6e, %.6e, %.6e]" % (E, E, E),
           "    G: [%.6e, %.6e, %.6e]" % (G_iso, G_iso, G_iso),
           "    nu: [%.6f, %.6f, %.6f]" % (nu, nu, nu)]
m45_mat = ["materials:", "- name: m45", "  density: 1600.0", "  elastic:",
           "    E: [1.420000e+11, 9.800000e+09, 9.800000e+09]",
           "    G: [6.000000e+09, 6.000000e+09, 4.800000e+09]",
           "    nu: [0.300000, 0.300000, 0.420000]"]


def sec_txt(entries):
    o = ["sections:"]
    for sname, mname, t, ang in entries:
        o += ["- type: shell", "  elementSet: %s" % sname, "  layup:",
              "  - - %s" % mname, "    - %.6e" % t, "    - %.1f" % ang]
    return o


# ---- (a) iso 0/90 cross cell, pitch p = L/sqrt(2), t = 0.01 ---------------
Lc = 1.0
p_pitch = Lc/np.sqrt(2.0)
t3 = 0.01
nsg = 12
t0 = time.perf_counter()
write_ring_yaml("wf_cross.yaml",
                [((-p_pitch/2, 0.0), (p_pitch/2, 0.0), 2*nsg),
                 ((0.0, -p_pitch/2), (0.0, p_pitch/2), 2*nsg)],
                sec_txt([("layup_0", "iso", t3, 0.0)]), iso_mat,
                [("layup_0", [0, 1])])
Bc = build_solid_bundle("wf_cross.yaml", cell_area=p_pitch**2)
Dc = np.asarray(Bc["D_eff"]); Dc = 0.5*(Dc + Dc.T)
c44_exact = 0.5*Ep*(t3/p_pitch)**3*p_pitch**2
print("(a) iso 0/90 CROSS cell (pitch %.4f, t=%.2f)  [%.1f s]"
      % (p_pitch, t3, time.perf_counter() - t0))
print("    D22 = %.5e  D33 = %.5e  (E't p = %.5e)"
      % (Dc[1, 1], Dc[2, 2], Ep*t3*p_pitch))
print("    D23 = %.5e   -> |D23|/D22 = %.2e  (machine zero?)"
      % (Dc[1, 2], abs(Dc[1, 2])/Dc[1, 1]))
print("    D44 = %.5e   vs exact relaxed %.5e  -> ratio %.3f"
      " (known ~4x state)" % (Dc[3, 3], c44_exact, Dc[3, 3]/c44_exact))

# ---- (b) iso X-cell: same lattice rotated 45 deg --------------------------
t0 = time.perf_counter()
write_ring_yaml("wf_xcell.yaml",
                [((-Lc/2, -Lc/2), (Lc/2, Lc/2), 2*nsg),
                 ((-Lc/2, Lc/2), (Lc/2, -Lc/2), 2*nsg)],
                sec_txt([("layup_0", "iso", t3, 0.0)]), iso_mat,
                [("layup_0", [0, 1])])
Bx = build_solid_bundle("wf_xcell.yaml", cell_area=Lc**2)
Dx = np.asarray(Bx["D_eff"]); Dx = 0.5*(Dx + Dx.T)
# Bond rotation of the cross cell by 45 deg, scaled to cell area L^2
sc = Lc**2/p_pitch**2
D23_rot = sc*(0.25*(Dc[1, 1] + Dc[2, 2]) + 0.5*Dc[1, 2] - Dc[3, 3])
D23_membr = Ep*t3*Lc/np.sqrt(2.0)
print("(b) iso X-CELL (+-45 walls corner-to-corner, L=%.1f, t=%.2f)  [%.1f s]"
      % (Lc, t3, time.perf_counter() - t0))
print("    D23 = %.5e" % Dx[1, 2])
print("    thin-limit membrane prediction  E' t L/sqrt2   = %.5e (ratio %.4f)"
      % (D23_membr, Dx[1, 2]/D23_membr))
print("    rotation of computed cross cell (uses its D44) = %.5e (ratio %.4f)"
      % (D23_rot, Dx[1, 2]/D23_rot))
print("    D22 = %.5e  D33 = %.5e  D44 = %.5e"
      % (Dx[1, 1], Dx[2, 2], Dx[3, 3]))

# ---- (c) fresh m45 square ring (copy of the repo yaml, clean code) --------
t0 = time.perf_counter()
open("wf_m45_square.yaml", "w").write(
    open("../square_tube_1Dshell.yaml").read())
Bm = build_solid_bundle("wf_m45_square.yaml", cell_area=0.12)
Cm = np.asarray(Bm["C3D"]); Cm = 0.5*(Cm + Cm.T)
print("(c) FRESH m45 square ring (45-deg ply all four walls)  [%.1f s]"
      % (time.perf_counter() - t0))
print("    C22 = %.5e  C33 = %.5e" % (Cm[1, 1], Cm[2, 2]))
print("    C23 = %.5e   -> |C23|/C22 = %.2e  (machine zero?)"
      % (Cm[1, 2], abs(Cm[1, 2])/Cm[1, 1]))
print("    stored (possibly e_nn-tainted) run had C23 = 4.48777e-20")

# ---- (d) mixed ring: 45-ply on horizontal walls, 0-ply on vertical --------
a_sq, t_sq = 1.0, 0.03
t0 = time.perf_counter()
write_ring_yaml("wf_mix45.yaml",
                [((-a_sq/2, -a_sq/2), (a_sq/2, -a_sq/2), nsg),   # bottom H
                 ((a_sq/2, -a_sq/2), (a_sq/2, a_sq/2), nsg),     # right  V
                 ((a_sq/2, a_sq/2), (-a_sq/2, a_sq/2), nsg),     # top    H
                 ((-a_sq/2, a_sq/2), (-a_sq/2, -a_sq/2), nsg)],  # left   V
                sec_txt([("layup_h", "m45", t_sq, 45.0),
                         ("layup_v", "m45", t_sq, 0.0)]), m45_mat,
                [("layup_h", [0, 2]), ("layup_v", [1, 3])])
Bmm = build_solid_bundle("wf_mix45.yaml", cell_area=4*a_sq*t_sq)
Cmm = np.asarray(Bmm["C3D"]); Cmm = 0.5*(Cmm + Cmm.T)
print("(d) MIXED square ring: 45-ply horizontals, 0-ply verticals  [%.1f s]"
      % (time.perf_counter() - t0))
print("    C22 = %.5e  C33 = %.5e" % (Cmm[1, 1], Cmm[2, 2]))
print("    C23 = %.5e   -> |C23|/sqrt(C22*C33) = %.2e"
      % (Cmm[1, 2], abs(Cmm[1, 2])/np.sqrt(Cmm[1, 1]*Cmm[2, 2])))

print()
print("="*72)
print("done.")
