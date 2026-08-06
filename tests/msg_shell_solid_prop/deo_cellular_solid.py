"""Deo-Yu MAMS 30(9):1737 Sec 3.1 cellular solid -- OUR SOLID (2-D SG) route.

Same Gibson-Ashby lattice as deo_cellular_shell.py (l = h = 10 cm, t = 0.5 cm,
Al E = 68.9 GPa nu = 0.33, theta = +-15 deg), homogenized with the validated
periodic 2-D CST solid solver (the UDcomp machinery).  Unit cell = one full
vertical wall + the four half-flanges of its two junctions; the four cut faces
sit at flange MID-SPANS (clean perpendicular cuts away from junctions) and
pair under the lattice vectors a2 = (l cos, h + l sin) and a2 - a1:

    F+ tip of top junction  <->  F- tip of bottom junction   (+a2)
    F- tip of top junction  <->  F+ tip of bottom junction   (a2 - a1)

Mesh: gmsh OCC fuse of the 5 wall rectangles; the 4 cut edges carry identical
transfinite seeding so periodic partners mesh node-for-node.  3 translation
pins; C3D = (Dee + V0^T Dhe) / (2 l cos * (h + l sin)).

Reads the shell-route C from deo_cell_shell_*.out for the 3-way table.
Run AFTER deo_cellular_shell.py (from this folder):  python deo_cellular_solid.py
"""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

############### User Input #################################
l_f = 0.10
h_w = 0.10
t_w = 0.005
E, nu = 68.9e9, 0.33
n_cut = 21               # transfinite nodes across each cut face
msize = 0.00035           # target mesh size (m); C44 is bending-dominated and
                         # linear CST needs ~8 elements through t to de-lock
############################################################

PAPER = {
    15.0: {"C11": (4736.9, 4678.9), "C12": (1089.4, 1105.5),
           "C13": (381.81, 386.88), "C22": (2446.39, 2488.9),
           "C23": (847.44, 860.89), "C33": (306.99, 311.48),
           "C44": (4.3215, 4.1919), "C55": (564.15, 573.11),
           "C66": (997.52, 1000.1)},
    -15.0: {"C11": (7507.4, 7352.9), "C12": (1094.1, 1075.2),
            "C13": (-220.53, -213.94), "C22": (4154.9, 4080.0),
            "C23": (-847.45, -821.77), "C33": (180.75, 173.48),
            "C44": (2.6775, 2.4705), "C55": (332.17, 346.01),
            "C66": (1694.2, 1706.4)},
}
IDX = {"C11": (0, 0), "C12": (0, 1), "C13": (0, 2), "C22": (1, 1),
       "C23": (1, 2), "C33": (2, 2), "C44": (3, 3), "C55": (4, 4),
       "C66": (5, 5)}


def build_mesh(th_deg):
    """gmsh mesh of the two-star unit cell; returns nodes (nn,2), tris."""
    import gmsh
    th = np.radians(th_deg)
    lc, ls = l_f*np.cos(th), l_f*np.sin(th)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("cell")
    occ = gmsh.model.occ
    # walls: (junction point, direction angle, length)
    walls = [((0.0, -h_w), np.pi/2, h_w),                    # V: J1 -> J0
             ((0.0, 0.0), th, l_f/2),                        # F+ @ J0
             ((0.0, 0.0), np.pi - th, l_f/2),                # F- @ J0
             ((0.0, -h_w), -th, l_f/2),                      # F+ @ J1
             ((0.0, -h_w), np.pi + th, l_f/2)]               # F- @ J1
    tags = []
    for (px, py), ang, ln in walls:
        r = occ.addRectangle(0.0, -t_w/2, 0.0, ln, t_w)
        occ.rotate([(2, r)], 0, 0, 0, 0, 0, 1, ang)
        occ.translate([(2, r)], px, py, 0.0)
        tags.append((2, r))
    out, _ = occ.fuse([tags[0]], tags[1:])
    occ.synchronize()
    # cut-face tips (centers of the 4 periodic cut edges)
    tips = [np.array([lc/2, ls/2]),                          # F+ @ J0
            np.array([-lc/2, ls/2]),                         # F- @ J0
            np.array([lc/2, -h_w - ls/2]),                   # F+ @ J1
            np.array([-lc/2, -h_w - ls/2])]                  # F- @ J1
    for dim, s in out:
        for (d1, e) in gmsh.model.getBoundary([(dim, s)], oriented=False):
            xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(d1, e)
            mid = np.array([(xmin+xmax)/2, (ymin+ymax)/2])
            if any(np.linalg.norm(mid - tp) < t_w/4 for tp in tips):
                gmsh.model.mesh.setTransfiniteCurve(e, n_cut)
    gmsh.option.setNumber("Mesh.MeshSizeMax", msize)
    gmsh.option.setNumber("Mesh.MeshSizeMin", msize/3)
    gmsh.model.mesh.generate(2)
    ntags, coords, _ = gmsh.model.mesh.getNodes()
    xy_all = coords.reshape(-1, 3)[:, :2]
    remap = {int(t): i for i, t in enumerate(ntags)}
    etypes, _, enodes = gmsh.model.mesh.getElements(2)
    tri = None
    for et, en in zip(etypes, enodes):
        if et == 2:
            tri = np.array([remap[int(v)] for v in en]).reshape(-1, 3)
    gmsh.finalize()
    used = np.unique(tri)
    newid = -np.ones(len(xy_all), int); newid[used] = np.arange(len(used))
    return xy_all[used], newid[tri], lc, ls


def solve_case(th_deg):
    t0 = time.perf_counter()
    nd, tri, lc, ls = build_mesh(th_deg)
    nn, ne = len(nd), len(tri)
    A_cell = 2*lc*(h_w + ls)

    lam = E*nu/((1+nu)*(1-2*nu)); mu = E/(2*(1+nu))
    C0 = np.zeros((6, 6)); C0[:3, :3] = lam
    C0[np.arange(3), np.arange(3)] = lam + 2*mu
    C0[3, 3] = C0[4, 4] = C0[5, 5] = mu

    p1, p2, p3 = nd[tri[:, 0]], nd[tri[:, 1]], nd[tri[:, 2]]
    det = ((p2[:, 0]-p1[:, 0])*(p3[:, 1]-p1[:, 1])
           - (p3[:, 0]-p1[:, 0])*(p2[:, 1]-p1[:, 1]))
    area = 0.5*np.abs(det)
    A_mesh = float(area.sum())
    b = np.stack([p2[:, 1]-p3[:, 1], p3[:, 1]-p1[:, 1],
                  p1[:, 1]-p2[:, 1]], 1)/det[:, None]
    cs = np.stack([p3[:, 0]-p2[:, 0], p1[:, 0]-p3[:, 0],
                   p2[:, 0]-p1[:, 0]], 1)/det[:, None]
    B = np.zeros((ne, 6, 9))
    for a in range(3):
        B[:, 1, 3*a+1] = b[:, a]
        B[:, 2, 3*a+2] = cs[:, a]
        B[:, 3, 3*a+1] = cs[:, a]; B[:, 3, 3*a+2] = b[:, a]
        B[:, 4, 3*a] = cs[:, a]
        B[:, 5, 3*a] = b[:, a]
    Ke = np.einsum('e,eia,ij,ejb->eab', area, B, C0, B)
    Fe = np.einsum('e,eia,ij->eaj', area, B, C0)
    Dee = C0 * A_mesh

    from scipy.sparse import coo_matrix, csr_matrix, bmat
    ndof = 3*nn
    gdof = (3*tri[:, :, None] + np.arange(3)[None, None, :]).reshape(ne, 9)
    rr = np.broadcast_to(gdof[:, :, None], (ne, 9, 9)).ravel()
    cc = np.broadcast_to(gdof[:, None, :], (ne, 9, 9)).ravel()
    K = coo_matrix((Ke.ravel(), (rr, cc)), shape=(ndof, ndof)).tocsr()
    Dhe = np.zeros((ndof, 6))
    np.add.at(Dhe, gdof.ravel(), Fe.reshape(-1, 6))

    # periodic pairing: slave faces at J1 map onto master faces at J0
    th = np.radians(th_deg)
    dirs = {"Fp": np.array([np.cos(th), np.sin(th)]),
            "Fm": np.array([-np.cos(th), np.sin(th)])}
    tipJ0 = {"Fp": np.array([lc/2, ls/2]), "Fm": np.array([-lc/2, ls/2])}
    tipJ1 = {"Fp": np.array([lc/2, -h_w - ls/2]),
             "Fm": np.array([-lc/2, -h_w - ls/2])}
    a2 = np.array([lc, h_w + ls])

    def face_nodes(P, d):
        rel = nd - P
        perp = d[0]*rel[:, 1] - d[1]*rel[:, 0]
        on = (np.abs(rel @ d) < 1e-9) & (np.abs(perp) < t_w/2 + 1e-9)
        return np.where(on)[0]

    master = np.arange(nn)
    pairs = [(tipJ1["Fm"], dirs["Fp"], tipJ0["Fp"], a2),
             (tipJ1["Fp"], dirs["Fm"], tipJ0["Fm"], -a2 + 2*np.array([lc, 0]))]
    # second pair: (lc/2,-h-ls/2) + (-lc, h+ls) = (-lc/2, ls/2): shift = a2-a1
    pairs[1] = (tipJ1["Fp"], dirs["Fm"], tipJ0["Fm"],
                np.array([-lc, h_w + ls]))
    for tipS, dS, tipM, shift in pairs:
        sl = face_nodes(tipS, dS if tipS is tipJ1["Fm"] else dS)
        ms = face_nodes(tipM, dS)
        assert len(sl) == len(ms) > 0, "cut faces %d vs %d" % (len(sl), len(ms))
        for s in sl:
            tgt = nd[s] + shift
            dd = np.linalg.norm(nd[ms] - tgt, axis=1)
            assert dd.min() < 1e-7, "periodic partner miss %.2e" % dd.min()
            master[s] = ms[np.argmin(dd)]
    for _ in range(3):
        master = master[master]

    uniq, inv = np.unique(master, return_inverse=True)
    nred = 3*len(uniq)
    trows = 3*np.repeat(np.arange(nn), 3) + np.tile(np.arange(3), nn)
    tcols = 3*np.repeat(inv, 3) + np.tile(np.arange(3), nn)
    T = csr_matrix((np.ones(3*nn), (trows, tcols)), shape=(ndof, nred))
    Kr = (T.T @ K @ T).tocsr()
    Fr = T.T @ Dhe
    wA = np.zeros(nn)
    np.add.at(wA, tri.ravel(), np.repeat(area/3.0, 3))
    wAr = np.zeros(len(uniq))
    np.add.at(wAr, inv, wA)
    Cc = np.zeros((3, nred))
    Cc[0, 0::3] = wAr; Cc[1, 1::3] = wAr; Cc[2, 2::3] = wAr
    A = bmat([[Kr, csr_matrix(Cc).T], [csr_matrix(Cc), None]], format="csc")
    Rhs = np.zeros((nred+3, 6)); Rhs[:nred] = -Fr
    try:
        from pypardiso import spsolve as _sp
        V0r = np.column_stack([_sp(A, Rhs[:, k]) for k in range(6)])
    except Exception:
        from scipy.sparse.linalg import spsolve as _sp
        V0r = np.column_stack([_sp(A, Rhs[:, k]) for k in range(6)])
    V0 = T @ V0r[:nred]
    C_so = (Dee + V0.T @ Dhe) / A_cell
    C_so = 0.5*(C_so + C_so.T)
    dt = time.perf_counter() - t0
    print("theta=%+.0f: %d nodes, %d tris, A_mesh=%.6e (midline %.6e),"
          " A_cell=%.6e  [%.1f s]"
          % (th_deg, nn, ne, A_mesh, (h_w + 2*l_f)*t_w, A_cell, dt))
    return C_so, nd, tri, A_cell, dt


def read_shell_C(tag):
    C = []
    with open("deo_cell_shell_%s.out" % tag) as f:
        on = False
        for ln in f:
            if ln.startswith("# ---- C3D"):
                on = True; continue
            if on:
                if ln.startswith("#"):
                    break
                C.append([float(v) for v in ln.split()])
    return np.array(C)                                       # MPa


def constants(Cm):
    S = np.linalg.inv(Cm)
    return {"E1": 1/S[0, 0], "E2": 1/S[1, 1], "E3": 1/S[2, 2],
            "G23": 1/S[3, 3], "G13": 1/S[4, 4], "G12": 1/S[5, 5],
            "nu12": -S[0, 1]/S[0, 0], "nu13": -S[0, 2]/S[0, 0],
            "nu23": -S[1, 2]/S[1, 1]}


lines_all = []
meshes = {}
for th in (15.0, -15.0):
    tag = "p15" if th > 0 else "m15"
    C_so, nd, tri, A_cell, dt = solve_case(th)
    meshes[th] = (nd, tri)
    Cm = C_so / 1e6
    np.savetxt("deo_cell_solid_%s.out" % tag, Cm, fmt="%16.8e",
               header="our 2-D solid periodic UC (MPa), theta=%+.0f;"
                      " order [e11 e22 e33 2e23 2e13 2e12]; A_cell=%.8e;"
                      " %.1f s" % (th, A_cell, dt))
    C_sh = read_shell_C(tag)
    print("\ntheta=%+.0f deg [MPa]: ours RM-shell / ours solid / Deo TW /"
          " Deo solid:" % th)
    hdr = ("  %-5s %12s %12s %12s %12s %11s %11s"
           % ("term", "ours_shell", "ours_solid", "Deo_TW", "Deo_solid",
              "sh_vs_ourso", "ourso_vs_Deo"))
    print(hdr)
    lines_all += ["# theta = %+.0f deg  (C in MPa)" % th, "#" + hdr[1:]]
    for nm, (i, j) in IDX.items():
        tw, so = PAPER[th][nm]
        osh, oso = C_sh[i, j], Cm[i, j]
        row = ("  %-5s %12.4f %12.4f %12.4f %12.4f %+11.3f %+11.3f"
               % (nm, osh, oso, tw, so,
                  100*(osh-oso)/oso, 100*(oso-so)/so))
        print(row); lines_all.append(row)
    csh, cso = constants(C_sh), constants(Cm)

    def ptab(col):
        Cp = np.zeros((6, 6))
        for nm2, (i2, j2) in IDX.items():
            Cp[i2, j2] = Cp[j2, i2] = PAPER[th][nm2][col]
        return constants(Cp)
    ctw, cds = ptab(0), ptab(1)
    hdr2 = ("  %-6s %12s %12s %12s %12s" % ("const", "ours_shell",
            "ours_solid", "Deo_TW", "Deo_solid"))
    print(hdr2)
    lines_all += ["# 9 constants (E,G MPa; nu -)", "#" + hdr2[1:]]
    for kx in ("E1", "E2", "E3", "G23", "G13", "G12", "nu12", "nu13", "nu23"):
        row = ("  %-6s %12.5f %12.5f %12.5f %12.5f"
               % (kx, csh[kx], cso[kx], ctw[kx], cds[kx]))
        print(row); lines_all.append(row)
    lines_all.append("")

with open("deo_cell_compare3.dat", "w") as f:
    f.write("# Deo-Yu MAMS 30(9) Sec 3.1 cellular solid, theta=+-15:"
            " ours RM-shell / ours 2-D solid / Deo MSG-TW / Deo MSG-solid\n")
    f.write("# both our routes periodic (shell: merged junction DOFs;"
            " solid: cut-face master-slave); C in MPa\n")
    f.write("\n".join(lines_all) + "\n")

fig, axs = plt.subplots(1, 2, figsize=(11, 6))
for k, th in enumerate((15.0, -15.0)):
    nd, tri = meshes[th]
    axs[k].tripcolor(nd[:, 0], nd[:, 1], tri, facecolors=np.zeros(len(tri)),
                     cmap="Blues", vmin=-1, vmax=1, edgecolors="k",
                     linewidth=0.15)
    axs[k].set_aspect("equal")
    axs[k].set_xlabel("y2 (m)"); axs[k].set_ylabel("y3 (m)")
fig.tight_layout(); fig.savefig("deo_cell_solid_meshes.png", dpi=200)
print("\nwrote deo_cell_solid_p15.out/_m15.out, deo_cell_compare3.dat,"
      " deo_cell_solid_meshes.png")
