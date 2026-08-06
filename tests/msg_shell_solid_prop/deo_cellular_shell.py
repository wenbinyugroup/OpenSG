"""Deo-Yu MAMS 30(9):1737 (2023) Sec 3.1 cellular-solid benchmark -- SHELL route.

Gibson-Ashby lattice: vertical walls h = 10 cm, inclined flanges l = 10 cm at
flange angle theta from horizontal (+15 deg = hexagonal closed polygons,
-15 deg = re-entrant), wall thickness t = 0.5 cm, aluminum E = 68.9 GPa,
nu = 0.33.  Periodic unit cell = the minimal Y-cell:

    vertical wall (0,-h)->(0,0),  flanges (0,0)->(+-l cos, l sin),

with the THREE cut ends (vertical bottom + both flange tips) periodically
merged into a single DOF slot -- they are all images of one lattice junction,
so the merge reconstitutes the full Y junction (displacement + rotation
continuity across differently-oriented walls, the RM analog of the paper's
Eq. 11-12 junction conditions).  Cell area = |a1 x a2| = 2 l cos * (h + l sin)
including void (paper Eq. 13).  Kernel = 3 translations (periodicity kills the
rigid rotation).

Our segments are RM (MSG shell), the paper's are CLPT -- the paper names
transverse shear as its dominant error source (C44: 2.99% at +15, 7.73% at
-15), so the RM route should land closer to the paper's MSG-solid reference.

Voigt order [e11 e22 e33 2e23 2e13 2e12], 1 = extrusion axis.  Paper C in MPa.

Run (from this folder):  python deo_cellular_shell.py
"""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml as _yaml
from scipy.linalg import lu_factor, lu_solve

from opensg_shell.segment_indep import (assemble_segment_indep,
                                        assemble_constraint, NDOF6)
from opensg_shell.solid_props import (assemble_solid_macro, elastic_constants,
                                      GBAR_ORDER)
from opensg_shell.fe_jax.msg_rm_timo import build_C_Psi
from opensg_shell.oml_ring import load_ring_ref

############### User Input #################################
l_f = 0.10               # flange length (m)
h_w = 0.10               # vertical wall length (m)
t_w = 0.005              # wall thickness (m)
E, nu = 68.9e9, 0.33     # aluminum
nseg = 24                # elements per wall segment
############################################################

# Deo-Yu Tables 1-2 (MPa): columns MSG-TW (CLPT) / MSG-solid (reference)
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
G = E / (2*(1+nu))


def run_case(theta_deg):
    th = np.radians(theta_deg)
    lc, ls = l_f*np.cos(th), l_f*np.sin(th)
    A_cell = 2*lc*(h_w + ls)
    t0 = time.perf_counter()

    # ---- Y-cell contour: vertical + two flanges sharing the origin junction
    sV = np.linspace(0, 1, nseg+1)
    nV = np.stack([np.zeros(nseg+1), -h_w*(1-sV), np.zeros(nseg+1)], 1)  # bottom->0
    nFp = np.stack([lc*sV, ls*sV, np.zeros(nseg+1)], 1)                  # 0->tip
    nFm = np.stack([-lc*sV, ls*sV, np.zeros(nseg+1)], 1)
    nodes3 = np.vstack([nV, nFp[1:], nFm[1:]])       # origin appears once (nV[-1])
    iJ = nseg                                        # origin junction index
    iVb = 0                                          # vertical bottom end
    iFp = nseg + nseg                                # F+ tip
    iFm = nseg + 2*nseg                              # F- tip
    cells, ori = [], []

    def tri(tv):
        e2 = np.array([tv[0], tv[1], 0.0]); e2 /= np.linalg.norm(e2)
        e3 = np.cross([0.0, 0.0, 1.0], e2)
        return [0.0, 0.0, 1.0, e2[0], e2[1], 0.0, e3[0], e3[1], 0.0]

    for i in range(nseg):                            # vertical wall
        cells.append([i, i+1]); ori.append(tri([0.0, 1.0]))
    prev = iJ
    for i in range(nseg):                            # F+
        nxt = nseg + 1 + i if i < nseg else iFp
        cells.append([prev, nseg+1+i]); prev = nseg+1+i
        ori.append(tri([np.cos(th), np.sin(th)]))
    prev = iJ
    for i in range(nseg):                            # F-
        cells.append([prev, 2*nseg+1+i]); prev = 2*nseg+1+i
        ori.append(tri([-np.cos(th), np.sin(th)]))

    out = ["nodes:"]
    for x, y, z in nodes3:
        out.append("- [%.12f %.12f %.12f]" % (x, y, z))
    out.append("elements:")
    for a, b in cells:
        out.append("- [%d %d]" % (a+1, b+1))
    out.append("elementOrientations:")
    for r in ori:
        out.append("- [" + ", ".join("%.12f" % v for v in r) + "]")
    out += ["sections:",
            "- type: shell",
            "  elementSet: layup_0",
            "  layup:",
            "  - - al",
            "    - %.6e" % t_w,
            "    - 0.0",
            "materials:",
            "- name: al",
            "  density: 2700.0",
            "  elastic:",
            "    E: [%.6e, %.6e, %.6e]" % (E, E, E),
            "    G: [%.6e, %.6e, %.6e]" % (G, G, G),
            "    nu: [%.6f, %.6f, %.6f]" % (nu, nu, nu),
            "sets:",
            "  element:",
            "  - name: layup_0",
            "    labels:"]
    out += ["    - %d" % (e+1) for e in range(len(cells))]
    out.append("reference: center")
    yname = "deo_cell_%s.yaml" % ("p15" if theta_deg > 0 else "m15")
    with open(yname, "w") as f:
        f.write("\n".join(out) + "\n")

    R = load_ring_ref(yname, "center")
    rx, rcells = R["rx"], R["cells"]
    ax, cross = R["ax"], R["cross"]
    G_by = list(R["G_by"])
    from opensg_solid.rm_plate_1D.msg_rm_plate import rm_plate_msg
    from opensg_shell.emit_abd import material_db_from_yaml
    d_sh = _yaml.safe_load(open(yname))
    _mdb = material_db_from_yaml(d_sh["materials"])
    for si, sec in enumerate(d_sh["sections"]):
        _pl = [[str(p[0]), float(p[1]), float(p[2])] for p in sec["layup"]]
        _rr = rm_plate_msg([p[1] for p in _pl], [p[2] for p in _pl],
                           [p[0] for p in _pl], _mdb, fraction=0.5)
        if _rr["G_msg"] is not None:
            G_by[si] = np.asarray(_rr["G_msg"])

    # ---- periodic 3-way merge of the cut ends (one lattice junction)
    mloc = len(rx)

    def find_node(pt):
        i = np.argmin(np.linalg.norm(rx[:, cross] - np.asarray(pt), axis=1))
        assert np.linalg.norm(rx[i, cross] - pt) < 1e-9
        return i

    rep = np.arange(mloc)
    jb = find_node([0.0, -h_w])
    rep[find_node([lc, ls])] = jb
    rep[find_node([-lc, ls])] = jb
    _, node_map = np.unique(rep, return_inverse=True)

    # ---- strip assembly with merged dof_map, 3-translation kernel
    h_st = float(np.mean(np.linalg.norm(rx[rcells[:, 1]] - rx[rcells[:, 0]],
                                        axis=1)))
    ez = np.zeros(3); ez[ax] = 1.0
    nodes_st = np.vstack([rx, rx + h_st*ez])
    dof_map = np.concatenate([node_map, node_map])
    quads = np.array([[a, b, mloc+b, mloc+a] for a, b in rcells], int)
    e3q = np.asarray(R["re3"])

    Dhh, _, _, _, _, _ = assemble_segment_indep(
        nodes_st, quads, R["rsub"], e3q, R["D_by"], G_by, np.asarray(R["k22"]),
        cross, ax, kg_e=None, pen=0.0, dof_map=dof_map, shear="mitc4_g23")
    Gc, _, _ = assemble_constraint(nodes_st, quads, R["rsub"], e3q,
                                   np.asarray(R["k22"]), cross, ax,
                                   dof_map=dof_map, lam_space="elem")
    Dhe6, Dee6 = assemble_solid_macro(nodes_st, quads, R["rsub"], e3q,
                                      R["D_by"], G_by, cross, ax,
                                      dof_map=dof_map, shear="mitc4_g23")
    Dhh = np.asarray(Dhh)/h_st; Gc = Gc/h_st; Dhe6 = Dhe6/h_st; Dee6 = Dee6/h_st

    M = Dhh.shape[0]; P = Gc.shape[0]
    C5, _ = build_C_Psi(rx[:, cross], rcells, p=1)
    C3 = np.zeros((3, M))
    for n in range(mloc):
        s = NDOF6 * node_map[n]
        C3[:, s:s+5] += C5[:3, 5*n:5*n+5]
    naug = M + P
    A = np.zeros((naug + 3, naug + 3))
    A[:M, :M] = Dhh; A[:M, M:naug] = Gc.T; A[M:naug, :M] = Gc
    A[:M, naug:] = C3.T; A[naug:, :M] = C3
    R0 = np.zeros((naug + 3, 6)); R0[:M] = -Dhe6
    V0 = lu_solve(lu_factor(A), R0)[:naug]
    Deff = Dee6 + V0[:M].T @ Dhe6
    C_sh = 0.5*(Deff + Deff.T) / A_cell
    dt = time.perf_counter() - t0
    print("theta=%+.0f: %d nodes (3-way merged), %d elems, A_cell=%.6e m^2"
          "  [%.1f s]" % (theta_deg, mloc, len(rcells), A_cell, dt))
    return C_sh, A_cell, dt, rx, rcells, cross


results = {}
for th in (15.0, -15.0):
    results[th] = run_case(th)

# ---------------- outputs vs the paper --------------------------------------
lines_all = []
for th in (15.0, -15.0):
    C_sh, A_cell, dt, rx, rcells, cross = results[th]
    Cm = C_sh / 1e6                                  # -> MPa
    tag = "p15" if th > 0 else "m15"
    cons, S = elastic_constants(C_sh)
    with open("deo_cell_shell_%s.out" % tag, "w") as f:
        f.write("# Deo-Yu MAMS 30(9) Sec 3.1, theta=%+.0f deg; RM shell route,"
                " periodic Y-cell\n" % th)
        f.write("# l=h=0.1 m, t=0.005 m, Al E=68.9 GPa nu=0.33; %s\n"
                % GBAR_ORDER)
        f.write("# cell area=%.8e m^2; solve %.1f s\n" % (A_cell, dt))
        f.write("# ---- C3D stiffness (6x6, MPa) ----\n")
        for i in range(6):
            f.write(" ".join("%16.8e" % Cm[i, j] for j in range(6)) + "\n")
        f.write("# ---- 9 engineering constants (Pa / -) ----\n")
        for kx in ("E1", "E2", "E3", "G23", "G13", "G12",
                   "nu12", "nu13", "nu23"):
            f.write("%-6s %16.8e\n" % (kx, cons[kx]))

    print("\ntheta=%+.0f deg: ours (RM) vs paper MSG-TW (CLPT) vs paper"
          " MSG-solid [MPa]:" % th)
    print("  %-5s %12s %12s %12s %11s %11s"
          % ("term", "ours_RM", "Deo_TW", "Deo_solid",
             "ours_vs_sol", "TW_vs_sol"))
    lines_all.append("# theta = %+.0f deg  (C in MPa)" % th)
    lines_all.append("# %-5s %12s %12s %12s %11s %11s"
                     % ("term", "ours_RM", "Deo_TW", "Deo_solid",
                        "ours_vs_sol%", "TW_vs_sol%"))
    idx = {"C11": (0, 0), "C12": (0, 1), "C13": (0, 2), "C22": (1, 1),
           "C23": (1, 2), "C33": (2, 2), "C44": (3, 3), "C55": (4, 4),
           "C66": (5, 5)}
    for nm, (i, j) in idx.items():
        tw, so = PAPER[th][nm]
        ours = Cm[i, j]
        row = ("  %-5s %12.4f %12.4f %12.4f %+11.3f %+11.3f"
               % (nm, ours, tw, so, 100*(ours-so)/so, 100*(tw-so)/so))
        print(row); lines_all.append(row)

    # 9 constants: ours vs constants from the paper's tabulated 6x6
    def C_from_table(col):
        Cp = np.zeros((6, 6))
        for nm2, (i2, j2) in idx.items():
            Cp[i2, j2] = Cp[j2, i2] = PAPER[th][nm2][col]
        return Cp * 1e6
    cons_tw, _ = elastic_constants(C_from_table(0))
    cons_so, _ = elastic_constants(C_from_table(1))
    print("  %-6s %12s %12s %12s %11s"
          % ("const", "ours_RM", "Deo_TW", "Deo_solid", "ours_vs_sol"))
    lines_all.append("# 9 constants (E,G in MPa; nu -)")
    lines_all.append("# %-6s %12s %12s %12s %11s"
                     % ("const", "ours_RM", "Deo_TW", "Deo_solid",
                        "ours_vs_sol%"))
    for kx in ("E1", "E2", "E3", "G23", "G13", "G12", "nu12", "nu13", "nu23"):
        sc = 1e-6 if kx[0] in "EG" else 1.0
        a, b2, c2 = cons[kx]*sc, cons_tw[kx]*sc, cons_so[kx]*sc
        row = ("  %-6s %12.5f %12.5f %12.5f %+11.3f"
               % (kx, a, b2, c2, 100*(a-c2)/c2))
        print(row); lines_all.append(row)
    lines_all.append("")

with open("deo_cell_compare.dat", "w") as f:
    f.write("# Deo-Yu MAMS 30(9):1737 Sec 3.1 cellular solid, theta=+-15 deg\n")
    f.write("# ours = OpenSG RM-shell solid-props (periodic Y-cell);"
            " paper MSG-TW (CLPT) and MSG-solid (reference), Tables 1-2\n")
    f.write("\n".join(lines_all) + "\n")

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
for k2, th in enumerate((15.0, -15.0)):
    _, _, _, rx, rcells, cross = results[th]
    for a2, b2 in rcells:
        axs[k2].plot(rx[[a2, b2], cross[0]], rx[[a2, b2], cross[1]],
                     "b-", lw=1.2)
    axs[k2].plot(rx[:, cross[0]], rx[:, cross[1]], "k.", ms=2)
    thr2 = np.radians(th)
    for pt in ([0, -h_w], [l_f*np.cos(thr2), l_f*np.sin(thr2)],
               [-l_f*np.cos(thr2), l_f*np.sin(thr2)]):
        axs[k2].plot(pt[0], pt[1], "rs", ms=7)
    axs[k2].set_aspect("equal")
    axs[k2].set_xlabel("y2 (m)"); axs[k2].set_ylabel("y3 (m)")
fig.tight_layout(); fig.savefig("deo_cell_contours.png", dpi=200)
print("\nwrote deo_cell_shell_p15.out, deo_cell_shell_m15.out,"
      " deo_cell_compare.dat, deo_cell_contours.png")
