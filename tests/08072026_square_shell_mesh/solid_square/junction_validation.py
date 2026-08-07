"""Validation matrix for the junction census correction (junction="census").

  A  iso square ring   vs square_solid_bicg_iso.out   (material-area norm 0.12)
  B  m45 square ring   vs square_solid_newarch.out    (material-area norm 0.12)
  C  iso I-beam t = 0.03 / 0.10 / 0.20  vs msg_solid D_eff (ibeam_thick_sweep)
  D  flag-off identity: bundle without the flag == square_solid_msg_shell.out

Normal-block terms only (the correction touches nothing else by design).

Run (from this folder):  python junction_validation.py
"""
import numpy as np
import yaml as _yaml

from opensg_shell import build_solid_bundle

############### User Input #################################
E, nu, t0 = 70.0e9, 0.30, 0.03
b_ib, h_ib = 0.5, 1.0
T_IB = [0.03, 0.10, 0.20]
ISO_YAML = "square_tube_1Dshell_iso.yaml"
M45_YAML = "../square_tube_1Dshell.yaml"
ISO_SOLID = "square_solid_bicg_iso.out"
M45_SOLID = "square_solid_newarch.out"
MSH_OUT = "square_solid_msg_shell.out"
AREA_N = 0.12                      # material-area normalization of the .outs
############################################################

# msg_solid I-beam D_eff (ibeam_thick_sweep.dat, periodic, refined CST)
IB_SOLID = {0.03: {"D11": 4.57719e9, "D22": 2.32479e9, "D33": 2.42527e9,
                   "D12": 7.18585e8, "D13": 7.48730e8, "D23": 7.04961e7},
            0.10: {"D11": 1.49729e10, "D22": 7.91249e9, "D33": 9.04370e9,
                   "D12": 2.61845e9, "D13": 2.95782e9, "D23": 8.15688e8},
            0.20: {"D11": 2.92687e10, "D22": 1.65858e10, "D33": 2.14309e10,
                   "D12": 6.05435e9, "D13": 7.50788e9, "D23": 3.59539e9}}
TERMS = [("D11", 0, 0), ("D22", 1, 1), ("D33", 2, 2), ("D12", 0, 1),
         ("D13", 0, 2), ("D23", 1, 2), ("D44", 3, 3)]
for _t, _v in ((0.03, 2.28505e6), (0.10, 9.79450e7), (0.20, 9.97446e8)):
    IB_SOLID[_t]["D44"] = _v


def read_out_C(path):
    C, mode = [], None
    for ln in open(path):
        if ln.startswith("# ---- C"):
            mode = "C"; continue
        if ln.startswith("#"):
            continue
        p = ln.split()
        if mode == "C" and len(p) == 6:
            C.append([float(v) for v in p])
        if len(C) == 6:
            break
    return np.array(C)


def ring_table(name, yaml_path, solid_path):
    Cso = read_out_C(solid_path)
    b0 = build_solid_bundle(yaml_path)
    b1 = build_solid_bundle(yaml_path, junction="census")
    b2 = build_solid_bundle(yaml_path, junction="micro")
    b3 = build_solid_bundle(yaml_path, junction="microcell")
    nj = b1["junction"]["n_junctions"]
    print("---- %s: %d junctions detected ----" % (name, nj))
    print("  %-5s %13s %13s %13s %13s %9s %9s %9s %9s"
          % ("term", "census", "micro", "microcell", "solid",
             "off/so", "cen/so", "mic/so", "cell/so"))
    for nm, i, j in TERMS:
        so = Cso[i, j]*AREA_N
        thr = 1e-3*abs(Cso[0, 0]*AREA_N)
        vals = [b0["D_eff"][i, j], b1["D_eff"][i, j], b2["D_eff"][i, j],
                b3["D_eff"][i, j]]
        rs = [v/so if abs(so) > thr else np.inf for v in vals]
        print("  %-5s %13.5e %13.5e %13.5e %13.5e %9.4f %9.4f %9.4f %9.4f"
              % (nm, vals[1], vals[2], vals[3], so, rs[0], rs[1], rs[2],
                 rs[3]))
    return b0


def ibeam_yaml(t, nseg=16):
    pts, cells, tans, idx = [], [], [], {}

    def add_node(p):
        key = (round(p[0], 12), round(p[1], 12))
        if key not in idx:
            idx[key] = len(pts)
            pts.append(np.array(p))
        return idx[key]

    def add_wall(p0, p1, n):
        p0, p1 = np.array(p0, float), np.array(p1, float)
        tv = (p1-p0)/np.linalg.norm(p1-p0)
        ids = [add_node(p0 + (p1-p0)*k/n) for k in range(n+1)]
        for k in range(n):
            cells.append([ids[k], ids[k+1]])
            tans.append(tv)

    b, h = b_ib, h_ib
    add_wall([-b/2,  h/2], [0.0,  h/2], nseg)
    add_wall([0.0,  h/2], [b/2,  h/2], nseg)
    add_wall([-b/2, -h/2], [0.0, -h/2], nseg)
    add_wall([0.0, -h/2], [b/2, -h/2], nseg)
    add_wall([0.0, -h/2], [0.0,  h/2], 2*nseg)
    G = E/(2*(1+nu))
    o = ["nodes:"]
    for x, y in np.array(pts):
        o.append("- [%.12f %.12f 0.00000000]" % (x, y))
    o.append("elements:")
    for c1, c2 in cells:
        o.append("- [%d %d]" % (c1+1, c2+1))
    o.append("elementOrientations:")
    for tv in tans:
        e3 = np.cross([0, 0, 1.0], [tv[0], tv[1], 0])
        o.append("- [0.0, 0.0, 1.0, %.9f, %.9f, 0.0, %.9f, %.9f, 0.0]"
                 % (tv[0], tv[1], e3[0], e3[1]))
    o += ["sections:", "- type: shell", "  elementSet: layup_0", "  layup:",
          "  - - iso", "    - %.6e" % t, "    - 0.0",
          "materials:", "- name: iso", "  density: 2700.0", "  elastic:",
          "    E: [%.6e, %.6e, %.6e]" % (E, E, E),
          "    G: [%.6e, %.6e, %.6e]" % (G, G, G),
          "    nu: [%.6f, %.6f, %.6f]" % (nu, nu, nu),
          "sets:", "  element:", "  - name: layup_0", "    labels:"]
    o += ["    - %d" % (e+1) for e in range(len(cells))]
    o.append("reference: center")
    open("_jv_ibeam.yaml", "w").write("\n".join(o) + "\n")
    return "_jv_ibeam.yaml"


# ---- A / B: rings --------------------------------------------------------
b_iso = ring_table("A: iso square ring", ISO_YAML, ISO_SOLID)
ring_table("B: m45 square ring", M45_YAML, M45_SOLID)

# ---- C: iso I-beam sweep -------------------------------------------------
print("---- C: iso I-beam (b=%.2f h=%.2f), sh/so per term ----" % (b_ib, h_ib))
print("  %-6s" % "t/h"
      + "".join("   %s off/cen/mic" % nm for nm, _, _ in TERMS))
for t in T_IB:
    y = ibeam_yaml(t)
    b0 = build_solid_bundle(y, cell_area=b_ib*h_ib)
    b1 = build_solid_bundle(y, cell_area=b_ib*h_ib, junction="census")
    b2 = build_solid_bundle(y, cell_area=b_ib*h_ib, junction="micro")
    b3 = build_solid_bundle(y, cell_area=b_ib*h_ib, junction="microcell")
    row = "  %-6.3f" % (t/h_ib)
    for nm, i, j in TERMS:
        so = IB_SOLID[t][nm]
        row += " %6.3f %6.3f %6.3f %6.3f" % (b0["D_eff"][i, j]/so,
                                             b1["D_eff"][i, j]/so,
                                             b2["D_eff"][i, j]/so,
                                             b3["D_eff"][i, j]/so)
    print(row)
    if t == T_IB[0]:
        print("        (junctions detected: %d, types %s)"
              % (b2["junction"]["n_junctions"], b2["junction"]["types"]))

# ---- D: flag-off identity ------------------------------------------------
Cref = read_out_C(MSH_OUT)
b0 = build_solid_bundle(M45_YAML)
dmax = np.max(np.abs(b0["D_eff"]/AREA_N - Cref)) / np.max(np.abs(Cref))
print("---- D: flag-off identity vs %s: max rel diff = %.2e ----"
      % (MSH_OUT, dmax))
