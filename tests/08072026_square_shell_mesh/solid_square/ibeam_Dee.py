"""I-beam Dee: analytical line integration vs numerical msg_shell vs solid.

Center-ref I-section, iso: two flanges (width b, at y3 = +-h/2, tangent y2)
and one web (height h, at y2 = 0, tangent y3), thickness t everywhere.
Wall lengths: flanges 2b total, web h -- an ASYMMETRIC wall census, unlike the
square, so every family-attribution shows up as D22 != D33.

Analytical (current 8-strain formulation; Lf = 2b flange total, Lw = h web):
  D11 = A11 (Lf+Lw)      D12 = A12 Lf        D13 = A12 Lw      D23 = 0
  D22 = A11 Lf           D33 = A11 Lw
  D44 = Gm (Lf+Lw)       D55 = A66 Lw + Gm Lf     D66 = A66 Lf + Gm Lw

Predicted with the e_nn + t*C completion:  Dee = A_mat * C0  (every term).
Solid:  Dee = A_mat * C0,  A_mat = (Lf+Lw)*t  (midline census).

Dee involves no fluctuation, so it is BC-independent; dof_map is identity here.

Run (from this folder):  python ibeam_Dee.py
"""
import numpy as np
import yaml as _yaml

from opensg_shell.oml_ring import load_ring_ref
from opensg_shell.solid_props import assemble_solid_macro
from opensg_shell.emit_abd import material_db_from_yaml
from opensg_solid.rm_plate_1D.msg_rm_plate import rm_plate_msg

############### User Input #################################
b, h, t = 0.5, 1.0, 0.03          # flange width, web height, thickness
E, nu = 70.0e9, 0.30
nseg = 10                          # elements per half-flange and per half-web
############################################################

G = E/(2*(1+nu))
Ep = E/(1-nu**2)
A11, A12, A66 = Ep*t, nu*Ep*t, G*t
mat = {"name": "iso", "density": 2700.0,
       "elastic": {"E": [E]*3, "G": [G]*3, "nu": [nu]*3}}
Gm = float(np.asarray(rm_plate_msg([t], [0.0], ["iso"],
                                   material_db_from_yaml([mat]),
                                   fraction=0.5)["G_msg"])[0, 0])
Lf, Lw = 2*b, h

# ---- analytical (current formulation) --------------------------------------
Dee_an = np.zeros((6, 6))
Dee_an[0, 0] = A11*(Lf+Lw)
Dee_an[1, 1] = A11*Lf
Dee_an[2, 2] = A11*Lw
Dee_an[0, 1] = Dee_an[1, 0] = A12*Lf
Dee_an[0, 2] = Dee_an[2, 0] = A12*Lw
Dee_an[3, 3] = Gm*(Lf+Lw)
Dee_an[4, 4] = A66*Lw + Gm*Lf
Dee_an[5, 5] = A66*Lf + Gm*Lw

# ---- 1-D yaml: two flanges + web, junction nodes shared --------------------
def seg(p0, p1, n):
    return [p0 + (p1-p0)*k/n for k in range(n)] + [p1] if False else \
           [p0 + (p1-p0)*k/n for k in range(n+1)]

pts, cells, tans = [], [], []
idx = {}


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


add_wall([-b/2,  h/2], [0.0,  h/2], nseg)   # top flange left half
add_wall([0.0,  h/2], [b/2,  h/2], nseg)    # top flange right half
add_wall([-b/2, -h/2], [0.0, -h/2], nseg)   # bottom flange halves
add_wall([0.0, -h/2], [b/2, -h/2], nseg)
add_wall([0.0, -h/2], [0.0,  h/2], 2*nseg)  # web (shares junction nodes)

pts = np.array(pts)
o = ["nodes:"]
for x, y in pts:
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
open("_ibeam.yaml", "w").write("\n".join(o) + "\n")

# ---- numerical msg_shell Dee -----------------------------------------------
R = load_ring_ref("_ibeam.yaml", "center")
d_sh = _yaml.safe_load(open("_ibeam.yaml"))
G_by = list(R["G_by"])
G_by[0] = np.asarray(rm_plate_msg([t], [0.0], ["iso"],
                                  material_db_from_yaml(d_sh["materials"]),
                                  fraction=0.5)["G_msg"])
rx, rcells = R["rx"], R["cells"]
m = len(rx)
h_st = float(np.mean(np.linalg.norm(rx[rcells[:, 1]] - rx[rcells[:, 0]], axis=1)))
ez = np.zeros(3); ez[R["ax"]] = 1.0
nodes_st = np.vstack([rx, rx + h_st*ez])
dof_map = np.concatenate([np.arange(m), np.arange(m)])
quads = np.array([[q1, q2, m+q2, m+q1] for q1, q2 in rcells], int)
_, Dee_sh = assemble_solid_macro(nodes_st, quads, R["rsub"],
                                 np.asarray(R["re3"]), R["D_by"], G_by,
                                 R["cross"], R["ax"], dof_map=dof_map,
                                 shear="mitc4_g23")
Dee_sh = Dee_sh / h_st

# ---- solid -------------------------------------------------------------------
lam = E*nu/((1+nu)*(1-2*nu)); mu = G
C0 = np.zeros((6, 6)); C0[:3, :3] = lam
C0[np.arange(3), np.arange(3)] = lam + 2*mu
C0[3, 3] = C0[4, 4] = C0[5, 5] = mu
A_mat = (Lf+Lw)*t
Dee_so = A_mat*C0

print("I-beam b=%.2f h=%.2f t=%.3f iso;  Lf=%.2f Lw=%.2f  A_mat=%.4f"
      % (b, h, t, Lf, Lw, A_mat))
print("A11=%.5e A12=%.5e A66=%.5e Gm=%.5e\n" % (A11, A12, A66, Gm))
print("  %-5s %13s %13s %9s %13s %8s %13s"
      % ("term", "analytical", "shell num", "an/sh", "solid", "an/so",
         "with e_nn"))
thr = 1e-6*np.max(np.abs(Dee_so))
for i in range(6):
    for j in range(i, 6):
        an, sh, so = Dee_an[i, j], Dee_sh[i, j], Dee_so[i, j]
        if abs(an) > thr or abs(sh) > thr or abs(so) > thr:
            rs = an/sh if abs(sh) > thr else np.inf
            ro = an/so if abs(so) > thr else 0.0
            enn = so if i < 3 and j < 3 else an     # e_nn completion: normal
            print("  D%d%d   %13.5e %13.5e %9.6f %13.5e %8.4f %13.5e"
                  % (i+1, j+1, an, sh, rs, so, ro, enn))
