"""Which shell strain channel produces each stiffness entry?

For the square cross cell, decompose every resolved C_ij into the wall-law
channels, using the converged solution:

    e_p(mode) = Gamma_e[:, mode] + Gamma_h V0[:, mode]        (8 rows)
    C_ij = (1/w) int [ e_m(i)^T A e_m(j)  +  e_m(i)^T B k(j) + k(i)^T B e_m(j)
                      + k(i)^T D k(j)     +  g(i)^T G g(j) ] dA

rows 1-3 = e_m (membrane), 4-6 = k (curvature), 7-8 = g (transverse shear).
The channel columns must sum to the C3D entry (check printed).

Run (from this folder):  python strain_channels.py
"""
import numpy as np

from opensg_shell import build_solid_bundle
from opensg_shell.sg_homo import (solid_fluct_ops_batch,
                                      solid_macro_ops_batch, NDOF6)

############### User Input #################################
L, t_w = 1.0, 0.10
E, nu = 70.0e9, 0.30
nseg = 40
############################################################

G_iso = E/(2*(1+nu))
c = L/2

# ---- cross-cell shell yaml (same builder as the sweep) ---------------------
xs = np.linspace(0.0, L, nseg+1)
h_n = np.stack([xs, np.full(nseg+1, c)], 1)
v_n = np.stack([np.full(nseg+1, c), xs], 1)
ic = nseg//2
keep = [j for j in range(nseg+1) if j != ic]
pts = np.vstack([h_n, v_n[keep]])
vid, kk = {}, len(h_n)
for j in range(nseg+1):
    vid[j] = ic if j == ic else kk
    if j != ic:
        kk += 1
cells = ([[i, i+1] for i in range(nseg)]
         + [[vid[j], vid[j+1]] for j in range(nseg)])
ori = ([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]*nseg
       + [[0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0]]*nseg)
o = ["nodes:"]
for x, y in pts:
    o.append("- [%.12f %.12f 0.00000000]" % (x, y))
o.append("elements:")
for a_, b_ in cells:
    o.append("- [%d %d]" % (a_+1, b_+1))
o.append("elementOrientations:")
for r in ori:
    o.append("- [" + ", ".join("%.1f" % v for v in r) + "]")
o += ["sections:", "- type: shell", "  elementSet: layup_0", "  layup:",
      "  - - iso", "    - %.6e" % t_w, "    - 0.0",
      "materials:", "- name: iso", "  density: 2700.0", "  elastic:",
      "    E: [%.6e, %.6e, %.6e]" % (E, E, E),
      "    G: [%.6e, %.6e, %.6e]" % (G_iso, G_iso, G_iso),
      "    nu: [%.6f, %.6f, %.6f]" % (nu, nu, nu),
      "sets:", "  element:", "  - name: layup_0", "    labels:"]
o += ["    - %d" % (e+1) for e in range(len(cells))]
o.append("reference: center")
open("_chan.yaml", "w").write("\n".join(o) + "\n")

B = build_solid_bundle("_chan.yaml", cell_area=L*L)
C3D = np.asarray(B["C3D"])
V0 = np.asarray(B["V0"])                       # (M, 6)
rx = B["rx3"]; rcells = B["red_cells"]
ax, cross = B["ax"], B["cross"]
rsub = B["rsub"]; re3 = B["re3"]
D_by = {int(k): v for k, v in enumerate([np.asarray(d) for d in
        ([B[k] for k in ("D_by",)][0] if "D_by" in B else [])])} \
    if "D_by" in B else None

# rebuild the strip exactly as ring_solid does
from opensg_shell.sg_periodicity import mesh_to_periodic_sparse_assembly_map
from opensg_shell.sg_mesh import load_ring_ref
R = load_ring_ref("_chan.yaml", "center")
from opensg_solid.rm_plate_1D.msg_rm_plate import rm_plate_msg
from opensg_shell.sg_materials import material_db_from_yaml
import yaml as _yaml
d_sh = _yaml.safe_load(open("_chan.yaml"))
_mdb = material_db_from_yaml(d_sh["materials"])
G_by = list(R["G_by"])
for si, sec in enumerate(d_sh["sections"]):
    _pl = [[str(p[0]), float(p[1]), float(p[2])] for p in sec["layup"]]
    _rr = rm_plate_msg([p[1] for p in _pl], [p[2] for p in _pl],
                       [p[0] for p in _pl], _mdb, fraction=0.5)
    if _rr["G_msg"] is not None:
        G_by[si] = np.asarray(_rr["G_msg"])
rx, rcells = R["rx"], R["cells"]
m = len(rx)
h_st = float(np.mean(np.linalg.norm(rx[rcells[:, 1]] - rx[rcells[:, 0]], axis=1)))
ez = np.zeros(3); ez[R["ax"]] = 1.0
nodes_st = np.vstack([rx, rx + h_st*ez])
rc, _ = mesh_to_periodic_sparse_assembly_map(m, np.arange(m)[:, None],
                                             rx[:, R["cross"]], 3, NDOF6)
nm = np.asarray(rc, int).ravel()
dof_map = np.concatenate([nm, nm])
quads = np.array([[a, b, m+b, m+a] for a, b in rcells], int)
Xe = nodes_st[quads]; e3e = np.asarray(R["re3"])
sd = np.asarray(R["rsub"], int)
keys = sorted(set(int(s) for s in sd))
Darr = np.stack([np.asarray(R["D_by"][k], float) for k in keys])
Garr = np.stack([np.asarray(G_by[k], float) for k in keys])
pos = {k: i for i, k in enumerate(keys)}
sdi = np.array([pos[int(s)] for s in sd])
De = Darr[sdi]; Gm = Garr[sdi]
ne = len(quads)
g = (NDOF6*dof_map[quads])[:, :, None] + np.arange(NDOF6)[None, None, :]
g = g.reshape(ne, 24)

# channel-resolved energies
chan = {k: np.zeros((6, 6)) for k in ("A_mem", "B_coup", "D_bend", "G_shear")}
gpv = 1.0/np.sqrt(3.0)
for (xi, eta) in [(-gpv, -gpv), (gpv, -gpv), (gpv, gpv), (-gpv, gpv)]:
    BDh, BGh, _, dA = solid_fluct_ops_batch(Xe, e3e, xi, eta, R["cross"], R["ax"])
    BDe6, BGe6, _ = solid_macro_ops_batch(Xe, e3e, xi, eta, R["cross"], R["ax"])
    Vg = V0[g]                                   # (ne, 24, 6)
    em = BDe6 + np.einsum('era,eam->erm', BDh, Vg)   # (ne, 6, 6) rows 1-6
    gm = BGe6 + np.einsum('era,eam->erm', BGh, Vg)   # (ne, 2, 6) rows 7-8
    mem, kap = em[:, 0:3, :], em[:, 3:6, :]
    A_, B_, D_ = De[:, 0:3, 0:3], De[:, 0:3, 3:6], De[:, 3:6, 3:6]
    w = dA
    chan["A_mem"] += np.einsum('e,eri,ers,esj->ij', w, mem, A_, mem)
    chan["B_coup"] += (np.einsum('e,eri,ers,esj->ij', w, mem, B_, kap)
                       + np.einsum('e,eri,ers,esj->ij', w, kap,
                                   np.transpose(B_, (0, 2, 1)), mem))
    chan["D_bend"] += np.einsum('e,eri,ers,esj->ij', w, kap, D_, kap)
    chan["G_shear"] += np.einsum('e,eri,ers,esj->ij', w, gm, Gm, gm)

for k in chan:
    chan[k] = chan[k]/h_st/(L*L)
tot = sum(chan.values())

print("cross cell  L=%.2f t=%.2f iso  (channel sums vs C3D check: max diff %.2e)"
      % (L, t_w, np.max(np.abs(tot - C3D))))
print("\n  %-5s %13s | %13s %13s %13s %13s" %
      ("term", "C3D", "membrane A", "coupling B", "bending D", "shear G"))
for (i, j) in [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (3, 3), (4, 4), (5, 5)]:
    print("  C%d%d   %13.4e | %13.4e %13.4e %13.4e %13.4e"
          % (i+1, j+1, C3D[i, j], chan["A_mem"][i, j], chan["B_coup"][i, j],
             chan["D_bend"][i, j], chan["G_shear"][i, j]))
