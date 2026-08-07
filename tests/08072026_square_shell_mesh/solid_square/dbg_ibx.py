"""Debug: I-beam merged-X corner micro energies vs web-cut extension."""
import numpy as np
import yaml as _yaml

from opensg_shell import junction_micro as jm

d = _yaml.safe_load(open("_jv_ibeam.yaml"))
sections, materials = d["sections"], d["materials"]
stackA = [(0, False), (0, False)]      # merged flange 2t
stackB = [(0, False)]                  # web t

pliesA, tA = jm.stack_plies(stackA, sections, materials)
pliesB, tB = jm.stack_plies(stackB, sections, materials)
_, profA, zsA = jm._wall_farfield(pliesA)
_, profB, zsB = jm._wall_farfield(pliesB)
DshA, GshA = jm.stack_shell_law(stackA, sections, materials, "msg")
DshB, GshB = jm.stack_shell_law(stackB, sections, materials, "msg")
Ls = 5.0*max(tA, tB)
E_shell = jm._shell_patch("X", tA, tB, Ls, 0, 1, [DshA, DshB], [GshA, GshB])

nodes, tris, Ce, cuts, counted, blk = jm._patch_mesh(
    "X", pliesA, tA, pliesB, tB, Ls, 4, fill="A", extA=0.0, extB=0.0)
print("nodes: %d  x range [%.4f, %.4f]  y range [%.4f, %.4f]"
      % (len(nodes), nodes[:, 0].min(), nodes[:, 0].max(),
         nodes[:, 1].min(), nodes[:, 1].max()))
print("tA=%.3f tB=%.3f Ls=%.3f  pliesA=%s  pliesB=%s"
      % (tA, tB, Ls, [tk for _, tk in pliesA], [tk for _, tk in pliesB]))
p1, p2, p3 = nodes[tris[:, 0]], nodes[tris[:, 1]], nodes[tris[:, 2]]
det = ((p2[:, 0]-p1[:, 0])*(p3[:, 1]-p1[:, 1])
       - (p3[:, 0]-p1[:, 0])*(p2[:, 1]-p1[:, 1]))
area = 0.5*np.abs(det)
print("solid material area = %.6f  (expect ~0.052)" % area.sum())
Carr = np.stack(Ce)
raw = float(np.einsum('e,i,eij,j->', area, np.array([1., 0, 0, 0, 0, 0]),
                      Carr, np.array([1., 0, 0, 0, 0, 0])))
print("solid raw eigen E11 (no fluct) = %.4e  (expect ~C11*area ~ 4.9e9)"
      % raw)
E_solid = jm._solve_patch(nodes, tris, Ce, cuts, profA, zsA, profB, zsB, blk)
print("E_solid[0,0] = %.4e" % E_solid[0, 0])
print("E_shell[0,0] = %.4e" % E_shell[0, 0])
print("E_shell full:")
for r in range(3):
    print("  " + " ".join("%12.4e" % E_shell[r, c] for c in range(3)))
