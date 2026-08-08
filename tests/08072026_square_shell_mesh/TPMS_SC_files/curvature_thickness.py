"""Why the thick TPMS sample agrees poorly with the shell model.

1. Discrete Gaussian curvature of the Schwarz-P midsurface (angle defect),
   giving the area-averaged Kbar and a representative radius R = 1/sqrt|Kbar|.
2. The exact sheet relations for a surface with mean curvature H = 0:
        offset area   S(d)/A0 = 1 + K d^2
        sheet volume  V       = A0 t (1 + Kbar t^2/12)
        free area     S_free  = 2 A0 (1 + Kbar t^2/4)
   Solved for the VOLUME-CONSISTENT thickness of each sample, and checked
   against the independently measured free area.
3. The resulting t/R, against which first-order shell error scales.

Run (from this folder):  python curvature_thickness.py"""
import numpy as np

MSH = "schwarz_p_D2_shell.msh"

ln = open(MSH).read().split("\n")
i = ln.index("$Nodes")
nn = int(ln[i+1])
nd = np.array([[float(v) for v in ln[i+2+k].split()[1:4]] for k in range(nn)])
j = ln.index("$Elements")
ne = int(ln[j+1])
tri = []
for k in range(ne):
    p = ln[j+2+k].split()
    ntag = int(p[2])
    tri.append([int(v) for v in p[3+ntag:]])
tri = np.array(tri, int) - 1

p1, p2, p3 = nd[tri[:, 0]], nd[tri[:, 1]], nd[tri[:, 2]]
A_el = 0.5*np.linalg.norm(np.cross(p2-p1, p3-p1), axis=1)
A0 = A_el.sum()

# angle defect: K dA at each vertex = 2pi - sum(incident angles)
ang = np.zeros((len(tri), 3))
for a in range(3):
    u = nd[tri[:, (a+1) % 3]] - nd[tri[:, a]]
    v = nd[tri[:, (a+2) % 3]] - nd[tri[:, a]]
    cu = np.sum(u*v, 1)/(np.linalg.norm(u, axis=1)*np.linalg.norm(v, axis=1))
    ang[:, a] = np.arccos(np.clip(cu, -1, 1))
angsum = np.zeros(nn)
np.add.at(angsum, tri.ravel(), ang.ravel())
Avert = np.zeros(nn)
np.add.at(Avert, tri.ravel(), np.repeat(A_el/3.0, 3))

# boundary vertices sit on the cube faces -> exclude (their defect is not K)
tol = 1e-7
onface = np.zeros(nn, bool)
for ax in range(3):
    onface |= (np.abs(nd[:, ax] - nd[:, ax].min()) < tol) | \
              (np.abs(nd[:, ax] - nd[:, ax].max()) < tol)
interior = ~onface
KdA = 2*np.pi - angsum
tot_KdA = KdA[interior].sum()
A_int = Avert[interior].sum()
Kbar = tot_KdA/A_int
R = 1.0/np.sqrt(abs(Kbar))

print("Schwarz-P midsurface mesh: %d nodes, %d tris, A0 = %.4f"
      % (nn, len(tri), A0))
print("  interior vertices %d (%.1f%% of area); int K dA = %.3f"
      % (interior.sum(), 100*A_int/A0, tot_KdA))
print("  Kbar = %.3f  ->  representative R = 1/sqrt|Kbar| = %.4f\n"
      % (Kbar, R))


def t_from_volume(V):
    """solve V = A0 t (1 + Kbar t^2/12) for t"""
    t = V/A0
    for _ in range(60):
        t = V/(A0*(1 + Kbar*t*t/12.0))
    return t


print("  %-10s %8s %10s %10s %10s %8s %8s"
      % ("sample", "V", "t=V/A0", "t (V-cons)", "S/2 pred", "S/2 meas",
         "t/R"))
for nm, V, S2meas in (("Sample_1", 0.300000, 2.2440),
                      ("Sample_2", 0.085694, 2.3448)):
    t0 = V/A0
    t = t_from_volume(V)
    S2 = A0*(1 + Kbar*t*t/4.0)
    print("  %-10s %8.5f %10.5f %10.5f %10.4f %8.4f %8.3f"
          % (nm, V, t0, t, S2, S2meas, t/R))
print("\n  (S/2 pred vs meas validates the uniform-offset sheet assumption)")
