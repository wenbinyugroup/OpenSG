"""Symbolic (CAS) integration of Dee over the four square edges, with the
closed-form integrand of every entry -- in particular D23, showing what
cancels.

A wall at angle phi in the cross-section plane has (prismatic identities
X11 = 1, X21 = X31 = 0, X12 = 0, C31 = 0):

    tangent  X22 = cos(phi),  X32 = sin(phi)
    normal   C32 = -sin(phi), C33 = cos(phi)

Gamma_e is built symbolically from the derived rows, K = ABDG (A block,
B = 0 center ref, D block, G = diag(Gm, Gm)), and

    M(phi) = Gamma_e^T K Gamma_e        (per unit wall length)
    Dee    = sum_edges  a * M(phi_edge),   phi_edge in {0, pi/2, pi, 3pi/2}

Run (from this folder):  python symbolic_Dee_square.py
"""
import sympy as sp

phi, a = sp.symbols("phi a", real=True, positive=True)
A11, A12, A66, Gm = sp.symbols("A11 A12 A66 Gm", positive=True)
D11d, D12d, D66d = sp.symbols("D11 D12 D66", positive=True)

c, s = sp.cos(phi), sp.sin(phi)
X11, X21, X31 = 1, 0, 0
X12, X22, X32 = 0, c, s
C31, C32, C33 = 0, -s, c

# ---- Gamma_e (8x6), columns [G11 G22 G33 2G23 2G13 2G12] -------------------
Ge = sp.zeros(8, 6)
row = [X11**2, X21**2, X31**2, X31*X21, X11*X31, X11*X21]          # eps11
for j in range(6):
    Ge[0, j] = row[j]
row = [X12**2, X22**2, X32**2, X32*X22, X32*X12, X12*X22]          # eps22
for j in range(6):
    Ge[1, j] = row[j]
row = [2*X11*X12, 2*X22*X21, 2*X31*X32,                            # 2eps12
       X22*X31 + X32*X21, X12*X31 + X32*X11, X12*X21 + X11*X22]
for j in range(6):
    Ge[2, j] = row[j]
# rows 3-5: K11, K22, K12+K21 -> zero macro columns
row = [C31*X11, C32*X21, C33*X31,                                  # 2g13
       X31*C32 + X21*C33, X31*C31 + X11*C33, X21*C31 + X11*C32]
for j in range(6):
    Ge[6, j] = row[j]
row = [C31*X12, C32*X22, C33*X32,                                  # 2g23
       X32*C32 + X22*C33, X32*C31 + X12*C33, X22*C31 + X12*C32]
for j in range(6):
    Ge[7, j] = row[j]

# ---- wall law K = ABDG ------------------------------------------------------
K = sp.zeros(8, 8)
K[0, 0] = K[1, 1] = A11
K[0, 1] = K[1, 0] = A12
K[2, 2] = A66
K[3, 3] = K[4, 4] = D11d
K[3, 4] = K[4, 3] = D12d
K[5, 5] = D66d
K[6, 6] = K[7, 7] = Gm

M = sp.simplify(Ge.T * K * Ge)          # per unit wall length, function of phi

print("integrand per unit wall length, entry by entry (nonzero only):")
lbl = ["G11", "G22", "G33", "2G23", "2G13", "2G12"]
for i in range(6):
    for j in range(i, 6):
        e = sp.factor(sp.trigsimp(M[i, j]))
        if e != 0:
            print("  M[%s,%s] = %s" % (lbl[i], lbl[j], e))

print("\nD23 integrand (the G22-G33 entry):")
d23 = sp.factor(sp.trigsimp(M[1, 2]))
print("  M[G22,G33](phi) = %s" % d23)
print("  membrane path:  eps22(G22)*A22*eps22(G33) = %s"
      % sp.trigsimp(Ge[1, 1]*A11*Ge[1, 2]))
print("  shear path:     2g23(G22)*Gm*2g23(G33)    = %s"
      % sp.trigsimp(Ge[7, 1]*Gm*Ge[7, 2]))
print("  at the square's edges phi = 0, pi/2, pi, 3pi/2:"
      " sin(phi)*cos(phi) = 0  ->  each edge contributes 0")

# ---- sum over the four edges ------------------------------------------------
Dee = sp.zeros(6, 6)
for ph in (0, sp.pi/2, sp.pi, 3*sp.pi/2):
    Dee += a * M.subs(phi, ph)
Dee = sp.simplify(Dee)

print("\nDee (symbolic, four edges of side a):")
for i in range(6):
    for j in range(i, 6):
        if Dee[i, j] != 0:
            print("  D%d%d = %s" % (i+1, j+1, sp.factor(Dee[i, j])))

# ---- numbers ----------------------------------------------------------------
E, nu, t = 70.0e9, 0.30, 0.03
Ep = E/(1-nu**2); G = E/(2*(1+nu))
subs = {a: 1.0, A11: Ep*t, A12: nu*Ep*t, A66: G*t, Gm: 0.880537*G*t,
        D11d: Ep*t**3/12, D12d: nu*Ep*t**3/12, D66d: G*t**3/12}
print("\nnumeric check (a=1, t=0.03, iso):")
for i in range(6):
    for j in range(i, 6):
        v = float(Dee[i, j].subs(subs))
        if abs(v) > 1.0:
            print("  D%d%d = %.6e" % (i+1, j+1, v))
