"""Why is the m45 ring C44 ratio 2x, not the iso 4x?

The solid's merged lattice wall = tube-k wall (-45 in its own frame) + the
neighbour tube's wall, whose frame has the OPPOSITE normal -> in a common
frame the stack is the antisymmetric [+45/-45] 2t laminate: B16/B26 != 0,
and the effective cylindrical bending  D** = 1/[(D - B A^-1 B)^-1]_KK  is
knocked down.  The shell keeps two INDEPENDENT midline t-walls (no composite
action): bending 2*D_t exactly.

Predictions (frame, both families equal -> D44 = k = 6*EI_eff/span):
  shell: D44 = 12*D_t**/a          solid: D44 = 6*D_2t**/(a+t)
against the measured .out values (material-area norm 0.12).

Run (from this folder):  python ring_m45_c44_check.py
"""
import numpy as np

############### User Input #################################
a, t = 1.0, 0.03
E1, E2, G12, nu12 = 142.0e9, 9.8e9, 6.0e9, 0.30
C44_SHELL_OUT = 1.02865288e7        # square_solid_msg_shell.out (post-fix)
C44_SOLID_OUT = 2.04702777e7        # square_solid_newarch.out
############################################################


def Qbar(th):
    nu21 = nu12*E2/E1
    d = 1 - nu12*nu21
    Q = np.array([[E1/d, nu12*E2/d, 0], [nu12*E2/d, E2/d, 0], [0, 0, G12]])
    c, s = np.cos(np.radians(th)), np.sin(np.radians(th))
    T = np.array([[c*c, s*s, 2*c*s], [s*s, c*c, -2*c*s],
                  [-c*s, c*s, c*c - s*s]])
    R = np.diag([1.0, 1.0, 2.0])
    return np.linalg.inv(T) @ Q @ R @ T @ np.linalg.inv(R)


def abd(plies):
    A = np.zeros((3, 3)); B = np.zeros((3, 3)); D = np.zeros((3, 3))
    z = -sum(tk for _, tk in plies)/2
    for th, tk in plies:
        Q = Qbar(th); z1 = z + tk
        A += Q*(z1 - z); B += Q*(z1**2 - z**2)/2; D += Q*(z1**3 - z**3)/3
        z = z1
    return A, B, D


def d_eff(plies, k_index=1):
    """Effective cylindrical bending about the wall tangent: condense the
    membrane strains (B-coupling) and the other two curvatures."""
    A, B, D = abd(plies)
    Ds = D - B @ np.linalg.inv(A) @ B
    return 1.0/np.linalg.inv(Ds)[k_index, k_index]


D_t = d_eff([(-45.0, t)])                       # one shell wall, own midline
D_2t_anti = d_eff([(+45.0, t), (-45.0, t)])     # merged antisymmetric pair
D_2t_homo = d_eff([(-45.0, t), (-45.0, t)])     # (hypothetical) same-frame pair

sh_pred = 12*D_t/a/0.12
so_pred_anti = 6*D_2t_anti/(a + t)/0.12
so_pred_homo = 6*D_2t_homo/(a + t)/0.12

print("D_t (single -45, condensed)      = %.5e" % D_t)
print("D_2t antisym [+45/-45], condensed = %.5e   (homogeneous 8*D_t = %.5e)"
      % (D_2t_anti, 8*D_t))
print("knockdown D_2t_anti/(8 D_t)       = %.4f\n" % (D_2t_anti/(8*D_t)))
print("  %-34s %14s %14s %8s" % ("C44 (material-area norm)", "predicted",
                                 "measured", "meas/pred"))
print("  %-34s %14.5e %14.5e %8.4f" %
      ("shell: 12 D_t / a", sh_pred, C44_SHELL_OUT, C44_SHELL_OUT/sh_pred))
print("  %-34s %14.5e %14.5e %8.4f" %
      ("solid: 6 D_2t_anti/(a+t)", so_pred_anti, C44_SOLID_OUT,
       C44_SOLID_OUT/so_pred_anti))
print("  %-34s %14.5e %14s" %
      ("solid if same-frame merge (homo)", so_pred_homo, "-"))
print("\nratio solid/shell predicted = %.3f   measured = %.3f"
      % (so_pred_anti/sh_pred, C44_SOLID_OUT/C44_SHELL_OUT))
