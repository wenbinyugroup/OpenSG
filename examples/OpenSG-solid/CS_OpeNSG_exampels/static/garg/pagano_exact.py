"""pagano_exact.py -- THE Pagano exact 3-D solution, SELF-CONTAINED (solver merged in).

The single entry point for the benchmark's reference curves; nothing else in the
benchmark may use it (all inputs come from statics, see statics_fsdt.py).

PROBLEM (Pagano, J. Compos. Mater. 3 (1969) 398-411; restated as Garg Eqs. (18)-(24)):
a laminated strip 0 <= x <= L, 0 <= z <= h (layer 0 at the BOTTOM), infinite across
the width (cylindrical bending, d/dy = 0), simply supported, loaded on the TOP face by

    sigma_33(x, h) = q0 sin(p x),   p = n pi / L,
    sigma_33(x, 0) = sigma_13(x, 0) = sigma_13(x, h) = sigma_23 = 0     (Garg Eq. 21)
    continuity of (u, v, w, sigma_33, sigma_13, sigma_23) at every ply interface (Eq. 22)

METHOD.  State-space (transfer-matrix) restatement of Pagano's construction: with the
single-harmonic separation

    u = U(z) cos(px),  v = V(z) cos(px),  w = W(z) sin(px),
    sigma_13 = X(z) cos(px),  sigma_23 = Y(z) cos(px),  sigma_33 = Z(z) sin(px),

the full 3-D equations reduce EXACTLY (no approximation of any kind, valid at every
L/h) to the linear ODE system  s'(z) = A_k s(z)  per layer k, with state
s = [U, V, W, X, Y, Z].  Each layer propagator is a matrix exponential, so interface
continuity and the face tractions hold to machine precision by construction
(bc_residual ~ 1e-12).  Works for ANY monoclinic ply (Pagano's original f(z) route is
restricted to cross-ply).

Harmonic families (which station carries which quantity):
    sigma_{11,22,33}, w, eps  ~  sin(px)  ->  amplitudes = the values at x = L/2
    sigma_{13,23}, u, v       ~  cos(px)  ->  amplitudes = the values at x = 0

Voigt order everywhere: OpenSG's [11, 22, 33, 23, 13, 12].
"""
import os
import sys

import numpy as np
from scipy.linalg import expm

# repo root = the first ancestor holding opensg_jax/ (location-robust: this
# module also runs from the examples/RM_OpenSG_pagano copy)
_CC = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_CC, "src", "opensg_solid", "rm_plate_1D")):
    _up = os.path.dirname(_CC)
    if _up == _CC:
        raise RuntimeError("rm_plate repo root not found above " + __file__)
    _CC = _up
sys.path.insert(0, os.path.join(_CC, "src"))
from opensg_solid.rm_plate_1D.msg_materials import rotated_stiffness_6x6     # noqa: E402


def _layer_A(C, p):
    """The 6x6 state matrix A of one monoclinic layer, s' = A s.

    Variables
    ---------
    C        (6, 6) rotated ply stiffness in OpenSG Voigt order [11,22,33,23,13,12]
    p        wavenumber n pi / L of the single harmonic [1/m]
    Cij      the individual stiffness entries pulled out of C (C11 ... C66)
    Gs       (2, 2) transverse-shear block [[C55, C45], [C45, C44]]: maps
             [gamma13, gamma23] -> [sigma13, sigma23]
    Gi       (2, 2) its inverse (strains from the stress state variables X, Y)
    Q11,     sigma33-CONDENSED in-plane moduli Qab = Cab - Ca3 Cb3 / C33 -- they
    Q16, Q66 appear because sigma33 (= Z) is a state variable while eps33 is not
    A        (6, 6) the state matrix: rows are the z-derivatives of
             [U, V, W, X, Y, Z] expressed in the state itself:
               U' = -p W + Gi[0,0] X + Gi[0,1] Y        (gamma13 kinematics)
               V' =        Gi[1,0] X + Gi[1,1] Y        (gamma23 kinematics)
               W' = (p C13 U + p C36 V + Z)/C33         (eps33 constitutive)
               X' = p^2 Q11 U + p^2 Q16 V - p C13/C33 Z (x-equilibrium)
               Y' = p^2 Q16 U + p^2 Q66 V - p C36/C33 Z (y-equilibrium)
               Z' = p X                                 (z-equilibrium)

    Returns
    -------
    (A, con) with con = (C11, C12, C13, C16, C22, C23, C26, C33, C36), the entries
    the stress reconstruction needs again later.
    """
    C11, C12, C13, C16 = C[0, 0], C[0, 1], C[0, 2], C[0, 5]
    C22, C23, C26 = C[1, 1], C[1, 2], C[1, 5]
    C33, C36 = C[2, 2], C[2, 5]
    C44, C45, C55 = C[3, 3], C[3, 4], C[4, 4]
    C66 = C[5, 5]

    Gs = np.array([[C55, C45], [C45, C44]])
    Gi = np.linalg.inv(Gs)

    Q11 = C11 - C13 * C13 / C33
    Q16 = C16 - C13 * C36 / C33
    Q66 = C66 - C36 * C36 / C33

    A = np.zeros((6, 6))
    A[0, 2] = -p;            A[0, 3] = Gi[0, 0]; A[0, 4] = Gi[0, 1]
    A[1, 3] = Gi[1, 0];      A[1, 4] = Gi[1, 1]
    A[2, 0] = p * C13 / C33; A[2, 1] = p * C36 / C33; A[2, 5] = 1.0 / C33
    A[3, 0] = p * p * Q11;   A[3, 1] = p * p * Q16;   A[3, 5] = -p * C13 / C33
    A[4, 0] = p * p * Q16;   A[4, 1] = p * p * Q66;   A[4, 5] = -p * C36 / C33
    A[5, 3] = p
    return A, (C11, C12, C13, C16, C22, C23, C26, C33, C36)


class ExactCyl:
    """The exact cylindrical-bending solution of one laminate.

    Constructor variables
    ---------------------
    thick        list of ply thicknesses, layer 0 at the BOTTOM face [m]
    angles_deg   ply fibre angles about the thickness axis [deg]
    mat_names    ply material names (keys of material_db)
    material_db  {name: {"E": [E1,E2,E3], "G": [G12,G13,G23], "nu": [nu12,nu13,nu23]}}
    L            span of the strip [m]
    q0           TOP-face load amplitude: sigma33(x, h) = q0 sin(px) [Pa]
    q_bot        BOTTOM-face load amplitude: sigma33(x, 0) = q_bot sin(px) [Pa].
                 Default 0 (Garg/Pagano: top-loaded only).  Yu-2003 splits the
                 load s3 = b3 = p0/2 -> q0 = +p0/2, q_bot = -p0/2 (the bottom
                 TRACTION b3 acts on the -z normal, so sigma33(0) = -b3)
    n            harmonic number (p = n pi / L); the benchmark uses n = 1

    Internal state built in __init__
    --------------------------------
    self.h       total thickness = sum(thick) [m]
    self.p       wavenumber n pi / L [1/m]
    self.C       list of (6,6) rotated ply stiffnesses
    self.z_bot   (nlay+1,) ply interface positions from the bottom face, z_bot[0] = 0
    self._A      list of per-layer state matrices (from _layer_A)
    self._con    list of per-layer stiffness tuples for stress reconstruction
    self._T      list of CUMULATIVE transfer matrices: _T[k] maps the state at the
                 bottom face (z = 0) to the state at the bottom of layer k
    self.T_tot   transfer matrix over the whole thickness: s(h) = T_tot s(0)
    self.s0      the bottom-face state, fixed by the boundary conditions:
                 bottom shear-free with sigma33 = q_bot ->
                 s0 = [U0, V0, W0, 0, 0, q_bot]; the top conditions
                 X(h) = Y(h) = 0, Z(h) = q0 give the 3x3 system
                 T_tot[3:6, 0:3] [U0,V0,W0] = [0, 0, q0] - T_tot[3:6, 5] q_bot
                 solved for [U0, V0, W0].
    """

    def __init__(self, thick, angles_deg, mat_names, material_db, L, q0=1.0, n=1,
                 q_bot=0.0):
        self.thick = np.asarray(thick, float)
        self.h = float(self.thick.sum())
        self.L = float(L)
        self.q0 = float(q0)
        self.q_bot = float(q_bot)
        self.p = n * np.pi / self.L
        self.C = [rotated_stiffness_6x6(material_db[m]['E'], material_db[m]['G'],
                                        material_db[m]['nu'], a)
                  for m, a in zip(mat_names, angles_deg)]
        self.z_bot = np.concatenate([[0.0], np.cumsum(self.thick)])

        self._A = []
        self._con = []
        self._T = []
        T = np.eye(6)
        for k in range(len(self.thick)):
            Ak, ck = _layer_A(self.C[k], self.p)
            self._A.append(Ak)
            self._con.append(ck)
            self._T.append(T.copy())
            T = expm(Ak * self.thick[k]) @ T
        self.T_tot = T

        M = self.T_tot[3:6, 0:3]
        rhs = np.array([0.0, 0.0, self.q0]) - self.T_tot[3:6, 5] * self.q_bot
        uvw0 = np.linalg.solve(M, rhs)
        self.s0 = np.array([uvw0[0], uvw0[1], uvw0[2], 0.0, 0.0, self.q_bot])

    def _layer_of(self, z):
        """Index k of the layer containing height z (z from the BOTTOM face [m])."""
        k = int(np.searchsorted(self.z_bot[1:-1], z, side='left'))
        return int(np.clip(k, 0, len(self.thick) - 1))

    def state(self, z, layer=None):
        """The state vector s(z) = [U, V, W, X, Y, Z] at height z.

        Variables: z = height from the bottom face [m]; layer = optional layer index
        (resolves the two-sided limit exactly AT an interface); the return is the
        in-layer propagation expm(A_k (z - z_bot[k])) applied to the state at the
        layer bottom, _T[k] s0.
        """
        k = self._layer_of(z) if layer is None else layer
        return expm(self._A[k] * (z - self.z_bot[k])) @ (self._T[k] @ self.s0)

    def stress_amp(self, z, layer=None):
        """Stress AMPLITUDES [S11, S22, S33, S23, S13, S12] at height z.

        sigma_{11,22,33,12} multiply sin(px); sigma_{13,23} multiply cos(px).

        Variables: (U, V, W, X, Y, Z) = the state at z; dW = eps33 from the
        constitutive inversion (p C13 U + p C36 V + Z)/C33; S11/S22/S12 = the in-plane
        stresses from the ply constitutive law with eps11 = -pU, gamma12 = -pV(+...),
        eps33 = dW; the transverse components are the state variables themselves
        (S33 = Z, S23 = Y, S13 = X).
        """
        k = self._layer_of(z) if layer is None else layer
        U, V, W, X, Y, Z = self.state(z, layer=k)
        C11, C12, C13, C16, C22, C23, C26, C33, C36 = self._con[k]
        p = self.p
        dW = (p * C13 * U + p * C36 * V + Z) / C33
        S11 = -p * C11 * U - p * C16 * V + C13 * dW
        S22 = -p * C12 * U - p * C26 * V + C23 * dW
        S12 = -p * C16 * U - p * self.C[k][5, 5] * V + C36 * dW
        return np.array([S11, S22, Z, Y, X, S12])

    def strain_amp(self, z, layer=None):
        """Strain AMPLITUDES [e11, e22, e33, g23, g13, g12] (same sin/cos split).

        Variables: e11 = -pU (from u = U cos); e22 = 0 (cylindrical bending);
        e33 from the constitutive inversion; [g13, g23] = Gs^{-1} [X, Y] (transverse
        shear strains from the stress state variables); g12 = -pV.
        """
        k = self._layer_of(z) if layer is None else layer
        U, V, W, X, Y, Z = self.state(z, layer=k)
        C = self.C[k]
        C13, C33, C36 = C[0, 2], C[2, 2], C[2, 5]
        p = self.p
        e33 = (p * C13 * U + p * C36 * V + Z) / C33
        Gs = np.array([[C[4, 4], C[3, 4]], [C[3, 4], C[3, 3]]])
        g13, g23 = np.linalg.solve(Gs, np.array([X, Y]))
        return np.array([-p * U, 0.0, e33, g23, g13, -p * V])

    def disp_amp(self, z, layer=None):
        """Displacement amplitudes [U, V, W]; u, v multiply cos(px), w sin(px)."""
        return self.state(z, layer=layer)[:3]

    def profile(self, n_per_layer=41, eps=1e-12):
        """Through-thickness sampling honouring the material discontinuities.

        Variables: n_per_layer = sample points per ply; eps = interface offset so
        both one-sided limits are present; zs/lay = the concatenated per-ply grids
        and their layer indices; sig/eps6/uvw = the stacked amplitude rows.

        Returns (zc, sig, eps6, uvw): zc measured from the MID-surface [m];
        sig (n, 6) stress amplitudes; eps6 (n, 6) strain amplitudes; uvw (n, 3).
        """
        zs, lay = [], []
        for k in range(len(self.thick)):
            a, b = self.z_bot[k], self.z_bot[k + 1]
            zz = np.linspace(a + eps * (b - a), b - eps * (b - a), n_per_layer)
            zs.append(zz)
            lay.append(np.full(n_per_layer, k))
        zs = np.concatenate(zs)
        lay = np.concatenate(lay)
        sig = np.array([self.stress_amp(z, layer=k) for z, k in zip(zs, lay)])
        eps6 = np.array([self.strain_amp(z, layer=k) for z, k in zip(zs, lay)])
        uvw = np.array([self.disp_amp(z, layer=k) for z, k in zip(zs, lay)])
        return zs - self.h / 2.0, sig, eps6, uvw

    def bc_residual(self):
        """max |traction BC violation| / max|q| over both faces (should be ~1e-12):
        bottom X = Y = 0, Z = q_bot; top X = Y = 0, Z = q0."""
        sb = self.state(0.0, layer=0)
        st = self.state(self.h, layer=len(self.thick) - 1)
        qref = max(abs(self.q0), abs(self.q_bot))
        return max(abs(sb[3]), abs(sb[4]), abs(sb[5] - self.q_bot),
                   abs(st[3]), abs(st[4]), abs(st[5] - self.q0)) / qref


def pagano_profiles(thick, angles_deg, mat_names, material_db, a=1.0, q0=1.0e4,
                    n_per_layer=81):
    """The exact through-thickness AMPLITUDE profiles (the benchmark entry point).

    Variables: thick/angles_deg/mat_names/material_db = the laminate definition
    (layer 0 at the bottom); a = span [m]; q0 = load amplitude [Pa]; n_per_layer =
    sample points per ply; ex = the ExactCyl solver instance.

    Returns (zc, sig, uvw): zc (n,) from the MID-surface; sig (n, 6) Voigt
    [11, 22, 33, 23, 13, 12]; uvw (n, 3).  Remember the families: sig[:, 4]
    (sigma_13) is the x = 0 profile; sig[:, 0] / sig[:, 2] are the x = a/2 ones.
    """
    ex = ExactCyl(list(thick), list(angles_deg), list(mat_names), material_db, a, q0=q0)
    zc, sig, _, uvw = ex.profile(n_per_layer=n_per_layer)
    return zc, sig, uvw
