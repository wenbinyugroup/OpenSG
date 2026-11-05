"""The solid-beam (n_model 1) D route: the generalized-Timoshenko recovery
of the msg-shell procedure (sg_dehom.compute_fluctuations_gpu), seeded with
the beam FORCE and differentiated by the FIXED R1 = recov(e1).

Before the fix the chain was seeded with C^-2 F (sg_dehom.py:417 at
f31b196) and a load-dependent operator recov(st + e1) with R2/R3
product-rule terms: F2/F3 recovered 1/GA^2 of the applied shear and the
field was not linear in the load.  The gates below are exact identities of
the procedure (zeros ==, round-off on the resultants) or second-order
convergence against the exact nu = 0 parabola -- never a tuned bound.
T6b pins what the procedure (VABS 4.1's and msg-shell's) does when the
section origin moves -- the field changes by exactly the Delta F_cl state
of each origin -- and an xfail(strict=True) test states what 3-D theory
requires instead: one field about every origin.
"""
import os
import struct

import numpy as np
import pytest

assert struct.calcsize("P") == 8

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

import beam_sg_fixtures as fx  # noqa: E402
from opensg_solid.sg_dehom import (  # noqa: E402
    beam_expected_resultants, build_recovery_matrix, compute_fluctuations_gpu,
    dehom_fields, recover_gauss_point_fields)

CL = [0, 3, 4, 5]
Q = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, -1.0], [1.0, 0.0]])
B2, H3 = 1.0, 0.5                       # the T3 rectangle, y2 x y3 [m]
# T3's exact single-mesh deviation: at nu = 0 the warping is the cubic of
# the shear coordinate alone and the quad9 solution its nodal quadratic
# interpolant (1-D nodal exactness), so tau_h - tau = -(4 tau0/(3 L^2))
# (3 t^2 - h_e^2/4) in each element (t from its centre); at the outer
# 4-point Gauss abscissa xi^2 = (3 + 2 sqrt(6/5))/7 the deviation is
# (3 xi^2 - 1)/3 (h_e/L)^2 tau0 = 0.408222 (h_e/L)^2 tau0
C_GP4 = (2.0 + 6.0 * np.sqrt(1.2)) / 21.0
COUPLED_AT = np.array([0.13, -0.07])    # the coupled origin, centre's axes


def _rect(tmp_path_factory, name, nu, n2=40, n3=20, refined=1):
    d = tmp_path_factory.mktemp(name)
    nodes, cells = fx.rect_quad9(B2, H3, n2, n3)
    p = fx.write_yaml(os.path.join(str(d), name + ".yaml"), nodes, cells,
                      np.ones(len(cells), int), {1: fx.iso(200e9, nu)},
                      refined=refined)
    return fx.homogenize(p)


@pytest.fixture(scope="module")
def rect0(tmp_path_factory):
    return _rect(tmp_path_factory, "rect_nu0", 0.0)


@pytest.fixture(scope="module")
def rect0_fine(tmp_path_factory):
    return _rect(tmp_path_factory, "rect_nu0_fine", 0.0, 80, 40)


@pytest.fixture(scope="module")
def rect03(tmp_path_factory):
    return _rect(tmp_path_factory, "rect_nu03", 0.3)


def _coupled_section(tmp_path_factory, name, center):
    """0.4 x 0.2 m rectangle, quad9 16 x 8, centred at `center` (the
    section origin at -center from the centre); glass-epoxy at +30
    degrees in the upper half and at -45 degrees in the lower half.  The
    element order does not depend on `center`."""
    d = tmp_path_factory.mktemp(name)
    nodes, cells = fx.rect_quad9(0.4, 0.2, 16, 8, center=center)
    up = nodes[cells[:, 8], 1] > center[1]
    p = fx.write_yaml(os.path.join(str(d), name + ".yaml"), nodes, cells,
                      np.where(up, 1, 2),
                      {1: fx.ortho(**fx.GLASS, angle=30.0),
                       2: fx.ortho(**fx.GLASS, angle=-45.0)})
    return fx.homogenize(p)


@pytest.fixture(scope="module")
def coupled(tmp_path_factory):
    """The section origin at (y2, y3) = (0.13, -0.07) from the centre:
    every coupling of the 6x6 is live and Delta F_cl is 2.4 % of max|F|.
    (One glass at 30 degrees over the whole rectangle is not enough:
    there Delta F_cl = 2e-11 N, B - B^T has no part in the F1/M1 rows of
    a homogeneous section.)"""
    return _coupled_section(tmp_path_factory, "coupled", -COUPLED_AT)


@pytest.fixture(scope="module")
def coupled_centred(tmp_path_factory):
    """The same section meshed about its centre (the same elements in the
    same order, every node moved by COUPLED_AT)."""
    return _coupled_section(tmp_path_factory, "coupled_centred", (0.0, 0.0))


def _chain(r, F, V0=None, V1=None):
    """compute_fluctuations_gpu as dehom_fields seeds it from a force."""
    C = np.asarray(r["C_eff"], float)
    st = np.linalg.solve(C, F)
    out = compute_fluctuations_gpu(
        jnp.asarray(C), jnp.asarray(st),
        r["V0"] if V0 is None else V0, r["V1s"] if V1 is None else V1,
        jnp.asarray(np.asarray(F, float)))
    return [np.asarray(x) for x in out], st


def _rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.abs(a - b).max() / np.abs(b).max())


def _delta_F_cl(r, FF):
    """Delta F_cl = -A (I - QQ^T) A^-1 B A^-1 Q F_s: the classical
    resultants the generalized-Timoshenko field carries beyond the
    applied [F1 M1 M2 M3] under F_s = [F2 F3] (exact for the procedure;
    A = C_eff_EB, B = B_tim of the Timoshenko reduction)."""
    A = np.asarray(r["C_eff_EB"], float)
    B = np.asarray(r["B_tim"], float)
    Ai = np.linalg.inv(A)
    P = np.eye(4) - Q @ Q.T
    return -A @ P @ Ai @ B @ Ai @ Q @ np.asarray(FF, float)[[1, 2]]


# ------------------------------------------------------------------ T1, T2
def test_beam_macro_recovery_matches_shell():
    """T1: the solid chain IS the msg-shell _macro_recovery.  With V0 = V1
    = I and F0 = C st the four fields are (eps, eps', eps'', eps'); 20
    seeded SPD laws of condition 1e2..1e8."""
    from opensg_shell.kernels.msg_dehom import _macro_recovery, _recov
    R1 = np.asarray(build_recovery_matrix(jnp.asarray(np.eye(6)[0])))
    assert np.count_nonzero(R1) == 2
    assert R1[4, 2] == 1.0 and R1[5, 1] == -1.0
    assert np.array_equal(R1 @ R1, np.zeros((6, 6)))
    rng = np.random.default_rng(20261103)
    I4 = jnp.eye(4)
    for t in range(20):
        s = rng.standard_normal(6)
        assert np.array_equal(np.asarray(build_recovery_matrix(
            jnp.asarray(s))), _recov(s))
        cond = 10.0 ** (2.0 + 6.0 * t / 19.0)
        Qm, _ = np.linalg.qr(rng.standard_normal((6, 6)))
        C = (Qm * (np.logspace(0.0, np.log10(cond), 6) * 1e7)) @ Qm.T
        C = 0.5 * (C + C.T)
        st = rng.standard_normal(6) * 1e-4
        _, st_m, cl1, cl2 = _macro_recovery(C, st)
        w_1, w1s_1, w1s_2, w_2, eps = [np.asarray(x) for x in
                                       compute_fluctuations_gpu(
                                           jnp.asarray(C), jnp.asarray(st),
                                           I4, I4, jnp.asarray(C @ st))]
        for got, ref in ((w_1, st_m), (eps, st_m), (w1s_1, cl1),
                         (w_2, cl1)):
            assert np.abs(got - ref).max() <= 1e-13 * np.abs(ref).max()
        assert np.array_equal(cl2, np.zeros(4))
        assert np.array_equal(w1s_2, np.zeros(4))          # eps'' == 0
        # st and F0 must be one state: a solve passes, a mismatch is refused
        F0 = np.random.default_rng(t).standard_normal(6) * 1e3
        compute_fluctuations_gpu(jnp.asarray(C), jnp.asarray(
            np.linalg.solve(C, F0)), I4, I4, jnp.asarray(F0))
        with pytest.raises(ValueError, match="not one state"):
            compute_fluctuations_gpu(jnp.asarray(C), jnp.asarray(st), I4, I4,
                                     jnp.asarray(F0))


def test_beam_macro_recovery_linear_and_exact(rect03):
    """T2: pure F1/M1/M2/M3 leave the first-order fields exactly zero and
    eps = st[0,3,4,5]; pure F2/F3 give eps' = A^-1 [0, 0, F3, -F2]; the
    chain is linear (10 F) and superposes (Fa + Fb)."""
    r = rect03
    A = np.asarray(r["C_eff_EB"], float)
    I4 = jnp.eye(4)
    for k in (0, 3, 4, 5):
        F = np.zeros(6)
        F[k] = 1e3
        (w_1, w1s_1, w1s_2, w_2, eps), st = _chain(r, F)
        for z in (w1s_1, w1s_2, w_2):
            assert np.array_equal(z, np.zeros_like(z))
        assert np.array_equal(eps, st[CL])
    for k in (1, 2):
        F = np.zeros(6)
        F[k] = 1e3
        (_, _, w1s_2, eps1, _), _ = _chain(r, F, I4, I4)
        ref = np.linalg.solve(A, np.array([0.0, 0.0, F[2], -F[1]]))
        assert _rel(eps1, ref) <= 1e-12
        assert np.array_equal(w1s_2, np.zeros(4))
    rng = np.random.default_rng(7)
    Fa = rng.standard_normal(6) * np.array([1e5, 1e3, 1e3, 1e3, 1e3, 1e3])
    Fb = rng.standard_normal(6) * np.array([1e5, 1e3, 1e3, 1e3, 1e3, 1e3])
    a, _ = _chain(r, Fa)
    b, _ = _chain(r, Fb)
    ab, _ = _chain(r, Fa + Fb)
    a10, _ = _chain(r, 10.0 * Fa)
    for i in (0, 1, 3, 4):                      # w_1, w1s_1, w_2, eps
        assert _rel(a10[i], 10.0 * a[i]) <= 1e-14
        assert _rel(ab[i], a[i] + b[i]) <= 1e-14


# ---------------------------------------------------------------------- T3
def _shear_check(r, k):
    """Recover the unit-direction shear force 1000 N (k = 1: F2, k = 2:
    F3) on the nu = 0 rectangle; return the field, its resultants, the
    zeroth-order part's resultants and the pointwise deviation from the
    exact parabola (true section span) at the Gauss points."""
    FF = np.zeros(6)
    FF[k] = 1e3
    G, S, _ = dehom_fields(r, beam_force=FF)
    (w_1, _, _, _, eps), _ = _chain(r, FF)
    z = jnp.zeros_like(jnp.asarray(w_1))
    _, _, S0 = recover_gauss_point_fields(
        jnp.asarray(w_1), z, z, z, jnp.asarray(eps), r["dehomo_data"],
        r["C_ess"], r["reduced_periodic_cells"], r["phi_qn"], r["x_end"],
        1, r["n_sg"])
    A = float(np.asarray(r["dehomo_data"]["dV_q"]).sum())
    tau0 = 1.5 * FF[k] / A
    comp, ax, L = (5, 0, B2) if k == 1 else (4, 1, H3)
    c = fx.gauss_xy(r)[..., ax]
    dev = np.abs(S[..., comp] - tau0 * (1.0 - (2.0 * c / L) ** 2)).max()
    return dict(FF=FF, S=S, R=fx.resultants(r, S),
                R0=fx.resultants(r, np.asarray(S0)), tau0=tau0, comp=comp,
                dev=dev / tau0, area=A)


@pytest.mark.parametrize("k", [1, 2])
def test_beam_dehom_rectangle_shear_nu0(rect0, k):
    """T3, the reported symptom: F2 (or F3) = 1000 N on the isotropic
    nu = 0 rectangle 1.0 x 0.5 m (quad9 40 x 20).  The shear resultant
    closes, the other five vanish, the zeroth-order part carries no
    shear, every other stress component is zero, and the shear stress
    is the exact parabola 1.5 V/A (1 - (2 y/L)^2) at the Gauss points up
    to the quadratic-interpolation deviation C_GP4 (h_e/L)^2 tau0 of the
    cubic warping, to round-off (v2.1.0 recovered 5.76e-22 of the
    force)."""
    o = _shear_check(rect0, k)
    assert abs(o["area"] - B2 * H3) <= 1e-12 * B2 * H3
    R, R0 = o["R"], o["R0"]
    assert abs(R[k] - 1e3) <= 1e-10 * 1e3
    others = [i for i in range(6) if i != k]
    assert np.abs(R[others]).max() <= 1e-10 * 1e3
    assert np.abs(R0[[1, 2]]).max() <= 1e-10 * 1e3          # identity (i)
    rest = [i for i in range(6) if i != o["comp"]]
    assert np.abs(o["S"][..., rest]).max() <= 1e-12 * o["tau0"]
    h_e = (B2 / 40.0) if k == 1 else (H3 / 20.0)
    L = B2 if k == 1 else H3
    print("T3 k=%d: maxdev/tau0 = %.10e = %.10f (h_e/L)^2, C_GP4 %.10f"
          % (k, o["dev"], o["dev"] / (h_e / L) ** 2, C_GP4))
    assert abs(o["dev"] - C_GP4 * (h_e / L) ** 2) <= 1e-12


@pytest.mark.parametrize("k", [1, 2])
def test_beam_dehom_rectangle_shear_nu0_converges(rect0, rect0_fine, k):
    """T3 twin: halving h divides the pointwise deviation from the
    parabola by 4 (quad9 stresses at the 4 x 4 Gauss points, second
    order: 0.25 from C_GP4 (h_e/L)^2).  A first-order defect (ratio 0.5)
    fails the 0.3 gate."""
    e_h = _shear_check(rect0, k)["dev"]
    o = _shear_check(rect0_fine, k)
    assert abs(o["R"][k] - 1e3) <= 1e-10 * 1e3
    print("T3 twin k=%d: %.6e -> %.6e, ratio %.4f"
          % (k, e_h, o["dev"], o["dev"] / e_h))
    assert o["dev"] / e_h <= 0.3


# ---------------------------------------------------------------------- T4
@pytest.mark.parametrize("which", ["rect0", "rect03"])
@pytest.mark.parametrize("k", [1, 2])
def test_beam_dehom_pure_shear_strain(request, which, k):
    """T4: a `0:` pure shear strain 1e-3 recovers the force C_eff st (the
    section is doubly symmetric, so Delta F_cl = 0); v2.1.0 gave max|s|
    7.2e-14 Pa."""
    r = request.getfixturevalue(which)
    st = np.zeros(6)
    st[k] = 1e-3
    _, S, _ = dehom_fields(r, st)
    F = np.asarray(r["C_eff"], float) @ st
    assert np.abs(fx.resultants(r, S) - F).max() <= 1e-10 * np.abs(F).max()


# ---------------------------------------------------------------------- T5
def _ladder_v210(C, st, V0, V1):
    """The f31b196 (v2.1.0) ladder, re-implemented for the parity gate:
    seed C^-1 st, operator recov(st + e1), R2/R3 product-rule terms."""
    from opensg_shell.kernels.msg_dehom import _recov
    Comp = np.linalg.inv(C)
    F_1d = Comp @ st
    R1 = _recov(st + np.eye(6)[0])
    F1 = R1 @ F_1d
    s1 = Comp @ F1
    R2 = _recov(s1)
    F2 = R1 @ F1 + R2 @ F_1d
    s2 = Comp @ F2
    R3 = _recov(s2)
    F3 = 2.0 * (R2 @ F1) + R3 @ F_1d + R1 @ F2
    s3 = Comp @ F3
    eps = st[CL] + Q @ s1[1:3]
    eps1 = s1[CL] + Q @ s2[1:3]
    eps2 = s2[CL] + Q @ s3[1:3]
    V0, V1 = np.asarray(V0), np.asarray(V1)
    return V0 @ eps, V1 @ eps1, V1 @ eps2, V0 @ eps1, eps


@pytest.mark.parametrize("k,mag", [(0, 1e6), (3, 1e3), (4, 1e3), (5, 1e3)])
def test_beam_dehom_classical_loads_unchanged(rect03, k, mag):
    """T5: F1 = 1e6 N and M1, M2, M3 = 1e3 N m alone (nu = 0.3): the
    first-order fields are exactly zero, the resultants are the applied
    load, and the field equals the v2.1.0 field (whose C^-2-seeded
    first-order part is ~1e-22 of it).  The v2.1.0 strain build is the
    current recover_gauss_point_fields: its frame rotation was the
    identity for a frameless file and its C_stress equals C_ess."""
    r = rect03
    FF = np.zeros(6)
    FF[k] = mag
    (w_1, w1s_1, w1s_2, w_2, eps), st = _chain(r, FF)
    for z in (w1s_1, w1s_2, w_2):
        assert np.array_equal(z, np.zeros_like(z))
    _, S, _ = dehom_fields(r, beam_force=FF)
    assert np.abs(fx.resultants(r, S) - FF).max() <= 1e-10 * mag
    old = _ladder_v210(np.asarray(r["C_eff"], float), st, r["V0"], r["V1s"])
    _, _, S_old = recover_gauss_point_fields(
        *[jnp.asarray(x) for x in old], r["dehomo_data"], r["C_ess"],
        r["reduced_periodic_cells"], r["phi_qn"], r["x_end"], 1, r["n_sg"])
    S_old = np.asarray(S_old)
    assert np.abs(S - S_old).max() <= 1e-12 * np.abs(S).max()


# ---------------------------------------------------------------------- T6
def test_beam_dehom_coupled_resultant_identity(coupled):
    """T6: on a fully coupled section (offset origin, +30/-45 glass halves)
    all six loads at once: [F2, F3] close exactly and [F1, M1, M2, M3] =
    F_cl + Delta F_cl, Delta F_cl = -A (I - QQ^T) A^-1 B A^-1 Q F_s (the
    VABS IEA r020 M1 excess -4727.45 N m is this term)."""
    r = coupled
    FF = np.array([1e5, 1e3, 2e3, 1e3, 1e3, 1e3])
    _, S, _ = dehom_fields(r, beam_force=FF)
    R = fx.resultants(r, S)
    dF = _delta_F_cl(r, FF)
    print("T6: Delta F_cl =", dF)
    assert np.abs(dF).max() > 1e-6 * np.abs(FF).max()      # coupled case
    tol = 1e-10 * np.abs(FF).max()
    assert np.abs(R[[1, 2]] - FF[[1, 2]]).max() <= tol
    assert np.abs(R[CL] - (FF[CL] + dF)).max() <= tol
    ex = beam_expected_resultants(r, beam_force=FF)
    assert np.array_equal(ex["F"], FF)
    assert np.abs(ex["dF_cl"] - dF).max() <= 1e-12 * np.abs(dF).max()
    assert np.abs(R - ex["expected"]).max() <= tol


# ------------------------------------------------- the section origin (T6b)
def _origin_pair(rA, rB, k):
    """The unit-direction shear force 1000 N (k = 1: F2, k = 2: F3) at
    the coupled origin A and the same load at the centred twin's origin
    B: F_B = T F_A, M_B = M_A + r x F, r = (0, 0.13, -0.07) the position
    of A in B's axes.  Out: T, F_A, F_B, the two SG-global fields."""
    xyA, xyB = fx.gauss_xy(rA), fx.gauss_xy(rB)
    assert np.abs((xyA + COUPLED_AT) - xyB).max() <= 1e-14  # the same points
    rv = np.array([0.0, COUPLED_AT[0], COUPLED_AT[1]])
    T = np.eye(6)
    T[3:, :3] = np.array([[0.0, -rv[2], rv[1]], [rv[2], 0.0, -rv[0]],
                          [-rv[1], rv[0], 0.0]])
    FA = np.zeros(6)
    FA[k] = 1e3
    FB = T @ FA
    _, SA, _ = dehom_fields(rA, beam_force=FA)
    _, SB, _ = dehom_fields(rB, beam_force=FB)
    return T, FA, FB, SA, SB


def _classical_field(r, e):
    """The zeroth-order field Ge e + Gamma_h (V0 e) of a classical state e
    [eps11 k1 k2 k3]."""
    e = jnp.asarray(np.asarray(e, float))
    w = jnp.asarray(r["V0"]) @ e
    z = jnp.zeros_like(w)
    _, _, S = recover_gauss_point_fields(
        w, z, z, z, e, r["dehomo_data"], r["C_ess"],
        r["reduced_periodic_cells"], r["phi_qn"], r["x_end"], 1, r["n_sg"])
    return np.asarray(S)


@pytest.mark.parametrize("k", [1, 2])
def test_beam_dehom_origin_dependence_is_delta_F_cl(coupled, coupled_centred,
                                                    k):
    """T6b: the T6 section meshed about its offset origin A and about its
    centre B, under the same physical shear force.  The two 6x6 laws are
    one law (S_A = T^T S_B T); the recovered fields differ, and the whole
    difference is the Delta F_cl state of each origin: the field minus
    the classical field of A^-1 Delta F_cl is one field about both origins
    (the zeroth-order state that closes [F1 M1 M2 M3], with the same
    first-order part).  The recovery of VABS 4.1 and msg-shell, pinned."""
    rA, rB = coupled, coupled_centred
    T, FA, FB, SA, SB = _origin_pair(rA, rB, k)
    SA6 = np.linalg.inv(np.asarray(rA["C_eff"], float))
    SB6 = np.linalg.inv(np.asarray(rB["C_eff"], float))
    assert _rel(SA6, T.T @ SB6 @ T) <= 1e-12
    dA, dB = _delta_F_cl(rA, FA), _delta_F_cl(rB, FB)
    ZA = _classical_field(rA, np.linalg.solve(np.asarray(rA["C_eff_EB"]), dA))
    ZB = _classical_field(rB, np.linalg.solve(np.asarray(rB["C_eff_EB"]), dB))
    print("T6b k=%d: field A vs B %.6e of max|sigma|; F1 recovered %.6g"
          " (A) / %.6g (B); Delta F_cl A %s, B %s" % (
              k, _rel(SA, SB), fx.resultants(rA, SA)[0],
              fx.resultants(rB, SB)[0], dA, dB))
    assert _rel(SA - ZA, SB - ZB) <= 1e-10


@pytest.mark.xfail(strict=True, reason=(
    "the generalized-Timoshenko recovery of VABS 4.1 and msg-shell is not"
    " origin-covariant on a coupled section: its zeroth-order state"
    " carries A^-1 Delta F_cl, which depends on the section origin"
    " (T6b); which procedure to keep is Akshat's decision"))
@pytest.mark.parametrize("k", [1, 2])
def test_beam_dehom_field_does_not_depend_on_the_origin(coupled,
                                                        coupled_centred, k):
    """What 3-D theory requires of a recovery: the same load on the same
    section gives the same stress at every point whatever the section
    origin (a 3-D solution of the prismatic beam closes every resultant at
    every section).  XFAIL while the D route keeps VABS parity."""
    _, _, _, SA, SB = _origin_pair(coupled, coupled_centred, k)
    assert _rel(SA, SB) <= 1e-10


# ------------------------------------------------------- drivers, refined 0
def test_beam_dehom_force_and_strain_drivers_agree(coupled):
    """beam_force F and epsilon_bar C^-1 F are the same state: the two
    drivers give the same field (to the round-off of the C^-1 F round
    trip), and exactly one of them is accepted."""
    r = coupled
    FF = np.array([1e5, 1e3, 2e3, 1e3, 1e3, 1e3])
    _, S_f, _ = dehom_fields(r, beam_force=FF)
    st = np.linalg.solve(np.asarray(r["C_eff"], float), FF)
    _, S_e, _ = dehom_fields(r, st)
    assert np.abs(S_f - S_e).max() <= 1e-10 * np.abs(S_f).max()
    with pytest.raises(ValueError):
        dehom_fields(r, st, beam_force=FF)
    with pytest.raises(ValueError):
        dehom_fields(r)
    with pytest.raises(ValueError):
        dehom_fields(r, st, dE1=np.zeros(6))                 # plate-only


def test_beam_dehom_refined0_classical(tmp_path_factory, rect03):
    """D3: a refined: 0 beam recovers the classical field only (the 4x4
    law, V0): a classical load closes exactly and equals the refined: 1
    field (pure classical loads drive no first-order term); F2/F3 and a
    shear strain are refused."""
    r0 = _rect(tmp_path_factory, "rect_nu03_cl", 0.3, refined=0)
    assert r0["refined"] == 0 and rect03["refined"] == 1
    FF = np.array([1e6, 0.0, 0.0, 1e3, -2e3, 3e3])
    _, S0, _ = dehom_fields(r0, beam_force=FF)
    assert np.abs(fx.resultants(r0, S0) - FF).max() <= 1e-10 * 1e6
    _, S1, _ = dehom_fields(rect03, beam_force=FF)
    assert np.abs(S0 - S1).max() <= 1e-10 * np.abs(S1).max()
    st = np.array([1e-4, 0.0, 0.0, 1e-3, 2e-3, -1e-3])
    _, Se, _ = dehom_fields(r0, st)
    F_cl = np.asarray(r0["C_eff_EB"], float) @ st[CL]
    assert np.abs(fx.resultants(r0, Se)[CL] - F_cl).max() \
        <= 1e-10 * np.abs(F_cl).max()
    for bad in ([0.0, 1e3, 0, 0, 0, 0], [0.0, 0, 1e3, 0, 0, 0]):
        with pytest.raises(ValueError, match="refined: 0"):
            dehom_fields(r0, beam_force=np.asarray(bad, float))
    with pytest.raises(ValueError, match="refined: 0"):
        dehom_fields(r0, np.array([0.0, 1e-3, 0, 0, 0, 0]))
