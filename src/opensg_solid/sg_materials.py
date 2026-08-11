"""sg_materials.py -- the 6x6 stiffness tables of the general SG engine
(per material and per element), split out of rm_plate_2D.py.

PROVENANCE.  build_single_C_matrix / rotate_C_matrix / build_material_C
are the SSDM plate engine's (validated, n_model 2/3);
rotate_C_with_matrix and get_heterogeneous_C_matrix are the Beam_solid.py
lifts (the n_model=1 KKT path, see sg_homo.py).

DEDUPE vs rm_plate_1D.msg_materials (diffed, not force-merged):
- build_stiffness_6x6 is the SAME compliance inverse as
  build_single_C_matrix (verified identical); it is numpy -- this engine
  vmaps/jits the builder, so the traceable jnp version stays here.
- rotation_6x6(t) == rotate_C_matrix(-t): the OpenSG R_sig convention
  (rm_plate_1D, and Beam_solid's own rotate_C_matrix) is the OPPOSITE
  angle sign of the SSDM R_Sig below.  The 2-D engine keeps the SSDM
  sign its plate/beam validations were produced with -- do NOT swap one
  for the other.

Voigt order everywhere: (xx, yy, zz, yz, xz, xy), SwiftComp/laminate.

# ----------------------------------------------------------------------------
# ALL VARIABLES USED IN THIS MODULE  (axis suffixes: e element, s Voigt-6)
# ----------------------------------------------------------------------------
# inputs
#   params              (9,) engineering E1 E2 E3 G12 G13 G23 v12 v13 v23
#   t / angle           ply angle about y3 [deg]; 0.0 skips the rotation
#   sc                  parsed SG dict {dim, nodes, cells, mat_id,
#                       materials, scale} (sg_mesh.load_sg_input)
#   material_param      (n_mat, 9) engineering override or None (.sc blocks)
#   angles              (n_mat,) deg per material or None
#   R_flat              (9,) flattened per-element direction-cosine matrix
#   elem_rotation       (E, 9) flattened DCs or None (parser yields none)
# working
#   S, C, C_mat         (6, 6) compliance / stiffness, SwiftComp order
#   R_Sig               (6, 6) SSDM Voigt stress rotation (the validated sign)
#   K                   (6, 6) Voigt transform built from R_flat
#   ids                 (E,) 0-based per-element material ids (= mat_seq)
# outputs
#   C_m                 (n_mat, 6, 6) per-MATERIAL table (build_material_C)
#   C_mat_e, C_asm_e    (E, 6, 6) per-ELEMENT tables: pre-elem-rotation
#                       (recovery) and assembly (elem_rotation applied)
# ----------------------------------------------------------------------------
"""
import jax
import jax.numpy as jnp
import numpy as np


def build_single_C_matrix(params):
    """(9,) engineering constants E1 E2 E3 G12 G13 G23 v12 v13 v23 -> (6,6) C."""
    E1, E2, E3, G12, G13, G23, v12, v13, v23 = params
    S = jnp.zeros((6, 6))
    S = S.at[0, 0].set(1 / E1)
    S = S.at[1, 1].set(1 / E2)
    S = S.at[2, 2].set(1 / E3)
    S = S.at[3, 3].set(1 / G23)
    S = S.at[4, 4].set(1 / G13)
    S = S.at[5, 5].set(1 / G12)
    S = S.at[0, 1].set(-v12 / E1).at[1, 0].set(-v12 / E1)
    S = S.at[0, 2].set(-v13 / E1).at[2, 0].set(-v13 / E1)
    S = S.at[1, 2].set(-v23 / E2).at[2, 1].set(-v23 / E2)
    return jnp.linalg.inv(S)


def rotate_C_matrix(C, t):
    """Rotate a 6x6 stiffness by angle t [deg] about y3; skip when t = 0."""
    def do_rotation(operand):
        C_mat, angle = operand
        th = jnp.deg2rad(angle)
        c, s = jnp.cos(th), jnp.sin(th)
        cs = c * s
        R_Sig = jnp.array([
            [c ** 2, s ** 2, 0, 0, 0, 2 * cs],
            [s ** 2, c ** 2, 0, 0, 0, -2 * cs],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, c, -s, 0],
            [0, 0, 0, s, c, 0],
            [-cs, cs, 0, 0, 0, c ** 2 - s ** 2]])
        return R_Sig @ C_mat @ R_Sig.T

    def skip_rotation(operand):
        C_mat, _ = operand
        return C_mat

    return jax.lax.cond(t == 0.0, skip_rotation, do_rotation, (C, t))


def _single_C_np(params):
    """NumPy twin of build_single_C_matrix -- the identical compliance
    inverse (rm_plate_1D.build_stiffness_6x6 precedent).  The material
    table is a handful of 6x6s, so building it eagerly in numpy avoids
    re-tracing a vmapped builder in every fresh process (~0.6 s/call)."""
    E1, E2, E3, G12, G13, G23, v12, v13, v23 = (float(x) for x in params)
    S = np.zeros((6, 6))
    S[0, 0] = 1.0 / E1
    S[1, 1] = 1.0 / E2
    S[2, 2] = 1.0 / E3
    S[3, 3] = 1.0 / G23
    S[4, 4] = 1.0 / G13
    S[5, 5] = 1.0 / G12
    S[0, 1] = S[1, 0] = -v12 / E1
    S[0, 2] = S[2, 0] = -v13 / E1
    S[1, 2] = S[2, 1] = -v23 / E2
    return np.linalg.inv(S)


def _rotate_C_np(C, t):
    """NumPy twin of rotate_C_matrix -- the same SSDM R_Sig sign."""
    if float(t) == 0.0:
        return np.asarray(C, float)
    th = np.deg2rad(float(t))
    c, s = np.cos(th), np.sin(th)
    cs = c * s
    R_Sig = np.array([
        [c ** 2, s ** 2, 0, 0, 0, 2 * cs],
        [s ** 2, c ** 2, 0, 0, 0, -2 * cs],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, c, -s, 0],
        [0, 0, 0, s, c, 0],
        [-cs, cs, 0, 0, 0, c ** 2 - s ** 2]])
    return R_Sig @ np.asarray(C, float) @ R_Sig.T


def _material_table_np(sc, material_param=None, angles=None):
    """The per-MATERIAL (n_mat, 6, 6) table in plain numpy -- the shared
    core of build_material_C / get_heterogeneous_C_matrix (semantics
    documented on build_material_C)."""
    if material_param is not None:
        rows = [_single_C_np(p) for p in np.asarray(material_param, float)]
        blk_ang = None
    else:
        rows, blk_ang = [], []
        for mid in sorted(sc["materials"]):
            m = sc["materials"][mid]
            if m["type"] == 0:
                E, nu = float(m["E"]), float(m["nu"])
                G = E / (2.0 * (1.0 + nu))
                rows.append(_single_C_np([E, E, E, G, G, G, nu, nu, nu]))
            elif m["type"] == 1:
                rows.append(_single_C_np(m["engineering"]))
            else:
                rows.append(np.asarray(m["C"], float))
            blk_ang.append(float(m.get("angle", 0.0)))
    C_m = np.stack(rows)
    if angles is not None:
        C_m = np.stack([_rotate_C_np(C, t)
                        for C, t in zip(C_m, np.asarray(angles, float))])
    elif blk_ang is not None and any(a != 0.0 for a in blk_ang):
        C_m = np.stack([_rotate_C_np(C, t) for C, t in zip(C_m, blk_ang)])
    return C_m


def build_material_C(sc, material_param=None, angles=None):
    """The per-MATERIAL (n_mat, 6, 6) stiffness table.

    In:  sc dict (helper read_sc/convert); material_param (n_mat, 9)
         engineering override or None -> use the .sc material blocks
         (type 0 iso E/nu, type 1 the 9 constants, type 2 the stored 6x6 --
         NOTE type-2 .sc entries are typically PRE-ROTATED ply C's, so no
         angles should be given with them); angles (n_mat,) deg or None.
         A block may also carry its own `angle` (deg): applied per material
         when the `angles` argument is None, so an SG file can be
         self-contained (constants + angle in the file, no override) --
         the type-1/type-0 route; a type-2 block normally has no `angle`
         since its stored C is already rotated.
    Out: (n_mat, 6, 6) jnp -- rotated when angles (or block angles) are
         given."""
    return jnp.asarray(_material_table_np(sc, material_param, angles))


def rotate_C_with_matrix(C, R_flat):
    """6x6 stiffness rotation by a flattened (9,) direction-cosine matrix
    (Beam_solid.py, verbatim).  The r_ij indexing reads R column-major --
    the transform is by R^T of the stored layout; keep the caller's DC
    convention consistent with Beam_solid's elem_rotation blocks."""
    R = R_flat.reshape(3, 3)
    r11, r12, r13 = R[0, 0], R[1, 0], R[2, 0]
    r21, r22, r23 = R[0, 1], R[1, 1], R[2, 1]
    r31, r32, r33 = R[0, 2], R[1, 2], R[2, 2]
    K = jnp.array([
        [r11**2, r12**2, r13**2, 2*r12*r13, 2*r11*r13, 2*r11*r12],
        [r21**2, r22**2, r23**2, 2*r22*r23, 2*r21*r23, 2*r21*r22],
        [r31**2, r32**2, r33**2, 2*r32*r33, 2*r31*r33, 2*r31*r32],
        [r21*r31, r22*r32, r23*r33, r22*r33 + r23*r32, r21*r33 + r23*r31,
         r21*r32 + r22*r31],
        [r11*r31, r12*r32, r13*r33, r12*r33 + r13*r32, r11*r33 + r13*r31,
         r11*r32 + r12*r31],
        [r11*r21, r12*r22, r13*r23, r12*r23 + r13*r22, r11*r23 + r13*r21,
         r11*r22 + r12*r21]])
    return K @ C @ K.T


def elem_rotation_from_yaml(orientations):
    """(E, 9) `elem_rotation` in the SOLVER's global order, from an OpenSG
    yaml's `elementOrientations`.

    THE CONVENTION GAP THIS CLOSES.  An OpenSG SG yaml stores each material
    axis with its components ordered (y2, y3, axial) -- the two cross-section
    coordinates first, the prismatic direction last -- because the yaml's node
    rows are written the same way.  The solver indexes global axes as
    (1, 2, 3) = (axial, y2, y3).  Every 3-vector therefore has to be cycled

        [a, b, c]  ->  [c, a, b]

    before it can be used as a direction-cosine matrix.  Feeding the yaml order
    straight into `elem_rotation` points e1 along y3 instead of along the beam
    axis: the run still completes and still looks plausible, but it silently
    destroys the section's symmetry (a square tube comes out with EI2 != EI3)
    and lands on the properties of a uniformly-rotated lamina.  The
    OpenSG-FEniCSx solid reader applies the same cycle, as EE1 = (o2, o0, o1).

    In:  (E, 9) or (E, 3, 3) rows [e1 | e2 | e3] in yaml component order.
    Out: (E, 9) rows [e1 | e2 | e3] in solver component order.
    """
    o = np.asarray(orientations, float).reshape(-1, 3, 3)
    return o[:, :, [2, 0, 1]].reshape(-1, 9)


def get_heterogeneous_C_matrix(sc, material_param=None, angles=None,
                               elem_rotation=None):
    """Per-ELEMENT stiffness tables for the beam/KKT path (Beam_solid.py's
    get_heterogeneous_C_matrix on the repo input interface: mat_seq = the
    per-element material ids in sc["mat_id"], the per-material build +
    angle rotation = build_material_C).

    In:  sc dict; material_param/angles as build_material_C;
         elem_rotation (E, 9) flattened per-element DC matrices or None.
         The rows must be in the SOLVER's global order (axial, y2, y3) -- pass
         a yaml's elementOrientations through elem_rotation_from_yaml first,
         which cycles the components; see that docstring for what goes wrong
         otherwise.  sc["mat_id"] is 1-BASED here.
    Out: (C_asm_e, C_mat_e) -- both (E, 6, 6): the ASSEMBLY stiffness
         (elem_rotation applied when given) and the pre-elem-rotation
         stiffness the recovery multiplies the element-frame strain by."""
    C_np = _material_table_np(sc, material_param, angles)
    ids = np.asarray(sc["mat_id"], int) - 1
    C_mat_e = jnp.asarray(C_np[ids])       # numpy gather, ONE device put
    if elem_rotation is None:
        return C_mat_e, C_mat_e
    R = jnp.asarray(elem_rotation, float).reshape(C_mat_e.shape[0], 9)
    C_asm_e = jax.vmap(rotate_C_with_matrix)(C_mat_e, R)
    return C_asm_e, C_mat_e
