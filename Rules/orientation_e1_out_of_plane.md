# Rule — `e1` is out of plane by default in every OpenSG yaml

**Applies to:** every OpenSG structure-gene yaml, both dialects —
the 1-D shell SG (`elements` = 2-node lines) and the 2-D solid SG
(`elements` = triangles/quads). Both carry `elementOrientations`, one
9-component row per element, read as three stacked unit vectors:

```
[ e1x e1y e1z | e2x e2y e2z | e3x e3y e3z ]
```

## The rule

**`e1` is the BEAM AXIS (the prismatic / out-of-plane direction) by default.**
For a cross-section meshed in the (y2, y3) plane with the beam axis along z,
that means

```
e1 = [0, 0, 1]        out of plane          -> e1_z = 1
e2 = wall tangent      in plane
e3 = e1 x e2           in plane, wall normal -> e3_z = 0
```

`e2` and `e3` are in-plane and complete a right-handed triad. The ply / fibre
angle is **NOT** baked into `e1`; it is carried separately:

| dialect | where the fibre angle lives |
|---|---|
| 1-D shell yaml | `sections[].layup` entries `[material, thickness, angle]` |
| 2-D solid yaml | the `angles` argument of the homogenizer (per material), or a pre-rotated material |

Both yamls describing the same physical section must therefore show the **same**
`e1`, out of plane, regardless of the ply angle.

## Why this matters

A frame whose `e1` already contains the fibre rotation (`e1_z = cos θ3`, e.g.
0.7071 for a ±45° ply) is a *fibre* frame, not the section frame. Mixing the two
silently double-counts or drops the ply rotation:

- rotate the material by `angles` **and** hand it a fibre-tilted `e1` → the ply
  angle is applied twice;
- hand it a fibre-tilted `e1` and `angles = 0` → correct, but then the 1-D and
  2-D yamls disagree and are no longer comparable term by term.

Keeping `e1` out of plane in both dialects makes the shell and solid routes
directly comparable and keeps one single place where the ply angle is declared.

## PreVABS conversion — the concrete gotcha

`OpenSG_io/scripts/convert_sg_to_yaml.py` writes **two** files from a `.sg`:

| file | frame | `e1` |
|---|---|---|
| `<name>.yaml` | θ1 **and** θ3 | fibre direction, `e1_z = cos θ3` (tilted) |
| `<name>_t1only.yaml` | θ1 only | **beam axis, `e1 = [0,0,1]`** |

Per this rule the **`_t1only`** file is the OpenSG-conforming 2-D solid yaml —
use it, and supply the fibre angle through `angles`. The θ1+θ3 file is the
fibre-frame variant; use it only with `angles = 0`, and never compare it
element-by-element against a 1-D shell yaml.

The two are physically equivalent (a rotation by θ3 about the ply normal `e3`
turns one into the other) — this rule is about which one OpenSG treats as
canonical.

## Check before using any yaml

```python
ori = np.array(d["elementOrientations"], float)
e1, e3 = ori[:, 0:3], ori[:, 6:9]
assert np.allclose(np.abs(e1[:, 2]), 1.0)   # e1 out of plane
assert np.allclose(e3[:, 2], 0.0)           # e3 in plane
```

If `|e1_z| < 1`, the yaml carries a fibre frame — either switch to the
`_t1only` variant or set the fibre angle to zero downstream.

Worked example: `tests/08072026_square_shell_mesh/` (square tube, m45 ply at
−45°) — `square_tube_1Dshell.yaml` and `square_tube_2Dsolid_t1only.yaml` both
have `e1 = [0, 0, 1]`, while `square_tube_2Dsolid.yaml` has
`e1 = [−0.7071, 0, 0.7071]`.
