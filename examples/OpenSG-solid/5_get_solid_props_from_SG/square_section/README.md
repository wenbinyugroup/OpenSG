# Square thin-walled section — equivalent 3-D solid properties, two architectures

A square tube cross-section (wall midline side a = 1.0, wall thickness t = 0.03,
one orthotropic ply at −45°) homogenized to the equivalent 3-D solid
(`n_model = 3`) by **both** OpenSG solid engines, from the same 2-D SG.

| file | what |
|---|---|
| `square_tube_2Dsolid.yaml` | the 2-D SG: 3228 nodes, 5384 linear triangles, `elementOrientations` with `e1` out of plane (see `Rules/orientation_e1_out_of_plane.md`) |
| `square_tube.sc` | the same mesh in SwiftComp `.sc` form (what the old script reads) |
| `solid_props_new_architecture.py` | `opensg_solid.sg_homo.plate_homo_2d`, `n_model = 3` |
| `solid_props_old_architecture.py` | the original single-script JAX_BICGoptimize driver, `n_model = 3` |

Material (SI, Pa): E = [142.0e9, 9.8e9, 9.8e9], G = [6.0e9, 6.0e9, 4.8e9],
nu = [0.30, 0.30, 0.42], rho = 1600.

Voigt order in both outputs: `[e11 e22 e33 2e23 2e13 2e12]`, axis 1 = prismatic.
Both engines normalize by the **material** area (omega = 0.12), not the cell area.

## Running

```bash
python solid_props_new_architecture.py
python solid_props_old_architecture.py
```

The old script needs the `fe_jax` that ships with `~/OpenSG_2.0` (it puts that on
`sys.path` itself) and a jax new enough for its jit decorators — on the group
server that is `~/miniconda3/envs/jax_env/bin/python`. The new script runs under
`~/miniconda3/envs/opensg_2_0/bin/python`.

## Result — the two engines agree

Run head to head with the ply **unrotated** (`angles = 0`, no per-element frame),
which is the only setting the `.sc` route can express:

| | new architecture | old single script | diff |
|---|---|---|---|
| E1 | 1.420000e11 | 1.420000e11 | 0.000% |
| E2 | 5.117968e9 | 5.117967e9 | 0.000% |
| E3 | 5.117908e9 | 5.117908e9 | 0.000% |
| G23 | 9.645916e6 | 9.646245e6 | −0.003% |
| G13 | 3.144840e9 | 3.144840e9 | 0.000% |
| G12 | 3.144834e9 | 3.144834e9 | 0.000% |
| nu12 | 0.3000000 | 0.3000037 | −0.001% |
| nu23 | 0.0267505 | 0.0267543 | −0.014% |

5–7 significant figures. `E1 = 142 GPa` exactly, as it must be when the fibres
run along the beam axis.

## The one capability difference

A `.sc` carries **one material id per element** and the old driver applies **one
fibre angle per material**, so per-element material frames cannot be represented
there. The new architecture takes them through `elem_rotation`.

Set `USE_ORIENTATIONS = True` in `solid_props_new_architecture.py` to use the
yaml's per-element frames with the real −45° ply; that gives

```
E1 = 1.473164e10   (= the analytical -45 deg off-axis lamina modulus, 14.73 GPa)
E2 = 7.879232e9    E3 = 5.292977e9
G23 = 1.582810e7   G13 = 2.813562e9   G12 = 5.359323e9
nu12 = 0.2276368   nu13 = 0.3312396   nu23 = -0.0112453
```

which the old script cannot reproduce from a `.sc`.

The mesh came from PreVABS — see `.claude/skills/prevabs-xml-to-2dyaml/SKILL.md`
for the XML → `.sg` → 2-D yaml recipe, and `tests/08072026_square_shell_mesh/`
for the full working folder including the beam (Timoshenko) runs.
