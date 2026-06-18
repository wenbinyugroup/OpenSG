# Solid cross-section test cases (MSG thin-walled benchmark)

Self-contained FEniCS 2D-solid (VABS-equivalent) Timoshenko-beam test cases used
as the **solid reference** for the MSG thin-walled (RM / Kirchhoff shell)
homogenization study. Each case is a 2D-solid cross-section YAML you can run on
its own to get the 6×6 Timoshenko stiffness.

## Layout

```
examples/test/
├── generate_solid_meshes.py     # regenerate every YAML in meshes/ (pure numpy, no FEniCS)
├── run_solid_timo_from_yaml.py  # run one YAML -> print Timoshenko 6x6  (needs FEniCSx + opensg)
├── meshes/                      # 34 ready-to-run 2D-solid YAMLs
└── README.md
```

## Running

Inside your FEniCSx / `opensg` environment:

```bash
cd examples/test
python run_solid_timo_from_yaml.py meshes/stripthick_iso_h0.5_W1.0.yaml
python run_solid_timo_from_yaml.py meshes/tube_iso_hR0.06.yaml
```

The driver builds `opensg.mesh.segment.SolidBounMesh`, calls
`opensg.core.solid.compute_timo_boun`, and prints the 6×6. The stiffness ordering
is `[ext, shear2, shear3, twist, bend2, bend3]`, i.e. the diagonal is
`[EA, GA2, GA3, GJ, EI2, EI3]`.

To regenerate / extend the mesh set (no FEniCS needed — just numpy):

```bash
python generate_solid_meshes.py
```

## Case families (34 meshes)

| Family | Files | Geometry | Sweep |
|---|---|---|---|
| Tube | `tube_{iso,aniso}_hR{0.01,0.03,0.06,0.12,0.20}.yaml` | circular tube, R = 1 m | h/R |
| Strip | `strip_{iso,aniso}_hW{0.01,0.03,0.06,0.12,0.20}.yaml` | flat strip, W = 1 m | h/W |
| Strip width | `stripwidth_iso_W{0.1..2.5}_h0.05.yaml` | strip, h = 0.05 m fixed | W (h/W 0.5 → 0.02) |
| Strip thickness | `stripthick_iso_h{0.01..0.5}_W1.0.yaml` | strip, W = 1 m fixed | h (h/W 0.01 → 0.5, thin → thick) |

- **Isotropic:** E = 70 GPa, ν = 0.3.
- **Anisotropic:** `[-45/45]` (two through-thickness — for the tube, radial —
  bands), E = [37, 9, 9] GPa, G = [4, 4, 4] GPa, ν = 0.3, fibre baked into the
  per-element material frame EE1.
- Sections are centred at the origin so the centroid coincides with the
  centre-referenced thin-walled shell model.

## Mesh format

Standard OpenSG 2D-solid YAML (4-node quads):

```yaml
nodes:           # - [y2 y3 0.0]      (space-separated, centred at origin)
elements:        # - [n1 n2 n3 n4]    (1-indexed quads)
sets:            # element set -> material name
materials:       # E[3], G[3], nu[3], rho
elementOrientations:   # - [e1x e1y e1z  e2x e2y e2z  e3x e3y e3z] per element
```

`e3` is the through-thickness normal; `e1` is the fibre direction.

## Sanity-check values (isotropic strip thickness sweep, W = 1 m)

| case | h/W | C44 (GJ) | C11 (EA) |
|---|---|---|---|
| `stripthick_iso_h0.01_W1.0` | 0.01 | ≈ 8.93e3 | 7.00e8 |
| `stripthick_iso_h0.5_W1.0`  | 0.50 | ≈ 7.70e8 | 3.50e10 |

These are the VABS-solid references the RM and Kirchhoff shell models are
compared against (RM stays within ~0.3 % of GJ from thin → thick; Kirchhoff
over-stiffens torsion as the section thickens).
