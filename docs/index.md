# OpenSG-2.0

**Mechanics of Structure Genome (MSG) homogenization in JAX** — beam, plate and equivalent 3-D
solid properties from 1-D, 2-D and 3-D structure genes, through two codes that share the same
theory:

| code | input | produces |
|---|---|---|
| **msg-shell** (`opensg_shell`) | 1-D shell SG (cross-section), 3-D shell SG (surface) | Timoshenko $6\times6$, equivalent 3-D solid $6\times6$ |
| **msg-solid** (`opensg_solid`) | 1-D / 2-D / 3-D solid SG | plate ABD, Timoshenko $6\times6$, equivalent 3-D solid $6\times6$ |

Every run writes a timed `.out` in the SwiftComp `.K` layout — see
`Rules/output_format_and_timing.md`.

```{toctree}
:maxdepth: 1
:caption: Equivalent 3-D solid properties

tutorials/solid_props_shell_sg
tutorials/solid_props_3d_sg
```

```{toctree}
:maxdepth: 1
:caption: Beam properties

tutorials/taper_segment_aperiodic
```

```{toctree}
:maxdepth: 1
:caption: Theory & conventions

theory
```

## Validation status

| case | reference | result |
|---|---|---|
| TPMS 3-D solid SG (192k + 546k tets) | SwiftComp `.K` | digit-for-digit, all 9 entries |
| Cellular lattice $\pm15^\circ$ | Deo & Yu 2023 / SwiftComp | $C_{44}$ error +3.1 / +8.4 % → **+0.18 / +0.17 %** |
| Hierarchical square | Deo & Yu 2023 / SwiftComp | matches the paper's thin-walled column to ≤1 % |
| Periodic cross cell, $C_{44}$ | closed form $\tfrac12 E'(t/L)^3$ | +0.01 % at $t/L = 0.0125$ |
| Schwarz-P shell SG vs its own solid | SwiftComp-exact msg-solid | −3 % normals, −6 % shears, +0.3 % $\nu$ |
| Aperiodic segment, prismatic tube (iso + ±45) | its own boundary ring $6\times6$ | identity to $\le 5\times10^{-5}$ rel |

## Working rules

Formulation and workflow rules live in `Rules/` at the repository root:

- `gamma_e_gamma_h_consistency.md` — $\Gamma_e$ is $\Gamma_h$ on the macro embedding
- `junction_corrections.md` — when a junction term applies, and when it must stay off
- `output_format_and_timing.md` — the `.out` contract
- `benchmarking_solid_props.md` — reference hierarchy and how to compare
- `periodicity_in_solid_props.md` — periodicity is always on for solid properties
- `orientation_e1_out_of_plane.md` — $e_1$ convention for 1-D/2-D YAML
