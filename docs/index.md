# OpenSG-2.0

**Mechanics of Structure Genome (MSG) homogenization in JAX** — beam, plate and equivalent 3-D
solid properties from 1-D, 2-D and 3-D structure genes, through two codes that share the same
theory:

| code | input | produces |
|---|---|---|
| **msg-shell** (`opensg_shell`) | 1-D shell SG (cross-section), 3-D shell SG (surface) | Timoshenko $6\times6$, equivalent 3-D solid $6\times6$ |
| **msg-solid** (`opensg_solid`) | 1-D / 2-D / 3-D solid SG | plate ABD, Timoshenko $6\times6$, equivalent 3-D solid $6\times6$ |

Every run is the same shape: **a YAML (or SwiftComp `.sc`) goes in, a timed `.out` in the
SwiftComp `.K` layout comes back**. The core packages hold no paths and no `main` blocks — each
example script names its input file and calls one entry function.

```python
from opensg_shell import build_rm_bundle

B = build_rm_bundle("iea_s10_shell.yaml")   # writes iea_s10_shell_Timo.out
B["Timo"]                                    # the Timoshenko 6x6
```

```{toctree}
:maxdepth: 1
:caption: Start here

input_format
constitutive
architecture
```

```{toctree}
:maxdepth: 1
:caption: Tutorials — msg-shell

tutorials/msg_shell_beam
tutorials/solid_props_shell_sg
tutorials/taper_segment_aperiodic
tutorials/windio_blade_pipeline
```

```{toctree}
:maxdepth: 1
:caption: Tutorials — msg-solid

tutorials/msg_solid_plate
tutorials/msg_solid_beam_and_solid
tutorials/solid_props_3d_sg
```

```{toctree}
:maxdepth: 1
:caption: Theory & conventions

theory
```

## What each capability answers

| you have | you want | tutorial |
|---|---|---|
| a laminate stacking sequence | plate ABD / Reissner–Mindlin $8\times8$, and stress through the thickness | Plate Properties and Recovery (msg-solid) |
| a 2-D unit cell (honeycomb, lattice) | plate ABD, beam $6\times6$, or an equivalent 3-D solid | Plate / Beam and Equivalent-Solid (msg-solid) |
| a composite blade cross-section | Timoshenko $6\times6$ and 3-D wall stress from beam loads | Beam Properties from a Shell Cross-Section |
| a tapered blade segment (two different ends) | the segment's Timoshenko $6\times6$ without periodicity | Tapered / Aperiodic Shell Segment |
| a TPMS or lattice unit cell | the equivalent 3-D solid law | Equivalent 3-D Solid Properties |
| a windIO blade file | a full spanwise beam model for BeamDyn, plus recovery | Full Blade Pipeline |

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
- `periodicity_in_solid_props.md` — periodic vs aperiodic boundary treatment
- `orientation_e1_out_of_plane.md` — $e_1$ convention for 1-D/2-D YAML
