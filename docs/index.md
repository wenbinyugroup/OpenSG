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

## Running OpenSG

One command, one argument — **the file says what to do**:

```bash
opensg <sg.yaml>        # homogenization (the default)
```

```bash
opensg <sg.yaml> D      # dehomogenization: homogenize, then recover the local fields
```

`opensg` is a *dispatcher*: it reads the yaml's `msg:` key (`shell` or `solid`) and hands the
file to the engine that owns it. Without the key the mesh **dialect** decides, so every YAML
written before the key existed still runs. The two engine commands remain and behave
identically once the file has been resolved:

```bash
opensg_solid <sg.yaml> [H|D]     # the general 1-D / 2-D / 3-D SG engine
```

```bash
opensg_shell <sg.yaml> [H|D]     # the msg-shell contour / surface engine
```

(`python -m opensg_solid <sg.yaml>` and `python -m opensg_shell <sg.yaml>` are the same entry
points.) Everything else — the macro model, the refinement, the boundary treatment, the macro
state for a recovery — lives in the YAML header and is documented under
{doc}`input_format`. Every key is defaulted, so a headerless mesh still runs.

A run prints what it resolved, then the law, then where it was stored:

```text
 ============================================================================
 OpenSG -- a multiscale structural analysis tool based on the Mechanics of
 Structure Genome (MSG), developed by the Multiscale Structural Mechanics
 Group led by Prof. Wenbin Yu at Purdue University
 ============================================================================

 input     : /.../1_get_beam_props_from_shell_cross_section/iea_s10_shell.yaml
 msg       : shell
 SG dim    : 2D
 analysis  : homogenization
 macro model: beam, shear-refined

Timoshenko Beam Stiffness Matrix  [eps11 gam12 gam13 kappa1 kappa2 kappa3]:
[[ 2.76606e+10 -1.72307e-05  4.66505e-06  5.40202e-06  1.66864e+09 -2.07730e+09]
 ...
Homogenization stored in iea_s10_shell_Timo.out
Time taken: 2.58 sec
```

`SG dim` is the **ambient** dimension — the space the SG occupies, not the dimension of the SG
manifold. A cross-section ring and a 2-D cell both print `2D`; a Schwarz-P shell surface, whose
facets live in space, prints `3D`. A dehomogenization closes with
`Local field files are computed and stored.` and one `Time taken` covering the whole run
instead.

Every entry function is also importable, for driving a run from your own script:

```python
from opensg_shell import build_rm_bundle

B = build_rm_bundle("iea_s10_shell.yaml")   # writes iea_s10_shell_Timo.out
B["Timo"]                                    # the Timoshenko 6x6
```

```{toctree}
:maxdepth: 1
:caption: Start here

installation
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
