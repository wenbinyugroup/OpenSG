# OpenSG

OpenSG is a multiscale structural-analysis code based on the Mechanics of Structure Genome (MSG), developed by the
Multiscale Structural Mechanics Group led by Prof. Wenbin Yu at Purdue University. From a Structure Gene (SG), the
smallest building block of a structure, it computes the constitutive relations of solid, plate/shell and beam models
(homogenization), and it recovers the local stress, strain and displacement fields from the structural response
(dehomogenization).

```{figure} images/opensg_overview.png
:width: 70%
:name: fig-overview

Structure Genes homogenized into solid, plate/shell and beam models.
```

An SG is a 1-D, 2-D or 3-D finite-element mesh. OpenSG has two engines that share one theory and one command,
`opensg <sg.yaml>`:

| engine | SG | produces |
|---|---|---|
| `opensg_solid` | 1-D, 2-D or 3-D SG of solid elements | plate ABD (6x6) or Reissner-Mindlin ABDG (8x8), Timoshenko beam 6x6, equivalent 3-D solid 6x6 |
| `opensg_shell` | 1-D shell SG (cross-section contour), 3-D shell SG (surface) | Timoshenko beam 6x6, equivalent 3-D solid 6x6 |

::::{grid} 2
:gutter: 3

:::{grid-item-card} Getting started
:link: getting_started/index
:link-type: doc

Install OpenSG, run the first case, and follow the tutorial on a blade cross-section.
:::

:::{grid-item-card} User guide
:link: user_guide/index
:link-type: doc

How to prepare the yaml input: the analysis header, the mesh, materials and layups, recovery, converters and output files.
:::

:::{grid-item-card} Examples
:link: examples/index
:link-type: doc

Worked examples for every capability, each with its input, command, result and the reference it is compared with.
:::

:::{grid-item-card} Theory and reference
:link: theory/index
:link-type: doc

The constitutive relations, the architecture of the two engines, the formulation rules and the benchmarks.
:::
::::

## What each capability answers

| you have | you want | where to look |
|---|---|---|
| a laminate stacking sequence | plate ABD or Reissner-Mindlin 8x8, and stress through the thickness | {doc}`examples/solid_plate_1d_layup`, {doc}`examples/solid_plate_1d_recovery` |
| a 2-D unit cell (honeycomb, lattice) | plate ABD, beam 6x6, or an equivalent 3-D solid | {doc}`examples/solid_plate_2d_honeycomb`, {doc}`examples/solid_beam_2d`, {doc}`examples/solid_equivalent_2d` |
| a composite blade cross-section | Timoshenko 6x6 and 3-D wall stress from beam loads | {doc}`getting_started/tutorial`, {doc}`examples/shell_beam_cross_section`, {doc}`examples/shell_beam_recovery` |
| a tapered blade segment (two different ends) | the segment's Timoshenko 6x6 without periodicity | {doc}`examples/shell_taper_segment` |
| a TPMS or lattice unit cell | the equivalent 3-D solid law | {doc}`examples/solid_equivalent_3d_tpms`, {doc}`examples/shell_solid_3d_tpms`, {doc}`examples/shell_solid_cellular` |
| a windIO blade file | a full spanwise beam model for BeamDyn, plus recovery | {doc}`examples/blade_windio_pipeline`, {doc}`examples/blade_iea_beam` |

## Validation status

| case | reference | result |
|---|---|---|
| TPMS 3-D solid SG (192k + 546k tets) | SwiftComp `.K` | digit-for-digit, all 9 entries |
| Cellular lattice $\pm15^\circ$ | Deo & Yu 2023 / SwiftComp | $C_{44}$ error +3.1 / +8.4 % → **+0.18 / +0.17 %** |
| Hierarchical square | Deo & Yu 2023 / SwiftComp | matches the paper's thin-walled column to ≤1 % |
| Periodic cross cell, $C_{44}$ | closed form $\tfrac12 E'(t/L)^3$ | +0.01 % at $t/L = 0.0125$ |
| Schwarz-P shell SG vs its own solid | SwiftComp-exact msg-solid | −3 % normals, −6 % shears, +0.3 % $\nu$ |
| Aperiodic segment, prismatic tube (iso + ±45) | its own boundary ring $6\times6$ | identity to $\le 5\times10^{-5}$ rel |

```{toctree}
:hidden:
:maxdepth: 2

getting_started/index
user_guide/index
examples/index
theory/index
references
```
