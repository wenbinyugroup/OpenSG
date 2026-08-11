# Tapered / aperiodic shell segment (boundary Dirichlet)

The classical msg-shell analysis meshes only the cross-section contour (a 1-D
ring) and enforces periodicity along the beam axis. A **tapered** segment has
no such periodicity — its two end cross-sections differ. `opensg_shell`
handles this with the aperiodic segment pipeline
(`opensg_shell.sg_homo`), the msg-shell counterpart of the
OpenSG-FEniCSx boundary flow:

1. **Boundary extraction** — the two end cross-sections are found
   *topologically* (mesh edges used by exactly one quad = free edges; their
   connected components are the end rings) and written as **separate 1-D
   yamls** from the given 3-D shell yaml, together with the
   ring-node → segment-node map.
2. **Boundary solves** — each ring is homogenized on its own (`ring_indep`),
   giving its Timoshenko $6\times6$ and the warping fields $V_0$
   (Euler–Bernoulli) and $V_1$ (Timoshenko), all six DOFs per node including
   the drilling rotation $\omega_3$.
3. **Mapping** — the ring $V_0$/$V_1$ are transferred onto the segment's
   boundary nodes as **Dirichlet data**, replacing the periodic boundary
   condition (and fixing the segment's rigid-body modes).
4. **Segment solve** — one LU factorization of the constrained system serves
   both orders:

   $$D_{hh}\,V_0 = -D_{he}, \qquad D_{hh}\,V_1 = b(V_0),$$

   with $V_0 = V_0^{\text{ring}}$, $V_1 = V_1^{\text{ring}}$ on both ends;
   the generalized-Timoshenko finalization divides the energy by the segment
   length $L$.

## What the user runs

The whole pipeline is one call on the 3-D shell YAML — no boundary files to
prepare, because the ends are found from the mesh topology:

```python
from opensg_shell.sg_homo import segment_timo_from_3dyaml

r = segment_timo_from_3dyaml("meshes/seg_iso_hR0.1.yaml")
r["S6"]             # segment Timoshenko 6x6
r["C6L"], r["C6R"]  # the two boundary-ring 6x6s
r["L"]              # segment length used to normalize the energy
```

Each run writes the two boundary 1-D YAMLs, the `_boundaries.npz` bundle and
the timed `<yaml>_Timo.out` in the SwiftComp layout. The three orientation
PNGs (segment, left ring, right ring) are written only when they do not
already exist, so delete them after changing a mesh if you want them
refreshed.

## Seeing the capability

A tapered segment is a genuine 3-D surface mesh, not a cross-section. This is
the BAR-URC blade segment the example runs, coloured by its per-element
material frames — the surface on which the segment problem is solved:

![BAR-URC tapered segment with per-element material frames](../_static/aperiodic_segment_bar_urc.png)

The two ends are extracted automatically. A mesh edge used by exactly one quad
is a free edge, and each connected component of the free-edge graph is one end
cross-section, written out as a standalone 1-D SG and solved on its own:

![Left boundary ring](../_static/aperiodic_ring_L.png)

![Right boundary ring](../_static/aperiodic_ring_R.png)

For the prismatic validation case the same extraction runs on a cylinder,
where both ends are identical by construction:

![Prismatic cylinder segment used for the identity check](../_static/aperiodic_prismatic_segment.png)

## Acceptance test: the prismatic identity

For a **prismatic** segment the taper terms vanish and the segment
$6\times6$ must equal the boundary ring's own $6\times6$. Example
`examples/OpenSG_shell/5_taper_segment_aperiodic_boundaries` runs a
prismatic circular tube ($R=1$, $t/R=0.1$, $L=1.5$):

| case | worst diagonal error | worst term overall |
|---|---|---|
| isotropic (E = 70 GPa) | 0.0003 % (GA) | $2.7\times10^{-6}$ rel |
| ±45 anisotropic layup | 0.0007 % (GA) | $5.2\times10^{-5}$ rel |

Reproduce it from the example folder in two commands — the first writes the segment and its
two boundary-ring yamls, the second homogenizes all three and prints the identity table:

```bash
python make_cylinder_segment.py
```

```bash
python run_taper_segment.py
```

Any one of the generated yamls also runs on its own through the unified command, since the
segment header carries `msg: shell`, `n_model: 1` and `refined: 1`:

```bash
opensg meshes/seg_iso_hR0.1.yaml
```

## Tapered demonstration: BAR-URC segment 12

The same example runs a real tapered segment of the 100 m BAR-URC blade
(`run_bar_urc_segment.py`). The two boundary rings now differ, and every
segment diagonal falls **between** its two ring values
(ratio $=(S-R)/(L-R)\in[0,1]$):

| term | segment | ring L | ring R | ratio |
|---|---|---|---|---|
| EA  | 1.488e10 | 1.499e10 | 1.484e10 | 0.26 |
| GA2 | 6.098e8  | 6.248e8  | 6.031e8  | 0.31 |
| GA3 | 1.399e8  | 1.457e8  | 1.346e8  | 0.48 |
| GJ  | 3.354e8  | 3.684e8  | 3.057e8  | 0.47 |
| EI2 | 2.997e9  | 3.245e9  | 2.765e9  | 0.48 |
| EI3 | 8.972e9  | 9.621e9  | 8.397e9  | 0.47 |

## Shear-scheme rule (segments)

`shear="full"` (2×2 Gauss) is the production default for the segment
operators, and it is what `assemble_segment_indep` itself documents: with the
independent $\omega_3$, Dvorkin–Bathe tying aliases the algebraic drilling
shear. The tied variants (`mitc4_wonly`, `mitc4_g23`, `mitc4_both`) are
available as ablation options, not as recommended settings.
