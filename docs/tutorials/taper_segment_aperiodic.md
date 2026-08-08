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

```python
from opensg_shell.sg_homo import segment_timo_from_3dyaml

r = segment_timo_from_3dyaml("seg_iso_hR0.1.yaml")   # 3-D shell yaml in, ...
r["S6"]            # segment Timoshenko 6x6
r["C6L"], r["C6R"] # the two boundary-ring 6x6s
```

Each run writes the boundary 1-D yamls, the orientation-frame PNGs, and the
timed `<yaml>_Timo.out` (SwiftComp layout).

## Acceptance test: the prismatic identity

For a **prismatic** segment the taper terms vanish and the segment
$6\times6$ must equal the boundary ring's own $6\times6$. Example
`examples/OpenSG_shell/5_taper_segment_aperiodic_boundaries` runs a
prismatic circular tube ($R=1$, $t/R=0.1$, $L=1.5$):

| case | worst diagonal error | worst term overall |
|---|---|---|
| isotropic (E = 70 GPa) | 0.0003 % (GA) | $2.7\times10^{-6}$ rel |
| ±45 anisotropic layup | 0.0007 % (GA) | $5.2\times10^{-5}$ rel |

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
operators: MITC tying aliases the algebraic drilling content on flat-walled
and webbed sections. Reserve MITC ties for very thin walls
($t/R \sim 0.02$); below the RM validity range use Kirchhoff–Love.
