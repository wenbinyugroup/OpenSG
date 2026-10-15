# Tapered shell segment with aperiodic boundaries

## Problem description

A cross-section analysis meshes only the section contour (a 1-D ring) and assumes the beam is
periodic along its axis. A tapered segment has no such periodicity: its two end cross-sections
differ. This example meshes a slice of the beam as a 3-D shell surface of four-node quads and
homogenizes it with `opensg_shell` to a Timoshenko $6\times6$ (`n_model: 1`, `refined: 1`). The
aperiodic segment route works in four steps:

1. Boundary extraction: a mesh edge used by exactly one quad is a free edge, and each connected
   component of the free-edge graph is one end cross-section. Each end is written as its own 1-D
   shell structure gene (SG) yaml, with the ring-node to segment-node map.
2. Boundary solves: each ring is homogenized on its own, giving its Timoshenko $6\times6$ and its
   warping fields $V_0$ (Euler-Bernoulli) and $V_1$ (Timoshenko), six degrees of freedom per node
   including the drilling rotation $\omega_3$.
3. Mapping: the ring $V_0$ and $V_1$ are imposed on the segment's boundary nodes as Dirichlet data.
   This replaces axial periodicity and fixes the rigid-body modes.
4. Segment solve: one LU factorization of the constrained system serves both orders,

   $$D_{hh}\,V_0 = -D_{he}, \qquad D_{hh}\,V_1 = b(V_0),$$

   with $V_0 = V_0^{\text{ring}}$ and $V_1 = V_1^{\text{ring}}$ on both ends. The energy is divided
   by the segment length $L$.

% source: examples/OpenSG_shell/5_taper_segment_aperiodic_boundaries/make_cylinder_segment.py (R, L, HR); docs/tutorials/taper_segment_aperiodic.md
Two cases are run. The acceptance test is a prismatic circular tube ($R=1$, $t/R=0.1$, $L=1.5$,
{numref}`fig-taper-prismatic`), whose two ends are identical, so the segment $6\times6$ must equal
the ring's own $6\times6$. The demonstration is segment 12 of the 100 m BAR-URC blade
({numref}`fig-taper-bar-urc`), whose two rings differ ({numref}`fig-taper-ring-l` and
{numref}`fig-taper-ring-r`).

```{figure} ../_static/aperiodic_prismatic_segment.png
:width: 60%
:name: fig-taper-prismatic

Prismatic cylinder segment used for the identity check.
```

```{figure} ../_static/aperiodic_segment_bar_urc.png
:width: 80%
:name: fig-taper-bar-urc

BAR-URC tapered segment 12 with its per-element material frames.
```

```{figure} ../_static/aperiodic_ring_L.png
:width: 70%
:name: fig-taper-ring-l

Left boundary ring of the BAR-URC segment, extracted from the mesh and solved as a 1-D SG.
```

```{figure} ../_static/aperiodic_ring_R.png
:width: 70%
:name: fig-taper-ring-r

Right boundary ring of the BAR-URC segment.
```

% source: examples/OpenSG_shell/5_taper_segment_aperiodic_boundaries/meshes/seg_iso_hR0.1.yaml, meshes/seg_aniso_hR0.1.yaml, BAR_URC_numEl_52_segment_12.yaml
| Name | Density | $E_1$ | $E_2$ | $E_3$ | $G_{12}$ | $G_{13}$ | $G_{23}$ | $\nu_{12}$ | $\nu_{13}$ | $\nu_{23}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| | kg/m³ | Pa | Pa | Pa | Pa | Pa | Pa | | | |
| iso | 1800.0 | 70000000000.0 | 70000000000.0 | 70000000000.0 | 26923000000.0 | 26923000000.0 | 26923000000.0 | 0.3 | 0.3 | 0.3 |
| ani | 1800.0 | 37000000000.0 | 9000000000.0 | 9000000000.0 | 4000000000.0 | 4000000000.0 | 4000000000.0 | 0.3 | 0.3 | 0.3 |
| glass_triax | 1940.0 | 28211400000.0 | 16238800000.0 | 15835500000.0 | 8248220000.0 | 3491240000.0 | 3491240000.0 | 0.497511 | 0.18091 | 0.27481 |
| glass_uni | 1940.0 | 43700000000.0 | 16500000000.0 | 15450000000.0 | 3265000000.0 | 3495000000.0 | 3480000000.0 | 0.262 | 0.264 | 0.35 |
| carbon_uni_industry_baseline | 1600.0 | 157599000000.0 | 9100000000.0 | 9100000000.0 | 4131000000.0 | 4131000000.0 | 2689000000.0 | 0.3133 | 0.3133 | 0.47066 |
| medium_density_foam | 130.0 | 142500000.0 | 142500000.0 | 142500000.0 | 54000000.0 | 54000000.0 | 54000000.0 | 0.3194 | 0.3194 | 0.3194 |
| glass_biax | 1940.0 | 11023100000.0 | 11023100000.0 | 16047700000.0 | 13231400000.0 | 3487480000.0 | 3487480000.0 | 0.688074 | 0.117173 | 0.117173 |

The BAR-URC yaml also defines `Gelcoat`, `Adhesive`, `resin` and `steel`, which no layup of this
segment uses. Layups list the outer ply first, as [material, ply thickness in m, angle in °]:

% source: examples/OpenSG_shell/5_taper_segment_aperiodic_boundaries/meshes/seg_iso_hR0.1.yaml, meshes/seg_aniso_hR0.1.yaml, BAR_URC_numEl_52_segment_12.yaml
| Case | Layup | Ply 1 (outer) | Ply 2 | Ply 3 |
|---|---|---|---|---|
| tube, isotropic | tube | iso, 0.1, 0.0 | | |
| tube, ±45 | tube | ani, 0.05, 45.0 | ani, 0.05, -45.0 | |
| BAR-URC | layup_0 | glass_triax, 0.005, 0 | glass_uni, 0.01, 0 | glass_triax, 0.003, 0 |
| BAR-URC | layup_1 | glass_triax, 0.005, 0 | medium_density_foam, 0.048, 0 | glass_triax, 0.003, 0 |
| BAR-URC | layup_2 | glass_triax, 0.005, 0 | carbon_uni_industry_baseline, 0.04, 0 | glass_triax, 0.003, 0 |
| BAR-URC | layup_3 | glass_triax, 0.005, 0 | medium_density_foam, 0.07, 0 | glass_triax, 0.003, 0 |
| BAR-URC | layup_4 | glass_triax, 0.005, 0 | glass_uni, 0.005, 0 | glass_triax, 0.003, 0 |
| BAR-URC | layup_5 | glass_triax, 0.005, 0 | medium_density_foam, 0.038, 0 | glass_triax, 0.003, 0 |
| BAR-URC | layup_6 | glass_triax, 0.005, 0 | medium_density_foam, 0.076, 0 | glass_triax, 0.003, 0 |
| BAR-URC | layup_7 | glass_biax, 0.002, 0 | medium_density_foam, 0.049, 0 | glass_biax, 0.002, 0 |
| BAR-URC | layup_8 | glass_biax, 0.002, 0 | medium_density_foam, 0.029, 0 | glass_biax, 0.002, 0 |

Input files are in `examples/OpenSG_shell/5_taper_segment_aperiodic_boundaries/`.

## Input

Every segment yaml opens with the same header. `aperiodic: 1` documents that the ends carry the
boundary rings' own $V_0$ and $V_1$; on a surface mesh with `n_model: 1` the route is aperiodic by
construction.

% source: examples/OpenSG_shell/5_taper_segment_aperiodic_boundaries/BAR_URC_numEl_52_segment_12.yaml
```yaml
n_model: 1      # 1 = beam, 2 = plate (opensg_solid), 3 = solid -- the macro model this SG homogenizes to
refined: 1      # 0 = classical (beam EB 4x4, KL wall); 1 = shear-refined (beam Timoshenko 6x6, RM wall)
msg: shell      # the ENGINE this SG belongs to (opensg_shell); `opensg <yaml>` dispatches on it
aperiodic: 1    # the two end cross-sections carry their own V0/V1 as Dirichlet data (no axial periodicity)
```

The blocks `nodes`, `elements` (quads), `sets`, `sections`, `elementOrientations` and `materials`
follow. No boundary file is prepared by hand: the ends are found from the mesh topology. The tube
yamls are written by `make_cylinder_segment.py`, whose geometry parameters are:

% source: examples/OpenSG_shell/5_taper_segment_aperiodic_boundaries/make_cylinder_segment.py
```python
R = 1.0            # wall mid-surface radius [m]
L = 1.5            # segment length [m] (prismatic => value is immaterial to 6x6)
HR = [0.05, 0.1, 0.2]   # wall/radius ratios: thin, moderate, thick
```

% source: examples/OpenSG_shell/5_taper_segment_aperiodic_boundaries/make_cylinder_segment.py
The mesh has `NC = 160` quads around the hoop and `NL = 3` quads along the axis. The beam axis is
$x$; each element frame has $e_1$ axial, $e_2$ along the hoop and $e_3$ the inward normal.

## Run

From the example folder, the first command writes the tube yamls, the second homogenizes the
$t/R = 0.1$ tubes and writes the identity table:

```bash
python make_cylinder_segment.py
```

```bash
python run_taper_segment.py
```

Each generated yaml also runs on its own:

```bash
opensg meshes/seg_iso_hR0.1.yaml
```

% source: examples/OpenSG_shell/5_taper_segment_aperiodic_boundaries/meshes/seg_iso_hR0.1_Timo.out
The command prints the law under `Timoshenko Beam Stiffness Matrix  [eps11 gam12 gam13 kappa1 kappa2 kappa3]:`,
then `Homogenization stored in meshes/seg_iso_hR0.1_Timo.out` and the `Time taken` line (1.98 sec in
the shipped file). The tapered case runs with:

```bash
python run_bar_urc_segment.py
```

In Python the whole pipeline is one call on the 3-D shell yaml:

```python
from opensg_shell.sg_homo import segment_timo_from_3dyaml

r = segment_timo_from_3dyaml("meshes/seg_iso_hR0.1.yaml")
r["S6"]             # segment Timoshenko 6x6
r["C6L"], r["C6R"]  # the two boundary-ring 6x6s
r["L"]              # segment length used to normalize the energy
```

Each run writes the two boundary 1-D yamls (`boundary_<name>_boundaries_L.yaml`, `..._R.yaml`), the
`<name>_boundaries.npz` bundle, the timed `<name>_Timo.out` and three orientation PNGs (segment, left
ring, right ring). The PNGs are written only when they do not exist yet; delete them after changing
a mesh to refresh them.

:::{note}
The segment operators use `shear="full"` (2×2 Gauss) by default: with the independent $\omega_3$,
Dvorkin-Bathe tying aliases the algebraic drilling shear. The tied variants `mitc4_wonly`,
`mitc4_g23` and `mitc4_both` are ablation options, not recommended settings.
:::

## Result

### Acceptance test: the prismatic identity

% source: docs/tutorials/taper_segment_aperiodic.md (acceptance-test table); per-term errors in examples/OpenSG_shell/5_taper_segment_aperiodic_boundaries/seg_ring_identity.dat
| case | worst diagonal error | worst term overall |
|---|---|---|
| isotropic (E = 70 GPa) | 0.0003 % (GA) | $2.7\times10^{-6}$ rel |
| ±45 anisotropic layup | 0.0007 % (GA) | $5.2\times10^{-5}$ rel |

The diagonals, segment against ring, with the per-term error in percent:

% source: examples/OpenSG_shell/5_taper_segment_aperiodic_boundaries/seg_ring_identity.dat
| term | iso segment | iso ring | % | ±45 segment | ±45 ring | % |
|---|---|---|---|---|---|---|
| $EA$ [N] | 4.397947e+10 | 4.397947e+10 | 0.0000 | 7.706160e+09 | 7.706160e+09 | 0.0000 |
| $GA_2$ [N] | 8.473281e+09 | 8.473258e+09 | 0.0003 | 3.262832e+09 | 3.262807e+09 | 0.0007 |
| $GA_3$ [N] | 8.473281e+09 | 8.473258e+09 | 0.0003 | 3.262832e+09 | 3.262807e+09 | 0.0007 |
| $GJ$ [N·m²] | 1.694078e+10 | 1.694078e+10 | -0.0000 | 6.526891e+09 | 6.526891e+09 | -0.0000 |
| $EI_2$ [N·m²] | 2.200273e+10 | 2.200269e+10 | 0.0002 | 3.855225e+09 | 3.855204e+09 | 0.0006 |
| $EI_3$ [N·m²] | 2.200273e+10 | 2.200269e+10 | 0.0002 | 3.855225e+09 | 3.855204e+09 | 0.0006 |

The ±45 tube also carries the extension-twist coupling 7.871078e+07 in both matrices and the
shear-bending coupling -3.684965e+07 (segment) against -3.684772e+07 (ring), 0.0052 %. The two
solves take 2.0 s and 1.7 s. The identity holds on every diagonal to within the errors above, which
validates the boundary extraction and the $V_0$/$V_1$ mapping.

### Tapered demonstration: BAR-URC segment 12

% source: examples/OpenSG_shell/5_taper_segment_aperiodic_boundaries/BAR_URC_numEl_52_segment_12_Timo.out (banner)
The segment is $L$ = 3.45293 m long. Its Timoshenko stiffness, order
$[\varepsilon_{11}\;\gamma_{12}\;\gamma_{13}\;\kappa_1\;\kappa_2\;\kappa_3]$ (N, N·m, N·m²):

% source: examples/OpenSG_shell/5_taper_segment_aperiodic_boundaries/BAR_URC_numEl_52_segment_12_Timo.out
```text
 OpenSG msg-shell tapered/aperiodic segment (boundary V0/V1 Dirichlet, L=3.45293)

 The Effective Timoshenko Beam Stiffness Matrix
 --------------------------------------------
     1.4877161E+010     5.0696312E+006    -2.8650111E+005     2.0187837E+007     2.4197462E+008    -4.1789564E+009
     5.0696312E+006     6.0982290E+008    -4.4541479E+007    -4.5851602E+007    -3.4116361E+006    -1.5344127E+007
    -2.8650111E+005    -4.4541479E+007     1.3989455E+008     8.5130241E+007     3.4083338E+006     1.0288430E+006
     2.0187837E+007    -4.5851602E+007     8.5130241E+007     3.3535382E+008     1.1816262E+007     2.2311883E+007
     2.4197462E+008    -3.4116361E+006     3.4083338E+006     1.1816262E+007     2.9966857E+009     5.2379719E+008
    -4.1789564E+009    -1.5344127E+007     1.0288430E+006     2.2311883E+007     5.2379719E+008     8.9717075E+009
```

The file closes with ` Time taken: 18.46 sec`. Here no identity holds: each ring has its own
$6\times6$, and every segment diagonal falls between its two ring values. The ratio
$(S-R)/(L-R)$ lies in $[0, 1]$ for all six terms:

% source: examples/OpenSG_shell/5_taper_segment_aperiodic_boundaries/bar_urc_segment12.dat
| term | segment | ring L | ring R | ratio |
|---|---|---|---|---|
| $EA$ | 1.487716e+10 | 1.498819e+10 | 1.483824e+10 | 0.260 |
| $GA_2$ | 6.098229e+08 | 6.248039e+08 | 6.030954e+08 | 0.310 |
| $GA_3$ | 1.398945e+08 | 1.457255e+08 | 1.345727e+08 | 0.477 |
| $GJ$ | 3.353538e+08 | 3.684346e+08 | 3.057355e+08 | 0.472 |
| $EI_2$ | 2.996686e+09 | 3.244829e+09 | 2.765385e+09 | 0.482 |
| $EI_3$ | 8.971707e+09 | 9.620926e+09 | 8.397279e+09 | 0.469 |

## Files

| File | Content |
|---|---|
| `make_cylinder_segment.py` | writes `meshes/seg_{iso,aniso}_hR{0.05,0.1,0.2}.yaml` |
| `run_taper_segment.py` | prismatic identity, writes `seg_ring_identity.dat` |
| `run_bar_urc_segment.py` | tapered segment, writes `bar_urc_segment12.dat` |
| `BAR_URC_numEl_52_segment_12.yaml` | the BAR-URC tapered segment SG |
| `meshes/seg_*_hR0.1_Timo.out`, `BAR_URC_numEl_52_segment_12_Timo.out` | timed segment $6\times6$ and compliance |
| `boundary_*_boundaries_L.yaml`, `boundary_*_boundaries_R.yaml` | the extracted end rings as 1-D SGs |
| `*_boundaries.npz` | segment and ring meshes, element frames, materials, sections and the ring-to-segment node maps |
| `orient_*_boundaries_segment.png`, `orient_*_ring_L.png`, `orient_*_ring_R.png` | element frames of the segment and rings |
