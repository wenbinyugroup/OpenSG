# Equivalent 3-D Solid Properties from a Shell SG

Beyond the Timoshenko $6\times6$, the same 1-D shell structure gene can be homogenized to an
**equivalent 3-D solid** $6\times6$ — the anisotropic continuum stiffness of the cell, in the
solid Voigt order $[\,\bar\Gamma_{11}\;\bar\Gamma_{22}\;\bar\Gamma_{33}\;2\bar\Gamma_{23}\;
2\bar\Gamma_{13}\;2\bar\Gamma_{12}\,]$ with axis 1 along the prismatic direction. This is the
MSG thin-walled (MSG-TW) equivalent-continuum problem: a cellular or lattice cross-section
tiled periodically in the two transverse directions.

Where the beam problem drives the SG with four beam strains, the solid problem drives it with
the six macro strains, so **periodicity is always on** — opposite edges, and the corners, of the
2-D cell. The wall law is the Reissner–Mindlin $8\times8$ (`ABD` plus the MSG transverse-shear
$\mathbf{G}$), so the walls carry transverse shear that a classical (CLPT/Kirchhoff) wall model
discards.

```{note}
Runnable cases live in this repository under
`examples/OpenSG_shell/3_get_solid_props_from_shell_cross_section/`. Each folder holds a YAML
generator, the SG YAML, the driver, a timed `.out` in SwiftComp format, the mesh PNG, and a
`.dat` comparison table.
```

## Driver

```python
from opensg_shell import build_solid_bundle

b = build_solid_bundle("cell_1Dshell.yaml", cell_area=A_cell)
C3D = b["C3D"]            # 6x6 equivalent solid stiffness
b["solve_time"]           # seconds; also written into <yaml>_C3D.out
```

`build_solid_bundle` writes `<yaml>_C3D.out` by default — the SwiftComp `.K` layout (effective
stiffness, effective compliance, orthotropic-approximated engineering constants) with an
` OpenSG …` banner and a ` Time taken: … sec` footer.

## Validation — Deo & Yu (2023)

The reference is A. Deo and W. Yu, *Equivalent 3D properties of thin-walled composite structures
using mechanics of structure genome*, **Mechanics of Advanced Materials and Structures** 30(9),
2023, 1737–1748. Two columns are quoted from that paper: **MSG-TW**, its own thin-walled model
(classical wall law), and the reference **MSG solid** column, produced with SwiftComp — the
equivalent of an OpenSG 2-D-solid run. All values below are MPa; `%` is against the SwiftComp
column.

### Cellular solid, $\theta = +15^\circ$ (paper Table 1)

Aluminium $E=68.9$ GPa, $\nu=0.33$; $l=h=10$ cm, $t=5$ mm. Solve 1.2 s.

| term | OpenSG-TW (RM) | Deo MSG-TW | SwiftComp | ours % | Deo TW % |
|---|---|---|---|---|---|
| $C_{11}$ | 4733.84 | 4736.90 | 4678.90 | +1.17 | +1.24 |
| $C_{12}$ | 1086.01 | 1089.40 | 1105.50 | −1.76 | −1.46 |
| $C_{13}$ | 380.63 | 381.81 | 386.88 | −1.61 | −1.31 |
| $C_{22}$ | 2444.24 | 2446.39 | 2488.90 | −1.79 | −1.71 |
| $C_{23}$ | 846.69 | 847.44 | 860.89 | −1.65 | −1.56 |
| $C_{33}$ | 306.74 | 306.99 | 311.48 | −1.52 | −1.44 |
| **$C_{44}$** | **4.20** | 4.32 | 4.19 | **+0.18** | +3.09 |
| $C_{55}$ | 562.61 | 564.15 | 573.11 | −1.83 | −1.56 |
| $C_{66}$ | 993.78 | 997.52 | 1000.10 | −0.63 | −0.26 |

### Re-entrant cellular solid, $\theta = -15^\circ$ (paper Table 2)

The auxetic lattice, with the negative $C_{13}$, $C_{23}$ couplings. Solve 0.3 s.

| term | OpenSG-TW (RM) | Deo MSG-TW | SwiftComp | ours % | Deo TW % |
|---|---|---|---|---|---|
| $C_{11}$ | 7505.26 | 7507.40 | 7352.90 | +2.07 | +2.10 |
| $C_{12}$ | 1090.52 | 1094.10 | 1075.20 | +1.42 | +1.76 |
| $C_{13}$ | −219.81 | −220.53 | −213.94 | +2.74 | +3.08 |
| $C_{22}$ | 4151.29 | 4154.90 | 4080.00 | +1.75 | +1.84 |
| $C_{23}$ | −846.69 | −847.45 | −821.77 | +3.03 | +3.12 |
| $C_{33}$ | 180.61 | 180.75 | 173.48 | +4.11 | +4.19 |
| **$C_{44}$** | **2.47** | 2.68 | 2.47 | **+0.17** | +8.38 |
| $C_{55}$ | 331.26 | 332.17 | 346.01 | −4.26 | −4.00 |
| $C_{66}$ | 1687.82 | 1694.20 | 1706.40 | −1.09 | −0.71 |

$C_{44}$ is the in-plane shear channel — the one the paper flags as its largest error and
attributes to the classical wall model dropping the segment transverse shear, with a
first-order-shear wall model named as future work. **That is exactly the RM wall law used here:
the +3.1 % and +8.4 % errors drop to +0.18 % and +0.17 %.** Every other term tracks the paper's
own thin-walled column to a few tenths of a percent, so the two independent MSG-TW
implementations agree where they share physics.

### Hierarchical square (paper Table 3)

$R=10$ cm, $r=5$ cm, $t=5$ mm, same aluminium.

| term | OpenSG-TW (RM) | Deo MSG-TW | SwiftComp | ours % | Deo TW % |
|---|---|---|---|---|---|
| $C_{11}$ | 5183.48 | 5186.39 | 5099.03 | +1.66 | +1.71 |
| $C_{12}=C_{13}$ | 24.22 | 24.83 | 26.75 | −9.47 | −7.18 |
| $C_{22}=C_{33}$ | 46.23 | 47.13 | 50.46 | −8.39 | −6.60 |
| $C_{23}$ | 27.15 | 27.94 | 30.59 | −11.23 | −8.66 |
| $C_{44}$ | 3.20 | 3.22 | 3.39 | −5.60 | −5.01 |
| $C_{55}=C_{66}$ | 647.56 | 650.00 | 670.92 | −3.48 | −3.12 |

This lattice is **bending-dominated**: its transverse channels are $t^3$-scale
($C_{22}\approx 50$ MPa against $C_{11}\approx 5100$ MPa), so the load path is wall bending, not
wall stretching. Two consequences are worth recording.

*The RM result is mesh-converged and the offset is physics.* Refining from 10 to 40 elements per
segment moves $C_{23}$ by 1 %, and running the same model with shear-rigid walls (the Kirchhoff
limit) reproduces the paper's classical column to 0.0–0.3 % — $C_{12}=24.83$ and $C_{44}=3.22$
land exactly. The gap between the two columns is therefore the wall transverse-shear compliance,
which is real at the $t/l = 0.1$ ligaments; element order is not involved.

*Both thin-walled models miss the same junction stiffness.* The residual against SwiftComp is
the finite rotational stiffness of the joint regions, which no midline model carries. The
classical model's smaller apparent error is a partial cancellation — artificial shear rigidity
offsetting the missing joint stiffness — and the RM model removes the first error, exposing the
second.

### Composite square (paper Fig. 13 / Table 4)

Cell $25.4\times25.4$ mm, horizontal segments at $y_3=\pm5.84$ mm, one vertical segment through
the centre; every wall a $[15]_8$ laminate, $t=1.016$ mm; $E_1=141.96$, $E_2=E_3=9.79$,
$G=6.136$ GPa, $\nu=0.42$. Solve 1.7 s.

| term | OpenSG-TW (RM) | Deo MSG-TW | SwiftComp | ours % | Deo TW % |
|---|---|---|---|---|---|
| $C_{11}$ | 15263.83 | 16133.90 | 15518.11 | **−1.64** | +3.97 |
| $C_{12}$ | 936.74 | 1016.82 | 1008.76 | −7.14 | +0.80 |
| $C_{13}$ | 468.37 | 468.38 | 459.01 | +2.04 | +2.04 |
| $C_{15}$ | +1191.85 | −1192.81 | −1135.27 | sign | +5.07 |
| $C_{16}$ | 2383.70 | 2589.74 | 2535.23 | −5.98 | +2.15 |
| $C_{22}$ | 906.11 | 983.56 | 988.88 | −8.37 | −0.54 |
| $C_{23}$ | 0.00 | 0.00 | 20.44 | −100 | −100 |
| $C_{26}$ | 292.25 | 317.09 | 311.80 | −6.27 | +1.70 |
| $C_{33}$ | 453.06 | 453.06 | 456.25 | −0.70 | −0.70 |
| $C_{35}$ | +146.13 | −146.06 | −140.04 | sign | +4.30 |
| $C_{44}$ | 0.94 | 1.08 | 1.16 | −18.57 | −6.90 |
| $C_{55}$ | 547.31 | 547.32 | 545.43 | +0.35 | +0.35 |
| $C_{66}$ | 1094.63 | 1188.20 | 1188.30 | −7.88 | −0.01 |

$C_{13}$, $C_{33}$ and $C_{55}$ reproduce the paper's thin-walled column exactly, and $C_{11}$
improves on it against SwiftComp. Two open items are recorded honestly: $C_{15}$ and $C_{35}$
match the published magnitudes to 0.1 % with the **opposite sign**, a ply-angle handedness
convention still to be reconciled (and the likely cause of the −6…−8 % group), and $C_{23}$ is
zero in both thin-walled models — see below.

## Why $C_{23}$ vanishes on axis-aligned lattices

For a wall at angle $\varphi$ the only $\Gamma_e$ entries carrying weight in **both** the
$\bar\Gamma_{22}$ and $\bar\Gamma_{33}$ columns are the membrane $\varepsilon_{22}$ row
($\cos^2\varphi$, $\sin^2\varphi$) and the transverse-shear $2\gamma_{23}$ row
($-\sin\varphi\cos\varphi$, $+\sin\varphi\cos\varphi$), giving a per-length integrand

$$(A_{11}-G_{\rm msg})\,\sin^2\varphi\cos^2\varphi .$$

On a $0/90$ lattice every wall has $\sin^2\varphi\cos^2\varphi = 0$, so this $O(t)$ membrane path
vanishes identically and the whole of $C_{23}$ is the $O(t^2)$ **junction** contribution — the
wall-overlap blocks, which have zero measure on a midline mesh. Measured on a solid reference it
obeys $C_{23} \simeq C_{23}^{\rm 3D}\!\cdot A_{\rm junction}$ to within a few percent across
independent geometries.

Inclined-wall cells are unaffected: the same code returns $C_{23} = E' t L/\sqrt{2}$ to 0.02 %
on a $\pm45^\circ$ cell, which is why the cellular cases above carry $C_{23}$ correctly and only
the square lattices expose the junction term. Optional junction corrections
(`junction="census"`, `"micro"`, `"microcell"`) recover it for stretch-dominated cells; they do
**not** apply to bending-dominated lattices such as the hierarchical square, and are off by
default.
