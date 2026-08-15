# Plate Properties and Recovery (msg-solid)

A plate (or shell wall) is described by a $6\times6$ ABD law relating the section resultants
$[N_{11}, N_{22}, N_{12}, M_{11}, M_{22}, M_{12}]$ to the plate strains
$[\varepsilon_{11}, \varepsilon_{22}, \gamma_{12}, \kappa_{11}, \kappa_{22}, \kappa_{12}]$, or by
the $8\times8$ Reissner-Mindlin law that adds the transverse shears $Q_1, Q_2$ against
$2\gamma_{13}, 2\gamma_{23}$. `msg-solid` builds either one from a structure gene, and then runs
the same construction backwards to recover the 3-D stress and strain inside that gene. Which
structure gene you supply depends on what the plate is made of:

| route | structure gene | entry point | macro law |
|---|---|---|---|
| **through-thickness** | a 1-D line mesh through the wall thickness, one higher-order element per ply | `opensg_solid.rm_plate_1D.msg_rm_plate.rm_plate_msg` | $6\times6$ ABD (`A6`) or $8\times8$ ABDG |
| **in-plane cell** | a 2-D mesh of one periodic panel cell (a honeycomb, a corrugation, a truss core) | `opensg_solid.sg_homo.plate_homo_2d` | $6\times6$ ABD (`C_eff`); $8\times8$ with `shear_refined=True` |

The first route is the laminate case: the wall is a stack of plies and nothing varies in-plane, so
the gene is one-dimensional. The second route is the cellular case: the wall has in-plane
microstructure, so the gene is the cell itself. Both write the same timed SwiftComp `.K`-layout
`.out` file — effective stiffness, effective compliance, and a closing ` Time taken:` line in
seconds.

Four runnable examples cover the pair of capabilities in both directions:

| example folder | script | does |
|---|---|---|
| `examples/OpenSG-solid/1_get_plate_props_from_1DSG` | `1d_sg.py` | layup $\rightarrow$ 1-D SG $\rightarrow$ plate law |
| `examples/OpenSG-solid/2_get_plate_stress_from_1DSG` | `rm_dehom.py`, `rm_dehom_ff.py` | plate resultants $\rightarrow$ 3-D ply stress through the thickness |
| `examples/OpenSG-solid/3_get_plate_props_from_2D_SG` | `opensg <name>.yaml` | 2-D cell SG $\rightarrow$ plate ABD (`refined: 0`) or ABDG (`refined: 1`) |
| `examples/OpenSG-solid/4_get_plate_dehom_from_2DSG` | `plate_dehom_2dsg.py` | macro plate strain $\rightarrow$ 3-D fields inside the cell |

## A. Through-thickness 1-D SG $\rightarrow$ the plate law

### What it computes

The structure gene is the wall discretised through its own thickness: one element per ply, quartic
by default, with the nodes carrying the three warping (fluctuation) degrees of freedom. Minimising
the 3-D strain energy over that line mesh order by order in $h/\ell$ gives the warping ladder — the
zeroth-order columns $V_0$, and the first-gradient columns $V_{11}, V_{12}$ — from which the plate
stiffness follows as

$$A_6 = \big\langle(\Gamma_\varepsilon + \Gamma_h V_0)^{\mathsf T}\,C(z)\,
(\Gamma_\varepsilon + \Gamma_h V_0)\big\rangle ,$$

$\langle\cdot\rangle$ being the through-thickness integral and $C(z)$ the ply stiffness rotated by
its fibre angle. With `model: 1` the run also identifies the $2\times2$ MSG transverse-shear block
$G$ from the second-order energy and assembles the $8\times8$ law

$$\text{ABDG} = \begin{bmatrix} A_6 & 0 \\ 0 & G \end{bmatrix},$$

with the transverse shears uncoupled from the membrane and bending measures. Row and column order
is `[e11 e22 g12 k11 k22 k12 2g13 2g23]`, printed in the banner of every `.out`.

### The input you edit

Everything lives in `layup_db.yaml`. The script reads that file and nothing else — a new laminate
means editing the YAML, never the code.

| key | what it sets | value in the shipped file |
|---|---|---|
| `model` | `0` = classical $6\times6$ ABD only, `1` = $8\times8$ ABD + transverse shear | `1` |
| `fraction` | reference surface as a fraction of thickness: `0` = bottom/OML face, `0.5` = mid-surface, `1` = top | `0.5` |
| `mesh: n_per_layer` | SG elements per physical ply | `1` |
| `mesh: elem_order` | element order; `4` = 5-noded quartic | `4` |
| `materials` | per material `E: [E1, E2, E3]`, `G: [G12, G13, G23]`, `nu: [nu12, nu13, nu23]`, `rho` | `ge`, `herex` |
| `layup` | one entry per ply, **bottom ply first**: `{material, thickness, angle}` | 9 plies |

The shipped case is a $[0/90/0/90/\text{core}]_s$ sandwich: eight graphite/epoxy plies of
$0.9525$ mm ($E_1 = 128$ GPa, $E_2 = E_3 = 11$ GPa, $\rho = 1500$ kg/m³) around a $144.78$ mm
HEREX C70.130 PVC foam core ($E = 103.63$ MPa, $G = 50$ MPa, $\rho = 130$ kg/m³), total
$h = 0.1524$ m.

```{note}
`fraction` is a property of the **mesh**, not of the homogenization call: the generated node
coordinates are measured from the reference plane, so they run from $-\text{fraction}\cdot h$ to
$(1-\text{fraction})\cdot h$ and $x_3 = 0$ *is* the reference. The $B$ block of the ABD depends on
this choice, so homogenize, analyze and recover with the same `fraction`.
```

The same `layup_db.yaml` also carries `V2`, a `plate:` block and a `dehom:` block. Those are read
by the recovery scripts of section B; `1d_sg.py` itself uses only the six keys in the table.

### Run it

```bash
python 1d_sg.py
```

A different database file can be passed as the single positional argument:

```bash
python 1d_sg.py my_other_layup.yaml
```

### What comes out

| file | content |
|---|---|
| `1dsg.yaml` | the generated 1-D SG mesh: an `sg:` header (`type: plate_1d`, `elem_order: 4`, `n_per_layer: 1`, `reference_fraction: 0.5`, `thickness`, `n_ply`), 37 nodes given as single $x_3$ coordinates from $-0.0762$ to $+0.0762$ m, 9 five-noded elements, the material blocks with densities, and one element set plus one `sections:` entry per ply carrying its material, angle and thickness |
| `1dsg.png` | the mesh figure, plies coloured by material |
| `layup_db_plate_homo.out` | the plate law, named `<layup_db stem>_plate_homo.out` |

The `.out` opens with a banner that records the whole run, then the matrix whose name follows
`model` — `Reissner-Mindlin` here, `Classical Plate` at `model: 0`:

```
 OpenSG msg-solid plate model from 1dsg.yaml: 9 plies, h = 0.152400 m, fraction = 0.5, rho*h = 30.251400 kg/m^2, rows [e11 e22 g12 k11 k22 k12 2g13 2g23]

 The Effective Reissner-Mindlin Stiffness Matrix
 --------------------------------------------
     5.4916502E+008     2.6417019E+007     1.5524611E-010     3.6182957E-006     3.6122933E-006     1.0244011E-024     0.0000000E+000     0.0000000E+000
```

The effective compliance follows in the same layout, and the file closes with
` Time taken: 0.82 sec`. Engineering constants are printed only for the 3-D solid law, never for a
plate law.

### What the numbers mean

The eight diagonal entries of the shipped run, in SI (the code is unit-agnostic and simply follows
the input, which is metres and pascals here):

| term | value | units |
|---|---|---|
| $A_{11}$ | $5.4916502\times10^{8}$ | N/m |
| $A_{22}$ | $5.4916502\times10^{8}$ | N/m |
| $A_{66}$ | $4.1376600\times10^{7}$ | N/m |
| $D_{11}$ | $3.0005458\times10^{6}$ | N·m |
| $D_{22}$ | $2.9371144\times10^{6}$ | N·m |
| $D_{66}$ | $2.0111708\times10^{5}$ | N·m |
| $G_{11}$ | $7.7009772\times10^{6}$ | N/m |
| $G_{22}$ | $7.5423712\times10^{6}$ | N/m |

Three checks are worth making on any run of your own, all visible in this one:

- **The section mass.** The banner's $\rho h = 30.2514$ kg/m² is
  $8 \times 0.0009525 \times 1500 + 0.14478 \times 130 = 11.43 + 18.8214$ kg/m², so the layup that
  reached the solver is the layup you wrote.
- **The $B$ block.** The stack is symmetric about the mid-surface and `fraction: 0.5` puts the
  reference plane there, so membrane-bending coupling must vanish. It does: the largest entry of
  the $B$ block is $3.6\times10^{-6}$ against an $A_{11}$ of $5.5\times10^{8}$, fourteen orders
  down.
- **The transverse-shear block.** $G_{11} = 7.70\times10^{6}$ N/m sits 6 % above the crude
  core-only estimate $G_{\rm core} h_{\rm core} = 50\times10^{6} \times 0.14478 = 7.24\times10^{6}$
  N/m, which is the expected magnitude for a sandwich whose shear is carried almost entirely by
  the foam. $A_{11}$, by contrast, is set by the stiff faces.

![Through-thickness 1-D structure gene of the sandwich laminate](../_static/plate_1dsg.png)

The figure is `1dsg.png` as written by the run: the gene is a line, drawn with thickness for
legibility, the thin graphite/epoxy face plies at top and bottom and the foam core between them,
with the dashed line marking the reference surface at mid-thickness.

### What it is validated against

`src/opensg_solid/rm_plate_1D/tests/test_msg_rm_plate.py` gates the construction on analytic
anchors and on an independent plain-NumPy re-assembly of the same SG matrices: for an isotropic
plate with $\nu = 0$ the transverse-shear block comes out exactly $\tfrac56 G h$, the textbook
shear-correction factor, with the least-squares residual $U^*$ at machine zero;
$A_6$ equals the classical lamination-theory
`msg_materials.compute_ABD_matrix` at the matching reference plane; $A_6(\text{fraction})$ equals
the offset transform of $A_6(0)$; and the warping fields satisfy the Euler-Lagrange equations and
the gauge condition $\psi^{\mathsf T} H V = 0$ for $V_0$, $V_{11}$ and $V_{12}$ directly. The
formulation is Yu, Hodges and Volovoi, *Computers & Structures* 81 (2003) 439–454.

## B. Through-thickness recovery: plate resultants $\rightarrow$ 3-D ply stress

### What it computes

Homogenization threw information away; recovery puts it back. Given the section resultants at a
point of the plate, the recovery evaluates the 3-D strain, stress, warping and displacement at
every depth of the gene. The construction is linear in **42 drivers**,

$$D = [\,E_6,\; E_{6,1},\; E_{6,2},\; E_{6,11},\; E_{6,12},\; E_{6,22},\; qt_6\,],$$

so the code builds three operators once per depth from 42 unit evaluations —
$T_E, T_S \in \mathbb{R}^{6\times42}$ and $T_W \in \mathbb{R}^{3\times42}$ — and every station is
then one matrix-vector product, `jax.vmap`-ed over the whole field. The plate strains come from
the section law itself, $E_6 = A_6^{-1} F_6$, so a recovery never bypasses the homogenization.

Which driver feeds which component is the thing to understand before choosing an input:

| driver | where it comes from | what it produces | without it |
|---|---|---|---|
| $E_6$ | $A_6^{-1} F_6$ at the station | $\sigma_{11}, \sigma_{22}, \sigma_{12}$ layer by layer | nothing |
| $E_{6,1}, E_{6,2}$ | in-plane gradients of the resultants | $\sigma_{13}, \sigma_{23}$ (the $V_{11}/V_{12}$ warping) | $\sigma_{13} = \sigma_{23} \equiv 0$ |
| $E_{6,11}, E_{6,12}, E_{6,22}$ | second gradients | the $\sigma_{33}$ interior build-up | $\sigma_{33}$ keeps only its face terms |
| $qt_6$ | the **applied** surface pressure ladder $[q,\, q_{,1},\, q_{,2},\, q_{,11},\, q_{,12},\, q_{,22}]$ | the face traction $\sigma_{33} = -q$ | $\sigma_{33} = 0$ at that face |

$Q_1, Q_2$ are deliberately not part of $F_6$: there is no shear slot among the 42 drivers, because
in the asymptotic construction the transverse shears are produced by the gradient brackets. They
enter only as the per-station rescale targets $\int\sigma_{13}\,dz = Q_1$,
$\int\sigma_{23}\,dz = Q_2$, so the profile *shape* comes from the gradients and the carried force
is exactly the one you supplied.

### Two drivers, two kinds of input

**`rm_dehom.py` — one station, resultants typed into the script.** The input is the `USER INPUT`
block at the bottom of the file: `SG_YAML` (the mesh from section A), `F6` and `Q2v`, plus the
optional gradient and load ladders. `F6` and `Q2v` ship as `None` and the script raises rather than
run, by design — a built-in number would masquerade as a result. `LATTICE` selects the output
depths, `"gauss"` (default) or `"nodes"`.

```python
SG_YAML = os.path.join(HERE, "1dsg.yaml")
F6 = [1.0e6, 0.0, 0.0, 5.0e4, 0.0, 0.0]   # [N11, N22, N12, M11, M22, M12]  [N/m, N]
Q2v = [0.0, 0.0]                          # [Q1, Q2]  [N/m]
DE1 = DE2 = None                          # d(E6)/dx, d(E6)/dy
D11 = D12 = D22 = None                    # second gradients
QT6 = None                                # [q q,1 q,2 q,11 q,12 q,22]
LATTICE = "gauss"
```

```bash
python rm_dehom.py
```

It writes `1dsg_dehom.SM`, `.EM`, `.U` and `.vtk` — the VTK is part of the default output, not an
extra step — and prints `max|S11|` through `max|S23|`. With every optional input left at `None`
this is the honest classical recovery: $\sigma_{11}, \sigma_{22}, \sigma_{12}$ exact layer by
layer, $\sigma_{13} = \sigma_{23} = 0$ exactly, $\sigma_{33}$ at machine zero. One station has no
neighbours, so it cannot supply gradients, and the code does not invent them.

**`rm_dehom_ff.py` — a whole field from a `.ff` station table.** This is the general driver, and
the one to use for real work. A `.ff` file is the plate analogue of a VABS `.glb`: whitespace
separated, `#` comments, one row per in-plane station, in any row order and on any station
geometry. The shipped table was collected from an Abaqus S8R shell field, but any solver can
write it — the driver contains no Abaqus. Its first three lines state the format:

```
# plate dehom FF station table (collected from ex5_shell_S8R_field.dat)
# x[m] y[m] N11 N22 N12[N/m] M11 M22 M12[N] Q1 Q2[N/m] u1 u2 u3[m] wx wy[-]
# 40 x 40 rectangular station lattice, x-fastest
```

Ten columns are required (`x y N11 N22 N12 M11 M22 M12 Q1 Q2`); the optional columns 11–15 add the
mid-surface displacement $u_1, u_2, u_3$ and slopes $w_x, w_y$, and without them the `.U` file
carries the warping alone. **Every gradient driver is computed from the stations themselves**, by
one of two paths chosen automatically: exact tensor-grid finite differences when the coordinates
form a complete rectangular lattice, or a meshfree KD-tree patch fit ($k = 12$ nearest neighbours,
distance-weighted local quadratic least squares, first and second derivatives from one batched
solve) for any other cloud. `--scatter` forces the meshfree path.

The applied face pressure is *not* in the table — loads are input, not solver output — so it lives
in `layup_db.yaml`:

```yaml
dehom:
  ff: Abaqus_results/ex5_shell_S8R_field.ff
  loads:
    top: {q0: 68.9476e6, shape: sinsin}   # uniform | sin_x | sin_y | sinsin
    bottom: none
```

Each `shape` fixes the whole six-term ladder analytically, differentiated with `a` and `b` from the
`plate:` block. Bottom-face loads raise `NotImplementedError`: the 42-driver operator block carries
the top ladder only.

```bash
python rm_dehom_ff.py ex5_shell_S8R_field.ff
```

Pass the table explicitly in this folder. The shipped `dehom: ff:` key reads
`Abaqus_results/ex5_shell_S8R_field.ff`, and no `Abaqus_results` directory is shipped — the table
sits at the folder root — so a no-argument invocation fails until you either name the file on the
command line, as above, or correct the key. `--db` points at a database elsewhere.
The run prints its chosen gradient path (`lattice FD (40 x 40)` for this table), the operator build,
the vmap over stations, and the peak of each stress component.

### What comes out

Outputs are named after the table, `<ff stem>_ff`:

| file | content |
|---|---|
| `ex5_shell_S8R_field_ff.SM` | 3-D stress: columns `x y z` then `S11 S22 S33 S12 S13 S23` in Pa |
| `ex5_shell_S8R_field_ff.EM` | 3-D strain, same layout, `E11` through `E23` |
| `ex5_shell_S8R_field_ff.U` | 3-D displacement `U1 U2 U3` in m |
| `ex5_shell_S8R_field_ff_gauss.vtk` | ASCII `STRUCTURED_GRID`, `DIMENSIONS 40 40 45`, 72 000 points, written only on a lattice |

The three text files share a two-line header naming the recovery and its station set, for example
`1600 .ff stations [lattice FD (40 x 40)] x the 45 SG-element Gauss depths`. The conventions that
matter when you read them:

- $z$ is in the **SG frame**: $z = 0$ is the reference surface, not the mid-plane, unless
  `fraction` put them at the same place.
- The default depths are the Gauss abscissae of each five-noded element — five per ply here, 45
  in all — so every output point is strictly inside one ply and a material interface, where stress
  is genuinely double-valued, never appears in a file. `LATTICE = "nodes"` switches to the SG nodes
  in the single-station `rm_dehom.py`; `rm_dehom_ff.py` has no `LATTICE` variable at all and
  always writes the Gauss depths.
- Stress and strain are written in the **ply material frame** of each depth, which is where failure
  criteria live. The chain computes in the laminate frame — it must, since the gradients and the
  $Q$-rescale integrate across plies — and rotates only at write time. The displacement `.U` stays
  global.
- Component order in the files is `11 22 33 12 13 23`, reordered from the internal SG Voigt
  order `(11, 22, 33, 23, 13, 12)` on write.

### What it is validated against

The recovery is measured against reference elasticity solutions — the example tree carries
`pagano_S4/S10/S50.dat` and `yu_case1/2/3` reference data under
`examples/OpenSG-solid/CS_OpeNSG_exampels/static/`. The accuracy table in
`src/opensg_solid/rm_plate_1D/USER_GUIDE.md` records errors falling at $\mathcal{O}(S^{-2})$ in the
slenderness $S$: cross-ply $\sigma_{13}$ is 21 % at $S = 4$, 4.4 % at $S = 10$ and 0.18 % at
$S = 50$; equilibrium $\sigma_{33}$ is 5.5 % / 1.2 % / 0.05 %; the second-order in-plane
$\sigma_{11}$ is 27 % / 2.0 % / 0.04 % at $S = 64$. Soft-core sandwiches need a larger $S$ for the
in-plane second-order recovery; their $\sigma_{13}$, $\sigma_{33}$ and displacements stay accurate
at $S = 10$. Separately, the
finite-difference gradient path and the analytically differentiated known-mode path were run side
by side on the same field and agree to **better than 1 % on every component** in the interior (four
integration points trimmed per edge); the 11–15 % whole-field figure is entirely the one-sided
stencil band about two elements wide at the boundary. `dehom_README.md` in the example folder
derives the full chain equation by equation and states exactly which term is Yu's and which is this
project's.

## C. 2-D SG $\rightarrow$ plate ABD

### What it computes

When the wall has in-plane microstructure — a honeycomb core, a corrugation, a truss — the gene is
a 2-D mesh of one periodic cell, and `plate_homo_2d` homogenizes it to the plate ABD with
`n_model=2`. The same function covers `n_model=1` (Timoshenko beam) and `n_model=3` (equivalent
3-D solid) from the same mesh, and reads 1-D, 2-D or 3-D structure genes. Opposite faces, edges and
corners are tied through the sparse periodic map; periodic is the default and the correct treatment
for a periodic medium.

### The input you edit

The yaml **is** the whole problem — nothing lives in the code, and every header key has a
default. The header is the leading scalar keys above the mesh blocks:

```yaml
n_model: 2      # 1 = beam, 2 = plate, 3 = solid -- the macro model this SG homogenizes to
refined: 1      # 0 = classical (plate ABD / beam EB); 1 = shear-refined (plate ABDG / beam Timoshenko)
msg: solid      # the ENGINE this SG belongs to (opensg_solid); `opensg <yaml>` dispatches on it
```

`refined` upgrades the macro law within the chosen `n_model`: for the plate, `0` is the
classical $6\times6$ ABD and `1` the shear-refined $8\times8$ ABDG; for the beam, `0` is the
classical Euler–Bernoulli $4\times4$ and `1` the Timoshenko $6\times6$; the solid law has no
refined variant. `0` is the default when the key is absent; the file shipped here asks for
`1`, so it reports the ABDG. There is no `dim` and no
`scale` key: the SG dimension is inferred from the mesh and the measure $\omega$ is computed
from it. `analysis:` defaults to `H`; with `analysis: D` the header additionally carries
`epsilon_bar:`, the $(6,)$ macro
state, and the run recovers the local 3-D fields right after the homogenization, writing the
`<name>_dehom.*` files of section D. Explicit arguments to `plate_homo_2d` always override
the header (the API's legacy beam default remains Timoshenko, which is what examples 7–8
rely on).

The materials live in the SG YAML itself, exactly as in a SwiftComp `.sc`: each entry of the
file's `materials:` block defines one material **either** by its $6\times6$ elastic stiffness
(`type: 2`, stored pre-rotated, used as-is) **or** by the nine orthotropic engineering constants
`[E1, E2, E3, G12, G13, G23, nu12, nu13, nu23]` together with the ply angle (`type: 1` plus
`angle:`, rebuilt and rotated in-code). This example ships the constants + angle form:

```yaml
materials:
  1:
    type: 1
    engineering: [108000.0, 8000.0, 8000.0, 4000.0, 4000.0, 3000.0, 0.32, 0.32, 0.30]
    angle: 45.0
```

The units are the mesh's: this cell is in millimetres with moduli in MPa, so $108\times10^{3}$
MPa is a 108 GPa fibre direction. (The `material_param=` / `angles=` arguments of
`plate_homo_2d` remain available as an override that takes precedence over the file's blocks —
examples 4 and 7 use that route.)

The solver reads `<name>.yaml`. A SwiftComp `.sc` is converted once with the packaged helper.
The module defines no `__main__` block and reads no command-line arguments, so import its
`convert` function rather than invoking it with `python -m`:

```python
from opensg_solid.io.sc_to_yaml import convert

convert("RHC_SW_2UC_45.sc")   # writes RHC_SW_2UC_45.yaml + .msh
```

### Run it

One command, one argument — the file says what to do:

```bash
opensg RHC_SW_2UC_45.yaml
```

The unified `opensg` command reads `msg: solid` from the header and forwards the file
unchanged; `opensg_solid RHC_SW_2UC_45.yaml` and `python -m opensg_solid RHC_SW_2UC_45.yaml`
are the same entry point. The example's `plate_homo_2dsg.py` is the identical call through the
Python API — one line, `plate_homo_2d(name + ".yaml")` — for driving it from your own script.

### What comes out

| file | content |
|---|---|
| `RHC_SW_2UC_45.out` | the timed stiffness and compliance in the `.K` layout, titled `Classical Plate` at `refined: 0` and `Reissner-Mindlin` at `refined: 1` |
| `RHC_SW_2UC_45_mesh.png` | the cell mesh, elements coloured by material |

Both routes print `r["law"]` under its `r["law_title"]` header — the same matrix the `.out`
reports, selected by the header, so the driver contains no branching. The `.out` banner
records the SG measure and the boundary treatment; the shipped file asks for `refined: 1`, so
what comes back is the $8\times8$:

```
 OpenSG msg-solid plate model, omega 35.248001, periodic

 The Effective Reissner-Mindlin Stiffness Matrix
 --------------------------------------------
     3.3585973E+005     7.7500043E+004    -1.4366892E+002     4.1488406E-001     1.4861903E+000     3.6602857E-002     0.0000000E+000     0.0000000E+000
```

closing with ` Time taken: 1.89 sec` for this 4 251-node, 6 640-triangle cell. The leading
$6\times6$ block is the classical ABD `refined: 0` would have reported on its own, and the two
extra rows carry the transverse shears $2.6974186\times10^{4}$ and $9.7898064\times10^{1}$.

### What the numbers mean

$\omega = 35.248001$ is the cell's in-plane period, and it is directly checkable against the mesh:
the nodes run from $y_1 = -17.624$ to $+17.624$ mm. The diagonal, in the mesh's own units:

| term | value | units |
|---|---|---|
| $A_{11}$ | $3.3585973\times10^{5}$ | N/mm |
| $A_{22}$ | $1.1291139\times10^{5}$ | N/mm |
| $A_{66}$ | $1.0551654\times10^{5}$ | N/mm |
| $D_{11}$ | $9.0367521\times10^{7}$ | N·mm |
| $D_{22}$ | $5.7011050\times10^{7}$ | N·mm |
| $D_{66}$ | $4.6026320\times10^{7}$ | N·mm |

The membrane law is strongly anisotropic, $A_{11}/A_{22} = 2.97$, while the bending law is much
closer to isotropic, $D_{11}/D_{22} = 1.59$ — the two directions see the cell very differently in
stretching and much less differently in bending. The largest entry of the whole $B$ block is
$4.95$, against $A$ terms of order $10^{5}$ and $D$ terms of order $10^{7}$, which is what a cell
symmetric about its mid-plane should give and is the first health check to make on a run of your
own.

![Two-unit-cell honeycomb sandwich structure gene, elements coloured by material](../_static/rhc_2dsg_mesh.png)

The mesh figure shows what the gene is: $\pm45^\circ$ face laminae (materials 1 and 2) as the thin
horizontal bands at $y_2 = \pm23$ mm, and the isotropic core web (material 3) folding between them
over the in-plane period. The through-thickness coordinate is $y_2$, spanning $\pm23.35$ mm.

### The Reissner-Mindlin option

Setting `refined: 1` in the yaml header (or passing `refined=1` to `plate_homo_2d`;
`shear_refined=True` is the legacy spelling, plate only) additionally runs the RM first-order
warping ladder
and returns `r["G_msg"]` (the $2\times2$ transverse-shear block), `r["ABDG"]` (the same $8\times8$
block form as section A) and `r["A6_ladder"]`. When the ladder produces an
SPD fit, the `.out` is written from the $8\times8$ and its matrices are titled
`Reissner-Mindlin` instead of `Classical Plate`; otherwise the run falls back to the
$6\times6$. The same switch serves the beam route: `n_model: 1` with `refined: 0` reports the
classical Euler–Bernoulli $4\times4$ (the `.out` titled `Euler-Bernoulli`), with `refined: 1`
the Timoshenko $6\times6$.

```python
from opensg_solid.sg_homo import plate_homo_2d

r = plate_homo_2d("RHC_SW_2UC_45.yaml", n_model=2, refined=1)
print(r["ABDG"])
```

The same folder also ships `Plate_1D_SG_2UC_45.yaml`, a **1-D** structure gene (two-node
line elements through a stack — the SG dimension follows from the element node count, nothing
declares it), which runs through the identical call by changing `name`; its
result is `Plate_1D_SG_2UC_45.out`, because `plate_homo_2d` always names its report after the
input basename. The `*_plate_ABD.out` files shipped beside it are `np.savetxt` leftovers from an
older script style — a bare $6\times6$ under a single comment line, not the timed `.K` layout.
`plate_homo_2d` reads 1-D, 2-D and 3-D genes from the
same code path, so a stack of plies can be homogenized either here or through the dedicated
through-thickness route of section A.

## D. 2-D SG plate dehomogenization

### What it computes

Given a converged plate solution, you know the macro plate strain at the point of interest. Feeding
it back into the homogenized cell recovers the local 3-D strain and stress at **every Gauss point
of the gene**, which is where a honeycomb actually fails — in the web, not in the smeared plate.
`dehom_fields` returns the strain, the stress and the fluctuation displacement in one pass, using
the warping columns $V_0$ that the homogenization already solved; nothing is re-solved.

### The input you edit

`plate_dehom_2dsg.py` takes the same `name` as section C plus its own `material_param` and
`angles` (this example still carries the materials in its `User Input` block, through the
override route), and one new line — the macro plate strain, in the order
`[e11, e22, 2e12, k11, k22, 2k12]`:

```python
epsilon_bar = jnp.array([0.0, 0.1, 0.0, 0.1, 0.0, 0.0])
```

The shipped state combines a transverse extension $\varepsilon_{22} = 0.1$ with a bending curvature
$\kappa_{11} = 0.1$. This vector is yours: it comes from your plate solution at the station you care
about, and there is no default that would be meaningful.

### Run it

```bash
python plate_dehom_2dsg.py
```

### What comes out

| file | content |
|---|---|
| `RHC_SW_2UC_45_plate_ABD.out` | the plain-text $6\times6$ ABD, header `plate 6x6 [N11 N22 N12 M11 M22 M12] from the 2D SG RHC_SW_2UC_45` |
| `RHC_SW_2UC_45_dehom.txt` | one row per Gauss point: `GP_Index`, `Coord_X`, `Coord_Y`, then `Eps_xx Eps_yy Eps_zz Eps_yz Eps_xz Eps_xy` and `Sig_xx` through `Sig_xy` — SwiftComp component order |
| `RHC_SW_2UC_45_dehom.vtk` | ASCII `UNSTRUCTURED_GRID` point cloud of the same 19 920 points, each strain and stress component as its own `SCALARS` array, ready for ParaView |
| `RHC_SW_2UC_45_dehom.SM` / `.EM` / `.U` | the same fields in the section-B layout: `x y z` then `S11 S22 S33 S12 S13 S23` (stress), `E11` through `E23` (strain), `U1 U2 U3` (displacement) |
| `RHC_SW_2UC_45_mesh.png` | the mesh figure, from the `plate_homo_2d` call the script makes first |

Because `plate_homo_2d` runs inside the script, it also writes the timed `.K`-layout
`RHC_SW_2UC_45.out` of section C. That file is regenerated on every run rather than read from
the folder, so its `Time taken` footer is this run's, not the shipped copy's. The script prints
the ABD diagonal and the peak of each stress component.

Two conventions are worth knowing before reading the files. In the `.SM`/`.EM`/`.U` tables the SG
coordinates are placed in the **last** slots of the `x y z` triple, so a 2-D gene leaves the `x`
column at zero and puts $y_1$ in `y` and the through-thickness $y_2$ in `z`. And `.U` is the
**fluctuation-only** displacement — its own header says so — because a unit-state SG recovery has
no macro displacement to add.

### Reading the result

The first Gauss point of the shipped run sits at $(y_1, y_2) = (17.54067,\ 23.29375)$ mm — right at
the top face of the cell, whose half-thickness is $23.35$ mm — and reports

$$\varepsilon_{11} = 2.329375, \qquad
\sigma_{11} = 8.755003\times10^{4},\quad \sigma_{22} = 7.102971\times10^{4},\quad
\sigma_{12} = -6.542276\times10^{4}$$

in MPa. That $\varepsilon_{11}$ is exactly $\kappa_{11} y_2 = 0.1 \times 23.29375$ — the macro part
of the strain, reproduced to every printed digit, which is the cheapest possible check that the
strain state you passed is the strain state that arrived. The other five components are what the
recovery adds: $\varepsilon_{22} = 0.2643$ and $\varepsilon_{33} = -0.9187$ are the fluctuation
response of the cell, and $\sigma_{12}$ is large even though the applied macro shear strain
$2\varepsilon_{12}$ is zero, which is consistent with the off-axis face laminae the mesh figure
shows at that height. Local states of this kind are exactly what a smeared plate model cannot show
you, and the reason to run a dehomogenization at all.

## Files at a glance

| example | you edit | you run | it writes |
|---|---|---|---|
| 1 | `layup_db.yaml` | `python 1d_sg.py` | `1dsg.yaml`, `1dsg.png`, `layup_db_plate_homo.out` |
| 2 | the `USER INPUT` block | `python rm_dehom.py` | `1dsg_dehom.SM/.EM/.U/.vtk` |
| 2 | a `.ff` table + `dehom:` in `layup_db.yaml` | `python rm_dehom_ff.py ex5_shell_S8R_field.ff` | `<ff stem>_ff.SM/.EM/.U`, `_ff_gauss.vtk` |
| 3 | the yaml header (`n_model`, `refined`, `analysis`) — materials + angles live in the yaml too | `opensg <name>.yaml` | `<name>.out`, `<name>_mesh.png` |
| 4 | `name`, `material_param`, `angles`, `epsilon_bar` | `python plate_dehom_2dsg.py` | `<name>.out`, `<name>_plate_ABD.out`, `<name>_dehom.txt/.vtk/.SM/.EM/.U`, `<name>_mesh.png` |
