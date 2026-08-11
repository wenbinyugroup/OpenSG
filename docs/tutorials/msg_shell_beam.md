# Beam Properties and Recovery from a Shell Cross-Section (msg-shell)

This page walks through the two `opensg_shell` examples that turn a thin-walled composite
cross-section into a Timoshenko beam and then back into 3-D ply stress:

| example folder | driver | produces |
|---|---|---|
| `examples/OpenSG_shell/1_get_beam_props_from_shell_cross_section` | `beam_homo_shell.py` | Timoshenko $6\times6$, per-wall RM plate laws, mesh and orientation PNGs |
| `examples/OpenSG_shell/2_get_beam_dehom_from_shell_cross_section` | `beam_dehom_shell.py` | 3-D stress at every wall depth for a given beam load |

Both run on the same input, `iea_s10_shell.yaml` — station $r/R = 0.20$ of the IEA-22 MW
reference blade.

## What a 1-D shell SG is

A **1-D shell structure gene** is the *contour* of the cross-section: a chain of straight line
elements along the wall mid-surface, each element carrying a laminate stack (the layup) as
material data rather than as geometry. Thickness never appears in the mesh; it lives inside the
wall's plate law and inside the through-thickness recovery. A 2-D solid cross-section mesh, by
contrast, discretizes the wall *through* its thickness with area elements, so every ply interface
must be resolved by the mesh.

The consequence for the user is size. The station below spans about 7.2 m in $y_2$ and 2.5 m in
$y_3$, has spar caps 87.6 mm thick and three shear webs, and is fully described by 307 nodes and
310 line elements. The price is that the walls are modelled as Reissner–Mindlin plates, so
wall-thickness effects enter through the plate law and the recovery ladder rather than through the
mesh.

### The YAML blocks

`build_rm_bundle` reads one file. These are its blocks, with the values in `iea_s10_shell.yaml`:

| block | content | this station |
|---|---|---|
| `nodes` | `[y2 y3 0]` contour coordinates, metres | 307 nodes |
| `elements` | 1-based two-node line connectivity | 310 elements |
| `sets: element` | named element sets, one per wall region | 6 sets |
| `sections` | per set: `elementSet` plus `layup` as `[material, thickness, angle]` triples, outer ply first | 6 sections |
| `elementOrientations` | per element, the flattened $3\times3$ frame $[e_1\;e_2\;e_3]$ | 310 rows |
| `materials` | `name`, `density`, `elastic: {E, G, nu}` as 3-vectors | 6 materials |
| `reference` | the reference surface the whole run is referred to | `center` |

The `reference` field is the single source of truth. `build_rm_bundle(yaml)` reads it and maps it
to a thickness fraction (`center` $\to$ 0.5, `oml` $\to$ 0.0, `iml` and `oml_flip` $\to$ 1.0)
that is then used consistently for the ring laminate reference, the plate-SG $z$ origin, the
emitted ABD cache and the recovery depth conversion. Pass `ref=` explicitly only to override it.
Because the reference here is the mid-surface, quantities that depend on the reference axis (in
particular $GJ$ and the bending–extension couplings) are mid-surface values.

![Cross-section contour coloured by layup](../_static/shell_xsec_mesh.png)

The mesh figure above is the one the driver writes. Each colour is one `sets: element` entry:

| layup | plies, outer first | total thickness (m) | elements |
|---|---|---|---|
| `layup_0` | gelcoat, glass_triax, glass_uniax, glass_triax | 0.03260055 | 25 |
| `layup_1` | gelcoat, glass_triax, medium_density_foam, glass_triax | 0.07711403 | 161 |
| `layup_2` | gelcoat, glass_triax, carbon_uniax (0.08106 m), glass_triax | 0.08756579 | 13 |
| `layup_3` | gelcoat, glass_triax, glass_uniax, glass_triax | 0.02808872 | 8 |
| `layup_4` | gelcoat, glass_triax, carbon_uniax (0.07707 m), glass_triax | 0.0835804 | 14 |
| `layup_5` | glass_biax, medium_density_foam, glass_biax | 0.046 | 89 |

`layup_2` and `layup_4` are the carbon spar caps on the upper and lower surfaces, `layup_1` is the
foam-cored skin that covers most of the contour, `layup_5` is the three shear webs, `layup_0` is
the trailing-edge reinforcement and `layup_3` the leading edge. Every ply angle in this file is
$0^\circ$; the biax and triax fabrics are supplied as smeared orthotropic materials rather than as
individual plies, so the "material frame" of the recovery coincides with the wall frame here.

### Orientation frames

The example-1 driver emits an orientation PNG by calling `auto_emit` explicitly —
`build_rm_bundle` never calls it, so a homogenization run on its own produces no such figure.
Make the same call in your own driver, because a wrong `elementOrientations` block is the most
common way to get a plausible-looking but wrong answer.
$e_1$ is the beam axis (out of the page), $e_2$ (blue) is the in-plane ply-flow direction along
the contour, and $e_3$ (green) is the wall normal, which sets the OML $\to$ IML sense used by the
recovery.

![Per-element e2 and e3 material axes on the contour](../_static/shell_xsec_orient.png)

Check that the green arrows point consistently into the section on the skin and consistently to
one side on each web before trusting any number below.

## The element and the wall law

The ring is assembled as a one-quad-deep prismatic strip: the contour is extruded by the mean
element length along the beam axis and the extruded node row is DOF-mapped back onto the original
row, which enforces the prismatic (span-invariant) condition exactly. All matrices are then
divided by that depth, so the result is per unit length.

Each node carries **six** degrees of freedom, $[w_1\;w_2\;w_3\;\omega_1\;\omega_2\;\omega_3]$.
The drilling rotation $\omega_3$ is an *independent* field rather than an algebraic function of
the displacements. Making it independent is what allows adjacent walls meeting at a junction to
share a consistent rotation, but it also introduces a spurious mode, which is removed by a weak
drilling constraint carried by one **element-constant Lagrange multiplier** per element
(`lam_space="elem"`). Element-constant is the inf-sup-stable choice; equal-order nodal
multipliers over-constrain as the mesh is refined.

Transverse shear uses assumed-strain (MITC) tying, and the production ring scheme
(`shear="mitc4_g23"`) ties **only** the $\gamma_{23}$ row. The reason is specific to a prismatic
strip: under span invariance $\gamma_{13}$ carries no fluctuation gradient and is algebraic in
the directors, so tying it would alias the drilling mode instead of curing shear locking. Only
$\gamma_{23}$ pairs a differentiated displacement with a rotation.

The wall constitutive law is therefore the $8\times8$ Reissner–Mindlin plate law

$$\mathbf{ABDG} = \begin{bmatrix}A & B & 0\\ B & D & 0\\ 0 & 0 & G\end{bmatrix},
\qquad
\bar{\varepsilon} = [\varepsilon_{11}\;\varepsilon_{22}\;2\varepsilon_{12}\;\;
K_{11}\;K_{22}\;K_{12}+K_{21}\;\; 2\gamma_{13}\;2\gamma_{23}]^{T} ,$$

with the row grouping written out in the header of `<name>_shell_ABDG.out`.

The $2\times2$ transverse-shear block $G$ is built by **one** route, with no switch to set: the
MSG/VAM second-order-energy projection (the Yu-2002 least-squares construction, Eq. 61 of Yu,
Hodges & Volovoi, *Computers & Structures* **81**:439–454, 2003), computed by
`opensg_solid.rm_plate_1D.msg_rm_plate.rm_plate_msg`. (The former `g_source="whitney"`
alternative — the coupling-aware complementary-energy shear flow — has been retired; the keyword
is still accepted by the old signatures and is ignored. The complementary-energy $G$ survives
only as the numerical fallback on a laminate whose MSG $U^*$-fit is not SPD.) The MSG
construction matters most on the foam-cored
walls: for `layup_1` the MSG $G$ block is $4.146\times10^{6}$ and $4.145\times10^{6}$ N/m, some
seventy times smaller than the carbon cap's $2.867\times10^{8}$ and $2.127\times10^{8}$ N/m, so
the soft core dominates the section's shear compliance and must not be assigned by a rule of
thumb.

## Example 1 — the Timoshenko $6\times6$

### Input the user edits

The driver has a single user-input line:

```python
name = "iea_s10"               # reads <name>_shell.yaml
```

### Run it

```bash
cd examples/OpenSG_shell/1_get_beam_props_from_shell_cross_section
```

```bash
opensg iea_s10_shell.yaml
```

That is the whole homogenization: the header carries `msg: shell`, `n_model: 1` and
`refined: 1`, so the unified command resolves the msg-shell engine, runs the Reissner–Mindlin
ring and writes `iea_s10_shell_Timo.out` and `iea_s10_shell_ABDG.out`. The example's driver
does the same through the API and adds the two figures the CLI does not emit:

```bash
python beam_homo_shell.py
```

### The one call

Everything is behind one function:

```python
import numpy as np
from opensg_shell import build_rm_bundle, auto_emit

B = build_rm_bundle("iea_s10_shell.yaml")   # RM 6-DOF ring + MSG wall G
C6 = np.asarray(B["Timo"])                  # (6, 6) Timoshenko stiffness
print(B["ref"], B["g_source"])              # "center msg"
print(np.diag(C6))
auto_emit("iea_s10_shell.yaml", out_png="iea_s10_orient.png")
```

`build_rm_bundle` returns a **bundle**, not just a matrix, because the dehomogenization in example
2 must reuse the identical discretization. The bundle carries:

| key | content |
|---|---|
| `Timo` | the $(6,6)$ Timoshenko stiffness |
| `V0`, `V1` | the zeroth- and first-order warping fields, $(6m,4)$ each, including the drilling $\omega_3$ boundary values |
| `corners`, `red_cells`, `rx3`, `re3`, `k22` | the contour geometry, connectivity, wall normals and curvature per element |
| `strip` | the prismatic strip actually assembled, as `(nodes, quads, h)` |
| `ax`, `cross` | the beam-axis index and the two cross-section axis indices |
| `layup_per_elem`, `layup_db`, `material_db` | the layup name of each element and the geometry-free plate-SG databases |
| `frac`, `ref`, `g_source` | the reference fraction, its name, and the wall-$G$ source |

### What comes out

| file | written by | content |
|---|---|---|
| `iea_s10_shell_Timo.out` | `build_rm_bundle` (via `write_sc_K`) | the effective Timoshenko stiffness and compliance in the SwiftComp `.K` layout, with an `OpenSG` banner and a `Time taken` footer |
| `iea_s10_shell_ABDG.out` | `build_rm_bundle` (via `write_abdg_out`) | one $8\times8$ RM plate stiffness and compliance block per section, each headed by its layup |
| `abd/iea_s10_shell_abd.yaml` | `emit_station_abd` | a per-layup cache of the $8\times8$ wall law plus ply lists, thickness and mass per area, for reuse by shell buckling and other station-level tools |
| `iea_s10_mesh.png` | the driver | the contour coloured by layup |
| `iea_s10_orient.png` | `auto_emit` | the $e_2$/$e_3$ arrows |

Two notes on the cache. It is written **only if it does not already exist**, so delete
`abd/iea_s10_shell_abd.yaml` after changing the layups (or after upgrading OpenSG — a cache left
over from an older version is never refreshed in place). And it records `g_source: msg`:
`emit_station_abd` builds the same MSG wall $G$ the ring uses, at the station's own reference
surface, so the cached $8\times8$ agrees with the per-section blocks of `_ABDG.out`.

### The numbers

This is the stiffness block of `iea_s10_shell_Timo.out` as written, in SI units, on the strain
set $[\varepsilon_{11}\;\gamma_{12}\;\gamma_{13}\;\kappa_1\;\kappa_2\;\kappa_3]$ (extension, the
two transverse shears, twist, and the two bending curvatures — the VABS ordering):

```text
 The Effective Timoshenko Beam Stiffness Matrix
 --------------------------------------------
     2.7660599E+010    -1.7230658E-005     4.6650498E-006     5.4020232E-006     1.6686413E+009    -2.0773040E+009
    -1.7230658E-005     7.1861326E+008     5.7520370E+007    -1.0192044E+008     2.3079953E-006     5.2738749E-006
     4.6650498E-006     5.7520370E+007     4.2174428E+008    -1.6217837E+008     8.4816767E-006     2.7931393E-007
     5.4020232E-006    -1.0192044E+008    -1.6217837E+008     2.4375653E+009     1.6449557E-006    -8.1140553E-006
     1.6686413E+009     2.3079953E-006     8.4816767E-006     1.6449557E-006     3.5143887E+010    -5.5571249E+009
    -2.0773040E+009     5.2738749E-006     2.7931393E-007    -8.1140553E-006    -5.5571249E+009     6.8129724E+010
```

The diagonal is what the driver prints:

| term | value | units | meaning |
|---|---|---|---|
| $EA$ | $2.76606\times10^{10}$ | N | axial stiffness |
| $GA_2$ | $7.18613\times10^{8}$ | N | transverse shear, chordwise |
| $GA_3$ | $4.21744\times10^{8}$ | N | transverse shear, flapwise |
| $GJ$ | $2.43757\times10^{9}$ | N·m² | torsional stiffness about the mid-surface reference |
| $EI_2$ | $3.51439\times10^{10}$ | N·m² | flapwise bending |
| $EI_3$ | $6.81297\times10^{10}$ | N·m² | edgewise bending |

The off-diagonals are the physics a beam table without couplings would throw away. The
extension–bending entries $1.6686\times10^{9}$ and $-2.0773\times10^{9}$ N·m locate the tension
centre away from the mid-surface reference point; $-5.5571\times10^{9}$ N·m² is the flap–edge
bending coupling, which says the section's principal bending axes are not the $y_2$/$y_3$ axes;
and $-1.0192\times10^{8}$ and $-1.6218\times10^{8}$ N·m are the shear–twist couplings that place
the shear centre. Entries printed at $10^{-5}$ against diagonals of $10^{10}$ are exact zeros to
about fifteen digits — extension does not couple to transverse shear or to twist on this section,
which is a useful self-check that the drilling constraint and the rigid-body kernel are behaving.

The footer of the same file reads `Time taken: 2.58 sec` for this run; the same station rebuilt
inside example 2 took 2.93 s. Both cover the whole bundle build — reading the YAML, the six
per-layup MSG plate solves, the ring KKT factorization and the layup-database build.

## Example 2 — dehomogenization to 3-D ply stress

Homogenization gave the beam its stiffness. Dehomogenization does the inverse: given the beam's
internal forces at this station, it reconstructs the 3-D stress at every point of every wall,
which is what a failure criterion actually needs.

### Input the user edits

```python
name = "iea_s10"                  # reads <name>_shell.yaml
ff_file = "ff51_rmc_reform.dat"   # beam FF per station (# eta F1 F2 F3 M1 M2 M3)
station_row = 10                  # row 10 -> eta = r/R = 0.20 (iea_s10)
n_depth = 9                       # through-thickness recovery points per element
stress_frame = "material"         # "material" (ply axes) | "plate" (wall axes)
```

`ff51_rmc_reform.dat` is a plain table whose header line reads
`# eta F1 F2 F3 M1 M2 M3 (RM CENTER-ref BeamDyn, VABS order)`: one row per spanwise station,
the first column $\eta = r/R$ and the remaining six the beam force and moment resultants in the
same VABS order and the same SI units (N, N·m) as the $6\times6$. Row 10 is

$$FF = [\,4.1927\times10^{4}\;\;8.2674\times10^{3}\;\;1.4673\times10^{6}\;\;1.2191\times10^{6}\;\;{-6.0185\times10^{7}}\;\;3.0426\times10^{5}\,]$$

— a 60 MN·m flapwise moment with a 1.47 MN flapwise shear, which is what dominates everything
that follows. The forces must be referred to the same axis as the stiffness: this table is
center-referenced, matching `reference: center` in the YAML.

### Run it

```bash
cd examples/OpenSG_shell/2_get_beam_dehom_from_shell_cross_section
```

```bash
opensg iea_s10_shell.yaml D
```

The `D` argument (or `analysis: D` in the header) runs the homogenization and then the recovery
in one pass, taking the macro state from `iea_s10_shell.ff` next to the YAML — or from
`epsilon_bar:` in the header when there is no `.ff` — and writing
`iea_s10_shell_dehom.txt`, `.vtk` and `.junc`. It closes with
`Local field files are computed and stored.` and one `Time taken`.

The example's driver produces the same recovery through the API and adds the figures and the
VABS comparison:

```bash
python beam_dehom_shell.py
```

### The two-step chain

**Step 1, along the contour.** The macro beam strain is $\mathrm{st} = C_6^{-1}FF$. That strain
is fed through `_macro_fields`, which recombines the bundle's warping fields into the nodal
warping $w = V_0\,\mathrm{st}_m + V_1\,\mathrm{st}_{c1}$ and its axial derivative
$w' = V_0\,\mathrm{st}_{c1} + V_1\,\mathrm{st}_{c2}$, and then through `_rm_shell_strain`, which
applies the *same* element operators used to assemble the stiffness. The result is, per element,
the six shell strains $[\varepsilon_{11}\;\varepsilon_{22}\;2\varepsilon_{12}\;K_{11}\;K_{22}\;2K_{12}]$
at mid-arc. Because it reuses `quad_ops_indep`, step 1 is the exact energy-consistent adjoint of
the homogenization rather than a separate post-processing model.

The recovery also needs the *gradients* of those strains. The span gradient $\partial/\partial y_1$
comes from the operators directly; the arc gradient $\partial/\partial y_2$ is a finite difference
of nodally averaged values, and the averaging is **layup-boundary aware** — two element values are
averaged at a node only when both elements carry the same layup, so no difference is ever taken
across a section change where the strains legitimately jump.

**Step 2, through the thickness.** For each of the 6 layups, `rm_plate_msg` builds a 1-D MSG-RM
plate structure gene (the warping ladder plus the MSG $G$), and `msgrm_strain_at_depth` evaluates
the first-order recovery of Eq. (63) at a depth $z$ measured from the reference surface, returning
the 3-D Voigt strain, the 3-D Voigt stress $[\sigma_{11}\;\sigma_{22}\;\sigma_{33}\;\sigma_{23}\;\sigma_{13}\;\sigma_{12}]$
and the ply angle at that depth. Both functions are imported from
`opensg_solid.rm_plate_1D.msg_rm_plate`, shared verbatim with the plate and solid examples.

The driver evaluates `n_depth = 9` ply-interior stations per element, at
$\zeta = (k + \tfrac12)/9$ with $\zeta = 0$ at the OML and $\zeta = 1$ at the IML, giving
$310 \times 9 = 2790$ recovery points.

```python
import numpy as np
from opensg_shell import build_rm_bundle, _macro_fields, _rm_shell_strain
from opensg_solid.rm_plate_1D.msg_rm_plate import rm_plate_msg, msgrm_strain_at_depth

B = build_rm_bundle("iea_s10_shell.yaml")
FF = np.loadtxt("ff51_rmc_reform.dat")[10, 1:]
st, st_m, aA, aB = _macro_fields(B, beam_force_vabs=FF)
s6, s2 = _rm_shell_strain(B, 75, 0.5, st_m, aA, aB)   # element 75, mid-arc

ldb = B["layup_db"]; mdb = B["material_db"]; frac = float(B["frac"])
warp = rm_plate_msg(ldb["layup_2"]["thick"], ldb["layup_2"]["angles"],
                    ldb["layup_2"]["mat_names"], mdb, fraction=frac)
h = float(sum(ldb["layup_2"]["thick"]))
Gam, Sig, ply = msgrm_strain_at_depth(warp, (0.056 - frac) * h, np.asarray(s6, float),
                                      frame="material")
print(Sig / 1e6)      # MPa, Voigt [S11 S22 S33 S23 S13 S12]
```

### What comes out

From `opensg iea_s10_shell.yaml D`:

| file | content |
|---|---|
| `iea_s10_shell_dehom.txt` | 2790 rows, one per recovery point: `elem`, `zeta`, `y2`, `y3`, `z` from the reference surface, then the six stresses (Pa), the six strains and the three displacements — plus, at the default `junction: flag`, the two extra columns `jflag` and `jdist` |
| `iea_s10_shell_dehom.vtk` | the exploded recovery mesh: one quad per (element, depth band), each carrying its own station's stress as `CELL_DATA` — never node-averaged — with `jflag`/`jdist` as extra scalars |
| `iea_s10_shell_dehom.junc` | the junction sidecar: every junction's coordinates, topology, overlap area and per-wall ownership, headed by the census |

From `python beam_dehom_shell.py`:

| file | content |
|---|---|
| `iea_s10_dehom_shell.txt` | 2790 rows, one per recovery point: `elem`, `y2`, `y3`, `z` from the reference surface, `zeta`, `ply` angle in degrees, then the six stresses in MPa; the header repeats the frame, $\eta$ and the full `FF` |
| `iea_s10_dehom_stress.png` | the section stress cloud, $\sigma_{11}$ and $\sigma_{12}$ |
| `iea_s10_cap_tt.png` | $\sigma_{11}$ and $\sigma_{12}$ against $\zeta$ along a single through-thickness line at the arc-centre element of the peak-stress cap |

The driver reports the peak as

```text
peak |sigma11| = 318.18 MPa at (y2, y3) = (-0.677, 1.311), zeta = 0.06  [material frame]
```

### Junctions in the recovery

A midline shell SG has no material at a wall crossing, and the homogenization adds that missing
energy back through a census correction. The **recovery has the opposite problem**. It hangs a
full through-thickness column off every element mid-arc, so within about a wall thickness of a
junction the columns of two crossing walls cover the *same material twice*, with two different
wall laws — which is how a spar-cap ply stress ends up reported at a web station.

`opensg_shell.sg_dehom_junction` does not repair that. It **measures** the hazard and lets the
YAML header say what to do about it, through three keys the `D` route reads:

| `junction:` | behaviour |
|---|---|
| `off` | the pre-junction behaviour exactly — no extra columns, no sidecar, no census; the files are bit-identical to before the feature existed |
| `flag` | **the default**, and non-mutating: two extra columns `jflag` and `jdist` appended at the *end* of `<base>_dehom.txt`, the same two added as `.vtk` `CELL_DATA` scalars, the `<base>_dehom.junc` sidecar written, and the census printed. **No field value changes**, so every existing column index is untouched. |
| `exclude` | as `flag`, plus `NaN` in the 15 field columns of every duplicate (`jflag` 3) row. The row itself stays — row $i$ is still element $i/\!/n_{\rm depth}$ at depth $i\%n_{\rm depth}$ — so the cloud *partitions* the material instead of double-covering it. The `.vtk` values are never NaN-ed (legacy ASCII readers handle NaN badly); threshold the mesh on `jflag` instead. |

A fourth tier, `patch`, is **designed but deliberately not implemented** and raises if you ask
for it: the corner micro-solve it would call is admissible only for two straight walls of
uniform layup crossing at one node with clear stubs on every leg, which most real blade
junctions are not.

| `jflag` | meaning |
|---|---|
| 0 | clean — outside every junction's boundary layer |
| 1 | near — inside some $r_{\rm bl}$, but inside no other wall's material |
| 2 | overlapped, and **this** wall owns the block |
| 3 | duplicate — inside another wall's material, and another wall owns it |
| 4 | reserved for `patch`; never emitted |

`jdist` is $|p - J| / \max_i t_i$ at the nearest junction node $J$ (infinite when the section
has no junction at all), so `jflag >= 1` is exactly `jdist <= k_bl`. Two more optional keys tune
the measurement: `junction_bl:` (default `1.0`) is $k_{\rm bl}$, the boundary-layer radius in
units of the junction's thickest wall — a modelling choice, not a derived length, so widen it
to be conservative. `junction_ang:` (default `1.0`, degrees) is the tangent-grouping tolerance
of the detector; 1.0 is the homogenization census value and is deliberately tight, so on a
curved airfoil contour it also calls every kink of more than one degree a junction. Raise it to
5–15° for a census that lists the wall crossings only — it moves the advisory `near` count, not
the overlap flags.

Ownership at a crossing mirrors the homogenization's own patch mesh exactly, so the two sides
cannot drift: one through-family at a two-family node owns the block (a T); two through
families mitre by $|s_i|/t_j$ against $|s_j|/t_i$ (an X); two ending families mitre on the
normalized-band diagonal (an L); more than two families fall back to the nearest midline and
the junction is reported *unresolved*.

On this station the census reads

```text
 junctions : 97 detected (5 T, 91 L, 1 unresolved)
 stations  : 2790 total, 1032 near, 56 overlapped (50 duplicates)
```

— 97 "junctions" at the default 1° tolerance, of which only a handful are genuine wall
crossings and the rest are discretized contour curvature, which is exactly why the near count is
large while the overlap counts stay small. The sidecar
`iea_s10_shell_dehom.junc` lists every junction with its coordinates, topology, overlap area
$A_j = \sum t_i t_j / |\sin\theta_{ij}|$ and per-wall ownership decision.

```{warning}
`junction:` here is a **recovery bookkeeping tier**. It is not the homogenization junction
correction (`build_solid_bundle(junction="census"|"micro"|"microcell")`), which is an energy
correction with its own validity domain — see `Rules/junction_corrections.md`.
```

### Reading the result

![Recovered sigma11 and sigma12 over the cross-section](../_static/shell_dehom_stress.png)

Both panels are the full recovery-point cloud; points sit slightly off the contour because each
one is displaced from the mid-surface by its own depth $z$ along the wall normal, which is why the
cap points reach $y_3 = 1.311$ m while the contour stops at $1.273$ m.

The axial stress is confined almost entirely to the two spar caps, exactly as a flap-dominated
load state should be:

| quantity | value | where |
|---|---|---|
| minimum $\sigma_{11}$ | $-318.177$ MPa | element 75, `layup_2`, $(y_2,y_3) = (-0.677, 1.311)$, $\zeta = 0.056$ |
| maximum $\sigma_{11}$ | $+314.461$ MPa | element 150, `layup_4`, $(y_2,y_3) = (-0.145, -1.249)$, $\zeta = 0.056$ |
| maximum $\lvert\sigma_{12}\rvert$ | $5.573$ MPa | element 11, `layup_0`, $(y_2,y_3) = (3.895, 0.335)$ |

The upper cap is in compression and the lower cap in tension under this flap-dominated state, at
nearly equal magnitude, and both extremes fall at the outermost recovery station $\zeta = 0.056$,
which for these layups lies just inside the carbon uniaxial ply — that is where a maximum-stress
check on the carbon belongs. The in-plane shear tells a different story: it peaks not in the caps
but in the trailing-edge and leading-edge regions, where the closed cell carries the torque as
shear flow.

### What this route recovers, and what it does not

The repository's `IEA_blade_beam` benchmark folder compares this same station's recovery against a
VABS 2-D solid run of the identical section (`iea_s10.sg.SM`), and records the result in
`iea_s10_report.txt` as rms differences alongside the rms magnitude of the VABS field itself:

| component | region | rms difference (MPa) | rms of the VABS field (MPa) |
|---|---|---|---|
| $\sigma_{11}$ | skin | 9.529 | 96.676 |
| $\sigma_{11}$ | web | 43.073 | 44.440 |
| $\sigma_{12}$ | web | 6.799 | 22.026 |
| $\sigma_{11}$ | spar-cap through-thickness path | 38.355 | 285.457 |
| $\sigma_{11}$ | cap-left junction path | 1.265 | 286.837 |

The in-plane picture is sound where the load is: about 10 % rms on the skin, 13 % along the cap
thickness, and 0.4 % along the reference junction path. The web $\sigma_{11}$ difference is the
same size as the field itself, which is honest to state plainly — the webs carry almost no axial
stress at this state, so the comparison there is a ratio of two small numbers.

Two components are **not** delivered by this route and should not be used from the output file:

- $\sigma_{33}$ is machine-zero in the first-order recovery (rms $3.6\times10^{-13}$ MPa against a
  VABS rms of $5.6\times10^{-2}$ MPa), which is visible directly in the `S33` column of
  `iea_s10_dehom_shell.txt`: over its 2790 recovery points that column is of order $10^{-13}$ MPa
  and smaller, with a median of $1.96\times10^{-13}$ and a maximum of $9.15\times10^{-13}$.
- $\sigma_{13}$ is recovered at rms $6.3\times10^{-3}$ MPa against a VABS rms of
  $2.4\times10^{-1}$ MPa. This is structural, not a bug: on a prismatic ring the $\gamma_{13}$ row
  is untied by construction (see the element section above), so it carries the flat-wall drilling
  mode and cannot serve as a recovery target. The companion diagnostic
  `iea_s10_q_consistency.txt` quantifies this — the tied $\gamma_{23}$ channel closes its
  equilibrium identity $\int \sigma_{23}\,dz = (G_{\text{msg}}\,s_2)_2$ to 1.05 % median error,
  while the untied $\gamma_{13}$ channel does not, giving a $\sigma_{13}$ target that is invalid by
  a factor of order 100. A physical $\sigma_{13}$ needs the surface-quad route (a segment or 3-D
  shell SG, where both rows are tied), not a rescale of this ring.

The same benchmark folder also records that adding the second-order recovery rung (Eq. 66) changes
$\sigma_{11}$ by an rms of $2.0\times10^{-4}$ MPa against a first-order rms of 88.5 MPa, so the
first-order chain used by `beam_dehom_shell.py` is the right default for this class of section.
