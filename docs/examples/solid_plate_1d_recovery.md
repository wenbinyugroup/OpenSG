# Ply stresses of a sandwich plate from a 1-D SG

## Problem description

Homogenization keeps the plate law and discards the through-thickness detail; recovery
(dehomogenization) puts it back. Given the section resultants at a point of the plate, this
example evaluates the 3-D strain, stress, warping and displacement at every depth of the
through-thickness structure gene (SG) of {doc}`solid_plate_1d_layup`: the same
$[0/90/0/90/\text{core}]_s$ sandwich, the same `layup_db.yaml` and the same 1-D SG `1dsg.yaml`
({numref}`fig-sp1-sg`), both shipped again in this folder. The engine is `opensg_solid`, through
its Reissner-Mindlin (RM) plate recovery `opensg_solid.rm_plate_1D.rm_dehom`. The recovery
accuracy is measured against the Pagano cylindrical-bending elasticity solutions and the cases of
Yu, Hodges and Volovoi (2003).

Two driver scripts cover the two kinds of input:

- `rm_dehom.py` recovers one station from resultants typed into the script.
- `rm_dehom_ff.py` recovers a whole field from a `.ff` station table, and is the one to use for
  real work.

% source: examples/OpenSG-solid/2_get_plate_stress_from_1DSG/layup_db.yaml (plate:, dehom:), examples/OpenSG-solid/2_get_plate_stress_from_1DSG/ex5_shell_S8R_field.ff (header)
The shipped table holds the shell resultants of a square sandwich plate of side $a = b = 1.524$ m
under the double-sine pressure $q_0 \sin(\pi x/a)\sin(\pi y/b)$ with $q_0 = 68.9476\times10^{6}$ Pa.
They were collected from an Abaqus S8R shell solution on a $40\times40$ rectangular station
lattice, at the first deflection peak ($t = 2.65$ ms, increment 53) of the transient shell
analysis. Complete input files are in `examples/OpenSG-solid/2_get_plate_stress_from_1DSG/`.

% source: docs/tutorials/msg_solid_plate.md (section B, "What it computes")
The recovery is linear in 42 drivers,

$$D = [\,E_6,\; E_{6,1},\; E_{6,2},\; E_{6,11},\; E_{6,12},\; E_{6,22},\; qt_6\,],$$

so the code builds three operators once per depth from 42 unit evaluations,
$T_E, T_S \in \mathbb{R}^{6\times42}$ and $T_W \in \mathbb{R}^{3\times42}$, and every station is
then one matrix-vector product, `jax.vmap`-ed over the whole field. The plate strains come from
the section law itself, $E_6 = A_6^{-1} F_6$, so a recovery never bypasses the homogenization.
{numref}`tab-sp2-drivers` lists which driver feeds which stress component.

% source: docs/tutorials/msg_solid_plate.md (section B, driver table)
```{list-table} The recovery drivers
:header-rows: 1
:name: tab-sp2-drivers

* - Driver
  - Where it comes from
  - What it produces
  - Without it
* - $E_6$
  - $A_6^{-1} F_6$ at the station
  - $\sigma_{11}, \sigma_{22}, \sigma_{12}$ layer by layer
  - nothing
* - $E_{6,1}, E_{6,2}$
  - in-plane gradients of the resultants
  - $\sigma_{13}, \sigma_{23}$ (the $V_{11}/V_{12}$ warping)
  - $\sigma_{13} = \sigma_{23} \equiv 0$
* - $E_{6,11}, E_{6,12}, E_{6,22}$
  - second gradients
  - the $\sigma_{33}$ interior build-up
  - $\sigma_{33}$ keeps only its face terms
* - $qt_6$
  - the applied surface pressure ladder $[q,\, q_{,1},\, q_{,2},\, q_{,11},\, q_{,12},\, q_{,22}]$
  - the face traction $\sigma_{33} = -q$
  - $\sigma_{33} = 0$ at that face
```

$Q_1, Q_2$ are not part of $F_6$: there is no shear slot among the 42 drivers, because in the
asymptotic construction the transverse shears are produced by the gradient terms. They enter
only as per-station rescale targets, $\int\sigma_{13}\,dz = Q_1$ and $\int\sigma_{23}\,dz = Q_2$,
so the profile shape comes from the gradients and the carried force is exactly the one supplied.

## Input

### One station: `rm_dehom.py`

The input is the `USER INPUT` block of the script:

% source: examples/OpenSG-solid/2_get_plate_stress_from_1DSG/rm_dehom.py
```python
SG_YAML = os.path.join(HERE, "1dsg.yaml")
# FF data: the section resultants at the station.  NO defaults -- these
# are YOURS.  The script refuses to run until both are set.
F6 = [1e6, 0, 0, 0, 0, 0]     # [N11, N22, N12, M11, M22, M12]  [N/m, N]
Q2v = [0.0, 0.0]              # [Q1, Q2]  [N/m]  (drives the rescale)
# optional refinements -- None means zero; a bare FF gives the classical
# ply stresses (s11/s22/s12) and NO transverse shear or s33.  A SINGLE
# FF fixes no gradients (the equilibrium closure is REMOVED -- it is
# indeterminate); gradients come ONLY from finite differences of an FF
# FIELD (a station line or a 2-D lattice: rm_dehom_dat.py), or from a
# known analytical mode.  Details: dehom_README.md.
DE1 = DE2 = None              # d(E6)/dx, d(E6)/dy
D11 = D12 = D22 = None        # second gradients
QT6 = None                    # surface-load ladder [q q,1 q,2 q,11 q,12 q,22]
LATTICE = "gauss"             # "gauss" (default) or "nodes"
```

`F6` and `Q2v` must both be set; the script raises a `ValueError` when either is `None`, since a
built-in number would pass for a result. `LATTICE` selects the output depths, the Gauss points of
the SG elements (default) or the SG nodes. One station has no neighbours, so it cannot supply
gradients: with every optional input left at `None` the run is the classical recovery,
$\sigma_{11}, \sigma_{22}, \sigma_{12}$ exact layer by layer, $\sigma_{13} = \sigma_{23} = 0$
exactly and $\sigma_{33}$ at machine zero.

### A field of stations: `rm_dehom_ff.py`

A `.ff` file is the plate analogue of a VABS `.glb`: whitespace separated, `#` comments, one row
per in-plane station, in any row order and on any station geometry. Any solver can write it; the
driver contains no Abaqus. The first three lines of the shipped table state the format:

% source: examples/OpenSG-solid/2_get_plate_stress_from_1DSG/ex5_shell_S8R_field.ff
```text
# plate dehom FF station table (collected from ex5_shell_S8R_field.dat)
# x[m] y[m] N11 N22 N12[N/m] M11 M22 M12[N] Q1 Q2[N/m] u1 u2 u3[m] wx wy[-]
# 40 x 40 rectangular station lattice, x-fastest
```

% source: examples/OpenSG-solid/2_get_plate_stress_from_1DSG/rm_dehom_ff.py (docstring); docs/tutorials/msg_solid_plate.md (section B)
Ten columns are required (`x y N11 N22 N12 M11 M22 M12 Q1 Q2`). The optional columns 11 to 15 add
the mid-surface displacement $u_1, u_2, u_3$ and the slopes $w_x, w_y$; without them the `.U` file
carries the warping alone. Every gradient driver is computed from the stations themselves, by one
of two paths chosen automatically: exact tensor-grid finite differences when the coordinates form
a complete rectangular lattice, or a meshfree KD-tree patch fit ($k = 12$ nearest neighbours,
distance-weighted local quadratic least squares, first and second derivatives from one batched
solve) for any other cloud.

The applied face pressure is not in the table, because loads are input and not solver output. It
lives in the `dehom:` block of `layup_db.yaml`:

% source: examples/OpenSG-solid/2_get_plate_stress_from_1DSG/layup_db.yaml
```yaml
dehom:
  ff: Abaqus_results/ex5_shell_S8R_field.ff
  loads:
    top:                    # the ex5 double-sine pressure
      q0: 68.9476e6         #   amplitude [Pa]
      shape: sinsin         #   uniform | sin_x | sin_y | sinsin
    bottom: none            # traction-free
```

Each `shape` fixes the whole six-term ladder analytically, differentiated with `a` and `b` from
the `plate:` block. Bottom-face loads raise `NotImplementedError`: the 42-driver operator block
carries the top ladder only. The `V2` key of the same file sets the recovery order
({numref}`tab-sp2-v2`).

% source: examples/OpenSG-solid/2_get_plate_stress_from_1DSG/layup_db.yaml (V2 comment), examples/OpenSG-solid/2_get_plate_stress_from_1DSG/rm_dehom_ff.py
```{list-table} The recovery-order key of layup_db.yaml
:header-rows: 1
:name: tab-sp2-v2

* - Key
  - Value
  - Meaning
* - `V2`
  - `1` (default, and the shipped value)
  - second-order recovery, Yu Eq. (66): the $V_{21}/V_{22}/V_{23}$ strain columns, with the second gradients $E_{6,11}, E_{6,12}, E_{6,22}$ taken from the stations
* - `V2`
  - `0`
  - first-order recovery, Yu Eq. (63): the second gradients are set to zero
```

:::{warning}
The shipped `dehom: ff:` key reads `Abaqus_results/ex5_shell_S8R_field.ff`, but the table sits at
the folder root and no `Abaqus_results` folder is shipped. Name the table on the command line, as
in the Run section, or correct the key.
:::

## Run

```bash
cd examples/OpenSG-solid/2_get_plate_stress_from_1DSG
python rm_dehom.py
python rm_dehom_ff.py ex5_shell_S8R_field.ff
```

% source: examples/OpenSG-solid/2_get_plate_stress_from_1DSG/rm_dehom_ff.py (argparse)
```{list-table} Arguments of rm_dehom_ff.py
:header-rows: 1
:name: tab-sp2-args

* - Argument
  - Meaning
* - `ff` (positional, optional)
  - the `.ff` station table; default: `dehom: ff` in the database
* - `--db`
  - the `layup_db.yaml` to read; default: the one next to the script
* - `--scatter`
  - force the meshfree (patch fit) gradient path even when the stations form a lattice
```

Both scripts re-run the homogenization on the SG line before the recovery. `rm_dehom.py` writes
`1dsg_dehom.SM`, `.EM`, `.U` and `.vtk` (the VTK is part of the default output) and prints the
peak of each stress component. `rm_dehom_ff.py` prints its chosen gradient path, the operator
build, the `vmap` over the stations and the peak of each stress component:

% source: examples/OpenSG-solid/2_get_plate_stress_from_1DSG/rm_dehom_ff.py (the block shows the format of the script's print statements, start to end); counts from docs/tutorials/msg_solid_plate.md (section B)
```text
start: <date time>
homo re-run: 9 plies -> 45 SG-element Gauss depths
gradient drivers: lattice FD (40 x 40) in <seconds> s
building operators: (45 + 61 fine) x 42 unit drivers ...
operators built in <seconds> s
jax.vmap dehomogenization over 1600 stations ...
wrote ex5_shell_S8R_field_ff.SM / .EM / .U / _gauss.vtk  (72000 points)
  max|S11| = <value> Pa
  ...
end:   <date time>
```

## Result

The field outputs are named after the table, `<ff stem>_ff` ({numref}`tab-sp2-out`).

% source: docs/tutorials/msg_solid_plate.md (section B, "What comes out")
```{list-table} Output files of rm_dehom_ff.py
:header-rows: 1
:name: tab-sp2-out

* - File
  - Content
* - `ex5_shell_S8R_field_ff.SM`
  - 3-D stress: columns `x y z` then `S11 S22 S33 S12 S13 S23` in Pa
* - `ex5_shell_S8R_field_ff.EM`
  - 3-D strain, same layout, `E11` through `E23`
* - `ex5_shell_S8R_field_ff.U`
  - 3-D displacement `U1 U2 U3` in m
* - `ex5_shell_S8R_field_ff_gauss.vtk`
  - ASCII `STRUCTURED_GRID`, `DIMENSIONS 40 40 45`, 72 000 points, written only on a lattice
```

% source: docs/tutorials/msg_solid_plate.md (section B, "What comes out")
The three text files share a two-line header naming the recovery and its station set, for example
`1600 .ff stations [lattice FD (40 x 40)] x the 45 SG-element Gauss depths`. The conventions that
matter when reading them:

- $z$ is in the SG frame: $z = 0$ is the reference surface, not the mid-plane, unless `fraction`
  puts them at the same place.
- The depths are the Gauss abscissae of each five-noded element, five per ply and 45 in all, so
  every output point is strictly inside one ply and a material interface, where stress is
  double-valued, never appears in a file. `LATTICE = "nodes"` switches to the SG nodes in
  `rm_dehom.py`; `rm_dehom_ff.py` always writes the Gauss depths.
- Stress and strain are written in the ply material frame of each depth, where failure criteria
  live. The chain computes in the laminate frame, since the gradients and the $Q$ rescale
  integrate across plies, and rotates only at write time. The displacement `.U` stays global.
- The component order in the files is `11 22 33 12 13 23`, reordered on write from the internal
  SG Voigt order `(11, 22, 33, 23, 13, 12)`.

```{figure} ../_static/pv_plate_1dsg_s11.png
:width: 420px
:name: fig-sp2-s11

The recovered in-plane stress $\sigma_{11}$ (Pa) over the plate on one depth slice of the
recovered field, displayed in ParaView.
```

% source: docs/theory/rm_plate_1d_user_guide.md ("Accuracy guidance"), docs/tutorials/msg_solid_plate.md (section B, "What it is validated against")
The recovery is measured against reference elasticity solutions: the Pagano cylindrical-bending
solutions at slenderness $S = 4, 10, 50$ and the Yu 2003 cases 1 to 3. The errors fall at
$\mathcal{O}(S^{-2})$ in the slenderness $S$ ({numref}`tab-sp2-accuracy`). Soft-core sandwiches
need a larger $S$ for the in-plane second-order recovery; their $\sigma_{13}$, $\sigma_{33}$ and
displacements stay accurate at $S = 10$.

% source: docs/theory/rm_plate_1d_user_guide.md ("Accuracy guidance")
```{list-table} Recovery error against the elasticity solutions
:header-rows: 1
:name: tab-sp2-accuracy

* - Component (chain)
  - $S=4$
  - $S=10$
  - Thin
* - $\sigma_{13}$ (cross-ply)
  - 21 %
  - 4.4 %
  - 0.18 % at $S=50$
* - $\sigma_{33}$ equilibrium
  - 5.5 %
  - 1.2 %
  - 0.05 %
* - $\sigma_{11}$ second-order
  - 27 %
  - 2.0 %
  - 0.04 % at $S=64$
* - $\sigma_{12}$ angle-ply
  - 18 %
  - (not listed)
  - 0.07 % at $S=64$ (order 2.0)
* - $U_1$
  - 30 %
  - 1.4 %
  - 0.02 % at $S=64$
```

% source: examples/OpenSG-solid/2_get_plate_stress_from_1DSG/layup_db.yaml (V2 comment), examples/OpenSG-solid/2_get_plate_stress_from_1DSG/dehom_README.md
For the choice of `V2`, the second-order recovery gives the better $\sigma_{33}$ for every stack
checked against the Pagano static solutions, with exact face tractions: homogeneous 0.5 to 2.6 %,
$[15/-15]$ 1.1 %, $[0/90/0]$ 1.2 to 5.6 %, Yu case 3 0.8 to 4.3 %, sandwich 0.1 to 16.7 %. The
in-plane face stress of a thick section still prefers `V2: 0`: on this example the $\sigma_{xx}$
peak is off by +8.67 % at second order and +0.32 % at first order.

% source: docs/tutorials/msg_solid_plate.md (section B), examples/OpenSG-solid/2_get_plate_stress_from_1DSG/dehom_README.md
The two gradient routes confirm each other. The finite-difference path and the analytically
differentiated known-mode path, run side by side on the same field, agree to better than 1 % on
every component in the interior (four integration points trimmed per edge). The 11 to 15 %
whole-field difference is the one-sided stencil band, about two elements wide, at the boundary.
`dehom_README.md` in the example folder derives the full chain equation by equation.

## Files

```{list-table} Files of the example
:header-rows: 1
:name: tab-sp2-files

* - File
  - Role
* - `layup_db.yaml`
  - input: the laminate of {doc}`solid_plate_1d_layup`, plus `V2`, `plate:` and `dehom:`
* - `1dsg.yaml`
  - input: the 1-D SG mesh
* - `ex5_shell_S8R_field.ff`
  - input: the $40\times40$ station table of shell resultants and mid-surface displacements
* - `rm_dehom.py`
  - driver: one station, resultants in the `USER INPUT` block
* - `rm_dehom_ff.py`
  - driver: a field from a `.ff` table
* - `dehom_README.md`
  - the recovery chain, equation by equation
* - `1dsg_dehom.SM`, `.EM`, `.U`, `.vtk`
  - written by `rm_dehom.py`
* - `ex5_shell_S8R_field_ff.SM`, `.EM`, `.U`, `_gauss.vtk`
  - written by `rm_dehom_ff.py`
```

The recovery keys and output frames of the other routes are described in
{doc}`../user_guide/recovery`.
