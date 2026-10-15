# How to run OpenSG

OpenSG is a command-line program. Each run reads one structure gene (SG) input file, a yaml that holds the mesh, the
materials and the analysis request, and writes its results next to that file. The same file runs homogenization, which
computes the effective constitutive law of the macro model, and dehomogenization, which recovers the local 3-D stress,
strain and displacement fields from the macro state.

The examples used on this page ship with the repository under `examples/`. Install OpenSG first, as described in
{doc}`installation`.

## Quick start

If the installation has put the `opensg` command on your `PATH`, open a terminal and change to the example folder of a
composite blade cross-section:

```bash
cd examples/OpenSG_shell/1_get_beam_props_from_shell_cross_section
```

and type the following command:

```bash
opensg iea_s10_shell.yaml --center
```

% source: docs/input_format.md, "The 1-D shell SG YAML" (station, mesh size); examples/OpenSG_shell/1_get_beam_props_from_shell_cross_section/iea_s10_shell.yaml (header)
The first argument, `iea_s10_shell.yaml`, is the SG input file: station $r/R = 0.2$ of the IEA-22 blade, meshed as a
1-D shell SG with 307 nodes, 310 line elements, 6 layups and 6 materials. No analysis argument is given, so the run is a
homogenization, the default. The option `--center` states that the node contour is the laminate mid-surface, which is
how the IEA stations are meshed.

OpenSG reads the header of the file, which carries `msg: shell`, `n_model: 1` and `refined: 1`. The run therefore goes
to the msg-shell engine, solves the Reissner-Mindlin (RM) ring and reports the Timoshenko beam $6\times6$. The console
shows what was resolved, then the law and the file it was stored in (abridged):

% source: src/opensg_solid/cli.py (BANNER); src/opensg_shell/cli.py and src/opensg_shell/sg_reference.py (header lines); examples/OpenSG_shell/1_get_beam_props_from_shell_cross_section/iea_s10_shell_Timo.out (first row, Time taken); docs/index.md at 0ef5e43, section "Running OpenSG" (console layout)
```text
 ============================================================================
 OpenSG -- a multiscale structural analysis tool based on the Mechanics of
 Structure Genome (MSG), developed by the Multiscale Structural Mechanics
 Group led by Prof. Wenbin Yu at Purdue University
 ============================================================================

 input     : .../1_get_beam_props_from_shell_cross_section/iea_s10_shell.yaml
 msg       : shell
 SG dim    : 2D
 analysis  : homogenization
 macro model: beam, shear-refined
 reference : center (contour = laminate mid-surface; ABD shifted by t/2, recovery depths -t/2..+t/2)
 note      : the yaml's `reference: center` key is ignored (the reference is a run-time choice) -- it agrees with this run

Timoshenko Beam Stiffness Matrix  [eps11 gam12 gam13 kappa1 kappa2 kappa3]:
[[ 2.76606e+10 -1.72307e-05  4.66505e-06  5.40202e-06  1.66864e+09 -2.07730e+09]
 ...
Homogenization stored in iea_s10_shell_Timo.out
Time taken: 2.82 sec
```

`SG dim` is the ambient dimension, the space the SG occupies, not the dimension of the SG manifold: a cross-section
ring prints `2D`, and a Schwarz-P shell surface, whose facets live in space, prints `3D`.

Two files appear in the folder: `iea_s10_shell_Timo.out`, the effective Timoshenko stiffness and compliance, and
`iea_s10_shell_ABDG.out`, the $8\times8$ RM plate law of every wall section. Add `--mesh` to the command to also write
`iea_s10_shell_mesh.png` and the gmsh mesh `iea_s10_shell.msh`, which show the cross-section as it was read.
{numref}`fig-run-xsec-mesh` shows the same contour coloured by layup, as the example driver `beam_homo_shell.py` draws it.

```{figure} ../_static/shell_xsec_mesh.png
:width: 90%
:name: fig-run-xsec-mesh

The IEA-22 cross-section at $r/R = 0.2$, elements coloured by layup, as drawn by the example driver.
```

:::{note}
`opensg` is a dispatcher. It reads the `msg:` key of the yaml (`shell` or `solid`) and hands the file to the engine
that owns it; without the key, the mesh dialect decides. `opensg_shell iea_s10_shell.yaml --center` and
`python -m opensg_shell iea_s10_shell.yaml --center` run the same analysis and print the same lines.
:::

:::{tip}
The first run of a new problem size is slower than later ones, because JAX compiles its kernels once and caches them
(see {doc}`installation`). The `Time taken` values quoted in this documentation are from runs with a warm cache.
:::

## Command line options

OpenSG is executed with the command `opensg`, followed by the SG file and options, or by a subcommand that converts or
generates files:

% source: src/opensg/cli.py (the forms of the module docstring, printed by `python -m opensg --help`); docs/installation.md (`opensg <sg.yaml> [H|D]`)
```text
opensg <sg.yaml> [H|D] [options]
opensg <subcommand> <file> [options]
```

If no argument is given, or with `-h` / `--help`, `opensg` prints the list of options and subcommands and the solver
menu; `opensg_shell --help` and `opensg_solid --help` also print the header keys each engine reads. {numref}`tab-run-options` lists the options of an analysis run.

% source: output of `python -m opensg --help`, `python -m opensg_solid --help`, `python -m opensg_shell --help`
```{table} Command line options of an analysis run
:name: tab-run-options

| Option | Engine | Description |
|---|---|---|
| `<sg.yaml>` | both | The SG input file. The `msg:` key of its header selects the engine; without the key the mesh dialect decides. |
| `H` | both | Homogenization: compute the effective law of the macro model (the default). |
| `D` | both | Dehomogenization: homogenize, then recover the local fields from the macro state in `<base>.ff` or `epsilon_bar:` in the header. |
| `--center` | shell | The node contour is the laminate mid-surface: the wall ABD (extension, coupling and bending stiffness) is parallel-axis shifted by $t/2$ and the recovery depths run from $-t/2$ to $+t/2$ about the contour. Without it the contour is the outer mold line (OML) and every laminate stacks inward from it. The nodes are never moved. Refused for a solid SG. |
| `--mesh` | both | Also (re)draw `<base>_mesh.png`, elements coloured by material (solid) or by section (shell), and write the gmsh-2.2 mesh `<base>.msh`. A flag, not an analysis: it combines with `H` or `D`. `-m` and `mesh` are accepted too. |
| `--material` | both, `D` only | Output frame of the recovered stress and strain: each element's ply axes (the default). |
| `--global` | both, `D` only | Output frame: the SG axes (solid) or the wall (plate) axes (shell). |
| `--solver F [N]` | solid | Linear solver of the fluctuation solves: the family `direct` (assembled sparse factorization, CPU, the default family) or `iter` (preconditioned conjugate gradient (CG) on the JAX device, the GPU route), with an optional backend number `N`, or a backend name ({numref}`tab-run-solvers`). Without the flag the solver is picked by the number of degrees of freedom. |
| `--gpu` | solid | Shorthand for the `iter` default. |
| `--solver_help` | both | Print the solver menu alone. |
| `-h`, `--help` | both | Print the usage and the header keys the engine reads. |
```

The msg-shell routes always run the direct solver, because every one of them factorizes a saddle point (the drilling
Lagrange multipliers) that conjugate gradient cannot take; given `--solver` or `--gpu`, the shell engine prints a note
and runs the direct solver.

% source: src/opensg_solid/cli.py (SOLVER_MENU, printed by `python -m opensg --help`)
```{table} Linear solvers of the solid engine (option --solver)
:name: tab-run-solvers

| Selection | Name | Description |
|---|---|---|
| no flag | `auto` | `direct` below the wall, `amg` above, the same on every machine; the wall is 1.2e6 degrees of freedom, set by `OPENSG_DIRECT_WALL` |
| `direct 1` | `pardiso` | MKL, multithreaded (the direct default) |
| `direct 2` | `superlu` | scipy fallback: robust, slow |
| `iter 1` | `cg` | matrix-free element-by-element (EBE) Chebyshev CG |
| `iter 2` | `amg` | algebraic multigrid (AMG) preconditioned CG; runs on the GPU when JAX sees one (needs `pyamg`) |
| `iter 3` | `stream` | host-streamed EBE CG for more than 10M degrees of freedom (work in progress) |
| `iter 4` | `gamg` | PETSc GAMG-preconditioned CG via `petsc4py` (needs `petsc4py`; explicit only, `auto` never picks it) |
```

Every solver returns the same digits of the law. The family and the number are given as separate words
(`--solver direct 2`) or the backend by its name (`--solver superlu`).

Everything else about the analysis lives in the yaml header, not on the command line: the macro model (`n_model`),
the refinement (`refined`), the analysis switch (`analysis`), the macro state of a recovery (`epsilon_bar`), the SG
measure (`omega`), the boundary treatment (`aperiodic`) and, for the shell engine, the junction keys. Every key has a
default. {doc}`../user_guide/analysis_settings` documents them.

:::{note}
The analysis argument on the command line wins over an `analysis:` key in the header.
:::

The subcommands of {numref}`tab-run-subcommands` convert meshes and results between formats and run the blade routes.
Each prints its own options with `--help`, for example `opensg msh_to_yaml --help`.

% source: src/opensg/cli.py (module docstring); output of `python -m opensg <subcommand> --help` for each subcommand
```{table} Subcommands of opensg
:name: tab-run-subcommands

| Subcommand | Input and output | Main options |
|---|---|---|
| `gen_windio_cs <windio.yaml>` | blade yaml (windIO v1/v2 or pyNuMAD) to a VABS-layout `.out` record per station plus spanwise `.dat` tables | `--stations S` (`airfoil`, the default, an integer `N`, or comma-separated r values), `--mesh-size H` (default 0.01), `--center`, `--out DIR` (default `cross_sections`), `--prefix TAG` |
| `windio_st <windio.yaml> <r> [r ...]` | blade yaml and span stations r in [0, 1] to the Timoshenko $6\times6$, printed and stored as `<tag>.out` | `--mesh-size H`, `--center`, `--out DIR` (default: the current directory), `--prefix TAG` |
| `pynumad <blade.yaml> <st-id>` | pyNuMAD blade yaml to the Timoshenko $6\times6$ at one station (a 0-based index, or a span r with a decimal point), or at every station with `all` | `--mesh-size H`, `--center`, `--xml`; writes into the current directory |
| `sc_to_yaml <file.sc>` | SwiftComp `.sc` to the solid SG `<base>.yaml` plus `<base>.msh` | `--n_model {1,2,3}` (required), `--refine {0,1}` |
| `yaml_to_sc <file.yaml> [...]` | SG yaml to SwiftComp `<base>.sc` | `--out-base BASE`, `--orientation {bake,ignore}`, `--drop-density` |
| `ff_to_glb <file.ff> [...]` | `.ff` macro state to the SwiftComp dehomogenization input `<base>.sc.glb` | `--yaml YAML`, `--n_model {2,3}`, `--refined {0,1}`, `--out PATH`, `--id1 {auto,0,1}`, `--sc PATH`, `--no-check-sc` |
| `inp_to_yaml <file.inp> [...]` | Abaqus `.inp` mesh to a runnable `<base>.yaml` plus `<base>.msh` | `--n_model {1,2,3}` (required), `--refined`, `--material NAME`, `--map ELSET=MAT ...`, `--out-base BASE`, `--force` |
| `msh_to_yaml <file.msh> [...]` | gmsh 2.2 solid mesh to `<base>.yaml` | `--mat<K> NAME[:ANGLE]` (one per physical tag), `--n_model {1,2,3}`, `--refined {0,1}`, `--materials PATH`, `--h_refine [N]`, `--p_refine`, `--out-base BASE`, `--force` |
```

`gen_windio_cs`, `windio_st` and `pynumad` are also subcommands of `opensg_shell`. The converters are described in
{doc}`../user_guide/converters`.

## Output files

Every homogenization writes one timed `.out` file in the SwiftComp `.K` layout: a banner line naming the model,
`The Effective <Name> Stiffness Matrix`, `The Effective <Name> Compliance Matrix` and a `Time taken` footer in seconds.
`<Name>` is the title of the macro law, the same words the console prints above the matrix
({numref}`tab-run-law-names`). Only the 3-D solid law also prints
`The Engineering Constants (Approximated as Orthotropic)`.

% source: docs/input_format.md (introduction)
```{table} Name of the macro law in the .out file
:name: tab-run-law-names

| Macro law | `<Name>` |
|---|---|
| equivalent 3-D solid $6\times6$ | `Cauchy Continuum` |
| beam $6\times6$ (shear-refined) | `Timoshenko Beam` |
| beam $4\times4$ (classical) | `Euler-Bernoulli Beam` |
| plate ABD $6\times6$ (classical) | `Classical Plate` |
| plate ABDG $8\times8$ (shear-refined) | `Reissner-Mindlin` |
```

The quick-start run, for example, writes this file (abridged):

% source: examples/OpenSG_shell/1_get_beam_props_from_shell_cross_section/iea_s10_shell_Timo.out
```text
 OpenSG msg-shell beam model [ext sh2 sh3 twist bend2 bend3]

 The Effective Timoshenko Beam Stiffness Matrix
 --------------------------------------------
     2.7660599E+010    -1.7230658E-005     4.6650498E-006     5.4020232E-006     1.6686413E+009    -2.0773040E+009
 ...

 The Effective Timoshenko Beam Compliance Matrix
 --------------------------------------------
     3.6320750E-011     9.0011477E-025    -5.2176925E-025    -7.3251858E-026    -1.5696510E-012     9.7940355E-013
 ...

 Time taken: 2.82 sec
```

The file names depend on the engine and the route ({numref}`tab-run-output-files`); `<base>` is the name of the yaml
without its extension.

% source: output of `python -m opensg_shell --help` and `python -m opensg_solid --help`
```{table} Files written by a run
:name: tab-run-output-files

| Run | Files |
|---|---|
| shell, beam, `refined: 1` | `<base>_Timo.out`; the cross-section ring also `<base>_ABDG.out` |
| shell, beam, `refined: 0` | `<base>_EB.out` |
| shell, `n_model: 3` | `<base>_C3D.out` and `<base>_ABDG.out` |
| shell, `D` | in addition `<base>_dehom.txt` and `<base>_dehom.vtk`, plus the sidecar `<base>_dehom.junc` under the default `junction: flag` |
| solid, `H` | `<base>.out` and `<base>_mesh.png`, redrawn whenever the PNG is older than the yaml |
| solid, `D` | in addition `<base>_dehom.txt`, `.vtk`, `.SM`, `.EM` and `.U` |
| any run with `--mesh` | `<base>_mesh.png` and `<base>.msh` |
```

`<base>_ABDG.out` is the step-1 record of every route that reduces a layup to a wall plate law: one $8\times8$ RM ABDG
stiffness and its compliance per section. {doc}`../user_guide/output_files` describes every file and its columns.

## Running cases

Some typical use cases are given below.

### Case 1: Homogenize a shell cross-section

```bash
opensg <cross_section.yaml> [--center] --mesh
```

A 1-D shell SG is the contour of a thin-walled section, each element carrying a laminate. The header decides the
macro model: `n_model: 1` with `refined: 1` gives the Timoshenko $6\times6$ in `<base>_Timo.out`, `refined: 0` the
Euler-Bernoulli $4\times4$ in `<base>_EB.out`, and `n_model: 3` the equivalent 3-D solid in `<base>_C3D.out`. `--mesh`
writes the mesh picture and the gmsh mesh, to check the input before looking at the numbers. Pass `--center` only when
the contour is the laminate mid-surface, as for the IEA stations. For the IEA station of the quick start:

% source: docs/input_format.md, "Running it"
```bash
opensg iea_s10_shell.yaml --center --mesh
```

The example folder also ships a driver that runs the same homogenization through the Python API and adds the two
figures the command line does not emit, the contour coloured by layup (`iea_s10_mesh.png`) and the material
orientation check (`iea_s10_orient.png`):

```bash
python beam_homo_shell.py
```

A 3-D shell SG, a surface meshed with triangles or quadrilaterals, runs the same way. In
`examples/OpenSG_shell/4_get_solid_props_from_shell_3D_SG`, the header of `schwarz_p_3Dshell.yaml` asks for
`n_model: 3`, and

% source: docs/input_format.md, "The 3-D shell SG YAML"
```bash
opensg schwarz_p_3Dshell.yaml
```

writes `schwarz_p_3Dshell_C3D.out` and `schwarz_p_3Dshell_ABDG.out`.

:::{warning}
`--center` must match the mesh. Where the contour sits (outer mold line or mid-surface) is decided when the yaml is
generated, and OpenSG never moves the nodes. Running a mid-surface mesh without `--center`, or an OML mesh with it,
shifts the coupling and bending blocks of the wall ABD by a parallel-axis term of the wall thickness and changes every
bending and torsion entry of the beam $6\times6$.
:::

### Case 2: Homogenize a solid SG

```bash
opensg <sg.yaml>
```

A solid SG is a 1-D, 2-D or 3-D mesh of solid elements with a material per element. The 2-D honeycomb plate cell of
`examples/OpenSG-solid/3_get_plate_props_from_2D_SG` starts with this header:

% source: examples/OpenSG-solid/3_get_plate_props_from_2D_SG/RHC_SW_2UC_45.yaml
```yaml
n_model: 2      # 1 = beam, 2 = plate, 3 = solid -- the macro model this SG homogenizes to
refined: 1      # 0 = classical (plate ABD / beam EB); 1 = shear-refined (plate ABDG / beam Timoshenko)
msg: solid      # the ENGINE this SG belongs to (opensg_solid); `opensg <yaml>` dispatches on it
```

From that folder, the command

```bash
opensg RHC_SW_2UC_45.yaml
```

writes `RHC_SW_2UC_45.out` and `RHC_SW_2UC_45_mesh.png`. Since the file asks for `refined: 1`, the law is the
$8\times8$ RM plate stiffness ABDG, with the banner and title

% source: examples/OpenSG-solid/3_get_plate_props_from_2D_SG/RHC_SW_2UC_45.out
```text
 OpenSG msg-solid plate model, omega 35.248001, periodic

 The Effective Reissner-Mindlin Stiffness Matrix
```

and the footer ` Time taken: 1.78 sec`. With `refined: 0` the same file reports the classical $6\times6$ ABD, titled
`The Effective Classical Plate Stiffness Matrix`. The number after `omega` is the SG measure, taken from the mesh.

For a large mesh, pick the linear solver with `--solver` ({numref}`tab-run-solvers`); on a GPU, pass `--solver iter`.

:::{tip}
A plain laminate needs no mesh file: `layup_db.yaml` in `examples/OpenSG-solid/1_get_plate_props_from_1DSG` lists the
plies, and `python 1d_sg.py` generates the through-thickness 1-D SG and homogenizes it.
:::

### Case 3: Recover the 3-D fields

```bash
opensg <sg.yaml> D
```

After a structural analysis of the macro model, the local stress and strain at a location follow from the macro state
there. The `D` argument runs the homogenization and the recovery in one pass, so no separate homogenization run is
needed. The macro state is read from `<base>.ff` next to the yaml, or from `epsilon_bar:` in the header when there is
no `.ff`. For the blade cross-section, the example folder `examples/OpenSG_shell/2_get_beam_dehom_from_shell_cross_section`
carries the beam forces of that station in `iea_s10_shell.ff`:

% source: examples/OpenSG_shell/2_get_beam_dehom_from_shell_cross_section/iea_s10_shell.ff
```yaml
# OpenSG dehomogenization macro state -- read by `opensg_shell iea_s10_shell.yaml D`
# The SG yaml describes the STRUCTURE; this file carries the LOADING.
u:     [0.0, 0.0, 0.0]      # macro displacement u1 u2 u3
theta: [0.0, 0.0, 0.0]      # macro rotation t1 t2 t3 [rad] (about y1 y2 y3)
C:     [[1.0, 0.0, 0.0],       # direction cosines of the macro frame
        [0.0, 1.0, 0.0],       #   (identity = unrotated; overrides theta)
        [0.0, 0.0, 1.0]]
# generalized macro forces, beam [F1 F2 F3 M1 M2 M3]
# IEA-22 station r/R = 0.20 (ff51_rmc_reform.dat row 10, RM center-ref BeamDyn)
FF:    [4.19268867e+04, 8.26739746e+03, 1.46733512e+06, 1.21914500e+06, -6.01849720e+07, 3.04264594e+05]
```

From that folder, the command

```bash
opensg iea_s10_shell.yaml D --center
```

% source: src/opensg_shell/cli.py (dehom_write, n_depth=9; console lines); docs/tutorials/msg_shell_beam.md ("opensg iea_s10_shell.yaml D --center")
writes `iea_s10_shell_dehom.txt`, `iea_s10_shell_dehom.vtk` and `iea_s10_shell_dehom.junc`, and closes with
`Local field files are computed and stored.` and one `Time taken` line for the whole run. The `.txt` holds one row per
recovery point, nine through the thickness of every element, and the `.vtk` is a legacy VTK file for a viewer such as
ParaView. The driver
`beam_dehom_shell.py` in the same folder runs the same recovery through the Python API and plots it
({numref}`fig-run-dehom-stress`).

```{figure} ../_static/shell_dehom_stress.png
:width: 80%
:name: fig-run-dehom-stress

Axial stress $\sigma_{11}$ (top) and in-plane shear stress $\sigma_{12}$ (bottom), in MPa, recovered over the IEA-22
cross-section at $r/R = 0.2$.
```

A solid SG recovers the same way. In `examples/OpenSG-solid/4_get_plate_dehom_from_2DSG`, `RHC_SW_2UC_45.ff` carries the
plate forces `[N11 N22 N12 M11 M22 M12]`, and

% source: examples/OpenSG-solid/4_get_plate_dehom_from_2DSG/RHC_SW_2UC_45.ff (header comment)
```bash
opensg_solid RHC_SW_2UC_45.yaml D
```

writes `RHC_SW_2UC_45_dehom.txt`, `.vtk`, `.SM`, `.EM` and `.U`.

:::{note}
The macro forces must be referred to the same axis as the stiffness. The forces of `iea_s10_shell.ff` are referred to
the mid-surface, which is why the recovery runs with `--center`.
:::

:::{note}
The recovered stress and strain are written in each element's ply axes by default. Add `--global` for the SG axes
(solid) or the wall axes (shell). A solid SG with per-element frames (`elementOrientations`) refuses `--material`.
The displacement is never rotated.
:::

The `.ff` format and the recovered fields are described in {doc}`../user_guide/recovery`.

### Case 4: Convert a mesh from gmsh or Abaqus

```bash
opensg msh_to_yaml <file.msh> --mat<K> NAME[:ANGLE] --n_model {1,2,3}
```

A gmsh mesh carries geometry and physical tags only. `msh_to_yaml` writes the mesh side of the solid SG yaml from a
gmsh 2.2 ASCII `.msh` and fills the materials from the material library, one `--mat<K>` flag per physical tag `K`:

% source: docs/input_format.md, "From a gmsh mesh (solid SG)"
```bash
opensg msh_to_yaml cell.msh --mat1 Al69 --mat2 epoxy --n_model 3
```

Tags without a flag, and `n_model` without `--n_model`, stay `FILL_IN` placeholders, and OpenSG refuses to run the yaml
until they are filled. A mesh can also go straight to the analysis: given a `.msh`, `opensg` converts it with the
conversion flags and analyses the yaml it emits,

% source: src/opensg/cli.py, main()
```bash
opensg core.msh --mat1 Al --n_model 2
```

The converter reads the legacy ASCII format 2.2 only. It refuses gmsh's default format 4.1 and binary files and prints
the fix, `gmsh cell.msh -format msh22 -o cell_msh22.msh -save`.

An Abaqus deck carries its own materials, so the yaml that `inp_to_yaml` writes is runnable; `--n_model` is required and
never guessed:

% source: output of `python -m opensg inp_to_yaml --help`
```bash
opensg inp_to_yaml <file.inp> --n_model {1,2,3} [--refined]
```

The element-set to material binding follows the section cards of the deck. `--map` sets it explicitly, and a value
`MATERIAL@DEGREES` makes plies that share one deck material but differ in orientation separate SG materials, for
example `--map P1=Aluminum L1=Laminate@45 L2=Laminate@-45`.

A SwiftComp `.sc` converts with `opensg sc_to_yaml <file.sc> --n_model {1,2,3}`, and `yaml_to_sc` and `ff_to_glb`
write the SwiftComp inputs back from an SG yaml and a `.ff`. A shell surface mesh from gmsh goes through the Python
helper `opensg_shell.helper.msh_to_yaml`. {doc}`../user_guide/converters` covers every converter.

## Running from Python

Every entry function is also importable, for driving a run from your own script. The shell cross-section of the quick
start:

% source: docs/tutorials/msg_shell_beam.md, "The one call"
```python
from opensg_shell import build_rm_bundle

B = build_rm_bundle("iea_s10_shell.yaml", ref="center")   # writes iea_s10_shell_Timo.out
B["Timo"]                                                  # the Timoshenko 6x6
```

The solid plate cell of Case 2:

% source: docs/input_format.md, "The solid SG YAML"
```python
from opensg_solid.sg_homo import plate_homo_2d

r = plate_homo_2d("RHC_SW_2UC_45.yaml")   # header + materials from the file
print(r["law_title"], r["law"])
```

Every example folder ships such a driver script, run as `python <script>.py` from its folder; {doc}`../examples/index`
lists them.
