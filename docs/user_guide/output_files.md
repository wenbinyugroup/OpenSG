# Output files

Every homogenization writes one timed text file, the `.out`, in the layout of a SwiftComp `.K`
file, so the results drop straight into existing comparison tooling. One writer,
`opensg_solid.sg_homo.write_sc_K`, produces it for every route of both engines. A shell route
also writes the wall plate law it passed through, a dehomogenization (`D`) run adds the
recovered local fields, and the mesh figures appear on request. This page lists each file, its
name and its contents.

## The `.out` layout

% source: docs/theory/rules/output_format_and_timing.md
```text
 OpenSG <model>, omega <value>

 The Effective <Name> Stiffness Matrix
 --------------------------------------------
     <6 columns, Fortran E+00x>
 ...
 The Effective <Name> Compliance Matrix
 --------------------------------------------
 ...
 The 6X6 Mass Matrix                                         <- beam route, mass= given
 ========================================================

 The Engineering Constants (Approximated as Orthotropic)     <- 3-D laws only
 ----------------------------------------------------------
  E1  = ...  nu23= ...

 Time taken: <x.xx> sec
```

- The ` OpenSG ...` banner is the **first line**. It names the model and, where the route
  reports them, the SG measure $\omega$ and the boundary treatment; the shell beam banner below
  carries the strain order instead.
- The compliance is the exact inverse of the stiffness.
- ` Time taken: <t> sec` is the **last line**, in seconds.
- The engineering-constants block appears only for a 3-D law; the mass matrix only on the beam
  route when a mass matrix is computed.
- Units follow the input: an SG in metres and pascals gives N, N·m and Pa; one in millimetres and
  MPa gives the corresponding mixed units.

The terminal prints the same matrix under the same title, then `Homogenization stored in` and
the file name, then `Time taken`. The beam law of the IEA-22 cross-section at $r/R = 0.20$ reads:

% source: examples/OpenSG_shell/1_get_beam_props_from_shell_cross_section/iea_s10_shell_Timo.out
```text
 OpenSG msg-shell beam model [ext sh2 sh3 twist bend2 bend3]

 The Effective Timoshenko Beam Stiffness Matrix
 --------------------------------------------
     2.7660599E+010    -1.7230658E-005     4.6650498E-006     5.4020232E-006     1.6686413E+009    -2.0773040E+009
    -1.7230658E-005     7.1861326E+008     5.7520370E+007    -1.0192044E+008     2.3079953E-006     5.2738749E-006
     4.6650498E-006     5.7520370E+007     4.2174428E+008    -1.6217837E+008     8.4816767E-006     2.7931393E-007
     5.4020232E-006    -1.0192044E+008    -1.6217837E+008     2.4375653E+009     1.6449557E-006    -8.1140553E-006
     1.6686413E+009     2.3079953E-006     8.4816767E-006     1.6449557E-006     3.5143887E+010    -5.5571249E+009
    -2.0773040E+009     5.2738749E-006     2.7931393E-007    -8.1140553E-006    -5.5571249E+009     6.8129724E+010

 The Effective Timoshenko Beam Compliance Matrix
 --------------------------------------------
     3.6320750E-011     9.0011477E-025    -5.2176925E-025    -7.3251858E-026    -1.5696510E-012     9.7940355E-013
     9.0011477E-025     1.4122557E-009    -1.7436709E-010     4.7448635E-011    -1.0896890E-025    -8.4399248E-026
    -5.2176925E-025    -1.7436709E-010     2.4548905E-009     1.5604036E-010    -5.6993055E-025    -4.0379194E-026
    -7.3251858E-026     4.7448635E-011     1.5604036E-010     4.2261118E-010    -5.0809674E-026     3.9641229E-026
    -1.5696510E-012    -1.0896890E-025    -5.6993055E-025    -5.0809674E-026     2.8894079E-011     2.3089388E-012
     9.7940355E-013    -8.4399248E-026    -4.0379194E-026     3.9641229E-026     2.3089388E-012     1.4896077E-011

 Time taken: 2.82 sec
```

## Matrix titles and row order

`<Name>` is always the console title of the macro law, so the line in the file and the line the
terminal printed are word for word the same. There are exactly five:

% source: docs/theory/constitutive.md; docs/theory/rules/output_format_and_timing.md
| macro law | `<Name>` | size | rows |
|---|---|---|---|
| 3-D solid (Cauchy continuum) | `Cauchy Continuum` | $6\times6$ | $[11\;22\;33\;23\;13\;12]$, engineering shears |
| beam, shear-refined | `Timoshenko Beam` | $6\times6$ | $[\bar\epsilon_{11}\;\gamma_{12}\;\gamma_{13}\;\kappa_1\;\kappa_2\;\kappa_3]$; the diagonal is $EA$, $GA_2$, $GA_3$, $GJ$, $EI_2$, $EI_3$ |
| beam, classical | `Euler-Bernoulli Beam` | $4\times4$ | $[\bar\epsilon_{11}\;\kappa_1\;\kappa_2\;\kappa_3]$ |
| plate ABD, classical | `Classical Plate` | $6\times6$ | `[N11 N22 N12 M11 M22 M12]` against $[\varepsilon_{11}\;\varepsilon_{22}\;2\varepsilon_{12}\;\kappa_{11}\;\kappa_{22}\;\kappa_{12}]$ |
| plate ABDG, shear-refined | `Reissner-Mindlin` | $8\times8$ | `[e11 e22 g12 k11 k22 k12 2g13 2g23]` |

Here ABD is the extension, coupling and bending stiffness of a plate, and ABDG adds the
transverse-shear block $G$ of the Reissner-Mindlin (RM) model. An empty name would give the
anonymous ` The Effective Stiffness Matrix`; no macro law uses it. The per-section wall laws of a
shell run come from a different writer, `opensg_shell.sg_homo.write_abdg_out`, and keep the
title `The Effective Reissner-Mindlin Plate Stiffness Matrix`.

## File names

% source: docs/theory/rules/output_format_and_timing.md; examples/OpenSG-solid/1_get_plate_props_from_1DSG/1d_sg.py
| route | file |
|---|---|
| msg-solid beam, plate or 3-D solid (`opensg <base>.yaml`) | `<base>.out` |
| msg-solid through-thickness laminate (`1d_sg.py`) | `<layup_db stem>_plate_homo.out` |
| msg-shell beam, Timoshenko (cross-section ring or 3-D segment) | `<base>_Timo.out` |
| msg-shell beam, classical (Kirchhoff-Love wall) | `<base>_EB.out` |
| msg-shell equivalent solid of a cross-section SG | `<base>_C3D.out` |
| msg-shell 3-D shell SG | `<base>_C3D.out` |
| the wall plate law of any shell route that builds one | `<base>_ABDG.out` |

There is one file per model and no `_K` suffix. When two routes can run on the same yaml, the
suffix tells them apart (`_Timo` and `_C3D`); a run that reuses a name overwrites the earlier
file.

:::{note}
The `*_plate_ABD.out` files shipped in examples 3 and 4, and `RHC_SW_2UC_45_beam_Timo.out` in
example 8, are plain `np.savetxt` matrices under a
single comment line, not the timed `.K` layout.
:::

## Banners and the SG measure

The banner states $\omega$ because the same stiffness has several legitimate normalizations. In
every case $\omega$ is measured from the mesh unless the yaml header declares `omega:`, which wins:

- **Cross-section SG:** the node bounding box (the periodic cell that the assembly ties), not the
  convex hull of the contour; the two differ on every honeycomb and lattice.
- **3-D shell SG** and **3-D solid SG:** the node bounding-box volume, not the summed element
  volume, so the `.out` compares directly with a SwiftComp `.K`.

Banners copied from the shipped results:

% source: examples/OpenSG-solid/3_get_plate_props_from_2D_SG/RHC_SW_2UC_45.out, examples/OpenSG-solid/7_get_beam_props_from_SG/RHC_SW_2UC_45.out, examples/OpenSG-solid/6_get_solid_props_from_3D_SG/sample_1/Sample_1.out, examples/OpenSG_shell/4_get_solid_props_from_shell_3D_SG/schwarz_p_3Dshell_C3D.out
```text
 OpenSG msg-solid plate model, omega 35.248001, periodic
 OpenSG msg-solid beam model, omega 1
 OpenSG msg-solid 3D elastic model, omega 1, periodic
 OpenSG msg-shell equivalent 3D solid (3-D shell SG, 13536 nodes, 26360 elems [26360 tri3/MITC3], 0 junction edges, periodic in 3 dirs, 26360 element drilling multipliers, per unit cell 1)
```

:::{tip}
A whole-matrix ratio that is uniform across every entry (for example exactly 0.5000) between two
results is a normalization difference, not a physics discrepancy. Check $\omega$ first.
:::

## The 3-D law: engineering constants

Only a 3-D law prints the block of engineering constants, computed from the compliance
$\mathbf S = \mathbf C^{-1}$ by $E_1 = 1/S_{11}$, $E_2 = 1/S_{22}$, $E_3 = 1/S_{33}$,
$G_{12} = 1/S_{66}$, $G_{13} = 1/S_{55}$, $G_{23} = 1/S_{44}$, $\nu_{12} = -S_{12}/S_{11}$,
$\nu_{13} = -S_{13}/S_{11}$, $\nu_{23} = -S_{23}/S_{22}$. For the aluminium TPMS cell `Sample_1`
of example 6 (SI units):

% source: examples/OpenSG-solid/6_get_solid_props_from_3D_SG/sample_1/Sample_1.out
```text
 The Engineering Constants (Approximated as Orthotropic)
 ----------------------------------------------------------
  E1  =     6.1641278E+009
  E2  =     6.1641815E+009
  E3  =     6.1640078E+009
  G12 =     4.5197549E+009
  G13 =     4.5196366E+009
  G23 =     4.5196402E+009
  nu12=     3.5652386E-001
  nu13=     3.5654253E-001
  nu23=     3.5653028E-001
```

The word *approximated* matters: the nine values are exact only for a law that is orthotropic in
the SG axes. The matrix, not the constants block, is the result.

## The wall plate law: `<base>_ABDG.out`

Every shell route that reduces a layup to a wall plate law (the cross-section rings for the
`refined: 1` beam and the equivalent solid, and the 3-D shell SG) writes one $8\times8$ ABDG
stiffness and its compliance per section. The file header states the row convention, and each
block is preceded by its section and full layup:

% source: examples/OpenSG_shell/1_get_beam_props_from_shell_cross_section/iea_s10_shell_ABDG.out
```text
 OpenSG msg-shell wall plate laws, one block per section
 rows [eps11 eps22 2eps12 | K11 K22 K12+K21 | 2g13 2g23]

 section 0: layup_0  layup [['gelcoat', 0.0005, 0.0], ['glass_triax', 0.00300294, 0.0], ['glass_uniax', 0.02609467, 0.0], ['glass_triax', 0.00300294, 0.0]]
 The Effective Reissner-Mindlin Plate Stiffness Matrix
```

The ABD block is referenced to the surface the run chose (the outer mold line, or the
mid-surface with `--center`), so $B$ depends on that choice. The $G$ block is always the
least-squares result of the mechanics of structure genome (MSG) plate solution, never a
shear-correction estimate. The driver `beam_homo_shell.py` of example 1 also caches these laws as
`abd/iea_s10_shell_abd.yaml` for reuse by the recovery and by shell buckling.

## Figures and gmsh meshes

% source: src/opensg/cli.py; src/opensg_solid/cli.py; src/opensg_shell/cli.py
| file | written by | content |
|---|---|---|
| `<base>_mesh.png` | a solid run when the PNG is missing or older than the yaml; any run with `--mesh` | the mesh, elements coloured by material (solid) or by section (shell, with the laminate band at the active reference) |
| `<base>.msh` | `--mesh` | the gmsh 2.2 twin: material ids (solid) or one physical group per section (shell) as tags; an existing solid `.msh` is kept |
| `<name>_orient.png` | the example drivers, through `opensg_shell.auto_emit` | the material orientation check, $e_2$ blue and $e_3$ green at element centroids |

No homogenization entry point draws the orientation figure by itself; call `auto_emit` from
your own driver, as `examples/OpenSG_shell/1_get_beam_props_from_shell_cross_section/beam_homo_shell.py`
does ({numref}`fig-out-orient`).

```{figure} ../_static/shell_xsec_orient.png
:width: 80%
:name: fig-out-orient

Material orientation of the IEA-22 cross-section at $r/R = 0.20$: $e_2$ in blue, tangent to
every wall, and $e_3$ in green, pointing inward on the skin.
```

## Dehomogenization files

A `D` run (see {doc}`recovery`) writes, besides the `.out`, the files below. It ends with
`Local field files are computed and stored.` and one `Time taken` line.

**msg-solid**, prefix `<base>_dehom`:

% source: src/opensg_solid/sg_dehom.py (export_gauss)
| file | content |
|---|---|
| `<base>_dehom.txt` | one row per Gauss point: `GP_Index`, `Coord_X` (and `Coord_Y`, `Coord_Z` up to the SG dimension), `Eps_xx Eps_yy Eps_zz Eps_yz Eps_xz Eps_xy`, `Sig_xx Sig_yy Sig_zz Sig_yz Sig_xz Sig_xy` |
| `<base>_dehom.vtk` | ASCII `UNSTRUCTURED_GRID` of the exploded SG mesh: each element with its own corner nodes, carrying the mean of its Gauss values (`Eps_*`, `Sig_*`, `mat_id`) as `CELL_DATA`, so the contour stays discontinuous across materials |
| `<base>_dehom.SM` | stress: `x y z`, then the components `11 22 33 12 13 23` |
| `<base>_dehom.EM` | strain, the same layout |
| `<base>_dehom.U` | displacement `U1 U2 U3` |
| `<base>_dehom_elemental.SM`, `_elemental.U` | the per-element mean at the element centroid, for comparison with a centroidal 3-D finite element stress |

Each `.SM`, `.EM` and `.U` file opens with two comment lines (the title, then the point set, the
frame, and `z = 0 is the reference surface`) and a column header. With `--global` the stress and
strain files are named `_global.SM`, `_global.EM` and `_elemental_global.SM`; `.U` keeps its
name, since it is never rotated.

**msg-shell**, prefix `<base>_dehom`:

% source: src/opensg_shell/cli.py (dehom_write)
| file | content |
|---|---|
| `<base>_dehom.txt` | one row per recovery point: `elem`, `zeta` (0 at the outer mold line, 1 at the inner), `y2`, `y3`, `z` from the reference surface, `S11 S22 S33 S23 S13 S12` (Pa), `E11 E22 E33 2E23 2E13 2E12`, `u1 u2 u3` (m), and, unless `junction: off`, the two extra columns `jflag`, `jdist` |
| `<base>_dehom.vtk` | the exploded recovery mesh: one quad per element and depth band, each carrying its own point's stress as `CELL_DATA`, never node-averaged, plus `jflag` and `jdist` unless `junction: off` |
| `<base>_dehom.junc` | the junction sidecar: every junction's coordinates, topology, overlap area and per-wall ownership, headed by the census; not written with `junction: off` |

**Plate field** (`rm_dehom_ff.py`), named after the station table:

% source: docs/tutorials/msg_solid_plate.md
| file | content |
|---|---|
| `<ff stem>_ff.SM` | 3-D stress: `x y z`, then `S11 S22 S33 S12 S13 S23` in Pa |
| `<ff stem>_ff.EM` | 3-D strain, the same layout, `E11` to `E23` |
| `<ff stem>_ff.U` | 3-D displacement `U1 U2 U3` in m |
| `<ff stem>_ff_gauss.vtk` | ASCII `STRUCTURED_GRID`, written only when the stations form a lattice (`DIMENSIONS 40 40 45`, 72 000 points for the shipped table) |

The single-station driver `rm_dehom.py` writes `1dsg_dehom.SM`, `.EM`, `.U` and `.vtk` in the
same layout.

## See also

- {doc}`../theory/constitutive` for the derivation of every law and the two recovery chains.
- {doc}`../theory/rules/output_format_and_timing` for the rule that fixes this format.
- {doc}`converters` for the files the converters write.
