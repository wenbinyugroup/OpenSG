# Plate_deck_analysis -- the 13-step plate benchmark pipeline

Deck generation, Abaqus runs, station extraction, dehomogenization and the
3-curve comparison figures for a pressure-loaded plate built from one 2-D/3-D
SG.  Distilled 2026-09-03 from the validated TPMS pressure_case study
(`M:\Abaqus\pressure_case\fea_driven`, see its README for the full diagnosis).

## The 13 steps

| step | module | does |
|---|---|---|
| 01 | `step01_make_decks` | plate deck from the SG yaml + law (S8R, subdiv = 3 DEFAULT, SS-1, cellmap) [level 2] |
| 02 | `step02_levels` | the level-1 / level-2 contract; cellmap readers + integrity checks |
| 03 | `make_load_column_deck` (step03 `inject` = legacy) | the plate deck WITH the load term: edge traction `-F_N` as consistent `*Cload`; `F_N` from Yu 2003 Eq. 47, F = V0^T L, no dehom run |
| 04 | `step04_solid_inp` | the 3-D FEA reference deck (tiled cell, C3D10, same SS-1, pressure on the top material faces) |
| 05 | `step05_dump_rpts` | odb -> .rpt dumpers for BOTH runs; standalone, run under `abaqus python` |
| 06 | `step06_fea_path_dat` | 3-D FEA rpt + path.coords (+ cell offset) -> the FEA path .dat (unified layout) |
| 07 | `step07_elem_grid_derivs` | strain + ALL first/second strain derivatives at a station from the ELEMENT grid (subdiv sub-element centroids, h = pitch/subdiv): 5-point central differences BY DEFAULT, 3-point fallback near an edge |
| 08 | `step08_station_ff` | plate rpt + cell (i, j) -> the station .ff, with the measured SE/SK/SM conventions of plate_rm and step07 derivatives |
| 09 | `step09_run_dehom` | stage a work dir and run `opensg_solid <yaml> D --global` (shear-refined) |
| 10 | `step10_opensg_path_dat` | elemental .SM/.U + path.coords -> the OpenSG dehom path .dat |
| 11 | `step11_classical` | the classical twin: `refined: 0` yaml, FF-only .ff from the BARE deck's station (classical has no F), then steps 9-10 |
| 12 | `step12_plots` | 6 stress + 3 displacement figures per path, three curves |
| 13 | `step13_run_pipeline` | the orchestrator: steps 1-12 as resumable stages; TWO plate jobs (bare + withF), per-cell 3-D reports with offsets, Eq. 47 F |

## Level 1 / level 2

- **Level 1** = one SG cell: the object `opensg_solid` analyses, and the
  footprint of one station (`--cell DI DJ` = signed offsets from the
  plate-centre cell).
- **Level 2** = the subdiv x subdiv sub-elements of that cell in the plate
  deck (subdiv = 3 default).  The plate .rpt is centroidal PER SUB-ELEMENT,
  so the element grid samples the macro field subdiv times finer than the
  cell grid.  The cellmap CSV ties the two levels together; step02 checks
  the contract.

## The unified path .dat layout (steps 06, 10, 12)

Whitespace table, one row per path point, comment lines start with `#`:

    s x y z S11 S22 S33 S12 S13 S23 U1 U2 U3

stress in Pa, displacement in m, `s` non-dimensional in [0, 1] along the
path.  step06 writes it from the 3-D FEA, step10 from either dehom, step12
reads all three.

## Defaults that are decisions

- **subdiv = 3** sub-elements per cell per direction: subdiv 3 vs 5 moved
  the second strain derivatives 0.78 % and the fourth 1.88 % on the TPMS
  benchmark -- refinement is not the lever.
- **5-point central differences on the ELEMENT grid** (step07), 3-point
  fallback with one sub-element of margin: every cell gets a real stencil,
  where cell-grid differencing left only the centre cell with the 5-point
  rule.
- **Third/fourth strain derivatives are EXCLUDED**: measured dead on the
  TPMS benchmark (the eps,ab conversion moved ring sigma22 by -0.12 kPa,
  away from the 3-D FEA).  `plate_rm/ff_high_derivs.py` remains available
  as a diagnostic only.

## The load-column physics (corrected 2026-09-04)

A pressure-loaded SG yields the plate law {N; M} = [ABD]{e; k} + F, with F
the load-related term of Yu, Hodges & Volovoi, Comput. Struct. 81 (2003)
439-454, Eq. 61 (their Eq. 47 gives F; Eq. 45 defines the load column
V_{1L}).  `*Shell General Section` implements the law WITHOUT F.

- **F comes from Eq. 47, not from integration.**  For a load that does not
  vary in plane, F = V0^T L exactly; `sg_homo` stores `F_unit = V0^T L` per
  unit face pressure (plus the gradient blocks `G1_unit`, `G2_unit` for
  in-plane-varying loads) and `sg_dehom.load_column_F` scales it.
  `make_load_column_deck` builds the withF deck from it; the old zero-strain
  dehom + stress-integration route is gone.  Gate on the TPMS cell: F_N
  identical to machine precision, F_M to 1.7e-4.
- **The membrane term is restored as an edge traction, and that is exact.**
  Integrating the F_N term of Eq. 61 by parts over the reference surface
  gives oint du_n F_nn ds (the tangential part vanishes because SS-1 holds
  u_t), so Yu's problem == the F-less deck + outward edge traction -F_nn on
  the free normal DOF.  Self-equilibration is a CONSEQUENCE (F constant),
  and a check: the four edges sum to zero and RF3 is unchanged.  Validated:
  eps 2.577e-8 vs 3-D 2.540e-8 (1.5 %); without it eps = 0 (100 % wrong).
  Derivation: `M:\Abaqus\pressure_case\FINAL\docs\edge_load_equivalence.pdf`.
- **The moment term is NOT applied, and the reason is empirical.**  The same
  integration by parts makes an edge moment -F_M on the free bending
  rotation the exact equivalent of F_M (`--with-moment`), and it is correct
  Yu -- but on the TPMS plate it moves the centre-cell total moment from
  0.5 % to 2.8 % off the 3-D value, and the 3-D cell's own M - D.kappa is
  -129..-187 N, never the -91 N the SG assigns to F_M.  At a/h = 5
  (U* = 0.155) the RM moment law with this F_M does not describe the cell;
  omitting F_M matches better by accident of that limitation.  Do NOT
  correct the station curvature by D^-1(M - F_M) either: it took ring
  sigma22 from 1.3 % to 6.0 % error.  The deck's SM is the STRAIN part
  D.kappa; the 3-D cell integral is the TOTAL D.kappa + F_M -- comparing
  them directly is comparing two different quantities (the "6 % moment
  gap" was exactly that; corrected, the interior agrees to 0.1-0.5 %).
- **The classical model has no F at all** (Yu Eq. 33), so the classical
  twin is driven by the BARE deck's station, never the withF one.  step13
  runs both plate jobs for that reason.
- **VAM is an interior solution** (Yu p. 443-444): cells in the ring nearest
  the plate boundary degrade (moment 3-7 % vs 0.1-0.5 % interior) and are
  excluded from validation, not "fixed".

## Plot conventions (step12)

Nine figures per (cell, path): S11 S22 S33 S12 S13 S23 in kPa and U1 U2 U3
in m, NPT = 25 evenly spaced points of the s-window per curve, x axis = s,
grid light, legend outside right (frameless), no dotted lines, files
`path<N>_<comp>.png` at dpi 200.  The only title allowed is "Path N".

| curve | color | lw | marker | label |
|---|---|---|---|---|
| 3-D FEA | tab:blue | 2.8 | o (filled) | 3-D FEA (Abaqus) |
| shear-refined | tab:orange | 1.3 | s (open) | Shear-refined plate model (OpenSG) |
| classical | tab:green | 1.3 | ^ (open) | Classical plate model (OpenSG) |

## Orchestrator layout (step13)

Fixed tree under `--out-root`:

    decks/     plate.inp, plate_withF.inp, solid.inp, cellmap.csv
    rpts/      plate.rpt, solid.rpt (+ odbs; Abaqus runs here)
    stations/  <tag>.ff              (tag = i0j0, i1Nj2, ...)
    dehom/<tag>/, dehom_classical/<tag>/
    dats/      <tag>_{fea|refined|classical}_path<N>.dat
    plots/<tag>/path<N>_<comp>.png
    pipeline_state.json              (stage -> done; --force reruns)

The pipeline runs the plate Abaqus job itself (seconds) when
`--run-abaqus plate` (the default) or `solid`; the expensive solid job is
only run with `--run-abaqus solid`, otherwise the exact commands are
printed and the pipeline stops -- rerun the same command line to resume
once `rpts/solid.rpt` exists.

## Worked example (the TPMS pressure_case study)

Schwarz-P rho = 0.3, 5 x 5 cells, a/h = 5, top-face pressure
q = 7946.1 Pa, SS-1.  On the server (msg, opensg env):

    cd /home/msg/a/bagla0/Abaqus/pressure_case
    python -m opensg_solid.helper.Plate_deck_analysis.step13_run_pipeline \
        --sg sg/bare_rho0.3.yaml --law sg/bare_rho0.3.out \
        --out-root pipeline_run --nx 5 --ny 5 --q 7946.1 \
        --paths path/path_circumferential.coords path/path_thickness.coords \
        --cells "0 0" --measure-fn --run-abaqus plate

The run stops at stage 5 after the plate solve, printing the solid job
commands.  Run those yourself (Abaqus in the Remote Desktop session), then
rerun the identical command line: stages 1-4 skip, 5 resumes from the
.rpt, 6-12 finish.  Off-centre stations afterwards, reusing every deck and
rpt:

    python -m opensg_solid.helper.Plate_deck_analysis.step13_run_pipeline \
        --sg sg/bare_rho0.3.yaml --law sg/bare_rho0.3.out \
        --out-root pipeline_run --nx 5 --ny 5 --q 7946.1 \
        --paths path/path_circumferential.coords path/path_thickness.coords \
        --cells "1 1,1 2" --measure-fn --stages 6-12 --force

Single steps run standalone the same way, e.g. the junction-window figures
of the study (path 1, |s - 0.5| <= 0.10):

    python -m opensg_solid.helper.Plate_deck_analysis.step12_plots \
        --fea dats/i0j0_fea_path1.dat --refined dats/i0j0_refined_path1.dat \
        --classical dats/i0j0_classical_path1.dat --path-label 1 \
        --out-dir plots/i0j0_junction --window "0.40 0.60"

## Validation (TPMS Schwarz-P rho = 0.3, AR5, q = 7946.1 Pa)

- Centre-cell ring sigma22: the shipped (F-less) pipeline had 9.2 % max
  error; the withF deck brings it to 1.3 %, and driving the recovery with
  the FEA cell's own load-column-consistent forces reaches 0.9 % -- the
  macro state, not the recovery, was the defect.
- Off-centre stations keep a ~13-16 % residual that is RECOVERY-side (an
  in-cell redistribution the V2 family does not capture), not a
  macro-state error: the resultant identities close there too.
- The two shipped-pipeline errors that used to cancel: the plate M is 6 %
  low while the missing F_M subtraction is +5.5 %, leaving the shipped
  station curvature only 0.51 % off -- which is why the moment defect went
  unnoticed and why the membrane term (93 % of the repair) is the one that
  matters.
