"""Plate_deck_analysis -- the complete plate-benchmark pipeline of a 2-D/3-D SG:
deck generation, Abaqus runs, station extraction, dehomogenization and the
3-curve comparison plots, as thirteen numbered steps (one module each) plus the
orchestrator.  Established 2026-09-03 from the TPMS pressure_case study
(M:\\Abaqus\\pressure_case\\fea_driven, see its README for the validation).

  step01_make_decks       plate deck from the SG yaml + law (S8R, subdiv = 3
                          DEFAULT, SS-1, cellmap)                      [level 2]
  step02_levels           the level-1 / level-2 contract: level 1 = one SG cell
                          (what `opensg_solid` analyses), level 2 = the subdiv^2
                          sub-elements of that cell in the plate deck; cellmap
                          readers + integrity checks
  step03_plate_inp_withF  the plate deck WITH the pressure-induced in-plane
                          load term: edge traction -F_N as consistent *Cload
                          (face-pressure + SS-1 case), F_N measured from the
                          SG's own load column
  step04_solid_inp        the 3-D FEA reference deck (tiled cell, C3D10,
                          same SS-1, pressure on the top material faces)
  step05_dump_rpts        odb -> .rpt dumpers for BOTH runs; standalone,
                          run under `abaqus python`
  step06_fea_path_dat     3-D FEA rpt + path.coords (+ cell offset) ->
                          fea stress .dat and displacement .dat
  step07_elem_grid_derivs strain + ALL first/second strain derivatives at a
                          station from the ELEMENT grid (subdiv sub-element
                          centroids, h = pitch/subdiv): 5-point central
                          differences BY DEFAULT, 3-point fallback near an
                          edge -- the 2026-09-03 rule, every cell gets a
                          real stencil
  step08_station_ff       plate rpt + cell (i, j) -> the station .ff, with
                          the measured SE/SK/SM conventions of plate_rm and
                          step07 derivatives
  step09_run_dehom        stage a work dir and run `opensg_solid <yaml> D
                          --global` (shear-refined)
  step10_opensg_path_dat  elemental .SM/.U + path.coords -> the OpenSG
                          dehom path .dat (stress + displacement columns)
  step11_classical        the classical twin: refined: 0 yaml, FF-only .ff,
                          then steps 9-10
  step12_plots            6 stress + 3 displacement figures per path, three
                          curves -- 3-D FEA thick blue, shear-refined ORANGE,
                          classical GREEN, all with markers, s in [0, 1]
  step13_run_pipeline     the orchestrator: steps 1-12 with resumable stages
                          (Abaqus solves can be run by the pipeline for
                          plate-size jobs or left to the user)

Third/fourth strain derivatives are NOT part of the pipeline: measured dead on
the TPMS benchmark (the eps,ab conversion moved ring sigma22 by -0.12 kPa, away
from the 3-D FEA).  `plate_rm/ff_high_derivs.py` remains available as a
diagnostic only.

Nothing here imports jax at package level; each step pulls what it needs."""
