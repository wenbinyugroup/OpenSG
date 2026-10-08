"""plate_rm -- every helper of the RM plate dehomogenization chain:

  sg_plate_station   level-1 station state (cell strain, u/theta,
                     derivative stencils, packing measurement)
  station_ff         the ff-side CLI: (di, dj) cell offset -> .ff +
                     station-location PNG
  fea_cell           the 3-D FEA-side CLI: the SAME (di, dj) offset ->
                     the cell's .SM/.U along the paths, suffixed with
                     the cell tag (_i1j1, negatives _i1Nj2N)
  abq_cell_dump      the odb->rpt stage of fea_cell (runs under
                     `abaqus python`; the only file here that touches
                     an odb)
"""
