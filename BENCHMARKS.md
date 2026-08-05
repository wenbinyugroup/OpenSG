# opensg_solid benchmarks — RHC / 1-D SG suite

Wall times of the `plate_homo_2d` call (fresh process each run; the
~2.3 s `import` cost is excluded; input parse + mesh PNG are included).
Cold = persistent JAX compilation cache cleared; warm = second identical
run (cache populated).  Machine: msg.ecn.purdue.edu, CPU, conda env
`opensg_2_0`.  Solver = the default `"direct"` (one pypardiso
factorization for all RHS columns; the SSDM Chebyshev-CG pipeline stays
selectable with `solver="cg"`).

| case  | SG (dim, file)                | n_elements | n_nodes | #dofs | cold [s] | warm [s] |
|-------|-------------------------------|-----------:|--------:|------:|---------:|---------:|
| beam (n_model=1, Timo 6x6)  | 2-D `RHC_SW_2UC_45.yaml`      | 6640 | 4251 | 12753 | 12.6 |  9.2 |
| plate (n_model=2, ABD 6x6)  | 2-D `RHC_SW_2UC_45.yaml`      | 6640 | 4251 | 12696 | 10.8 |  7.7 |
| solid (n_model=3, C 6x6)    | 2-D `RHC_SW_2UC_45.yaml`      | 6640 | 4251 | 12273 | 11.4 |  8.1 |
| plate (n_model=2, ABD 6x6)  | 1-D `Plate_1D_SG_2UC_45.yaml` |    9 |   10 |    30 |  5.5 |  3.8 |

Notes.

- #dofs = the actual solve size: unique periodic dofs (3 per master
  node).  The beam KKT factorization additionally carries 4 Lagrange
  constraint rows.
- n_model=3 on the 2-D RHC SG is legitimate in this engine (the SG is
  the unit cell, periodic in both in-plane directions).  The bundled
  3-D `SW_2UC_45.sc` remains BLOCKED: 9 nonzero connectivity slots per
  element, not a known SwiftComp element slicing (examples 5/6 carry
  the open item).
- Before the optimization the RHC plate run took 138 s end-to-end; the
  measured breakdown put 130 s in the per-column Chebyshev-CG solves
  (assembly incl. the per-element `jacfwd` K_e: 0.8 s; XLA compile of
  the fused pipeline: ~0 s).  The direct factorization removes that
  cost; `C_eff` is stationary in the fluctuation, so the direct and
  CG(1e-6) answers agree in all reported digits (gate values
  A11 = 3.35860e+05, D11 = 9.03675e+07 unchanged).
- End-to-end example scripts (import + homo + exports): example 3
  plate 14.2 s cold / 10.5 s warm; example 7 beam 13.5 / 12.0 s;
  example 8 beam dehom 14.2 s.
