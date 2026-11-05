# Solid-beam mesh convergence

Two linear-triangle meshes were compared: a 1.0 mm target mesh and a 0.5 mm target mesh. Values are the diagonal entries of the Timoshenko stiffness matrix about the quarter-chord reference axis.

## NACA 0012

| Term | 1.0 mm | 0.5 mm | Change (fine vs coarse) |
|---|---:|---:|---:|
| K11 (EA) [N] | 7.1842995e+08 | 7.1842076e+08 | -0.0013% |
| K22 (GA2) [N] | 63709903 | 63703272 | -0.0104% |
| K33 (GA3) [N] | 5977451.6 | 5961633.5 | -0.2646% |
| K44 (torsion) [N m²] | 589363.9 | 589015.2 | -0.0592% |
| K55 (EI2) [N m²] | 1786852.1 | 1786807.9 | -0.0025% |
| K66 (EI3) [N m²] | 29157293 | 29157129 | -0.0006% |

Maximum absolute diagonal-term change: **0.2646%**.

## NACA 4412

| Term | 1.0 mm | 0.5 mm | Change (fine vs coarse) |
|---|---:|---:|---:|
| K11 (EA) [N] | 7.1998734e+08 | 7.1998029e+08 | -0.0010% |
| K22 (GA2) [N] | 63687243 | 63681582 | -0.0089% |
| K33 (GA3) [N] | 6261328.2 | 6249606.6 | -0.1872% |
| K44 (torsion) [N m²] | 725373.08 | 725043.46 | -0.0454% |
| K55 (EI2) [N m²] | 2688035.4 | 2687999.6 | -0.0013% |
| K66 (EI3) [N m²] | 29280311 | 29280165 | -0.0005% |

Maximum absolute diagonal-term change: **0.1872%**.

