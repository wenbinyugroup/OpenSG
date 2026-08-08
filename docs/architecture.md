# Architecture

OpenSG-2.0 is two engines over one theory, both driven the same way: **a yaml
(or SwiftComp `.sc`) in, a timed SwiftComp-layout `.out` back**. The core
packages contain no `main` blocks and no directory assumptions — every path
decision lives in the example scripts, which call one entry function on the
input file.

| layer | package | role |
|---|---|---|
| user | `examples/` | yaml in → `.out` out; owns all paths |
| msg-shell | `src/opensg_shell` | RM shell SGs: ring, aperiodic segment, 3-D shell cell |
| msg-solid | `src/opensg_solid` | solid SGs: plate ABD, beam KKT, 3-D solid law |
| FE core | `src/fe_jax` | JAX FE architecture: basis/quadrature, periodic assembly map, jitted element kernels, EBE Chebyshev-CG, sparse direct |

## msg-shell: 1-D ring → Timoshenko 6×6

```{mermaid}
flowchart TD
    Y1["1-D shell yaml<br/>(nodes, elements, layups, materials)"] --> LR["oml_ring.load_ring_ref<br/>contour arrays + ABD/G per section"]
    LR --> MB["solve_segment_jax._material_by_section<br/>ABD 6x6 (chosen reference) + G 2x2"]
    MB --> GMSG["opensg_solid rm_plate_msg<br/>MSG (Yu-2002 LS) wall G"]
    LR --> K22["segment_element.compute_k22<br/>hoop curvature per edge"]
    GMSG --> RI["run_ring_indep.ring_indep<br/>6-DOF ring, drilling Lagrange, MITC g23"]
    K22 --> RI
    RI --> ASM["segment_indep.assemble_segment_indep<br/>strip energy blocks Dhh..Dle (/h)"]
    RI --> CON["segment_indep.assemble_constraint<br/>drilling rows Gc/Gl/Ge"]
    ASM --> LU["one LU of the KKT<br/>V0 = LU \ (-Dhe)"]
    CON --> LU
    LU --> V1["fe_jax.msg_solver.prepare_v1_rhs<br/>V1 = LU \ b(V0)"]
    V1 --> FIN["finalize_v1_and_compute_deff<br/>generalized-Timoshenko 6x6"]
    FIN --> OUT1[["&lt;yaml&gt;_Timo.out + &lt;yaml&gt;_ABDG.out<br/>(.K layout, timed)"]]
```

## msg-shell: 3-D segment yaml → aperiodic Timoshenko 6×6

```{mermaid}
flowchart TD
    Y2["3-D shell segment yaml"] --> EX["boundary_from_yaml.extract<br/>free edges -> two end rings + node2seg"]
    EX --> BY[["boundary 1-D yamls (L, R)<br/>+ orientation PNGs"]]
    EX --> RL["ring_indep (side L)<br/>C6L, V0, V1"]
    EX --> RR["ring_indep (side R)<br/>C6R, V0, V1"]
    EX --> SA["assemble_segment_indep + constraint<br/>full 3-D quad mesh, drilling saddle point"]
    RL --> DIR["segment_taper._boundary_dirichlet<br/>ring V0/V1 -> Dirichlet on end nodes"]
    RR --> DIR
    SA --> SLV["dirichlet_factor: one interior LU<br/>V0 then V1 (shared factorization)"]
    DIR --> SLV
    SLV --> FIN2["finalize (energy / L)<br/>segment S6"]
    FIN2 --> OUT2[["&lt;yaml&gt;_Timo.out<br/>prismatic gate: S6 == ring 6x6"]]
```

## msg-shell: 3-D shell SG → equivalent solid C3D

```{mermaid}
flowchart TD
    Y3["3-D shell SG yaml (TPMS-class)"] --> SG3["shell_sg3d.shell_sg3d"]
    SG3 --> PM["periodic_multiscale map<br/>faces+edges+corners tied (default)"]
    SG3 --> AP["aperiodic: w = 0 Dirichlet<br/>on bounding-box nodes (on request)"]
    PM --> KAS["batched Gamma_h / Gamma_e assembly<br/>sparse K, Dhe, Dee"]
    AP --> KAS
    KAS --> SPLU["scipy splu, one factorization<br/>V0 for all 6 macro strains"]
    SPLU --> DEFF["Deff = Dee + V0^T Dhe<br/>C3D = Deff / omega"]
    DEFF --> OUT3[["&lt;yaml&gt;_C3D.out<br/>(per unit cell; digit gate vs SwiftComp)"]]
```

## msg-solid over the fe_jax core

One driver, `sg_homo.plate_homo_2d(sc_path, n_model=…)`, serves all three
macro models; the fe_jax layer supplies the FE machinery:

```{mermaid}
flowchart TD
    SC[".sc / yaml"] --> LOAD["sg_mesh.load_sg_input<br/>dim, nodes, cells, materials"]
    LOAD --> BQ["fe_jax.basis_quadrature<br/>quadrature + basis tables"]
    LOAD --> MJ["fe_jax.setup.mesh_to_jax<br/>x_end (E,N,d)"]
    LOAD --> PMAP["fe_jax mesh_to_periodic_sparse_assembly_map<br/>master connectivity + dof map"]
    LOAD --> CMAT["sg_materials.get_heterogeneous_C_matrix<br/>C_ess (E,6,6), per-element frames"]
    BQ --> MODEL{n_model}
    MJ --> MODEL
    PMAP --> MODEL
    CMAT --> MODEL
    MODEL -->|"1: beam"| KKT["_beam_homo_kkt<br/>EB 4-mode KKT + l-chain V1s"]
    MODEL -->|"2: plate"| PL["_homo_direct: jitted kernels -> CSR<br/>pardiso direct (CPU)"]
    MODEL -->|"3: solid"| PL
    PL -.->|'solver=cg'| CG["fe_jax EBE Chebyshev-CG<br/>fully jitted, device-resident (GPU path)"]
    KKT --> OUTB[["&lt;base&gt;.out — Timoshenko<br/>stiffness + compliance"]]
    PL --> OUTP[["&lt;base&gt;.out — Classical /<br/>Reissner-Mindlin Plate, or 3-D solid + constants"]]
    CG --> OUTP
```

## GPU readiness

The element kernels (`calculate_RHS_and_Ke_batch_periodic`, jitted `jacfwd`
stiffness), the EBE operator, block/Chebyshev preconditioners and the CG
solver are pure JAX and run device-resident: on a GPU host,
`plate_homo_2d(..., solver="cg")` is the GPU path (documented digit-safe —
`C_eff` is stationary in `V0`, so solver tolerance enters at second order).
The default `solver="direct"` hands the assembled CSR to pardiso/SuperLU on
CPU — fastest on workstations, and the only stage that leaves the device.
The msg-shell routes assemble batched NumPy einsum contractions (two-step
`C·B` then `Bᵀ(CB)` form) into sparse SuperLU solves; their operator batches
are `jnp`-portable, with the factorization as the remaining CPU stage.
