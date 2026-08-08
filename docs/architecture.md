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

## What a homogenization actually does

Every pipeline below is the same variational statement, specialised. A
structure gene (SG) is the smallest piece of the structure that carries the
heterogeneity: a laminate through the thickness, a cross-section contour, a
unit cell. Its displacement field is split into a macro part driven by the
macro strains $\bar\varepsilon$ and a fluctuation (warping) field $w$:

$$\varepsilon = \Gamma_e\,\bar\varepsilon + \Gamma_h\,w .$$

Substituting into the strain energy gives the quadratic form the code
assembles, in the block names used throughout the source:

$$2U = \bar\varepsilon^T D_{ee}\,\bar\varepsilon
      + 2\,w^T D_{he}\,\bar\varepsilon + w^T D_{hh}\,w .$$

Minimising over $w$ (subject to the SG's boundary treatment — periodicity,
Dirichlet boundary data, or a rigid-body constraint) gives
$D_{hh}V_0 = -D_{he}$, and the effective law follows as

$$D_{\rm eff} = D_{ee} + V_0^T D_{he},
\qquad C = D_{\rm eff}/\omega ,$$

with $\omega$ the SG measure that stays in the model (thickness, area,
volume, perimeter). That is the **zeroth-order** answer: the plate ABD, the
Euler–Bernoulli 4×4, the 3-D solid law. Recovering transverse shear —
the Reissner–Mindlin $G$ block, the Timoshenko 6×6 — takes a **first-order**
step: a second solve for $V_1$ against a right-hand side built from $V_0$
(`prepare_v1_rhs`), then an energy transformation
(`finalize_v1_and_compute_deff`). Both solves share one factorization.

Running the chain backwards is **dehomogenization**: given macro strains or
forces, rebuild $w$ from the stored $V_0$/$V_1$ and evaluate pointwise 3-D
stress inside the SG.

## msg-shell: 1-D ring → Timoshenko 6×6

```{mermaid}
flowchart TD
    Y1["1-D shell yaml<br/>(nodes, elements, layups, materials)"] --> LR["sg_mesh.load_ring_ref<br/>contour arrays + ABD/G per section"]
    LR --> MB["sg_materials._material_by_section<br/>ABD 6x6 (chosen reference) + G 2x2"]
    MB --> GMSG["opensg_solid rm_plate_msg<br/>MSG (Yu-2002 LS) wall G"]
    LR --> K22["sg_assembly.compute_k22<br/>hoop curvature per edge"]
    GMSG --> RI["sg_homo.ring_indep<br/>6-DOF ring, drilling Lagrange, MITC g23"]
    K22 --> RI
    RI --> ASM["sg_assembly.assemble_segment_indep<br/>strip energy blocks Dhh..Dle (/h)"]
    RI --> CON["sg_assembly.assemble_constraint<br/>drilling rows Gc/Gl/Ge"]
    ASM --> LU["one LU of the KKT<br/>V0 = LU \ (-Dhe)"]
    CON --> LU
    LU --> V1["fe_jax.msg_solver.prepare_v1_rhs<br/>V1 = LU \ b(V0)"]
    V1 --> FIN["finalize_v1_and_compute_deff<br/>generalized-Timoshenko 6x6"]
    FIN --> OUT1[["&lt;yaml&gt;_Timo.out + &lt;yaml&gt;_ABDG.out<br/>(.K layout, timed)"]]
```

## msg-shell: 3-D segment yaml → aperiodic Timoshenko 6×6

```{mermaid}
flowchart TD
    Y2["3-D shell segment yaml"] --> EX["sg_mesh.extract<br/>free edges -> two end rings + node2seg"]
    EX --> BY[["boundary 1-D yamls (L, R)<br/>+ orientation PNGs"]]
    EX --> RL["ring_indep (side L)<br/>C6L, V0, V1"]
    EX --> RR["ring_indep (side R)<br/>C6R, V0, V1"]
    EX --> SA["assemble_segment_indep + constraint<br/>full 3-D quad mesh, drilling saddle point"]
    RL --> DIR["sg_homo._boundary_dirichlet<br/>ring V0/V1 -> Dirichlet on end nodes"]
    RR --> DIR
    SA --> SLV["dirichlet_factor: one interior LU<br/>V0 then V1 (shared factorization)"]
    DIR --> SLV
    SLV --> FIN2["finalize (energy / L)<br/>segment S6"]
    FIN2 --> OUT2[["&lt;yaml&gt;_Timo.out<br/>prismatic gate: S6 == ring 6x6"]]
```

What the segment pipeline adds over the ring: the ring is periodic along the
beam axis, so one cross-section suffices. A tapered segment is not, so its
two end cross-sections are extracted from the 3-D mesh, solved as rings in
their own right, and their warping fields are imposed as Dirichlet data on
the segment's end nodes — the boundary condition that replaces periodicity.
The figure below shows a BAR-URC blade segment with its per-element material
frames, the surface the segment solve runs on:

![BAR-URC tapered shell segment with element material frames](_static/aperiodic_segment_bar_urc.png)

The two end rings are extracted topologically — a mesh edge used by exactly
one quad is a free edge, and the connected components of the free-edge graph
are the end cross-sections. Each is written as a standalone 1-D SG yaml and
solved on its own:

![Left boundary ring extracted from the segment](_static/aperiodic_ring_L.png)
![Right boundary ring extracted from the segment](_static/aperiodic_ring_R.png)

## msg-shell: 3-D shell SG → equivalent solid C3D

```{mermaid}
flowchart TD
    Y3["3-D shell SG yaml (TPMS-class)"] --> SG3["sg_homo.shell_sg3d"]
    SG3 --> PM["sg_periodicity map<br/>faces+edges+corners tied (default)"]
    SG3 --> AP["aperiodic: w = 0 Dirichlet<br/>on bounding-box nodes (on request)"]
    PM --> KAS["batched Gamma_h / Gamma_e assembly<br/>sparse K, Dhe, Dee"]
    AP --> KAS
    KAS --> SPLU["scipy splu, one factorization<br/>V0 for all 6 macro strains"]
    SPLU --> DEFF["Deff = Dee + V0^T Dhe<br/>C3D = Deff / omega"]
    DEFF --> OUT3[["&lt;yaml&gt;_C3D.out<br/>(per unit cell; digit gate vs SwiftComp)"]]
```

Because the wall law is a Reissner–Mindlin plate law, thickness is a
parameter of the material model rather than of the mesh — the same surface
mesh serves any sheet thickness, which is what makes the TPMS sweep cheap:

![Schwarz-P TPMS shell unit cell](_static/schwarz_p_mesh.png)

## msg-solid over the fe_jax core

One driver, `sg_homo.plate_homo_2d(sc_path, n_model=…)`, serves all three
macro models; the fe_jax layer supplies the FE machinery. `n_model` selects
which macro strain set $\Gamma_e$ is built for — 1 routes to the beam KKT
engine (four Euler–Bernoulli modes plus the $V_1$ chain to Timoshenko), 2
builds the plate modes $\bar\varepsilon = [\varepsilon_{11}\,\varepsilon_{22}\,
2\varepsilon_{12}\,\kappa_{11}\,\kappa_{22}\,\kappa_{12}]$, and 3 the six
solid macro strains:

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
