# Rule — solid properties are always computed on a PERIODIC SG

**Applies to:** every route that homogenizes a structure gene to an equivalent
3-D solid (`n_model = 3`) — the solid engines (`opensg_solid`, and the original
JAX_BICGoptimize single script) and **msg_shell** (`opensg_shell.solid_props`,
`build_solid_bundle` / `ring_solid`).

## The rule

Periodicity is **always** part of getting solid properties. It is not optional
and not a per-case choice:

| SG | what is tied |
|---|---|
| 2-D SG | opposite face **edges**, and the **corners** |
| 3-D SG | all opposite **faces**, **edges**, and **corners** |

The tie is applied exactly as `fe_jax/periodic_multiscale.py` does it for the
solid architecture — the same function, the same switch, mirrored for the shell
in `opensg_shell/periodic_multiscale.py` with 6 DOFs per node instead of 3:

```
periodic_map(points, n_model, atol, ndof_per_node)
  n_sg = 2, n_model = 3   ->  pair x and y
  n_sg = 3, n_model = 3   ->  pair x, y and z
  for _ in range(n_sg): dof_map = dof_map[dof_map]   # edge + corner chains
```

Pairing is by bounding-box faces (`max` face onto `min` face, shifted by the box
length, matched with `cdist`), then the repeated `dof_map[dof_map]` resolves
nodes that belong to two or three faces at once — that is what makes the edges
and corners periodic, not just the face interiors.

Periodicity then rides in the **local→global sparse assembly map**: the element
connectivity is re-pointed at the master nodes, so every scatter/gather lands on
the master DOF. No master–slave transformation matrix is formed and no
constraint rows are added.

## Why it is mandatory

A free (non-periodic) SG cannot produce a meaningful equivalent solid. The
fluctuation fields

```
w1 = -2*G12*y2 - 2*G13*y3 ,  w2 = -G22*y2 - G23*y3 ,  w3 = -G33*y3 - G23*y2
```

are affine, admissible on any domain, and cancel five of the six macro strains
pointwise at **zero energy**. Only `Gamma_11` survives (it has no fluctuation
gradient available in a prismatic SG), so `C3D` comes out **rank one** — C11
only, with every other entry at roundoff. Periodicity is precisely what forbids
those affine fields, because they are not periodic.

Verified numerically: a fully filled square cell with free faces returns
`C11 = E` exactly (uniaxial stress) and zeros elsewhere; the **same mesh** with
periodic faces returns the material's own isotropic stiffness exactly
(`lambda + 2*mu`, `lambda`, `mu`).

## Defaults in code

`ring_solid(...)` and `build_solid_bundle(...)` take `periodic=True` by default.
`periodic=False` is retained for diagnostics only (to demonstrate the rank-one
theorem); do not use it for reported properties.

## Choosing the cell — the one thing still on the user

Periodicity ties whatever cell you hand it, so the cell must be a genuine unit
cell. Walls that lie **on** the cell boundary are shared with the neighbour and
get integrated twice (and, in a solid mesh, bonded face-to-face into a wall of
twice the thickness). Put the walls **through** the cell instead — one wall of
each family, interior to the cell — so each is owned by exactly one cell.

Related: `Rules/orientation_e1_out_of_plane.md`.
