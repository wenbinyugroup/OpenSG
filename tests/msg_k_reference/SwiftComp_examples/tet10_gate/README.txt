tet10_gate -- the live SwiftComp 2.1 run (2026-09-07, ANSHIKA laptop,
C:\SwiftComp\Windows\SwiftComp.exe <deck> 3D H) that pinned the tet10
element record io.sg_input.write_sc emits, and the `T rho` aux-pair order.

build_gate.py      builds two tet10 unit cubes (2x2x2 hex cells, Kuhn
                   6-tet split, midsides on the half-lattice; 125 nodes,
                   48 elements): `iso` = one material (E 70000, nu 0.30,
                   rho 2700) and `bilayer` = that below z = 0.5, a soft
                   one (E 7000, nu 0.35, rho 1200) above.  Writes the
                   OpenSG yaml of each (cells in GMSH tet10 order) and
                   THREE `.sc` decks per cube that differ ONLY in the
                   element record:
                     A  corners 1-4, slot 5 = 0, midsides 6-11 on edges
                        (12, 23, 13, 14, 24, 34), zeros to 20
                     B  same slots, midsides in GMSH order (…, 34, 24)
                     C  contiguous fill 1-10, slot 5 /= 0
                   then runs opensg_solid on both yamls (cube_*.out).
compare_gate.py    scores each `.sc.k` against the OpenSG `.out` and, for
                   `iso`, the exact isotropic C.
cube_*_?.sc        the six decks;  cube_*_?.sc.k  what SwiftComp wrote;
cube_*_?.swiftcomp.log  the console of each run.

RESULT
  A  finished successfully on both cubes.  iso: C == exact to machine
     precision and == OpenSG; `Effective Density = 2.7000000E+003` (the
     deck's `0 2700` aux line read back as `T rho`).  bilayer: == OpenSG's
     own tet10 solve; effective density 1950 = the volume average.
  B, C  "determinant of Jacobian matrix less than 0 for element 1" and a
     ZERO-BYTE `.k` -- SwiftComp keeps going after that message and
     reports success-looking lines, so the empty `.k` is the tell.

So write_sc's _slots emits A (the yaml's last two midsides swapped, the
inverse of the swap sc_to_yaml.read_sc applies), and _guard_sc_aux writes
the density in the second aux slot.  test_sg_input_writer.py::
test_sc_tet10_slot_layout re-checks these artifacts.
