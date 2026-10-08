micro1D.sc / micro2D.sc / micro3D.sc -- the three example input decks
AnalySwift distributes beside SwiftComp.exe (SwiftComp 2.1, Windows
package; copied verbatim 2026-09-07 from C:\SwiftComp\Windows on the
ANSHIKA laptop).  Kept here as EVIDENCE for io.sg_input.write_sc, not as
test inputs to run:

  * the material aux pair is `temperature density` in that order -- each
    deck carries `100 0.5   # temperature density`, BOTH numbers nonzero
    and the order named in the vendor's own comment (SCManual 8.2 says the
    same: "T_i ... is the temperature, rho is the density");
  * micro3D.sc shows the 20-slot hex20 record (corners 1-8, midsides
    9-20) and micro2D.sc the 9-slot quad8 record (corners 1-4, midsides
    5-8, slot 9 = 0).

tet10_gate/ -- the live SwiftComp 2.1 run (2026-09-07) that pinned the
tet10 slot layout write_sc emits; see its README.
