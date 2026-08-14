"""User-updatable blade SECTION: edit one station's resolved stacks.

The three-line entry point plus the per-section design move:

    import opensg
    blade = opensg.read("IEA-15-240-RWT.yaml")
    K, M = blade(4)

blade.section(st) returns the station's RESOLVED layup (the pyNuMAD
12-region stacks + webs) as an editable object registered as a live
override: the next blade(st) computes from the edit, section.reset()
returns to the definition.  Unlike scale_layer_thickness (a SPANWISE
move on the definition), a Section edit touches exactly one station.

Variables
---------
blade : opensg.read result (editable Blade, pyNuMAD dialect)
K0    : baseline Timoshenko 6x6 at station 4 (r = 0.3288)
S     : the editable Section; S.stacks {region: [[mat, t, ang], ...]},
        S.webs [per-web ply lists]
K1    : spar caps x1.3 at THIS station only -> EI2 (flap) rises
K2    : after S.reset() -- must reproduce K0 exactly

Run (from this folder):  python blade_section_edit_demo.py
"""
import numpy as np

import opensg

LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]

blade = opensg.read("IEA-15-240-RWT.yaml")
K0, M0 = blade(4)

S = blade.section(4)
print("editable regions:", ", ".join(nm for nm in S.stacks if S.stacks[nm]))
print("webs:", len(S.webs))
print("LP_SPAR stack (outer->inner):")
for m, t, a in S.stacks["LP_SPAR"]:
    print("   %-22s t=%.5f m  ang=%g" % (m, t, a))

S.scale_thickness(1.3, region="LP_SPAR")
S.scale_thickness(1.3, region="HP_SPAR")
K1, M1 = blade(4)

S.reset()
K2, M2 = blade(4)

print("\nstation 4 diagonal, baseline vs spar caps x1.3 (this station only):")
for i in range(6):
    print("  %-4s %14.6e %14.6e   %+7.2f %%"
          % (LBL[i], K0[i, i], K1[i, i],
             100.0 * (K1[i, i] - K0[i, i]) / K0[i, i]))
print("\nreset restores the baseline exactly:",
      bool(np.allclose(K0, K2, rtol=1e-12, atol=0.0)))
