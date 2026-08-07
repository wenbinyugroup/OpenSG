# Blocked: 3-D solid SG examples (former examples 5 and 6)

Retired from the active example list until a runnable 3-D structure gene
exists.  The bundled `SW_2UC_45.sc` carries **9 nonzero connectivity slots
per element**, which is not tet4 (4), hex8 (8) or tet10 (10) under any
known SwiftComp slot convention, so `helper/sc_to_yaml.py` refuses it
rather than emit phantom nodes.

The engine path itself is complete and unblocked: `sg_homo.plate_homo_2d`
takes `n_model=3` and the 3-D branches are exercised by the shared
kernels.  To revive these, supply either
  * a 3-D `.sc` whose element convention is confirmed, or
  * the slot layout of this file, so the converter can read it,
then move the folders back and renumber.
