"""sg_read_abaqus_rpt_for_elemental_stress_output.py -- turn the Abaqus
ELEMENTAL output of a tiled 3-D reference run (the abq_dump_3d csvs:
one CENTROIDAL value per C3D4 = its single integration point) into
`.SM`-type and `.U`-type dats, so the reference reads exactly like an
OpenSG dehom field file (same columns, same header discipline) and
every .SM-consuming tool works on it unchanged.

Use:  from opensg import helper
      helper.abaqus_to_sm("AR5_3D_S_material.csv", "AR5_3D.SM")
      helper.abaqus_to_u("AR5_3D_U.csv", "AR5_3D.U")

In:  the abq_dump_3d csvs -- stress `elem,x,y,z,S11,S22,S33,S12,S13,
     S23` (material frame, as stored in the odb) and displacement
     `node,x,y,z,U1,U2,U3`
Out: <out> with rows `x y z S11 S22 S33 S12 S13 S23` (.SM layout) or
     `x y z U1 U2 U3` (.U layout), '#' headers stating provenance
"""
import os

import numpy as np


def abaqus_to_sm(csv_path, out_path=None, frame="material"):
    """Elemental stress csv -> a .SM-layout dat (one row per element:
    the C3D4 single-IP value at the centroid).

    In:  csv_path str; out_path str | None (None -> csv stem + '.SM');
         frame str -- stated in the header ('material' = odb as-stored
         with *Orientation, the dehom .SM default)
    Out: the written path."""
    a = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    out = out_path or os.path.splitext(csv_path)[0] + ".SM"
    with open(out, "w") as f:
        f.write("# %s -- Abaqus 3-D FEA elemental stress (one C3D4"
                " integration point per element, position=CENTROIDAL)\n"
                % os.path.basename(csv_path))
        f.write("# frame: %s; columns follow the OpenSG dehom .SM"
                " layout\n" % frame)
        f.write("# %12s %14s %14s %14s %14s %14s %14s %14s %14s\n"
                % ("x", "y", "z", "S11", "S22", "S33", "S12", "S13",
                   "S23"))
        np.savetxt(f, a[:, 1:10], fmt="%14.6e")
    print("wrote %s : %d elements" % (out, len(a)))
    return out


def abaqus_to_u(csv_path, out_path=None):
    """Nodal displacement csv -> a .U-layout dat.

    In:  csv_path str -- node,x,y,z,U1,U2,U3; out_path str | None
    Out: the written path."""
    a = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    out = out_path or os.path.splitext(csv_path)[0] + ".U"
    with open(out, "w") as f:
        f.write("# %s -- Abaqus 3-D FEA nodal displacement (TOTAL, not"
                " fluctuation-only like a dehom .U)\n"
                % os.path.basename(csv_path))
        f.write("# %12s %14s %14s %14s %14s %14s\n"
                % ("x", "y", "z", "U1", "U2", "U3"))
        np.savetxt(f, a[:, 1:7], fmt="%14.6e")
    print("wrote %s : %d nodes" % (out, len(a)))
    return out
