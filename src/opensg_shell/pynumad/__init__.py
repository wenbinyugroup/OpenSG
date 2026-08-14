"""opensg_shell.pynumad -- the pyNuMAD blade dialect, SEPARATE from windio.

pyNuMAD exports look like windIO v1 but place TE/LE reinforcements with
width-based forms (start/end/midpoint_nd_arc + width [m]) that the plain
windio reader would drop from the laminate.  This package owns that dialect:

    sg_mesh.py       pynumad-owned blade reader (windIO-v1 shape) and
                     cross-section builder -- _interp, WindIOBladeV1,
                     build_cross_section.
    sg_pynumad.py    PyNuMADBlade reader (width-form placement resolution,
                     loud skip warnings), st-id resolution, the file's own
                     elastic_properties_mb 6x6 accessor, and the IN-MEMORY
                     station / all-stations drivers (VABS-layout .out is
                     the only artifact; no 1-D yaml emission).
    sg_homo.py       STANDALONE in-memory homogenizer: ring driver, laws,
                     curvature, mass, VABS .K writer -- `timo(blade, st)`.

    The package imports NOTHING from opensg_shell.windio: that machinery's
    only role here was emitting 1-D yamls, which this route does not need.

Terminal route:

    opensg pynumad <blade.yaml> <st-id>     Timoshenko 6x6 at one station
                                            (st-id = 0-based station index,
                                            a span r in [0, 1], or "all")
"""
from .sg_pynumad import (PyNuMADBlade, load_blade_pynumad, resolve_station,  # noqa: F401
                         file_six_by_six, FILE_DIAG_TO_VABS,
                         station_timo, generate_cross_sections)
from .sg_homo import timo                                                    # noqa: F401
