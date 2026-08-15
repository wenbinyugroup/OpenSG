"""opensg_shell.pynumad -- the COMPLETE blade workflow (all dialects).

One package owns everything from the blade yaml to BeamDyn and back:
reading (windIO v1, windIO v2 and the pyNuMAD width-form dialect),
cross-section resolution, the editable Blade object, the in-memory
homogenizer, the VABS-layout records, the BeamDyn coupling and the
stress recovery.  The former opensg_shell.windio package is DELETED;
its BeamDyn/recovery modules live here now.

    sg_mesh.py       blade readers (WindIOBlade v2, WindIOBladeV1) and the
                     cross-section builder + 1-D shell SG yaml emitter.
    sg_pynumad.py    PyNuMADBlade reader (width-form placement resolution,
                     loud skip warnings), load_blade auto-detect, st-id
                     resolution, the file's own elastic_properties_mb 6x6
                     accessor, and the IN-MEMORY station / all-stations
                     drivers (VABS-layout .out is the only artifact).
    sg_blade.py      the editable Blade object (opensg.read entry) +
                     the per-station Section override.
    sg_homo.py       STANDALONE in-memory homogenizer: ring driver, laws,
                     curvature, mass, VABS .K writer -- `timo(blade, st)`.
    sg_props.py      1-D shell SG yaml -> Timoshenko + mass + .K record
                     (beam_props / mass_matrix_ring -- the artifacts-route
                     consumer of the emitted station yaml).
    sg_beamdyn.py    VABS .K reader, VABS<->BeamDyn frame transforms,
                     BeamDyn input writers, driver runner, .ff extraction.
    sg_recovery.py   per-station two-step RM dehomogenization driven by a
                     .ff file; VABS-layout .SM / .EM / .U writers.

Terminal route (any dialect -- windIO v1/v2 or pyNuMAD):

    opensg pynumad <blade.yaml> <st-id>     Timoshenko 6x6 at one station
                                            (st-id = 0-based station index,
                                            a span r in [0, 1], or "all")
"""
from .sg_pynumad import (PyNuMADBlade, load_blade_pynumad, load_blade,       # noqa: F401
                         blade_stations, blade_length, resolve_station,
                         file_six_by_six, FILE_DIAG_TO_VABS,
                         station_timo, generate_cross_sections)
from .sg_mesh import (WindIOBlade, WindIOBladeV1, build_cross_section,       # noqa: F401
                      emit_shell_yaml, emit_prevabs_xml)
from .sg_homo import timo, timo_cs, write_vabs_k                             # noqa: F401
from .sg_props import mass_matrix_ring, beam_props                           # noqa: F401
from .sg_beamdyn import (read_k_file, vabs_to_beamdyn, write_bd_props,       # noqa: F401
                         write_bd_primary, write_bd_driver, station_loads,
                         run_beamdyn, read_bd_out, extract_ff, write_ff,
                         read_ff, wm_to_dcm, dcm_to_wm)
from .sg_recovery import (dehom_station, dehom_to_files, write_vabs_sm,      # noqa: F401
                          write_vabs_em, write_vabs_u)
from .sg_blade import Blade, Section, opensg_blade                           # noqa: F401
