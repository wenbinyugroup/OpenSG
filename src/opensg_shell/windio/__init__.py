"""opensg_shell.windio -- windIO blade -> beam pipeline (standalone, no external
wrapper): cross-sections, beam properties, BeamDyn coupling, stress recovery.

    sg_windio.py     windIO v1/v2 blade reader (plain yaml, no windIO package),
                     station cross-section builder, 1-D shell SG yaml emitter
                     (reference-axis origin, center/OML reference)
    sg_props.py      RM shell homogenization driver + 6x6 mass matrix of the
                     1-D ring (thickness-integrated density moments), VABS
                     .K-layout output writer
    sg_beamdyn.py    VABS .K reader, VABS<->BeamDyn frame transforms, BeamDyn
                     blade-props / primary / driver writers (nodal loads),
                     beamdyn_driver runner, per-station .ff extraction
                     (u, theta, DCM, F, M in the VABS frame)
    sg_recovery.py   per-station two-step RM dehomogenization driven by a .ff
                     file; VABS-layout .SM / .EM / .U writers

Terminal route for step 1 (all stations, PreVABS XML byproduct on by
default), and the one-shot station bypass (windIO -> Timoshenko 6x6, steps
1 + 2 fused for one station):

    opensg gen_windio_cs <windio.yaml>          (sg_windio.generate_cross_sections)
    opensg windio_st <windio.yaml> <r> [r ...]  (sg_props.station_timo)

Pipeline (examples/OpenSG_shell/windio):
    1  windIO yaml         -> <tag>_shell.yaml per station     (sg_windio)
    2  <tag>_shell.yaml    -> <tag>.K  (Timoshenko 6x6 + mass) (sg_props)
    3  <tag>.K x stations  -> BeamDyn blade props file          (sg_beamdyn)
    4  loads + geometry    -> BeamDyn primary + driver inputs   (sg_beamdyn)
    5  beamdyn_driver run  -> per-station <tag>.ff              (sg_beamdyn)
    6  <tag>.ff + yaml     -> <tag>.SM / .EM / .U               (sg_recovery)
"""
from .sg_windio import (load_blade, build_cross_section, emit_shell_yaml,   # noqa: F401
                        emit_prevabs_xml, blade_stations, blade_length,
                        generate_cross_sections, oml_load_integrals)
from .sg_props import (mass_matrix_ring, beam_props, write_vabs_k,          # noqa: F401
                       station_timo)
from .sg_beamdyn import (read_k_file, vabs_to_beamdyn, write_bd_props,      # noqa: F401
                         write_bd_primary, write_bd_driver, station_loads,
                         run_beamdyn, read_bd_out, extract_ff, write_ff,
                         read_ff, wm_to_dcm, dcm_to_wm)
from .sg_recovery import (dehom_station, dehom_to_files, write_vabs_sm,     # noqa: F401
                          write_vabs_em, write_vabs_u)
