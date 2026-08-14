"""opensg_shell command line -- the shell twin of `opensg_solid`:

    opensg_shell <sg.yaml>        homogenization (the default)
    opensg_shell <sg.yaml> D      dehomogenization: homogenize, then recover

    opensg_shell gen_windio_cs <windio.yaml>
                                  every blade station -> VABS-layout .out
                                  station record + spanwise .dat tables,
                                  fully in memory (the pynumad drivers; any
                                  dialect: windIO v1/v2 or pyNuMAD; see
                                  gen_windio_cs --help for --stations,
                                  --mesh-size, --out, --prefix)

    opensg_shell windio_st <windio.yaml> <r> [r ...]
                                  one-shot station bypass: blade + span r ->
                                  Timoshenko 6x6 printed and stored as the
                                  VABS-layout <tag>.out, nothing else --
                                  fully in memory (windio_st --help for
                                  --mesh-size, --out, --prefix)

    opensg_shell pynumad <blade.yaml> <st-id>
                                  the blade route by st-id (any dialect;
                                  pyNuMAD width-placed reinforcements
                                  resolved): st-id = 0-based station index |
                                  span r | "all".  One station prints the
                                  Timoshenko 6x6 plus the cross-check vs the
                                  file's own elastic_properties_mb; "all"
                                  sweeps every station (pynumad --help)

No flags, no codes: everything else lives in the yaml header (the
leading scalar keys above the mesh blocks), and every key has a default
-- a headerless msg-shell SG runs as a classical beam homogenization.

    msg: shell          # which ENGINE owns this file: `shell` = this one,
                        #   `solid` = opensg_solid.  Omit it and the mesh
                        #   dialect decides (nodes/elements/sets/sections/
                        #   elementOrientations/materials = shell), so older
                        #   files keep working; the unified `opensg`
                        #   command dispatches on exactly this.
    n_model: 1          # 1 = beam, 2 = plate, 3 = 3-D solid
                        #   2 is the opensg_solid route: opensg_shell has NO
                        #   plate macro model (the shell WALL is the plate;
                        #   the cross-section ring routes write its law as
                        #   <base>_ABDG.out)
    refined: 0          # 0 = classical (Kirchhoff-Love wall -> Euler-
                        #     Bernoulli beam 4x4), 1 = shear-refined
                        #     (Reissner-Mindlin wall -> Timoshenko beam
                        #     6x6); ignored for the solid model
    analysis: H         # the same H | D switch, when you would rather
                        #   carry it in the file than on the command
                        #   line (the argument wins)
    epsilon_bar: [...]  # (6,) macro beam strain for D --
                        #   [ext sh2 sh3 twist bend2 bend3]
    omega: <area>       # OPTIONAL user SG measure for the n_model 3
                        #   cross-section route (e.g. the wall MATERIAL
                        #   area) -- overrides the measured bounding-box
                        #   cell area, exactly like the drivers'
                        #   build_solid_bundle(..., cell_area=omega)
    junction: flag      # OPTIONAL, D only -- junction-aware recovery:
                        #   off     = no extra columns, no sidecar (the
                        #             pre-junction behaviour, bit-identical)
                        #   flag    = (default) NON-MUTATING: append jflag
                        #             and jdist to <base>_dehom.txt, add
                        #             them as .vtk CELL_DATA, write the
                        #             <base>_dehom.junc sidecar and print
                        #             the census; NO field value changes
                        #   exclude = as flag, plus NaN in the 15 field
                        #             columns of every duplicate (jflag 3)
                        #             row, so the cloud PARTITIONS the
                        #             material instead of double-covering
                        #             it at wall crossings
                        #   (see sg_dehom_junction; NOT the homogenization
                        #   junction correction, a different switch)
    junction_bl: 1.0    # OPTIONAL k_bl -- junction boundary-layer radius in
                        #   units of the thickest wall there (a modelling
                        #   choice, not a derived length)
    junction_ang: 1.0   # OPTIONAL tangent-grouping tolerance (deg) of the
                        #   junction detector.  1.0 = the homogenization
                        #   census value, which also calls every >1 deg
                        #   contour KINK a junction; 5-15 lists the wall
                        #   crossings only (it moves the advisory `near`
                        #   count, not the overlap flags)
    aperiodic: 1        # OPTIONAL, and only for the n_model 3 3-D shell
                        #   SG: replace periodicity by the BOUNDARY
                        #   SOLUTION (w = 0 Dirichlet on the bounding-box
                        #   faces, a kinematic upper bound on one cell).
                        #   Omit the key for the periodic default.  The
                        #   3-D shell SEGMENT route (n_model 1 on a
                        #   surface mesh) is aperiodic by construction --
                        #   its ends carry the boundary rings' own V0/V1
                        #   -- so there the key only documents the case.

The MESH picks the engine inside a macro model -- 2-node line elements
are a 1-D cross-section (ring) SG, 3/4-node elements a 3-D shell SG:

    n_model 1, ring  -> RM ring        (sg_homo.build_rm_bundle)
    n_model 1, 3-D   -> aperiodic/tapered segment
                        (sg_homo.segment_timo_from_3dyaml)
    n_model 3, ring  -> equivalent solid of a cross-section
                        (sg_homo.build_solid_bundle)
    n_model 3, 3-D   -> equivalent solid of a 3-D shell SG
                        (sg_homo.shell_sg3d)

H writes the timed SwiftComp-layout .out the engine owns (<base>_Timo.out
for a beam, <base>_EB.out for the classical beam, <base>_C3D.out for a
solid); every route that reduces a layup to a wall plate law -- the
cross-section rings (beam refined: 1 and the n_model 3 equivalent solid) AND
the 3-D shell SG -- also writes that step-1 record as <base>_ABDG.out
(8x8 Reissner-Mindlin ABDG + compliance, one block per section).  D drives the
RM two-step recovery from the `.ff` macro state next to the yaml and
additionally writes <base>_dehom.txt/.vtk.
"""
import os
import sys

from opensg_solid.cli import BANNER, read_ff_state   # ONE banner and ONE .ff
                                                     # reader for both CLIs

_MDL = {1: "beam", 2: "plate", 3: "3-D solid"}
# reference surface -> laminate-thickness fraction, the SAME map
# build_rm_bundle / build_solid_bundle use
_FRAC = {"center": 0.5, "oml": 0.0, "oml_flip": 1.0, "iml": 1.0}


def mesh_kind(path):
    """1-D cross-section ring or 3-D shell SG, read from the connectivity.

    The msg-shell dialect has no `dim:` key: an element with 2 nodes is a
    ring (contour) segment, one with 3 or 4 nodes a shell facet of a 3-D
    SG -- exactly what the loaders assume.

    This is also the SG's intrinsic dimension (sg_dim): the ring IS 1-D and
    the surface mesh IS 2-D, however many coordinates its nodes carry.

    In:  path str -- an msg-shell SG yaml
    Out: int 1 (1-D ring contour) | 2 (3-D shell surface mesh)."""
    from opensg_solid.sg_mesh import elem_node_count   # ONE element scanner

    return 1 if elem_node_count(path) <= 2 else 2


def sg_reference(path, default="center"):
    """The reference surface the yaml records (`reference:`), read cheaply.

    In:  path str -- an msg-shell SG yaml; default str when the key is absent
    Out: str -- "center" | "oml" | "oml_flip" | "iml"."""
    with open(path) as f:
        for ln in f:
            if ln.startswith("reference:"):
                return ln.split(":", 1)[1].split("#")[0].strip().strip("'\"") \
                    or default
    return default


def node_extents(path):
    """The SG bounding box, measured from the node block (never declared).

    In:  path str -- an msg-shell SG yaml
    Out: (3,) float ndarray -- the per-axis extent of the node cloud."""
    import numpy as np

    vals, on = [], False
    with open(path) as f:
        for ln in f:
            top = (":" in ln) and not ln[:1].isspace() and not ln.startswith("-")
            if top:
                if on:
                    break
                on = ln.split(":", 1)[0].strip() == "nodes"
                continue
            if not on:
                continue
            s = ln.strip()
            if s.startswith("- "):              # strip the list dash only --
                s = s[2:]                       # never a number's minus sign
            vals.extend(float(t) for t in
                        s.replace("[", " ").replace("]", " ")
                         .replace(",", " ").split())
    a = np.asarray(vals, float)
    if a.size == 0 or a.size % 3:
        import yaml as _yaml                    # fallback: let yaml do it
        from .sg_mesh import _row
        d = _yaml.safe_load(open(path))
        a = np.array([[float(v) for v in _row(r)][:3] for r in d["nodes"]])
    a = a.reshape(-1, 3)
    return a.max(0) - a.min(0)


def sg_cell_area(path):
    """The cross-section SG cell measure (omega) of an msg-shell yaml.

    ONE rule, so the CLI and the example drivers cannot drift apart: the
    `omega:` header key when the file declares one (e.g. the wall MATERIAL
    area of a closed tube, which is not a geometric property of the node
    cloud), otherwise the MEASURED periodic cell -- the node bounding box,
    because the assembly map ties opposite box faces.  The bounding box is
    NOT the convex hull of the contour; the two differ on any non-convex
    cell (honeycomb, lattice).

    In:  path str -- an msg-shell SG yaml
    Out: float -- the cell area to divide the 3-D solid law by."""
    from opensg_solid.sg_mesh import read_yaml_header

    om = read_yaml_header(path).get("omega")
    if om is not None:
        return float(om)
    ext = node_extents(path)              # ring loaders: ax = 2, cross = [0, 1]
    return float(ext[0]*ext[1])


def gen_windio_cs(argv):
    """`gen_windio_cs <windio.yaml>` -- every blade station, fully in memory.

    The pynumad sweep driver on any dialect (windIO v1/v2 or pyNuMAD):
    per station the VABS-layout <tag>.out record, plus the spanwise
    stations/timo_by_r/mass_by_r .dat tables, all under --out.  The 1-D
    yaml / XML / PNG emission of the deleted windio machinery is gone.

    In:  argv list[str] -- [windio_path, flags...] (the token after
         `gen_windio_cs`); see --help
    Out: int exit code (0 ok, 2 usage)."""
    import argparse

    p = argparse.ArgumentParser(
        prog="opensg gen_windio_cs",
        description="blade yaml (windIO v1/v2 or pyNuMAD) -> VABS-layout"
                    " .out station records + spanwise .dat tables, fully"
                    " in memory.")
    p.add_argument("windio", help="blade yaml (windIO v1/v2 or pyNuMAD)")
    p.add_argument("--stations", default="airfoil", metavar="S",
                   help='"airfoil" = the blade\'s own airfoil positions (the'
                        " default); an int N = N uniform stations r = i/(N-1);"
                        " or comma-separated r values, e.g. 0.2,0.5,0.7")
    p.add_argument("--mesh-size", type=float, default=0.01, metavar="H",
                   help="target element arc length / chord (default 0.01)")
    p.add_argument("--no-xml", action="store_true",
                   help="accepted for compatibility; ignored")
    p.add_argument("--out", default="cross_sections", metavar="DIR",
                   help="output folder (default cross_sections)")
    p.add_argument("--prefix", default=None, metavar="TAG",
                   help="station tag prefix (default: the blade file stem)")
    a = p.parse_args(argv)
    if not os.path.exists(a.windio):
        raise SystemExit("no such file: %s" % a.windio)
    st = a.stations
    if st != "airfoil":
        try:
            st = int(st)
        except ValueError:
            st = [float(v) for v in st.replace(",", " ").split()]

    import time as _time

    _t0 = _time.perf_counter()
    from .pynumad import generate_cross_sections
    out = generate_cross_sections(a.windio, out_dir=a.out, stations=st,
                                  mesh_size=a.mesh_size, prefix=a.prefix)
    print("Station records stored in %s (%d stations: .out + spanwise"
          " .dat tables)" % (a.out, len(out)))
    print("Time taken: %.2f sec" % (_time.perf_counter() - _t0))
    return 0


def windio_st(argv):
    """`windio_st <windio.yaml> <r> [r ...]` -- Timoshenko 6x6, one shot.

    The pynumad in-memory station driver on any dialect: blade + span r ->
    Timoshenko 6x6 printed and stored as the VABS-layout <tag>.out record,
    nothing else (the yaml/XML/PNG emission of the deleted windio
    machinery is gone).

    In:  argv list[str] -- [windio_path, r, ..., flags] (the tokens after
         `windio_st`); see --help
    Out: int exit code (0 ok, 2 usage)."""
    import argparse

    p = argparse.ArgumentParser(
        prog="opensg windio_st",
        description="blade yaml + station r -> Timoshenko 6x6 (RM ring),"
                    " fully in memory; stores the VABS-layout <tag>.out.")
    p.add_argument("windio", help="blade yaml (windIO v1/v2 or pyNuMAD)")
    p.add_argument("r", nargs="+", type=float,
                   help="non-dimensional span station(s) in [0, 1]")
    p.add_argument("--mesh-size", type=float, default=0.01, metavar="H",
                   help="target element arc length / chord (default 0.01)")
    p.add_argument("--out", default=".", metavar="DIR",
                   help="output folder (default: the current directory)")
    p.add_argument("--prefix", default=None, metavar="TAG",
                   help="station tag prefix (default: the blade file stem)")
    p.add_argument("--xml", action="store_true",
                   help="accepted for compatibility; ignored")
    p.add_argument("--view", action="store_true",
                   help="accepted for compatibility; ignored")
    a = p.parse_args(argv)
    if not os.path.exists(a.windio):
        raise SystemExit("no such file: %s" % a.windio)
    for r in a.r:
        if not 0.0 <= r <= 1.0:
            raise SystemExit("station r must be in [0, 1], got %r" % r)

    import time as _time

    from .pynumad import station_timo
    for r in a.r:
        t0 = _time.perf_counter()
        P = station_timo(a.windio, "%.10f" % r, mesh_size=a.mesh_size,
                         out_dir=a.out, prefix=a.prefix)
        m = P["mesh"]
        print(" station   : r = %.4f   chord = %.3f m   %d nodes  %d elems"
              "  %d sets  %d webs"
              % (r, P["chord"], m["n_nodes"], m["n_elems"],
                 m["n_sets"], m["n_webs"]))
        print("Timoshenko Beam Stiffness Matrix  "
              "[eps11 gam12 gam13 kappa1 kappa2 kappa3]:")
        print(P["Timo"])
        print(" mass/span = %.6g kg/m" % P["info"]["mpus"])
        print("Homogenization stored in %s" % P["k_file"])
        print("Time taken: %.2f sec" % (_time.perf_counter() - t0))
    return 0


def pynumad(argv):
    """`pynumad <blade.yaml> <st-id>` -- the pyNuMAD blade dialect route.

    st-id = a 0-based integer index into the blade file's OWN spanwise
    stations, a span r in [0, 1] (any token with a decimal point), or "all".
    One station: Timoshenko 6x6 printed and stored as the VABS-layout
    <tag>.out record, nothing else, with the cross-check table against the
    file's own elastic_properties_mb appended when the block is present.
    "all": every station -> <tag>.out + the spanwise
    stations/timo_by_r/mass_by_r .dat tables under --out, fully in memory
    (the gen_windio_cs equivalent for this dialect).

    In:  argv list[str] -- [blade_path, st_id, flags...] (the tokens after
         `pynumad`); see --help
    Out: int exit code (0 ok, 2 usage)."""
    import argparse

    p = argparse.ArgumentParser(
        prog="opensg pynumad",
        description="pyNuMAD blade yaml (windIO-v1 dialect with width-placed"
                    " layers) -> Timoshenko 6x6 at one station, or all"
                    " cross-sections with st-id \"all\".")
    p.add_argument("blade", help="pyNuMAD blade yaml")
    p.add_argument("station", nargs="?", default="all",
                   help='st-id: 0-based station index, a span r in [0, 1]'
                        ' (with a decimal point), or "all" -- the default'
                        " when omitted")
    p.add_argument("--mesh-size", type=float, default=0.01, metavar="H",
                   help="target element arc length / chord (default 0.01)")
    p.add_argument("--out", default=None, metavar="DIR",
                   help="output folder (default: cross_sections for"
                        ' "all", the current directory for one station)')
    p.add_argument("--no-xml", action="store_true",
                   help="accepted for compatibility; ignored")
    p.add_argument("--xml", action="store_true",
                   help="accepted for compatibility; ignored")
    p.add_argument("--view", action="store_true",
                   help="accepted for compatibility; ignored")
    a = p.parse_args(argv)
    if not os.path.exists(a.blade):
        raise SystemExit("no such file: %s" % a.blade)

    import time as _time

    from .pynumad import load_blade_pynumad, station_timo

    if a.station == "all":
        import numpy as np

        rs = load_blade_pynumad(a.blade).stations()
        out = a.out or "cross_sections"
        stem = os.path.splitext(os.path.basename(a.blade))[0]
        rows, rows_k, rows_m = [], [], []
        _t0 = _time.perf_counter()
        for r in rs:
            P = station_timo(a.blade, "%.10f" % r, mesh_size=a.mesh_size,
                             out_dir=out)
            print(" r = %.4f" % P["r"])
            m = P["mesh"]
            rows.append([P["r"], P["chord"], P["twist"], m["n_nodes"],
                         m["n_elems"], m["n_sets"], m["n_webs"]])
            rows_k.append(np.r_[P["r"], np.asarray(P["Timo"]).flatten()])
            rows_m.append(np.r_[P["r"], P["info"]["mpus"],
                                np.asarray(P["Mass"]).flatten()])
        np.savetxt(os.path.join(out, stem + "_stations.dat"), np.array(rows),
                   fmt="%.4f %10.4f %10.6f %6d %6d %3d %3d",
                   header="r chord[m] twist[windIO unit] n_nodes n_elems"
                          " n_sets n_webs")
        np.savetxt(os.path.join(out, stem + "_timo_by_r.dat"),
                   np.array(rows_k), fmt="%.8e",
                   header="r then Timoshenko 6x6 row-major (VABS order)")
        np.savetxt(os.path.join(out, stem + "_mass_by_r.dat"),
                   np.array(rows_m), fmt="%.8e",
                   header="r mass_per_span then mass 6x6 row-major"
                          " (VABS frame)")
        print("Cross-sections + Timoshenko stored in %s (%d stations)"
              % (out, len(rs)))
        print("Time taken: %.2f sec" % (_time.perf_counter() - _t0))
        return 0

    _t0 = _time.perf_counter()
    P = station_timo(a.blade, a.station, mesh_size=a.mesh_size,
                     out_dir=a.out or ".")
    print("Timoshenko Beam Stiffness Matrix  "
          "[eps11 gam12 gam13 kappa1 kappa2 kappa3]:")
    print(P["Timo"])
    print("Mass Matrix  [F1 F2 F3 M1 M2 M3]:")
    print(P["Mass"])
    print("Homogenization stored in %s" % P["k_file"])
    print("Time taken: %.2f sec" % (_time.perf_counter() - _t0))
    return 0


def main(argv=None):
    """Run the analysis the shell SG yaml (and an optional H|D argument) asks for.

    In:  argv (list[str] | None) -- [yaml_path, "H"|"D" (optional)], or
         ["gen_windio_cs", windio_path, flags...] for the windIO
         cross-section generator, or ["windio_st", windio_path, r, ...] for
         the one-shot windIO station Timoshenko bypass, or
         ["pynumad", blade_path, st_id, ...] for the pyNuMAD blade dialect;
         None reads sys.argv
    Out: int exit code (0 ok, 2 usage)."""
    print(BANNER)
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "gen_windio_cs":
        return gen_windio_cs(argv[1:])
    if argv and argv[0] == "windio_st":
        return windio_st(argv[1:])
    if argv and argv[0] == "pynumad":
        return pynumad(argv[1:])
    if not 1 <= len(argv) <= 2 or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 2
    path = argv[0]
    if not os.path.exists(path):
        alt = os.path.splitext(path)[0] + ".yaml"
        raise SystemExit("no such file: %s%s" % (
            path, "" if not os.path.exists(alt) else
            "\ndid you mean %s ?" % alt))

    import time as _time

    import numpy as np

    # the opensg_solid header reader: its stop-key set already covers the
    # msg-shell block keys (nodes/elements/sets/sections/...)
    from opensg_solid.sg_mesh import read_yaml_header, resolve_msg, sg_dim, node_span_dim

    # an msh_to_yaml TEMPLATE carries the mesh but only FILL_IN placeholders
    # for the layup/materials: name the missing fields instead of failing
    # deep inside the ABD build (cheap line scan, never a full parse)
    from .helper.msh_to_yaml import check_filled
    _todo = check_filled(path)
    if _todo:
        raise SystemExit(_todo)

    hdr = read_yaml_header(path)
    analysis = str(argv[1] if len(argv) == 2
                   else hdr.get("analysis", "H")).strip().upper()
    if analysis not in ("H", "D"):
        raise SystemExit("the analysis argument must be H (homogenization)"
                         " or D (dehomogenization), got %r" % argv[1])
    n_model = int(hdr.get("n_model", 1))
    refined = int(hdr.get("refined", 0))
    if n_model == 2:
        raise SystemExit(
            "n_model 2 (plate) is the opensg_solid route: opensg_shell has no"
            " plate macro\nmodel -- the shell WALL is the plate here, and the"
            " cross-section ring routes\nwrite its ABD/ABDG law as"
            " <base>_ABDG.out."
            "\nRun  opensg_solid %s\nfor a plate homogenization, or set"
            " n_model 1 (beam) / 3 (3-D solid) in the header."
            % os.path.basename(path))
    if n_model not in (1, 3):
        raise SystemExit("n_model must be 1 (beam), 2 (plate, opensg_solid)"
                         " or 3 (3-D solid), got %r" % n_model)
    if refined not in (0, 1):
        raise SystemExit("refined must be 0 (classical: beam Euler-Bernoulli)"
                         " or 1 (shear-refined: beam Timoshenko), got %r"
                         % refined)
    # junction-aware recovery tier (D only, but validated here with the rest
    # of the header so a typo never survives an H run)
    from .sg_dehom_junction import read_tier
    junction, junction_bl, junction_ang = read_tier(hdr)

    _t0 = _time.perf_counter()
    print(" input     : %s" % os.path.abspath(path))
    print(" msg       : %s" % resolve_msg(path))     # the engine that owns it
    print(" SG dim    : %dD" % node_span_dim(path))   # the space the SG occupies
    print(" analysis  : %s" % ("homogenization" if analysis == "H"
                               else "dehomogenization"))
    print(" macro model: %s, %s%s"
          % (_MDL[n_model], "shear-refined" if refined else "classical",
             ", aperiodic" if int(hdr.get("aperiodic", 0) or 0) else ""))
    print("")

    base = os.path.splitext(path)[0]
    kind = mesh_kind(path)
    B = None                      # the RM bundle, when the route builds one

    if n_model == 1 and refined:
        # Reissner-Mindlin wall -> Timoshenko beam 6x6
        if kind == 1:
            from .sg_homo import build_rm_bundle
            B = build_rm_bundle(path)
            law, out_path = np.asarray(B["Timo"]), base + "_Timo.out"
            solve_time = float(B.get("solve_time", _time.perf_counter() - _t0))
        else:
            from .sg_homo import segment_timo_from_3dyaml
            S = segment_timo_from_3dyaml(path)
            law, out_path = np.asarray(S["S6"]), base + "_Timo.out"
            solve_time = float(S["solve_time"])
        law_title = ("Timoshenko Beam Stiffness Matrix  "
                     "[eps11 gam12 gam13 kappa1 kappa2 kappa3]")

    elif n_model == 1:
        # classical: the Kirchhoff-Love (Hermite C1) counterpart of the same
        # ring -- a genuine 4x4, not a slice of the Timoshenko 6x6
        if kind != 1:
            raise SystemExit(
                "the 3-D shell SEGMENT route (aperiodic boundary V0/V1) returns"
                " the\nTimoshenko 6x6 only -- there is no classical segment"
                " result to report.\nSet refined: 1 in the header.")
        from .fe_jax.msg_hermite import solve_tw_from_yaml
        from opensg_solid.sg_homo import write_sc_K
        ref = sg_reference(path)
        KL = solve_tw_from_yaml(path, frac=_FRAC.get(ref, 0.0))
        law = np.asarray(KL["EB"])
        law, out_path = 0.5*(law + law.T), base + "_EB.out"
        solve_time = _time.perf_counter() - _t0
        write_sc_K(out_path, law, solve_time=solve_time,
                   model="msg-shell beam model (Kirchhoff-Love wall)"
                         " [ext twist bend2 bend3]",
                   constants=False, name="Euler-Bernoulli Beam")
        law_title = ("Euler-Bernoulli Beam Stiffness Matrix  "
                     "[eps11 kappa1 kappa2 kappa3]")

    elif kind == 1:
        # cross-section SG -> equivalent 3-D solid.  omega is MEASURED, like
        # every other OpenSG SG measure: the periodic cell of this SG is the
        # node bounding box (the assembly map ties opposite box faces), NOT
        # the convex hull of the contour -- they differ on any non-convex
        # cell (honeycomb, lattice).  An `omega:` header key overrides the
        # measure (e.g. the wall MATERIAL area), exactly the drivers'
        # build_solid_bundle(..., cell_area=...) user measure.
        from .sg_homo import build_solid_bundle
        B = build_solid_bundle(path, cell_area=sg_cell_area(path))
        law, out_path = np.asarray(B["C3D"]), base + "_C3D.out"
        solve_time = float(B["solve_time"])
        law_title = "Cauchy Continuum Stiffness Matrix  [11 22 33 23 13 12]"

    else:
        # 3-D shell SG -> equivalent 3-D solid (periodic in 3 directions)
        from .sg_homo import shell_sg3d
        # everything this route needs is IN the yaml: the layup/material come
        # from sections:/materials:, and an optional `omega:` header overrides
        # the MEASURED SG measure -- the node bounding-box VOLUME, the volume
        # the equivalent continuum occupies and the cell the periodic assembly
        # map ties (nothing is supplied in code)
        r = shell_sg3d(path, omega=hdr.get("omega"))
        # the .out normalizes per unit cell (SwiftComp parity); print THAT.
        # With the default measure this is exactly r["C3D"]; the cell volume
        # comes back from the solve, so the 4 MB node block is scanned once.
        law = np.asarray(r["D_eff"])/float(r["cell_volume"])
        out_path = base + "_C3D.out"
        solve_time = float(r["solve_time"])
        law_title = "Cauchy Continuum Stiffness Matrix  [11 22 33 23 13 12]"

    print(law_title + ":")
    print(law)
    if analysis == "H":
        print("Homogenization stored in %s" % out_path)
        print("Time taken: %.2f sec" % solve_time)

    if analysis == "D":
        if B is None or n_model != 1:
            raise SystemExit(
                "dehomogenization is the RM two-step recovery of a 1-D shell"
                " cross-section\n(beam macro model, refined: 1): st = C6^-1 FF"
                " -> RM shell strains -> plate\nthrough-thickness SG.\nThe %s"
                " route has no recovery in opensg_shell."
                % ("equivalent-solid" if n_model == 3 else
                   "classical (Kirchhoff-Love / Euler-Bernoulli) beam"
                   if kind == 1 else "3-D shell segment"))
        state = read_ff_state(base + ".ff")
        if state is not None:
            eps = np.linalg.solve(np.asarray(law, float), state["FF"])
        else:
            eps = hdr.get("epsilon_bar")
            if eps is None or len(eps) != 6:
                raise SystemExit(
                    "analysis D needs the macro state: either %s.ff (u, theta,"
                    " C, FF) or `epsilon_bar:` in the yaml header"
                    % os.path.basename(base))
            eps = np.asarray([float(x) for x in eps], float)
        dehom_write(B, eps, base + "_dehom", state=state,
                    junction=junction, junction_bl=junction_bl,
                    junction_ang=junction_ang)
        print("Local field files are computed and stored.")
        print("Time taken: %.2f sec" % (_time.perf_counter() - _t0))
    return 0


def recovery_points(B, n_depth=9):
    """The through-thickness recovery grid of an RM ring bundle.

    One column of ``n_depth`` ply-interior stations per ring element, hung
    off the element mid-arc along the inward wall normal -- the sampling
    the shell dehom example uses.

    In:  B dict -- build_rm_bundle bundle; n_depth int -- stations per wall
    Out: dict {pts (E*n_depth, 2) section coords, emid (E, 2) mid-arc points,
         nvec (E, 2) inward wall normal, h (E,) layup thickness,
         zeta (n_depth,) 0 = OML .. 1 = IML, frac float}."""
    import numpy as np

    corners = np.asarray(B["corners"]); rc = np.asarray(B["red_cells"])
    cen = corners.mean(0)
    hth = {ln: float(sum(i["thick"])) for ln, i in B["layup_db"].items()}
    h = np.array([hth[ln] for ln in B["layup_per_elem"]])
    T = corners[rc[:, 1]] - corners[rc[:, 0]]
    L = np.hypot(T[:, 0], T[:, 1])
    tun = T/L[:, None]
    nvec = np.column_stack([tun[:, 1], -tun[:, 0]])
    emid = 0.5*(corners[rc[:, 0]] + corners[rc[:, 1]])
    flip = ((cen - emid)*nvec).sum(1) < 0.0             # OML -> IML
    nvec[flip] *= -1.0
    frac = float(B.get("frac", 0.0))
    zeta = (np.arange(n_depth) + 0.5)/n_depth
    z = (zeta[None, :] - frac)*h[:, None]               # depth from the ref
    pts = (emid[:, None, :] + z[:, :, None]*nvec[:, None, :]).reshape(-1, 2)
    return {"pts": pts, "emid": emid, "nvec": nvec, "h": h, "zeta": zeta,
            "frac": frac, "z": z}


def dehom_write(B, eps, out_base, state=None, n_depth=9, frame="material",
                junction="flag", junction_bl=1.0, junction_ang=1.0):
    """RM two-step recovery per (element, zeta), written as .txt + exploded .vtk.

    The exact code path of the shell dehom example (beam_dehom_shell.py):
    step 1 is sg_dehom.ring_wall_strains -- the RM (C0, MITC-g23) shell
    strains, span/arc gradients and layup-boundary-aware nodal averages per
    ring ELEMENT, never a nearest-element re-projection (junction points keep
    their own element's layup); step 2 is the MSG-RM through-thickness plate
    SG (rm_plate_msg / msgrm_strain_at_depth per layup at each depth), which
    carries the transverse-shear rows S23/S13.  The displacement is the RM
    warping at the element mid-arc (mid-surface warping + z (omega x e3))
    plus, when a .ff macro state is given, the macro rigid motion
    u_i + (C_ij - d_ij) y_j  (VABS recovery form).

    ``junction`` is the yaml `junction:` tier (sg_dehom_junction): "off"
    reproduces the pre-junction files byte for byte; "flag" (the default)
    appends the two junction columns jflag/jdist to the .txt, adds them as
    extra .vtk CELL_DATA scalars, writes the <out_base>.junc sidecar and
    prints the census, changing NO field value; "exclude" additionally puts
    NaN in the 15 field columns of every duplicate (jflag 3) ROW of the .txt
    -- the row stays so row i is still element i//n_depth at depth i%n_depth,
    and the cloud then partitions the material domain instead of
    double-covering it.  The .vtk field values are NEVER NaN-ed (legacy VTK
    ASCII readers handle NaN badly): threshold the .vtk on jflag instead.

    In:  B dict -- build_rm_bundle bundle; eps (6,) macro beam strain
         [ext sh2 sh3 twist bend2 bend3]; out_base str -- path stem;
         state dict | None -- read_ff_state result (its FF drives step 1
         when given, exactly like the example); n_depth int;
         frame str -- "material" (ply axes) | "plate" (wall axes);
         junction str -- "off" | "flag" | "exclude";
         junction_bl float -- k_bl, the boundary-layer radius in wall
         thicknesses; junction_ang float -- the tangent-grouping tolerance
         (deg) of the junction detector
    Out: writes <out_base>.txt, <out_base>.vtk and (tier != "off")
         <out_base>.junc; returns None."""
    import numpy as np

    from opensg_solid.rm_plate_1D.msg_rm_plate import (rm_plate_msg,
                                                       msgrm_strain_at_depth)
    from .sg_dehom import ring_wall_strains

    G = recovery_points(B, n_depth=n_depth)
    pts, zeta, frac = G["pts"], G["zeta"], G["frac"]
    if state is not None:
        F = ring_wall_strains(B, beam_force_vabs=state["FF"])
    else:
        F = ring_wall_strains(B, beam_strain=np.asarray(eps, float))
    rc = np.asarray(B["red_cells"])
    layups = B["layup_per_elem"]; ldb = B["layup_db"]; mdb = B["material_db"]
    warpM = {ln: rm_plate_msg(i["thick"], i["angles"], i["mat_names"], mdb,
                              fraction=frac) for ln, i in ldb.items()}
    wn = np.asarray(F["aA"]).reshape(-1, 6)   # per-node [u1 u2 u3 om1 om2 om3]
    s6n, s6mid = F["s6n"], F["s6mid"]
    n_el = len(G["emid"])
    stress = np.zeros((n_el*n_depth, 6)); strain = np.zeros((n_el*n_depth, 6))
    U = np.zeros((n_el*n_depth, 3))
    for e in range(n_el):
        ln = layups[e]; h = float(G["h"][e])
        c0, c1 = int(rc[e, 0]), int(rc[e, 1])
        s6 = s6mid[e].copy()
        for row in (2, 5):                # contour-derivative rows: nodal interp
            v0 = s6n[c0, row] if np.isfinite(s6n[c0, row]) else s6mid[e, row]
            v1 = s6n[c1, row] if np.isfinite(s6n[c1, row]) else s6mid[e, row]
            s6[row] = 0.5*(v0 + v1)
        umid = 0.5*(wn[c0, 0:3] + wn[c1, 0:3])          # mid-arc warping
        om = 0.5*(wn[c0, 3:6] + wn[c1, 3:6])            # director rotation
        e3 = np.array([0.0, G["nvec"][e, 0], G["nvec"][e, 1]])
        for k in range(n_depth):
            z = (zeta[k] - frac)*h        # depth from the reference surface
            i = e*n_depth + k
            Gam, Sig, _ply = msgrm_strain_at_depth(warpM[ln], z, s6,
                                                   F["dE1"][e], F["dE2"][e],
                                                   frame=frame)
            stress[i] = np.asarray(Sig, float)
            strain[i] = np.asarray(Gam, float)
            U[i] = umid + z*np.cross(om, e3)            # + z (omega x e3)
    if state is not None:
        # total displacement = macro rigid motion + SG warping (beam axis y1)
        y = np.column_stack([np.zeros(len(pts)), pts])
        U = U + state["u"] + y @ (state["C"] - np.eye(3)).T

    R = {"stress": stress, "strain": strain}
    ez = np.repeat(np.arange(n_el), n_depth)
    tab = np.column_stack([ez, np.tile(G["zeta"], n_el), pts,
                           G["z"].ravel(), R["stress"], R["strain"], U])
    fmt = "%6d %6.3f %13.6e %13.6e %13.6e " + " ".join(["%14.6e"]*15)
    head = ("RM two-step shell dehomogenization, %s frame\n"
            "macro beam strain [ext sh2 sh3 twist bend2 bend3] = %s\n"
            "elem  zeta(0=OML,1=IML)  y2(m)  y3(m)  z(m, from the %s"
            " reference)\n  S11 S22 S33 S23 S13 S12 (Pa)  "
            "E11 E22 E33 2E23 2E13 2E12  u1 u2 u3 (m)"
            % (frame, np.array2string(np.asarray(eps, float), precision=6),
               B.get("ref", "center")))
    # junction-aware recovery: the two EXTRA columns go at the END, so every
    # existing column index is untouched and `junction: off` is byte-identical
    jflag = None
    tier = str(junction).strip().lower()
    if tier != "off":
        from .sg_dehom_junction import (census_text, flag_recovery_points,
                                        write_sidecar)
        JR = flag_recovery_points(B, G=G, k_bl=float(junction_bl),
                                  ang_tol_deg=float(junction_ang))
        jflag, jdist = JR["jflag"], JR["jdist"]
        if tier == "exclude":
            tab[jflag == 3, 5:] = np.nan       # the 15 FIELD columns only --
        tab = np.column_stack([tab, jflag, jdist])   # elem/zeta/y2/y3/z stay
        fmt += " %5d %13.6e"
        head += ("\n  jflag (0 clean, 1 near, 2 overlapped-owned,"
                 " 3 duplicate)  jdist (|p-J| / max t there)")
        if tier == "exclude":
            head += ("\n  NaN fields = a jflag-3 duplicate station, dropped"
                     " by junction: exclude")
        write_sidecar(out_base + ".junc", B, JR, tier=tier,
                      ang_tol_deg=float(junction_ang))
        print(census_text(JR["census"], tier))
    np.savetxt(out_base + ".txt", tab, fmt=fmt, header=head)

    # exploded-mesh VTK: one quad per (element, depth band), each carrying the
    # stress of its own recovery station as CELL_DATA -- never node-averaged
    emid, nvec, h, frac = G["emid"], G["nvec"], G["h"], G["frac"]
    corners = np.asarray(B["corners"]); rc = np.asarray(B["red_cells"])
    edge = np.arange(n_depth + 1)/n_depth
    P = np.zeros((4*n_el*n_depth, 3))
    for e in range(n_el):
        p0, p1 = corners[rc[e, 0]], corners[rc[e, 1]]
        for k in range(n_depth):
            z0 = (edge[k] - frac)*h[e]
            z1 = (edge[k + 1] - frac)*h[e]
            q = 4*(e*n_depth + k)
            P[q + 0, 1:] = p0 + z0*nvec[e]
            P[q + 1, 1:] = p1 + z0*nvec[e]
            P[q + 2, 1:] = p1 + z1*nvec[e]
            P[q + 3, 1:] = p0 + z1*nvec[e]
    nc = n_el*n_depth
    with open(out_base + ".vtk", "w") as f:
        f.write("# vtk DataFile Version 3.0\nOpenSG msg-shell RM dehom\n"
                "ASCII\nDATASET UNSTRUCTURED_GRID\nPOINTS %d float\n" % (4*nc))
        for p in P:
            f.write("%.8e %.8e %.8e\n" % (p[0], p[1], p[2]))
        f.write("\nCELLS %d %d\n" % (nc, 5*nc))
        for c in range(nc):
            f.write("4 %d %d %d %d\n" % (4*c, 4*c + 1, 4*c + 2, 4*c + 3))
        f.write("\nCELL_TYPES %d\n" % nc)
        f.write("".join(["9\n"]*nc))
        f.write("\nCELL_DATA %d\n" % nc)
        for j, nm in enumerate(("S11", "S22", "S33", "S23", "S13", "S12")):
            f.write("SCALARS %s float 1\nLOOKUP_TABLE default\n" % nm)
            f.write("".join("%.8e\n" % v for v in R["stress"][:, j]))
        if jflag is not None:
            # the junction bookkeeping travels with the cloud: threshold the
            # mesh on jflag (== 3 is the duplicate cover).  jdist is written
            # with -1 where there is no junction in the section at all --
            # legacy VTK ASCII readers do not take `inf`.
            f.write("SCALARS jflag int 1\nLOOKUP_TABLE default\n")
            f.write("".join("%d\n" % v for v in jflag))
            f.write("SCALARS jdist float 1\nLOOKUP_TABLE default\n")
            f.write("".join("%.8e\n" % v for v in
                            np.where(np.isfinite(jdist), jdist, -1.0)))
