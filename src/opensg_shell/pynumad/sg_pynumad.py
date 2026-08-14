"""pyNuMAD blade yaml -> OpenSG 1-D shell SG, a SEPARATE dialect from windio.

pyNuMAD exports (e.g. sandialabs/pyNuMAD examples/example_data/IEA-15-240-RWT)
look like windIO v1 (outer_shape_bem / internal_structure_2d_fem) but place
some skin layers with pyNuMAD's width forms instead of explicit arc bounds:

    start_nd_arc + width        TE_reinforcement_SS (width [m] toward the LE)
    end_nd_arc + width          TE_reinforcement_PS (width [m] from the end)
    midpoint_nd_arc + width     LE-style reinforcements (half each side)

The plain windio v1 reader keeps only explicit start/end bounds, so those
layers would drop out of the laminate (measured on IEA-15: EI3 up to -84%,
mass/span -19%).  PyNuMADBlade resolves the width forms against the section
perimeter and WARNS (once per layer) on anything still unresolvable -- material
is never dropped silently.  The file's own `elastic_properties_mb.six_x_six`
block (pyNuMAD/WISDEM beam properties, 3 = axial frame) is exposed for the
station cross-check the `opensg pynumad` command prints.
"""
import os
import time

import numpy as np

import yaml as _yaml

from .sg_mesh import (WindIOBlade, WindIOBladeV1, _interp,
                      build_cross_section)


class PyNuMADBlade(WindIOBladeV1):
    """windIO-v1-shaped reader with the pyNuMAD layer-placement forms."""

    def __init__(self, yaml_path):
        super().__init__(yaml_path)
        self._warned = set()

    def _perim(self, r):
        """Section perimeter [m] at span r (chord-scaled blended contour).

        In:  r float, non-dimensional span.
        Out: float [m]."""
        xy = self.airfoil_coords(r) * (self.scalar("chord", r) or 1.0)
        return float(np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1])).sum())

    def layers_at(self, r, tol=1e-6):
        """Skin layers at span r, resolving pyNuMAD width-based placements.

        In:  r float; tol float, thickness cutoff [m].
        Out: list of dict(name, material, s, e, t, fiber, order) -- order =
             layer index = outer -> inner stacking; s/e always resolved."""
        out = []
        for idx, L in enumerate(self._layers):
            if L.get("web"):
                continue
            t = _interp(L.get("thickness"), r)
            if not t or t < tol:
                continue
            s = _interp(L.get("start_nd_arc"), r)
            e = _interp(L.get("end_nd_arc"), r)
            w = _interp(L.get("width"), r)
            m = _interp(L.get("midpoint_nd_arc"), r)
            if w is not None and w > 0 and (s is None or e is None):
                dw = w / self._perim(r)          # width [m] -> arc fraction
                if s is not None:
                    e = s + dw
                elif e is not None:
                    s = e - dw
                elif m is not None:
                    s, e = m - 0.5 * dw, m + 0.5 * dw
            if s is None or e is None:
                if L["name"] not in self._warned:
                    self._warned.add(L["name"])
                    print("[pynumad] layer %r: unsupported placement (no"
                          " resolvable start/end_nd_arc) -- layer SKIPPED"
                          % L["name"])
                continue
            s = float(np.clip(s, 0.0, 1.0)); e = float(np.clip(e, 0.0, 1.0))
            out.append(dict(name=L["name"], material=L["material"], s=s, e=e,
                            t=t, fiber=_interp(L.get("fiber_orientation"), r) or 0.0,
                            order=idx))
        return out


def load_blade_pynumad(path):
    """Read a pyNuMAD blade yaml.

    In:  path str, pyNuMAD blade yaml (windIO-v1 dialect).
    Out: PyNuMADBlade."""
    return PyNuMADBlade(path)


def load_blade(path):
    """Auto-detect the blade dialect and return the matching reader.

    In:  path str -- blade yaml: windIO v1 / pyNuMAD dialect
         (outer_shape_bem) or windIO v2 (outer_shape).
    Out: PyNuMADBlade (v1: the width-form placements resolved) |
         WindIOBlade (v2)."""
    bl = _yaml.safe_load(open(path))["components"]["blade"]
    return PyNuMADBlade(path) if "outer_shape_bem" in bl \
        else WindIOBlade(path)


def blade_stations(blade):
    """The blade's own airfoil station positions.

    In:  blade reader.
    Out: list of float r in [0, 1]."""
    return blade.stations()


def blade_length(blade):
    """Blade length [m] from the reference-axis polyline.

    In:  blade reader.
    Out: float [m]."""
    return blade.length()


def resolve_station(blade, token):
    """Turn the CLI st-id token into a span r.

    In:
        blade: PyNuMADBlade.
        token: str -- a 0-based int index into the blade's own station list,
            or a float span r in [0, 1] (a token with a decimal point).
    Out:
        float r.
    """
    rs = blade.stations()
    tok = str(token).strip()
    if "." not in tok:
        i = int(tok)
        if not 0 <= i < len(rs):
            raise SystemExit(
                "station id %d out of range -- this blade has %d stations:\n%s"
                % (i, len(rs), "\n".join("  %2d : r = %.4f" % (k, v)
                                         for k, v in enumerate(rs))))
        return float(rs[i])
    r = float(tok)
    if not 0.0 <= r <= 1.0:
        raise SystemExit("station r must be in [0, 1], got %r" % tok)
    return r


def file_six_by_six(blade, r):
    """The blade file's own beam matrices at span r, when it carries them.

    pyNuMAD/WISDEM files store `elastic_properties_mb.six_x_six` with 21
    upper-triangle values per grid point, in the blade frame with 3 = axial
    (diag order [GA3 GA2 EA EI3 EI2 GJ] against the VABS-frame diag via
    x_bd = (x3, -x2, x1)).

    In:
        blade: PyNuMADBlade; r: float span.
    Out:
        (K6, M6) float (6,6) file matrices at r, or None when the block is
        absent.
    """
    ep = blade.bl.get("elastic_properties_mb")
    if not ep or "six_x_six" not in ep:
        return None

    def mat(spec):
        g = np.asarray(spec["grid"], float)
        V = np.asarray(spec["values"], float)
        row = np.array([np.interp(r, g, V[:, j]) for j in range(V.shape[1])])
        M = np.zeros((6, 6))
        iu = np.triu_indices(6)
        M[iu] = row
        return M + M.T - np.diag(np.diag(M))
    sx = ep["six_x_six"]
    return mat(sx["stiff_matrix"]), mat(sx["inertia_matrix"])


# VABS index of each file-frame diagonal slot [11..66] -> [GA3 GA2 EA EI3 EI2 GJ]
FILE_DIAG_TO_VABS = (2, 1, 0, 5, 4, 3)


def pynumad_segments(blade, r):
    """The exact pyNuMAD 12-region contour segmentation at span r.

    Built from the same keypoint scheme pyNuMAD's stackdb uses
    (te,e,d,c,b,a,le,a,b,c,d,e,te from the LE band / spar cap / TE band
    placements); the lazy import avoids a module cycle with sg_blade.

    In:  blade PyNuMADBlade; r float span.
    Out: list of (s_a, s_b, laminate) for build_cross_section(segments=).
    """
    from .sg_blade import Definition, Geometry, KeyPoints, StackDatabase
    geo = Geometry().generate(blade)
    kp = KeyPoints().generate(Definition(blade, "v1"), geo, blade)
    return StackDatabase().generate(kp, blade).segments(r)


def _append_file_crosscheck(out_path, K6, mpus, file_K, file_M):
    """Append the elastic_properties_mb cross-check block to the station .out.

    The console stays lean (Timoshenko + mass matrices only); the diagonal
    comparison against the blade file's own pyNuMAD/WISDEM 6x6 lives in the
    stored record instead.

    In:
        out_path: str, the station VABS-layout .out file (appended in place).
        K6: (6,6) OpenSG Timoshenko; mpus: float mass/span [kg/m];
        file_K/file_M: (6,6) the blade file's stiffness/inertia at r.
    Out:
        None.
    """
    LBL = ("EA", "GA2", "GA3", "GJ", "EI2", "EI3")
    dv = np.diag(np.asarray(K6, float))
    df = np.diag(np.asarray(file_K, float))
    L = ["\n Cross-check vs the blade file's own elastic_properties_mb"
         " (pyNuMAD/WISDEM)\n"
         " ========================================================\n"]
    for b, v in enumerate(FILE_DIAG_TO_VABS):
        L.append("   %-3s  OpenSG %14.6E   file %14.6E   %+8.2f %%\n"
                 % (LBL[v], dv[v], df[b], 100.0 * (dv[v] / df[b] - 1.0)))
    mu_f = float(np.asarray(file_M, float)[0, 0])
    L.append("   %-3s  OpenSG %14.6E   file %14.6E   %+8.2f %%\n"
             % ("mu", mpus, mu_f, 100.0 * (mpus / mu_f - 1.0)))
    with open(out_path, "a") as f:
        f.write("".join(L))


def station_timo(blade_yaml, station, mesh_size=0.01,
                 out_dir=".", prefix=None, xml=False, view=False,
                 blade=None):
    """Timoshenko 6x6 of one pyNuMAD blade station, fully IN MEMORY.

    The whole station runs through the pynumad-owned chain
    (sg_mesh.build_cross_section with the 12-region segmentation ->
    sg_homo.timo_cs); the ONLY artifact is the VABS-layout <tag>.out
    record (with the elastic_properties_mb cross-check appended when the
    file carries the block).  The 1-D yaml emission of the old
    windio-backed route is gone -- that was windio's job, and this route
    does not need it.

    In:
        blade_yaml: str, pyNuMAD blade yaml.
        station: str | int | float, st-id token (see resolve_station).
        mesh_size: float, element arc length / chord.  The shell reference
            surface is ALWAYS "oml" on this route (no option); a center-
            reference section is an SG-engine study via
            emit_shell_yaml(..., reference="center").
        out_dir: str, folder of the <tag>.out record.
        prefix: str | None, tag prefix (None = the blade-file stem).
        xml, view: accepted for CLI compatibility, IGNORED (no yaml/XML/PNG
            artifacts on this route).
        blade: PyNuMADBlade | None -- a pre-loaded reader (sweep reuse).
    Out:
        dict: "Timo"/"Mass" (6,6), "info" mass/geometry dict, "k_file" str,
        "tag" str, "mesh" dict(n_nodes, n_elems, n_sets, n_webs), "chord",
        "twist", "r" float, "file_K"/"file_M" (6,6) | None.
    """
    from .sg_homo import timo_cs, write_vabs_k

    t0 = time.perf_counter()
    if blade is None:
        blade = load_blade(blade_yaml)
    r = resolve_station(blade, station)
    # the 12-region segmentation is the v1/pyNuMAD scheme; windIO v2
    # sections keep their data-driven breakpoints
    segs = pynumad_segments(blade, r) \
        if isinstance(blade, WindIOBladeV1) else None
    cs = build_cross_section(blade, r, mesh_size=mesh_size, segments=segs)
    K, M, info, arrays = timo_cs(cs, "oml")
    stem = prefix or os.path.splitext(os.path.basename(blade_yaml))[0]
    tag = "%s_r%04d" % (stem, round(r * 1000))
    os.makedirs(out_dir, exist_ok=True)
    k_file = os.path.join(out_dir, tag + ".out")
    write_vabs_k(k_file, K, M, info,
                 model="OpenSG msg-shell beam model (pynumad in-memory,"
                       " reference=oml)",
                 solve_time=time.perf_counter() - t0)
    P = dict(Timo=K, Mass=M, info=info, k_file=k_file, tag=tag, r=r,
             mesh=dict(n_nodes=len(arrays["rx"]),
                       n_elems=len(arrays["cells"]),
                       n_sets=len(arrays["sections"]),
                       n_webs=arrays["n_webs"]),
             chord=cs["chord"], twist=cs["twist"])
    ref6 = file_six_by_six(blade, r)
    P.update(file_K=None if ref6 is None else ref6[0],
             file_M=None if ref6 is None else ref6[1])
    if ref6 is not None:
        _append_file_crosscheck(k_file, K, float(info["mpus"]),
                                ref6[0], ref6[1])
    return P


def generate_cross_sections(blade_yaml, out_dir="cross_sections",
                            stations="airfoil", mesh_size=0.01,
                            xml=False, plots=False,
                            prefix=None, verbose=True):
    """Every pyNuMAD blade station, fully IN MEMORY -> .out + spanwise .dat.

    Per station: the VABS-layout <tag>.out record (station_timo); at the
    end the spanwise <stem>_{stations,timo_by_r,mass_by_r}.dat tables.
    The 1-D yaml / XML / PNG emission of the old windio-backed route is
    gone; xml/plots are accepted for compatibility and ignored.

    In:
        blade_yaml: str, pyNuMAD blade yaml.
        out_dir: str, output folder.
        stations: "airfoil" (the blade's own) | int N (uniform grid) |
            list of float r.
        mesh_size, prefix: as station_timo (reference surface always oml).
        verbose: bool, one "r = X" line per station.
    Out:
        list of the per-station station_timo dicts (station order).
    """
    blade = load_blade(blade_yaml)
    if stations == "airfoil":
        rs = blade.stations()
    elif isinstance(stations, int):
        rs = list(np.linspace(0.0, 1.0, stations))
    else:
        rs = [float(v) for v in stations]
    stem = prefix or os.path.splitext(os.path.basename(blade_yaml))[0]
    os.makedirs(out_dir, exist_ok=True)
    out, rows, rows_k, rows_m = [], [], [], []
    for r in rs:
        P = station_timo(blade_yaml, "%.10f" % r, mesh_size=mesh_size,
                         out_dir=out_dir, prefix=prefix, blade=blade)
        if verbose:
            print(" r = %.4f" % P["r"])
        m = P["mesh"]
        rows.append([P["r"], P["chord"], P["twist"], m["n_nodes"],
                     m["n_elems"], m["n_sets"], m["n_webs"]])
        rows_k.append(np.r_[P["r"], np.asarray(P["Timo"]).flatten()])
        rows_m.append(np.r_[P["r"], P["info"]["mpus"],
                            np.asarray(P["Mass"]).flatten()])
        out.append(P)
    np.savetxt(os.path.join(out_dir, stem + "_stations.dat"),
               np.array(rows), fmt="%.4f %10.4f %10.6f %6d %6d %3d %3d",
               header="r chord[m] twist[windIO unit] n_nodes n_elems"
                      " n_sets n_webs")
    np.savetxt(os.path.join(out_dir, stem + "_timo_by_r.dat"),
               np.array(rows_k), fmt="%.8e",
               header="r then Timoshenko 6x6 row-major (VABS order)")
    np.savetxt(os.path.join(out_dir, stem + "_mass_by_r.dat"),
               np.array(rows_m), fmt="%.8e",
               header="r mass_per_span then mass 6x6 row-major"
                      " (VABS frame)")
    return out
