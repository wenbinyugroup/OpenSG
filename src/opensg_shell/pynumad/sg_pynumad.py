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
import numpy as np

from ..windio.sg_windio import WindIOBladeV1, _interp
from ..windio.sg_props import station_timo as _station_timo
from ..windio.sg_windio import generate_cross_sections as _generate


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


def station_timo(blade_yaml, station, mesh_size=0.01, reference="center",
                 out_dir=".", prefix=None, xml=False, view=False):
    """Timoshenko 6x6 of one pyNuMAD blade station (steps 1 + 2 fused).

    The pynumad twin of windio's station_timo: same emitted artifacts
    (<tag>_shell.yaml, <tag>_shell_Timo.out, VABS-layout <tag>.K, nothing
    else), built with the PyNuMADBlade reader so the width-placed layers are
    in the laminate.

    In:
        blade_yaml: str, pyNuMAD blade yaml.
        station: str | int | float, st-id token (see resolve_station).
        mesh_size, reference, out_dir, prefix: as windio.station_timo.
    Out:
        the windio station_timo dict + "r" float + "file_K"/"file_M" the
        blade file's own 6x6 at r ((6,6) or None).
    """
    blade = load_blade_pynumad(blade_yaml)
    r = resolve_station(blade, station)
    P = _station_timo(blade_yaml, r, mesh_size=mesh_size, reference=reference,
                      out_dir=out_dir, prefix=prefix, blade=blade,
                      xml=xml, view=view)
    ref = file_six_by_six(blade, r)
    P.update(r=r, file_K=None if ref is None else ref[0],
             file_M=None if ref is None else ref[1])
    return P


def generate_cross_sections(blade_yaml, out_dir="cross_sections",
                            stations="airfoil", mesh_size=0.01,
                            reference="center", xml=True, plots=True,
                            prefix=None, verbose=True):
    """All pyNuMAD blade stations -> 1-D shell SG yamls (+ XML byproduct).

    The pynumad twin of windio's generate_cross_sections (same outputs,
    PyNuMADBlade reader).

    In/Out: as windio.generate_cross_sections.
    """
    return _generate(blade_yaml, out_dir=out_dir, stations=stations,
                     mesh_size=mesh_size, reference=reference, xml=xml,
                     plots=plots, prefix=prefix, verbose=verbose,
                     blade=load_blade_pynumad(blade_yaml))
