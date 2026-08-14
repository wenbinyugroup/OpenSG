"""opensg_shell.sg_blade -- the editable Blade object (pyNuMAD-style workflow).

Modeled on pynumad.objects.blade.Blade: ONE object reads the blade file
(windIO v1/v2 or the pyNuMAD dialect), exposes the WHOLE definition as
editable plain-python structures, and computes cross-sections and Timoshenko
properties FROM THE CURRENT STATE -- an optimization loop edits the blade
object, never a yaml file:

    from opensg_shell import Blade

    b = Blade("IEA-15-240-RWT.yaml")
    b.scale_layer_thickness("Spar_Cap_SS", 1.2)     # design move
    b.update_blade()                                # re-sync derived views
    R = b.timo(4)                                   # st-id (0-based) or r
    print(R["Timo"], R["Mass"])

    rows = b.timo_all()                             # every station + tables

Editing contract (mirrors pyNuMAD's definition -> update_blade -> analysis):
- `blade.raw` is the full parsed yaml dict; `blade.layers[name]`,
  `blade.webs`, `blade.materials[name]`, `blade.airfoils[name]`,
  `blade.chord`, `blade.twist`, `blade.offset` are REFERENCES into it --
  editing them (or `raw` directly) edits the definition.
- VALUE edits (thickness values, chord values, material constants)
  propagate immediately; after STRUCTURAL edits (adding/removing layers,
  webs, materials, airfoils) call `update_blade()` to rebuild the views.
  Calling `update_blade()` after every edit is always safe.
- The convenience helpers (scale_layer_thickness, set_material, ...) cover
  the common optimization moves.

Every compute goes through the SAME production route as the terminal
commands (`opensg pynumad` / `opensg windio_st`): build_cross_section ->
emit_shell_yaml -> beam_props (RM ring + ring mass matrix), station
artifacts under `blade.workdir` (a fresh temp dir unless given).
"""
import os
import tempfile

import numpy as np
import yaml

try:
    from yaml import CSafeLoader as _Loader
except ImportError:
    from yaml import SafeLoader as _Loader

from .windio.sg_windio import WindIOBlade, build_cross_section, emit_shell_yaml
from .windio.sg_props import station_timo as _station_timo
from .pynumad.sg_pynumad import (PyNuMADBlade, resolve_station,
                                 file_six_by_six, _append_file_crosscheck)


class Blade:
    """The editable blade definition + its cross-section/beam analyses.

    In (constructor):
        path: str | None, blade yaml (windIO v1/v2 or pyNuMAD dialect);
            None = empty object, call read_yaml later.
        reference: "oml" | "center", shell reference surface for every
            emitted station (default oml, the pynumad-route default).
        mesh_size: float, target element arc length / chord.
        workdir: str | None, station artifact folder (None = fresh temp dir).
    Out:
        Blade with `raw`, `dialect`, the definition views (layers, webs,
        materials, airfoils, chord, twist, offset) and the compute methods.
    """

    def __init__(self, path=None, reference="oml", mesh_size=0.01,
                 workdir=None):
        self.reference = reference
        self.mesh_size = mesh_size
        self.workdir = workdir or tempfile.mkdtemp(prefix="opensg_blade_")
        self.raw = None
        self.dialect = None
        self._path = None
        self._reader = None
        if path is not None:
            self.read_yaml(path)

    # ------------------------------------------------------------------ I/O
    def read_yaml(self, path):
        """Parse the blade yaml into the editable definition.

        In:  path str -- windIO v1/v2 or pyNuMAD blade yaml.
        Out: self (views rebuilt via update_blade)."""
        self.raw = yaml.load(open(path), Loader=_Loader)
        self._path = str(path)
        return self.update_blade()

    def update_blade(self):
        """Rebuild the reader + definition views from `raw` (pyNuMAD's
        definition -> update_blade step).

        Call after structural edits (add/remove layers, webs, materials,
        airfoils); value edits propagate without it.

        In:  -- (uses self.raw)
        Out: self."""
        bl = self.raw["components"]["blade"]
        self.dialect = "v1" if "outer_shape_bem" in bl else "v2"
        # the readers BIND to self.raw (dict-in constructors), so definition
        # edits reach every later cross-section/homogenization
        self._reader = (PyNuMADBlade(self.raw) if self.dialect == "v1"
                        else WindIOBlade(self.raw))
        osh = self._reader.osh
        self.chord = osh["chord"]
        self.twist = osh["twist"]
        self.offset = osh.get("pitch_axis" if self.dialect == "v1"
                              else "section_offset_y")
        src = (self._reader._layers if self.dialect == "v1"
               else self._reader.st["layers"])
        self.layers = {L["name"]: L for L in src}
        self.webs = (self._reader._webs if self.dialect == "v1"
                     else self._reader.st["webs"])
        self.materials = self._reader.mats
        self.airfoils = self._reader.afs
        return self

    # ------------------------------------------------------------- stations
    @property
    def stations(self):
        """The blade's own spanwise station list (list of float r)."""
        return self._reader.stations()

    def resolve(self, st):
        """st-id token -> span r (0-based index | float r | str)."""
        return resolve_station(self._reader, st)

    def _stem(self):
        return (os.path.splitext(os.path.basename(self._path))[0]
                if self._path else "blade")

    # -------------------------------------------------------------- compute
    def cross_section(self, st):
        """Resolved 2-D cross-section dict of one station (current state).

        In:  st -- st-id token (0-based index | r | str).
        Out: dict from windio.build_cross_section."""
        return build_cross_section(self._reader, self.resolve(st),
                                   mesh_size=self.mesh_size)

    def write_station_yaml(self, st, path=None):
        """Emit the 1-D shell SG yaml of one station from the current state.

        In:  st -- st-id token; path str | None (None = workdir/<tag>).
        Out: dict(n_nodes, n_elems, n_sets, n_webs, out, r)."""
        r = self.resolve(st)
        tag = "%s_r%04d" % (self._stem(), round(r * 1000))
        path = path or os.path.join(self.workdir, tag + "_shell.yaml")
        info = emit_shell_yaml(self.cross_section(r), path,
                               reference=self.reference)
        info["r"] = r
        return info

    def timo(self, st, xml=False, view=False):
        """Timoshenko 6x6 + mass 6x6 of one station FROM THE CURRENT STATE.

        The same production route as `opensg pynumad <yaml> <st-id>`:
        station yaml + _Timo.out + VABS-layout .out land in self.workdir;
        when the file carries elastic_properties_mb, the cross-check block
        is appended inside the .out and returned as file_K/file_M.

        In:  st -- st-id token; xml/view bool -- per-station opt-ins.
        Out: dict("Timo" (6,6), "Mass" (6,6), "info", "bundle", "k_file",
             "yaml", "tag", "mesh", "chord", "twist", "r",
             "file_K"/"file_M" (6,6) | None)."""
        r = self.resolve(st)
        P = _station_timo(self._path or "blade.yaml", r,
                          mesh_size=self.mesh_size, reference=self.reference,
                          out_dir=self.workdir, prefix=self._stem(),
                          blade=self._reader, xml=xml, view=view,
                          k_ext=".out")
        ref = file_six_by_six(self._reader, r)
        P.update(r=r, file_K=None if ref is None else ref[0],
                 file_M=None if ref is None else ref[1])
        if ref is not None:
            _append_file_crosscheck(P["k_file"], P["Timo"],
                                    float(P["info"]["mpus"]), ref[0], ref[1])
        return P

    def timo_all(self, xml=False, view=False, verbose=True):
        """Timoshenko sweep over every station of the current state.

        Writes the spanwise <stem>_{stations,timo_by_r,mass_by_r}.dat tables
        into self.workdir next to the per-station artifacts.

        In:  xml/view bool -- per-station opt-ins; verbose bool -- r lines.
        Out: list of the per-station timo() dicts (station order)."""
        out, rows, rk, rm = [], [], [], []
        for r in self.stations:
            P = self.timo("%.10f" % r, xml=xml, view=view)
            if verbose:
                print(" r = %.4f" % P["r"])
            m = P["mesh"]
            rows.append([P["r"], P["chord"], P["twist"], m["n_nodes"],
                         m["n_elems"], m["n_sets"], m["n_webs"]])
            rk.append(np.r_[P["r"], np.asarray(P["Timo"]).flatten()])
            rm.append(np.r_[P["r"], P["info"]["mpus"],
                            np.asarray(P["Mass"]).flatten()])
            out.append(P)
        stem = self._stem()
        np.savetxt(os.path.join(self.workdir, stem + "_stations.dat"),
                   np.array(rows), fmt="%.4f %10.4f %10.6f %6d %6d %3d %3d",
                   header="r chord[m] twist[windIO unit] n_nodes n_elems"
                          " n_sets n_webs")
        np.savetxt(os.path.join(self.workdir, stem + "_timo_by_r.dat"),
                   np.array(rk), fmt="%.8e",
                   header="r then Timoshenko 6x6 row-major (VABS order)")
        np.savetxt(os.path.join(self.workdir, stem + "_mass_by_r.dat"),
                   np.array(rm), fmt="%.8e",
                   header="r mass_per_span then mass 6x6 row-major"
                          " (VABS frame)")
        return out

    # ----------------------------------------------------- editing helpers
    def scale_layer_thickness(self, name, factor):
        """Multiply one layer's thickness distribution by a factor.

        In:  name str -- layer name (see blade.layers); factor float.
        Out: self."""
        t = self.layers[name]["thickness"]
        if isinstance(t, dict):
            t["values"] = [float(v) * float(factor) for v in t["values"]]
        else:
            self.layers[name]["thickness"] = float(t) * float(factor)
        return self

    def set_layer_thickness(self, name, values, grid=None):
        """Replace one layer's thickness distribution.

        In:  name str; values list[float]; grid list[float] | None (None =
             keep the existing grid; lengths must match it).
        Out: self."""
        t = self.layers[name]["thickness"]
        if not isinstance(t, dict):
            self.layers[name]["thickness"] = t = {"grid": list(grid or []),
                                                  "values": []}
        if grid is not None:
            t["grid"] = [float(g) for g in grid]
        assert len(values) == len(t["grid"]), \
            "values length must match the thickness grid"
        t["values"] = [float(v) for v in values]
        return self

    def set_material(self, name, **props):
        """Update material constants in place (E, G, nu, rho, ...).

        In:  name str -- material name; props -- keys of the material card.
        Out: self."""
        self.materials[name].update(props)
        return self

    def scale_chord(self, factor):
        """Multiply the chord distribution by a factor.

        In:  factor float.
        Out: self."""
        self.chord["values"] = [float(v) * float(factor)
                                for v in self.chord["values"]]
        return self

    def set_twist(self, values, grid=None):
        """Replace the twist distribution (file's own unit).

        In:  values list[float]; grid list[float] | None (None = keep).
        Out: self."""
        if grid is not None:
            self.twist["grid"] = [float(g) for g in grid]
        assert len(values) == len(self.twist["grid"]), \
            "values length must match the twist grid"
        self.twist["values"] = [float(v) for v in values]
        return self
