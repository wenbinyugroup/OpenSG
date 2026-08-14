"""opensg_shell.sg_blade -- the editable Blade object (pyNuMAD-style workflow).

Modeled on pynumad.objects.blade.Blade: ONE object reads the blade file
(windIO v1/v2 or the pyNuMAD dialect), exposes the WHOLE definition as
editable plain-python structures, and computes cross-sections and Timoshenko
properties FROM THE CURRENT STATE -- an optimization loop edits the blade
object, never a yaml file:

    import opensg

    blade = opensg.read("IEA-15-240-RWT.yaml")
    K, M = blade(4)                                 # st-id (0-based) or r
    Ks, Ms = blade()                                # full spanwise sweep

    blade.scale_layer_thickness("Spar_Cap_SS", 1.2) # spanwise design move
    blade.update_blade()                            # re-sync derived views

    S = blade.section(4)                            # ONE station's stacks
    S.scale_thickness(1.3, region="LP_SPAR")        # per-section design move
    K2, M2 = blade(4)                               # honors the edit
    S.reset()                                       # back to the definition

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


REGION_NAMES = ("HP_TE_FLAT", "HP_TE_REINF", "HP_TE_PANEL", "HP_SPAR",
                "HP_LE_PANEL", "HP_LE", "LP_LE", "LP_LE_PANEL", "LP_SPAR",
                "LP_TE_PANEL", "LP_TE_REINF", "LP_TE_FLAT")
KEY_LABELS = ("te", "e", "d", "c", "b", "a", "le", "a", "b", "c", "d", "e",
              "te")


class Definition:
    """The EDITABLE blade definition (pyNuMAD's definition object).

    All attributes are live references into the parsed yaml dict -- editing
    them edits the blade; update_blade() re-syncs the derived objects.

    In (constructor): reader -- a dialect reader bound to the raw dict.
    Out: chord/twist/offset (grid dicts), layers {name: layer}, webs,
         materials {name: card}, airfoils {name: airfoil}."""

    def __init__(self, reader, dialect):
        osh = reader.osh
        self.chord = osh["chord"]
        self.twist = osh["twist"]
        self.offset = osh.get("pitch_axis" if dialect == "v1"
                              else "section_offset_y")
        src = reader._layers if dialect == "v1" else reader.st["layers"]
        self.layers = {L["name"]: L for L in src}
        self.webs = reader._webs if dialect == "v1" else reader.st["webs"]
        self.materials = reader.mats
        self.airfoils = reader.afs


class Geometry:
    """GENERATED station geometry accessor (pyNuMAD's geometry object).

    In (generate): reader; Out: stations list + contour/chord accessors."""

    def generate(self, reader):
        self._reader = reader
        self.stations = reader.stations()
        return self

    def chord(self, r):
        """Chord [m] at span r."""
        return self._reader.scalar("chord", r)

    def contour(self, r):
        """Chord-scaled section contour (n, 2) at span r."""
        return self._reader.airfoil_coords(r) * (self.chord(r) or 1.0)

    def le_arc(self, r):
        """Normalised arc position s of the LE (min-x point) at span r."""
        xy = self.contour(r)
        d = np.r_[0.0, np.cumsum(np.hypot(np.diff(xy[:, 0]),
                                          np.diff(xy[:, 1])))]
        return float(d[int(np.argmin(xy[:, 0]))] / d[-1])


class KeyPoints:
    """GENERATED pyNuMAD keypoints: te,e,d,c,b,a,le,a,b,c,d,e,te per station.

    The keypoint arcs come from the SAME definition placements pyNuMAD uses:
    a = LE band edges (LE_reinforcement span), b/c = spar-cap edges, d = TE
    band inner edge (TE_reinforcement span), e = TE band outer edge (the
    flat/TE side; e == te when there is no flatback).  All in the TE(0) ->
    suction -> LE -> pressure -> TE(1) normalised arc of the OpenSG contour.

    In (generate): definition, geometry, reader.
    Out: key_arcs(r) -> (13,) float s; regions(r) -> [(name, s_a, s_b)]."""

    _SIDES = {"HP": ("_PS", "hp"), "LP": ("_SS", "lp")}

    def generate(self, definition, geometry, reader):
        self._def = definition
        self._geo = geometry
        self._reader = reader
        return self

    def _layer_span(self, name_frags, r):
        """Resolved (s, e) of the first present layer matching any fragment."""
        for L in self._reader.layers_at(r):
            nm = L["name"].lower()
            if any(f in nm for f in name_frags):
                return float(L["s"]), float(L["e"])
        return None

    def key_arcs(self, r):
        """The 13 keypoint arc positions s at span r (HP=pressure side first
        in pyNuMAD label order te..le..te; here expressed in OpenSG s where
        s=0 is the TE on the suction (LP) side).

        In:  r float span.
        Out: (13,) float s, ordered per KEY_LABELS (HP te -> LP te)."""
        le = self._geo.le_arc(r)
        sp_lp = self._layer_span(("spar_cap_ss",), r)
        sp_hp = self._layer_span(("spar_cap_ps",), r)
        leb = self._layer_span(("le_reinf",), r)
        te_lp = self._layer_span(("te_reinforcement_ss",), r)
        te_hp = self._layer_span(("te_reinforcement_ps",), r)

        def lo(x, d):
            return d if x is None else min(x)

        def hi(x, d):
            return d if x is None else max(x)

        a_lp, a_hp = lo(leb, le), hi(leb, le)
        b_lp, c_lp = (hi(sp_lp, le), lo(sp_lp, le))
        b_hp, c_hp = (lo(sp_hp, le), hi(sp_hp, le))
        e_lp, d_lp = (lo(te_lp, 0.0), hi(te_lp, 0.0))
        e_hp, d_hp = (hi(te_hp, 1.0), lo(te_hp, 1.0))
        # pyNuMAD's keypoint clamps (keypoints.generate), expressed as
        # distance-from-LE fractions of each surface arc: a in [1%, 10%],
        # b >= 15%, c <= 80%, d in [85%, 98% HP / 96% LP]
        L_lp, L_hp = le, 1.0 - le

        def cl(dist, lo_f, hi_f, L):
            return min(max(dist, lo_f * L), hi_f * L)

        a_lp = le - cl(le - a_lp, 0.01, 0.10, L_lp)
        a_hp = le + cl(a_hp - le, 0.01, 0.10, L_hp)
        b_lp = le - max(le - b_lp, 0.15 * L_lp)
        b_hp = le + max(b_hp - le, 0.15 * L_hp)
        c_lp = le - min(le - c_lp, 0.80 * L_lp)
        c_hp = le + min(c_hp - le, 0.80 * L_hp)
        d_lp = le - cl(le - d_lp, 0.85, 0.96, L_lp)
        d_hp = le + cl(d_hp - le, 0.85, 0.98, L_hp)
        return np.array([1.0, e_hp, d_hp, c_hp, b_hp, a_hp, le,
                         a_lp, b_lp, c_lp, d_lp, e_lp, 0.0])

    def regions(self, r):
        """The 12 canonical regions at span r (zero-width ones included).

        In:  r float span.
        Out: list of (name, s_a, s_b) with s_a <= s_b, REGION_NAMES order."""
        k = self.key_arcs(r)
        bounds = [(k[1], k[0]), (k[2], k[1]), (k[3], k[2]), (k[4], k[3]),
                  (k[5], k[4]), (k[6], k[5]), (k[7], k[6]), (k[8], k[7]),
                  (k[9], k[8]), (k[10], k[9]), (k[11], k[10]),
                  (k[12], k[11])]
        return [(nm, float(min(a, b)), float(max(a, b)))
                for nm, (a, b) in zip(REGION_NAMES, bounds)]


class StackDatabase:
    """GENERATED per-region stacks (pyNuMAD's stackdb).

    In (generate): keypoints, reader.
    Out: stacks(r) -> {region: laminate tuple ((mat, t, ang) outer->inner)};
         swstacks(r) -> [web laminate tuples]; segments(r) -> the
         build_cross_section segments list."""

    def generate(self, keypoints, reader):
        self._kp = keypoints
        self._reader = reader
        # pyNuMAD component classes are fixed for the WHOLE blade: any layer
        # whose arc span reaches >= 0.9 somewhere is a full-surface 'shell'
        # component (te..te in every region at every station)
        self._full = set()
        for rs in reader.stations():
            for L in reader.layers_at(rs):
                if max(L["s"], L["e"]) - min(L["s"], L["e"]) >= 0.9:
                    self._full.add(L["name"])
        return self

    def _ply_quantize(self, name, mat, t_resolved, r):
        """pyNuMAD ply quantization: n = round(interp(grid,
        round(values*1000/layerthickness), r)), thickness = n plies.

        In:  name/mat str; t_resolved float [m] fallback; r float span.
        Out: float thickness [m] (0.0 when the count rounds to zero)."""
        ply_mm = 1000.0 * float(self._kp._def.materials
                                .get(mat, {}).get("ply_t") or 0.001)
        raw = self._kp._def.layers.get(name, {}).get("thickness")
        if isinstance(raw, dict) and "grid" in raw:
            counts = np.round(np.asarray(raw["values"], float) * 1000.0
                              / ply_mm)
            n = float(np.round(np.interp(r, np.asarray(raw["grid"], float),
                                         counts)))
        else:
            n = float(np.round(t_resolved * 1000.0 / ply_mm))
        return n * ply_mm / 1000.0

    _ROLES = (("te_reinforcement_ss", ("LP_TE_REINF", "LP_TE_FLAT")),
              ("te_reinforcement_ps", ("HP_TE_REINF", "HP_TE_FLAT")),
              ("spar_cap_ss", ("LP_SPAR",)),
              ("spar_cap_ps", ("HP_SPAR",)),
              ("le_reinf", ("HP_LE", "LP_LE")))

    def _laminate_at(self, r, a, b, region=None):
        # pyNuMAD component-to-region assignment: full-surface 'shell'
        # classes sit everywhere; PLACED components (caps, reinforcements)
        # belong ONLY to their designated regions; everything else (fillers)
        # by overlap with the region interval (boundary touches excluded)
        cov = []
        for L in self._reader.layers_at(r):
            nm = L["name"].lower()
            role = next((regs for frag, regs in self._ROLES if frag in nm),
                        None)
            if L["name"] in self._full:
                ok = True
            elif role is not None:
                ok = region in role
            else:
                lo, hi = min(L["s"], L["e"]), max(L["s"], L["e"])
                ok = min(hi, b) - max(lo, a) > 1e-4
            if ok:
                cov.append(L)
        cov.sort(key=lambda L: L["order"])
        out = []
        for L in cov:
            t = self._ply_quantize(L["name"], L["material"], L["t"], r)
            if t > 1e-9:
                out.append((L["material"], round(t, 8),
                            round(L["fiber"], 3)))
        return tuple(out)

    def stacks(self, r):
        """{region name: laminate} at span r (region-midpoint coverage)."""
        out = {}
        for nm, a, b in self._kp.regions(r):
            out[nm] = self._laminate_at(r, a, b, nm) if b - a > 1e-9 \
                else tuple()
        # pyNuMAD extends the TE band component into the TE flat region
        for side in ("HP", "LP"):
            if not out[side + "_TE_FLAT"]:
                out[side + "_TE_FLAT"] = out[side + "_TE_REINF"]
        return out

    def swstacks(self, r):
        """Web laminates at span r ([(mat, t, ang) ...] per web)."""
        out = []
        for w in self._reader.webs_at(r):
            if not w["layers"]:
                continue
            lam = []
            for L in w["layers"]:
                t = self._ply_quantize(L["name"], L["material"], L["t"], r)
                if t > 1e-9:
                    lam.append((L["material"], round(t, 8),
                                round(L["fiber"], 3)))
            out.append(tuple(lam))
        return out

    def segments(self, r):
        """The 12-region contour segmentation for build_cross_section.

        In:  r float span.
        Out: list of (s_a, s_b, laminate) with zero-width regions dropped,
             ordered by arc position."""
        segs = [(a, b, self._laminate_at(r, a, b, nm))
                for nm, a, b in self._kp.regions(r) if b - a > 1e-9]
        return sorted(segs, key=lambda t: t[0])


class MaterialDatabase:
    """GENERATED material card view (pyNuMAD's materialdb).

    In (generate): definition.  Out: dict-like name -> card (live refs)."""

    def generate(self, definition):
        self.materials = definition.materials
        return self

    def __getitem__(self, name):
        return self.materials[name]

    def __contains__(self, name):
        return name in self.materials


class Section:
    """One station's RESOLVED layup, user-editable (per-station override).

    Creating a Section registers it on the blade: every later
    cross_section/timo at this station computes from the EDITED stacks until
    reset().  The stacks are captured through the same resolution
    (ply-quantized) the un-edited route uses, so an untouched Section
    reproduces the baseline digit-for-digit; edited thicknesses are used
    verbatim (no re-quantization).

    In (constructor):
        blade: Blade (v1/pyNuMAD dialect).
        r: float, non-dimensional span of the station.
    Out:
        .stacks {region: [[mat, t, ang], ...] outer->inner} -- editable;
        .webs   [[[mat, t, ang], ...] per web]              -- editable;
        edit helpers scale_thickness / set_ply / add_ply / drop_ply / reset.
    """

    def __init__(self, blade, r):
        self.blade = blade
        self.r = float(r)
        kp = blade.keypoints
        sd = blade.stackdb
        # capture per REGION exactly as stackdb.segments resolves it (no
        # TE_FLAT fallback), so an un-edited Section mirrors the baseline
        self.stacks = {nm: [list(p) for p in sd._laminate_at(self.r, a, b,
                                                             nm)]
                       for nm, a, b in kp.regions(self.r)}
        # webs are captured from the cross-section's OWN resolution (before
        # this override registers), so an un-edited web repoints to the
        # identical laminate set and the baseline yaml is unchanged
        cs = blade.cross_section(self.r)
        by_id = {v: k for k, v in cs["laminates"].items()}
        self.webs = [[list(p) for p in by_id[w["lam"]]] for w in cs["webs"]]
        blade._overrides[self._key(self.r)] = self

    @staticmethod
    def _key(r):
        return round(float(r), 9)

    # ------------------------------------------------------------- editing
    def _lam_ref(self, region):
        """The editable ply list: region name (str) or web index (int)."""
        return (self.stacks[region] if isinstance(region, str)
                else self.webs[int(region)])

    def scale_thickness(self, factor, region=None, web=None):
        """Scale ply thicknesses of one region, one web, or the whole section.

        In:  factor float; region str | None; web int | None (both None =
             every skin region and every web).
        Out: self."""
        if region is not None:
            tgt = [self.stacks[region]]
        elif web is not None:
            tgt = [self.webs[int(web)]]
        else:
            tgt = list(self.stacks.values()) + self.webs
        for lam in tgt:
            for p in lam:
                p[1] = float(p[1]) * float(factor)
        return self

    def set_ply(self, region, ply, material=None, thickness=None,
                angle=None):
        """Update one ply in place.

        In:  region str (region name) | int (web index); ply int (0 =
             outermost); material str | None; thickness float [m] | None;
             angle float [deg] | None (None = keep).
        Out: self."""
        p = self._lam_ref(region)[int(ply)]
        if material is not None:
            p[0] = str(material)
        if thickness is not None:
            p[1] = float(thickness)
        if angle is not None:
            p[2] = float(angle)
        return self

    def add_ply(self, region, material, thickness, angle=0.0, index=None):
        """Insert a ply (default innermost).

        In:  region str | int; material str; thickness float [m]; angle
             float [deg]; index int | None (None = append inside).
        Out: self."""
        lam = self._lam_ref(region)
        ply = [str(material), float(thickness), float(angle)]
        lam.insert(len(lam) if index is None else int(index), ply)
        return self

    def drop_ply(self, region, ply):
        """Remove one ply.

        In:  region str | int; ply int index.
        Out: self."""
        del self._lam_ref(region)[int(ply)]
        return self

    def reset(self):
        """Drop this override: the station computes from the definition again.

        In:  --
        Out: self (deregistered)."""
        self.blade._overrides.pop(self._key(self.r), None)
        return self

    # -------------------------------------------------- compute resolution
    def _laminate(self, region):
        return tuple((str(m), round(float(t), 8), round(float(a), 3))
                     for m, t, a in self.stacks[region] if float(t) > 1e-9)

    def segments(self):
        """The contour segmentation from the EDITED stacks (baseline arcs).

        In:  --
        Out: list of (s_a, s_b, laminate tuple), stackdb.segments layout."""
        kp = self.blade.keypoints
        segs = [(a, b, self._laminate(nm))
                for nm, a, b in kp.regions(self.r) if b - a > 1e-9]
        return sorted(segs, key=lambda t: t[0])

    def web_laminates(self):
        """The edited web laminates, resolution-ready.

        In:  --
        Out: list of laminate tuples (web order; empty tuple = unchanged)."""
        return [tuple((str(m), round(float(t), 8), round(float(a), 3))
                      for m, t, a in lam if float(t) > 1e-9)
                for lam in self.webs]


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
        self._overrides = {}
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
        # the pyNuMAD chain: definition -> geometry -> keypoints -> stackdb
        # -> materialdb (all views over the SAME raw dict)
        self.definition = Definition(self._reader, self.dialect)
        self.geometry = Geometry().generate(self._reader)
        self.keypoints = KeyPoints().generate(self.definition, self.geometry,
                                              self._reader)
        self.stackdb = StackDatabase().generate(self.keypoints, self._reader)
        self.materialdb = MaterialDatabase().generate(self.definition)
        # flat aliases (kept for the initial Blade API)
        self.chord = self.definition.chord
        self.twist = self.definition.twist
        self.offset = self.definition.offset
        self.layers = self.definition.layers
        self.webs = self.definition.webs
        self.materials = self.definition.materials
        self.airfoils = self.definition.airfoils
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

        pyNuMAD dialect: the contour carries the EXACT pyNuMAD 12-region
        segmentation (stackdb.segments); v2 windIO keeps the data-driven
        breakpoints.

        In:  st -- st-id token (0-based index | r | str).
        Out: dict from windio.build_cross_section."""
        r = self.resolve(st)
        ov = self._overrides.get(Section._key(r))
        if self.dialect != "v1":
            segs = None
        elif ov is not None:
            segs = ov.segments()
        else:
            segs = self.stackdb.segments(r)
        cs = build_cross_section(self._reader, r,
                                 mesh_size=self.mesh_size, segments=segs)
        if ov is not None and self.dialect == "v1":
            # web overrides: repoint each web at its edited laminate's set id
            for w, lam in zip(cs["webs"], ov.web_laminates()):
                if not lam:
                    continue
                if lam not in cs["laminates"]:
                    cs["laminates"][lam] = (max(cs["laminates"].values())
                                            + 1)
                w["lam"] = cs["laminates"][lam]
        return cs

    def write_station_yaml(self, st, path=None):
        """Emit the 1-D shell SG yaml of one station from the current state.

        In:  st -- st-id token; path str | None (None = workdir/<tag>).
        Out: dict(n_nodes, n_elems, n_sets, n_webs, out, r)."""
        r = self.resolve(st)
        tag = "%s_r%04d" % (self._stem(), round(r * 1000))
        path = path or os.path.join(self.workdir, tag + "_shell.yaml")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        info = emit_shell_yaml(self.cross_section(r), path,
                               reference=self.reference)
        info["r"] = r
        info["out"] = path
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
        info = self.write_station_yaml(r)
        from .windio.sg_props import beam_props
        P = beam_props(info["out"], out_k=info["out"]
                       .replace("_shell.yaml", ".out"), abd_out=False)
        if xml:
            from .windio.sg_windio import emit_prevabs_xml
            tag = os.path.basename(info["out"])[:-len("_shell.yaml")]
            emit_prevabs_xml(self.cross_section(r),
                             os.path.join(self.workdir, "xml", tag), name=tag)
        if view:
            from opensg_shell import auto_emit
            from .windio.sg_windio import plot_ring_mesh
            plot_ring_mesh(info["out"], info["out"]
                           .replace("_shell.yaml", "_mesh.png"))
            auto_emit(info["out"], out_png=info["out"]
                      .replace("_shell.yaml", "_orient.png"))
        tag = os.path.basename(info["out"])[:-len("_shell.yaml")]
        P.update(yaml=info["out"], tag=tag, mesh=info,
                 chord=self.geometry.chord(r),
                 twist=self._reader.scalar("twist", r))
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

    def __call__(self, st=None, xml=False, view=False):
        """The shortest interface:  K, M = blade(st).

            import opensg
            blade = opensg.read("IEA-15-240-RWT.yaml")
            K, M = blade(4)          # one station (st-id or span r)
            Ks, Ms = blade()         # every station (two lists)

        In:  st -- st-id token (0-based index | span r | str) | None
             (None = full sweep); xml/view bool -- per-station opt-ins.
        Out: (K, M) -- (6,6) Timoshenko stiffness (VABS order) and mass
             matrix; for st=None two station-ordered lists."""
        if st is None:
            out = self.timo_all(verbose=False)
            return ([np.asarray(P["Timo"]) for P in out],
                    [np.asarray(P["Mass"]) for P in out])
        P = self.timo(st, xml=xml, view=view)
        return np.asarray(P["Timo"]), np.asarray(P["Mass"])

    # ----------------------------------------------------- editing helpers
    def section(self, st):
        """The user-updatable SECTION at one station (v1/pyNuMAD dialect).

        Returns the station's resolved layup as an editable Section and
        registers it as a live override: every later cross_section/timo at
        this station uses the edits until section.reset().

            S = blade.section(4)
            S.stacks["LP_SPAR"]                  # [[mat, t, ang], ...]
            S.scale_thickness(1.3, region="LP_SPAR")
            K, M = blade(4)                      # computed from the edit

        In:  st -- st-id token (0-based index | span r | str).
        Out: Section (an existing override at that station is returned
             instead of being recaptured)."""
        if self.dialect != "v1":
            raise NotImplementedError(
                "section() edits the resolved pyNuMAD 12-region stacks; for "
                "windIO v2 blades edit blade.layers / blade.webs and call "
                "update_blade()")
        r = self.resolve(st)
        ov = self._overrides.get(Section._key(r))
        return ov if ov is not None else Section(self, r)

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


def opensg_blade(blade, st=None, **kw):
    """One call: blade (file or Blade) + station -> Timoshenko K and mass M.

    The shortest optimization interface:

        from opensg_shell import Blade, opensg_blade

        b = Blade("IEA-15-240-RWT.yaml")
        K, M = opensg_blade(b, 4)          # st-id (0-based) or span r
        b.scale_layer_thickness("Spar_Cap_SS", 1.2)
        b.update_blade()
        K2, M2 = opensg_blade(b, 4)        # the edited design

    In:
        blade: str | Blade -- blade yaml path (a Blade is built with **kw:
            reference, mesh_size, workdir) or an existing editable Blade.
        st: st-id token (0-based index | span r | str) | None -- None
            returns the FULL sweep as (list of K, list of M) station order.
    Out:
        (K, M): (6,6) float ndarrays -- Timoshenko stiffness (VABS order,
        1 = axial) and mass matrix; for st=None, two lists over stations.
    """
    b = blade if isinstance(blade, Blade) else Blade(blade, **kw)
    if st is None:
        out = b.timo_all(verbose=False)
        return ([np.asarray(P["Timo"]) for P in out],
                [np.asarray(P["Mass"]) for P in out])
    P = b.timo(st)
    return np.asarray(P["Timo"]), np.asarray(P["Mass"])
