"""windIO blade -> OpenSG 1-D shell SG yaml, standalone (plain yaml, no windIO package).

Reads any windIO file: v2 (outer_shape / structure / anchors, e.g. IEA-22-280-RWT) and
v1 (outer_shape_bem / internal_structure_2d_fem, e.g. the NREL BAR designs) share one
downstream pipeline.  Laminate stacking order = the windIO layer order (outer -> inner);
materials accept scalar (isotropic) or 3-list (orthotropic) E/G/nu (G filled from E, nu
if absent).

Conventions:
  nd_arc s in [0,1] runs TE(s=0) -> suction/upper -> LE(s~0.5) -> pressure/lower -> TE(s=1),
  matching the windIO airfoil coordinate ordering (coords start at TE upper, x: 1->0->1).
  Ply angle = fiber_orientation (deg, about the spanwise/beam axis e1 = OpenSG theta3).
  The emitted section origin is the windIO REFERENCE AXIS x1 (chordwise shift by
  section_offset_y / pitch_axis), the same origin the 6x6 and the BeamDyn loads use.
  e1 = +z (span), e2 = arc tangent, e3 = inward wall normal (skin) / e1 x e2 (web).
"""
import numpy as np
import yaml

try:
    from yaml import CSafeLoader as _Loader
except ImportError:                                       # libyaml absent -> pure-python
    from yaml import SafeLoader as _Loader


def _interp(spec, r):
    """Evaluate a windIO {grid, values} spec (or scalar) at span r.

    In:
        spec: dict {grid, values} | float | None.
        r: float, non-dimensional span in [0, 1].
    Out:
        float value at r, or None when spec is None.
    """
    if isinstance(spec, dict) and "grid" in spec:
        return float(np.interp(r, spec["grid"], spec["values"]))
    if isinstance(spec, (int, float)):
        return float(spec)
    return None


def _arc_param(xy):
    """Normalised cumulative arc length along an open point list.

    In:
        xy: (n, 2) float, airfoil points.
    Out:
        (n,) float, s in [0, 1].
    """
    d = np.r_[0.0, np.cumsum(np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1])))]
    return d / d[-1]


class WindIOBlade:
    """windIO v2 reader (outer_shape / structure / anchors)."""

    def __init__(self, yaml_path):
        self.d = yaml.load(open(yaml_path), Loader=_Loader)
        self.bl = self.d["components"]["blade"]
        self.osh = self.bl["outer_shape"]
        self.st = self.bl["structure"]
        self.mats = {m["name"]: m for m in self.d["materials"]}
        self.afs = {a["name"]: a for a in self.d["airfoils"]}
        self.anch = {}
        for a in self.st["anchors"]:
            self.anch[a["name"]] = a
        for w in self.st["webs"]:
            for a in w.get("anchors", []):
                self.anch[a["name"]] = a

    def scalar(self, name, r):
        """Outer-shape scalar (chord [m], twist [rad], ...) at span r."""
        return _interp(self.osh[name], r)

    def resolve(self, spec, r):
        """Resolve a direct spec or an {anchor: {name, handle}} reference at span r."""
        if spec is None:
            return None
        if "anchor" in spec:
            ref = spec["anchor"]
            return _interp(self.anch[ref["name"]].get(ref["handle"]), r)
        return _interp(spec, r)

    def offset_y(self, r):
        """Chordwise distance LE -> reference axis [m] at span r (section_offset_y)."""
        return _interp(self.osh["section_offset_y"], r) or 0.0

    def length(self):
        """Blade length [m] from the reference axis polyline."""
        ra = self.bl["reference_axis"]
        x, y, z = (np.array(ra[k]["values"], float) for k in ("x", "y", "z"))
        return float(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2).sum())

    def stations(self):
        """The blade's own airfoil spanwise positions (sorted, unique)."""
        return sorted({float(a["spanwise_position"]) for a in self.osh["airfoils"]})

    def airfoil_coords(self, r, n=400):
        """Blend the two bracketing airfoils at r.

        In:
            r: float span; n: int resample count.
        Out:
            (n, 2) float, chord-normalised contour (x: TE=1..LE=0..TE=1, y up/down).
        """
        ent = sorted(self.osh["airfoils"], key=lambda a: a["spanwise_position"])
        pos = [a["spanwise_position"] for a in ent]
        i = int(np.clip(np.searchsorted(pos, r) - 1, 0, len(ent) - 2))
        a0, a1 = ent[i], ent[i + 1]
        w = 0.0 if a1["spanwise_position"] == a0["spanwise_position"] else \
            (r - a0["spanwise_position"]) / (a1["spanwise_position"] - a0["spanwise_position"])
        w = float(np.clip(w, 0, 1))

        def resample(afname):
            c = self.afs[afname]["coordinates"]
            xy = np.column_stack([c["x"], c["y"]]).astype(float)
            s = _arc_param(xy); sn = np.linspace(0, 1, n)
            return np.column_stack([np.interp(sn, s, xy[:, 0]), np.interp(sn, s, xy[:, 1])])
        p0 = resample(a0["name"]); p1 = resample(a1["name"])
        return (1 - w) * p0 + w * p1

    def layers_at(self, r, tol=1e-6):
        """Skin layers present at span r.

        Out:
            list of dict(name, material, s, e, t, fiber, order) -- order = windIO
            layer index = outer -> inner stacking.
        """
        out = []
        for idx, L in enumerate(self.st["layers"]):
            t = _interp(L.get("thickness"), r)
            if not t or t < tol:
                continue
            out.append(dict(name=L["name"], material=L["material"],
                            s=self.resolve(L.get("start_nd_arc"), r),
                            e=self.resolve(L.get("end_nd_arc"), r),
                            t=t, fiber=_interp(L.get("fiber_orientation"), r) or 0.0, order=idx))
        return out

    def webs_at(self, r, tol=1e-6):
        """Webs present at span r (laminate = the layers named after the web).

        Out:
            list of dict(name, s, e, layers).
        """
        webs = []
        for w in self.st["webs"]:
            s = self.resolve(w.get("start_nd_arc"), r); e = self.resolve(w.get("end_nd_arc"), r)
            if s is None or e is None:
                continue
            lam = [L for L in self.layers_at(r, tol) if L["name"].startswith(w["name"])]
            webs.append(dict(name=w["name"], s=s, e=e, layers=lam))
        return webs


class WindIOBladeV1(WindIOBlade):
    """windIO v1 reader (outer_shape_bem / internal_structure_2d_fem), same interface.
    v1 has no anchors -- start/end_nd_arc are direct grid/values; web layers carry a
    `web:` key; airfoils are placed by outer_shape_bem.airfoil_position."""

    def __init__(self, yaml_path):
        self.d = yaml.load(open(yaml_path), Loader=_Loader)
        self.bl = self.d["components"]["blade"]
        self.osh = self.bl["outer_shape_bem"]
        self.ist = self.bl["internal_structure_2d_fem"]
        self.mats = {m["name"]: m for m in self.d["materials"]}
        self.afs = {a["name"]: a for a in self.d["airfoils"]}
        self._layers = self.ist["layers"]
        self._webs = self.ist["webs"]

    def scalar(self, name, r):
        return _interp(self.osh[name], r)

    def resolve(self, spec, r):
        return _interp(spec, r)                          # no anchors in v1

    def offset_y(self, r):
        """LE -> reference axis [m]: v1 pitch_axis is a chord fraction."""
        pa = _interp(self.osh.get("pitch_axis"), r)
        return (pa or 0.0) * (self.scalar("chord", r) or 0.0)

    def length(self):
        ra = self.osh["reference_axis"]
        x, y, z = (np.array(ra[k]["values"], float) for k in ("x", "y", "z"))
        return float(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2).sum())

    def stations(self):
        return sorted({float(g) for g in self.osh["airfoil_position"]["grid"]})

    def airfoil_coords(self, r, n=400):
        ap = self.osh["airfoil_position"]; grid = ap["grid"]; labels = ap["labels"]
        i = int(np.clip(np.searchsorted(grid, r) - 1, 0, len(grid) - 2))
        w = 0.0 if grid[i + 1] == grid[i] else (r - grid[i]) / (grid[i + 1] - grid[i])
        w = float(np.clip(w, 0, 1))

        def resample(afname):
            c = self.afs[afname]["coordinates"]
            xy = np.column_stack([c["x"], c["y"]]).astype(float)
            s = _arc_param(xy); sn = np.linspace(0, 1, n)
            return np.column_stack([np.interp(sn, s, xy[:, 0]), np.interp(sn, s, xy[:, 1])])
        return (1 - w) * resample(labels[i]) + w * resample(labels[i + 1])

    def layers_at(self, r, tol=1e-6):
        out = []
        for idx, L in enumerate(self._layers):
            if L.get("web"):
                continue
            t = _interp(L.get("thickness"), r)
            if not t or t < tol:
                continue
            out.append(dict(name=L["name"], material=L["material"],
                            s=_interp(L.get("start_nd_arc"), r), e=_interp(L.get("end_nd_arc"), r),
                            t=t, fiber=_interp(L.get("fiber_orientation"), r) or 0.0, order=idx))
        return out

    def webs_at(self, r, tol=1e-6):
        webs = []
        for w in self._webs:
            s = _interp(w.get("start_nd_arc"), r); e = _interp(w.get("end_nd_arc"), r)
            if s is None or e is None:
                continue
            lam = []
            for idx, L in enumerate(self._layers):
                if L.get("web") == w["name"]:
                    t = _interp(L.get("thickness"), r)
                    if t and t >= tol:
                        lam.append(dict(name=L["name"], material=L["material"], s=None, e=None,
                                        t=t, fiber=_interp(L.get("fiber_orientation"), r) or 0.0,
                                        order=idx))
            webs.append(dict(name=w["name"], s=s, e=e, layers=lam))
        return webs


def load_blade(path):
    """Auto-detect windIO v1 vs v2 and return the matching reader.

    In:
        path: str, windIO blade yaml.
    Out:
        WindIOBlade | WindIOBladeV1.
    """
    bl = yaml.load(open(path), Loader=_Loader)["components"]["blade"]
    return WindIOBladeV1(path) if "outer_shape_bem" in bl else WindIOBlade(path)


def blade_stations(blade):
    """The blade's own airfoil station positions (non-dimensional span).

    In:
        blade: WindIOBlade | WindIOBladeV1.
    Out:
        list of float r in [0, 1].
    """
    return blade.stations()


def blade_length(blade):
    """Blade length [m] from the windIO reference axis.

    In:
        blade: WindIOBlade | WindIOBladeV1.
    Out:
        float [m].
    """
    return blade.length()


def build_cross_section(blade, r, mesh_size=0.01):
    """Resolve one span station into a 2-D cross-section (nodes, line elements, laminates).

    In:
        blade: WindIOBlade | WindIOBladeV1.
        r: float, non-dimensional span in [0, 1].
        mesh_size: float, target element arc length in chord-normalised units.
    Out:
        dict cs: "r", "chord" [m], "twist" [rad], "nodes" (n,2) [m] (LE origin, shifted
        to the reference axis at emit), "elems" [(n1,n2)], "elem_lam" [int], "laminates"
        {tuple -> set id}, "webs" [dict(a, b, lam, name, s, e)], "blade", "xy" (contour),
        "s_arc", "perim" [m].
    """
    chord = blade.scalar("chord", r)
    xy = blade.airfoil_coords(r) * chord
    s_arc = _arc_param(xy)

    skin_layers = [L for L in blade.layers_at(r) if not L["name"].startswith("web")]
    webs = blade.webs_at(r)

    # segment breakpoints = all skin-layer start/end + web attachment arc positions
    brks = {0.0, 1.0}
    for L in skin_layers:
        for v in (L["s"], L["e"]):
            if v is not None:
                brks.add(float(np.clip(v, 0, 1)))
    for w in webs:
        brks.add(float(np.clip(w["s"], 0, 1))); brks.add(float(np.clip(w["e"], 0, 1)))
    if len([v for v in brks if 1e-6 < v < 1 - 1e-6]) < 2:
        brks |= {0.25, 0.5, 0.75}                        # closed uniform skin: inject dividers
    brks = np.array(sorted(brks))

    def laminate_at(smid):
        cov = [L for L in skin_layers if (L["s"] is not None and L["e"] is not None
                                          and min(L["s"], L["e"]) - 1e-9 <= smid
                                          <= max(L["s"], L["e"]) + 1e-9)]
        cov.sort(key=lambda L: L["order"])               # outer -> inner = windIO layer order
        return tuple((L["material"], round(L["t"], 8), round(L["fiber"], 3)) for L in cov)

    def pt(starc):
        return np.array([np.interp(starc, s_arc, xy[:, 0]), np.interp(starc, s_arc, xy[:, 1])])

    perim = float(np.r_[0, np.cumsum(np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1])))][-1])
    nodes = []; node_arc = []; elems = []; elem_lam = []
    laminates = {}
    segments = []                                # ordered skin (s_a, s_b, set_id) for the XML

    def add_node(starc):
        nodes.append(pt(starc)); node_arc.append(starc); return len(nodes) - 1

    prev = None
    for bi in range(len(brks) - 1):
        a, b = brks[bi], brks[bi + 1]
        seg_len = (b - a) * perim
        nsub = max(1, int(round(seg_len / (mesh_size * chord))))
        ss = np.linspace(a, b, nsub + 1)
        lam = laminate_at(0.5 * (a + b))
        if lam not in laminates:
            laminates[lam] = len(laminates)
        lid = laminates[lam]
        segments.append(dict(s_a=float(a), s_b=float(b), set_id=lid))
        if prev is None:
            prev = add_node(ss[0])
        for sN in ss[1:]:
            cur = add_node(sN)
            elems.append((prev, cur)); elem_lam.append(lid); prev = cur
    nodes = np.array(nodes)
    if np.linalg.norm(nodes[-1] - nodes[0]) < 1e-9 * chord:   # close the loop at the TE
        elems[-1] = (elems[-1][0], 0); nodes = nodes[:-1]; node_arc = node_arc[:-1]
    else:
        elems.append((len(nodes) - 1, 0)); elem_lam.append(elem_lam[-1])

    node_arc = np.array(node_arc)
    web_chains = []
    for w in webs:
        if not w["layers"]:
            continue
        na = int(np.argmin(np.abs(node_arc - w["s"]))); nb = int(np.argmin(np.abs(node_arc - w["e"])))
        lam = tuple((L["material"], round(L["t"], 8), round(L["fiber"], 3)) for L in w["layers"])
        if lam not in laminates:
            laminates[lam] = len(laminates)
        web_chains.append(dict(a=na, b=nb, lam=laminates[lam], name=w["name"],
                               s=float(w["s"]), e=float(w["e"])))

    return dict(r=r, chord=chord, twist=blade.scalar("twist", r), nodes=nodes, elems=elems,
                elem_lam=elem_lam, laminates=laminates, webs=web_chains, blade=blade,
                segments=segments, xy=xy, s_arc=s_arc, perim=perim)


class _Flow(list):
    pass


yaml.add_representer(_Flow, lambda d, data:
                     d.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True))


def _mat_block(blade, name):
    """OpenSG material card (elastic-nested) from the windIO material.

    In:
        blade: reader (for blade.mats); name: str material name.
    Out:
        dict(name, density, elastic={E, G, nu} 3-lists).
    """
    m = blade.mats[name]
    E, G, nu = m.get("E"), m.get("G"), m.get("nu")
    if not isinstance(E, (list, tuple)):                 # isotropic -> replicate
        nu = 0.3 if nu is None else nu
        G = E / (2.0 * (1.0 + nu)) if G is None else G
        E = [E, E, E]; G = [G, G, G]; nu = [nu, nu, nu]
    return dict(name=name, density=float(m.get("rho", 1.0)),
                elastic=dict(E=[float(x) for x in E], G=[float(x) for x in G],
                             nu=[float(x) for x in nu]))


def emit_shell_yaml(cs, out_path, web_mesh=None, reference="center"):
    """Write the OpenSG 1-D shell SG yaml with the reference-axis origin.

    The skin contour is offset from the OML to the laminate reference surface
    ("center" = mid-surface, fraction 0.5; "oml" = no offset) before the webs are
    built, then every node is shifted chordwise so the windIO reference axis x1 is
    the origin -- the same origin the Timoshenko/mass 6x6 and the BeamDyn loads use.
    The chosen reference is recorded in the yaml's `reference` field (the single
    source of truth consumed by build_rm_bundle and the dehom), and the header
    carries `refined: 1` so `opensg <yaml>` runs the shear-refined RM ->
    Timoshenko route by default (not the classical EB 4x4).

    In:
        cs: dict from build_cross_section.
        out_path: str, output yaml path.
        web_mesh: float | None, web element length [m] (default 0.01 * chord).
        reference: "center" | "oml", shell reference surface.
    Out:
        dict(n_nodes, n_elems, n_sets, n_webs, n_mats, out).
    """
    blade = cs["blade"]; chord = cs["chord"]
    fraction = {"center": 0.5, "oml": 0.0}[reference]
    nodes = [np.asarray(p, float) for p in cs["nodes"]]
    elems = list(cs["elems"]); elem_lam = list(cs["elem_lam"])
    web_mesh = web_mesh if web_mesh else 0.01 * chord
    set_of_lam = {v: k for k, v in cs["laminates"].items()}

    # reference-surface offset: OML -> laminate reference surface at `fraction`
    nskin = len(cs["nodes"])
    web_sets_pre = {w["lam"] for w in cs["webs"]}
    skin_xy0 = np.asarray(cs["nodes"])
    area2 = float(np.sum(skin_xy0[:, 0] * np.roll(skin_xy0[:, 1], -1)
                         - np.roll(skin_xy0[:, 0], -1) * skin_xy0[:, 1]))
    inward0 = 1.0 if area2 > 0 else -1.0                 # e3=[-e2y,e2x] inward for a CCW loop
    if fraction:
        acc_n = np.zeros((nskin, 2)); acc_t = np.zeros(nskin); acc_c = np.zeros(nskin)
        for ei, (n1, n2) in enumerate(elems):
            if elem_lam[ei] in web_sets_pre:
                continue
            e2 = nodes[n2] - nodes[n1]; e2 = e2 / (np.linalg.norm(e2) + 1e-30)
            nin = inward0 * np.array([-e2[1], e2[0]])
            tlam = float(sum(t for (_m, t, _a) in set_of_lam[elem_lam[ei]]))
            for nd in (n1, n2):
                if nd < nskin:
                    acc_n[nd] += nin; acc_t[nd] += tlam; acc_c[nd] += 1.0
        offs = [np.asarray(p, float) for p in nodes]
        for i in range(nskin):
            if acc_c[i] > 0 and np.linalg.norm(acc_n[i]) > 1e-12:
                nrm = acc_n[i] / np.linalg.norm(acc_n[i])
                offs[i] = nodes[i] + fraction * (acc_t[i] / acc_c[i]) * nrm
        nodes = offs

    # web chains: bottom(-y3) -> top(+y3) so e2 = +y3, e3 = e1 x e2 = -y2 (2-D-solid match)
    for w in cs["webs"]:
        a, b = w["a"], w["b"]
        if nodes[a][1] > nodes[b][1]:
            a, b = b, a
        Pa, Pb = nodes[a].copy(), nodes[b].copy()
        nseg = max(2, int(round(np.linalg.norm(Pb - Pa) / web_mesh)))
        ts = np.linspace(0, 1, nseg + 1)
        chain = [a]
        for t in ts[1:-1]:
            nodes.append(Pa + t * (Pb - Pa)); chain.append(len(nodes) - 1)
        chain.append(b)
        for ia, ib in zip(chain[:-1], chain[1:]):
            elems.append((ia, ib)); elem_lam.append(w["lam"])

    # reference-axis origin: chordwise shift LE -> x1 (windIO section_offset_y / pitch_axis)
    dx = blade.offset_y(cs["r"])
    nodes = np.array(nodes); nodes[:, 0] -= dx

    nsets = len(cs["laminates"])
    web_sets = {w["lam"] for w in cs["webs"]}
    # consistent skin inward normal from the CLOSED-LOOP ORIENTATION (stable at the TE,
    # where a centroid heuristic flips: skins ~horizontal -> normal ~vertical, C-cen ~horizontal)
    skin_xy = np.asarray(cs["nodes"])
    area2 = float(np.sum(skin_xy[:, 0] * np.roll(skin_xy[:, 1], -1)
                         - np.roll(skin_xy[:, 0], -1) * skin_xy[:, 1]))
    inward = 1.0 if area2 > 0 else -1.0

    # `msg` names the ENGINE that owns the file (`opensg <yaml>` dispatches on
    # it); `refined: 1` = shear-refined (RM wall -> Timoshenko 6x6), the model
    # the windIO blade pipeline is built on -- without it the terminal route
    # defaults to the classical EB 4x4; `reference` is the layup reference
    # surface -- all three are header keys, so they sit above the mesh blocks
    seg = {"msg": "shell", "refined": 1, "reference": reference,
           "nodes": [], "elements": [], "sets": {"element": []},
           "sections": [], "elementOrientations": [], "materials": []}
    for (X, Y) in nodes:
        seg["nodes"].append(_Flow(["%.8f %.8f %.8f" % (X, Y, 0.0)]))
    for (n1, n2) in elems:
        seg["elements"].append(_Flow(["%d %d" % (n1 + 1, n2 + 1)]))
    for k in range(nsets):
        labels = [i + 1 for i, s in enumerate(elem_lam) if s == k]
        seg["sets"]["element"].append({"name": "layup_%d" % k, "labels": labels})
    for k in range(nsets):
        layup = [[mat, float(t), float(ang)] for (mat, t, ang) in set_of_lam[k]]
        seg["sections"].append({"type": "shell", "elementSet": "layup_%d" % k, "layup": layup})
    for ei, (n1, n2) in enumerate(elems):
        P1, P2 = nodes[n1], nodes[n2]
        e2 = (P2 - P1) / (np.linalg.norm(P2 - P1) + 1e-30)
        e3 = np.array([-e2[1], e2[0]])
        if elem_lam[ei] not in web_sets:
            e3 = inward * e3
        seg["elementOrientations"].append(_Flow([0.0, 0.0, 1.0, float(e2[0]), float(e2[1]), 0.0,
                                                 float(e3[0]), float(e3[1]), 0.0]))
    used = []
    for k in range(nsets):
        for (mat, t, ang) in set_of_lam[k]:
            if mat not in used:
                used.append(mat)
    for name in used:
        seg["materials"].append(_mat_block(blade, name))
    with open(out_path, "w") as f:
        yaml.dump(seg, f, sort_keys=False, default_flow_style=False)
    return dict(n_nodes=len(nodes), n_elems=len(elems), n_sets=nsets,
                n_webs=len(cs["webs"]), n_mats=len(used), out=out_path)


def _mat_xml(blade, name):
    """PreVABS material card (isotropic replicated to orthotropic so the .sg
    parser gets uniform cards)."""
    m = blade.mats[name]
    E, G, nu, rho = m.get("E"), m.get("G"), m.get("nu"), float(m.get("rho", 1.0))
    if not isinstance(E, (list, tuple)):
        nu = 0.3 if nu is None else nu
        G = E / (2.0 * (1.0 + nu)) if G is None else G
        E = [E, E, E]; G = [G, G, G]; nu = [nu, nu, nu]
    return ('  <material name="%s" type="orthotropic">\n    <density>%g</density>\n    <elastic>\n'
            '      <e1>%g</e1><e2>%g</e2><e3>%g</e3>\n      <g12>%g</g12><g13>%g</g13><g23>%g</g23>\n'
            '      <nu12>%g</nu12><nu13>%g</nu13><nu23>%g</nu23>\n    </elastic>\n  </material>\n'
            % (name, rho, E[0], E[1], E[2], G[0], G[1], G[2], nu[0], nu[1], nu[2]))


def emit_prevabs_xml(cs, outdir, name="xsec", mesh_size=0.005):
    """Write the PreVABS XML byproduct of a station: {name}.dat (normalised
    airfoil), materials.xml (materials + laminae) and {name}.xml (baselines,
    dividing points, webs, layups) -- the cross-check input for the established
    XML -> prevabs -> 2-D-solid pathway (the same emitter validated against the
    2-D solid orientations; the 1-D yaml route stays the primary output).

    In:
        cs: dict from build_cross_section (carries `segments`).
        outdir: str, output directory (created).
        name: str, file stem.
        mesh_size: float, PreVABS 2-D mesh size (chord-normalised).
    Out:
        dict(dat, xml, materials, n_layups, n_webs, out).
    """
    import os
    blade = cs["blade"]; chord = cs["chord"]
    os.makedirs(outdir, exist_ok=True)
    inv = {v: k for k, v in cs["laminates"].items()}
    web_sets = {w["lam"] for w in cs["webs"]}
    used = []
    for k in range(len(inv)):
        for (mat, t, a) in inv[k]:
            if mat not in used:
                used.append(mat)

    def ply_t(mat):
        pt = blade.mats[mat].get("ply_t")
        return float(pt) if pt else 0.001            # cores have no manufacturing ply_t

    # per-(material, thickness) lamina, through-thickness ply count CAPPED (a thick
    # core split into 1 mm plies makes high-aspect layers that crash the mesher)
    MAX_PLIES = 8
    lam_reg = {}
    for k in range(len(inv)):
        for (mat, t, a) in inv[k]:
            key = (mat, round(t, 8))
            if key not in lam_reg:
                n = min(max(1, int(round(t / ply_t(mat)))), MAX_PLIES)
                lam_reg[key] = ("la_%s_%d" % (mat, len(lam_reg)), n, t / n)

    xyn = cs["xy"] / chord
    with open(os.path.join(outdir, name + ".dat"), "w") as f:
        f.write("%s\n" % name)
        for X, Y in xyn:
            f.write("% .8f % .8f\n" % (X, Y))

    mx = "<materials>\n" + "".join(_mat_xml(blade, m) for m in used)
    for (mat, _tr), (lname, _n, lthk) in lam_reg.items():
        mx += ('  <lamina name="%s">\n    <material>%s</material>\n    <thickness>%g</thickness>\n  </lamina>\n'
               % (lname, mat, lthk))
    mx += "</materials>\n"
    open(os.path.join(outdir, "materials.xml"), "w").write(mx)

    def layup_xml(k):
        s = '    <layup name="layup_%d">\n' % k
        for (mat, t, a) in inv[k]:
            lname, n, _lthk = lam_reg[(mat, round(t, 8))]
            s += '      <layer lamina="%s">%g:%d</layer>\n' % (lname, a, n)
        return s + '    </layup>\n'

    s_arc = cs["s_arc"]; xyc = cs["xy"]; segs = cs["segments"]

    LE_XMIN = 0.02      # keep dividing points off the degenerate LE nose (by-x placement)

    def xn(s):
        x = float(np.interp(s, s_arc, xyc[:, 0])) / chord
        return x if x >= LE_XMIN else LE_XMIN

    def side(s):
        return "top" if s < 0.5 else "bottom"
    bks = sorted({round(float(v), 6) for seg in segs for v in (seg["s_a"], seg["s_b"])
                  if 1e-6 < v < 1 - 1e-6})
    pn = {s: "d%d" % i for i, s in enumerate(bks)}
    pts = "".join('    <point name="%s" on="ln_af" by="x2" which="%s">%.6f</point>\n'
                  % (pn[s], side(s), xn(s)) for s in bks)
    skin = [seg for seg in segs if seg["set_id"] not in web_sets]
    bls = ""; comps = ""; bi = 0
    te_first = skin[0]; te_last = skin[-1]
    for seg in skin[1:-1]:
        a, b = round(seg["s_a"], 6), round(seg["s_b"], 6)
        bls += '    <line name="bl_%d"><points>%s:%s</points></line>\n' % (bi, pn[a], pn[b])
        comps += ('    <segment name="sg_%d">\n      <baseline>bl_%d</baseline>\n'
                  '      <layup>layup_%d</layup>\n    </segment>\n' % (bi, bi, seg["set_id"]))
        bi += 1
    sm = round(te_last["s_a"], 6); s1 = round(te_first["s_b"], 6)
    bls += '    <line name="bl_te"><points>%s:%s</points></line>\n' % (pn[sm], pn[s1])
    comps += ('    <segment name="sg_te">\n      <baseline>bl_te</baseline>\n'
              '      <layup>layup_%d</layup>\n    </segment>\n' % te_first["set_id"])

    # webs: midpoint on the web line + the ACTUAL web angle from the attachments
    web_xml = ""; web_comp = ""
    for wi, w in enumerate(cs["webs"]):
        Ps = np.asarray(cs["nodes"][w["a"]], float); Pe = np.asarray(cs["nodes"][w["b"]], float)
        if Ps[1] < Pe[1]:
            Ps, Pe = Pe, Ps
        ang = float(np.degrees(np.arctan2(Ps[1] - Pe[1], Ps[0] - Pe[0])))
        M = 0.5 * (Ps + Pe) / chord
        web_xml += ('    <point name="wp_%d">%.6f %.6f</point>\n'
                    '    <line name="bl_web_%d"><point>wp_%d</point><angle>%.4f</angle></line>\n'
                    % (wi, M[0], M[1], wi, wi, ang))
        web_comp += ('  <component name="web_%d" depend="surface">\n    <segment name="sg_web_%d">\n'
                     '      <baseline>bl_web_%d</baseline>\n      <layup>layup_%d</layup>\n'
                     '    </segment>\n  </component>\n' % (wi, wi, wi, w["lam"]))

    layups = "".join(layup_xml(k) for k in range(len(inv)))
    xml = ('<cross_section name="%s">\n  <include><material>materials</material></include>\n'
           '  <analysis><model>1</model></analysis>\n'
           '  <general>\n    <scale>%.6f</scale>\n    <mesh_size>%g</mesh_size>\n'
           '    <element_type>linear</element_type>\n  </general>\n'
           '  <baselines>\n    <line name="ln_af" type="airfoil"><points data="file" format="1" header="1">%s.dat</points></line>\n'
           '%s%s%s  </baselines>\n  <layups>\n%s  </layups>\n'
           '  <component name="surface">\n%s  </component>\n%s'
           '  <global><loads>1 2 3 4 5 6</loads></global>\n</cross_section>\n'
           % (name, chord, mesh_size, name, pts, bls, web_xml, layups, comps, web_comp))
    open(os.path.join(outdir, name + ".xml"), "w").write(xml)
    return dict(dat=name + ".dat", xml=name + ".xml", materials="materials.xml",
                n_layups=len(inv), n_webs=len(cs["webs"]), out=outdir)


def generate_cross_sections(windio_path, out_dir="cross_sections",
                            stations="airfoil", mesh_size=0.01,
                            reference="center", xml=True, plots=True,
                            prefix=None, verbose=True):
    """windIO blade -> one 1-D shell SG yaml per station (+ PreVABS XML byproduct).

    The terminal route (`opensg gen_windio_cs <windio.yaml>`): every station
    gets <prefix>_rXXXX_shell.yaml (XXXX = round(r*1000)) with the
    reference-axis origin and the reference surface recorded in the yaml, the
    layup-colored ring-mesh PNG and the e1/e2/e3 orientation PNG, and (xml)
    the PreVABS XML cross-check input under <out_dir>/xml/<tag>/.  A station
    table <prefix>_stations.dat closes the run (twist is passed through in the
    windIO file's own unit).

    In:
        windio_path: str, windIO blade yaml (v1 or v2).
        out_dir: str, output folder (created).
        stations: "airfoil" = the blade's own airfoil positions | int N =
            N uniform stations r = i/(N-1) | iterable of float r.
        mesh_size: float, target element arc length (chord-normalised).
        reference: "center" | "oml", shell reference surface.
        xml: bool, ALSO emit the PreVABS XML per station (default True).
        plots: bool, ALSO emit the mesh + orientation PNGs (default True).
        prefix: str | None, station tag prefix (None = the windIO file stem).
        verbose: bool, per-station progress lines.
    Out:
        dict(rows (n, 7) float station table [r chord twist n_nodes n_elems
             n_sets n_webs], yamls list[str], dat str station-table path,
             out_dir str, prefix str).
    """
    import os

    blade = load_blade(windio_path)
    if stations == "airfoil":
        rs = blade.stations()
    elif isinstance(stations, int):
        rs = [i / (stations - 1.0) for i in range(stations)]
    else:
        rs = [float(r) for r in stations]
    if prefix is None:
        prefix = os.path.splitext(os.path.basename(windio_path))[0]
    os.makedirs(out_dir, exist_ok=True)
    if verbose:
        print("%s: %d stations, reference=%s%s"
              % (windio_path, len(rs), reference, ", xml" if xml else ""))

    if plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from opensg_shell import auto_emit

    rows, yamls = [], []
    for r in rs:
        tag = "%s_r%04d" % (prefix, round(r * 1000))
        cs = build_cross_section(blade, r, mesh_size=mesh_size)
        ypath = os.path.join(out_dir, tag + "_shell.yaml")
        info = emit_shell_yaml(cs, ypath, reference=reference)
        if xml:
            emit_prevabs_xml(cs, os.path.join(out_dir, "xml", tag), name=tag)
        rows.append([r, cs["chord"], cs["twist"], info["n_nodes"],
                     info["n_elems"], info["n_sets"], info["n_webs"]])
        yamls.append(ypath)
        if verbose:
            print("  %s  r=%.4f  chord=%7.3f m  %4d nodes  %4d elems"
                  "  %d sets  %d webs"
                  % (tag, r, cs["chord"], info["n_nodes"], info["n_elems"],
                     info["n_sets"], info["n_webs"]))
        if plots:
            # figures come from the EMITTED file, never the in-memory state
            d = yaml.load(open(ypath), Loader=_Loader)
            nd = np.array([[float(v) for v in row[0].split()][:2]
                           for row in d["nodes"]])
            el = np.array([[int(v) for v in row[0].split()]
                           for row in d["elements"]]) - 1
            lam = np.zeros(len(el), int)
            for k, grp in enumerate(d["sets"]["element"]):
                for lab in grp["labels"]:
                    lam[int(lab) - 1] = k
            cmap = plt.get_cmap("tab20")
            fig, ax = plt.subplots(figsize=(9.0, 5.0))
            for k in range(lam.max() + 1):
                first = True
                for e in np.where(lam == k)[0]:
                    sxy = nd[el[e]]
                    ax.plot(sxy[:, 0], sxy[:, 1], "-", color=cmap(k % 20),
                            lw=2.0, label="layup_%d" % k if first else None)
                    first = False
            ax.plot(nd[:, 0], nd[:, 1], ".", color="0.25", ms=1.5)
            ax.set_aspect("equal")
            ax.set_xlabel("y2 (m)"); ax.set_ylabel("y3 (m)")
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                      frameon=False, fontsize=8)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, tag + "_mesh.png"), dpi=150,
                        bbox_inches="tight")
            plt.close(fig)
            auto_emit(ypath, out_png=os.path.join(out_dir, tag + "_orient.png"))

    dat = os.path.join(out_dir, prefix + "_stations.dat")
    np.savetxt(dat, np.array(rows), fmt="%.4f %10.4f %10.6f %6d %6d %3d %3d",
               header="r chord[m] twist[windIO unit] n_nodes n_elems"
                      " n_sets n_webs")
    return dict(rows=rows, yamls=yamls, dat=dat, out_dir=out_dir, prefix=prefix)


def oml_load_integrals(cs):
    """OML contour integrals for the surface-traction -> beam-load model, about x1.

    A uniform fixed-direction traction p [Pa] on the OML gives, per unit span,
    f_flap = p * P and a torsion m1 = p * integral(x2 ds) with x2 measured FROM the
    reference axis (the same origin as the emitted section and the 6x6).

    In:
        cs: dict from build_cross_section.
    Out:
        (P, Sx2ds): float perimeter [m], float integral of x2 ds [m^2] about x1.
    """
    xy = np.asarray(cs["xy"], float).copy()
    xy[:, 0] -= cs["blade"].offset_y(cs["r"])            # LE origin -> reference axis
    closed = np.vstack([xy, xy[0]])
    seg = np.diff(closed, axis=0)
    ds = np.hypot(seg[:, 0], seg[:, 1])
    x2mid = 0.5 * (closed[:-1, 0] + closed[1:, 0])
    return float(ds.sum()), float((x2mid * ds).sum())
