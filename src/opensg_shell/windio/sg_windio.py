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
                xy=xy, s_arc=s_arc, perim=perim)


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
    source of truth consumed by build_rm_bundle and the dehom).

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

    seg = {"reference": reference, "nodes": [], "elements": [], "sets": {"element": []},
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
