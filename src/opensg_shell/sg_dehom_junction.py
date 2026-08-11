"""sg_dehom_junction.py -- junction-aware DEHOMOGENIZATION (the recovery side).

A midline shell SG has no material at a wall crossing: the t_i x t_j overlap
block has zero measure, which is what the HOMOGENIZATION census correction
(sg_homo.junction_census_correction) adds back as energy.  The RECOVERY has
the opposite problem.  It hangs a full through-thickness column off every
element mid-arc, so within about a wall thickness of a junction the columns of
two crossing walls cover the SAME material TWICE -- with two different wall
laws.  That double cover is what reports a spar-cap ply stress at a web
station (the measured 274 MPa error at the IEA-22 spar-cap/web junctions).

This module does NOT repair the recovery.  It MEASURES the hazard and lets the
yaml header say what to do about it, through the `junction:` key:

    off      today's behaviour EXACTLY -- no extra columns, no sidecar, no
             census, no field touched
    flag     (the default) NON-MUTATING: two EXTRA columns at the END of
             <base>_dehom.txt (jflag, jdist), the same two as extra .vtk
             CELL_DATA scalars, the <base>_dehom.junc sidecar and the stdout
             census.  It changes NO field value -- it corrects nothing, so it
             does not violate the default-OFF rule of
             Rules/junction_corrections.md (that rule governs the
             homogenization Dee correction, a different validity domain --
             see the note at the end of this docstring).
    exclude  as `flag`, plus every duplicate (jflag 3) row carries NaN in its
             15 field columns.  The row itself stays, so row i is still
             element i//n_depth at depth i%n_depth, and the cloud PARTITIONS
             the material domain instead of double-covering it.

    junction_bl: 1.0   OPTIONAL k_bl -- the boundary-layer radius in units of
             the junction's thickest wall.  A MODELLING CHOICE, not a derived
             length: the RM wall solution is polluted within roughly one wall
             thickness of a crossing and 1.0 is the conventional St-Venant
             decay scale, nothing more.  Widen it to be conservative.

    junction_ang: 1.0  OPTIONAL ang_tol_deg -- the tangent-grouping tolerance
             handed to sg_homo.junction_inventory.  1.0 (the default) is the
             homogenization census value and it is DELIBERATELY tight: on a
             curved airfoil contour it calls every node whose two elements
             turn by more than 1 deg a two-family L junction, so an IEA-22
             r/R=0.2 station reports 97 "junctions" of which only 8 are wall
             crossings -- the other 89 are discretized curvature, and they
             raise the advisory flag-1 count from 114 to 1032 of 2790
             stations.  The flag-2/3 counts barely move (the material-band
             test, not the detector, decides those), so this key only tunes
             how much curvature noise the near-field advisory carries.
             Raise it (5-15 deg) for a census that lists wall crossings only.

jflag, per recovery station
    0  clean       outside every junction's r_bl
    1  near        inside some r_bl, but inside NO other wall's material
    2  overlapped  inside another wall's material AND this wall owns the block
    3  duplicate   inside another wall's material and ANOTHER wall owns it
    4  RESERVED for the `patch` tier -- DESIGNED BUT DELIBERATELY NOT
       IMPLEMENTED.  `patch` would replace the overlap block by the
       sg_junction corner micro-solve (corner_micro_law / _patch_mesh), whose
       Dirichlet far-field stubs are admissible only for two STRAIGHT walls of
       UNIFORM layup crossing at ONE node with stubs ~5t of clear wall on
       every leg.  Real blade junctions -- a web landing on a tapering spar
       cap, three-way LE/web/panel nodes, coincident web/cap doublers, an
       element shorter than the crossing wall's thickness -- violate that at
       most stations, so a `patch` implementation would REFUSE the majority of
       the junctions it was asked about and silently fall back to `exclude`
       anyway.  It is specified here and left unbuilt on purpose; nothing in
       this module ever emits flag 4.

jdist, per recovery station
    |p - J| / max_i t_i at the NEAREST junction node J (inf when the section
    has no junction at all), so `jflag >= 1` is exactly `jdist <= k_bl`.

Ownership mirrors sg_junction._patch_mesh.owner EXACTLY (sg_junction.py
lines 294-304), so the two sides of the code cannot drift:
    * one THROUGH family (weight 1.0) at a 2-family node -- it owns the block
      (the `return "A"` branch of a T);
    * two THROUGH families (X) -- mitre by |s_i|/t_j vs |s_j|/t_i (the
      `fill == "mitre4"` branch);
    * two ENDING families (L) -- mitre by the normalized-band diagonal
      xi > eta (the `topo == "L"` branch), generalized from the patch's
      midline-centred bands to the recovery's frac-shifted ones;
    * more than two families -- nearest midline, argmin |z_i|/t_i, and the
      junction is reported "unresolved" in the census and the sidecar.

The overlap area A_j = t_i t_j / |sin theta_ij| is the SAME formula
sg_homo.junction_census_correction uses (sg_homo.py:528): the recovery side
and the homogenization side stay on ONE definition of the block.

NOT the same switch as the homogenization correction.  `junction:` here is a
RECOVERY bookkeeping tier and must never be wired to `build_solid_bundle
(junction="census"|"micro"|"microcell")`: that one is an energy correction
with its own validity domain (stretch-dominated cells only -- it is invalid on
bending-dominated lattices, see Rules/junction_corrections.md) and it stays
default-OFF.  `flag` corrects nothing at all.

ALL VARIABLES USED IN THIS MODULE
---------------------------------
  B                 build_rm_bundle bundle (corners, red_cells, rx3, cross,
                    layup_per_elem, layup_db, frac, ref)
  G                 cli.recovery_points result -- pts/emid/nvec/h/zeta/frac/z;
                    its nvec IS the recovery's own inward normal (never
                    recomputed here: the cross-product sign flips on webs)
  fams              junction_families output, one record per junction node
  walls / mem       wall families at a node / their incident ELEMENTS
  lo, hi            per-element recovery band [-frac*t, (1-frac)*t] (+ any
                    per-element registration offset)
  t_max, r_bl, r_ov junction thickest wall, boundary-layer radius k_bl*t_max,
                    overlap shortlist radius max (t_i+t_j)/(2 sin theta)
  jflag, jdist      per-recovery-station flag and normalized junction distance
  census            counts printed on stdout and written into the sidecar
"""
import numpy as np

TIERS = ("off", "flag", "exclude")
_EPS = 1e-12


def read_tier(hdr):
    """The `junction:`, `junction_bl:` and `junction_ang:` keys of a header.

    yaml 1.1 resolves a bare ``off`` to the BOOLEAN False (and ``on`` to
    True), so the boolean spellings are normalized here rather than left to
    surprise the caller.

    In:  hdr dict -- opensg_solid.sg_mesh.read_yaml_header result
    Out: (tier str in TIERS, k_bl float, ang_tol_deg float)."""
    v = hdr.get("junction", "flag")
    if v is False:                       # `junction: off`  -> yaml False
        v = "off"
    elif v is True:                      # `junction: on`   -> yaml True
        v = "on"
    tier = str(v).strip().lower()
    if tier == "patch":
        raise SystemExit(
            "junction: patch is DESIGNED BUT NOT IMPLEMENTED -- the corner"
            " micro-solve it would\ncall (sg_junction.corner_micro_law) is"
            " admissible only for two straight walls of\nuniform layup"
            " crossing at one node, which most real blade junctions are not."
            "\nUse junction: exclude (drop the duplicate stations) or"
            " junction: flag (report them).")
    if tier not in TIERS:
        raise SystemExit("junction must be %s, got %r"
                         % (" | ".join(TIERS), v))
    try:
        k_bl = float(hdr.get("junction_bl", 1.0))
    except (TypeError, ValueError):
        raise SystemExit("junction_bl must be a number (the boundary-layer"
                         " radius in wall thicknesses), got %r"
                         % (hdr.get("junction_bl"),))
    if not (k_bl >= 0.0):
        raise SystemExit("junction_bl must be >= 0, got %r" % (k_bl,))
    try:
        ang = float(hdr.get("junction_ang", 1.0))
    except (TypeError, ValueError):
        raise SystemExit("junction_ang must be a number (the tangent-grouping"
                         " tolerance in degrees), got %r"
                         % (hdr.get("junction_ang"),))
    if not (0.0 < ang < 90.0):
        raise SystemExit("junction_ang must be in (0, 90) degrees, got %r"
                         % (ang,))
    return tier, k_bl, ang


def _canon(d):
    """The canonical (sign-fixed) direction of a 2-D unit vector, the SAME
    convention sg_homo.junction_inventory groups tangents with.

    In:  d (2,) float -- a unit direction
    Out: (2,) float -- d or -d, whichever has the canonical sign."""
    if d[0] < -1e-9 or (abs(d[0]) < 1e-9 and d[1] < 0.0):
        return -d
    return d


def _rsub_of(B):
    """The per-element section index junction_inventory groups by.

    build_rm_bundle does not carry ``rsub`` in the bundle, but it carries the
    layup NAME per element, and the section index is only ever used as a group
    label -- a stable first-appearance index over the names reproduces the
    grouping exactly.

    In:  B dict -- build_rm_bundle bundle
    Out: (n_el,) int ndarray."""
    rs = B.get("rsub")
    if rs is not None:
        return np.asarray(rs, int)
    idx = {}
    for ln in B["layup_per_elem"]:
        idx.setdefault(ln, len(idx))
    return np.array([idx[ln] for ln in B["layup_per_elem"]], int)


def junction_families(B, G=None, n_depth=9, ang_tol_deg=1.0):
    """Junction nodes of the recovery ring, with PER-ELEMENT wall geometry.

    A thin wrapper around sg_homo.junction_inventory -- the ONE detector,
    shared with the homogenization census correction, so the two sides cannot
    disagree about what a junction is.  The wrapper exists for one caveat:
    junction_inventory reports ONE ``section`` per tangent group
    (``sorted(secs)[0]``), which is lossy whenever a group mixes layups (a
    spar cap and a panel meeting head-on at a web).  Every incident ELEMENT is
    therefore re-attached to its family here, and the thickness is taken per
    element from B["layup_per_elem"] + B["layup_db"].  junction_inventory
    itself is left untouched: the census correction depends on it.

    The wall geometry is the RECOVERY's own: the inward normal is nvec[e] from
    recovery_points / ring_wall_strains, never recomputed from a cross product
    (that sign flips on webs), and the material band is
    [-frac*t, (1-frac)*t] with frac = B["frac"] plus any per-element
    registration offset (B["z_offset_per_elem"], absent = 0.0).

    In:  B dict -- build_rm_bundle bundle; G dict | None -- a
         cli.recovery_points result (built here when None); n_depth int --
         only used to build G; ang_tol_deg float -- tangent grouping
         tolerance handed to junction_inventory
    Out: list of dicts, one per junction node, each
         {"node" int, "xy" (2,), "topo" "L"|"T"|"X"|"N", "resolved" bool,
          "A_j" float (sum over wall pairs of t_i t_j / |sin theta|),
          "t_max" float, "r_ov" float,
          "walls": [{"dir" (2,), "weight" float, "t" float, "sec" int,
                     "layups" [str], "nrm" (2,), "elems" [int],
                     "mem": [{"elem", "tan", "nrm", "lo", "hi", "L"}]}]}."""
    from .sg_homo import junction_inventory

    if G is None:
        from .cli import recovery_points
        G = recovery_points(B, n_depth=n_depth)
    corners = np.asarray(B["corners"], float)
    rc = np.asarray(B["red_cells"], int)
    nvec = np.asarray(G["nvec"], float)
    h = np.asarray(G["h"], float)
    frac = float(G.get("frac", B.get("frac", 0.0)))
    n_el = rc.shape[0]
    zoff = np.asarray(B.get("z_offset_per_elem", np.zeros(n_el)), float)
    lo = -frac*h + zoff
    hi = (1.0 - frac)*h + zoff
    Lel = np.hypot(*(corners[rc[:, 1]] - corners[rc[:, 0]]).T)
    layups = list(B["layup_per_elem"])

    inv = junction_inventory(np.asarray(B["rx3"], float), rc, _rsub_of(B),
                             list(B["cross"]), ang_tol_deg=ang_tol_deg)
    adj = {}
    for e in range(n_el):
        adj.setdefault(int(rc[e, 0]), []).append(e)
        adj.setdefault(int(rc[e, 1]), []).append(e)
    ctol = np.cos(np.radians(ang_tol_deg))

    out = []
    for J in inv:
        nd = int(J["node"])
        walls = [{"dir": np.asarray(d, float), "weight": float(w),
                  "sec": int(sec), "mem": [], "elems": [], "layups": []}
                 for (sec, d, w) in J["walls"]]
        for e in adj.get(nd, []):
            o = int(rc[e, 1]) if int(rc[e, 0]) == nd else int(rc[e, 0])
            v = corners[o] - corners[nd]
            tan = v/np.linalg.norm(v)              # AWAY from the node
            dcan = _canon(tan.copy())
            dots = [abs(float(np.dot(w["dir"], dcan))) for w in walls]
            k = int(np.argmax(dots))
            if dots[k] < ctol:                     # never seen in practice --
                continue                           # inventory made the groups
            walls[k]["mem"].append({"elem": e, "tan": tan,
                                    "nrm": nvec[e].copy(),
                                    "lo": float(lo[e]), "hi": float(hi[e]),
                                    "L": float(Lel[e])})
            walls[k]["elems"].append(e)
            if layups[e] not in walls[k]["layups"]:
                walls[k]["layups"].append(layups[e])
        walls = [w for w in walls if w["mem"]]
        if len(walls) < 2:
            continue
        for w in walls:
            w["t"] = max(float(h[m["elem"]]) for m in w["mem"])
            w["nrm"] = w["mem"][0]["nrm"].copy()
        A_j, r_ov = 0.0, 0.0
        for i in range(len(walls)):
            for j in range(i + 1, len(walls)):
                di, dj = walls[i]["dir"], walls[j]["dir"]
                sinth = abs(float(di[0]*dj[1] - di[1]*dj[0]))
                if sinth < 1e-6:
                    continue
                ti, tj = walls[i]["t"], walls[j]["t"]
                A_j += ti*tj/sinth                 # sg_homo.py:528, verbatim
                r_ov = max(r_ov, 0.5*(ti + tj)/sinth)
        if len(walls) == 2:
            thru = sum(1 for w in walls if w["weight"] >= 1.0)
            topo = {0: "L", 1: "T", 2: "X"}[thru]
        else:
            topo = "N"
        out.append({"node": nd, "xy": corners[nd].copy(), "topo": topo,
                    "resolved": topo != "N", "A_j": float(A_j),
                    "t_max": max(w["t"] for w in walls),
                    "r_ov": float(r_ov), "walls": walls})
    return out


def _in_family(w, p, Jp):
    """Is point ``p`` inside wall family ``w``'s material at junction ``Jp``?

    True when SOME incident element of the family has the point inside its own
    band (lo <= (p-J).n <= hi) and inside its own arc span (0 <= (p-J).t <= L).

    In:  w dict -- a junction_families wall; p (2,); Jp (2,) junction node
    Out: bool."""
    d = p - Jp
    for m in w["mem"]:
        z = float(d @ m["nrm"])
        if not (m["lo"] - _EPS <= z <= m["hi"] + _EPS):
            continue
        s = float(d @ m["tan"])
        if -_EPS <= s <= m["L"] + _EPS:
            return True
    return False


def _band_u(wj, p, Jp, along):
    """Normalized position of ``p`` across family ``wj``'s thickness band,
    oriented so 0 is the face on the -``along`` side and 1 the +``along`` face.

    This is the patch mitre's xi = (cx + tB/2)/tB generalized from the
    midline-centred band of sg_junction._patch_mesh to the recovery's
    frac-shifted band.

    In:  wj dict -- the wall whose band is crossed; p (2,); Jp (2,);
         along (2,) -- the other wall's extension direction
    Out: float, normally in [0, 1]."""
    m = wj["mem"][0]
    z = float((p - Jp) @ m["nrm"])
    t = max(m["hi"] - m["lo"], 1e-30)
    u = (z - m["lo"])/t
    return 1.0 - u if float(m["nrm"] @ along) < 0.0 else u


def _owner(J, cont, p, Jp):
    """Which wall family owns the overlap block at ``p`` -- the EXACT rules of
    sg_junction._patch_mesh.owner (sg_junction.py:294-304).

    In:  J dict -- a junction_families record; cont list[int] -- the families
         whose material contains p; p (2,); Jp (2,)
    Out: int -- the owning family index."""
    walls = J["walls"]
    if len(walls) == 2 and len(cont) == 2:
        i, j = cont
        wi, wj = walls[i], walls[j]
        ti, tj = wi["weight"] >= 1.0, wj["weight"] >= 1.0
        if ti != tj:                       # T: the THROUGH wall owns the block
            return i if ti else j
        if ti and tj:                      # X: fill == "mitre4"
            si = abs(float((p - Jp) @ wi["dir"]))
            sj = abs(float((p - Jp) @ wj["dir"]))
            return i if si/wj["t"] > sj/wi["t"] else j
        u = _band_u(wj, p, Jp, wi["mem"][0]["tan"])            # L: xi
        v = _band_u(wi, p, Jp, wj["mem"][0]["tan"])            # L: eta
        return i if u > v else j
    # > 2 families: no patch topology exists -- nearest midline, and the
    # junction is reported unresolved
    return min(cont, key=lambda k: abs(float((p - Jp) @ walls[k]["nrm"]))
               / max(walls[k]["t"], 1e-30))


def flag_recovery_points(B, G=None, n_depth=9, k_bl=1.0, ang_tol_deg=1.0):
    """Per-station junction flag and normalized junction distance.

    Non-mutating by construction: it reads the recovery grid and returns two
    arrays in the SAME (element, depth) order -- point i is element
    i//n_depth at depth i%n_depth, the order of cli.recovery_points,
    cli.dehom_write and windio.sg_recovery.dehom_station alike.  A consumer of
    the fixed-format VABS cloud can therefore partition it without touching
    the writers::

        from opensg_shell.sg_dehom_junction import flag_recovery_points
        D = dehom_to_files(...)            # or dehom_station(...)
        keep = flag_recovery_points(D["bundle"], n_depth=9)["jflag"] != 3
        write_vabs_sm(base + ".SM", D["pts"][keep], D["sig"][keep])

    In:  B dict -- build_rm_bundle bundle; G dict | None -- recovery_points
         result; n_depth int -- stations per wall (only used when G is None);
         k_bl float -- boundary-layer radius in wall thicknesses;
         ang_tol_deg float -- tangent grouping tolerance
    Out: dict {"jflag" (P,) int, "jdist" (P,) float, "junctions" list (the
         junction_families records, each with the extra key "n_own"/"n_dup"),
         "census" dict, "n_depth" int, "k_bl" float}."""
    if G is None:
        from .cli import recovery_points
        G = recovery_points(B, n_depth=n_depth)
    pts = np.asarray(G["pts"], float)
    n_el = len(G["emid"])
    nd_ = len(G["zeta"])
    P = len(pts)
    jflag = np.zeros(P, int)
    jdist = np.full(P, np.inf)
    fams = junction_families(B, G=G, ang_tol_deg=ang_tol_deg)
    pe = np.repeat(np.arange(n_el), nd_)              # station -> element

    for J in fams:
        J["n_own"] = 0
        J["n_dup"] = 0
        Jp = J["xy"]
        r = np.hypot(pts[:, 0] - Jp[0], pts[:, 1] - Jp[1])
        jdist = np.minimum(jdist, r/max(J["t_max"], 1e-30))
        r_bl = k_bl*J["t_max"]
        near = r <= r_bl
        jflag[near & (jflag < 1)] = 1
        fam_of = {}
        for fi, w in enumerate(J["walls"]):
            for m in w["mem"]:
                fam_of[int(m["elem"])] = fi
        for ip in np.nonzero(r <= J["r_ov"])[0]:      # r_ov: SHORTLIST only
            p = pts[ip]
            cont = [fi for fi in range(len(J["walls"]))
                    if _in_family(J["walls"][fi], p, Jp)]
            if not cont:
                continue
            mine = fam_of.get(int(pe[ip]))
            if mine is None:
                # a station of an element that does NOT touch this node, yet
                # sitting inside an incident wall's material: the incident
                # wall covers that material too, so this column duplicates it
                jflag[ip] = max(jflag[ip], 3)
                J["n_dup"] += 1
                continue
            if mine not in cont:
                cont.append(mine)                     # its own wall, always
            if len(cont) < 2:
                continue
            own = _owner(J, sorted(cont), p, Jp)
            f = 2 if own == mine else 3
            J["n_own" if f == 2 else "n_dup"] += 1
            jflag[ip] = max(jflag[ip], f)

    by_topo = {k: sum(1 for J in fams if J["topo"] == k)
               for k in ("T", "L", "X", "N")}
    census = {"n_junc": len(fams), "by_topo": by_topo,
              "unresolved": sum(1 for J in fams if not J["resolved"]),
              "n_station": P, "n_elem": n_el, "n_depth": nd_,
              "n_near": int((jflag == 1).sum()),
              "n_over": int(((jflag == 2) | (jflag == 3)).sum()),
              "n_dup": int((jflag == 3).sum()), "k_bl": float(k_bl)}
    return {"jflag": jflag, "jdist": jdist, "junctions": fams,
            "census": census, "n_depth": nd_, "k_bl": float(k_bl)}


def census_text(census, tier="flag"):
    """The two-line stdout census of a D run.

    In:  census dict -- flag_recovery_points()["census"]; tier str
    Out: str (two lines, no trailing newline)."""
    bt = census["by_topo"]
    parts = ["%d %s" % (bt[k], k) for k in ("T", "L", "X") if bt[k]]
    if census["unresolved"]:
        parts.append("%d unresolved" % census["unresolved"])
    return (" junctions : %d detected%s\n"
            " stations  : %d total, %d near, %d overlapped (%d duplicates%s)"
            % (census["n_junc"],
               " (%s)" % ", ".join(parts) if parts else "",
               census["n_station"], census["n_near"], census["n_over"],
               census["n_dup"], " excluded" if tier == "exclude" else ""))


def write_sidecar(path, B, R, tier="flag", ang_tol_deg=1.0):
    """The <base>_dehom.junc sidecar: every junction, every wall, every
    affected element and station.

    Layout (blank-separated, one record per line, `#` comments)::

        JUNC id y2 y3 n_walls topo A_j resolved n_own n_dup
        WALL id iw sec t weight t2 t3 n2 n3 owner layups
        FLAG class n_elem n_station
        ELEM class e ...
        STAT class i ...

    ``n_own``/``n_dup`` are the stations THIS junction classified 2 / 3, so a
    duplicate can be traced back to the crossing that caused it (a station
    inside two junctions' blocks is counted by both, but carries the worst
    flag once).

    ``owner`` is `own` (this wall always owns the block -- the through wall of
    a T), `none` (it never does), `mitre` (ownership is point-by-point, an L
    or X) or `nearest` (an unresolved >2-wall node: argmin |z|/t).

    In:  path str; B dict -- the bundle (its frac/ref are recorded);
         R dict -- flag_recovery_points result; tier str; ang_tol_deg float
    Out: str path."""
    jf = R["jflag"]
    jd = R["jdist"]
    fams = R["junctions"]
    c = R["census"]
    nd_ = R["n_depth"]
    ez = np.repeat(np.arange(c["n_elem"]), nd_)
    fin = np.isfinite(jd)
    with open(path, "w") as f:
        f.write("# OpenSG msg-shell dehomogenization junction sidecar\n"
                "# reference       : %s\n"
                "# frac            : %.6f\n"
                "# tier            : %s\n"
                "# junction_bl     : %.4f   (k_bl, r_bl = k_bl * max t)\n"
                "# ang_tol_deg     : %.4f\n"
                "# n_depth         : %d\n"
                % (B.get("ref", "center"), float(B.get("frac", 0.0)), tier,
                   c["k_bl"], ang_tol_deg, nd_))
        f.write("# jflag  0 clean | 1 near | 2 overlapped, this wall owns |"
                " 3 overlapped, another\n"
                "#        wall owns (duplicate) | 4 reserved for the patch"
                " tier (never emitted)\n"
                "# jdist  |p - nearest junction node| / max t there"
                "  (inf: no junction)\n#\n")
        f.write("# %s\n# %s\n#\n" % tuple(census_text(c, tier)
                                          .strip("\n").split("\n")))
        f.write("# A_j = sum over wall pairs of t_i t_j / |sin theta_ij|"
                " (sg_homo.py:528)\n")
        f.write("# JUNC  id            y2            y3  n_walls topo"
                "           A_j  resolved  n_own  n_dup\n")
        f.write("# WALL  id  iw  sec             t   weight"
                "            t2            t3            n2            n3"
                "  owner    layups\n")
        for jid, J in enumerate(fams):
            f.write("JUNC %5d %13.6e %13.6e %8d %4s %13.6e %9d %6d %6d\n"
                    % (jid, J["xy"][0], J["xy"][1], len(J["walls"]),
                       J["topo"], J["A_j"], int(J["resolved"]),
                       J.get("n_own", 0), J.get("n_dup", 0)))
            thru = [w["weight"] >= 1.0 for w in J["walls"]]
            for iw, w in enumerate(J["walls"]):
                if J["topo"] == "T":
                    own = "own" if thru[iw] else "none"
                elif J["topo"] in ("L", "X"):
                    own = "mitre"
                else:
                    own = "nearest"
                f.write("WALL %5d %3d %3d %13.6e %8.3f %13.6e %13.6e"
                        " %13.6e %13.6e  %-7s  %s\n"
                        % (jid, iw, w["sec"], w["t"], w["weight"],
                           w["dir"][0], w["dir"][1], w["nrm"][0], w["nrm"][1],
                           own, "|".join(w["layups"])))
        f.write("#\n# affected entities per flag class\n")
        for cl in (1, 2, 3):
            sel = np.nonzero(jf == cl)[0]
            els = np.unique(ez[sel])
            f.write("FLAG %d %d %d\n" % (cl, len(els), len(sel)))
            if len(els):
                f.write("ELEM %d %s\n" % (cl, " ".join("%d" % e
                                                       for e in els)))
                f.write("STAT %d %s\n" % (cl, " ".join("%d" % i
                                                       for i in sel)))
        f.write("#\n# jdist: %d of %d stations have a junction in the"
                " section; min %s\n"
                % (int(fin.sum()), len(jd),
                   "%.4f" % jd[fin].min() if fin.any() else "inf"))
    return path
