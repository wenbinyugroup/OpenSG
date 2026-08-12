"""The ONE `.K` reader/verifier shared by both MSG engines.

A `.K` file is the vendor (and OpenSG) plain-text dump of a homogenized
law.  Three dialects land in this repo and all three are read here -- the
parser is block-driven, not line-counted, so a dialect is recognized by
the block titles it carries rather than by its file name:

VABS beam cross-section `.K` (examples/OpenSG_shell/windio/vabs_K/*.K,
examples/OpenSG_shell/windio/props/*.K), 146 lines, in this order:
    The 6X6 Mass Matrix                                     6x6
    The Mass Center of the Cross Section                    2
    The 6X6 Mass Matrix at the Mass Center                  6x6
    The Mass Properties with respect to Principal Inertial Axes
                              mpus, i11, i22p, i33p, angle, rgyr scalars
    The Geometric Center of the Cross Section               2
    The Area of the Cross Section                    Area = scalar
    Classical Stiffness / Compliance Matrix
                            (1-extension; 2-twist; 3,4-bending)     4x4
    The Tension Center of the Cross Section                 2
    EA, GJ, principal EI22 / EI33 + rotation angle       scalars
    Timoshenko Stiffness / Compliance Matrix
              (1-extension; 2,3-shear, 4-twist; 5,6-bending)        6x6
    The Shear Center of the Cross Section                   2
    Principal shear stiffness GA22 / GA33 + rotation angle  scalars
    Classical Stiffness / Compliance Matrix at Shear Center 4x4
    The Mass Matrix at Shear Center                         6x6
    The Mass Center with respect to Shear Center            2
    The Mass Properties at Shear Center                  scalars

SwiftComp 3-D solid `.sc.k` (Sample_1.sc.k, Sample_2.sc.k), 32 lines:
    The Effective Stiffness Matrix                          6x6 (MPa)
    The Effective Compliance Matrix                         6x6
    The Engineering Constants (Approximated as Orthotropic)
                     E1 E2 E3 G12 G13 G23 nu12 nu13 nu23 scalars
    Effective Density                                       scalar

OpenSG's own `.out` (opensg_solid.sg_homo.write_sc_K) is the SwiftComp
layout with the macro-law name spliced into the title ("The Effective
Cauchy Continuum Stiffness Matrix", "... Timoshenko Beam ..."), an
optional VABS-style `The 6X6 Mass Matrix` block, and a trailing
`Time taken:` line -- the infix is stripped, so the same key reads it.

Beyond reading, `compare_to_K` is the reusable verifier: it lines an
OpenSG law up against a `.K` term by term under an EXPLICIT unit and
index convention and returns a report dict (plus a printable table).
Unit and index conventions are never guessed -- a `.K` carries no units
and no index legend that can be trusted programmatically, so the caller
states them and an unresolvable request raises instead of silently
comparing the wrong things.
"""
import os
from collections import OrderedDict

import numpy as np

# --------------------------------------------------------------- block titles
# canonical key -> (lower-case title test, matcher kind)
# "eq": the stripped lower-case title equals the tag
# "in": the tag is a substring
# "pre": the title starts with the tag
_MATRIX_TAGS = [
    ("mass_at_shear_center", "the mass matrix at shear center", "pre"),
    ("mass_at_mass_center", "the 6x6 mass matrix at the mass center", "pre"),
    ("mass", "the 6x6 mass matrix", "pre"),
    ("classical_stiffness_at_shear_center",
     "classical stiffness matrix at shear center", "pre"),
    ("classical_compliance_at_shear_center",
     "classical compliance matrix at shear center", "pre"),
    ("classical_stiffness", "classical stiffness matrix", "pre"),
    ("classical_compliance", "classical compliance matrix", "pre"),
    ("timoshenko_stiffness", "timoshenko stiffness matrix", "pre"),
    ("timoshenko_compliance", "timoshenko compliance matrix", "pre"),
]

# scalar tags: canonical key -> title substring (value = last token of the line)
_SCALAR_TAGS = OrderedDict([
    ("mpus", "mass per unit span"),
    ("i11", "mass moment of inertia i11"),
    ("i22p", "principal mass moments of inertia i22"),
    ("i33p", "principal mass moments of inertia i33"),
    ("rgyr", "mass-weighted radius of gyration"),
    ("area", "area ="),
    ("ea", "the extension stiffness ea"),
    ("gj", "the torsional stiffness gj"),
    ("ei22", "principal bending stiffness ei22"),
    ("ei33", "principal bending stiffness ei33"),
    ("ga22", "principal shear stiffness ga22"),
    ("ga33", "principal shear stiffness ga33"),
    ("density", "effective density"),
    ("E1", "e1  ="), ("E2", "e2  ="), ("E3", "e3  ="),
    ("G12", "g12 ="), ("G13", "g13 ="), ("G23", "g23 ="),
    ("nu12", "nu12="), ("nu13", "nu13="), ("nu23", "nu23="),
])

# vector tags (n floats on the first numeric line after the title)
_VECTOR_TAGS = [
    ("mass_center_wrt_shear_center",
     "the mass center with respect to shear center", 2),
    ("mass_center", "the mass center of the cross section", 2),
    ("geometric_center", "the geometric center of the cross section", 2),
    ("tension_center", "the tension center of the cross section", 2),
    ("shear_center", "the shear center of the cross section", 2),
]

# angle-carrying prose lines ("... rotated from ... by\n  8.73  degrees ...")
_ANGLE_TAGS = [
    ("ei_angle_deg", "principal bending axes rotated"),
    ("ga_angle_deg", "principal shear axes rotated"),
    ("m_angle_deg", "principal inertial axes rotated"),
]


def _floats(line):
    """Every token of `line` as a float, or None if any token is not one.

    In:  line str.
    Out: list[float] | None (None also for a blank/separator line)."""
    toks = line.split()
    if not toks:
        return None
    out = []
    for t in toks:
        try:
            out.append(float(t))
        except ValueError:
            return None
    return out


def _read_matrix(lines, i0):
    """Collect the square numeric block that follows line index `i0`.

    Separator/blank lines between the title and the data are skipped; the
    first all-numeric line fixes the width n and n such rows are taken.

    In:  lines list[str]; i0 int index of the title line.
    Out: (M ndarray (n, n), j int index one past the block) or (None, i0)."""
    j = i0 + 1
    while j < len(lines) and _floats(lines[j]) is None:
        s = lines[j].strip()
        if s and not set(s) <= set("=-_ "):
            return None, i0            # prose, not a matrix block
        j += 1
    rows = []
    n = None
    while j < len(lines):
        v = _floats(lines[j])
        if v is None:
            break
        if n is None:
            n = len(v)
        elif len(v) != n:
            break
        rows.append(v)
        j += 1
        if len(rows) == n:
            break
    if n is None or len(rows) != n:
        return None, i0
    return np.array(rows, float), j


def read_k_blocks(path):
    """Parse EVERY recognized block of a `.K` / `.out` file.

    Dialect-agnostic: VABS beam, SwiftComp 3-D solid and OpenSG's own
    write_sc_K output all go through this one scan.  The macro-law infix
    of an OpenSG title ("The Effective Cauchy Continuum Stiffness
    Matrix") is stripped, so `effective_stiffness` reads a SwiftComp
    `.sc.k` and an OpenSG `.out` alike.

    A VABS `.K` restates several scalars (mass per unit span, i11, i22,
    i33, the principal angles) once about the user origin and again at the
    shear centre.  FIRST occurrence wins for every scalar, vector and
    matrix, so the whole dict is consistently the USER-ORIGIN set -- the
    frame `mass` and `timoshenko_stiffness` are in; the shear-centre
    matrices keep their own explicit keys.

    In:
        path: str | os.PathLike, the file to read.
    Out:
        dict: "path" str; "matrices" {key: (n, n) ndarray};
        "scalars" {key: float}; "vectors" {key: list[float]};
        "titles" list[str] of the block titles actually seen (the thing to
        quote when a lookup fails).
    """
    path = os.fspath(path)
    lines = open(path).read().splitlines()
    mats, scal, vecs, titles = {}, {}, {}, []

    i = 0
    while i < len(lines):
        raw = lines[i]
        s = raw.strip().lower()
        if not s:
            i += 1
            continue

        # ---- scalars printed inline, with ("Mass per unit span = 7.7E+02")
        # or without ("The extension stiffness EA   2.8E+10") an equals sign
        hit = False
        for key, tag in _SCALAR_TAGS.items():
            if tag in s:
                tail = raw.split("=")[-1] if "=" in raw else raw
                try:
                    v = float(tail.split()[-1])
                    scal.setdefault(key, v)     # FIRST occurrence wins
                    hit = True
                except (ValueError, IndexError):
                    pass
                break
        if hit:
            i += 1
            continue

        # ---- prose angle lines: value sits on the NEXT line
        for key, tag in _ANGLE_TAGS:
            if tag in s:
                for j in range(i + 1, min(i + 3, len(lines))):
                    for tok in lines[j].split():
                        try:
                            scal.setdefault(key, float(tok))
                            hit = True
                            break
                        except ValueError:
                            continue
                    if hit:
                        break
                break
        if hit:
            i += 1
            continue

        # ---- 2-component centers
        for key, tag, nv in _VECTOR_TAGS:
            if s.startswith(tag):
                j = i + 1
                while j < len(lines) and _floats(lines[j]) is None:
                    j += 1
                if j < len(lines):
                    v = _floats(lines[j])
                    if len(v) >= nv:
                        vecs.setdefault(key, v[:nv])
                        titles.append(raw.strip())
                        hit = True
                break
        if hit:
            i += 1
            continue

        # ---- "The Effective <macro law> Stiffness/Compliance Matrix"
        key = None
        if s.startswith("the effective") and s.endswith("matrix"):
            if "compliance" in s:
                key = "effective_compliance"
            elif "stiffness" in s:
                key = "effective_stiffness"
        if key is None:
            for k, tag, kind in _MATRIX_TAGS:
                ok = (s == tag if kind == "eq" else
                      tag in s if kind == "in" else s.startswith(tag))
                if ok:
                    key = k
                    break
        # legacy OpenSG "Stiffness :" block
        if key is None and s.startswith("stiffness") and "matrix" not in s:
            key = "effective_stiffness"

        if key is not None and key not in mats:
            M, j = _read_matrix(lines, i)
            if M is not None:
                mats[key] = M
                titles.append(raw.strip())
                i = j
                continue
        i += 1

    return {"path": path, "matrices": mats, "scalars": scal,
            "vectors": vecs, "titles": titles}


def read_beam_k(path):
    """VABS beam `.K` -> (Timoshenko 6x6, mass 6x6), both in VABS order.

    VABS beam index convention: 1 extension, 2 and 3 transverse shear,
    4 twist, 5 and 6 bending.

    In:
        path: str, a native VABS `.K`, an OpenSG write_vabs_k `.K`, or the
        legacy OpenSG "Stiffness :" `.out`.
    Out:
        (K, M): two (6, 6) ndarrays.
    """
    blk = read_k_blocks(path)
    m = blk["matrices"]
    K = m.get("timoshenko_stiffness")
    if K is None:
        K = m.get("effective_stiffness")
    M = m.get("mass")
    if K is None or K.shape != (6, 6):
        raise ValueError(
            "no Timoshenko stiffness 6x6 in %s (blocks read: %s)"
            % (path, sorted(m)))
    if M is None or M.shape != (6, 6):
        raise ValueError("no mass 6x6 in %s (blocks read: %s)"
                         % (path, sorted(m)))
    return K, M


def read_solid_k(path):
    """SwiftComp 3-D solid `.sc.k` (or an OpenSG solid `.out`) -> its law.

    Voigt order [11 22 33 23 13 12] with ENGINEERING shear strains, the
    order both SwiftComp and opensg_solid write.

    In:
        path: str, the `.sc.k` / `.out` file.
    Out:
        dict: "C" (6, 6) effective stiffness; "S" (6, 6) | None compliance;
        "constants" {E1 E2 E3 G12 G13 G23 nu12 nu13 nu23} (may be empty);
        "density" float | None; "path" str.
    """
    blk = read_k_blocks(path)
    C = blk["matrices"].get("effective_stiffness")
    if C is None or C.shape != (6, 6):
        raise ValueError(
            "no 6x6 effective stiffness in %s (blocks read: %s)"
            % (path, sorted(blk["matrices"])))
    keys = ("E1", "E2", "E3", "G12", "G13", "G23", "nu12", "nu13", "nu23")
    return {"C": C,
            "S": blk["matrices"].get("effective_compliance"),
            "constants": {k: blk["scalars"][k] for k in keys
                          if k in blk["scalars"]},
            "density": blk["scalars"].get("density"),
            "path": blk["path"]}


# ------------------------------------------------------------------ verifier
#: unit name -> factor to SI base (Pa for a stiffness, kg-m for a mass).
UNITS = {"SI": 1.0, "one": 1.0, "Pa": 1.0, "MPa": 1.0e6, "GPa": 1.0e9,
         "kPa": 1.0e3, "psi": 6894.757293168362}

_SOLID_TERMS = OrderedDict([
    ("C11", (0, 0)), ("C12", (0, 1)), ("C13", (0, 2)),
    ("C22", (1, 1)), ("C23", (1, 2)), ("C33", (2, 2)),
    ("C44", (3, 3)), ("C55", (4, 4)), ("C66", (5, 5))])

_BEAM_TERMS = OrderedDict([
    ("EA", (0, 0)), ("GA2", (1, 1)), ("GA3", (2, 2)),
    ("GJ", (3, 3)), ("EI2", (4, 4)), ("EI3", (5, 5))])

_MASS_TERMS = OrderedDict([
    ("mu", (0, 0)), ("i11", (3, 3)), ("i22", (4, 4)), ("i33", (5, 5)),
    ("S3", (0, 4)), ("-S2", (0, 5)), ("-I23", (4, 5))])

#: convention name -> the block, the index legend and the default terms.
CONVENTIONS = {
    "vabs_beam": {
        "n": 6, "block": "timoshenko_stiffness",
        "index_labels": ("1 extension", "2 shear", "3 shear",
                         "4 twist", "5 bending", "6 bending"),
        "terms": _BEAM_TERMS,
        "note": "VABS Timoshenko: 1 extension, 2,3 shear, 4 twist, "
                "5,6 bending; SI (N, N-m, N-m^2)"},
    "vabs_beam_classical": {
        "n": 4, "block": "classical_stiffness",
        "index_labels": ("1 extension", "2 twist", "3 bending",
                         "4 bending"),
        "terms": OrderedDict([("EA", (0, 0)), ("GJ", (1, 1)),
                              ("EI2", (2, 2)), ("EI3", (3, 3))]),
        "note": "VABS classical 4x4: 1 extension, 2 twist, 3,4 bending"},
    "vabs_beam_mass": {
        "n": 6, "block": "mass",
        "index_labels": ("1", "2", "3", "4", "5", "6"),
        "terms": _MASS_TERMS,
        "note": "VABS 6x6 mass matrix about the user origin"},
    "swiftcomp_solid": {
        "n": 6, "block": "effective_stiffness",
        "index_labels": ("11", "22", "33", "23", "13", "12"),
        "terms": _SOLID_TERMS,
        "note": "SwiftComp/opensg_solid 3-D Voigt [11 22 33 23 13 12], "
                "ENGINEERING shears; SwiftComp .sc.k is normally MPa"},
    "swiftcomp_plate": {
        "n": 6, "block": "effective_stiffness",
        "index_labels": ("N11", "N22", "N12", "M11", "M22", "M12"),
        "terms": OrderedDict(
            [("A%d%d" % (i + 1, j + 1), (i, j))
             for i in range(3) for j in range(i, 3)] +
            [("D%d%d" % (i - 2, j - 2), (i, j))
             for i in range(3, 6) for j in range(i, 6)]),
        "note": "Classical plate ABD [N11 N22 N12 M11 M22 M12]"},
}


def _resolve_terms(conv, terms):
    """Turn the `terms` request into [(label, i, j), ...]; raise if unknown."""
    n = conv["n"]
    if terms is None:
        return [(k, i, j) for k, (i, j) in conv["terms"].items()]
    if isinstance(terms, str):
        if terms == "default":
            return [(k, i, j) for k, (i, j) in conv["terms"].items()]
        if terms == "full":
            return [("[%d,%d]" % (i + 1, j + 1), i, j)
                    for i in range(n) for j in range(i, n)]
        if terms == "diagonal":
            return [("[%d,%d]" % (i + 1, i + 1), i, i) for i in range(n)]
        raise ValueError(
            "terms must be 'default', 'full', 'diagonal', a list of labels "
            "or a list of (label, i, j); got %r" % terms)
    out = []
    for t in terms:
        if isinstance(t, str):
            if t not in conv["terms"]:
                raise ValueError(
                    "unknown term %r for this convention; known terms: %s"
                    % (t, ", ".join(conv["terms"])))
            i, j = conv["terms"][t]
        else:
            if len(t) != 3:
                raise ValueError("a term tuple must be (label, i, j), got %r"
                                 % (t,))
            _lab, i, j = t
            t = _lab
        if not (0 <= i < n and 0 <= j < n):
            raise ValueError("term %r index (%d, %d) outside the %dx%d law"
                             % (t, i, j, n, n))
        out.append((t, int(i), int(j)))
    return out


def _resolve_rtol(rtol, term_list):
    """Broadcast `rtol` (float | {label: float} | sequence) over the terms."""
    n = len(term_list)
    if isinstance(rtol, dict):
        missing = [lab for lab, _, _ in term_list if lab not in rtol]
        if missing:
            raise ValueError("rtol dict is missing a gate for %s"
                             % ", ".join(missing))
        return [float(rtol[lab]) for lab, _, _ in term_list]
    if np.isscalar(rtol):
        return [float(rtol)] * n
    g = list(rtol)
    if len(g) != n:
        raise ValueError("rtol sequence has %d entries for %d terms"
                         % (len(g), n))
    return [float(v) for v in g]


def compare_to_K(value, k_path, convention, ref_unit=None, value_unit="SI",
                 terms=None, rtol=1.0e-6, atol=0.0, permutation=None,
                 block=None, symmetrize=False, label=""):
    """Compare an OpenSG law with a vendor `.K`, term by term.

    The comparison the tests assert on AND the thing to print when a run
    is being eyeballed -- it returns data, it does not assert.

    Conventions are stated, never guessed.  A `.K` file records no units
    and no index legend, so `convention` and `ref_unit` are the caller's
    declaration; anything this function cannot resolve (unknown
    convention, missing unit, wrong shape, absent block, bad permutation)
    raises ValueError rather than comparing the wrong numbers.

    In:
        value: (n, n) array-like, OR a str/os.PathLike to an OpenSG `.out`
            / `.K` from which `block` is read.
        k_path: str, the reference `.K` file.
        convention: str, a key of CONVENTIONS -- REQUIRED, no default.
            "vabs_beam" (Timoshenko 6x6: 1 extension, 2,3 shear, 4 twist,
            5,6 bending), "vabs_beam_classical" (4x4), "vabs_beam_mass",
            "swiftcomp_solid" (Voigt [11 22 33 23 13 12]),
            "swiftcomp_plate" (ABD).
        ref_unit: str, unit of the REFERENCE file -- REQUIRED (a
            SwiftComp `.sc.k` is normally "MPa", a VABS beam `.K` "SI").
            One of UNITS.
        value_unit: str, unit of `value` (opensg_solid writes "Pa"/"SI").
        terms: None | "default" | "full" | "diagonal" | list of labels |
            list of (label, i, j).  None == "default".
        rtol: float | {label: float} | sequence aligned with the resolved
            terms -- the relative gate.  Per-term gates matter when the
            terms disagree by construction (a shell-vs-2D-solid beam law
            agrees far better in EI2 than in GA2); a dict must cover every
            requested term.
        atol: float, absolute gate per term IN `ref_unit` (a term passes
            on either gate; atol alone covers reference zeros).
        permutation: None | sequence of n ints, p[k] = the REFERENCE index
            that OpenSG index k maps to.  None is the identity.  Use it
            only to state a real convention difference -- it is validated
            as a permutation.
        block: str | None, override the reference block key (see
            read_k_blocks); None takes the convention's own.
        symmetrize: bool, replace `value` with (value + value.T)/2 first.
        label: str, free text carried into the report/table header.
    Out:
        dict report: "passed" bool, "n_fail" int, "worst" the term dict
        with the largest rel_diff, "terms" list of per-term dicts
        {"term", "i", "j", "opensg", "reference", "abs_diff", "rel_diff",
        "pass"} (values expressed in `ref_unit`), "table" the formatted
        text, plus "convention", "block", "k_path", "ref_unit",
        "value_unit", "scale", "rtol", "atol", "permutation".
    """
    if convention not in CONVENTIONS:
        raise ValueError("unknown convention %r; known: %s"
                         % (convention, ", ".join(sorted(CONVENTIONS))))
    conv = CONVENTIONS[convention]
    n = conv["n"]
    if ref_unit is None:
        raise ValueError(
            "ref_unit is required: a .K file records no units, so the "
            "caller must state the reference's (one of %s).  A SwiftComp "
            ".sc.k is normally 'MPa'; a VABS beam .K is 'SI'."
            % ", ".join(sorted(UNITS)))
    for nm, u in (("ref_unit", ref_unit), ("value_unit", value_unit)):
        if u not in UNITS:
            raise ValueError("unknown %s %r; known: %s"
                             % (nm, u, ", ".join(sorted(UNITS))))
    key = block or conv["block"]

    # ---- the OpenSG side
    if isinstance(value, (str, bytes, os.PathLike)):
        vblk = read_k_blocks(value)
        A = vblk["matrices"].get(key)
        if A is None and key != "effective_stiffness":
            A = vblk["matrices"].get("effective_stiffness")
        if A is None:
            raise ValueError(
                "no %r block in the OpenSG file %s (blocks read: %s)"
                % (key, value, sorted(vblk["matrices"])))
        src = os.fspath(value)
    else:
        A = np.asarray(value, float)
        src = "<array>"
    if symmetrize:
        A = 0.5 * (A + A.T)
    if A.shape != (n, n):
        raise ValueError(
            "convention %r expects a %dx%d law, got %s from %s"
            % (convention, n, n, A.shape, src))

    # ---- the reference side
    rblk = read_k_blocks(k_path)
    R = rblk["matrices"].get(key)
    if R is None:
        raise ValueError(
            "reference %s has no %r block; blocks present: %s"
            % (k_path, key, sorted(rblk["matrices"])))
    if R.shape != (n, n):
        raise ValueError(
            "reference %s block %r is %s, convention %r expects %dx%d"
            % (k_path, key, R.shape, convention, n, n))

    if permutation is not None:
        p = np.asarray(permutation, int)
        if p.shape != (n,) or sorted(p.tolist()) != list(range(n)):
            raise ValueError(
                "permutation must be a permutation of 0..%d, got %r"
                % (n - 1, permutation))
        A = A[np.ix_(p, p)]
        permutation = tuple(int(v) for v in p)

    scale = UNITS[value_unit] / UNITS[ref_unit]
    tl = _resolve_terms(conv, terms)
    gates = _resolve_rtol(rtol, tl)
    rows = []
    for (lab, i, j), g in zip(tl, gates):
        a = float(A[i, j]) * scale
        r = float(R[i, j])
        d = abs(a - r)
        rel = d / abs(r) if r != 0.0 else (0.0 if d == 0.0 else float("inf"))
        rows.append({"term": lab, "i": i, "j": j, "opensg": a,
                     "reference": r, "abs_diff": d, "rel_diff": rel,
                     "rtol": g, "pass": bool(d <= atol or rel <= g)})

    worst = max(rows, key=lambda x: x["rel_diff"]) if rows else None
    rep = {"label": label, "convention": convention, "block": key,
           "k_path": os.fspath(k_path), "value_source": src,
           "ref_unit": ref_unit, "value_unit": value_unit, "scale": scale,
           "rtol": rtol, "atol": atol, "permutation": permutation,
           "index_labels": conv["index_labels"], "note": conv["note"],
           "terms": rows, "worst": worst,
           "n_terms": len(rows),
           "n_fail": sum(0 if x["pass"] else 1 for x in rows)}
    rep["passed"] = rep["n_fail"] == 0
    rep["table"] = format_k_report(rep)
    return rep


def format_k_report(rep):
    """Render a compare_to_K report as a printable table.

    In:  rep dict from compare_to_K.
    Out: str (no trailing newline)."""
    head = "%s vs %s" % (rep["label"] or "OpenSG",
                         os.path.basename(rep["k_path"]))
    gates = sorted({x["rtol"] for x in rep["terms"]})
    gtxt = ("%.3g" % gates[0]) if len(gates) == 1 else "per term"
    L = ["# %s" % head,
         "# convention %s [%s] | block %s | %s"
         % (rep["convention"], ", ".join(rep["index_labels"]), rep["block"],
            rep["note"]),
         "# values in %s (OpenSG read as %s, x%.6g) | gate rtol %s"
         % (rep["ref_unit"], rep["value_unit"], rep["scale"], gtxt)
         + ("" if rep["atol"] == 0 else " atol %.3g" % rep["atol"])
         + ("" if rep["permutation"] is None
            else " | permutation %s" % (rep["permutation"],)),
         "# %-6s %18s %18s %12s %12s %10s  %s"
         % ("term", "opensg", "reference", "abs diff", "rel diff", "rtol",
            "ok")]
    for x in rep["terms"]:
        L.append("  %-6s %18.8e %18.8e %12.4e %12.4e %10.3g  %s"
                 % (x["term"], x["opensg"], x["reference"], x["abs_diff"],
                    x["rel_diff"], x["rtol"],
                    "." if x["pass"] else "FAIL"))
    if rep["worst"] is not None:
        w = rep["worst"]
        L.append("# worst term %s: rel %.4e (abs %.4e) | %d/%d fail"
                 % (w["term"], w["rel_diff"], w["abs_diff"], rep["n_fail"],
                    rep["n_terms"]))
    return "\n".join(L)
