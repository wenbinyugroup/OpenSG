"""sg_reference.py -- the ONE place the msg-shell laminate reference is defined.

The laminate reference surface of an msg-shell SG is a RUN-TIME choice, never a
property recorded in the yaml:

    oml      (default)   the node contour is the OUTER MOLD LINE; every wall
                         laminate stacks INWARD from it (the ABD is referenced
                         to the outer face, no shift)
    center   (--center)  the node contour is the laminate MID-SURFACE; the ABD
                         is parallel-axis shifted by t/2
                         (B' = B - (t/2) A,  D' = D - t B + (t^2/4) A), and the
                         plate-SG z origin, the mass moments and the dehom
                         recovery depths (-t/2 .. +t/2 about the contour) follow

``oml_flip`` / ``iml`` (contour on the inner mold line, full-thickness shift)
stay available to the python API as diagnostics; the CLI exposes --center only.

Where the contour sits (OML or mid-surface) is decided when the 1-D yaml is
GENERATED (pynumad emit_shell_yaml(reference=...), OpenSG_io fraction=...).
OpenSG never moves the nodes -- it only builds the wall law referenced to them.
A leftover ``reference:`` key in a yaml is ignored (the CLI prints a note).
"""

REFS = ("oml", "center", "oml_flip", "iml")
DEFAULT = "oml"
# reference -> fraction of the laminate thickness from the OUTER face at which
# the contour sits: 0 = OML, 0.5 = mid-surface, 1 = IML
FRAC = {"oml": 0.0, "center": 0.5, "oml_flip": 1.0, "iml": 1.0}

_DESCRIBE = {
    "oml": "oml (contour = outer mold line, laminate stacked inward;"
           " pass --center for a mid-surface contour)",
    "center": "center (contour = laminate mid-surface; ABD shifted by t/2,"
              " recovery depths -t/2..+t/2)",
    "oml_flip": "oml_flip (diagnostic: full-thickness shift)",
    "iml": "iml (contour = inner mold line, laminate stacked outward)",
}


def norm_ref(ref):
    """Validate a reference name; ``None`` means the default ("oml").

    In:  ref str | None
    Out: str -- one of REFS (lower-cased)
    Raises ValueError on anything else: a typo must never silently run as OML."""
    if ref is None:
        return DEFAULT
    r = str(ref).strip().lower()
    if r not in FRAC:
        raise ValueError("unknown laminate reference %r -- expected one of %s"
                         " (CLI: the default is oml, --center selects the"
                         " mid-surface)" % (ref, ", ".join(REFS)))
    return r


def frac_of(ref):
    """Reference -> thickness fraction from the OUTER face (0 = OML, 0.5 = mid)."""
    return FRAC[norm_ref(ref)]


def ref_from_flag(center):
    """The CLI rule: --center -> "center", otherwise the default "oml"."""
    return "center" if center else DEFAULT


def describe(ref):
    """One line for the run banner."""
    return _DESCRIBE[norm_ref(ref)]


def yaml_reference_key(path):
    """The value of a top-level ``reference:`` line still present in a yaml, or None.

    Cheap line scan (the key is IGNORED by every route; the CLI only reports it)."""
    with open(path) as f:
        for ln in f:
            if ln.startswith("reference:"):
                v = ln.split(":", 1)[1].split("#")[0].strip().strip("'\"")
                return v or ""
    return None


def reference_note(path, ref):
    """The line the CLI prints when the yaml still carries a ``reference:`` key.

    In:  path str -- the yaml; ref str -- the reference the run uses
    Out: str | None -- a WARNING when the key disagrees with the run, a note
         when it agrees, None when there is no key."""
    key = yaml_reference_key(path)
    if key is None:
        return None
    run = norm_ref(ref)
    if key.strip().lower() == run:
        return (" note      : the yaml's `reference: %s` key is ignored (the"
                " reference is a run-time choice) -- it agrees with this run"
                % key)
    return (" WARNING   : the yaml's `reference: %s` key is IGNORED -- this run"
            " uses `%s` (default oml; pass --center for a mid-surface contour)"
            % (key, run))
