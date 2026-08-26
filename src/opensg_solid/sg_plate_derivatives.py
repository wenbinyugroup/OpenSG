"""Central-difference plate-strain derivatives + the .ff writer.

The refined recovery (Eq. 63 / Eq. 64-66 / the load ladder) is driven by
the IN-PLANE DERIVATIVES of the plate strain measures at the recovery
station.  This module owns that arithmetic:

  * `sg_center_diff` -- the CORE, one station: a (3, 3, 6) stencil of
    strain states around the station element -> the 12 first-derivative
    components (E,1 E,2) and the 18 second-order components
    (E,11 E,12 E,22).  Pure elementwise arithmetic, no shape branching,
    so `jax.vmap(sg_center_diff, in_axes=(0, None, None))` runs it over
    as many stations as wanted.
  * `build_ff` -- the helper that consumes the core's output and writes
    the `<base>.ff` the CLI's read_ff_state parses (`opensg <yaml> D`).

Where the stencil comes from is the CALLER's business (Abaqus rpt
parsing stays in the example scripts): each E[i, j] is the plate state
[e11 e22 2e12 k11 k22 2k12] at the element centred one pitch away,
E[1, 1] being the station itself.

Validation heritage: the same stencil operators were checked on the
pm45 chain analytically -- d2k11 by FD 6.418e-8 vs the closed form
6.452e-8 -- and the law-independent second-difference check is
d2M11/dx1^2 = +q under a uniform pressure.
"""
import os


def sg_center_diff(E, h1, h2):
    """First and second central differences of the plate strain state at
    ONE station.

    In:  E (3, 3, 6) array -- the plate strain state
         [e11 e22 2e12 k11 k22 2k12] on the 3x3 station stencil;
         E[i, j] = the state at (x1 + (i-1) h1, x2 + (j-1) h2), so
         E[1, 1] is the station element itself, index i runs along x1
         and j along x2.  numpy and jax arrays both work -- the body is
         elementwise arithmetic only, so jax.vmap over a leading
         stations axis traces cleanly.
         h1, h2 floats -- the stencil pitches (element size along
         x1 / x2).
    Out: dict {dE1, dE2, dE11, dE12, dE22}, each (6,) --
         dE1/dE2 = d E / d x1, d x2       (the 12 first-derivative
                                           components),
         dE11/dE12/dE22 = the second derivatives E,11 E,12 E,22
                                           (the 18 second-order
                                           components).
    Accuracy: O(h^2); the first differences are exact for a quadratic
    field, the second differences exact for a cubic-free quartic term
    pattern (standard 3x3 stencil results)."""
    dE1 = (E[2, 1] - E[0, 1]) / (2.0 * h1)
    dE2 = (E[1, 2] - E[1, 0]) / (2.0 * h2)
    dE11 = (E[2, 1] - 2.0 * E[1, 1] + E[0, 1]) / (h1 * h1)
    dE22 = (E[1, 2] - 2.0 * E[1, 1] + E[1, 0]) / (h2 * h2)
    dE12 = (E[2, 2] - E[2, 0] - E[0, 2] + E[0, 0]) / (4.0 * h1 * h2)
    return {"dE1": dE1, "dE2": dE2,
            "dE11": dE11, "dE12": dE12, "dE22": dE22}


def _row(f, key, vals, fmt="%.10g"):
    """One `key: [v, v, ...]` yaml line, every value through float() --
    writing a NumPy-2 scalar with %r emits "np.float64(...)" which
    read_ff_state cannot parse."""
    f.write("%s: [%s]\n" % (key, ", ".join(fmt % float(v) for v in vals)))


def build_ff(path, FF, E_stencil=None, h1=None, h2=None,
             u=(0.0, 0.0, 0.0), theta=(0.0, 0.0, 0.0), C=None,
             Q=None, qt6=None, qb6=None):
    """Write the `<base>.ff` macro-state file for `opensg <yaml> D`.

    In:  path str -- the .ff to write (conventionally <yaml stem>.ff);
         FF (6,) -- the generalized macro forces
         [N11 N22 N12 M11 M22 M12];
         E_stencil (3, 3, 6) | None -- the strain stencil around the
         station; when given, sg_center_diff(E_stencil, h1, h2) supplies
         deps_dx1/deps_dx2 AND d2eps_* (h1/h2 then required);
         u (3,), theta (3,) -- macro displacement / rotation;
         C (3, 3) | None -- macro direction cosines (omitted -> the
         reader builds the small-rotation frame from theta);
         Q (2,) | None -- [Q1 Q2] for the Q-consistency rescale;
         qt6/qb6 (6,) | None -- [q q,1 q,2 q,11 q,12 q,22] of the
         TOP/BOTTOM face pressure, q positive pushing INTO the face
         (a UNIFORM pressure is [q, 0, 0, 0, 0, 0]).
    Out: path str -- the file written; keys exactly the read_ff_state
         layout, derivative keys only when a stencil was given, optional
         keys only when given."""
    d = None
    if E_stencil is not None:
        if h1 is None or h2 is None:
            raise ValueError("E_stencil needs its pitches: h1 and h2")
        d = sg_center_diff(E_stencil, h1, h2)
    with open(path, "w") as f:
        f.write("# macro state for `opensg <yaml> D` -- read_ff_state"
                " layout;\n# derivatives by sg_plate_derivatives."
                "sg_center_diff (3x3 stencil, O(h^2))\n")
        _row(f, "u", u)
        _row(f, "theta", theta)
        if C is not None:
            f.write("C:\n")
            for r3 in C:
                f.write("- [%s]\n" % ", ".join("%.10g" % float(v)
                                               for v in r3))
        _row(f, "FF", FF)
        if Q is not None:
            _row(f, "Q", Q)
        if d is not None:
            _row(f, "deps_dx1", d["dE1"])
            _row(f, "deps_dx2", d["dE2"])
            _row(f, "d2eps_dx1dx1", d["dE11"])
            _row(f, "d2eps_dx1dx2", d["dE12"])
            _row(f, "d2eps_dx2dx2", d["dE22"])
        if qt6 is not None:
            _row(f, "qt6", qt6)
        if qb6 is not None:
            _row(f, "qb6", qb6)
    return os.path.abspath(path)
