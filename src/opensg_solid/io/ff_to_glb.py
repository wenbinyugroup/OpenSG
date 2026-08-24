"""ff_to_glb.py -- OpenSG `.ff` macro state -> the SwiftComp
dehomogenization input `<name>.sc.glb`, behind `opensg ff_to_glb <ff>`.

SwiftComp's dehomogenization (`SwiftComp <name>.sc 2D L`) reads a file
named `<input file name>.glb` -- the input file name INCLUDING its
extension, so `x.sc` pairs with `x.sc.glb` (SCManual 2.1, section 9).
For the elastic analysis (analysis = 0) that file is exactly:

    v1 v2 v3            macro displacements
    C11 C12 C13         macro rotation matrix (rows), Bi = Cij bj
    C21 C22 C23
    C31 C32 C33
    id1                 0 = the next line is generalized STRESSES,
                        1 = generalized strains
    sigma_bar           Kirchhoff-Love plate: N11 N22 N12 M11 M22 M12
                        Reissner-Mindlin:     ... + N13 N23  (8 values)
                        3-D Cauchy:           s11 s22 s33 s23 s13 s12
    <blank line>        (the manual asks both inputs to end blank)

The OpenSG `.ff` carries the same state (u, theta/C, FF, optional Q),
so the map is a transcription with two decisions made here:

  id1 = 0 ALWAYS.  The `.ff` FF are the plate's own section forces;
      feeding stresses lets SwiftComp turn them into strains with ITS
      law -- the clean cross-code comparison.  (An id1 = 1 file would
      need a law to convert with, and whose law it is becomes ambiguous.)
  N13 N23 for a shear-refined (refined: 1) model come from the `.ff`
      `Q:` key when present, else 0 0 with a printed note -- OpenSG's
      own convention is that Q is not a user input (the Eq. 63 route),
      but the SwiftComp RM dehom slot must be filled.

The `.ff` is the input (`opensg ff_to_glb <ff>`); n_model / refined --
which fix the resultant COUNT -- come from the sibling `<stem>.yaml`
when one exists (or --yaml / explicit arguments), else the RM-plate
default is assumed and printed: n_model 2 + refined 0 -> 6 resultants,
refined 1 -> 8; n_model 3 -> the 6 solid stresses.  n_model 1 (beam) is
refused -- the beam `.sc` header is unvalidated in this tree
(io.sg_input), so a beam .glb has nothing to pair with.

In:  the `.ff` (+ optionally the SG yaml whose header shapes the file)
Out: `<base>.sc.glb`, ready for `SwiftComp <base>.sc 2D|3D L`
"""
import os

import numpy as np


def write_glb(state, path, n_model=2, refined=0, precision=14):
    """Write one SwiftComp elastic `.glb` from a parsed `.ff` state.

    In:  state dict -- opensg_solid.cli.read_ff_state output (u (3,),
         C (3, 3), FF (6,), Q (2,) | None); path str -- the `.sc.glb`
         to write; n_model 2 plate | 3 solid (1 beam refused); refined
         0 classical | 1 shear-refined (plate slot count); precision
         int -- decimals of the %e format
    Out: dict {path, n_model, refined, id1, sigma_bar (list), q_filled
         bool -- True when N13/N23 came from the .ff Q key}."""
    if int(n_model) == 1:
        raise NotImplementedError(
            "a beam `.glb` has nothing to pair with: the beam `.sc` "
            "header is unvalidated in this tree, so io.sg_input refuses "
            "to write the .sc it would drive (see NOT VALIDATED there).  "
            "Use the plate or solid macro model, or the VABS `.sg` route "
            "for a beam cross-section.")
    if int(n_model) not in (2, 3):
        raise ValueError("n_model must be 2 (plate) or 3 (solid), got %r"
                         % (n_model,))
    FF = np.asarray(state["FF"], float).reshape(6)
    sig = list(FF)
    q_filled = False
    if int(n_model) == 2 and int(refined) == 1:
        Q = state.get("Q")
        if Q is not None:
            sig += [float(Q[0]), float(Q[1])]
            q_filled = True
        else:
            sig += [0.0, 0.0]
    fe = "%%.%de" % int(precision)
    u = np.asarray(state["u"], float).reshape(3)
    C = np.asarray(state["C"], float).reshape(3, 3)
    with open(path, "w") as f:
        f.write(" ".join(fe % v for v in u) + "\n")
        for row in C:
            f.write(" ".join(fe % v for v in row) + "\n")
        f.write("0\n")                      # id1: generalized STRESSES
        f.write(" ".join(fe % v for v in sig) + "\n")
        f.write("\n")                       # the manual's blank ending
    return {"path": path, "n_model": int(n_model),
            "refined": int(refined), "id1": 0, "sigma_bar": sig,
            "q_filled": q_filled}


def convert(ff_path, yaml_path=None, out_path=None, n_model=None,
            refined=None, **kw):
    """One `.ff` -> `<base>.sc.glb`, report printed.

    The model shape (how many resultants the .glb line carries) comes,
    in order of authority: the explicit n_model/refined arguments; else
    the yaml (named by yaml_path, or `<ff stem>.yaml` beside the .ff
    when it exists); else the defaults n_model 2, refined 1 (the RM
    plate, 8 resultants) -- printed loudly, because a wrong slot count
    in a list-directed SwiftComp read is silent and total.

    In:  ff_path str -- the OpenSG `.ff` (u / theta|C / FF [/ Q]);
         yaml_path str | None -- the SG yaml whose header shapes the
         .glb; out_path str | None -- the `.glb` (None -> `<ff stem>
         .sc.glb`, the name `SwiftComp <stem>.sc ... L` looks for);
         n_model / refined int | None -- explicit overrides;
         **kw -> write_glb
    Out: dict, the write_glb report."""
    from opensg_solid.cli import read_ff_state

    state = read_ff_state(ff_path)
    if state is None:
        raise FileNotFoundError(
            "no usable `.ff` at %s -- the file is absent or carries no "
            "FF: block (the whitespace station-table route has no single "
            "macro state to hand SwiftComp)" % ff_path)
    base = os.path.splitext(ff_path)[0]
    if yaml_path is None and os.path.exists(base + ".yaml"):
        yaml_path = base + ".yaml"
    if yaml_path is not None:
        from opensg_solid.sg_mesh import read_yaml_header
        hdr = read_yaml_header(yaml_path)
        if n_model is None:
            n_model = int(hdr.get("n_model", 2))
        if refined is None:
            refined = int(hdr.get("refined", 0) or 0)
        shape_src = os.path.basename(yaml_path)
    else:
        shape_src = None

    if out_path is None:
        out_path = base + ".sc.glb"
    r = write_glb(state, out_path, n_model=n_model, refined=refined, **kw)
    print("ff_to_glb: %s -> %s  (%s, %d resultants)"
          % (os.path.basename(ff_path), r["path"],
             "Classical plate" if n_model == 2 else "3-D solid",
             len(r["sigma_bar"]), shape_src))
    sc_name = os.path.basename(out_path)
    if sc_name.endswith(".glb"):
        sc_name = sc_name[:-4]
    print("ff_to_glb: run as  SwiftComp %s %dD L   (the .glb name "
          "must stay <input file name>.glb)"
          % (sc_name, 2 if n_model == 2 else 3))
    return r
