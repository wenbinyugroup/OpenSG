"""ff_to_glb.py -- OpenSG `.ff` macro state -> the SwiftComp
dehomogenization input `<name>.sc.glb`, behind `opensg ff_to_glb <ff>`.

SwiftComp's dehomogenization (`SwiftComp <name>.sc 2D L`) reads a file
named `<input file name>.glb` -- the input file name INCLUDING its
extension, so `x.sc` pairs with `x.sc.glb` (SCManual 2.1, section 9).
For the elastic analysis (analysis = 0) that file is exactly:

    v1 v2 v3            macro displacements
    C11 C12 C13         macro DIRECTION COSINES, Bi = Cij bj
    C21 C22 C23
    C31 C32 C33
    id1                 0 = the next line is generalized STRESSES,
                        1 = generalized strains
    eps_bar | sig_bar   Kirchhoff-Love plate: e11 e22 2e12 k11 k22 2k12
                                            | N11 N22 N12 M11 M22 M12
                        Reissner-Mindlin:      ... + g13 g23 | N13 N23
                        3-D Cauchy:           e11 e22 e33 2e23 2e13 2e12
                                            | s11 s22 s33 s23 s13 s12
    <blank line>        (the manual asks both inputs to end blank)

The OpenSG `.ff` carries the same state (u, theta/C, the `0:` macro
strain, the `1:`/FF macro force, optional Q), so the map is a
transcription with three decisions, each pinned to the manual:

  Cij IS NOT `I + skew(theta)`.  SCManual Eqs. (38) (3-D) and (39) (KL
      plate) define Cij by Bi = Cij bj -- ROW i holds the deformed base
      vector Bi resolved on the undeformed bj, the TRANSPOSE of the
      rotation OpenSG's own recovery applies (cli.py: `y @ (C - I).T`)
      -- and it carries the macro STRAIN on the diagonal as well as the
      rotation off it:

          Cij = [[1+u1,1,  u2,1,    u3,1  ]        (Eq. 38, 3-D)
                 [ u1,2,  1+u2,2,   u3,2  ]
                 [ u1,3,   u2,3,   1+u3,3 ]]

          Cij = [[1+u1,1,  u2,1,    u3,1  ]        (Eq. 39, KL plate --
                 [ u1,2,  1+u2,2,   u3,2  ]         the 3rd row is the
                 [-u3,1,  -u3,2, 1+u1,1+u2,2]]      KL normal, not free)

      built here from the primitives the `.ff` states, via the standard
      decomposition u_i,j = e_ij - eps_ijk theta_k: u_(a,a) = e_aa,
      u_(1,2) / u_(2,1) = e12 -/+ theta3, and the KL slopes u3,1 =
      -theta2, u3,2 = +theta1.  Writing `I + skew(theta)` -- the
      previous behaviour -- TRANSPOSED the rotation and dropped the
      strain diagonal; it only ever looked right because the two agree
      to O(theta) whenever theta and the macro strain are both ~0.
  id1 follows the `.ff`.  "auto" (the default) writes the `0:` macro
      STRAIN with id1 = 1 when the `.ff` carries one, because that is
      the state OpenSG's own recovery consumed directly ("macro state:
      0: (global strain, used directly)") -- so both codes dehomogenize
      from the SAME generalized strain and the comparison isolates the
      recovery instead of the 5th-digit difference between the two 6x6
      laws.  With no `0:` line, or id1=0 asked for explicitly, the
      `1:`/FF forces go in with id1 = 0 and SwiftComp inverts them
      through ITS law; on this repo's bending-dominated HC states that
      costs ~1e-3 relative on the (tiny, near-cancelling) in-plane
      strains and ~3e-8 on the curvatures.
  N13 N23 / g13 g23 for a shear-refined (refined: 1) model come from
      the `.ff` `Q:` key when present, else 0 0 with a printed note --
      OpenSG's own convention is that Q is not a user input (the Eq. 63
      route), but the SwiftComp RM dehom slot must be filled.

THE COUNT MUST MATCH THE `.sc`.  SwiftComp reads the resultants with a
list-directed READ sized by the `.sc` SUBMODEL line (SCManual 8.1:
0 = Kirchhoff-Love -> 6, 1 = Reissner-Mindlin -> 8), so a 6-value .glb
against a submodel-1 deck runs off the end of the file and the run dies
with every local-field output (.u .sn .sg .snm .sgm) left at ZERO BYTES
and nothing in the .ech -- the echo covers the `.sc` alone, so there is
no error text anywhere to read.  convert() therefore reads the submodel
line of the sibling `<base>.sc` when one is there and REFUSES to write
a .glb that disagrees with it.

The `.ff` is the input (`opensg ff_to_glb <ff>`); n_model / refined --
which fix the resultant COUNT -- come from the sibling `<stem>.yaml`
when one exists (or --yaml / explicit arguments), else the classical
plate is assumed and printed: n_model 2 + refined 0 -> 6 resultants,
refined 1 -> 8; n_model 3 -> the 6 solid components.  A HEADER
`refined: 1` under n_model 2 is downgraded to 0 (printed), the same rule
io.sg_input.write_sc applies to the `.sc` this file pairs with -- a
SwiftComp plate deck is the classical one (its measured evidence sits at
that downgrade); an explicit --refined 1 still sizes the line to 8 for
a submodel-1 deck that already exists.  n_model 1 (beam)
is refused -- the beam `.sc` header is unvalidated in this tree
(io.sg_input), so a beam .glb has nothing to pair with.

In:  the `.ff` (+ optionally the SG yaml whose header shapes the file,
     and the `.sc` the .glb must agree with)
Out: `<base>.sc.glb`, ready for `SwiftComp <base>.sc 2D|3D L`
"""
import os

import numpy as np


def macro_C(state, n_model):
    """The SCManual direction cosines Cij (Bi = Cij bj) of a `.ff` state.

    Eq. (38) for a 3-D macro model, Eq. (39) for a plate/shell -- see THE
    C MATRIX in the module docstring for why this is NOT the rotation
    matrix OpenSG's own recovery uses, but its transpose plus the macro
    strain on the diagonal.

    An explicit `C:` in the `.ff` is honoured, TRANSPOSED on the way out:
    read_ff_state's C is the rotation R (cli.py applies it as
    `y @ (C - I).T`, i.e. x_deformed = C y), and the manual's Cij is
    R^T.  Otherwise C is assembled from `theta:` and the `0:` macro
    strain, whichever of them the file carries.

    In:  state dict -- read_ff_state output; n_model int, 2 plate | 3 solid
    Out: (3, 3) float, the manual's Cij."""
    if state.get("C_explicit") is not None:
        return np.asarray(state["C_explicit"], float).reshape(3, 3).T
    th = np.asarray(state.get("theta", np.zeros(3)), float).reshape(3)
    E = state.get("EPS")
    E = np.zeros(6) if E is None else np.asarray(E, float).reshape(6)
    if int(n_model) == 3:                    # Eq. 38: e = [11 22 33 23 13 12]
        e11, e22, e33 = E[0], E[1], E[2]
        e23, e13, e12 = 0.5 * E[3], 0.5 * E[4], 0.5 * E[5]
    else:                                    # Eq. 39: e = [11 22 2*12 k..]
        e11, e22, e33 = E[0], E[1], 0.0
        e23, e13, e12 = 0.0, 0.0, 0.5 * E[2]
    # u_i,j = e_ij - eps_ijk theta_k
    u11, u22, u33 = e11, e22, e33
    u12, u21 = e12 - th[2], e12 + th[2]
    u13, u31 = e13 + th[1], e13 - th[1]
    u23, u32 = e23 - th[0], e23 + th[0]
    if int(n_model) == 3:
        return np.array([[1.0 + u11, u21, u31],
                         [u12, 1.0 + u22, u32],
                         [u13, u23, 1.0 + u33]])
    return np.array([[1.0 + u11, u21, u31],          # Eq. 39: the third
                     [u12, 1.0 + u22, u32],          # row is the KL
                     [-u31, -u32, 1.0 + u11 + u22]])  # normal, not free


def write_glb(state, path, n_model=2, refined=0, precision=14, id1="auto"):
    """Write one SwiftComp elastic `.glb` from a parsed `.ff` state.

    In:  state dict -- opensg_solid.cli.read_ff_state output (u (3,),
         theta (3,), C (3, 3), EPS (6,) | None, FF (6,) | None,
         Q (2,) | None); path str -- the `.sc.glb` to write; n_model
         2 plate | 3 solid (1 beam refused); refined 0 classical |
         1 shear-refined (plate slot count); precision int -- decimals
         of the %e format; id1 "auto" | 0 | 1 -- which macro state to
         write, "auto" preferring the `.ff` `0:` STRAIN (id1 = 1, what
         OpenSG's own recovery consumed) and falling back to the
         `1:`/FF forces (id1 = 0)
    Out: dict {path, n_model, refined, id1, sigma_bar (list -- the
         values written, strains or forces per id1), q_filled bool --
         True when the last two slots came from the .ff Q key}."""
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
    if id1 == "auto":
        id1 = 1 if state.get("EPS") is not None else 0
    id1 = int(id1)
    if id1 not in (0, 1):
        raise ValueError("id1 must be 0 (generalized stresses), 1 "
                         "(generalized strains) or \"auto\", got %r" % (id1,))
    key = "EPS" if id1 == 1 else "FF"
    if state.get(key) is None:
        raise ValueError(
            "id1 = %d asks for the macro %s, but the `.ff` carries no `%s:`"
            " block" % (id1, "STRAIN" if id1 else "FORCE",
                        "0" if id1 else "1"))
    sig = list(np.asarray(state[key], float).reshape(6))
    q_filled = False
    if int(n_model) == 2 and int(refined) == 1:
        # the RM slots: N13 N23 (id1 = 0) or gamma13 gamma23 (id1 = 1).
        # `Q:` is the transverse-shear RESULTANT, so it fills the stress
        # form only -- with strains there is nothing in the .ff to put
        # here and the honest value is zero.
        Q = state.get("Q")
        if Q is not None and id1 == 0:
            sig += [float(Q[0]), float(Q[1])]
            q_filled = True
        else:
            sig += [0.0, 0.0]
    fe = "%%.%de" % int(precision)
    u = np.asarray(state["u"], float).reshape(3)
    C = macro_C(state, n_model)
    with open(path, "w") as f:
        f.write(" ".join(fe % v for v in u) + "\n")
        for row in C:
            f.write(" ".join(fe % v for v in row) + "\n")
        f.write("%d\n" % id1)      # 0 = generalized STRESSES, 1 = STRAINS
        f.write(" ".join(fe % v for v in sig) + "\n")
        f.write("\n")                       # the manual's blank ending
    return {"path": path, "n_model": int(n_model),
            "refined": int(refined), "id1": id1, "sigma_bar": sig,
            "q_filled": q_filled}


def read_sc_submodel(sc_path):
    """The SUBMODEL line of a `.sc`, or None when the deck has no such line.

    SCManual 8.1: a beam/plate deck opens with one integer -- 0 classical
    (Euler-Bernoulli / Kirchhoff-Love), 1 shear-refined (Timoshenko /
    Reissner-Mindlin) -- while a 3-D deck starts straight at the control
    line.  The two are told apart by what FOLLOWS: a plate's next record
    is the two-value curvature line, a 3-D deck's is the four-value
    `analysis elem_flag trans_flag temp_flag` control line.

    In:  sc_path str -- the SwiftComp `.sc`
    Out: int 0 | 1, or None when the first records do not look like a
         dimensionally-reducible header (a 3-D deck, or an unreadable
         file)."""
    try:
        with open(sc_path) as f:
            recs = []
            for ln in f:
                if ln.split():
                    recs.append(ln.split())
                if len(recs) == 2:
                    break
    except OSError:
        return None
    if len(recs) < 2 or len(recs[0]) != 1:
        return None
    try:
        sub = int(recs[0][0])
    except ValueError:
        return None
    # a plate curvature line is 2 numbers, a beam's 3; 4 integers means we
    # were looking at a 3-D deck's control line and there is no submodel
    return sub if len(recs[1]) in (2, 3) and sub in (0, 1) else None


def convert(ff_path, yaml_path=None, out_path=None, n_model=None,
            refined=None, sc_path=None, check_sc=True, **kw):
    """One `.ff` -> `<base>.sc.glb`, report printed.

    The model shape (how many resultants the .glb line carries) comes,
    in order of authority: the explicit n_model/refined arguments; else
    the yaml (named by yaml_path, or `<ff stem>.yaml` beside the .ff
    when it exists); else the defaults n_model 2, refined 0 (the
    classical plate, 6 resultants) -- printed loudly, because a wrong
    slot count in a list-directed SwiftComp read is silent and total.

    Then the deck itself gets the last word as a GATE: when the `.sc`
    the .glb will be read beside exists, its submodel line is compared
    with `refined` and a disagreement RAISES instead of writing a file
    SwiftComp would die on (see THE COUNT MUST MATCH THE `.sc` in the
    module docstring -- the failure mode is zero-byte outputs and an
    .ech with no error in it).

    In:  ff_path str -- the OpenSG `.ff` (u / theta|C / 0: / 1: [/ Q]);
         yaml_path str | None -- the SG yaml whose header shapes the
         .glb; out_path str | None -- the `.glb` (None -> `<ff stem>
         .sc.glb`, the name `SwiftComp <stem>.sc ... L` looks for);
         n_model / refined int | None -- explicit overrides; sc_path
         str | None -- the `.sc` to gate against (None -> the .glb path
         minus `.glb`); check_sc bool -- False downgrades that gate to
         a printed warning; **kw -> write_glb (id1=, precision=)
    Out: dict, the write_glb report."""
    from opensg_solid.cli import read_ff_state

    state = read_ff_state(ff_path)
    if state is None:
        raise FileNotFoundError(
            "no usable `.ff` at %s -- the file is absent or carries no "
            "macro state (the whitespace station-table route has no "
            "single state to hand SwiftComp)" % ff_path)
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
            if int(n_model) == 2 and refined == 1:
                # the same rule io.sg_input.write_sc applies to the `.sc`
                # this .glb pairs with: a header `refined: 1` is the OpenSG
                # RM macro model, but the SwiftComp plate deck written for
                # it is the CLASSICAL one (submodel 0, 6 resultants) -- see
                # the measured evidence at write_sc's downgrade.  Sizing
                # the .glb to 8 here would be exactly the mismatch the gate
                # below exists to catch.  --refined 1 still forces 8.
                refined = 0
                print("ff_to_glb: header `refined: 1` in %s -> the .glb is"
                      " sized for the CLASSICAL plate (6 resultants), the"
                      " deck io.sg_input.write_sc writes for a plate; pass"
                      " --refined 1 to size it for a submodel-1 deck"
                      % os.path.basename(yaml_path))
        shape_src = os.path.basename(yaml_path)
    else:
        shape_src = None
    if n_model is None or refined is None:
        n_model = 2 if n_model is None else n_model
        refined = 0 if refined is None else refined
        print("ff_to_glb: no SG yaml beside %s and no --n_model/--refined"
              " -- ASSUMING the classical plate (n_model %d, refined %d)."
              "  A wrong resultant count is silent in SwiftComp's read"
              % (os.path.basename(ff_path), n_model, refined))

    if out_path is None:
        out_path = base + ".sc.glb"
    if sc_path is None:
        sc_path = (out_path[:-4] if out_path.endswith(".glb") else None)
    sub = None if not sc_path else read_sc_submodel(sc_path)
    if sub is not None and int(n_model) == 2 and sub != int(refined):
        msg = ("%s is submodel %d (%s, %d resultants) but this .glb would"
               " be written for refined %d (%s, %d) -- SwiftComp reads the"
               " resultants with a list-directed READ sized by the deck,"
               " so the mismatch does not warn: it runs off the end of the"
               " .glb and leaves every local-field output at zero bytes"
               " with no error in the .ech.  Rewrite the .sc from the same"
               " yaml (`opensg yaml_to_sc %s`) so the pair agrees, or pass"
               " --refined %d if the deck is the one you mean"
               % (os.path.basename(sc_path), sub,
                  "Reissner-Mindlin" if sub else "Kirchhoff-Love",
                  8 if sub else 6, int(refined),
                  "Reissner-Mindlin" if refined else "Kirchhoff-Love",
                  8 if refined else 6,
                  os.path.basename(base + ".yaml"), sub))
        if check_sc:
            raise ValueError(msg)
        print("ff_to_glb: WARNING: " + msg)

    r = write_glb(state, out_path, n_model=n_model, refined=refined, **kw)
    print("ff_to_glb: %s -> %s  (%s, %d %s%s)"
          % (os.path.basename(ff_path), r["path"],
             ("3-D solid" if int(n_model) == 3 else
              "Reissner-Mindlin plate" if int(refined) else
              "Kirchhoff-Love plate"),
             len(r["sigma_bar"]),
             "generalized strains, id1 1" if r["id1"] else
             "generalized stresses, id1 0",
             "" if shape_src is None else "; shape from %s" % shape_src))
    if int(n_model) == 2 and int(refined) == 1 and not r["q_filled"]:
        print("ff_to_glb: the last two RM slots are 0 0 -- the `.ff` has no"
              " `Q:` to fill them with")
    if sub is not None:
        print("ff_to_glb: %s submodel %d agrees with the .glb"
              % (os.path.basename(sc_path), sub))
    sc_name = os.path.basename(out_path)
    if sc_name.endswith(".glb"):
        sc_name = sc_name[:-4]
    print("ff_to_glb: run as  SwiftComp %s %dD L   (the .glb name "
          "must stay <input file name>.glb)"
          % (sc_name, 2 if int(n_model) == 2 else 3))
    return r
