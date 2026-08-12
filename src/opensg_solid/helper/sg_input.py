"""sg_input.py -- an OpenSG SG yaml -> a NATIVE Yu-group SG INPUT file, in
EITHER of the two dialects of that family: the SwiftComp `.sc` and the VABS
`.sg`.  The write side of helper.sc_to_yaml (`.sc` -> yaml), and the one
yaml -> SG-input path in the repo (the historical
opensg_solid.rm_plate_1D.helper.yaml_to_sc is now a shim onto this module).

No third-party writer is involved.  EVERY layout claim below is measured off
the SG-input decks committed in this repo, and each is named where it is
used; nothing here rests on a manual this repo does not ship.  The decks,
and how much authority each carries:

  RHC_SW_2UC_45.sc          2-D SG, plate macro model.  Third-party.
  Plate_1D_SG_2UC_45.sc     1-D SG, plate macro model.  Third-party.
  Sample_1.sc / Sample_2.sc 3-D SG.  The STRONGEST evidence in the tree:
                            their SwiftComp outputs Sample_{1,2}.sc.k sit
                            beside them, so SwiftComp demonstrably read
                            these two files and produced a sane 6x6 from
                            them.  Where a rule is in doubt, these win.
  square_tube.sg            VABS `.sg` written by PreVABS 1.6, and the
  iea_s10.sg                twin of square_tube_2Dsolid.yaml -- the same
                            3228 nodes / 5384 elements, so the two can be
                            diffed element by element (they agree: see
                            THE VABS LAYUP DECOMPOSITION below).
  SW_2UC_45.sc              3-D SG, but BLOCKED: 9 nonzero connectivity
                            slots under no known slicing, and read_sc
                            refuses it.  Not used as evidence.

WHY TWO WRITERS AND NOT ONE.  `.sc` and `.sg` are the same FAMILY -- header
flags, node block, element block, per-material blocks, a trailing scalar --
but they are NOT the same file, and a writer that conflated them would emit
something that parses and means something else.  Every place they differ:

  ---------------------------------------------------------------------
                       SwiftComp `.sc`            VABS `.sg`
  ---------------------------------------------------------------------
  macro model    ANY: 1-D beam / 2-D plate  BEAM only.  The SG is always
                 / 3-D solid, and the SG    the 2-D cross-section; there
                 may be 1-D, 2-D or 3-D.    is no 1-D or 3-D SG and no
                 The macro model is NOT in  plate/solid macro model.
                 the file -- it is the
                 command-line argument
                 (`SwiftComp f.sc 2D H`).
  model header   1/2/3 EXTRA lines BEFORE   ALWAYS 3 lines:
                 the control line.  The     `format_flag nlayer`
                 submodel line is ALWAYS    `Timoshenko damping thermal`
                 there (0 = classical,      `curve oblique trapeze Vlasov`
                 1 = shear-refined,         (+ a curvature line when
                 2 = Vlasov, 3 = trapeze),  curve_flag, + an oblique line
                 + a curvature line for a   when oblique_flag).
                 plate (k12 k21) or a beam
                 (k11 k12 k13), + an
                 oblique cos pair for a
                 beam.  See THE SUBMODEL
                 LINE below -- it is there
                 for a 3-D macro model too.
  control line   `analysis elem_flag        no equivalent: the analysis
                 trans_flag temp_flag`      type is a command-line
                 (0 = elastic ...).         argument of VABS.
  size line      `nSG nnode nelem nmate     `nnode nelem nmate` -- no
                 nslave nlayer` -- carries  nSG (always the 2-D section),
                 the SG dimension, the      no nslave (VABS pairs the
                 periodic slave-node count  boundary itself), and nlayer
                 and the layer count.       lives on line 1 instead.
  node record    `id y1 [y2 [y3]]`, nSG     `id x2 x3`, always exactly 2
                 coordinate columns.        columns.
  element rec.   `id mat_id n1..n20` -- the `id n1..n9` -- NO material on
                 material (or, when         the connectivity line, and the
                 nlayer /= 0, the LAYER)    slot count is always 9.
                 rides on the connectivity
                 line, and the slot count
                 is ALWAYS 20, whatever the
                 SG dimension.  See
                 SC_SLOTS.
  per-element    OPTIONAL, only when        MANDATORY, one line per
  orientation    trans_flag = 1: a separate element:
                 block `id a1 a2 a3 b1 b2   `id layer_id theta1`
                 b3 c1 c2 c3` -- three      (format_flag 1), or
                 POINTS a, b, c with        `id mat_id theta3 theta1(9)`
                 e1 = (a-c)/|a-c| and       (old format).  theta1 is the
                 (b-c) in the e1-e2 plane.  LAYER PLANE ANGLE, a single
                 NOT angles.                right-handed rotation about
                                            the beam axis x1.
  layer table    OPTIONAL (nlayer /= 0):    MANDATORY when format_flag=1:
                 `layer_id mat_id angle`.   `layer_id mat_id theta3`,
                                            theta3 = the ply angle about
                                            the layer normal y3.
  material       `mat_id isotropy ntemp`    `mat_id orth`
  block          then, per temperature,     then the constants, then the
                 `T rho` and the            density on its OWN line --
                 constants.  The density    NO temperature line and the
                 comes BEFORE the           density comes AFTER.
                 constants.
  isotropy /     same codes and the same    same codes and the same
  orth codes     constant layouts:          layouts (VABS calls the field
                 0 iso   `E nu`             `orth`, SwiftComp `isotropy`).
                 1 orth  `E1 E2 E3 /
                          G12 G13 G23 /
                          nu12 nu13 nu23`
                 2 aniso 21 upper-tri c_ij
                          over 6 lines
  trailing       omega, "the volume of the  NONE.  A VABS `.sg` ends at the
  normalization  domain spanned by the      last material block; the beam
                 remaining coordinates":    measure is the section itself.
                 the SG volume for a 3-D
                 macro model, the in-plane
                 y1(-y2) measure for a
                 plate, 1.0 for a 2-D beam
                 cross-section.
  comments       NEITHER is evidenced, and both writers REFUSE
                 comments=True.  MEASURED: not one of the 17 `.sc` and 2
                 `.sg` decks in this tree carries a comment line of any
                 kind, and this repo's own sc_to_yaml.read_sc would take
                 a `# nSG nnode nelem nmate nslave nlayer` banner for the
                 size line (it picks the first record with >= 5 tokens).
  ---------------------------------------------------------------------

THE SUBMODEL LINE IS ALWAYS THERE, INCLUDING FOR A 3-D MACRO MODEL.  This
one is worth stating because getting it wrong is silent and total.  Both
3-D decks this repo can prove SwiftComp accepted -- Sample_1.sc and
Sample_2.sc, whose Sample_{1,2}.sc.k outputs sit beside them -- begin

    0                      <- submodel
    0 0 0 0                <- analysis elem_flag trans_flag temp_flag
    3 116851 545741 1 0 0  <- nSG nnode nelem nmate nslave nlayer

so the line is present with no curvature line after it.  It cannot be a
stray: SwiftComp's read is list-directed, so an unexpected leading record
would be swallowed as the control line's first integer, the control read
would run on into the next record, and the size line would come back as
nSG = 0 -- the run could not then have produced the sane 6x6 that sits in
Sample_1.sc.k.  preovios_try.sc, a third 3-D deck, begins the same way.
(The one 3-D file here WITHOUT the line, SW_2UC_45.sc, is the blocked
deck read_sc already refuses; it is not evidence.)

THE VABS LAYUP DECOMPOSITION (write_sg's only non-transcription step).
The ply frame (y1, y2, y3) is the beam frame (x1, x2, x3) rotated
right-handed about x1 by theta1, and the material frame (e1, e2, e3) is
that frame rotated right-handed about y3 by theta3.  So a per-element
material frame decomposes EXACTLY, with no fitting:

    theta1 = atan2(-e3.x2, e3.x3)          (e3 is the layer normal y3)
    y2     = (0, cos theta1, sin theta1)
    theta3 = atan2(e1.y2, e1.x1)

and a frame that is not of that form (e1 tilted out of the x1-y2 plane, a
non-orthonormal or left-handed triad) is REJECTED rather than projected --
see frame_to_vabs_angles.

This is not asserted from a figure -- it is MEASURED.  square_tube.sg was
written by PreVABS for the same section as square_tube_2Dsolid.yaml, with
the same 3228 nodes and 5384 elements in the same order, so the two can be
compared element by element: feeding the yaml's frames through
frame_to_vabs_angles reproduces PreVABS's theta1 on all 5384 elements to
0.0 deg, four walls at 0 / 90 / 180 / -90.  Note -90 and not 270: atan2
puts theta1 in (-180, 180], and PreVABS writes it the same way.

THETA3 AND THE MATERIAL `angle:` ADD -- they do not conflict.  A rotation
by the block's `angle:` and the theta3 read off the frame are both
rotations about the SAME axis y3, so they compose by addition, and that is
exactly what the solver does: sg_materials builds C_asm as

    rotate_C_with_matrix(_rotate_C_np(C, block angle), elem_rotation)

-- the block angle about y3 first, then the element frame.  A yaml that
carries per-element frames AND a nonzero material `angle:` is therefore
well posed and common (it is how a wall-frame-plus-ply-angle section is
naturally written), so element_layups sums the two into theta3 and
_bake_frames folds the angle in before the frame.  Neither rejects it.

WHAT A `.sc` CANNOT SAY, AND WHAT write_sc DOES ABOUT IT.  With
trans_flag = 0 and nlayer = 0 -- the only shape helper.sc_to_yaml.read_sc
can read back -- a `.sc` carries ONE material id per element and no frame.
An SG yaml's per-element `elementOrientations` therefore has no slot, and
write_sc refuses to guess: `orientation=` must be stated as

    "bake"   fold each distinct element frame into its own PRE-ROTATED
             type-2 (21-constant) material block.  Exact, reader-visible,
             and the only lossless choice -- but the material count grows
             with the number of distinct frames, so `max_baked` caps it.
    "points" the native SwiftComp trans_flag=1 block (a, b, c points).
             REFUSED -- named so the caller is told why, not that it is a
             typo.  No deck in this tree is trans_flag=1, so the block's
             component ordering is unevidenced, and the two candidates
             disagree: a, b, c read as POINTS live in the SG's own
             coordinates (the node columns' ordering), while this module's
             frames are in BEAM order.  One fixed choice cannot be right
             for the 1-D, 2-D and 3-D SG at once.  read_sc does not parse
             the block either, so nothing here could even catch it.
    "ignore" drop the frames.  Only correct when they are all identity.

A material `angle:` has no slot either, for the same reason -- the `.sc`
layer table that would carry it exists only when nlayer /= 0.  Dropping it
would silently change the physics, so write_sc always FOLDS a nonzero
`angle:` into the material it belongs to, re-spelling that block as a
pre-rotated type-2 (21-constant) one.  Lossless in C, reader-visible, and
the only thing an nlayer = 0 file can say; the cost is that the block
comes back from read_sc spelled `type: 2` rather than `type: 1` + `angle`.

`.sg` needs none of this: theta1 is per element by construction and the
`angle:` lands in the layer table as theta3.

THE TRAILING omega BELONGS TO A MACRO MODEL, NOT TO THE FILE.  An SG yaml
may state `omega:` in its header, and write_sc will re-emit that number --
but ONLY when the macro model being written is the one the header
describes (`n_model:`, defaulting to 2).  Override n_model and the header
number is no longer the measure of anything: it is re-derived with
sg_omega for the model actually being written.  This is not hypothetical
-- a yaml with `n_model: 3` and `omega: 7.0` written at n_model=2 used to
emit 7.0 where the plate measure of that same mesh was 6.0, a wrong
number in a solver input with nothing to flag it (pinned by
tests/msg_k_reference/test_sg_input_writer.py::
test_omega_is_remeasured_when_the_macro_model_is_overridden).  The
writer's report says which happened, in `omega_source`: "header",
"measured" or "argument".

----------------------------------------------------------------------------
VALIDATED / NOT VALIDATED -- the boundary of this module
----------------------------------------------------------------------------
Everything below is a statement about EVIDENCE IN THIS TREE, not about
what SwiftComp or VABS can do.  "Validated" means a shipped third-party
file says so; a file this repo itself wrote is not evidence for the
convention this repo used to write it.

VALIDATED
  `.sc` element record 20 slots wide at every SG dimension -- measured on
      RHC_SW_2UC_45.sc (2-D) and Sample_{1,2}.sc (3-D).
  `.sc` submodel line present for a 3-D macro model -- Sample_{1,2}.sc,
      the two decks SwiftComp demonstrably read (their .sc.k sit beside
      them), begin `0` / `0 0 0 0` / the size line.
  `.sc` PLATE header: submodel, a 2-value curvature line, the control
      line -- RHC_SW_2UC_45.sc and Plate_1D_SG_2UC_45.sc.
  `.sc` node / element / material blocks and the trailing omega, field
      for field against RHC_SW_2UC_45.sc, Plate_1D_SG_2UC_45.sc and
      Sample_{1,2}.sc; the round trip through read_sc is lossless on all
      four (max |dx| 0.0-5e-16).
  `.sc` 3-D tet4 and (as the same contiguous fill, one node wider) hex8.
  `.sg` EVERY field: write_sg reproduces PreVABS's square_tube.sg and
      iea_s10.sg exactly -- flags, nodes, connectivity, all 55434 theta1
      values, the layer table, the material blocks AND their NONZERO
      density (1.6e3 / 1.94e3) on its own line after the constants.
  The (theta1, theta3) convention itself, against VABS's own output:
      rebuilding iea_s10.sg's frames from its layup and homogenizing
      reproduces iea_s10.sg.K to 3.7e-11, with a negative control that
      fails 6/6.

NOT VALIDATED -- these RAISE rather than emit a guess
  `.sc` beam macro model (n_model=1).  No deck in this tree has a beam
      header, and the repo's own beam examples are fed the byte-identical
      PLATE-header RHC_SW_2UC_45.sc, so the 3-value curvature line and
      the oblique pair are unevidenced here.  write_sc refuses.
  `.sc` orientation="points" (the trans_flag=1 a/b/c block).  Every deck
      in this tree is trans_flag=0, so the block's component ordering is
      unevidenced -- and the natural reading is that a/b/c are POINTS in
      the SG's own coordinates, i.e. the node columns' ordering, while
      this module's frames are in BEAM order (x1 axial first).  The two
      orderings cannot both be right.  write_sc refuses; use "bake"
      (exact, and read_sc reads it back) or "ignore".
  `.sc` a nonzero density (or temperature).  The aux pair is
      `T rho` in this repo's reader, but all four third-party decks write
      `0 0`, which discriminates nothing; the decks here that DO carry a
      nonzero one (square_tube.sc, four_case/*.sc) were written by this
      module's own predecessor.  write_sc refuses unless the caller says
      drop_density=True, which writes the `0 0` line every vendor deck
      actually contains.  The `.sg` density is a different field and IS
      validated (above).
  `.sc` tet10.  All three 3-D decks here are tet4; the split layout
      (corners 1-4, slots 5-6 zero, midsides 7-12) is this repo's own
      sc_to_yaml convention read back by this repo.  _slots refuses.
  comments=True in EITHER dialect -- see the table above.

NOT VALIDATED, and currently emitted anyway -- documented, not guarded
  `.sg` orth 0 (`E nu`) and orth 2 (21 constants) material layouts: every
      vendor `.sg` here is orth 1, so those two are the SwiftComp layouts
      carried over by analogy (see _write_material_sg).
  `.sg` curve_flag / oblique_flag 1 and their extra lines, and a nonzero
      `.sc` plate curvature: both vendor `.sg` files and all four vendor
      `.sc` files carry zeros, so the line SHAPES are evidenced but the
      nonzero values have never been exercised.
  A curved (quad8 / quad9 / tri6) mesh measured by integrated_measure,
      which uses the straight-sided corner polygon.

# ----------------------------------------------------------------------------
# ALL VARIABLES USED IN THIS MODULE
# ----------------------------------------------------------------------------
# inputs
#   path, out_path      the SG yaml in; the .sc/.sg out
#   sg                  the SG dict this module passes around, below
#   n_model, refined    the macro model (1 beam, 2 plate, 3 solid) and the
#                       classical/shear-refined switch (.sc header)
#   orientation         "bake" | "ignore" for write_sc ("points" refused)
#   drop_density        write the vendor-evidenced `0 0` aux line instead
#                       of refusing an unvalidated nonzero `.sc` density
#   precision           decimals of the %e number format
# working
#   raw                 the yaml dict as loaded (CBaseLoader: all strings)
#   conn                per-element connectivity, 0-based internally
#   ori                 (E, 9) yaml-order element frames -> (E, 3, 3) beam
#   t1, t3              per-element theta1 / theta3 [deg]
#   layers              [(mat_id, theta3)] in first-appearance order
#   slots               the zero-padded connectivity record
# outputs
#   sg                  {dim, nodes (N, 3), cells (0-based), mat_id (E,)
#                        1-based, materials {id: block}, orientation
#                        (E, 9) | None, n_model, refined, scale}
#   <out>.sc / <out>.sg the SG input file
# ----------------------------------------------------------------------------
"""
import os

import numpy as np
import yaml

#: connectivity slots in ONE SwiftComp element record -- 20, whatever the SG
#: dimension.  NOT the per-dimension node maxima (1-D 5, 2-D 9, 3-D 20): those
#: are element-TYPE capacities, not the width of the record, and using them
#: would emit a 2-D SG at 9 slots where every 2-D vendor deck here writes 20.
#: Measured: RHC_SW_2UC_45.sc is a 2-D SG padded to 20, and Sample_{1,2}.sc --
#: the decks proven accepted by SwiftComp (their .sc.k outputs sit beside
#: them) -- pad to 20.  A short record is not benign: the read is
#: list-directed, so it runs on into the next line and shears the mesh.
SC_SLOTS = 20
VABS_SLOTS = 9

#: SwiftComp `isotropy` / VABS `orth` -- the SAME three codes in both codes
ISO, ORTHO, ANISO = 0, 1, 2

#: what write_sc will do with per-element frames.  "points" -- the native
#: SwiftComp trans_flag=1 a/b/c block -- is NAMED here and refused in
#: write_sc rather than dropped from the vocabulary, so a caller who asks
#: for it is told why instead of being told it is a typo.
_ORIENTATION_MODES = ("bake", "points", "ignore")

#: the one refusal text both writers share
_NO_COMMENTS = (
    "comments=True is not validated for the %s dialect: not one of the 17 "
    "`.sc` and 2 `.sg` decks in this tree carries a comment line, so there "
    "is no evidence that either reader skips a `%s` record -- and this "
    "repo's own sc_to_yaml.read_sc demonstrably does NOT, because it takes "
    "the first record with >= 5 tokens for the size line and a block banner "
    "has more than that.  A comment in a list-directed read is not benign: "
    "it is swallowed as data and shears every field after it.  Write the "
    "file without comments.")

# the shell SG yaml dialect -- a different engine's file, not a solid SG
_SHELL_KEYS = ("sections", "elementOrientations", "elements")


# ---------------------------------------------------------------- yaml input
def _tokens(row):
    """The numeric tokens of one yaml mesh row, whatever flow style it used.

    `- [1 2]` is a single yaml scalar "1 2" while `- [1, 2]` is a two-item
    list; both are one element record.

    In:  row -- a yaml list, scalar or string
    Out: list[str] of whitespace-separated tokens."""
    if not isinstance(row, (list, tuple)):
        row = [row]
    return " ".join(str(v) for v in row).replace(",", " ").split()


def _num(v):
    """A yaml scalar as a python number when it is one, else unchanged.

    CBaseLoader returns every scalar as a string; this is the local
    equivalent of sg_mesh._numify, kept here so the helper does not have to
    import the fe_jax-backed sg_mesh.

    In:  v -- any yaml value
    Out: int | float | the value unchanged (recursing into list/dict)."""
    if isinstance(v, dict):
        return {k: _num(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_num(x) for x in v]
    if isinstance(v, str):
        try:
            f = float(v)
        except ValueError:
            return v
        return int(f) if (f.is_integer() and "." not in v
                          and "e" not in v.lower()) else f
    return v


def _load_yaml(path):
    """The whole SG yaml, parsed fast (libyaml, untyped).

    In:  path str
    Out: dict (raises ValueError when the file is not a yaml mapping)."""
    try:
        from yaml import CBaseLoader as _YL     # libyaml, everything a str
    except ImportError:                          # pragma: no cover
        from yaml import SafeLoader as _YL
    with open(path) as f:
        raw = yaml.load(f, Loader=_YL)
    if not isinstance(raw, dict):
        raise ValueError("%s is not an SG yaml (top level is %s, not a "
                         "mapping)" % (path, type(raw).__name__))
    return raw


def _material_from_list_entry(m):
    """One entry of the LIST-form `materials:` block -> a canonical block.

    The 2-D solid dialect spells a material either flat
    (`name/E/G/nu/rho`, opensg_io) or nested (`name/density/elastic:
    {E,G,nu}`, helper.msh_to_yaml).  Both are the nine engineering
    constants, i.e. a type-1 block.

    In:  m dict
    Out: {"type": 1, "engineering": [9], "density": rho, "name": str}."""
    m = _num(m)
    el = m.get("elastic", m)
    if not all(k in el for k in ("E", "G", "nu")):
        raise ValueError(
            "material %r carries no E/G/nu (nor an `elastic:` sub-block) --"
            " the list-form `materials:` block of the 2-D solid SG yaml is"
            " nine engineering constants" % m.get("name", m))

    def three(v):
        a = np.asarray(v, float).ravel()
        return list(np.broadcast_to(a, (3,)))
    out = {"type": ORTHO,
           "engineering": (three(el["E"]) + three(el["G"])
                           + three(el["nu"]))}
    rho = m.get("rho", m.get("density"))
    if rho is not None and float(rho) > 0.0:
        out["density"] = float(rho)
    ang = m.get("angle")
    if ang is not None and float(ang) != 0.0:
        out["angle"] = float(ang)
    if "name" in m:
        out["name"] = str(m["name"])
    return out


def _material_from_dict_entry(m):
    """One entry of the DICT-form `materials:` block -> a canonical block.

    The canonical solid dialect (what helper.sc_to_yaml emits and
    sg_mesh.load_sg_input reads): `type` 0 (E/nu), 1 (nine `engineering`
    constants) or 2 (the stored 6x6 `C`), plus optional `angle`/`density`.

    In:  m dict
    Out: the same dict, numified and validated."""
    m = _num(m)
    t = int(m.get("type", ORTHO))
    out = {"type": t}
    if t == ISO:
        out["E"], out["nu"] = float(m["E"]), float(m["nu"])
    elif t == ORTHO:
        e = [float(v) for v in m["engineering"]]
        if len(e) != 9:
            raise ValueError("a type-1 material needs 9 engineering "
                             "constants, got %d" % len(e))
        out["engineering"] = e
    elif t == ANISO:
        C = np.asarray(m["C"], float)
        if C.shape != (6, 6):
            raise ValueError("a type-2 material needs a 6x6 C, got %s"
                             % (C.shape,))
        out["C"] = C
    else:
        raise ValueError("material type must be 0, 1 or 2, got %r" % t)
    for k in ("angle", "density"):
        if m.get(k) is not None and float(m[k]) != 0.0:
            out[k] = float(m[k])
    if "name" in m:
        out["name"] = str(m["name"])
    return out


def _mat_id_from_sets(sets, n_elem, materials, mat_names):
    """(n_elem,) 1-based material ids from the `sets: element:` block.

    Sets are matched to materials by NAME when every set name is a material
    name, and by POSITION otherwise -- the contract helper.msh_to_yaml
    writes (one set per material, same order).

    In:  sets list of {name, labels (1-based)}; n_elem int;
         materials list of canonical blocks; mat_names list[str | None]
    Out: (n_elem,) int, 1-based."""
    if len(sets) != len(materials):
        raise ValueError(
            "the yaml has %d element sets but %d materials -- the 2-D solid"
            " dialect pairs them one for one" % (len(sets), len(materials)))
    names = [str(s.get("name", "")) for s in sets]
    by_name = (all(n in mat_names for n in names)
               and len(set(names)) == len(names))
    mat_id = np.zeros(int(n_elem), int)
    for k, s in enumerate(sets):
        mid = (mat_names.index(names[k]) + 1) if by_name else (k + 1)
        lab = np.asarray([int(v) for v in s.get("labels", [])], int)
        if lab.size and (lab.min() < 1 or lab.max() > n_elem):
            raise ValueError(
                "element set %r references label %d outside 1..%d"
                % (names[k], int(lab.max()), n_elem))
        mat_id[lab - 1] = mid
    if (mat_id == 0).any():
        miss = int((mat_id == 0).sum())
        raise ValueError(
            "%d of %d elements belong to no element set, so they have no "
            "material -- an SG input has one material per element"
            % (miss, n_elem))
    return mat_id


def read_opensg_yaml(path, dim=None):
    """An OpenSG SOLID SG yaml -> the SG dict this module writes from.

    Both solid spellings are accepted (sg_mesh.sg_dialect calls them one
    dialect):

      canonical   `nodes` / `cells` (0-based) / `mat_id` (1-based) /
                  `materials` (a MAPPING id -> block) -- what
                  helper.sc_to_yaml emits and sg_mesh.load_sg_input reads.
      2-D solid   `nodes` / `elements` (1-based) / `sets: element:` /
                  `elementOrientations` / `materials` (a LIST) -- what
                  helper.msh_to_yaml and the opensg_io mesh pipelines emit.

    The msg-SHELL dialect (`sections` + `elementOrientations` + `elements`,
    i.e. a contour with a LAYUP) is refused: a layup is not a mesh, so
    there is no SG input to write from it -- go through
    opensg_shell.windio.sg_windio.emit_prevabs_xml and PreVABS instead.

    In:  path str -- the SG yaml (or an already-parsed SG dict, returned
         unchanged); dim int | None -- the SG dimension, None = inferred
         from the node columns that actually vary (load_sg_input's rule)
    Out: dict {dim int, nodes (N, 3) float, cells list of 0-based lists,
         mat_id (E,) int 1-based, materials {int: block}, orientation
         (E, 9) float | None (yaml component order), n_model int | None,
         refined int | None, omega float | None (the SG measure the header
         states FOR ITS OWN `n_model:` -- write_sc re-emits it only when
         that is the macro model being written, and re-measures
         otherwise), scale float}."""
    if isinstance(path, dict) and "cells" in path:
        sg = dict(path)
        sg.setdefault("orientation", None)
        sg.setdefault("n_model", None)
        sg.setdefault("refined", None)
        sg.setdefault("omega", None)
        sg.setdefault("scale", 1.0)
        return sg
    raw = _load_yaml(path)
    keys = set(raw)
    if all(k in keys for k in _SHELL_KEYS):
        raise ValueError(
            "%s is the msg-SHELL SG yaml dialect (sections + "
            "elementOrientations + elements): its `sections` carry a LAYUP,"
            " not a solid mesh, so there is no VABS/SwiftComp SG input to"
            " write from it directly.  Build the 2-D section first --"
            " opensg_shell.windio.sg_windio.emit_prevabs_xml + PreVABS --"
            " and write the resulting 2-D solid yaml instead." % path)

    nd = [[float(v) for v in _tokens(r)] for r in raw["nodes"]]
    w = max(len(r) for r in nd)
    nodes = np.zeros((len(nd), max(3, w)))
    for i, r in enumerate(nd):
        nodes[i, :len(r)] = r
    nodes = nodes[:, :3] if w <= 3 else nodes

    if "cells" in raw:
        cells = [[int(v) for v in _tokens(r)] for r in raw["cells"]]
    elif "elements" in raw:
        cells = [[int(v) - 1 for v in _tokens(r)] for r in raw["elements"]]
    else:
        raise ValueError(
            "%s carries neither a `cells:` nor an `elements:` connectivity"
            " block -- it is not an SG mesh" % path)
    n_elem = len(cells)

    mats_raw = raw.get("materials")
    if mats_raw is None:
        raise ValueError("%s carries no `materials:` block" % path)
    if isinstance(mats_raw, dict):
        materials = {int(k): _material_from_dict_entry(v)
                     for k, v in mats_raw.items()}
        mat_names = None
    else:
        blocks = [_material_from_list_entry(m) for m in mats_raw]
        materials = {k + 1: b for k, b in enumerate(blocks)}
        mat_names = [b.get("name") for b in blocks]

    if "mat_id" in raw:
        mat_id = np.asarray([int(v) for v in _tokens(raw["mat_id"])], int)
        if mat_id.size != n_elem:
            raise ValueError("%s: %d mat_id entries for %d elements"
                             % (path, mat_id.size, n_elem))
    elif raw.get("sets", {}).get("element"):
        mat_id = _mat_id_from_sets(_num(raw["sets"]["element"]), n_elem,
                                   list(materials.values()),
                                   mat_names or [])
    elif len(materials) == 1:
        mat_id = np.full(n_elem, min(materials), int)
    else:
        raise ValueError(
            "%s has %d materials but neither a `mat_id:` block nor element"
            " sets, so no element can be given a material"
            % (path, len(materials)))

    ori = raw.get("elementOrientations")
    if ori is not None:
        ori = np.asarray([[float(v) for v in _tokens(r)] for r in ori],
                         float)
        if ori.shape != (n_elem, 9):
            raise ValueError(
                "%s: elementOrientations is %s, expected (%d, 9)"
                % (path, (ori.shape,), n_elem))

    if dim is None:
        span = nodes[:, :3].max(axis=0) - nodes[:, :3].min(axis=0)
        used = span > 1e-9 * max(float(span.max()), 1.0)
        dim = int(np.nonzero(used)[0].max() + 1) if used.any() else 1
    hdr = _num({k: raw[k] for k in ("n_model", "refined", "scale", "omega")
                if k in raw})
    return {"dim": int(dim), "nodes": nodes, "cells": cells,
            "mat_id": mat_id, "materials": materials, "orientation": ori,
            "n_model": (None if hdr.get("n_model") is None
                        else int(hdr["n_model"])),
            "refined": (None if hdr.get("refined") is None
                        else int(hdr["refined"])),
            "omega": (None if hdr.get("omega") is None
                      else float(hdr["omega"])),
            "scale": float(hdr.get("scale", 1.0))}


# ------------------------------------------------- the VABS layup decomposition
def yaml_frames_to_beam(ori):
    """(E, 9) yaml `elementOrientations` -> (E, 3, 3) rows [e1; e2; e3] with
    components in BEAM order (x1 axial, x2, x3).

    The same component cycle sg_materials.elem_rotation_from_yaml applies:
    an SG yaml stores each axis as (y2, y3, axial) because its node rows are
    written that way, so (a, b, c) -> (c, a, b).

    In:  ori (E, 9) | (E, 3, 3) float, yaml component order
    Out: (E, 3, 3) float, row i = e_{i+1} in (x1, x2, x3)."""
    o = np.asarray(ori, float).reshape(-1, 3, 3)
    return o[:, :, [2, 0, 1]]


def vabs_angles_to_frame(theta1, theta3):
    """The VABS material frame of a (theta1, theta3) layup, in beam order.

    The ply frame is the beam frame rotated right-handed about x1 by
    theta1; the material frame is the ply frame rotated right-handed about
    y3 by theta3.  This is the exact inverse of frame_to_vabs_angles, and
    the pair is checked against VABS itself: rebuilding iea_s10.sg's frames
    from its (theta1, theta3) and homogenizing reproduces the Timoshenko
    6x6 in iea_s10.sg.K to 3.7e-11 (tests/msg_k_reference/
    test_sg_input_writer.py).

    In:  theta1, theta3 float [deg]
    Out: (3, 3) float, row i = e_{i+1} in (x1, x2, x3)."""
    c1, s1 = np.cos(np.radians(theta1)), np.sin(np.radians(theta1))
    c3, s3 = np.cos(np.radians(theta3)), np.sin(np.radians(theta3))
    y1 = np.array([1.0, 0.0, 0.0])
    y2 = np.array([0.0, c1, s1])
    y3 = np.array([0.0, -s1, c1])
    return np.stack([c3 * y1 + s3 * y2, -s3 * y1 + c3 * y2, y3])


def frame_to_vabs_angles(R, atol=1.0e-8):
    """A per-element material frame -> the VABS (theta1, theta3) layup.

    EXACT, not fitted: theta1 is read off the layer normal e3 and theta3
    off e1 in the ply plane, and the pair is then substituted back and
    compared with the input frame.  A frame that is not
    R1(theta1) . R3(theta3) -- e1 tilted out of the x1-y2 plane, a
    non-orthonormal triad, a left-handed triad -- CANNOT be said in a `.sg`
    and raises here instead of being silently projected onto one that can.

    In:  R (3, 3) float, rows [e1; e2; e3] in beam order (x1, x2, x3);
         atol float, the reconstruction tolerance
    Out: (theta1, theta3) floats [deg], theta1 in (-180, 180]."""
    R = np.asarray(R, float).reshape(3, 3)
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-6):
        raise ValueError(
            "the element frame is not orthonormal (R R^T deviates by %.3g)"
            " -- a VABS layup is a rotation" % np.abs(R @ R.T
                                                      - np.eye(3)).max())
    if np.linalg.det(R) < 0:
        raise ValueError(
            "the element frame is left-handed (det = %.3g); VABS theta1 /"
            " theta3 are right-handed rotations" % np.linalg.det(R))
    e1, e3 = R[0], R[2]
    t1 = np.degrees(np.arctan2(-e3[1], e3[2]))
    y2 = np.array([0.0, np.cos(np.radians(t1)), np.sin(np.radians(t1))])
    t3 = np.degrees(np.arctan2(float(e1 @ y2), e1[0]))
    err = np.abs(vabs_angles_to_frame(t1, t3) - R).max()
    if err > atol:
        raise ValueError(
            "the element frame is not a VABS layup: R1(%.6g) R3(%.6g) "
            "reproduces it only to %.3g.  A `.sg` can express a rotation "
            "about the beam axis followed by a rotation about the layer "
            "normal and nothing else -- e1 must stay in the x1-y2 plane."
            % (t1, t3, err))
    return float(t1), float(t3)


def element_layups(sg, angle_tol=1.0e-6):
    """The per-element (theta1, layer) table a VABS `.sg` is built from.

    A LAYER is a unique (material, theta3) pair -- the layer table of the
    two vendor `.sg` files here is exactly that: square_tube.sg has one
    layer (material 1, theta3 -45) for four different theta1, and
    iea_s10.sg has six, one per material, all at theta3 0.

    theta1 always comes from the element frame (0 without one).  theta3 is
    the SUM of the frame's own theta3 and the material block's `angle:` --
    both are rotations about y3, so they add, and the solver composes them
    the same way; see THETA3 AND THE MATERIAL `angle:` in the module
    docstring.  A type-2 block normally carries no `angle:`, its C being
    rotated already.

    In:  sg dict (read_opensg_yaml); angle_tol float, the theta3 rounding
         that decides whether two elements share a layer
    Out: (theta1 (E,) float [deg], layer_of (E,) int 1-based,
         layers list[(mat_id int, theta3 float)] in first-appearance
         order)."""
    mat_id = np.asarray(sg["mat_id"], int)
    n_elem = mat_id.size
    bad = sorted(set(int(m) for m in mat_id) - set(sg["materials"]))
    if bad:
        raise ValueError("element material id(s) %s have no material block"
                         " (blocks present: %s)"
                         % (bad, sorted(sg["materials"])))
    blk = np.array([float(sg["materials"][int(m)].get("angle", 0.0))
                    for m in mat_id])
    if sg.get("orientation") is None:
        t1, t3 = np.zeros(n_elem), blk
    else:
        R = yaml_frames_to_beam(sg["orientation"])
        ang = np.array([frame_to_vabs_angles(R[e]) for e in range(n_elem)])
        t1, t3 = ang[:, 0], ang[:, 1] + blk        # both are about y3
    q = np.round(np.asarray(t3, float) / angle_tol) * angle_tol
    layers, layer_of = [], np.zeros(n_elem, int)
    index = {}
    for e in range(n_elem):
        key = (int(mat_id[e]), float(q[e]))
        if key not in index:
            index[key] = len(layers) + 1
            layers.append(key)
        layer_of[e] = index[key]
    return np.asarray(t1, float), layer_of, layers


# ------------------------------------------------------------- the SG measure
def integrated_measure(nodes, cells, dim):
    """The summed ELEMENT measure of the mesh -- total length (1-D) or area
    (2-D), i.e. the material measure, which for a hollow or lattice SG is
    smaller than the bounding box.

    Exact for straight-sided elements (line2, tri3, quad4), which is what
    the corner nodes describe; a curved quad8/quad9 or tri6 edge is measured
    by its straight-sided corner polygon, so a strongly curved
    higher-order mesh is approximated here.

    In:  nodes (N, >=dim) float; cells list of 0-based connectivity lists;
         dim int (1 or 2)
    Out: float measure."""
    nd = np.asarray(nodes, float)
    total = 0.0
    for c in cells:
        p = nd[np.asarray(c, int)][:, :dim]
        if dim == 1:
            total += float(p[:, 0].max() - p[:, 0].min())
        else:
            q = p[:3] if len(c) in (3, 6) else p[:4]
            x, y = q[:, 0], q[:, 1]
            total += 0.5 * abs(float(np.dot(x, np.roll(y, -1))
                                     - np.dot(y, np.roll(x, -1))))
    return float(total)


def sg_omega(nodes, dim, n_model, cells=None):
    """The trailing `.sc` normalization omega, from the mesh and the model.

    omega is the measure of the domain the macro model does NOT resolve --
    the number the assembled energy is divided by to become a stiffness.
    This MIRRORS sg_assembly (compute_D_bar_and_omega / _solid_omega) term
    for term, because a `.sc` whose trailing line disagrees with the
    measure the solver applies to the same mesh says something the solver
    does not mean:

        n_model 3, 3-D SG   the node bounding-box VOLUME -- a periodic unit
                            cell is replaced by a continuum filling the
                            whole cell, holes included
        n_model 3, 1/2-D SG the INTEGRATED material measure of the SG's own
                            dimension -- NOT the bounding box.  For the
                            square tube (2-D SG, 1.03 outer, 0.03 wall)
                            that is the wall area 0.12, while the bounding
                            box is 1.0609: an 8.8x error if confused.
                            Needs `cells`; without them this falls back to
                            the bounding box and is only right when the SG
                            is solid through.
        n_model 2 (plate)   the in-plane span(s): ptp(y1) ptp(y2) for a 3-D
                            SG, ptp(y1) for a 2-D SG, 1.0 for a 1-D SG
        n_model 1 (beam)    ptp(y1) for a 3-D SG, 1.0 otherwise -- the
                            section itself is the measure

    Measured against this repo's decks: RHC_SW_2UC_45.sc's trailing 35.248
    is the y1 span of its 2-D plate SG (the mesh gives 35.248001, the
    vendor rounded), Plate_1D_SG_2UC_45.sc and the TPMS Sample_{1,2}.sc
    carry 1.0.

    In:  nodes (N, >=dim) float; dim int SG dimension; n_model 1|2|3;
         cells list | None -- required for the n_model 3, 1/2-D SG case
    Out: float omega (1.0 when the macro model resolves every SG
         coordinate)."""
    dim, n_model = int(dim), int(n_model)
    if n_model == 3 and dim < 3:
        if cells is None:
            nd = np.asarray(nodes, float)
            return float(np.prod((nd.max(axis=0) - nd.min(axis=0))[:dim]))
        return integrated_measure(nodes, cells, dim)
    keep = {3: dim, 2: dim - 1, 1: dim - 2}[n_model]
    if keep <= 0:
        return 1.0
    nd = np.asarray(nodes, float)
    span = nd.max(axis=0) - nd.min(axis=0)
    return float(np.prod(span[:keep]))


# ---------------------------------------------------------- number formatting
def _fe(p):
    """The %e field the writers use, `p` decimals in a width that keeps the
    columns aligned."""
    return "%%%d.%de" % (p + 8, p)


def _fb(p):
    """The same number format with no column padding (single-value lines)."""
    return "%%.%de" % p


def _renumber(materials, mat_id):
    """Material ids compacted to 1..nmate in sorted order.

    Both vendor codes size their material table by nmate and index it by
    the id on the element/layer line, so a sparse id set is a trap; the ids
    of an OpenSG yaml are already 1..n in every shipped file, which makes
    this a no-op there.

    In:  materials {id: block}; mat_id (E,) int 1-based
    Out: (materials {1..n: block}, mat_id (E,) int remapped)."""
    order = sorted(materials)
    if order == list(range(1, len(order) + 1)):
        return materials, np.asarray(mat_id, int)
    remap = {old: k + 1 for k, old in enumerate(order)}
    bad = sorted(set(int(m) for m in mat_id) - set(remap))
    if bad:
        raise ValueError("element material id(s) %s have no material block"
                         " (blocks present: %s)" % (bad, order))
    return ({remap[o]: materials[o] for o in order},
            np.asarray([remap[int(m)] for m in mat_id], int))


def _slots(conn, dim, width):
    """One zero-padded connectivity record, 1-BASED.

    The layout written is CONTIGUOUS fill: the element's nodes in slots
    1..n, zeros to the end of the record.  That is what every vendor deck
    in this tree does -- the 2-D decks (RHC_SW_2UC_45.sc, tri3/quad4 in
    slots 1-3 / 1-4 of 20) and all three 3-D decks (Sample_{1,2}.sc and
    preovios_try.sc, tet4 in slots 1-4 of 20).

    The 10-node tetrahedron is REFUSED, because it is the one case that is
    not a contiguous fill: the layout helper.sc_to_yaml.read_sc decodes
    (4 corners, slots 5-6 zero, the 6 midsides in slots 7-12) is this
    repo's own convention read back by this repo, and no shipped deck
    exercises it.  See NOT VALIDATED in the module docstring.

    In:  conn list[int] 1-based; dim int; width int slots
    Out: list[int] of length `width`."""
    n, row = len(conn), [0] * width
    if dim == 3:
        if n in (4, 8):
            row[:n] = conn
        elif n == 10:
            raise NotImplementedError(
                "a 10-node tetrahedron has no VALIDATED SwiftComp slot "
                "layout here.  Every 3-D deck in this tree -- Sample_1.sc "
                "and Sample_2.sc (the two SwiftComp demonstrably read, "
                "their .sc.k sit beside them) and preovios_try.sc -- is "
                "tet4 filling slots 1-4, so only the CONTIGUOUS fill is "
                "evidenced.  The split layout sc_to_yaml.read_sc decodes "
                "(corners 1-4, slots 5-6 zero, midsides 7-12) is this "
                "repo's own convention read back by this repo, and a wrong "
                "slot layout is silent -- the read is list-directed, so it "
                "runs on into the next record and shears the mesh.  Write "
                "the tet4 mesh, or pin the layout against a "
                "SwiftComp-accepted tet10 deck first.")
        else:
            raise ValueError(
                "a 3-D SG element with %d nodes has no slot convention here"
                " -- tet4 and hex8 are what this writer emits" % n)
    elif dim == 2:
        if n in (3, 4, 8, 9):
            row[:n] = conn
        else:
            raise ValueError(
                "a 2-D SG element with %d nodes has no unambiguous slot "
                "layout (slot 4 = 0 is what marks a triangle, so a 6-node "
                "triangle cannot be written straight) -- tri3, quad4, quad8"
                " and quad9 are supported" % n)
    else:
        if 2 <= n <= 5:
            row[:n] = conn
        else:
            raise ValueError("a 1-D SG element has 2..5 nodes, got %d" % n)
    return row


# --------------------------------------------------------- SwiftComp `.sc`
def _rotate_C_dcm(C, R):
    """A 6x6 stiffness rotated by a flattened (9,) direction-cosine matrix.

    The numpy twin of sg_materials.rotate_C_with_matrix, verbatim including
    its column-major r_ij indexing, so a baked material reproduces the
    solver's own C_asm exactly (tests/msg_k_reference asserts the two agree).

    In:  C (6, 6) float; R (9,) float, the frame in SOLVER order
    Out: (6, 6) float."""
    M = np.asarray(R, float).reshape(3, 3)
    r11, r12, r13 = M[0, 0], M[1, 0], M[2, 0]
    r21, r22, r23 = M[0, 1], M[1, 1], M[2, 1]
    r31, r32, r33 = M[0, 2], M[1, 2], M[2, 2]
    K = np.array([
        [r11**2, r12**2, r13**2, 2*r12*r13, 2*r11*r13, 2*r11*r12],
        [r21**2, r22**2, r23**2, 2*r22*r23, 2*r21*r23, 2*r21*r22],
        [r31**2, r32**2, r33**2, 2*r32*r33, 2*r31*r33, 2*r31*r32],
        [r21*r31, r22*r32, r23*r33, r22*r33 + r23*r32, r21*r33 + r23*r31,
         r21*r32 + r22*r31],
        [r11*r31, r12*r32, r13*r33, r12*r33 + r13*r32, r11*r33 + r13*r31,
         r11*r32 + r12*r31],
        [r11*r21, r12*r22, r13*r23, r12*r23 + r13*r22, r11*r23 + r13*r21,
         r11*r22 + r12*r21]])
    return K @ np.asarray(C, float) @ K.T


def _engineering_C(p):
    """(9,) engineering constants -> the 6x6 stiffness, in numpy.

    The same compliance inverse as sg_materials.build_single_C_matrix, in
    the SwiftComp Voigt order (11, 22, 33, 23, 13, 12).

    In:  p (9,) E1 E2 E3 G12 G13 G23 nu12 nu13 nu23
    Out: (6, 6) float."""
    E1, E2, E3, G12, G13, G23, v12, v13, v23 = (float(x) for x in p)
    S = np.zeros((6, 6))
    S[0, 0], S[1, 1], S[2, 2] = 1.0 / E1, 1.0 / E2, 1.0 / E3
    S[3, 3], S[4, 4], S[5, 5] = 1.0 / G23, 1.0 / G13, 1.0 / G12
    S[0, 1] = S[1, 0] = -v12 / E1
    S[0, 2] = S[2, 0] = -v13 / E1
    S[1, 2] = S[2, 1] = -v23 / E2
    return np.linalg.inv(S)


def material_C(block):
    """The 6x6 stiffness of ONE canonical material block, unrotated.

    In:  block dict {type 0 E/nu | 1 engineering | 2 C}
    Out: (6, 6) float."""
    t = int(block["type"])
    if t == ISO:
        E, nu = float(block["E"]), float(block["nu"])
        G = E / (2.0 * (1.0 + nu))
        return _engineering_C([E, E, E, G, G, G, nu, nu, nu])
    if t == ORTHO:
        return _engineering_C(block["engineering"])
    return np.asarray(block["C"], float)


def _rotate_C_angle(C, t):
    """A 6x6 stiffness rotated by the block `angle:` [deg] about y3.

    The numpy twin of sg_materials._rotate_C_np / rotate_C_matrix, verbatim
    including its R_Sig sign, so a folded angle reproduces the solver's own
    material table exactly.

    In:  C (6, 6) float; t float [deg]
    Out: (6, 6) float."""
    if float(t) == 0.0:
        return np.asarray(C, float)
    th = np.deg2rad(float(t))
    c, s = np.cos(th), np.sin(th)
    cs = c * s
    R_Sig = np.array([
        [c ** 2, s ** 2, 0, 0, 0, 2 * cs],
        [s ** 2, c ** 2, 0, 0, 0, -2 * cs],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, c, -s, 0],
        [0, 0, 0, s, c, 0],
        [-cs, cs, 0, 0, 0, c ** 2 - s ** 2]])
    return R_Sig @ np.asarray(C, float) @ R_Sig.T


def fold_angles(materials):
    """Material blocks with their `angle:` folded into the constants.

    An nlayer = 0 `.sc` has nowhere to say a per-material ply rotation (the
    layer table that would carry it exists only when nlayer /= 0), so a
    nonzero `angle:` is folded into the stiffness and the block is
    re-spelled as a pre-rotated type-2 one rather than dropped -- dropping
    it would silently emit the UNROTATED lamina.  Blocks with no angle are
    passed through untouched, so the common case re-spells nothing.

    In:  materials {id: canonical block}
    Out: ({id: block}, folded list[int] -- the ids that were re-spelled)."""
    out, folded = {}, []
    for mid, blk in materials.items():
        ang = float(blk.get("angle", 0.0))
        if ang == 0.0:
            out[mid] = blk
            continue
        new = {"type": ANISO, "C": _rotate_C_angle(material_C(blk), ang)}
        for k in ("density", "name"):
            if k in blk:
                new[k] = blk[k]
        out[mid] = new
        folded.append(int(mid))
    return out, folded


def _guard_sc_aux(materials, temperature, drop_density):
    """The `.sc` aux pair guard: refuse to write an unvalidated one.

    Every SwiftComp material block carries, per temperature, a two-number
    line before the constants.  This repo reads it as `T rho`
    (sc_to_yaml.read_sc), and this module writes it that way -- but that
    ORDER is supported by nothing in the tree.  All four third-party decks
    (RHC_SW_2UC_45.sc, Plate_1D_SG_2UC_45.sc and the two SwiftComp
    demonstrably read, Sample_{1,2}.sc) write `0 0`, which cannot tell the
    two orders apart; the decks here that DO carry a nonzero one
    (square_tube.sc, four_case/*.sc) were written by this module's own
    predecessor through this same convention, so they are circular.

    So: `0 0` is the only aux line this writer will emit unasked.  A
    nonzero density raises unless the caller passes drop_density=True, in
    which case the density is dropped (an elastic SwiftComp run does not
    use it) and reported -- never guessed into a slot.  A nonzero
    temperature has no such escape; pass 0.0.

    The VABS `.sg` density is a DIFFERENT field with its own line and is
    validated -- see _write_material_sg.

    In:  materials {id: block}; temperature float; drop_density bool
    Out: ({id: block} with the densities dropped when asked,
         dropped list[int] -- the ids whose nonzero density was zeroed)."""
    if float(temperature) != 0.0:
        raise NotImplementedError(
            "temperature=%r: the `.sc` material aux line is a PAIR and its "
            "order is not validated in this tree -- every third-party deck "
            "writes `0 0`, which cannot tell `T rho` from `rho T`.  Only "
            "0.0 can be written without guessing." % (temperature,))
    hot = sorted(int(m) for m, b in materials.items()
                 if float(b.get("density", 0.0)) != 0.0)
    if not hot:
        return materials, []
    if not drop_density:
        raise NotImplementedError(
            "material(s) %s carry a nonzero density, and WHERE a density "
            "goes in a `.sc` is not validated in this tree.  The aux line "
            "is two numbers; this repo reads them as `T rho`, but all four "
            "third-party decks write `0 0` -- which discriminates nothing "
            "-- and the decks here with a nonzero one were written by this "
            "writer's own predecessor, so they are circular evidence.  "
            "Emitting the density in the wrong slot would set a "
            "TEMPERATURE of %g and a density of 0 with nothing to flag it. "
            " Pass drop_density=True to write the `0 0` line every vendor "
            "deck actually contains (the stiffness is unaffected; an "
            "elastic run does not use the density), or write the VABS "
            "`.sg` dialect, whose density field IS validated against "
            "PreVABS."
            % (hot, float(materials[hot[0]]["density"])))
    out = {}
    for mid, blk in materials.items():
        if float(blk.get("density", 0.0)) != 0.0:
            blk = {k: v for k, v in blk.items() if k != "density"}
        out[mid] = blk
    return out, hot


def _bake_frames(sg, max_baked, atol):
    """Fold per-element frames into PRE-ROTATED type-2 materials.

    Every distinct (material, frame) pair becomes one type-2 block holding
    the stiffness the solver would have assembled -- the only way a
    trans_flag = 0 `.sc` can carry element frames without losing them.

    A block `angle:` is folded in FIRST and the element frame applied on
    top, which is the solver's own order (rotate_C_with_matrix of
    _rotate_C_np of C); the two are not in conflict and neither is dropped.

    In:  sg dict; max_baked int, the guard on the material count; atol
         float, the frame rounding that decides "distinct"
    Out: (materials {id: block}, mat_id (E,) int 1-based)."""
    R = yaml_frames_to_beam(sg["orientation"]).reshape(-1, 9)
    mat_id = np.asarray(sg["mat_id"], int)
    base = {m: _rotate_C_angle(material_C(b), b.get("angle", 0.0))
            for m, b in sg["materials"].items()}
    key = np.round(R / atol) * atol
    out, new_id, index = {}, np.zeros(mat_id.size, int), {}
    for e in range(mat_id.size):
        k = (int(mat_id[e]), tuple(key[e]))
        if k not in index:
            if len(out) >= int(max_baked):
                raise ValueError(
                    "orientation='bake' would need more than max_baked=%d "
                    "material blocks (the SG has that many distinct "
                    "(material, frame) pairs).  Raise max_baked, or write "
                    "the VABS `.sg` dialect, whose per-element theta1 says "
                    "this in one number." % max_baked)
            src = sg["materials"][int(mat_id[e])]
            blk = {"type": ANISO,
                   "C": _rotate_C_dcm(base[int(mat_id[e])], R[e])}
            if "density" in src:
                blk["density"] = src["density"]
            if "name" in src:
                blk["name"] = src["name"]
            index[k] = len(out) + 1
            out[index[k]] = blk
        new_id[e] = index[k]
    return out, new_id


def _write_material_sc(f, mid, blk, temperature, fe):
    """One SwiftComp material block: `mat_id isotropy ntemp`, `T rho`, the
    constants -- the shape read off the vendor decks (RHC_SW_2UC_45.sc
    heads a type-2 block `1 2 1`, Sample_1.sc a type-0 block `1 0 1`, both
    with the aux pair on the next line, BEFORE the constants) and the
    shape helper.sc_to_yaml.read_sc parses back.

    The PAIR's shape and position are validated; the ORDER of its two
    numbers is not, so _guard_sc_aux has already ensured both are 0.0 by
    the time this runs -- the `0 0` line every third-party deck contains.

    In:  f open file; mid int; blk canonical block; temperature float;
         fe str number format
    Out: None."""
    t = int(blk["type"])
    f.write("%d %d 1\n" % (int(mid), t))
    f.write((fe + fe + "\n") % (float(temperature),
                                float(blk.get("density", 0.0))))
    if t == ISO:
        f.write((fe + fe + "\n") % (blk["E"], blk["nu"]))
    elif t == ORTHO:
        e = blk["engineering"]
        for k in range(0, 9, 3):
            f.write((fe * 3 + "\n") % tuple(e[k:k + 3]))
    else:
        C = np.asarray(blk["C"], float)
        for r in range(6):
            f.write((fe * (6 - r) + "\n") % tuple(C[r, r:]))
    f.write("\n")


def write_sc(sg, path, n_model=None, refined=None, analysis=0, elem_flag=0,
             temp_flag=0, curvature=None, omega=None,
             temperature=0.0, orientation=None, max_baked=64,
             frame_atol=1.0e-9, precision=14, drop_density=False,
             comments=False):
    """Write an SG as a SwiftComp `.sc`.

    The layout is the one the vendor decks in this repo use (named in the
    module docstring), and only the shape helper.sc_to_yaml.read_sc reads
    back is written by default: nslave = 0 (SwiftComp pairs the periodic
    boundary itself) and nlayer = 0 (the element record carries the
    material id directly).

    In:
        sg: dict from read_opensg_yaml (or a path / an sc dict).
        path: str, the `.sc` to write.
        n_model: 2 plate | 3 solid -- the MACRO model.  It is not a field
            of the file; it decides how many header lines the file has and
            how omega is measured, and it is the `2D`/`3D` argument the
            SwiftComp run must then be given.  None takes the yaml
            header's `n_model:`, defaulting to 2 (plate).  1 (beam) is
            REFUSED: no deck in this tree carries a beam header, so its
            layout is unvalidated here -- see NOT VALIDATED in the module
            docstring.
        refined: 0 classical | 1 shear-refined -- the `submodel` line, which
            every macro model carries (see THE SUBMODEL LINE in the module
            docstring; a 3-D macro model has the line but no meaning for
            the value, and both SwiftComp-accepted 3-D decks here write 0).
            None takes the yaml header's `refined:`, defaulting to 0.
        analysis: int, the SwiftComp analysis code (0 = elastic).
        elem_flag: int, 0 = regular elements (the only kind implemented).
        temp_flag: int, 0 = uniform temperature.
        curvature: sequence | None -- (k12, k21) for the plate macro
            model; None = flat (zeros, which is what every vendor deck
            here carries).
        omega: float | None, the trailing normalization.  None takes the
            yaml header's own `omega:` ONLY when the macro model being
            written is the one that header describes; override n_model and
            the measure is re-derived with sg_omega for the model actually
            being written.  See THE TRAILING omega in the module
            docstring.
        temperature: float, the T of the single material property set.
            Only 0.0 is validated -- every vendor deck writes `0 0`.
        orientation: "bake" | "ignore" -- REQUIRED when the SG carries
            per-element frames, see the module docstring.  "points" (the
            trans_flag=1 a/b/c block) is named and REFUSED: no deck here
            is trans_flag=1, so its component ordering is unvalidated.
        max_baked: int, the guard on how many materials "bake" may create.
        frame_atol: float, the rounding that decides two frames are one.
        precision: int, decimals of the %e number format (15 significant
            digits is what SwiftComp reads).
        drop_density: bool.  A nonzero material density is REFUSED by
            default (the aux pair's order is unevidenced -- all four
            third-party decks write `0 0`).  True writes that evidenced
            `0 0` line instead of raising, losing only the density, which
            an elastic run does not use.
        comments: bool -- REFUSED when True, see the module docstring.
    Out:
        dict {path, dim, n_nodes, n_elems, n_mats, n_model, refined,
        omega, omega_source, trans_flag, orientation, folded_angles,
        dropped_density} -- what was actually written.  `omega_source` is
        "header" | "measured" | "argument"; `folded_angles` lists the
        material ids whose `angle:` was folded into a pre-rotated type-2
        block (see fold_angles); `dropped_density` lists the ids whose
        nonzero density was zeroed under drop_density.
    """
    if comments:
        raise NotImplementedError(_NO_COMMENTS % ("SwiftComp `.sc`", "#"))
    sg = read_opensg_yaml(sg) if not isinstance(sg, dict) else sg
    dim = int(sg["dim"])
    if dim not in (1, 2, 3):
        raise ValueError("an SG is 1-, 2- or 3-D, got dim=%d" % dim)
    # the macro model the HEADER describes -- what its `omega:` measures
    hdr_model = 2 if sg.get("n_model") is None else int(sg["n_model"])
    n_model = hdr_model if n_model is None else int(n_model)
    if n_model not in (1, 2, 3):
        raise ValueError("n_model must be 1 (beam), 2 (plate) or 3 (solid),"
                         " got %r" % n_model)
    if n_model == 1:
        raise NotImplementedError(
            "the BEAM macro-model `.sc` header is not validated in this "
            "tree, so this writer will not emit one.  A beam header is "
            "said to be the submodel line, a THREE-value curvature line "
            "(k11 k12 k13) and an oblique cosine pair before the control "
            "line -- but no deck here has it: the four third-party decks "
            "are a 2-D/1-D PLATE header (submodel + a 2-value curvature "
            "line) and a 3-D one (submodel only), and even this repo's "
            "own beam examples (examples/OpenSG-solid/7_get_beam_props_"
            "from_SG) are fed the byte-identical PLATE-header "
            "RHC_SW_2UC_45.sc.  Getting a header line wrong is silent and "
            "total, because SwiftComp's read is list-directed.  Write the "
            "VABS `.sg` dialect for a beam cross-section -- it is the "
            "beam format and it IS validated field for field -- or pin "
            "the beam header against a SwiftComp-accepted beam deck.")
    refined = (int(sg.get("refined") or 0) if refined is None
               else int(refined))

    materials, mat_id = dict(sg["materials"]), np.asarray(sg["mat_id"], int)
    trans = 0                        # trans_flag: every deck in this tree
    if sg.get("orientation") is not None:
        if orientation not in _ORIENTATION_MODES:
            raise ValueError(
                "this SG carries per-element `elementOrientations`, which a "
                "trans_flag=0 `.sc` has no slot for.  State what to do with "
                "them: orientation='bake' (fold each frame into its own "
                "pre-rotated type-2 material -- exact, and read_sc reads it "
                "back) or 'ignore' (drop them; correct only when they are "
                "all the identity).  'points' (the trans_flag=1 a/b/c "
                "block) is a recognised name but is REFUSED as "
                "unvalidated.  Got orientation=%r." % (orientation,))
        if orientation == "points":
            raise NotImplementedError(
                "orientation='points' -- the native SwiftComp trans_flag=1 "
                "a/b/c block -- is not validated in this tree.  EVERY deck "
                "here is trans_flag=0, so nothing says what coordinate "
                "ordering the three points are written in, and the two "
                "candidates disagree: the node columns of this same file "
                "are the SG's own (y1, y2[, y3]) ordering, while the "
                "frames this module carries are in BEAM order (x1 axial "
                "first, the [2, 0, 1] cycle of yaml_frames_to_beam).  A "
                "single fixed choice cannot be right for the 1-D, 2-D and "
                "3-D SG at once, and the error is silent -- the file "
                "parses and the material axes point somewhere else.  Use "
                "orientation='bake' (exact, reader-visible, and read_sc "
                "reads it back) or 'ignore' (only when the frames are all "
                "the identity); for a beam cross-section the VABS `.sg` "
                "dialect says the same thing in one validated number per "
                "element.")
        if orientation == "bake":
            materials, mat_id = _bake_frames(sg, max_baked, frame_atol)
    materials, folded = fold_angles(materials)
    materials, dropped = _guard_sc_aux(materials, temperature, drop_density)
    if omega is not None:
        omega, omega_source = float(omega), "argument"
    elif sg.get("omega") is not None and n_model == hdr_model:
        # the header states the measure for THIS macro model -- re-emit it
        omega, omega_source = float(sg["omega"]), "header"
    else:
        # no header measure, or the caller overrode the macro model the
        # header's measure belongs to: re-derive it for the model actually
        # being written (see THE TRAILING omega in the module docstring)
        omega = sg_omega(sg["nodes"], dim, n_model, sg["cells"])
        omega_source = "measured"
    materials, mat_id = _renumber(materials, mat_id)

    nodes = np.asarray(sg["nodes"], float)
    fe, fb = _fe(int(precision)), _fb(int(precision))
    width = SC_SLOTS

    with open(path, "w") as f:
        # the submodel line is there for EVERY macro model, 3-D included --
        # see THE SUBMODEL LINE in the module docstring for the evidence
        f.write("%d\n" % refined)
        if n_model == 2:              # only a plate has a curvature line
            k = ([float(v) for v in curvature] if curvature is not None
                 else [0.0, 0.0])
            f.write(" ".join(fb % v for v in k) + "\n")
        f.write("%d %d %d %d\n\n" % (int(analysis), int(elem_flag), trans,
                                     int(temp_flag)))
        f.write("%d %d %d %d 0 0\n\n"
                % (dim, len(nodes), len(sg["cells"]), len(materials)))

        for i, x in enumerate(nodes):
            f.write(("%d" + fe * dim + "\n") % ((i + 1,) + tuple(x[:dim])))
        f.write("\n")

        for e, c in enumerate(sg["cells"]):
            row = _slots([int(v) + 1 for v in c], dim, width)
            f.write("%d %d %s\n" % (e + 1, int(mat_id[e]),
                                    " ".join(str(v) for v in row)))
        f.write("\n")

        for mid in sorted(materials):
            _write_material_sc(f, mid, materials[mid], temperature, fe)
        f.write((fb + "\n") % float(omega))

    return {"path": path, "dim": dim, "n_nodes": len(nodes),
            "n_elems": len(sg["cells"]), "n_mats": len(materials),
            "n_model": n_model, "refined": refined, "omega": float(omega),
            "omega_source": omega_source, "trans_flag": trans,
            "orientation": orientation, "folded_angles": folded,
            "dropped_density": dropped}


# --------------------------------------------------------------- VABS `.sg`
def _write_material_sg(f, mid, blk, fe):
    """One VABS material block: `mat_id orth`, the constants, then the
    density on its OWN line and no temperature line -- the shape both
    PreVABS decks here write (square_tube.sg / iea_s10.sg: `1 1`, three
    rows of three, then the density).

    Unlike its `.sc` counterpart the density here is VALIDATED at a
    NONZERO value, which is why write_sg needs no drop_density guard:
    square_tube.sg carries 1.600000e+03 and iea_s10.sg 1.6e3 / 1.94e3 on
    exactly this line, and the writer reproduces both files field for
    field including those numbers.  There is no ordering ambiguity because
    the density is alone on its line -- the `.sc` aux PAIR is the case
    that cannot be pinned; see _guard_sc_aux.

    NOTE the orth 0 (`E nu`) and orth 2 (21 constants) layouts are the
    SwiftComp ones by analogy; every vendor `.sg` in this repo is orth 1,
    so those two are UNVERIFIED here (module docstring, NOT VALIDATED and
    currently emitted anyway).

    In:  f open file; mid int; blk canonical block; fe str number format
    Out: None."""
    t = int(blk["type"])
    f.write("%8d%8d\n" % (int(mid), t))
    if t == ISO:
        f.write((fe + fe + "\n") % (blk["E"], blk["nu"]))
    elif t == ORTHO:
        e = blk["engineering"]
        for k in range(0, 9, 3):
            f.write((fe * 3 + "\n") % tuple(e[k:k + 3]))
    else:
        C = np.asarray(blk["C"], float)
        for r in range(6):
            f.write((fe * (6 - r) + "\n") % tuple(C[r, r:]))
    f.write((fe + "\n") % float(blk.get("density", 0.0)))


def write_sg(sg, path, timoshenko=1, damping=0, thermal=0, curvature=None,
             oblique=None, trapeze=0, vlasov=0, angle_tol=1.0e-6,
             precision=9, comments=False):
    """Write an SG as a VABS `.sg` (the new, format_flag = 1 layout).

    VABS is a BEAM code: the SG must be the 2-D cross-section, so a 1-D or
    3-D SG is refused rather than reinterpreted.  Unlike a `.sc`, this file
    needs no orientation argument -- theta1 is per element by construction
    and every distinct (material, theta3) pair becomes a LAYER.

    In:
        sg: dict from read_opensg_yaml (or a path / an sc dict).
        path: str, the `.sg` to write.
        timoshenko: 0 classical only | 1 also the Timoshenko model.
        damping: 0 | 1 -- 1 additionally expects a damping coefficient per
            layer, which this writer does not have, so only 0 is accepted.
        thermal: 0 mechanical | 3 one-way thermoelastic -- 3 additionally
            expects a CTE per material, so only 0 is accepted.
        curvature: (k1, k2, k3) | None -- None writes curve_flag 0.
        oblique: (cos1, cos2) | None -- None writes oblique_flag 0.
        trapeze, vlasov: 0 | 1, the two remaining VABS model flags.
        angle_tol: float, the theta3 rounding that merges two layers.
        precision: int, decimals of the %e number format (PreVABS writes 6).
        comments: bool -- REFUSED when True.  `!` is the VABS comment
            character by report, but neither PreVABS deck in this tree
            writes one, so nothing here shows VABS skips the record.
    Out:
        dict {path, n_nodes, n_elems, n_mats, n_layers, layers} -- what was
        actually written; `layers` is the (mat_id, theta3) table.
    """
    if comments:
        raise NotImplementedError(_NO_COMMENTS % ("VABS `.sg`", "!"))
    sg = read_opensg_yaml(sg) if not isinstance(sg, dict) else sg
    if int(sg["dim"]) != 2:
        raise ValueError(
            "a VABS `.sg` is a BEAM cross-section, so its SG is 2-D; this "
            "one is %d-D.  Write the SwiftComp `.sc` dialect instead -- it "
            "takes a 1-D, 2-D or 3-D SG." % int(sg["dim"]))
    if int(damping) not in (0,) or int(thermal) not in (0,):
        raise ValueError(
            "damping=1 needs a damping coefficient per layer and thermal=3 "
            "a CTE per material; neither is in an OpenSG SG yaml, so this "
            "writer emits damping=0 thermal=0 only")
    for c in sg["cells"]:
        if len(c) not in (3, 4):
            raise ValueError(
                "VABS 2-D elements are tri3 or quad4 here (slot 4 = 0 is "
                "what marks a triangle, so a 6-node triangle has no "
                "unambiguous slot layout); got %d nodes" % len(c))

    materials, mat_id = _renumber(dict(sg["materials"]),
                                  np.asarray(sg["mat_id"], int))
    sg = dict(sg, materials=materials, mat_id=mat_id)
    t1, layer_of, layers = element_layups(sg, angle_tol)
    for mid, _t3 in layers:
        if int(materials[mid]["type"]) == ANISO and _t3 != 0.0:
            raise ValueError(
                "material %d is a stored 6x6 C (type 2, already rotated) "
                "but the SG asks for theta3 = %.6g on it -- VABS would "
                "rotate it a second time" % (mid, _t3))
    nodes = np.asarray(sg["nodes"], float)
    fe = _fe(int(precision))

    with open(path, "w") as f:
        f.write("%8d%8d\n\n" % (1, len(layers)))
        f.write("%8d%8d%8d\n\n" % (int(timoshenko), 0, 0))
        f.write("%8d%8d%8d%8d\n\n"
                % (0 if curvature is None else 1,
                   0 if oblique is None else 1, int(trapeze), int(vlasov)))
        if curvature is not None:
            f.write((fe * 3 + "\n\n") % tuple(float(v) for v in curvature))
        if oblique is not None:
            f.write((fe * 2 + "\n\n") % tuple(float(v) for v in oblique))

        f.write("%8d%8d%8d\n\n"
                % (len(nodes), len(sg["cells"]), len(materials)))

        for i, x in enumerate(nodes):
            f.write(("%8d" + fe * 2 + "\n") % (i + 1, x[0], x[1]))
        f.write("\n")

        for e, c in enumerate(sg["cells"]):
            row = _slots([int(v) + 1 for v in c], 2, VABS_SLOTS)
            f.write("%8d%s\n" % (e + 1, "".join("%8d" % v for v in row)))
        f.write("\n")

        for e in range(len(sg["cells"])):
            f.write(("%8d%8d" + fe + "\n")
                    % (e + 1, int(layer_of[e]), float(t1[e])))
        f.write("\n")

        for k, (mid, t3) in enumerate(layers):
            f.write(("%8d%8d" + fe + "\n") % (k + 1, int(mid), float(t3)))
        f.write("\n")

        for mid in sorted(materials):
            _write_material_sg(f, mid, materials[mid], fe)

    return {"path": path, "n_nodes": len(nodes),
            "n_elems": len(sg["cells"]), "n_mats": len(materials),
            "n_layers": len(layers), "layers": layers}


# ------------------------------------------------------------------ front door
def convert(yaml_path, out_path=None, dialect="sc", **kw):
    """An OpenSG SG yaml -> an SG input file in the named dialect.

    The ONE yaml -> SG-input entry point of the repo.  The dialect is
    always explicit: `.sc` and `.sg` are different files (see the module
    docstring), and picking one from a file extension would be exactly the
    silent conflation this module exists to avoid.

    In:  yaml_path str -- the SG yaml (or an already-parsed SG dict);
         out_path str | None -- the output; None -> the yaml stem plus the
         dialect's extension (a dict input then requires out_path);
         dialect "sc" (SwiftComp) | "sg" (VABS); **kw -- passed to
         write_sc / write_sg
    Out: dict, the writer's report (its `path` is the file written)."""
    if dialect not in ("sc", "sg"):
        raise ValueError("dialect must be 'sc' (SwiftComp) or 'sg' (VABS),"
                         " got %r" % (dialect,))
    if out_path is None:
        if not isinstance(yaml_path, str):
            raise ValueError("out_path is required when the SG comes in as"
                             " a parsed dict")
        out_path = os.path.splitext(yaml_path)[0] + "." + dialect
    sg = read_opensg_yaml(yaml_path)
    w = write_sc if dialect == "sc" else write_sg
    return w(sg, out_path, **kw)


def convert_yaml_to_sc(yaml_path, out_path=None, dim=2, **kw):
    """The legacy rm_plate_1D.helper.yaml_to_sc.convert signature.

    Kept so the historical import path keeps working unchanged; new code
    should call convert(..., dialect="sc") and state `orientation=` for an
    SG that carries element frames.

    In:  yaml_path str; out_path str | None (None -> the yaml stem + .sc);
         dim int -- the SG dimension override; **kw -> write_sc
    Out: str, the path written (the legacy return value)."""
    if out_path is None:
        out_path = os.path.splitext(yaml_path)[0] + ".sc"
    sg = read_opensg_yaml(yaml_path, dim=dim)
    r = write_sc(sg, out_path, **kw)
    print("yaml_to_sc: %dD SG, %d nodes, %d cells, %d materials -> %s"
          % (r["dim"], r["n_nodes"], r["n_elems"], r["n_mats"], r["path"]))
    return r["path"]
