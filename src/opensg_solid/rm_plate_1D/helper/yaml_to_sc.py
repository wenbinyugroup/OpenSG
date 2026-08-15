"""MOVED -- the yaml -> SG-input writer now lives in
opensg_solid.io.sg_input (the solid-side `helper` subpackage, beside
sc_to_yaml.py, which reads the same files back, and k_file.py); this
module re-exports it so the historical import path keeps working.

WHAT CHANGED, and why the move rather than a rename in place.  The old
module wrote only a SwiftComp `.sc`, predated the current yaml header
(msg / n_model / refined) and had three defects a second, disagreeing
yaml -> SG-input path would have kept alive:

  * every material was written with isotropy flag 0 (isotropic, `E nu`)
    and then given NINE orthotropic constants, so the reader took E1/E2
    as (E, nu) and the remaining seven lines desynchronized the block;
  * the element record carried only (nodes + 1) connectivity slots
    instead of the 20 every vendor deck writes at every SG dimension,
    which a list-directed read fills by running on into the NEXT line;
  * no trailing omega line at all, and no way at all to say a
    per-element material frame.

opensg_solid.io.sg_input fixes all three, adds the VABS `.sg`
dialect beside the SwiftComp one, and states every place the two differ.
`convert` below keeps the legacy (yaml_path, out_path, dim) signature and
its printed one-liner; new code should call
`opensg_solid.io.sg_input.convert(path, dialect="sc" | "sg")`.

NOTE for the one caller in the tree (tests/08072026_square_shell_mesh/
four_case/make_inputs.py): the new writer REFUSES rather than guesses.
Those yamls carry `elementOrientations` and a nonzero density, so the
call now needs `orientation="bake"` (or "ignore") and
`drop_density=True`; see VALIDATED / NOT VALIDATED in the sg_input module
docstring for why each is a refusal and not a default.
"""
from opensg_solid.io.sg_input import (             # noqa: F401
    convert_yaml_to_sc as convert, read_opensg_yaml, write_sc, write_sg)
