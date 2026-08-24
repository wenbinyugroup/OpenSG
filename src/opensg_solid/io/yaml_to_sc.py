"""yaml_to_sc.py -- OpenSG SG yaml -> SwiftComp `.sc`, the named io module
behind the `opensg yaml_to_sc <yaml>` subcommand (the exact reverse of
io.sc_to_yaml).

Everything format-critical lives in io.sg_input -- the ONE yaml ->
SG-input writer, whose `.sc` layout is measured field for field off the
third-party decks in this tree (RHC_SW_2UC_45.sc, Plate_1D_SG_2UC_45.sc,
Sample_{1,2}.sc) and cross-checked against the SwiftComp 2.1 user manual
(SCManual.pdf, sections 8.1-8.2: submodel + curvature header lines for a
plate, the `analysis elem_flag trans_flag temp_flag` control line, the
`nSG nnode nelem nmate nslave nlayer` size line, nSG coordinate columns
per node, zero-padded element records, `mat_id isotropy ntemp` material
blocks with the `T rho` aux line, and the trailing omega -- for a plate
2-D SG the y2 span, not the area).  This module only fixes the pieces the
subcommand promises:

    the macro model and the SG dimension come from the YAML ITSELF --
    `n_model:` (2 plate | 3 solid; 1 beam is refused by sg_input, whose
    docstring says why) and `refined:` (0 classical -> submodel 0,
    1 shear-refined -> submodel 1) from the header, the SG dimension from
    the node columns that actually vary.

A material `angle:` is FOLDED into a pre-rotated type-2 (21-constant)
block by sg_input.fold_angles -- the emitted `.sc` carries the same
physical stiffness the OpenSG engine assembles, with no dependence on
either code's layup-angle sign convention.  Per-element frames
(`elementOrientations`) need an explicit `orientation=` choice ("bake" |
"ignore"); the sg_input error text explains the options when it is
missing.

In:  the canonical SG yaml (nodes/cells/mat_id/materials) or the 2-D
     solid mesh dialect (elements/sets/elementOrientations)
Out: <stem>.sc, runnable as `SwiftComp <stem>.sc 2D H` (n_model 2) or
     `SwiftComp <stem>.sc 3D H` (n_model 3)
"""
import os

from .sg_input import convert as _convert
from .sg_input import read_opensg_yaml, write_sc  # noqa: F401  (re-export)


def convert(yaml_path, out_path=None, **kw):
    """One SG yaml -> `.sc`, with the report printed the converter way.

    In:  yaml_path str -- the SG yaml; out_path str | None (None -> the
         yaml stem + .sc); **kw -> sg_input.write_sc (orientation=,
         drop_density=, curvature=, ...).  n_model/refined are NOT
         accepted here: the subcommand's contract is that the yaml header
         decides them (call sg_input.convert directly to override).
    Out: dict, the write_sc report (path, dim, n_nodes, n_elems, n_mats,
         n_model, refined, omega, omega_source, folded_angles, ...)."""
    for k in ("n_model", "refined"):
        if k in kw:
            raise TypeError(
                "yaml_to_sc.convert takes %s from the yaml header itself"
                " (the subcommand contract); call io.sg_input.convert"
                " with dialect='sc' to override it" % k)
    r = _convert(yaml_path, out_path=out_path, dialect="sc", **kw)
    print("yaml_to_sc: %dD SG, %d nodes, %d cells, %d materials,"
          " n_model %d (%s), omega %.10g (%s) -> %s"
          % (r["dim"], r["n_nodes"], r["n_elems"], r["n_mats"],
             r["n_model"], "shear-refined" if r["refined"] else "classical",
             r["omega"], r["omega_source"], r["path"]))
    if r["folded_angles"]:
        print("yaml_to_sc: material `angle:` folded into pre-rotated"
              " type-2 blocks for ids %s (an nlayer=0 .sc has no angle"
              " slot; the stiffness is identical)" % r["folded_angles"])
    print("yaml_to_sc: run as  SwiftComp %s %dD H"
          % (os.path.basename(r["path"]), 2 if r["n_model"] == 2 else 3))
    return r
