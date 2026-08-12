"""Gate: the yaml -> SG-INPUT writer (opensg_solid.helper.sg_input).

The failure mode this exists to prevent is NOT a crash.  It is a file that
parses and means something else -- a connectivity record one slot short, a
header line missing, a ply rotation applied once instead of twice -- which
no amount of "it imported and ran" will catch.  So every claim here is
pinned to a file some other code actually accepted, and every substantive
assertion carries a negative control.

REFERENCES, and how much authority each carries:

    examples/OpenSG-solid/3_get_plate_props_from_2D_SG/
        RHC_SW_2UC_45.sc            2-D SG, plate model, three type-2
                                    (21-constant, pre-rotated) materials
        Plate_1D_SG_2UC_45.sc       1-D SG, plate model
    tests/msg_shell_solid_prop/SC_files/SC_files/Sample_{1,2}.sc
                                    3-D SG.  THE strongest evidence in the
                                    tree: SwiftComp's own output for these
                                    two sits beside them as
                                    tests/08072026_square_shell_mesh/
                                    TPMS_SC_files/Sample_{1,2}.sc.k, so
                                    SwiftComp demonstrably read them.  Where
                                    a layout rule is in doubt, these decide.
    tests/08072026_square_shell_mesh/prevabs/square_tube.sg
                                    VABS `.sg` written by PreVABS for the
                                    same section as
                                    examples/OpenSG-solid/5_get_solid_props_
                                    from_SG/square_section/
                                    square_tube_2Dsolid.yaml -- same 3228
                                    nodes and 5384 elements in the same
                                    order, so the two are diffable element
                                    by element.
    examples/OpenSG_shell/windio/vabs_K/iea_s10.sg
                                    a second, much larger PreVABS `.sg`
                                    (6 materials, 6 layers, per-element
                                    theta1 over the whole range).

THE TWO LAYOUT RULES THAT WERE WRONG AND ARE PINNED HERE.

 1. The `.sc` element record is 20 slots wide for EVERY SG dimension, not
    5/9/20 by dimension.  RHC_SW_2UC_45.sc is a 2-D SG and pads to 20;
    Sample_{1,2}.sc pad to 20.  Nothing in the tree pads a 2-D SG to 9.
 2. The submodel line is present for a 3-D macro model too.  Sample_{1,2}.sc
    begin `0` / `0 0 0 0` / the size line, and SwiftComp's read is
    list-directed, so a stray leading record would have been swallowed as
    the control line's first integer and the size line would have come back
    as nSG = 0 -- which could not have produced the sane 6x6 in the .sc.k.

THE SILENT DEFECT THAT WAS FOUND AND IS PINNED HERE.  The trailing omega
was taken from the yaml header even when the caller OVERRODE n_model, so a
yaml stating `n_model: 3` / `omega: 7.0` written at n_model=2 emitted 7.0
where the plate measure of that same mesh is 6.0 -- a wrong normalization
in a solver input with nothing to flag it.  See
test_omega_is_remeasured_when_the_macro_model_is_overridden, which fails
on the old code.

THE PATHS THAT REFUSE RATHER THAN GUESS.  Four fields are positionally
plausible and supported by NOTHING in this tree: the beam macro-model
`.sc` header, orientation="points" (the trans_flag=1 a/b/c block), a
nonzero `.sc` material density (the aux PAIR's order), and a tet10 `.sc`
record -- plus comments in either dialect.  Each raises, and
test_writer_refuses_the_unvalidated_paths re-measures the evidence for
each refusal off the vendor files, so the day a deck arrives that DOES
settle one, the test that guards it fails and says so.

THE PHYSICS RULE THAT WAS WRONG.  A yaml may carry per-element frames AND a
material `angle:`; they do not conflict.  Both are rotations about y3 --
theta3 from the frame and the block angle add -- and the solver composes
them in that order (sg_materials: rotate_C_with_matrix of _rotate_C_np of
C).  The repo's own driver does exactly this: examples/.../square_section/
solid_props_new_architecture.py passes angles=[-45] together with
elem_rotation_from_yaml(ori).

Fast: no homogenization.  The three tests that parse a multi-MB SG are
marked `slow` and deselected by `-m "not slow"` like the rest of the repo.

Run:  pytest tests/msg_k_reference -q -m "not slow"
      pytest tests/msg_k_reference -q                 (everything)
(env opensg_2_0)
"""
import inspect
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

S3 = os.path.join(ROOT, "examples", "OpenSG-solid",
                  "3_get_plate_props_from_2D_SG")
RHC = os.path.join(S3, "RHC_SW_2UC_45.sc")
P1D = os.path.join(S3, "Plate_1D_SG_2UC_45.sc")
TPMS = os.path.join(ROOT, "tests", "msg_shell_solid_prop", "SC_files",
                    "SC_files")
SQ_SG = os.path.join(ROOT, "tests", "08072026_square_shell_mesh", "prevabs",
                     "square_tube.sg")
SQ_YAML = os.path.join(ROOT, "examples", "OpenSG-solid",
                       "5_get_solid_props_from_SG", "square_section",
                       "square_tube_2Dsolid.yaml")
IEA_SG = os.path.join(ROOT, "examples", "OpenSG_shell", "windio", "vabs_K",
                      "iea_s10.sg")

# the square tube: 1.03 outer, 0.03 wall -> the four wall frames PreVABS
# writes as theta1 = 0 / 90 / 180 / -90, and the ply it writes as theta3
ENG = [142.0e9, 9.8e9, 9.8e9, 6.0e9, 6.0e9, 4.8e9, 0.30, 0.30, 0.42]


# ------------------------------------------------------------------ helpers
def sc_fields(path):
    """The FIELD layout of a `.sc`, read straight off the text -- the model
    header lines, the META integers, the coordinate columns and the
    connectivity slot width."""
    L = [ln.strip() for ln in open(path) if ln.strip()]
    mi = next(i for i, ln in enumerate(L) if len(ln.split()) >= 5)
    meta = [int(v) for v in L[mi].split()[:6]]
    nn, ne = meta[1], meta[2]
    e0 = mi + 1 + nn
    return {
        "header": L[:mi],
        "meta": meta,
        "ncoord": sorted({len(L[mi + 1 + k].split()) - 1 for k in range(nn)}),
        "slots": sorted({len(L[e0 + k].split()) - 2 for k in range(ne)}),
        "nonzero": sorted({len([t for t in L[e0 + k].split()[2:] if t != "0"])
                           for k in range(ne)}),
        "mat_head": L[e0 + ne],
        "trailing": float(L[-1].split()[0]),
    }


def sg_fields(path):
    """Every field of a VABS `.sg`, straight off the text."""
    L = [ln.strip() for ln in open(path) if ln.strip()]
    ff, nlayer = (int(v) for v in L[0].split())
    timo = [int(v) for v in L[1].split()]
    flags = [int(v) for v in L[2].split()]
    assert not (flags[0] or flags[1]), "curve/oblique flags unhandled here"
    nn, ne, nm = (int(v) for v in L[3].split())
    k = 4
    nodes = np.array([[float(x) for x in L[k + i].split()[1:]]
                      for i in range(nn)])
    nid = [int(L[k + i].split()[0]) for i in range(nn)]
    k += nn
    conn = [[int(x) for x in L[k + e].split()[1:]] for e in range(ne)]
    k += ne
    lay = np.array([int(L[k + e].split()[1]) for e in range(ne)])
    t1 = np.array([float(L[k + e].split()[2]) for e in range(ne)])
    k += ne
    layers = [(int(L[k + i].split()[1]), float(L[k + i].split()[2]))
              for i in range(nlayer)]
    k += nlayer
    mats = {}
    for _ in range(nm):
        mid, orth = (int(v) for v in L[k].split())
        eng = []
        for r in range(3):
            eng += [float(v) for v in L[k + 1 + r].split()]
        mats[mid] = {"type": orth, "engineering": eng,
                     "density": float(L[k + 4].split()[0])}
        k += 5
    return dict(ff=ff, nlayer=nlayer, timo=timo, flags=flags, nn=nn, ne=ne,
                nm=nm, nodes=nodes, nid=nid, conn=conn, lay=lay, t1=t1,
                layers=layers, mats=mats, tail=L[k:])


def yaml_order(B):
    """(E, 3, 3) beam-order frames -> (E, 9) in yaml component order, the
    inverse of sg_input.yaml_frames_to_beam's [2, 0, 1] cycle."""
    return np.asarray(B)[:, :, [1, 2, 0]].reshape(-1, 9)


def tube_sg(angle=None, frames=True, density=1600.0):
    """A 4-element stand-in for the square tube: one orthotropic ply, the
    four wall frames, and optionally the ply angle as a block `angle:`.

    `density=None` drops the density, which the `.sc` writer refuses to
    place (its aux-pair order is unvalidated) but the `.sg` writer carries
    -- see test_writer_refuses_the_unvalidated_paths."""
    from opensg_solid.helper.sg_input import vabs_angles_to_frame

    nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0],
                      [0.0, 1.0, 0.0], [0.5, 0.5, 0.0], [2.0, 0.0, 0.0]])
    cells = [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]
    blk = {"type": 1, "engineering": list(ENG)}
    if density is not None:
        blk["density"] = float(density)
    if angle is not None:
        blk["angle"] = float(angle)
    sg = {"dim": 2, "nodes": nodes, "cells": cells,
          "mat_id": np.ones(4, int), "materials": {1: blk},
          "orientation": None, "n_model": None, "refined": None,
          "omega": None, "scale": 1.0}
    if frames:
        B = np.stack([vabs_angles_to_frame(t, 0.0)
                      for t in (0.0, 90.0, 180.0, -90.0)])
        sg["orientation"] = yaml_order(B)
    return sg


# =================================================== the .sc element record
@pytest.mark.parametrize("path,dim", [(RHC, 2), (P1D, 1)])
def test_vendor_2d_sc_pads_to_twenty_slots_not_nine(path, dim):
    """Rule 1, stated against the vendor files themselves: no `.sc` in this
    repo pads a 2-D SG to 9 slots.  RHC_SW_2UC_45.sc is nSG = 2 with 20."""
    f = sc_fields(path)
    assert f["meta"][0] == dim
    if dim == 2:
        assert f["slots"] == [20], (
            "the 2-D vendor deck pads to %r -- if this ever reads [9] the "
            "5/9/20 misreading is back" % f["slots"])


def test_write_sc_uses_twenty_slots_at_every_dimension(tmp_path):
    """What we WRITE: 20 slots for a 1-, 2- and 3-D SG alike, with the
    padding zeros beyond the real nodes."""
    from opensg_solid.helper.sg_input import SC_SLOTS, write_sc

    assert SC_SLOTS == 20, "SC_SLOTS is the record width, not a per-dim map"
    for dim, cells, nodes in [
            (1, [[0, 1], [1, 2]], np.array([[0.], [1.], [2.]])),
            (2, [[0, 1, 2]], np.array([[0., 0.], [1., 0.], [0., 1.]])),
            (3, [[0, 1, 2, 3]], np.array([[0., 0., 0.], [1., 0., 0.],
                                          [0., 1., 0.], [0., 0., 1.]]))]:
        n3 = np.zeros((len(nodes), 3))
        n3[:, :nodes.shape[1]] = nodes
        sg = {"dim": dim, "nodes": n3, "cells": cells,
              "mat_id": np.ones(len(cells), int),
              "materials": {1: {"type": 0, "E": 69.0e9, "nu": 0.3}},
              "orientation": None, "n_model": None, "refined": None,
              "omega": None, "scale": 1.0}
        p = str(tmp_path / ("w%d.sc" % dim))
        write_sc(sg, p, n_model=2 if dim < 3 else 3)
        f = sc_fields(p)
        assert f["slots"] == [20], (dim, f["slots"])
        assert f["nonzero"] == [len(cells[0])], (dim, f["nonzero"])
        assert f["ncoord"] == [dim]


# ====================================================== the .sc model header
def test_submodel_line_is_present_for_a_3d_macro_model(tmp_path):
    """Rule 2.  Both SwiftComp-accepted 3-D decks carry it; so must we, and
    the negative control is that dropping it shifts every later field."""
    from opensg_solid.helper.sg_input import write_sc

    for name in ("Sample_1.sc", "Sample_2.sc"):
        vend = sc_fields_header_only(os.path.join(TPMS, name))
        assert vend == ["0", "0 0 0 0"], (name, vend)

    nodes = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    sg = {"dim": 3, "nodes": nodes, "cells": [[0, 1, 2, 3]],
          "mat_id": np.ones(1, int),
          "materials": {1: {"type": 0, "E": 69.0e9, "nu": 0.3}},
          "orientation": None, "n_model": None, "refined": None,
          "omega": None, "scale": 1.0}
    p = str(tmp_path / "solid.sc")
    r = write_sc(sg, p, n_model=3)
    f = sc_fields(p)
    assert len(f["header"]) == 2, f["header"]
    assert [int(v) for v in f["header"][0].split()] == [0]
    assert [int(v) for v in f["header"][1].split()] == [0, 0, 0, 0]
    assert f["meta"] == [3, 4, 1, 1, 0, 0]
    assert r["n_model"] == 3


def sc_fields_header_only(path):
    """The model-header lines of a `.sc` without parsing the mesh (the TPMS
    decks are 250-660 k lines)."""
    out = []
    with open(path) as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            if len(ln.split()) >= 5:
                return out
            out.append(ln)
    raise AssertionError("no META line in %s" % path)


@pytest.mark.parametrize("n_model,nhdr", [(2, 3), (3, 2)])
def test_sc_header_line_count_by_macro_model(tmp_path, n_model, nhdr):
    """Everything before the META line: the submodel line always, then a
    2-value curvature line for the plate macro model, then the control
    line.  So 2 lines for a solid and 3 for a plate -- the two shapes the
    vendor decks show.  The BEAM header is refused, see
    test_writer_refuses_the_unvalidated_paths."""
    from opensg_solid.helper.sg_input import write_sc

    nodes = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    sg = {"dim": 2, "nodes": nodes, "cells": [[0, 1, 2]],
          "mat_id": np.ones(1, int),
          "materials": {1: {"type": 0, "E": 69.0e9, "nu": 0.3}},
          "orientation": None, "n_model": None, "refined": None,
          "omega": None, "scale": 1.0}
    p = str(tmp_path / "h.sc")
    write_sc(sg, p, n_model=n_model, refined=1 if n_model != 3 else 0)
    hdr = sc_fields(p)["header"]
    assert len(hdr) == nhdr, (n_model, hdr)
    assert int(hdr[0].split()[0]) == (1 if n_model != 3 else 0)
    assert [int(v) for v in hdr[-1].split()] == [0, 0, 0, 0]   # control
    if n_model == 2:
        assert len(hdr[1].split()) == 2          # k12 k21
    # and the shape matches the vendor decks of the same macro model
    vend = sc_fields_header_only(RHC if n_model == 2 else
                                 os.path.join(TPMS, "Sample_1.sc"))
    assert [len(l.split()) for l in hdr] == [len(l.split()) for l in vend]


# ================================================ round trip through read_sc
@pytest.mark.parametrize("path", [RHC, P1D], ids=["RHC_2D", "Plate_1D"])
def test_sc_round_trip_is_lossless_for_mesh_and_materials(tmp_path, path):
    """vendor .sc -> yaml -> our .sc -> read_sc must land on the same mesh
    and the same material constants as reading the vendor file directly."""
    from opensg_solid.helper.sc_to_yaml import convert as sc_to_yaml, read_sc
    from opensg_solid.helper import sg_input

    src = read_sc(path)
    base = str(tmp_path / "rt")
    import shutil
    shutil.copy2(path, base + ".sc")
    sc_to_yaml(base + ".sc", out_base=base)
    ours = base + "_ours.sc"
    sg_input.convert(base + ".yaml", ours, dialect="sc")
    got = read_sc(ours)

    assert got["dim"] == src["dim"]
    assert np.abs(got["nodes"] - src["nodes"]).max() == 0.0
    assert [list(c) for c in got["cells"]] == [list(c) for c in src["cells"]]
    assert np.array_equal(got["mat_id"], src["mat_id"])
    assert sorted(got["materials"]) == sorted(src["materials"])
    for mid, a in src["materials"].items():
        b = got["materials"][mid]
        assert int(a["type"]) == int(b["type"])
        if a["type"] == 2:
            assert np.abs(np.asarray(a["C"]) - np.asarray(b["C"])).max() == 0.0
        elif a["type"] == 1:
            assert np.allclose(a["engineering"], b["engineering"], rtol=0,
                               atol=0)
        else:
            assert (a["E"], a["nu"]) == (b["E"], b["nu"])


def test_omega_header_survives_the_round_trip(tmp_path):
    """A yaml that STATES its SG measure must re-emit that number, not a
    fresh measurement of the mesh -- the 3-D case where the solver consumes
    `omega:`."""
    from opensg_solid.helper.sg_input import read_opensg_yaml, write_sc

    y = tmp_path / "o.yaml"
    y.write_text(
        "n_model: 3\nrefined: 0\nomega: 7.5\n"
        "nodes:\n- [0.0, 0.0, 0.0]\n- [1.0, 0.0, 0.0]\n- [0.0, 1.0, 0.0]\n"
        "- [0.0, 0.0, 1.0]\ncells:\n- [0, 1, 2, 3]\nmat_id: [1]\n"
        "materials:\n  1:\n    type: 0\n    E: 69000.0\n    nu: 0.3\n")
    sg = read_opensg_yaml(str(y))
    assert sg["omega"] == 7.5 and sg["n_model"] == 3
    p = str(tmp_path / "o.sc")
    r = write_sc(sg, p)
    assert r["omega"] == 7.5 and r["omega_source"] == "header"
    assert abs(sc_fields(p)["trailing"] - 7.5) < 1e-12
    # and an explicit argument still wins over the header
    r = write_sc(sg, p, omega=2.25)
    assert r["omega"] == 2.25 and r["omega_source"] == "argument"


def test_omega_is_remeasured_when_the_macro_model_is_overridden(tmp_path):
    """The silent defect this pins: a yaml that STATES `omega:` states it
    for the macro model in its OWN header.  Override n_model and that
    number is the measure of nothing -- it must be re-derived with
    sg_omega for the model actually being written, or a wrong
    normalization goes into a solver input with nothing to flag it.

    Before the fix, writing this yaml at n_model=2 emitted the header's
    7.0 where the plate measure is 6.0.
    """
    from opensg_solid.helper.sg_input import (read_opensg_yaml, write_sc,
                                              sg_omega)

    # a tet whose three coordinate spans differ, so the solid measure
    # (2*3*5 = 30), the plate measure (2*3 = 6) and the header's 7.0 are
    # three different numbers -- an equal-span mesh would hide the bug
    y = tmp_path / "ov.yaml"
    y.write_text(
        "n_model: 3\nrefined: 0\nomega: 7.0\n"
        "nodes:\n- [0.0, 0.0, 0.0]\n- [2.0, 0.0, 0.0]\n- [0.0, 3.0, 0.0]\n"
        "- [0.0, 0.0, 5.0]\ncells:\n- [0, 1, 2, 3]\nmat_id: [1]\n"
        "materials:\n  1:\n    type: 0\n    E: 69000.0\n    nu: 0.3\n")
    sg = read_opensg_yaml(str(y))
    assert (sg["n_model"], sg["omega"], sg["dim"]) == (3, 7.0, 3)

    want = {3: 30.0, 2: 6.0}
    for nm, w in want.items():
        assert abs(sg_omega(sg["nodes"], 3, nm, sg["cells"]) - w) < 1e-12

    # the header's OWN macro model: the stated number is re-emitted
    p3 = str(tmp_path / "m3.sc")
    r3 = write_sc(sg, p3, n_model=3)
    assert r3["omega"] == 7.0 and r3["omega_source"] == "header"
    assert abs(sc_fields(p3)["trailing"] - 7.0) < 1e-12

    # OVERRIDDEN: the trailing value is the PLATE measure, not the header's
    p2 = str(tmp_path / "m2.sc")
    r2 = write_sc(sg, p2, n_model=2)
    assert r2["omega_source"] == "measured", r2
    assert abs(r2["omega"] - 6.0) < 1e-12, r2["omega"]
    assert abs(sc_fields(p2)["trailing"] - 6.0) < 1e-12
    assert abs(sc_fields(p2)["trailing"] - 7.0) > 0.5, (
        "the header's omega is being copied across a macro-model override")

    # an explicit omega= still wins over both
    assert write_sc(sg, p2, n_model=2, omega=1.25)["omega"] == 1.25

    # and a yaml with NO header n_model: the default (plate) is what its
    # `omega:` describes, so a plate write re-emits it and a solid write
    # re-measures
    y2 = tmp_path / "ov2.yaml"
    y2.write_text(y.read_text().replace("n_model: 3\n", ""))
    sg2 = read_opensg_yaml(str(y2))
    assert sg2["n_model"] is None and sg2["omega"] == 7.0
    assert write_sc(sg2, p2)["omega_source"] == "header"
    r = write_sc(sg2, p2, n_model=3)
    assert r["omega_source"] == "measured" and abs(r["omega"] - 30.0) < 1e-12


# ============================================================ the SG measure
def test_sg_omega_mirrors_the_solver_measure():
    """omega must be the number sg_assembly divides by.  The trap: a 2-D SG
    under the SOLID macro model is measured by its INTEGRATED area, not its
    bounding box -- 0.12 vs 1.0609 for the square tube, an 8.8x error."""
    from opensg_solid.helper.sg_input import sg_omega, integrated_measure

    # a 1.03 square annulus of wall 0.03, meshed as two rectangles per wall
    o, i = 0.515, 0.485
    ring = []
    nodes, cells = [], []

    def quad(x0, y0, x1, y1):
        k = len(nodes)
        nodes.extend([[x0, y0, 0.], [x1, y0, 0.], [x1, y1, 0.], [x0, y1, 0.]])
        cells.append([k, k + 1, k + 2, k + 3])

    quad(-o, i, o, o)                     # top
    quad(-o, -o, o, -i)                   # bottom
    quad(-o, -i, -i, i)                   # left
    quad(i, -i, o, i)                     # right
    nodes = np.asarray(nodes)
    area = 2 * (2 * o) * (o - i) + 2 * (2 * i) * (o - i)
    assert abs(area - 0.12) < 1e-12
    assert abs(integrated_measure(nodes, cells, 2) - 0.12) < 1e-12
    assert abs(sg_omega(nodes, 2, 3, cells) - 0.12) < 1e-12
    # the bounding box is the WRONG answer here, and is what you get with
    # no cells to integrate over
    assert abs(sg_omega(nodes, 2, 3) - 1.0609) < 1e-9
    # the other combinations are coordinate spans, as the solver has them
    assert abs(sg_omega(nodes, 2, 2, cells) - 1.03) < 1e-12   # ptp(y1)
    assert sg_omega(nodes, 2, 1, cells) == 1.0
    assert sg_omega(nodes, 1, 2, cells) == 1.0
    n3 = np.c_[nodes[:, :2], np.linspace(0, 2, len(nodes))]
    assert abs(sg_omega(n3, 3, 3) - 1.03 * 1.03 * 2.0) < 1e-9


def test_sg_omega_reproduces_the_vendor_trailing_lines():
    """The two trailing values this repo can check: RHC's 35.248 is the y1
    span of its 2-D plate SG, and the TPMS cell's 1.0 is its bounding-box
    volume."""
    from opensg_solid.helper.sc_to_yaml import read_sc
    from opensg_solid.helper.sg_input import sg_omega

    rhc = read_sc(RHC)
    got = sg_omega(rhc["nodes"], 2, 2, rhc["cells"])
    assert abs(got / 35.248 - 1.0) < 1e-6, got     # vendor rounded to 5 sf
    p = read_sc(P1D)
    assert sg_omega(p["nodes"], 1, 2, p["cells"]) == 1.0 == p["scale"]


# ==================================== the VABS layup: frames <-> (t1, t3)
@pytest.mark.parametrize("t1,t3", [(0.0, 0.0), (90.0, 0.0), (180.0, 0.0),
                                   (-90.0, -45.0), (37.5, 12.25),
                                   (-164.0, 30.0)])
def test_frame_angle_round_trip_is_exact(t1, t3):
    from opensg_solid.helper.sg_input import (frame_to_vabs_angles,
                                              vabs_angles_to_frame)

    R = vabs_angles_to_frame(t1, t3)
    assert np.abs(R @ R.T - np.eye(3)).max() < 1e-14
    assert np.linalg.det(R) > 0
    g1, g3 = frame_to_vabs_angles(R)
    assert abs((g1 - t1 + 180.0) % 360.0 - 180.0) < 1e-9
    assert abs((g3 - t3 + 180.0) % 360.0 - 180.0) < 1e-9


def test_frame_to_vabs_angles_refuses_what_a_sg_cannot_say():
    """A `.sg` can express R1(theta1) R3(theta3) and nothing else, so a
    frame that is not one must raise rather than be projected onto one."""
    from opensg_solid.helper.sg_input import frame_to_vabs_angles

    with pytest.raises(ValueError, match="orthonormal"):
        frame_to_vabs_angles(np.diag([1.0, 1.0, 2.0]))
    with pytest.raises(ValueError, match="left-handed"):
        frame_to_vabs_angles(np.diag([1.0, 1.0, -1.0]))
    # e1 tilted out of the x1-y2 plane: orthonormal, right-handed, not a layup
    R = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    assert np.abs(R @ R.T - np.eye(3)).max() < 1e-14
    assert np.linalg.det(R) > 0
    with pytest.raises(ValueError, match="not a VABS layup"):
        frame_to_vabs_angles(R)


def test_theta3_from_the_frame_and_from_the_block_angle_add():
    """The rule that was inverted: frames + a material `angle:` compose,
    they do not conflict.  Saying -45 either way must give one answer."""
    from opensg_solid.helper.sg_input import element_layups

    a = element_layups(tube_sg(angle=-45.0, frames=True))
    from opensg_solid.helper.sg_input import vabs_angles_to_frame
    sg_b = tube_sg(angle=None, frames=True)
    sg_b["orientation"] = yaml_order(np.stack(
        [vabs_angles_to_frame(t, -45.0) for t in (0.0, 90.0, 180.0, -90.0)]))
    b = element_layups(sg_b)
    assert np.abs(a[0] - b[0]).max() < 1e-9            # theta1
    assert np.array_equal(a[1], b[1])                  # layer of each element
    assert len(a[2]) == len(b[2]) == 1
    assert abs(a[2][0][1] - (-45.0)) < 1e-9
    assert abs(b[2][0][1] - (-45.0)) < 1e-9
    # negative control: with no ply angle at all the layer is at 0, and the
    # two answers must NOT agree
    c = element_layups(tube_sg(angle=None, frames=True))
    assert abs(c[2][0][1]) < 1e-12


# ============================================ what "bake" means, numerically
def test_bake_reproduces_the_solver_assembly_stiffness_exactly(tmp_path):
    """orientation='bake' must fold the block angle AND the element frame
    into each type-2 block, so the .sc read back gives element for element
    the same C_asm the solver would have assembled from the yaml."""
    from opensg_solid.helper.sc_to_yaml import read_sc
    from opensg_solid.helper.sg_input import write_sc
    from opensg_solid.sg_materials import (elem_rotation_from_yaml,
                                           get_heterogeneous_C_matrix,
                                           build_material_C)

    sg = tube_sg(angle=-45.0, frames=True)
    ref, _ = get_heterogeneous_C_matrix(
        {"materials": sg["materials"], "mat_id": sg["mat_id"]},
        elem_rotation=elem_rotation_from_yaml(sg["orientation"]))
    ref = np.asarray(ref, float)

    p = str(tmp_path / "bake.sc")
    r = write_sc(sg, p, n_model=2, orientation="bake", drop_density=True)
    assert r["n_mats"] == 4, "four distinct frames -> four baked materials"
    got_sc = read_sc(p)
    table = np.asarray(build_material_C(got_sc), float)
    got = table[np.asarray(got_sc["mat_id"], int) - 1]
    rel = np.abs(got - ref).max() / np.abs(ref).max()
    assert rel < 1e-12, rel

    # NEGATIVE CONTROL: dropping the frames is a different material law
    q = str(tmp_path / "ign.sc")
    write_sc(sg, q, n_model=2, orientation="ignore", drop_density=True)
    ig_sc = read_sc(q)
    ig = np.asarray(build_material_C(ig_sc), float)[
        np.asarray(ig_sc["mat_id"], int) - 1]
    assert np.abs(ig - ref).max() / np.abs(ref).max() > 0.1

    # NEGATIVE CONTROL: so is dropping the ply angle
    sg0 = tube_sg(angle=None, frames=True)
    s = str(tmp_path / "noang.sc")
    write_sc(sg0, s, n_model=2, orientation="bake", drop_density=True)
    n_sc = read_sc(s)
    na = np.asarray(build_material_C(n_sc), float)[
        np.asarray(n_sc["mat_id"], int) - 1]
    assert np.abs(na - ref).max() / np.abs(ref).max() > 0.1


def test_fold_angles_matches_the_solver_material_table():
    """With no frames a `.sc` still has nowhere to put a ply angle, so it is
    folded into the constants -- and the folded C must be the one the solver
    builds from `angle:`."""
    from opensg_solid.helper.sg_input import fold_angles, write_sc
    from opensg_solid.sg_materials import build_material_C

    blk = {"type": 1, "engineering": list(ENG), "angle": -45.0}
    want = np.asarray(build_material_C({"materials": {1: blk}}), float)[0]
    folded, ids = fold_angles({1: blk})
    assert ids == [1] and folded[1]["type"] == 2
    assert np.abs(folded[1]["C"] - want).max() / np.abs(want).max() < 1e-14
    # a block with no angle is passed through untouched, same object
    plain = {"type": 1, "engineering": list(ENG)}
    out, ids0 = fold_angles({1: plain})
    assert ids0 == [] and out[1] is plain


def test_write_sc_does_not_silently_drop_a_ply_angle(tmp_path):
    """The regression this guards: an `angle:` that never reaches the file.
    The emitted material must differ from the unrotated one."""
    from opensg_solid.helper.sc_to_yaml import read_sc
    from opensg_solid.helper.sg_input import write_sc
    from opensg_solid.sg_materials import build_material_C

    sg = tube_sg(angle=30.0, frames=False)
    p = str(tmp_path / "ang.sc")
    r = write_sc(sg, p, n_model=2, drop_density=True)
    assert r["folded_angles"] == [1]
    got = np.asarray(build_material_C(read_sc(p)), float)[0]
    want = np.asarray(build_material_C(
        {"materials": {1: sg["materials"][1]}}), float)[0]
    assert np.abs(got - want).max() / np.abs(want).max() < 1e-14
    # NEGATIVE CONTROL: it is NOT the unrotated lamina
    bare = np.asarray(build_material_C(
        {"materials": {1: {"type": 1, "engineering": list(ENG)}}}), float)[0]
    assert np.abs(got - bare).max() / np.abs(bare).max() > 0.01


# ================================================ the VABS `.sg`, vs PreVABS
def _sg_from_vendor(v, variant):
    """Rebuild the OpenSG SG that a vendor `.sg` describes.  variant "A"
    puts theta3 in the element frame, "B" in the material `angle:`."""
    from opensg_solid.helper.sg_input import vabs_angles_to_frame

    t3 = np.array([v["layers"][l - 1][1] for l in v["lay"]])
    mat = np.array([v["layers"][l - 1][0] for l in v["lay"]], int)
    if variant == "A":
        B = np.stack([vabs_angles_to_frame(a, b) for a, b in zip(v["t1"], t3)])
        mats = {m: dict(v["mats"][m]) for m in v["mats"]}
    else:
        B = np.stack([vabs_angles_to_frame(a, 0.0) for a in v["t1"]])
        per = {}
        for m, t in zip(mat, t3):
            per.setdefault(int(m), float(t))
        mats = {m: dict(v["mats"][m], angle=per[m]) for m in v["mats"]}
    return {"dim": 2, "nodes": np.c_[v["nodes"], np.zeros(v["nn"])],
            "cells": [[c - 1 for c in row if c != 0] for row in v["conn"]],
            "mat_id": mat, "materials": mats, "orientation": yaml_order(B),
            "n_model": None, "refined": None, "omega": None, "scale": 1.0}


def _assert_sg_matches_vendor(vend_path, tmp_path, node_atol):
    from opensg_solid.helper.sg_input import write_sg

    v = sg_fields(vend_path)
    paths = {}
    for variant in ("A", "B"):
        p = str(tmp_path / ("ours_%s.sg" % variant))
        write_sg(_sg_from_vendor(v, variant), p, timoshenko=v["timo"][0],
                 precision=6)
        paths[variant] = p
    # theta3 said through the frame and through `angle:` are the same file
    assert open(paths["A"]).read() == open(paths["B"]).read()

    o = sg_fields(paths["A"])
    assert [o["ff"], o["nlayer"]] == [v["ff"], v["nlayer"]]
    assert o["timo"] == v["timo"] and o["flags"] == v["flags"]
    assert [o["nn"], o["ne"], o["nm"]] == [v["nn"], v["ne"], v["nm"]]
    assert o["nid"] == v["nid"] == list(range(1, v["nn"] + 1))
    assert np.abs(o["nodes"] - v["nodes"]).max() <= node_atol
    assert o["conn"] == v["conn"]
    assert sorted({len(r) for r in o["conn"]}) == [9]
    assert np.abs((o["t1"] - v["t1"] + 180.0) % 360.0 - 180.0).max() < 1e-9
    assert o["layers"] == v["layers"]
    assert np.array_equal(o["lay"], v["lay"])
    for m in v["mats"]:
        assert o["mats"][m]["type"] == v["mats"][m]["type"]
        assert np.allclose(o["mats"][m]["engineering"],
                           v["mats"][m]["engineering"], rtol=1e-12, atol=0)
        assert o["mats"][m]["density"] == v["mats"][m]["density"]
    assert o["tail"] == v["tail"] == [], "a .sg ends at the last material"
    return v, o


def test_sg_reproduces_the_prevabs_square_tube_field_for_field(tmp_path):
    """PreVABS wrote square_tube.sg; from the same section we must write the
    same file.  Node coordinates are exact here (PreVABS printed 6 decimals
    and so do we at precision=6)."""
    v, o = _assert_sg_matches_vendor(SQ_SG, tmp_path, node_atol=0.0)
    assert v["nn"] == 3228 and v["ne"] == 5384 and v["nlayer"] == 1
    assert v["layers"] == [(1, -45.0)]
    assert sorted(set(np.round(v["t1"], 6))) == [-90.0, 0.0, 90.0, 180.0]


@pytest.mark.slow
def test_sg_reproduces_iea_s10_field_for_field(tmp_path):
    """The larger PreVABS deck: 6 materials, 6 layers, per-element theta1
    over the full range.  PreVABS printed the nodes at 9 decimals and we
    write 6, so the coordinates agree to 5e-7 -- a print width, not a
    convention."""
    v, o = _assert_sg_matches_vendor(IEA_SG, tmp_path, node_atol=5e-7)
    assert v["nn"] == 30271 and v["ne"] == 55434 and v["nlayer"] == 6


@pytest.mark.slow
def test_the_vabs_angle_convention_lands_on_vabs_own_answer(tmp_path):
    """The one cross-code check that is possible without VABS installed.

    The repo ships a vendor VABS INPUT (iea_s10.sg) together with the VABS
    OUTPUT it produced (iea_s10.sg.K).  Turn the file's (theta1, theta3)
    layup back into element frames with vabs_angles_to_frame -- the exact
    inverse of the decomposition write_sg uses -- run OpenSG's own beam
    homogenization, and the Timoshenko 6x6 must land on VABS's.

    If our theta1 meant anything other than what VABS means by it, this
    could not close.  MEASURED worst relative difference 3.7e-11, i.e.
    every digit the .K prints; the gate is 1e-8.  The control below drops
    the frames and moves the law by 54 % in GA3 -- so the closure is a
    statement about the convention, not an insensitive test.

    ~20 s (30 271 nodes, 55 434 tri3), hence `slow`.
    """
    from opensg_solid.helper.sg_input import vabs_angles_to_frame
    from opensg_solid.sg_materials import elem_rotation_from_yaml
    from opensg_solid.sg_homo import plate_homo_2d
    from opensg_solid.helper.k_file import compare_to_K

    kref = IEA_SG + ".K"
    assert os.path.exists(kref)
    v = sg_fields(IEA_SG)
    t3 = np.array([v["layers"][l - 1][1] for l in v["lay"]])
    mat = np.array([v["layers"][l - 1][0] for l in v["lay"]], int)
    B = np.stack([vabs_angles_to_frame(a, b) for a, b in zip(v["t1"], t3)])
    sc = {"dim": 2, "nodes": np.c_[v["nodes"], np.zeros(v["nn"])],
          "cells": [[c - 1 for c in row if c != 0] for row in v["conn"]],
          "mat_id": mat, "materials": v["mats"], "scale": 1.0}

    r = plate_homo_2d(dict(sc), n_model=1, refined=1, plot=False,
                      workdir=str(tmp_path),
                      elem_rotation=elem_rotation_from_yaml(yaml_order(B)))
    K = np.asarray(r["C_eff"], float)
    rep = compare_to_K(K, kref, "vabs_beam", ref_unit="SI", value_unit="SI",
                       symmetrize=True, rtol=1e-8, label="iea_s10 vs VABS")
    assert rep["passed"], "\n" + rep["table"]

    # CONTROL: the theta1 field is load-bearing -- drop it and VABS's own
    # answer is missed by half in the transverse shear
    r0 = plate_homo_2d(dict(sc), n_model=1, refined=1, plot=False,
                       workdir=str(tmp_path))
    bad = compare_to_K(np.asarray(r0["C_eff"], float), kref, "vabs_beam",
                       ref_unit="SI", value_unit="SI", symmetrize=True,
                       rtol=1e-8, label="CONTROL: frames dropped")
    assert not bad["passed"] and bad["worst"]["rel_diff"] > 0.5


def test_the_yaml_frames_of_the_square_tube_are_prevabs_theta1(tmp_path):
    """The decomposition against a real vendor file, on the real mesh:
    square_tube_2Dsolid.yaml and square_tube.sg are the same 5384 elements
    in the same order, so every theta1 must land on PreVABS's."""
    from opensg_solid.helper.sg_input import (read_opensg_yaml,
                                              yaml_frames_to_beam,
                                              frame_to_vabs_angles)

    sg = read_opensg_yaml(SQ_YAML)
    v = sg_fields(SQ_SG)
    assert len(sg["cells"]) == v["ne"] and len(sg["nodes"]) == v["nn"]
    assert [[c + 1 for c in row] for row in sg["cells"]] == \
        [[c for c in row if c != 0] for row in v["conn"]]
    B = yaml_frames_to_beam(sg["orientation"])
    got = np.array([frame_to_vabs_angles(B[e])[0] for e in range(v["ne"])])
    assert np.abs((got - v["t1"] + 180.0) % 360.0 - 180.0).max() == 0.0


# =========================================================== loud refusals
def test_writer_refuses_what_it_cannot_say():
    from opensg_solid.helper import sg_input

    sg = tube_sg(frames=True)
    with pytest.raises(ValueError, match="orientation="):
        sg_input.write_sc(sg, "/dev/null", n_model=2)
    with pytest.raises(ValueError, match="orientation="):
        sg_input.write_sc(sg, "/dev/null", n_model=2, orientation="guess")
    sg3 = dict(tube_sg(frames=False), dim=3)
    with pytest.raises(ValueError, match="BEAM cross-section"):
        sg_input.write_sg(sg3, "/dev/null")
    with pytest.raises(ValueError, match="damping|thermal"):
        sg_input.write_sg(tube_sg(frames=False), "/dev/null", damping=1)
    with pytest.raises(ValueError, match="dialect"):
        sg_input.convert("x.yaml", "y.out", dialect="vabs")


# ============================ the UNVALIDATED paths: refuse, do not guess
def test_writer_refuses_the_unvalidated_paths(tmp_path):
    """Each of these is positionally plausible and supported by NOTHING in
    this tree, so the writer must raise instead of emitting a convention
    into a solver input.  The failure they would cause is silent: the file
    parses and means something else.  See NOT VALIDATED in the sg_input
    module docstring; the evidence for each is re-measured here."""
    from opensg_solid.helper import sg_input

    plain = tube_sg(frames=False, density=None)

    # (a) the BEAM macro-model header.  Evidence: no deck in the tree has
    #     one -- the four third-party decks are 2- or 3-line plate/solid
    #     headers, and the repo's own beam example is fed the BYTE-
    #     IDENTICAL plate-header file.
    import filecmp
    beam_dir = os.path.join(ROOT, "examples", "OpenSG-solid",
                            "7_get_beam_props_from_SG", "RHC_SW_2UC_45.sc")
    assert filecmp.cmp(RHC, beam_dir, shallow=False), (
        "the beam example no longer reuses the plate deck -- re-check "
        "whether a beam header is now evidenced")
    for hdr in (sc_fields_header_only(RHC),
                sc_fields_header_only(P1D),
                sc_fields_header_only(os.path.join(TPMS, "Sample_1.sc")),
                sc_fields_header_only(os.path.join(TPMS, "Sample_2.sc"))):
        assert len(hdr) in (2, 3), hdr          # never the 4-line beam one
    with pytest.raises(NotImplementedError, match="BEAM macro-model"):
        sg_input.write_sc(plain, "/dev/null", n_model=1)

    # (b) orientation="points" -- the trans_flag=1 a/b/c block.  Evidence:
    #     every deck here is trans_flag=0 (the third control-line integer).
    for p in (RHC, P1D, os.path.join(TPMS, "Sample_1.sc"),
              os.path.join(TPMS, "Sample_2.sc")):
        ctrl = [int(v) for v in sc_fields_header_only(p)[-1].split()]
        assert ctrl[2] == 0, (p, ctrl)
    with pytest.raises(NotImplementedError, match="points"):
        sg_input.write_sc(tube_sg(frames=True, density=None), "/dev/null",
                          orientation="points")

    # (c) a nonzero `.sc` density.  Evidence: every third-party deck's aux
    #     line is `0 0`, which cannot tell `T rho` from `rho T`.
    for p in (RHC, P1D):
        L = [ln.strip() for ln in open(p) if ln.strip()]
        aux = [L[i + 1] for i, ln in enumerate(L)
               if len(ln.split()) == 3 and ln.split()[2] == "1"
               and ln.split()[1] in ("0", "1", "2")]
        assert aux and all(set(a.split()) <= {"0", "0.0"} for a in aux), aux
    hot = tube_sg(frames=False)                 # density 1600
    with pytest.raises(NotImplementedError, match="nonzero density"):
        sg_input.write_sc(hot, "/dev/null")
    with pytest.raises(NotImplementedError, match="aux line"):
        sg_input.write_sc(plain, "/dev/null", temperature=300.0)
    # ... and the escape writes the line the vendor decks DO contain
    q = str(tmp_path / "nodens.sc")
    r = sg_input.write_sc(hot, q, drop_density=True)
    assert r["dropped_density"] == [1]
    L = [ln.strip() for ln in open(q) if ln.strip()]
    head = next(i for i, ln in enumerate(L) if ln.split()[-1] == "1"
                and len(ln.split()) == 3)
    assert [float(v) for v in L[head + 1].split()] == [0.0, 0.0]

    # (d) tet10 in a `.sc`.  Evidence: all three 3-D decks are tet4.
    t10 = {"dim": 3, "nodes": np.zeros((10, 3)), "cells": [list(range(10))],
           "mat_id": np.ones(1, int),
           "materials": {1: {"type": 0, "E": 1.0, "nu": 0.3}},
           "orientation": None, "n_model": None, "refined": None,
           "omega": None, "scale": 1.0}
    with pytest.raises(NotImplementedError, match="10-node tetrahedron"):
        sg_input.write_sc(t10, str(tmp_path / "t10.sc"), n_model=3)

    # (e) comments, in EITHER dialect.  Evidence: no deck carries one.
    for p in (RHC, P1D, SQ_SG, IEA_SG):
        ch = "!" if p.endswith(".sg") else "#"
        assert not any(ln.lstrip().startswith(ch) for ln in open(p))
    with pytest.raises(NotImplementedError, match="comments=True"):
        sg_input.write_sc(plain, "/dev/null", comments=True)
    with pytest.raises(NotImplementedError, match="comments=True"):
        sg_input.write_sg(plain, "/dev/null", comments=True)


def test_write_sg_refuses_to_rotate_a_prerotated_C(tmp_path):
    """A type-2 block is already rotated; asking VABS for theta3 on top of
    it would rotate it twice."""
    from opensg_solid.helper.sg_input import write_sg, material_C

    sg = tube_sg(frames=False)
    sg["materials"] = {1: {"type": 2,
                           "C": material_C({"type": 1, "engineering": ENG}),
                           "angle": -45.0}}
    with pytest.raises(ValueError, match="rotate it a second time"):
        write_sg(sg, str(tmp_path / "x.sg"))


def test_shell_dialect_is_refused_with_the_prevabs_route_named(tmp_path):
    from opensg_solid.helper.sg_input import read_opensg_yaml

    y = tmp_path / "shell.yaml"
    y.write_text("nodes:\n- [0.0, 0.0, 0.0]\nelements:\n- [1, 1, 1, 1]\n"
                 "sections: []\nelementOrientations:\n- [1,0,0,0,1,0,0,0,1]\n"
                 "materials: []\n")
    with pytest.raises(ValueError, match="msg-SHELL"):
        read_opensg_yaml(str(y))


# ================================== gate (e): exactly ONE yaml -> SG-input
def test_only_one_yaml_to_sg_input_implementation_survives():
    """The historical module must be a pure re-export: no second writer, and
    the old import path unchanged."""
    from opensg_solid.rm_plate_1D.helper import yaml_to_sc
    from opensg_solid.helper import sg_input

    assert yaml_to_sc.convert is sg_input.convert_yaml_to_sc
    assert yaml_to_sc.write_sc is sg_input.write_sc
    assert yaml_to_sc.write_sg is sg_input.write_sg
    assert yaml_to_sc.read_opensg_yaml is sg_input.read_opensg_yaml
    src = inspect.getsource(yaml_to_sc)
    assert "def " not in src, (
        "yaml_to_sc defines something again -- it must stay a shim")


def test_legacy_convert_signature_still_writes_a_file(tmp_path):
    """The one caller in the tree (tests/08072026_square_shell_mesh/
    four_case/make_inputs.py) calls convert(yaml, out, dim=2)."""
    from opensg_solid.rm_plate_1D.helper.yaml_to_sc import convert

    y = tmp_path / "legacy.yaml"
    y.write_text(
        "nodes:\n- [0.0, 0.0, 0.0]\n- [1.0, 0.0, 0.0]\n- [0.0, 1.0, 0.0]\n"
        "cells:\n- [0, 1, 2]\nmat_id: [1]\n"
        "materials:\n  1:\n    type: 0\n    E: 69000.0\n    nu: 0.3\n")
    out = str(tmp_path / "legacy.sc")
    got = convert(str(y), out, dim=2)
    assert got == out and os.path.exists(out)
    f = sc_fields(out)
    assert f["meta"][:4] == [2, 3, 1, 1] and f["slots"] == [20]
