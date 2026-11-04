"""The msg-shell ring D .vtk carries all 15 fields of every recovery station.

`opensg <ring yaml> D` writes <base>_dehom.txt and <base>_dehom.vtk.  The
.vtk holds one quad per (element, depth band).  Each quad carries the values
of its own station as CELL_DATA, in this order:

    S11 S22 S33 S23 S13 S12      stresses (the older arrays, unchanged)
    jflag jdist                  the junction arrays (not with `junction: off`)
    E11 E22 E33 2E23 2E13 2E12   strains, engineering shears (the .txt names)
    U1 U2 U3                     displacement (the .txt's u1 u2 u3)
    VECTORS U                    the same displacement as one vector

Checked on OpenSG's shell example, the IEA-22 station of
examples/OpenSG_shell/2_get_beam_dehom_from_shell_cross_section at --center:
with its .ff (junction: flag, the default), and with an epsilon_bar header
under `junction: exclude` and `junction: off`.  The gates:
  * the array names, their kinds and their order in the file;
  * every array equals its .txt column row by row, to the .txt's digits
    (the rows `junction: exclude` sets to NaN in the .txt are skipped: the
    .vtk keeps the recovered values there);
  * VECTORS U equals U1 U2 U3 exactly, and every value is finite;
  * meshio, when it is installed, reads the same arrays.

Run:  pytest tests/unit/test_vtk_fields_shell.py -q   (env opensg_2_0)
"""
import os
import shutil
import struct

import numpy as np
import pytest

assert struct.calcsize("P") == 8

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
EX2 = os.path.join(ROOT, "examples", "OpenSG_shell",
                   "2_get_beam_dehom_from_shell_cross_section")
STEM = "iea_s10_shell"

STRESS = ("S11", "S22", "S33", "S23", "S13", "S12")
STRAIN = ("E11", "E22", "E33", "2E23", "2E13", "2E12")
DISP = ("U1", "U2", "U3")
# .txt columns: elem zeta y2 y3 z | 6 stresses | 6 strains | u1 u2 u3 | jflag jdist
COL = dict(zip(STRESS + STRAIN + DISP, range(5, 20)))
EPS = "[1.0e-4, 2.0e-5, -3.0e-5, 4.0e-4, -2.0e-4, 1.0e-4]"

# case: (header lines added on top of the yaml copy, copy the example's .ff)
CASES = {
    "ff_flag": ([], True),
    "eps_exclude": (["junction: exclude", "epsilon_bar: " + EPS], False),
    "eps_off": (["junction: off", "epsilon_bar: " + EPS], False),
}


def read_vtk(path):
    """CELL_DATA of a legacy ASCII .vtk: {name: array}, [(kind, name, type)]
    in file order, and the cell count."""
    with open(path) as f:
        lines = f.read().split("\n")
    arrays, order, nc, i = {}, [], None, 0
    while i < len(lines):
        t = lines[i].split()
        if t and t[0] == "CELL_DATA":
            nc = int(t[1])
        elif t and t[0] == "SCALARS":
            assert lines[i + 1].strip() == "LOOKUP_TABLE default"
            arrays[t[1]] = np.array([float(v) for v in lines[i + 2:i + 2 + nc]])
            order.append(("SCALARS", t[1], t[2]))
            i += 2 + nc
            continue
        elif t and t[0] == "VECTORS":
            arrays[t[1]] = np.array([[float(v) for v in ln.split()]
                                     for ln in lines[i + 1:i + 1 + nc]])
            order.append(("VECTORS", t[1], t[2]))
            i += 1 + nc
            continue
        i += 1
    return arrays, order, nc


@pytest.fixture(scope="module", params=list(CASES))
def run(request, tmp_path_factory):
    """One `opensg <yaml> D --center` run of the IEA-22 station per case."""
    from opensg.cli import main
    extra, with_ff = CASES[request.param]
    d = tmp_path_factory.mktemp(request.param)
    y = str(d / (STEM + ".yaml"))
    with open(os.path.join(EX2, STEM + ".yaml")) as f:
        txt = f.read()
    with open(y, "w") as f:
        f.write("".join(ln + "\n" for ln in extra) + txt)
    if with_ff:
        shutil.copy(os.path.join(EX2, STEM + ".ff"), str(d / (STEM + ".ff")))
    rc = main([y, "D", "--center"])
    assert rc in (0, None), rc
    base = y[:-5] + "_dehom"
    arrays, order, nc = read_vtk(base + ".vtk")
    tab = np.loadtxt(base + ".txt")
    return dict(case=request.param, base=base, arrays=arrays, order=order,
                nc=nc, tab=tab)


def test_array_names_kinds_and_order(run):
    """The older arrays first, unchanged in name and kind; then the strains,
    the displacement and VECTORS U."""
    junction = [] if run["case"] == "eps_off" else [("SCALARS", "jflag", "int"),
                                                    ("SCALARS", "jdist", "float")]
    want = ([("SCALARS", n, "float") for n in STRESS] + junction
            + [("SCALARS", n, "float") for n in STRAIN + DISP]
            + [("VECTORS", "U", "float")])
    assert run["order"] == want
    assert run["nc"] == run["tab"].shape[0] > 0           # one cell per .txt row
    assert run["nc"] % 9 == 0                             # 9 stations per element


def test_arrays_equal_the_txt_columns(run):
    """Every field array is the .txt column of the same row (the .txt prints
    7 digits, the .vtk 9); NaN rows of `junction: exclude` are skipped."""
    A, tab = run["arrays"], run["tab"]
    nan_rows = 0
    for name, c in COL.items():
        t, v = tab[:, c], A[name]
        ok = np.isfinite(t)
        nan_rows = max(nan_rows, int((~ok).sum()))
        np.testing.assert_allclose(v[ok], t[ok], rtol=1e-6, atol=0, err_msg=name)
    if run["case"] == "eps_off":
        assert tab.shape[1] == 20 and "jflag" not in A
        return
    np.testing.assert_array_equal(A["jflag"], tab[:, 20])
    fin = np.isfinite(tab[:, 21])
    np.testing.assert_allclose(A["jdist"][fin], tab[fin, 21], rtol=1e-6, atol=0)
    assert np.all(A["jdist"][~fin] == -1.0)
    dup = int((A["jflag"] == 3).sum())
    if run["case"] == "eps_exclude":
        assert dup > 0 and nan_rows == dup                # the exclude path ran
    else:
        assert nan_rows == 0


def test_vector_U_and_finite_values(run):
    """VECTORS U is (U1, U2, U3) digit for digit, and nothing is NaN or inf."""
    A = run["arrays"]
    np.testing.assert_array_equal(A["U"], np.column_stack([A[n] for n in DISP]))
    for name, v in A.items():
        assert np.all(np.isfinite(v)), name


def test_meshio_reads_the_same_arrays(run):
    """A standard reader sees the same 15 fields, U and the junction arrays
    (meshio reads the `float` arrays as float32)."""
    meshio = pytest.importorskip("meshio")
    m = meshio.read(run["base"] + ".vtk")
    cd = {k: np.asarray(v[0]) for k, v in m.cell_data.items()}
    assert set(cd) == set(run["arrays"])
    assert sum(len(c.data) for c in m.cells) == run["nc"]
    for name, v in run["arrays"].items():
        np.testing.assert_allclose(cd[name].reshape(v.shape), v, rtol=1e-6,
                                   atol=0, err_msg=name)
