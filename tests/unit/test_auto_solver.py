"""Unit gate of the --solver auto policy (sg_homo.resolve_auto_solver):
fake dof counts through the resolution function, no mesh, no solve.
The policy is DOF-BANDED ONLY and identical on every machine (GPU
presence changes where the iterative solvers execute, never which one
auto picks): direct below the $OPENSG_DIRECT_WALL (default 1.2e6),
above it amg -> cg -> warned direct as availability/legality allow;
stream is NEVER an auto choice."""
from opensg_solid.sg_homo import resolve_auto_solver


def _pick(dofs, amg, ok, wall=None):
    s, why, warn = resolve_auto_solver(dofs, amg, ok, wall=wall)
    assert s != "stream"           # iter 3 is manual-only, always
    return s, why, warn


def test_below_wall_direct_even_with_pyamg():
    s, why, warn = _pick(289_000, amg=True, ok=True)
    assert s == "direct" and warn is None and "wall" in why


def test_below_wall_direct_tiny():
    s, _, warn = _pick(30_000, amg=True, ok=True)
    assert s == "direct" and warn is None


def test_above_wall_amg():
    s, why, warn = _pick(2_080_000, amg=True, ok=True)
    assert s == "amg" and warn is None and "wall" in why


def test_above_wall_no_pyamg_falls_to_cheb_with_hint():
    s, why, warn = _pick(2_080_000, amg=False, ok=True)
    assert s == "cg" and warn is None and "pyamg" in why


def test_above_wall_iter_illegal_warns_and_attempts_direct():
    s, why, warn = _pick(2_080_000, amg=True, ok=False)
    assert s == "direct" and warn is not None and "wall" in warn


def test_ten_million_band_names_stream_in_reason():
    s, why, warn = _pick(20_000_000, amg=True, ok=True)
    assert s == "amg" and "stream" in why


def test_wall_override_moves_the_band():
    s, _, _ = _pick(1_500_000, amg=True, ok=True, wall=2e6)
    assert s == "direct"
    s, _, _ = _pick(1_500_000, amg=True, ok=True, wall=1e6)
    assert s == "amg"
