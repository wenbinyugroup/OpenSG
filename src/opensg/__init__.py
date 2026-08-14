"""opensg -- the unified OpenSG command.

A DISPATCHER and nothing else: the two engines stay where they are
(opensg_solid, the general 1-D/2-D/3-D SG engine; opensg_shell, the
msg-shell contour/surface engine), each with its own console script, and
this package only decides which of them a yaml belongs to.  It owns no
analysis, no header defaults, no banner and no output.

    from opensg.cli import main        # console script: `opensg <yaml> [H|D]`

plus ONE python convenience, the blade entry point (lazy import):

    import opensg

    blade = opensg.read("IEA-15-240-RWT.yaml")   # windIO v1/v2 or pyNuMAD
    K, M = blade(4)                              # one station's 6x6 pair
    Ks, Ms = blade()                             # full spanwise sweep
    S = blade.section(4); S.scale_thickness(1.3, region="LP_SPAR")

Deliberately import-light: nothing here pulls in jax, numpy or either
engine at import time, so the engine's own package __init__ (which sets
the LOAD-BEARING jax x64 flag) is still the first thing that runs once a
file has been resolved.
"""
from .cli import main                      # noqa: F401


def read(path, **kw):
    """Read a blade file into an editable Blade object.

    In:  path str -- blade yaml (windIO v1/v2 or the pyNuMAD dialect);
         kw -- Blade options (reference="oml"|"center", mesh_size, workdir).
    Out: opensg_shell.Blade -- callable: K, M = blade(st); K/M lists for
         blade(); blade.section(st) for per-station edits."""
    from opensg_shell import Blade
    return Blade(path, **kw)


def __getattr__(name):
    if name in ("Blade", "Section", "opensg_blade"):
        import opensg_shell
        return getattr(opensg_shell, name)
    raise AttributeError("module 'opensg' has no attribute %r" % name)


__all__ = ["main", "read", "Blade", "Section"]
