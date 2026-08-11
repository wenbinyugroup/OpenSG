"""opensg -- the unified OpenSG command.

A DISPATCHER and nothing else: the two engines stay where they are
(opensg_solid, the general 1-D/2-D/3-D SG engine; opensg_shell, the
msg-shell contour/surface engine), each with its own console script, and
this package only decides which of them a yaml belongs to.  It owns no
analysis, no header defaults, no banner and no output.

    from opensg.cli import main        # console script: `opensg <yaml> [H|D]`

Deliberately import-light: nothing here pulls in jax, numpy or either
engine at import time, so the engine's own package __init__ (which sets
the LOAD-BEARING jax x64 flag) is still the first thing that runs once a
file has been resolved.
"""
from .cli import main                      # noqa: F401

__all__ = ["main"]
