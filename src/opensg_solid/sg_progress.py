"""sg_progress.py -- the ONE five-chunk completion bar of a big run.

Chunks: init / assembly / solve / ladder / done, 20% each; the line
redraws ONLY when a chunk completes, so the whole feature costs five
string writes per run.  Armed when the periodic-reduced system is
>= OPENSG_PROGRESS_DOFS (default 1e6) AND stdout is a terminal;
OPENSG_PROGRESS=1|0 forces it on|off.  Skipped chunks (a classical run
has no ladder) simply jump the bar forward -- it never moves backwards
and never redraws twice for one chunk.
"""
import os
import sys

_STAGES = ("init", "assembly", "solve", "ladder", "done")
_DOFS = float(os.environ.get("OPENSG_PROGRESS_DOFS", 1e6))
_BAR = None            # furthest completed chunk of the active run


def start(dofs):
    """In:  dofs int -- periodic-reduced system size of the run
    Out: none (arms the bar when the gates pass; draws the 20% init)."""
    global _BAR
    env = os.environ.get("OPENSG_PROGRESS", "")
    on = env == "1" or (env != "0" and dofs >= _DOFS
                        and sys.stdout.isatty())
    _BAR = 0 if on else None
    if on:
        tick("init")


def tick(stage):
    """In:  stage str, one of _STAGES
    Out: none (redraws iff this chunk advances the bar)."""
    global _BAR
    if _BAR is None:
        return
    k = _STAGES.index(stage) + 1
    if k <= _BAR:
        return
    _BAR = k
    n = len(_STAGES)
    fill = "#" * (4 * k) + "." * (4 * (n - k))
    sys.stdout.write("\r [%s] %3d%% %-8s%s"
                     % (fill, 20 * k, stage, "\n" if k == n else ""))
    sys.stdout.flush()
    if k == n:
        _BAR = None
