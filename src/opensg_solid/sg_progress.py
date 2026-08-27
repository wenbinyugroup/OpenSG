"""sg_progress.py -- the solve-only progress bar of a big run.

The bar tracks ONLY the iterative fluctuation solves (sg_amg
solve_columns): that is where the wall clock of a big run goes, and
convergence gives a GENUINE completion fraction
p = log10(res0/res_worst) / log10(1/rtol), res_worst the worst
per-column RELATIVE residual across the RHS columns (so res0 = 1),
clamped to [0, 1].  sg_amg chunks its CG into 25-iteration device
blocks and reports res_worst once per chunk -- the line redraws only
when the percent advances, so the feature costs a few string writes
per solve and ZERO extra device syncs when dark.  Armed when the
periodic-reduced system is >= OPENSG_PROGRESS_DOFS (default 1e6) AND
stdout is a terminal; OPENSG_PROGRESS=1|0 forces it on|off.
Consecutive solves of one run (the V0 macro columns, then each ladder
solve) redraw the SAME line 0 -> 100; write_sc_K's finish() ends the
line.  Bar-less by design: the cheb-CG route (--solver cg -- its
vmapped device loop exposes no cheap mid-solve residual), the direct
factorizations, and gamg (PETSc owns that loop; a KSP monitor callback
could feed solve() -- follow-up).
"""
import os
import sys

_DOFS = float(os.environ.get("OPENSG_PROGRESS_DOFS", 1e6))
_CELLS = 20            # bar cells, 5% each
_ON = False            # armed for the active run (start())
_PCT = -1              # last drawn percent of the active solve line
_LINE = False          # a bar line is on screen, not yet newline'd


def start(dofs):
    """In:  dofs int -- periodic-reduced system size of the run
    Out: none (arms the bar when the gates pass; draws nothing -- the
         first chunked residual moves the line)."""
    global _ON, _PCT, _LINE
    env = os.environ.get("OPENSG_PROGRESS", "")
    _ON = env == "1" or (env != "0" and dofs >= _DOFS
                         and sys.stdout.isatty())
    _PCT, _LINE = -1, False


def active():
    """In:  --
    Out: bool -- sg_amg chunks its CG for mid-solve reads iff True."""
    return _ON


def solve(p):
    """In:  p float -- convergence fraction of the ACTIVE solve
    Out: none (redraws the solve line iff the clamped percent
         advances; 100% re-arms 0 for the next solve of the run,
         leaving the line for finish())."""
    global _PCT, _LINE
    if not _ON:
        return
    pct = int(100 * min(1.0, max(0.0, p)))
    if pct <= _PCT:
        return
    k = _CELLS * pct // 100
    sys.stdout.write("\r [%s] %3d%% solve"
                     % ("#" * k + "." * (_CELLS - k), pct))
    sys.stdout.flush()
    _PCT = -1 if pct == 100 else pct
    _LINE = True


def finish():
    """In:  --
    Out: none (ends a drawn solve line with its newline and disarms;
         called once, where the run's outputs are written)."""
    global _ON, _LINE
    if _LINE:
        sys.stdout.write("\n")
        sys.stdout.flush()
    _ON, _LINE = False, False
