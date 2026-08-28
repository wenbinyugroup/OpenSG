"""sg_progress.py -- the solve-only progress bar of a big run.

The bar tracks ONLY the fluctuation solves: that is where the wall
clock of a big run goes.  ONE line, redrawn in place, whatever the
route; write_sc_K's finish() ends it.  Armed when the periodic-reduced
system is >= OPENSG_PROGRESS_DOFS (default 1e6) AND stdout is a
terminal; OPENSG_PROGRESS=1|0 forces it on|off.  Consecutive solves of
one run (the V0 macro columns, then each ladder solve) sweep the SAME
line 0 -> 100.  Zero cost when dark: every hook below is behind
active(), so a dark run runs the historical program byte for byte --
no extra device syncs, no host round-trips, no callbacks.

WHAT DRIVES THE FRACTION, per route -- two kinds, never a fake one:

  RESIDUAL (a genuine convergence fraction) --
  p = log10(res0/res) / log10(1/rtol) on RELATIVE residuals (res0 = 1),
  clamped [0, 1], scaled across the column blocks of one solve.
    amg  (sg_amg.solve_columns)    chunked CG, _PROG_BLOCK iterations
         per host residual read; a single full-maxiter call when dark.
    gamg (sg_gamg.solve_columns)   a PETSc KSP monitor (attached only
         when armed) normalized by the its=0 rnorm the monitor sees --
         the KSP norm type is UNPRECONDITIONED, so that ratio is the
         same relative residual amg reports.
    cg   (sg_assembly.chunked_cg_columns)  chunked exactly as amg, the
         jax.scipy cg carry advanced across 25-iteration blocks.
    stream (sg_stream._pcg)        that loop is already host-resident
         and reads every column residual per iteration -- the bar just
         reads the `worst` it already had.

  COUNT (k of n completed solves) -- a factorization exposes NO
  mid-solve progress, so the direct routes never fake a percentage.
    direct/pardiso/superlu, SHEAR-REFINED plate
         (sg_homo.plate_shear_ladder): the ladder issues a KNOWN
         sequence of solves (V0, V1, [V2], [V1L, V2Lt, V2Lb]), so
         completed-solve count moves the line honestly.
    direct beam KKT (sg_homo._beam_homo_kkt): k of 2 -- the V0 solve
         then the V1s solve of the same augmented matrix.
  Deliberately bar-less: any route whose whole run is ONE monolithic
  factorization -- the classical (unrefined) direct plate/solid, the
  mixed and aperiodic batched solves, and the entire opensg_shell
  engine (its TW cross-sections are ~1e4-dof single pypardiso solves;
  shell_sg3d's drilling saddle point is one _kkt_solve, refined
  in-place).  There is nothing to report between call and return, and
  a percentage invented from a factorization's phases would be a lie.
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
    Out: bool -- the run draws a bar iff True.  EVERY hook is behind
         this: sg_amg/sg_assembly chunk their CG for mid-solve reads,
         sg_gamg attaches its KSP monitor, plate_shear_ladder counts
         its solves -- all only when it is True."""
    return _ON


def solve(p):
    """In:  p float -- completion fraction of the ACTIVE solve
         (residual-driven, or k/n completed solves on a direct ladder)
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
    # the bar and its percent, nothing else -- no label text
    sys.stdout.write("\r [%s] %3d%%"
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
