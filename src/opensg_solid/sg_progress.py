"""sg_progress.py -- the WHOLE-RUN progress bar of a big run.

ONE line, redrawn in place, whatever the route; write_sc_K's finish()
ends it.  Armed when the periodic-reduced system is >=
OPENSG_PROGRESS_DOFS (default 1e6) AND stdout is a terminal;
OPENSG_PROGRESS=1|0 forces it on|off.  Zero cost when dark: every hook
is behind active(), so a dark run runs the historical program byte for
byte -- no extra device syncs, no host round-trips, no callbacks, no
timing calls, NO THREAD, and not one character of output.

THE LINE.  " [####........]  27% assembly" -- the bar, its percent and
ONE short lowercase word (the phase).  The opaque phases add their
elapsed seconds, the solve phase its remaining-time estimate:
" [##########..........]  53% solve ~7m left".  Nothing else is ever
written to it.

PHASE WEIGHTS (the slice of the overall 0 -> 100 each phase owns; a
stage never moves the line backwards and never exceeds 100):

      0.00 - 0.30  assembly   the streamed element assembly
      0.30 - 0.50  setup      the preconditioner setup
      0.50 - 1.00  solve      the fluctuation solves

  and, when the run also runs the RM shear ladder (start(ladder=True)),
  the solve half is subdivided -- V0 first, then the ladder's own
  assembly and its solve sequence:

      0.50 - 0.65  solve      V0 (the macro columns)
      0.65 - 0.75  assembly   the ladder blocks
      0.75 - 1.00  solve      the ladder solves (equal sub-windows)

WHAT DRIVES EACH PHASE -- three kinds, never a fake percentage:

  COUNT (host-side, free) -- a Python loop over slabs or solves already
  knows k of n, so the fraction costs nothing and cannot lie.
    assembly  sg_assembly._streamed_rhs_and_csr (slab k of n) and
         sg_mixed.ladder_blocks (the ladder slab loop).
    direct/pardiso/superlu ladder (sg_homo.plate_shear_ladder): the
         KNOWN solve sequence V0, V1, [V2], [V1L, V2Lt, V2Lb] -- one
         equal sub-window each; a factorization exposes nothing
         mid-solve, so completed-solve count is the honest fraction.
    direct beam KKT (sg_homo._beam_homo_kkt): k of 2.

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

  INDETERMINATE (busy()/idle()) -- an OPAQUE call with no callback and
  no interior counter: pyamg's smoothed_aggregation_solver, PETSc's
  PC GAMG setup (ksp.setUp), the Chebyshev/EBE setup passes, and the
  first slab of an assembly (its jit compile).  Rather than invent a
  percentage, ONE daemon thread redraws the line every 0.5 s with a
  knight-rider cell bouncing inside the bar at the phase's low end,
  plus the elapsed seconds.  It starts only when the bar is armed, is
  a daemon (it can never block exit), writes under the SAME lock as
  solve(), and is stopped by idle(), by the next stage(), by the first
  real fraction, and by finish().

  ETA.  Only where the fraction is residual-driven AND the phase runs
  to 100 % (stage(eta=True) with hi = 1.0, one sub-window): the plain
  amg/gamg/cg/stream V0 solve.  A least-squares fit of the overall
  fraction against wall clock over the last _ETA_FIT host reads gives
  the rate; the remaining time is shown once the run is _ETA_MIN into
  the phase AND two consecutive estimates agree to 30 %, and the
  DISPLAYED value is clamped monotone non-increasing (no
  down-then-up jitter).  Omitted on the ladder (its sub-windows are
  separate solves of different cost -- a fit across them would not be
  an estimate of anything) and on the direct routes (a factorization
  has no rate to observe).

  Deliberately bar-less: any route whose whole run is ONE monolithic
  factorization -- the classical (unrefined) direct plate/solid, the
  mixed and aperiodic batched solves, and the entire opensg_shell
  engine (its TW cross-sections are ~1e4-dof single pypardiso solves;
  shell_sg3d's drilling saddle point is one _kkt_solve, refined
  in-place).  Their assembly still moves the line 0 -> 30.
"""
import os
import sys
import threading
import time

_DOFS = float(os.environ.get("OPENSG_PROGRESS_DOFS", 1e6))
_CELLS = 20            # bar cells, 5% each
_TICK = 0.5            # seconds between knight-rider frames
_ETA_MIN = 0.15        # into the phase before an estimate may show
_ETA_FIT = 5           # host reads in the rate fit
_ETA_TOL = 0.30        # two estimates within 30% -> stable
_ETA_LOW = 5.0         # under this many seconds there is nothing to say
_TH_NAME = "opensg-bar"    # the marker thread, named so a dark run can
#                            be gated on its ABSENCE (threading.enumerate)

# the phase weights of the overall bar (documented above)
W_ASSEMBLY = (0.00, 0.30)
W_SETUP = (0.30, 0.50)
W_SOLVE = (0.50, 1.00)
W_SOLVE_V0 = (0.50, 0.65)      # ... when a shear ladder follows
W_LADDER_ASM = (0.65, 0.75)
W_LADDER = (0.75, 1.00)

_ON = False            # armed for the active run (start())
_LADDER = False        # the run runs the RM shear ladder after V0
_PCT = -1              # last drawn OVERALL percent (the monotone floor)
_LAST = ""             # last drawn line (skip an identical redraw)
_WIDE = 0              # its width (pad the next one, erasing the tail)
_LINE = False          # a bar line is on screen, not yet newline'd

# the active stage: its [lo, hi] slice of the overall bar, its label,
# the number of EQUAL sub-sweeps inside it and how many have completed
_LO, _HI, _LBL, _N, _K = 0.0, 1.0, "", 1, 0
_ETA = False           # this stage's fraction supports a time estimate
_SAMP = []             # (monotonic clock, overall fraction) samples
_SHOW = None           # the DISPLAYED estimate (monotone non-increasing)
_PREV = None           # the previous raw estimate (the stability test)

_LOCK = threading.RLock()      # every writer of the line holds it
_TH = None                     # the knight-rider thread, or None
_EV = None                     # its stop event


def start(dofs, ladder=False):
    """In:  dofs int -- periodic-reduced system size of the run;
         ladder bool -- the run also runs the RM shear ladder (it
         subdivides the solve half of the bar)
    Out: none (arms the bar when the gates pass; draws nothing -- the
         first assembly slab moves the line)."""
    global _ON, _LADDER, _PCT, _LAST, _WIDE, _LINE
    env = os.environ.get("OPENSG_PROGRESS", "")
    _ON = env == "1" or (env != "0" and dofs >= _DOFS
                         and sys.stdout.isatty())
    _LADDER = bool(ladder)
    _PCT, _LAST, _WIDE, _LINE = -1, "", 0, False
    _set_stage(0.0, 1.0, "", 1, False)


def active():
    """In:  --
    Out: bool -- the run draws a bar iff True.  EVERY hook is behind
         this: sg_amg/sg_assembly chunk their CG for mid-solve reads,
         sg_gamg attaches its KSP monitor, the slab loops count their
         slabs, plate_shear_ladder stages its solves -- all only when
         it is True."""
    return _ON


def solve_window():
    """In:  --
    Out: (lo, hi) -- the V0 solve's slice of the bar: the whole solve
         half, or its first part when a shear ladder follows."""
    return W_SOLVE_V0 if _LADDER else W_SOLVE


def stage(win, label, n=1, eta=False):
    """Open a phase: the slice of the overall bar the fractions handed
    to solve() from here on are mapped into.  Stops a running marker
    (the phase it belonged to has ended).

    In:  win (lo, hi) fractions of the overall bar (a W_* weight);
         label str ONE short lowercase word; n int equal sub-sweeps
         inside the window (each solve(1.0) closes one); eta bool --
         the fractions are rate-observable (residual-driven), so a
         remaining-time estimate may be shown
    Out: none (draws nothing)."""
    if not _ON:
        return
    _stop_marker()
    _set_stage(float(win[0]), float(win[1]), label, max(1, int(n)),
               bool(eta))


def solve(p):
    """In:  p float -- completion fraction of the ACTIVE sub-sweep
         (residual-driven, or k/n of a counted loop)
    Out: none (maps p into the stage's window and redraws the one line
         iff it changes; never backwards, never past 100)."""
    global _K
    if not _ON:
        return
    if _TH is not None:
        _stop_marker()          # a real fraction supersedes the marker
    q = min(1.0, max(0.0, float(p)))
    # the stage NEVER spills past its window: a route that closes its
    # last sub-sweep and then reports 1.0 again (solve_columns does)
    # saturates at hi instead of walking into the next phase
    f = _LO + (_HI - _LO) * min(1.0, (_K + q) / _N)
    _draw(f, _estimate(f) if _ETA else "")
    if q >= 1.0:
        _K = min(_K + 1, _N)


def busy():
    """Start the INDETERMINATE marker of an opaque phase: one daemon
    thread redrawing the line every _TICK s with a knight-rider cell at
    the phase's low end plus its elapsed seconds.  Never starts when
    the bar is dark or when a marker already runs.

    In:  -- (the phase is the open stage)
    Out: none."""
    global _TH, _EV
    if not _ON or _TH is not None:
        return
    _EV = threading.Event()
    _TH = threading.Thread(target=_spin,
                           args=(_EV, _LO, time.monotonic()),
                           name=_TH_NAME, daemon=True)
    _TH.start()


def idle():
    """In:  --
    Out: none (stops and joins the knight-rider thread; a no-op when
         none runs, so it is safe in a finally)."""
    _stop_marker()


def finish():
    """In:  --
    Out: none (stops any marker, CLOSES a drawn line at 100 % -- it is
         called where the run's outputs are written, so the run really
         is done, whatever the last phase reported -- ends it with its
         newline and disarms)."""
    global _ON, _LINE
    _stop_marker()
    if _LINE:
        _set_stage(1.0, 1.0, _LBL, 1, False)
        _draw(1.0)
        sys.stdout.write("\n")
        sys.stdout.flush()
    _ON, _LINE = False, False


# ------------------------------------------------------------- internals
def _set_stage(lo, hi, label, n, eta):
    """In:  the stage fields  Out: none (resets the stage and its ETA
         history; the overall monotone floor _PCT is NOT reset)."""
    global _LO, _HI, _LBL, _N, _K, _ETA, _SHOW, _PREV
    _LO, _HI, _LBL, _N, _K, _ETA = lo, hi, label, n, 0, eta
    _SHOW, _PREV = None, None
    del _SAMP[:]


def _draw(f, tail="", mark=None):
    """The ONE writer of the line, under _LOCK (solve() and the marker
    thread share it).

    In:  f float overall fraction; tail str after the label ('' none);
         mark int | None -- a knight-rider cell index (a marker frame:
         it may redraw the same percent)
    Out: none."""
    global _PCT, _LAST, _WIDE, _LINE
    with _LOCK:
        pct = max(_PCT, int(100 * min(1.0, max(0.0, f))))
        k = _CELLS * pct // 100
        cells = ["#"] * k + ["."] * (_CELLS - k)
        if mark is not None and k < _CELLS:
            cells[min(_CELLS - 1, max(k, int(mark)))] = "#"
        s = " [%s] %3d%%" % ("".join(cells), pct)
        if _LBL:
            s += " " + _LBL
        if tail:
            s += " " + tail
        if s == _LAST:
            return
        sys.stdout.write("\r" + s + " " * max(0, _WIDE - len(s)))
        sys.stdout.flush()
        _PCT, _LAST, _WIDE, _LINE = pct, s, len(s), True


def _spin(ev, lo, t0):
    """The knight-rider body: bounce a cell inside the empty part of
    the bar every _TICK s until stopped.  Daemon; it writes ONLY
    through _draw (so it never interleaves with solve()).

    In:  ev threading.Event stop flag; lo float the phase's low end;
         t0 float its start (wall clock)
    Out: none."""
    k = _CELLS * int(100 * lo) // 100
    span = max(1, _CELLS - k)
    j, d = 0, 1
    while not ev.wait(_TICK):
        if ev.is_set():
            break
        _draw(lo, _clock(time.monotonic() - t0), mark=k + j)
        if span > 1:
            j += d
            if j >= span - 1 or j <= 0:
                d = -d
        if ev.is_set():
            break


def _stop_marker():
    """In:  --  Out: none (signals and joins the marker thread; leaves
         the line as drawn)."""
    global _TH, _EV
    th, ev = _TH, _EV
    if th is None:
        return
    _TH, _EV = None, None
    ev.set()
    th.join(timeout=4 * _TICK)      # never blocks a run: it is a daemon


def _clock(s):
    """In:  s float seconds  Out: str compact -- '43s', '7m', '1h12m'."""
    s = int(max(0.0, s))
    if s < 90:
        return "%ds" % s
    if s < 3600:
        return "%dm" % ((s + 30) // 60)
    return "%dh%02dm" % (s // 3600, (s % 3600) // 60)


def _estimate(f):
    """The solve phase's remaining-time text (module docstring, ETA):
    a least-squares rate from the last _ETA_FIT (t, fraction) host
    reads, shown only past _ETA_MIN of the phase, only once two
    consecutive estimates agree to _ETA_TOL, and clamped monotone
    non-increasing.

    In:  f float overall fraction just reached
    Out: str '~7m left' or '' (nothing to show yet)."""
    global _SHOW, _PREV
    _SAMP.append((time.monotonic(), f))
    if len(_SAMP) > _ETA_FIT:
        del _SAMP[0]
    if _HI < 1.0 or _N != 1 or f - _LO < _ETA_MIN * (_HI - _LO):
        return ""
    n = len(_SAMP)
    if n < 3:
        return ""
    mt = sum(s[0] for s in _SAMP) / n
    mf = sum(s[1] for s in _SAMP) / n
    num = sum((s[0] - mt) * (s[1] - mf) for s in _SAMP)
    den = sum((s[0] - mt) ** 2 for s in _SAMP)
    if den > 0.0 and num > 0.0:
        est = (1.0 - f) * den / num
        if _PREV is not None and abs(est - _PREV) <= _ETA_TOL * max(
                _PREV, 1e-12):
            _SHOW = est if _SHOW is None else min(_SHOW, est)
        _PREV = est
    if _SHOW is None or _SHOW < _ETA_LOW:
        return ""              # seconds away: the bar itself says it
    return "~%s left" % _clock(_SHOW)
