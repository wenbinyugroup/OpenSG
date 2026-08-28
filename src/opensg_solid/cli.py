"""opensg_solid command line -- the file is the whole problem:

    opensg_solid <sg.yaml>        homogenization (the default)
    opensg_solid <sg.yaml> D      dehomogenization: homogenize, then recover
    opensg_solid <sg.yaml> --mesh  the run above, AND force a redraw of
                                  <base>_mesh.png (a flag, not an
                                  analysis: combines with H | D)
    opensg_solid <sg.yaml> --solver F [N]  the linear solver of the
                                  fluctuation solves: family direct |
                                  iter (+ backend number), or a name
                                  (pardiso superlu cg amg stream gamg).
                                  Bare --solver prints the menu; no
                                  flag = auto by dofs.  A flag,
                                  combines with H | D

No flags, no codes: everything else lives in the yaml header (the
leading scalar keys above the mesh blocks), and every key has a default
-- a headerless mesh runs as a classical plate homogenization.

    msg: solid          # which ENGINE owns this file: `solid` = this one
                        #   (the general 1-D/2-D/3-D SG engine), `shell` =
                        #   opensg_shell.  Omit it and the mesh dialect
                        #   decides (nodes/cells/mat_id = solid), so older
                        #   files keep working; the unified `opensg`
                        #   command dispatches on exactly this.
    n_model: 2          # 1 = beam, 2 = plate, 3 = 3-D solid
    refined: 0          # 0 = classical (plate ABD 6x6 / beam EB 4x4)
                        # 1 = shear-refined (plate ABDG 8x8 / beam
                        #     Timoshenko 6x6); ignored for solid
    aperiodic: 1        # OPTIONAL, and only for an SG that is genuinely
                        #   not periodic: zero fluctuation on every
                        #   bounding-box-face node.  Periodic is the
                        #   default at every SG dimension -- omit the
                        #   key entirely for a periodic SG.
    omega: 1.0          # OPTIONAL user SG measure (3-D solid only).
                        #   The measure is normally taken from the mesh:
                        #   a 3-D SG is a periodic unit cell, so it is the
                        #   node BOUNDING-BOX volume, not the summed
                        #   element (material) volume.  Set this key only
                        #   when the equivalent continuum occupies
                        #   something else; it wins over the measurement.
    analysis: H         # the same H | D switch, when you would rather
                        #   carry it in the file than on the command
                        #   line (the argument wins)
    epsilon_bar: [...]  # (6,) macro state for D --
                        #   [e11 e22 2e12 k11 k22 2k12] (plate/solid) or
                        #   [ext sh2 sh3 twist bend2 bend3] (beam)

D drives the recovery from `epsilon_bar`.  A 1-D plate SG instead
recovers a whole FIELD from a `.ff` station table found next to the
yaml (or named by `ff:` in the header) -- see the 1-D dehom example.

D output frame: --material (the DEFAULT) rotates every element's
stress/strain into its own PLY axes (theta = +angle about the thickness
axis, the unified VABS-sign Abaqus equivalence `3, angle` -- what an
Abaqus odb shows UNTRANSFORMED when the section carries
*Orientation); --global keeps
the SG axes, the frame the recovery computes in and the one every .SM
written before this flag existed carries.  The .SM/.EM header names the
frame; .U is never rotated.  Block-angle yamls only -- per-element
frames (elementOrientations) refuse --material.

H writes the timed <base>.out (SwiftComp .K layout) + <base>_mesh.png
(redrawn whenever the PNG is older than the yaml, so a re-converted SG
never keeps the previous picture);
D additionally writes <base>_dehom.txt/.vtk/.SM/.EM/.U.
"""
import glob
import os
import sys

# XLA's C++ warnings (rematerialization etc.) are compiler internals the
# run cannot act on; must be set before jax first loads
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

BANNER = """
 ============================================================================
 OpenSG -- a multiscale structural analysis tool based on the Mechanics of
 Structure Genome (MSG), developed by the Multiscale Structural Mechanics
 Group led by Prof. Wenbin Yu at Purdue University
 ============================================================================
"""

SOLVER_MENU = """\
 --solver <family> [n] | <name>     no flag = auto by dofs
                                    (wall: OPENSG_DIRECT_WALL, 1.2e6)

   auto       direct below the wall, amg above -- same on every machine
   direct 1   pardiso  MKL, multithreaded (direct default)
   direct 2   superlu  scipy fallback -- robust, slow
   iter 1     cg       matrix-free EBE Chebyshev CG
   iter 2     amg      AMG-preconditioned CG; runs on the GPU when
                       jax sees one (needs pyamg)
   iter 3     stream   host-streamed EBE CG for >10M dofs  [WIP]
   iter 4     gamg     PETSc GAMG-preconditioned CG via jetsci
                       (needs petsc4py; explicit-only, auto never
                       picks it)

 names: --solver pardiso | superlu | cg | amg | stream | gamg
 every option returns the same law digits
"""


def main(argv=None):
    """Run the analysis the yaml (and an optional H|D argument) request.

    In:  argv (list[str] | None) -- [yaml_path, "H"|"D" (optional)];
         None reads sys.argv
    Out: int exit code (0 ok, 2 usage)."""
    print(BANNER)
    argv = sys.argv[1:] if argv is None else list(argv)
    # --mesh is a FLAG, not the analysis: strip it, keep the rest
    want_mesh = any(str(a).strip().lower() in ("--mesh", "-m", "mesh")
                    for a in argv)
    argv = [a for a in argv
            if str(a).strip().lower() not in ("--mesh", "-m", "mesh")]
    # --solver F [N] picks the linear-solver FAMILY of the fluctuation
    # solves (direct = assembled sparse factorization, CPU; iter =
    # matrix-free preconditioned CG, device-resident -- the GPU route)
    # and optionally the backend NUMBER inside it (`--solver direct 2`,
    # `--solver direct2`).  Bare `--solver` / `--solver help` prints the
    # menu.  --gpu / --solver cg = iter 1; pardiso / superlu = direct 1/2.
    solver_req, _kept, _i = None, [], 0
    while _i < len(argv):
        al = str(argv[_i]).strip().lower()
        if al in ("--solver_help", "--solver-help"):
            solver_req = "help"            # no value token to consume
        elif al.startswith("--solver"):
            if "=" in al:
                v = al.split("=", 1)[1]
            else:
                _i += 1
                v = (str(argv[_i]).strip().lower()
                     if _i < len(argv) else "")
            if (v in ("direct", "iter") and _i + 1 < len(argv)
                    and str(argv[_i + 1]).strip().isdigit()):
                _i += 1
                v += str(argv[_i]).strip()
            solver_req = v
        elif al in ("--gpu", "gpu"):
            solver_req = "iter1"
        else:
            _kept.append(argv[_i])
        _i += 1
    argv = _kept
    solver, force_superlu = None, False
    if solver_req is not None:
        _alias = {"direct": "direct1", "pardiso": "direct1",
                  "superlu": "direct2", "iter": "iter1", "cg": "iter1",
                  "cheb": "iter1", "amg": "iter2", "mumps": "direct3",
                  "cudss": "direct4", "stream": "iter3",
                  "gamg": "iter4"}
        s = _alias.get(solver_req, solver_req)
        if s in ("", "help", "list", "?"):
            print(SOLVER_MENU)
            return 2
        if s == "direct1":
            solver = "direct"
        elif s == "direct2":
            solver, force_superlu = "direct", True
        elif s == "iter1":
            solver = "cg"
        elif s == "iter2":
            solver = "amg"
        elif s == "iter3":
            solver = "stream"
        elif s == "iter4":
            solver = "gamg"
            # PETSc allocates its Mat/hierarchy on the SAME card, and
            # XLA's default pool grabs 75% of it up front (60 GB of an
            # A100-80), so the COO upload dies with cudaErrorMemory-
            # Allocation.  Let jax grow on demand instead -- read at
            # backend init, which is still ahead of us here, and inert
            # on a CPU-only machine.
            os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE",
                                  "false")
        elif s in ("direct3", "direct4"):
            raise SystemExit("--solver %s is on the menu but not wired"
                             " yet -- available today: direct 1"
                             " (pardiso, the default), direct 2"
                             " (superlu), iter 1 (cheb), iter 2 (amg),"
                             " iter 3 (stream), iter 4 (gamg)\n\n%s"
                             % (solver_req, SOLVER_MENU))
        else:
            print(SOLVER_MENU)
            raise SystemExit("unknown --solver %r" % solver_req)
    # D output frame: --material (ply axes, the DEFAULT) | --global (SG
    # axes, the frame every pre-flag .SM was written in).  A flag, not an
    # analysis -- strip it the same way; the last one given wins.
    frame = "material"
    for a in argv:
        if str(a).strip().lower() in ("--material", "--global"):
            frame = str(a).strip().lower().lstrip("-")
    argv = [a for a in argv
            if str(a).strip().lower() not in ("--material", "--global")]
    if not 1 <= len(argv) <= 2 or argv[0] in ("-h", "--help"):
        print(__doc__)
        print(SOLVER_MENU)
        return 2
    path = argv[0]
    if not os.path.exists(path):
        alt = os.path.splitext(path)[0] + ".yaml"
        raise SystemExit("no such file: %s%s" % (
            path, "" if not os.path.exists(alt) else
            "\ndid you mean %s ?" % alt))

    # an unfilled msh_to_yaml template: name the missing `materials:` fields
    # instead of failing deep inside the material parse (cheap line scan,
    # never a full parse)
    from .io.msh_to_yaml import check_filled
    _todo = check_filled(path)
    if _todo:
        raise SystemExit(_todo)

    from .sg_mesh import read_yaml_header, resolve_msg, sg_dim, node_span_dim
    from .sg_homo import plate_homo_2d

    hdr = read_yaml_header(path)
    analysis = str(argv[1] if len(argv) == 2
                   else hdr.get("analysis", "H")).strip().upper()
    if analysis not in ("H", "D"):
        raise SystemExit("the analysis argument must be H (homogenization)"
                         " or D (dehomogenization), got %r" % argv[1])
    if want_mesh:
        # --mesh does not replace the analysis, it GUARANTEES the figure:
        # drop the old PNG so the run's own plot step redraws it
        _png = os.path.splitext(path)[0] + "_mesh.png"
        if os.path.exists(_png):
            os.remove(_png)
    import time as _time
    _t0 = _time.perf_counter()
    _MDL = {1: "beam", 2: "plate", 3: "3-D solid"}
    _ap = os.path.abspath(path).replace("\\", "/").split("/")
    print(" input     : %s" % (".../" + "/".join(_ap[-2:])
                               if len(_ap) > 2 else "/".join(_ap)))
    # engine, SG space and element type on ONE line; the mesh line below
    # then carries the dofs alone
    _eng, _dim, _otag = resolve_msg(path), node_span_dim(path), ""
    try:
        # mesh size up front (parse is npz-sidecar cached, so this costs
        # nothing the run would not pay later); guarded -- a print must
        # never kill an analysis on an exotic dialect
        from .sg_mesh import load_sg_input as _lsi
        _d = _lsi(path)
        _nc = _d.get("cells")
        # a list may be per-element ROWS (one connectivity list each, the
        # msh/sets dialect) or ragged BLOCKS of (Ei, npe_i) arrays (mixed
        # shell); rows count as themselves, blocks sum their cells
        if isinstance(_nc, (list, tuple)) and len(_nc):
            _c0 = _nc[0]
            _rows = (getattr(_c0, "ndim", None) == 1
                     or (isinstance(_c0, (list, tuple)) and len(_c0)
                         and not hasattr(_c0[0], "__len__")))
            _ne = len(_nc) if _rows else sum(len(c) for c in _nc)
            _npe = len(_c0) if _rows else None
        else:
            _ne = len(_nc)
            _npe = (int(_nc.shape[1])
                    if getattr(_nc, "ndim", 1) == 2 else None)
        # a 3-D cell arity names the element (the name alone carries the
        # order; other arities untagged)
        _otag = ({4: "- tet4", 10: "- tet10"}.get(_npe, "")
                 if _dim == 3 else "")
        del _ne
        _dofs = 3 * len(_d["nodes"])
    except Exception:
        _dofs = None
    print(" msg       : %s (%dD SG%s)" % (_eng, _dim, _otag))
    if _dofs is not None:
        print(" dofs      : %d" % _dofs)
    print(" analysis  : %s" % ("homo" if analysis == "H" else "dehom"))
    print(" macro     : %s, %s%s"
          % (_MDL.get(int(hdr.get("n_model", 2)), "?"),
             "shear-refined" if int(hdr.get("refined", 0)) else "classical",
             ", aperiodic" if int(hdr.get("aperiodic", 0)) else ""))
    if force_superlu:
        from . import sg_assembly as _sga
        _sga.DIRECT_BACKEND = "superlu"
    # the solver line is printed by sg_homo once the family is RESOLVED
    # (an explicit --solver and an auto pick print identically)
    print("")

    # the terminal route is classical by default for EVERY macro model;
    # `refined: 1` in the file is the explicit upgrade.  For a dehom run
    # the .ff is peeked FIRST: its optional `q_reaction:` key is a
    # LOADING-side declaration the homogenization's load ladder consumes
    # (the yaml stays purely structural).
    base = os.path.splitext(path)[0]
    state = read_ff_state(base + ".ff") if analysis == "D" else None
    q_react = None if state is None else state.get("q_reaction")
    has_d2 = state is not None and any(
        state.get(k) is not None for k in ("dE11", "dE12", "dE22"))
    if has_d2 and q_react == "tau":
        # the forbidden pairing (2026-08-25 audit): the tau-reacted
        # pressure column already carries the moment-gradient (shear-
        # outflow) transverse content the Eq. 64-66 d2eps chains add,
        # so running both double-counts it.  Respect the explicit
        # override, but say so.
        print("WARNING: q_reaction: tau combined with d2eps_* drivers"
              " double-counts the moment-gradient transverse content"
              " (sigma_i3/sigma_33 overshoot) -- drop the d2eps lines"
              " (tau subsumes them) or use uniform")
    r = plate_homo_2d(path, refined=int(hdr.get("refined", 0)),
                      q_reaction=q_react,
                      solver=solver or "auto",
                      recovery=(analysis == "D"))
    print(r["law_title"] + ":")
    print(r["law"])
    if analysis == "H":
        print("Homogenization stored in %s.out" % base)
        print("Time taken: %.2f sec" % float(r["solve_time"]))
        _print_gpu_peak()

    if analysis == "D":
        import numpy as np

        from .sg_dehom import dehom_fields, export_gauss, gauss_coords

        # state was already read above (its q_reaction fed the ladder)
        if state is not None:
            eps = np.linalg.solve(np.asarray(r["C_eff"], float),
                                  state["FF"])
        else:
            eps = hdr.get("epsilon_bar")
            if eps is None or len(eps) != 6:
                raise SystemExit(
                    "analysis D needs the macro state: either %s.ff (u, theta,"
                    " C, FF) or `epsilon_bar:` in the yaml header"
                    % os.path.basename(base))
            eps = np.asarray([float(x) for x in eps], float)
        # the Eq. 63 refined recovery: strain-derivative + Q drivers
        # ride in the .ff as OPTIONAL keys; a classical run (refined: 0)
        # cannot consume them -- warn instead of silently dropping
        dE1 = dE2 = Qff = None
        if state is not None and (state.get("dE1") is not None
                                  or state.get("dE2") is not None):
            if r.get("V11bar") is None:
                print("note: %s.ff carries strain derivatives, but the"
                      " yaml ran classical (refined: 0) -- V2 recovery"
                      " skipped; set `refined: 1` to use them"
                      % os.path.basename(base))
            else:
                dE1, dE2 = state.get("dE1"), state.get("dE2")
                Qff = state.get("Q")
                print("V2 recovery: deps_dx1/deps_dx2 drive the Eq. 63"
                      " refined term%s" % ("" if Qff is None else
                                           ", Q-consistency rescale on"))
        qt6 = qb6 = None
        if state is not None and (state.get("qt6") is not None
                                  or state.get("qb6") is not None):
            if r.get("V1Lt") is None:
                print("note: %s.ff carries face pressures (qt6/qb6),"
                      " but this SG stores no load ladder (needs a"
                      " refined single-batch plate run) -- pressure"
                      " recovery skipped" % os.path.basename(base))
            else:
                qt6, qb6 = state.get("qt6"), state.get("qb6")
                print("q recovery: face pressure drives the load-ladder"
                      " term (qt %s, qb %s)"
                      % ("-" if qt6 is None else "%g" % qt6[0],
                         "-" if qb6 is None else "%g" % qb6[0]))
        dE11 = dE12 = dE22 = None
        if state is not None and any(
                state.get(k) is not None
                for k in ("dE11", "dE12", "dE22")):
            if r.get("V21") is None:
                print("note: %s.ff carries second strain derivatives"
                      " (d2eps_*), but this SG stores no V2 chains"
                      " (needs a refined single-batch plate run)"
                      " -- second-order recovery skipped"
                      % os.path.basename(base))
            elif (r.get("q_reaction") == "tau"
                  and (state.get("q_reaction") or "") != "tau"):
                # AUTO picked tau (in-plane-heterogeneous cell): the
                # tau-reacted load column subsumes the moment-gradient
                # content, so the d2eps chains would double-count it
                # (the HC_pm45-validated mode is tau WITHOUT d2).  The
                # Yu d2 composition remains available by forcing
                # q_reaction: uniform in the .ff.
                print("d2eps drivers DROPPED: the auto-selected tau"
                      " reaction already carries the moment-gradient"
                      " content on this heterogeneous cell (HC_pm45"
                      " mode; tau + d2eps double-counts -- 2026-08-25"
                      " audit).  Force q_reaction: uniform in the .ff"
                      " to run the Yu d2 composition instead")
            else:
                dE11 = state.get("dE11")
                dE12 = state.get("dE12")
                dE22 = state.get("dE22")
                print("V2 second order: d2eps drive the Eq. 64-66"
                      " two-chain recovery (tilt/detilt row split)")
        Gam, Sig, U = dehom_fields(r, eps, dE1=dE1, dE2=dE2, Q=Qff,
                                   qt6=qt6, qb6=qb6,
                                   dE11=dE11, dE12=dE12, dE22=dE22)
        # recovery computes in the SG-global frame (angle baked into C);
        # the default output rotates each element into its PLY frame
        if frame == "material":
            from .sg_dehom import material_frame_fields
            Gam, Sig = material_frame_fields(Gam, Sig, r)
            frame_note = ("material (ply axes, theta = +angle about the"
                          " thickness axis; U stays SG-global)")
            print("output frame: MATERIAL (ply) -- pass --global for the"
                  " SG axes")
        else:
            frame_note = "global (SG axes)"
            print("output frame: GLOBAL (SG axes)")
        if state is not None:
            # total displacement = macro rigid motion + SG warping:
            # U_i = u_i + (C_ij - d_ij) y_j + w_i  (VABS recovery form)
            xy = np.asarray(gauss_coords(r))
            xy = xy.reshape(-1, xy.shape[-1])      # (N, n_sg), n_sg = 1..3
            y = np.zeros((xy.shape[0], 3))
            y[:, :min(3, xy.shape[1])] = xy[:, :3]
            rig = state["u"] + y @ (state["C"] - np.eye(3)).T
            U = np.asarray(U).reshape(-1, 3) + rig
            U = U.reshape(np.asarray(Gam).shape[:-1] + (3,))
        export_gauss(r, Gam, Sig, base + "_dehom", U_eqd=U,
                     frame=frame_note)
        print("Local field files are computed and stored.")
        print("Time taken: %.2f sec" % (_time.perf_counter() - _t0))
        _print_gpu_peak()
    return 0


def _print_gpu_peak():
    """One line of TRUE device usage after a GPU run -- the nvidia-smi /
    Colab gauge shows the XLA preallocation pool (75% of the card), not
    what the run needed.  Silent on CPU and on any backend that does not
    expose memory_stats."""
    try:
        import jax as _jax
        dev = _jax.local_devices()[0]
        if dev.platform != "gpu":
            return
        peak = dev.memory_stats().get("peak_bytes_in_use")
        if peak:
            print("GPU peak: %.1f GB" % (peak / 1e9))
    except Exception:
        pass


def read_ff_state(path):
    """The macro state of a dehomogenization, from <yaml stem>.ff.

    The file is the analysis's loading side, kept out of the SG yaml
    (which describes the structure alone).  Four blocks, all optional
    except FF:

        u:     [u1 u2 u3]          macro displacement
        theta: [t1 t2 t3]          macro rotation [rad], about y1 y2 y3
        C:     3x3 rows            direction cosines of the macro frame;
                                   defaults to the rotation `theta` builds
        FF:    [6 values]          generalized macro forces, in the same
                                   order as the model's stiffness --
                                   plate [N11 N22 N12 M11 M22 M12],
                                   beam  [F1 F2 F3 M1 M2 M3],
                                   solid [S11 S22 S33 S23 S13 S12]

    OPTIONAL refined-recovery drivers (plate, refined: 1 only):

        Q:         [Q1 Q2]         transverse shear resultants -- turns
                                   on the Q-consistency rescale of the
                                   recovered sigma_13/sigma_23
        deps_dx1:  [6 values]      d/dx1 of the plate strain measures
        deps_dx2:  [6 values]      d/dx2 -- both in the engineering
                                   order [e11 e22 2e12 k11 k22 2k12];
                                   they drive the Eq. 63 V2 term
        qt6:       [6 values]      TOP-face pressure and its in-plane
                                   derivatives [q q,1 q,2 q,11 q,12
                                   q,22], q positive pushing INTO the
                                   face -- drives the load-ladder
                                   recovery (sigma33 face content,
                                   sigma22 load content)
        qb6:       [6 values]      same for the BOTTOM face
        q_reaction: uniform | tau  how the pressure load column reacts
                                   its net face force inside the cell:
                                   'uniform' (default, the <w> = 0 KKT
                                   reaction) or 'tau' (along the cell's
                                   own sigma_xz path under unit Q1 --
                                   for in-plane-heterogeneous cells,
                                   see plate_homo_2d's q_reaction
                                   docstring).  A LOADING declaration,
                                   so it lives here, not in the yaml
        d2eps_dx1dx1: [6 values]   SECOND derivatives of the plate
        d2eps_dx1dx2: [6 values]   measures (E,11 E,12 E,22) -- drive
        d2eps_dx2dx2: [6 values]   the Eq. 64-66 second-order two-chain
                                   recovery (2-D plate SGs)

    The macro strain is C_eff^-1 FF, and u/C add the rigid motion to the
    recovered displacement.

    In:  path str -- <base>.ff
    Out: dict {u (3,), theta (3,), C (3, 3), FF (6,), dE1 (6,)|None,
         dE2 (6,)|None, Q (2,)|None, qt6 (6,)|None, qb6 (6,)|None} |
         None when the file is absent or is not this format (a
         whitespace station table, the 1-D plate field route, reads as
         None)."""
    import numpy as np
    import yaml as _yaml

    if not os.path.exists(path):
        return None
    try:
        d = _yaml.safe_load(open(path))
    except _yaml.YAMLError:
        return None
    if not isinstance(d, dict) or "FF" not in d:
        return None
    u = np.asarray(d.get("u", [0.0, 0.0, 0.0]), float).reshape(3)
    th = np.asarray(d.get("theta", [0.0, 0.0, 0.0]), float).reshape(3)
    if d.get("C") is not None:
        C = np.asarray(d["C"], float).reshape(3, 3)
    else:                                   # small-rotation frame from theta
        C = np.eye(3) + np.array([[0.0, -th[2], th[1]],
                                  [th[2], 0.0, -th[0]],
                                  [-th[1], th[0], 0.0]])
    opt = lambda k, n: (None if d.get(k) is None else       # noqa: E731
                        np.asarray(d[k], float).reshape(n))
    return {"u": u, "theta": th, "C": C,
            "FF": np.asarray(d["FF"], float).reshape(6),
            "dE1": opt("deps_dx1", 6), "dE2": opt("deps_dx2", 6),
            "Q": opt("Q", 2),
            "qt6": opt("qt6", 6), "qb6": opt("qb6", 6),
            "dE11": opt("d2eps_dx1dx1", 6),
            "dE12": opt("d2eps_dx1dx2", 6),
            "dE22": opt("d2eps_dx2dx2", 6),
            "q_reaction": (str(d["q_reaction"]).strip().lower()
                           if d.get("q_reaction") else None)}
