"""opensg_solid command line -- the file is the whole problem:

    opensg_solid <sg.yaml>        homogenization (the default)
    opensg_solid <sg.yaml> D      dehomogenization: homogenize, then recover
    opensg_solid <sg.yaml> --mesh  the run above, AND force a redraw of
                                  <base>_mesh.png (a flag, not an
                                  analysis: combines with H | D)

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
stress/strain into its own PLY axes (theta = -angle about the thickness
axis, the flat_pm45-gated Abaqus equivalence -- what an Abaqus odb shows
UNTRANSFORMED when the section carries *Orientation); --global keeps
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

BANNER = """
 ============================================================================
 OpenSG -- a multiscale structural analysis tool based on the Mechanics of
 Structure Genome (MSG), developed by the Multiscale Structural Mechanics
 Group led by Prof. Wenbin Yu at Purdue University
 ============================================================================
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
    print(" input     : %s" % os.path.abspath(path))
    print(" msg       : %s" % resolve_msg(path))     # the engine that owns it
    print(" SG dim    : %dD" % node_span_dim(path))   # the space the SG occupies
    print(" analysis  : %s" % ("homogenization" if analysis == "H"
                               else "dehomogenization"))
    print(" macro model: %s, %s%s"
          % (_MDL.get(int(hdr.get("n_model", 2)), "?"),
             "shear-refined" if int(hdr.get("refined", 0)) else "classical",
             ", aperiodic" if int(hdr.get("aperiodic", 0)) else ""))
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
                      q_reaction=q_react)
    print(r["law_title"] + ":")
    print(r["law"])
    if analysis == "H":
        print("Homogenization stored in %s.out" % base)
        print("Time taken: %.2f sec" % float(r["solve_time"]))

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
            frame_note = ("material (ply axes, theta = -angle about the"
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
    return 0


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
