"""opensg_solid command line -- the file is the whole problem:

    opensg_solid <sg.yaml>        homogenization (the default)
    opensg_solid <sg.yaml> D      dehomogenization: homogenize, then recover

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

H writes the timed <base>.out (SwiftComp .K layout) + <base>_mesh.png;
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
    if not 1 <= len(argv) <= 2 or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 2
    path = argv[0]
    if not os.path.exists(path):
        alt = os.path.splitext(path)[0] + ".yaml"
        raise SystemExit("no such file: %s%s" % (
            path, "" if not os.path.exists(alt) else
            "\ndid you mean %s ?" % alt))

    from .sg_mesh import read_yaml_header, resolve_msg, sg_dim, node_span_dim
    from .sg_homo import plate_homo_2d

    hdr = read_yaml_header(path)
    analysis = str(argv[1] if len(argv) == 2
                   else hdr.get("analysis", "H")).strip().upper()
    if analysis not in ("H", "D"):
        raise SystemExit("the analysis argument must be H (homogenization)"
                         " or D (dehomogenization), got %r" % argv[1])
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
    # `refined: 1` in the file is the explicit upgrade
    r = plate_homo_2d(path, refined=int(hdr.get("refined", 0)))
    base = os.path.splitext(path)[0]
    print(r["law_title"] + ":")
    print(r["law"])
    if analysis == "H":
        print("Homogenization stored in %s.out" % base)
        print("Time taken: %.2f sec" % float(r["solve_time"]))

    if analysis == "D":
        import numpy as np

        from .sg_dehom import dehom_fields, export_gauss, gauss_coords

        state = read_ff_state(base + ".ff")
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
        Gam, Sig, U = dehom_fields(r, eps)
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
        export_gauss(r, Gam, Sig, base + "_dehom", U_eqd=U)
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

    The macro strain is C_eff^-1 FF, and u/C add the rigid motion to the
    recovered displacement.

    In:  path str -- <base>.ff
    Out: dict {u (3,), theta (3,), C (3, 3), FF (6,)} | None when the
         file is absent or is not this format (a whitespace station
         table, the 1-D plate field route, reads as None)."""
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
    return {"u": u, "theta": th, "C": C,
            "FF": np.asarray(d["FF"], float).reshape(6)}
