"""opensg command line -- ONE command for both MSG engines:

    opensg <sg.yaml>        homogenization (the default)
    opensg <sg.yaml> D      dehomogenization: homogenize, then recover

    opensg gen_windio_cs <windio.yaml>
                            windIO blade -> one 1-D shell cross-section SG
                            yaml per station (+ the PreVABS XML byproduct by
                            default) -- the opensg_shell windio route
                            (gen_windio_cs --help for the flags)

The file says which engine owns it.  `msg: shell` in the yaml header sends
it to opensg_shell (the msg-shell contour / surface SG), `msg: solid` to
opensg_solid (the general 1-D/2-D/3-D SG engine); without the key the mesh
DIALECT decides, so every yaml written before the key existed still runs.

Nothing else is decided here.  n_model, refined, analysis, epsilon_bar,
omega, aperiodic and the rest of the header are read by the engine that
runs, exactly as when its own command is used -- `opensg <yaml>` and
`opensg_shell <yaml>` / `opensg_solid <yaml>` print the same thing, because
this dispatcher prints nothing of its own.

    opensg_shell --help     the msg-shell header contract
    opensg_solid --help     the general SG engine's header contract
"""
import os
import sys


def main(argv=None):
    """Resolve the engine the yaml names and hand it the untouched argv.

    In:  argv (list[str] | None) -- [yaml_path, "H"|"D" (optional)];
         None reads sys.argv
    Out: int exit code -- whatever the engine's own main() returns
         (0 ok, 2 usage)."""
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "gen_windio_cs":
        # the windIO cross-section generator lives in the shell engine
        from opensg_shell.cli import main as _engine
        return _engine(argv)
    if not 1 <= len(argv) <= 2 or argv[0] in ("-h", "--help"):
        from opensg_solid.cli import BANNER          # the ONE banner
        print(BANNER)
        print(__doc__)
        return 2

    if not os.path.exists(argv[0]):
        # the "no such file" (and "did you mean <stem>.yaml ?") message is the
        # engines' own and is engine-neutral: let one of them raise it rather
        # than write a second copy here
        from opensg_solid.cli import main as _engine
        return _engine(argv)

    from opensg_solid.sg_mesh import resolve_msg
    if resolve_msg(argv[0]) == "shell":
        from opensg_shell.cli import main as _engine
    else:
        from opensg_solid.cli import main as _engine
    return _engine(argv)
