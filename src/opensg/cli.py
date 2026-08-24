"""opensg command line -- ONE command for both MSG engines:

    opensg <sg.yaml>        homogenization (the default)
    opensg <sg.yaml> D      dehomogenization: homogenize, then recover.
                            Output frame: --material (each element's PLY
                            axes, the DEFAULT) | --global (the SG axes
                            every pre-flag .SM was written in)
    opensg <sg.yaml> --mesh  run as above AND (re)draw <base>_mesh.png,
                            elements coloured by material -- a FLAG, not
                            an analysis, so it combines: `<sg.yaml> D
                            --mesh` dehomogenizes and redraws.  Every run
                            writes that PNG anyway when it is missing or
                            older than the yaml; --mesh forces it
                            (`-m` / `mesh` are accepted too)

    opensg gen_windio_cs <windio.yaml>
                            every blade station -> VABS-layout .out records
                            + spanwise .dat tables, fully in memory (any
                            dialect: windIO v1/v2 or pyNuMAD;
                            gen_windio_cs --help for the flags)

    opensg windio_st <windio.yaml> <r> [r ...]
                            one-shot station bypass: blade + span r ->
                            Timoshenko 6x6 printed and stored as the
                            VABS-layout <tag>.out, nothing else
                            (windio_st --help for the flags)

    opensg pynumad <blade.yaml> <st-id>
                            the blade route by st-id (any dialect; pyNuMAD
                            width-placed reinforcements resolved): st-id =
                            0-based station index | span r | "all" -- one
                            station prints the Timoshenko 6x6 + the
                            cross-check vs the file's own
                            elastic_properties_mb; "all" sweeps every
                            station (pynumad --help for the flags)

    opensg sc_to_yaml <file.sc> [...] --n_model {1,2,3}
                            SwiftComp .sc -> <base>.yaml + <base>.msh (the
                            solid-dialect SG yaml opensg_solid reads).
                            --n_model is the ONLY flag; everything else
                            (refined: 1, omega, ...) the user adds BY HAND
                            in the emitted yaml header.  An existing
                            <base>.yaml is therefore KEPT (hand edits
                            win) -- delete it to re-emit

    opensg yaml_to_sc <file.yaml> [...]
                            the REVERSE: SG yaml -> SwiftComp <base>.sc.
                            No model flags -- the yaml header's n_model:
                            (2 plate | 3 solid) and refined: decide the
                            .sc header lines, and the SG dimension is
                            measured from the mesh, so the file is run as
                            `SwiftComp <base>.sc 2D|3D H`.  Material
                            `angle:` blocks are folded into pre-rotated
                            type-2 (21-constant) materials (an nlayer=0
                            .sc has no angle slot; the stiffness is
                            identical); per-element frames need
                            --orientation bake|ignore.  Globs expand and
                            every call REWRITES the .sc
                            (yaml_to_sc --help for the flags)

    opensg ff_to_glb <file.ff> [...]
                            the DEHOM side of yaml_to_sc: the `.ff` macro
                            state (u / theta|C / FF [/ Q]) -> the
                            SwiftComp dehom input <base>.sc.glb (SCManual
                            section 9: v, Cij, id1=0, the generalized
                            STRESSES -- 8 resultants for a refined: 1
                            plate, N13/N23 from the .ff `Q:` or 0 0 with
                            a note; 6 otherwise).  The resultant count
                            comes from `<stem>.yaml` beside the .ff when
                            it exists, or --yaml / --n_model/--refined;
                            with none of those the RM plate is assumed
                            and printed.  Then run
                            `SwiftComp <base>.sc 2D|3D L`
                            (ff_to_glb --help for the flags)

    opensg inp_to_yaml <file.inp> [...] --n_model {1,2,3} [--refined]
                            Abaqus .inp mesh -> <base>.yaml + <base>.msh,
                            RUNNABLE (materials come from the deck; the
                            binding follows its section cards -- without
                            any, ALL deck materials are emitted, every
                            element defaults to material 1 with a loud
                            note, and the user edits mat_id:/materials:
                            in the yaml; --material NAME / --map
                            ELSET=MAT are optional overrides).  A mixed
                            S4R+S3 mesh comes out as ragged cells for
                            the batched homo.  --n_model REQUIRED (never
                            guessed); globs expand; --force re-emits
                            (inp_to_yaml --help for the flags)

    opensg msh_to_yaml <file.msh> [...] [--n_model {1,2,3}]
                            gmsh 2.2 solid mesh -> <base>.yaml TEMPLATE:
                            the mesh side (nodes / elements /
                            elementOrientations / sets, one set per gmsh
                            physical tag, named from $PhysicalNames) is
                            complete; `materials:` -- and n_model, unless
                            the flag gives it -- come out as FILL_IN
                            placeholders, because a .msh carries no
                            constitutive data (and no orientation either:
                            every element gets the constant
                            e1-out-of-plane frame).  opensg refuses to run
                            the yaml until the FILL_IN fields are
                            replaced.  Globs expand, up-to-date yamls are
                            skipped unless --force (msh_to_yaml --help
                            for the flags)

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
    if argv and argv[0] in ("gen_windio_cs", "windio_st", "pynumad"):
        # the blade routes live in the shell engine's pynumad package
        from opensg_shell.cli import main as _engine
        return _engine(argv)
    if argv and argv[0] == "sc_to_yaml":
        # format conversion, owned here: the emitted yaml carries the
        # `msg: solid` header, so the engine choice stays in the FILE
        return sc_to_yaml(argv[1:])
    if argv and argv[0] == "msh_to_yaml":
        # same ownership rule; the emitted yaml is a `msg: solid` template
        return msh_to_yaml(argv[1:])
    if argv and argv[0] == "inp_to_yaml":
        # same ownership rule; an Abaqus deck HAS materials, so the yaml
        # comes out runnable (unlike the msh template)
        return inp_to_yaml(argv[1:])
    if argv and argv[0] == "yaml_to_sc":
        # the reverse of sc_to_yaml: the yaml itself states n_model /
        # refined / the SG dimension, so no model flags exist here
        return yaml_to_sc(argv[1:])
    if argv and argv[0] == "ff_to_glb":
        # the dehom side of yaml_to_sc: .ff macro state -> <base>.sc.glb
        return ff_to_glb(argv[1:])
    # --mesh and the D frame flags are consumed by the ENGINE; they must
    # not count toward the <yaml> [H|D] arity here
    _n = len([a for a in argv if str(a).strip().lower()
              not in ("--mesh", "-m", "mesh", "--material", "--global")])
    if not 1 <= _n <= 2 or argv[0] in ("-h", "--help"):
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


def _yaml_header_matches(yml, n_model, refined):
    """Does an already-emitted SG yaml carry the REQUESTED header?

    The sidecar freshness check is otherwise mtime-only, so re-running
    a converter with a different --n_model/--refined would silently
    keep the old file (the emitted header is exactly what those flags
    control).  A cheap leading-line scan -- never a mesh parse, so the
    skip path stays engine-free.

    In:  yml str -- the emitted yaml; n_model int | None; refined int
    Out: bool -- True when the file already states this header (an
         unreadable or headerless file returns False: re-emit)."""
    want = {"n_model": None if n_model is None else int(n_model),
            "refined": int(refined)}
    got = {}
    try:
        with open(yml) as f:
            for ln in f:
                key = ln.split(":", 1)[0].strip()
                if key in ("nodes", "cells", "mat_id", "materials",
                           "elements"):
                    break                      # mesh starts; header done
                if key in want:
                    try:
                        got[key] = int(ln.split(":", 1)[1]
                                       .split("#")[0].strip())
                    except (IndexError, ValueError):
                        return False
    except OSError:
        return False
    for k, v in want.items():
        if v is not None and got.get(k) != v:
            return False
    return True


def sc_to_yaml(argv):
    """The `opensg sc_to_yaml` subcommand: SwiftComp .sc -> SG yaml + .msh.

    A thin batch driver over opensg_solid.io.sc_to_yaml.convert.
    --n_model is the ONLY flag: every other header key (refined:,
    omega:, ...) the user adds BY HAND in the emitted yaml afterwards.
    For exactly that reason an existing UP-TO-DATE <base>.yaml is KEPT,
    never overwritten (convert()'s hand-edits-win rule); a stale pair
    (the .sc is newer) is re-emitted with the requested n_model.
    Delete the yaml to force a fresh emit.

    In:  argv (list[str]) -- everything after the `sc_to_yaml` keyword:
         .sc paths/globs + --n_model {1,2,3}
    Out: int exit code (0 ok, 1 any file failed, 2 usage)."""
    import argparse
    import glob

    p = argparse.ArgumentParser(
        prog="opensg sc_to_yaml",
        description="SwiftComp .sc -> OpenSG solid SG yaml (+ gmsh "
                    ".msh).  An existing up-to-date <base>.yaml is kept"
                    " -- hand edits win; delete it to re-emit")
    p.add_argument("sc", nargs="+",
                   help=".sc file(s); globs expand")
    p.add_argument("--n_model", "--n-model", dest="n_model", type=int,
                   required=True, choices=(1, 2, 3),
                   help="yaml header n_model: 1 beam, 2 plate, 3 solid")
    a = p.parse_args(argv)

    paths = sum((sorted(glob.glob(s)) or [s] for s in a.sc), [])
    failed = 0
    for sc in paths:
        base = os.path.splitext(sc)[0]
        yml, msh = base + ".yaml", base + ".msh"
        try:
            if not os.path.exists(sc):
                raise FileNotFoundError("no such file: %s" % sc)
            kept = (os.path.exists(yml) and os.path.exists(msh)
                    and os.path.getmtime(yml) >= os.path.getmtime(sc)
                    and os.path.getmtime(msh) >= os.path.getmtime(sc))
            from opensg_solid.io.sc_to_yaml import convert
            convert(sc, n_model=a.n_model)
            if kept:
                print("sc_to_yaml: kept the existing %s.yaml (hand"
                      " edits win) -- delete it to re-emit from the"
                      " .sc" % base)
        except Exception as e:
            print("sc_to_yaml: %s FAILED: %s" % (sc, e))
            failed += 1
    return 1 if failed else 0


def yaml_to_sc(argv):
    """The `opensg yaml_to_sc` subcommand: SG yaml -> SwiftComp .sc.

    A thin batch driver over opensg_solid.io.yaml_to_sc.convert (itself
    the io.sg_input `.sc` writer).  Deliberately NO --n_model/--refined
    flags: the subcommand's contract is that the yaml header states the
    macro model (`n_model:` 2 plate | 3 solid; 1 beam is refused with the
    sg_input rationale) and the submodel (`refined:` 0 classical |
    1 shear-refined), and the SG dimension is measured from the node
    columns that vary -- the same self-describing-file rule `opensg
    <yaml>` runs by.  Every call REWRITES the .sc: writing one is cheap
    next to the SwiftComp run it feeds, and the failure worth avoiding
    is the other one -- an mtime-fresh .sc quietly handing SwiftComp the
    previous mesh.

    In:  argv (list[str]) -- everything after the `yaml_to_sc` keyword:
         yaml paths/globs, plus --out-base BASE (single input only),
         --orientation {bake,ignore} (required only when the yaml carries
         per-element elementOrientations), --drop-density
    Out: int exit code (0 ok, 1 any file failed, 2 usage)."""
    import argparse
    import glob

    p = argparse.ArgumentParser(
        prog="opensg yaml_to_sc",
        description="OpenSG SG yaml -> SwiftComp .sc; n_model/refined come "
                    "from the yaml header and the SG dimension from the "
                    "mesh (nothing is guessed beyond the file's own "
                    "defaults: n_model 2, refined 0).  Every call "
                    "rewrites the .sc")
    p.add_argument("yaml", nargs="+",
                   help="SG yaml file(s); globs expand (PowerShell passes "
                        "them through literally)")
    p.add_argument("--out-base", default=None, metavar="BASE",
                   help="basename for the emitted .sc (single input only; "
                        "default: the yaml stem)")
    p.add_argument("--orientation", default=None,
                   choices=("bake", "ignore"),
                   help="what to do with per-element elementOrientations "
                        "frames, which a trans_flag=0 .sc has no slot "
                        "for: 'bake' folds each distinct frame into its "
                        "own pre-rotated type-2 material (exact), "
                        "'ignore' drops them (only correct when they are "
                        "all the identity).  Only needed when the yaml "
                        "carries such frames")
    p.add_argument("--drop-density", action="store_true",
                   help="write the vendor-evidenced `0 0` aux line for a "
                        "material with NONZERO density instead of "
                        "refusing (an elastic .sc run does not use it)")
    a = p.parse_args(argv)

    paths = sum((sorted(glob.glob(s)) or [s] for s in a.yaml), [])
    if a.out_base is not None and len(paths) != 1:
        raise SystemExit("--out-base needs exactly one input yaml, got %d"
                         % len(paths))
    failed = 0
    for yml in paths:
        base = a.out_base or os.path.splitext(yml)[0]
        sc = base + ".sc"
        try:
            if not os.path.exists(yml):
                raise FileNotFoundError("no such file: %s" % yml)
            from opensg_solid.io.yaml_to_sc import convert
            convert(yml, out_path=sc, orientation=a.orientation,
                    drop_density=a.drop_density)
        except Exception as e:
            print("yaml_to_sc: %s FAILED: %s" % (yml, e))
            failed += 1
    return 1 if failed else 0


def msh_to_yaml(argv):
    """The `opensg msh_to_yaml` subcommand: gmsh .msh -> solid SG yaml
    TEMPLATE.

    A thin batch driver over opensg_solid.io.msh_to_yaml.convert with NO
    materials: a gmsh mesh carries geometry and physical tags only, so the
    emitted yaml is complete on the mesh side and a marked FILL_IN template
    on the `materials:` side -- `opensg <yaml>` refuses to run it until the
    placeholders are replaced (io.msh_to_yaml.check_filled).  The
    orientation is not in a .msh either; every element gets convert()'s
    constant e1-out-of-plane default frame.

    In:  argv (list[str]) -- everything after the `msh_to_yaml` keyword:
         .msh paths/globs, plus --out-base BASE (single input only), --force
    Out: int exit code (0 ok, 1 any file failed, 2 usage)."""
    import argparse
    import glob

    p = argparse.ArgumentParser(
        prog="opensg msh_to_yaml",
        description="gmsh 2.2 solid mesh -> OpenSG solid SG yaml TEMPLATE"
                    " (mesh blocks complete, `materials:` as FILL_IN"
                    " placeholders -- a .msh carries no constitutive data);"
                    " existing up-to-date yamls are skipped unless --force")
    p.add_argument("msh", nargs="+",
                   help=".msh file(s); globs expand (PowerShell passes "
                        "them through literally)")
    p.add_argument("--out-base", default=None, metavar="BASE",
                   help="basename for the emitted yaml (single input only; "
                        "default: the .msh stem)")
    p.add_argument("--n_model", "--n-model", dest="n_model", type=int,
                   default=None, choices=(1, 2, 3),
                   help="yaml header n_model: 1 beam, 2 plate, 3 solid -- "
                        "the macro model this SG homogenizes to.  Never "
                        "guessed from the mesh; omit to leave a FILL_IN "
                        "placeholder in the template")
    p.add_argument("--force", action="store_true",
                   help="re-emit even when <base>.yaml is newer than the "
                        ".msh")
    a = p.parse_args(argv)

    paths = sum((sorted(glob.glob(s)) or [s] for s in a.msh), [])
    if a.out_base is not None and len(paths) != 1:
        raise SystemExit("--out-base needs exactly one input .msh, got %d"
                         % len(paths))
    failed = 0
    for msh in paths:
        base = a.out_base or os.path.splitext(msh)[0]
        yml = base + ".yaml"
        try:
            if not os.path.exists(msh):
                raise FileNotFoundError("no such file: %s" % msh)
            fresh = (not a.force and os.path.exists(yml)
                     and os.path.getmtime(yml) >= os.path.getmtime(msh))
            if fresh:
                print("msh_to_yaml: %s up to date (--force to re-emit)"
                      % yml)
                continue
            if os.path.exists(yml):
                os.remove(yml)
            from opensg_solid.io.msh_to_yaml import convert
            r = convert(msh, out_path=yml, n_model=a.n_model)
            print("msh_to_yaml: %s -> %s\n"
                  "  %d nodes, %d elements (%d-node), set(s): %s\n"
                  "  MATERIALS / LAYUP NOT ADDED%s: `materials:` is a"
                  " FILL_IN template -- opensg\n  refuses to run this yaml"
                  " until every FILL_IN field is replaced"
                  % (os.path.basename(msh), os.path.basename(r["path"]),
                     r["n_nodes"], r["n_elements"], r["npe"],
                     ", ".join("%s (%d)" % kv for kv in r["sets"].items()),
                     "" if a.n_model is not None else
                     " (nor n_model -- that placeholder too)"))
        except Exception as e:
            print("msh_to_yaml: %s FAILED: %s" % (msh, e))
            failed += 1
    return 1 if failed else 0


def inp_to_yaml(argv):
    """The `opensg inp_to_yaml` subcommand: Abaqus .inp -> solid SG yaml.

    A thin batch driver over opensg_solid.io.inp_to_yaml.convert.  The
    deck carries its own materials, so the emitted yaml is RUNNABLE.
    The elset -> material binding comes from the deck's section cards
    when it has them; --material NAME / --map ELSET=MATERIAL are
    OPTIONAL overrides.  With neither, ALL deck materials are emitted
    and every element defaults to material 1 with a loud note -- the
    binding is the user's edit in the yaml, never a CLI requirement
    (material sets vary deck to deck).  A MIXED mesh (e.g. S4R + S3)
    comes out as ragged cells for the batched homo.

    In:  argv (list[str]) -- everything after the `inp_to_yaml` keyword:
         .inp paths/globs, --n_model {1,2,3} (REQUIRED, never guessed),
         --refined, --material NAME, --map ELSET=MAT [ELSET=MAT ...],
         --out-base BASE (single input only), --force
    Out: int exit code (0 ok, 1 any file failed, 2 usage)."""
    import argparse
    import glob

    p = argparse.ArgumentParser(
        prog="opensg inp_to_yaml",
        description="Abaqus .inp mesh -> OpenSG solid SG yaml (+ gmsh"
                    " .msh); materials come from the deck; existing"
                    " up-to-date sidecars are skipped unless --force")
    p.add_argument("inp", nargs="+",
                   help=".inp file(s); globs expand (PowerShell passes"
                        " them through literally)")
    p.add_argument("--n_model", "--n-model", dest="n_model", type=int,
                   required=True, choices=(1, 2, 3),
                   help="REQUIRED. yaml header n_model: 1 beam, 2 plate,"
                        " 3 solid -- never guessed from the mesh")
    p.add_argument("--refined", action="store_true",
                   help="emit `refined: 1` (shear-refined: plate ABDG /"
                        " beam Timoshenko).  Default 0 = classical")
    p.add_argument("--material", default=None, metavar="NAME",
                   help="OPTIONAL: one deck material for every element."
                        " Without it (and without sections/--map) all"
                        " deck materials are emitted, elements default"
                        " to material 1, and the binding is edited in"
                        " the yaml")
    p.add_argument("--map", nargs="+", default=None, metavar="ELSET=MAT",
                   help="OPTIONAL explicit elset -> material pairs (win"
                        " over the deck's section cards).  A value may"
                        " be MATERIAL@DEGREES so plies sharing one deck"
                        " material but differing in orientation become"
                        " separate SG materials, e.g."
                        " --map P1=Aluminum L1=Laminate@45"
                        " L2=Laminate@-45")
    p.add_argument("--out-base", default=None, metavar="BASE",
                   help="basename for the emitted sidecars (single input"
                        " only; default: the .inp stem)")
    p.add_argument("--force", action="store_true",
                   help="re-emit even when <base>.yaml/.msh are newer"
                        " than the .inp AND already carry the requested"
                        " n_model/refined (a changed header re-emits on"
                        " its own)")
    a = p.parse_args(argv)

    emap = None
    if a.map:
        emap = {}
        for pair in a.map:
            if "=" not in pair:
                raise SystemExit("--map takes ELSET=MATERIAL pairs, got"
                                 " %r" % pair)
            k, v = pair.split("=", 1)
            emap[k] = v
    paths = sum((sorted(glob.glob(s)) or [s] for s in a.inp), [])
    if a.out_base is not None and len(paths) != 1:
        raise SystemExit("--out-base needs exactly one input .inp, got"
                         " %d" % len(paths))
    failed = 0
    for inp in paths:
        base = a.out_base or os.path.splitext(inp)[0]
        yml, msh = base + ".yaml", base + ".msh"
        try:
            if not os.path.exists(inp):
                raise FileNotFoundError("no such file: %s" % inp)
            # an explicit binding (--map / --material) ALWAYS re-emits:
            # it changes mat_id/materials, which the header scan cannot
            # see, so a stale sidecar would silently ignore it
            fresh = (not a.force and a.map is None and a.material is None
                     and os.path.exists(yml)
                     and os.path.exists(msh)
                     and os.path.getmtime(yml) >= os.path.getmtime(inp)
                     and os.path.getmtime(msh) >= os.path.getmtime(inp)
                     and _yaml_header_matches(yml, a.n_model,
                                              int(a.refined)))
            if fresh:
                print("inp_to_yaml: %s.yaml/.msh up to date (--force to"
                      " re-emit)" % base)
                continue
            for stale in (yml, msh, base + "_sg.npz"):
                if os.path.exists(stale):
                    os.remove(stale)
            from opensg_solid.io.inp_to_yaml import convert
            convert(inp, out_base=a.out_base, n_model=a.n_model,
                    refined=int(a.refined), material=a.material,
                    elset_map=emap)
        except Exception as e:
            print("inp_to_yaml: %s FAILED: %s" % (inp, e))
            failed += 1
    return 1 if failed else 0

def ff_to_glb(argv):
    """The `opensg ff_to_glb` subcommand: `.ff` -> SwiftComp `.sc.glb`.

    A thin batch driver over opensg_solid.io.ff_to_glb.convert -- the
    dehomogenization side of `opensg yaml_to_sc`.  The `.ff` is the
    input; the resultant COUNT (RM plate 8, classical plate / solid 6)
    comes from `<stem>.yaml` beside it when one exists, or --yaml /
    --n_model/--refined, else the RM plate is assumed with a printed
    note.  Every call REWRITES the .glb, for the same reason yaml_to_sc
    rewrites the .sc.

    In:  argv (list[str]) -- everything after the `ff_to_glb` keyword:
         .ff paths/globs, plus --yaml YAML (single input only),
         --n_model {2,3} / --refined {0,1} explicit overrides,
         --out PATH (single input only; default <ff stem>.sc.glb)
    Out: int exit code (0 ok, 1 any file failed, 2 usage)."""
    import argparse
    import glob

    p = argparse.ArgumentParser(
        prog="opensg ff_to_glb",
        description="OpenSG .ff macro state -> SwiftComp dehom input "
                    "<base>.sc.glb.  The resultant count follows "
                    "<stem>.yaml beside the .ff (or --yaml / --n_model/"
                    "--refined; RM plate assumed otherwise) so the .glb "
                    "matches the .sc yaml_to_sc wrote.  Every call "
                    "rewrites the .glb")
    p.add_argument("ff", nargs="+",
                   help=".ff file(s); globs expand.  The .ff supplies "
                        "the numbers (u / theta|C / FF [/ Q])")
    p.add_argument("--yaml", default=None, metavar="YAML",
                   help="the SG yaml whose header (n_model/refined) "
                        "shapes the .glb (single input only; default: "
                        "<ff stem>.yaml beside the .ff when it exists)")
    p.add_argument("--n_model", "--n-model", dest="n_model", type=int,
                   default=None, choices=(2, 3),
                   help="explicit macro model: 2 plate, 3 solid (beam "
                        "is refused -- its .sc header is unvalidated)")
    p.add_argument("--refined", type=int, default=None, choices=(0, 1),
                   help="explicit submodel: 0 classical (6 resultants), "
                        "1 shear-refined RM (8)")
    p.add_argument("--out", default=None, metavar="PATH",
                   help="the .glb to write (single input only; default: "
                        "<ff stem>.sc.glb -- the name `SwiftComp "
                        "<stem>.sc ... L` looks for)")
    a = p.parse_args(argv)

    paths = sum((sorted(glob.glob(s)) or [s] for s in a.ff), [])
    if (a.yaml is not None or a.out is not None) and len(paths) != 1:
        raise SystemExit("--yaml/--out need exactly one input .ff, got %d"
                         % len(paths))
    failed = 0
    for ff in paths:
        try:
            if not os.path.exists(ff):
                raise FileNotFoundError("no such file: %s" % ff)
            from opensg_solid.io.ff_to_glb import convert
            convert(ff, yaml_path=a.yaml, out_path=a.out,
                    n_model=a.n_model, refined=a.refined)
        except Exception as e:
            print("ff_to_glb: %s FAILED: %s" % (ff, e))
            failed += 1
    return 1 if failed else 0
