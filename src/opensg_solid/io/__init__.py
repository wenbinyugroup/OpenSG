"""opensg_solid.io -- format conversions of the general SG engine.

    sc_to_yaml.convert     SwiftComp `.sc` -> the canonical SG yaml + .msh
    sc_to_yaml.sg_to_yaml  PreVABS/VABS `.sg` -> the 2-D solid MESH-dialect
                           yaml (theta1 x theta3 material frames; read back
                           with sg_input.read_opensg_yaml)
    msh_to_yaml.convert    gmsh mesh -> the solid mesh dialect (materials
                           optional: none -> a FILL_IN template the
                           check_filled gate refuses to run)
    sg_input               yaml -> SwiftComp `.sc` / VABS `.sg` writers
    yaml_to_sc.convert     the named `.sc` entry behind `opensg yaml_to_sc`
                           (a printed-report front onto sg_input, with
                           n_model/refined ALWAYS from the yaml header)
    ff_to_glb.convert      its dehom side (`opensg ff_to_glb`): the `.ff`
                           macro state -> the SwiftComp `<base>.sc.glb`
                           (SCManual section 9; id1=0 generalized
                           stresses, RM plate 8 resultants / else 6)
    k_file                 the ONE `.K` / `.sc.k` reader + compare_to_K

(the opensg_shell.helper
analog on the solid side): sc_to_yaml turns a SwiftComp `.sc` structure
gene into the OpenSG solid SG yaml (+ a gmsh `.msh` sidecar), and
msh_to_yaml goes the other way round the loop -- a gmsh `.msh` mesh plus
the caller's materials into the `nodes`/`elements`/`elementOrientations`
solid SG yaml dialect the 2-D SG drivers read.

sg_input.py closes that loop: it is the ONE yaml -> SG-INPUT writer, in
either dialect of the Yu-group SG-input family -- the SwiftComp `.sc`
sc_to_yaml reads back, and the VABS `.sg` that until now could only be
produced by going out to PreVABS.  It is explicit about which of the two
it is writing and documents every place they differ (header flags,
element record, layer table, material block, trailing normalization);
opensg_solid.rm_plate_1D.helper.yaml_to_sc is a shim onto it.  It is also
explicit about what it CANNOT say: the paths no shipped deck evidences
(the beam `.sc` header, the trans_flag=1 orientation block, a nonzero
`.sc` density, tet10, comments) raise rather than emit a guess, and its
module docstring carries the validated / not-validated list.

k_file.py is the OUTPUT-side counterpart and the one place a `.K` is
parsed: the VABS beam cross-section `.K`, the SwiftComp 3-D solid
`.sc.k` and OpenSG's own write_sc_K / write_vabs_k `.out` all go through
read_k_blocks, with read_beam_k / read_solid_k the two typed views and
compare_to_K the reusable term-by-term verifier against a vendor
reference.  It lives on the solid side because the dependency runs one
way -- opensg_shell imports opensg_solid, never the reverse -- so both
engines can reach it, and the shell side's
`opensg_shell.pynumad.read_k_file` is the entry point to point at it.
Import it explicitly
(`from opensg_solid.io.k_file import ...`); this package __init__
stays import-light on purpose.
"""
