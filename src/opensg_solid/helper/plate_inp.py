"""plate_inp.py -- the plate .msh -> a runnable Abaqus deck.

Defaults follow the HC_pm45 benchmark conventions, with the lessons of
that study baked in:

  step      *Static, nlgeom=NO -- LINEAR, matching the linear
            OpenSG/SwiftComp chain (the HC study's nonlinear reference
            cost a full apples-to-apples rerun);
  BC        `clamped`  = ENCASTRE all four side faces (the square-plate
            default), `clamped-x` = the two x faces only (the HC panel
            style);
  load      q = 1 MPa pushing DOWN on the TOP surface, applied as
            element-face *Dload lines -- so a NONUNIFORM pressure needs
            no user subroutine, each face just carries its own value:
              uniform    q(x, y) = q
              linear-x   q(x, y) = q x / a      (hydrostatic head on a
                                                 lock gate / tank floor:
                                                 constant q,1 -- the
                                                 refined-recovery driver)
              linear-y   q(x, y) = q y / b
            Other realistic q,1/q,2 loads worth running the same way:
            a bilinear ramp q x y / (a b), a sinusoidal (Navier) patch
            q sin(pi x / a) sin(pi y / b) -- both are one-line changes
            in _qval below;
  output    *Node Output U + *Element Output, position=CENTROIDAL S --
            the granularity every extraction script in the project
            already pairs against;
  material  from the SG yaml (the .msh carries only material TAGS):
            <msh stem minus _plate>.yaml by default, or `yaml_path`.
            type 0 -> *Elastic; type 1 -> ENGINEERING CONSTANTS with an
            *Orientation `3, -angle` card (OpenSG `angle: a` == Abaqus
            `3, -a`, the flat_pm45-gated map);
  elements  tet4 -> C3D4, hex8 -> C3D8I (incompatible modes: the
            established choice for linear-hex bending).

In:  the plate .msh (plate_mesh output), the SG yaml for materials
Out: <msh stem>.inp (or `out`)
"""
import os
import time

import numpy as np

# Abaqus face tables: which face label carries a pressure when ALL its
# nodes sit on the loaded surface (0-based local node ids)
_FACES = {4: [("P1", (0, 1, 2)), ("P2", (0, 3, 1)),
              ("P3", (1, 3, 2)), ("P4", (2, 3, 0))],
          8: [("P1", (0, 1, 2, 3)), ("P2", (4, 5, 6, 7)),
              ("P3", (0, 1, 5, 4)), ("P4", (1, 2, 6, 5)),
              ("P5", (2, 3, 7, 6)), ("P6", (3, 0, 4, 7))]}


def _is_iso(blk, rtol=1e-6):
    """Is this type-1 (nine engineering constants) block ONE isotropic
    material?  The loaders broadcast an isotropic E/G/nu to three, so the
    test is: equal triples and G12 = E/(2(1+nu)), no ply angle.

    In:  blk dict -- canonical material block; rtol float
    Out: bool."""
    if float(blk.get("angle", 0.0) or 0.0) != 0.0:
        return False
    e = [float(v) for v in blk["engineering"]]
    E, G, nu = e[0], e[3], e[6]
    same = all(abs(v - w) <= rtol * abs(v)
               for v, w in ((e[0], e[1]), (e[1], e[2]), (e[3], e[4]),
                            (e[4], e[5]), (e[6], e[7]), (e[7], e[8])))
    return same and abs(G - E / (2.0 * (1.0 + nu))) <= rtol * G


def _read_msh(path):
    """gmsh 2.2 (the plate_mesh/write_msh dialect) -> arrays.

    In:  path str
    Out: (nodes (N, 3), cells (E, k) 0-based, mats (E,))."""
    with open(path) as f:
        lines = f.read().split("\n")
    i = lines.index("$Nodes")
    nn = int(lines[i + 1])
    nodes = np.loadtxt(lines[i + 2:i + 2 + nn], usecols=(1, 2, 3))
    j = lines.index("$Elements")
    ne = int(lines[j + 1])
    raw = np.loadtxt(lines[j + 2:j + 2 + ne], dtype=np.int64)
    k = {4: 4, 5: 8}[int(raw[0, 1])]
    return nodes, raw[:, 5:5 + k] - 1, raw[:, 3].astype(int)


def plate_inp(msh_path, yaml_path=None, out=None, q=1.0, a=None,
              b=None, bc="clamped", load="uniform", job_note=""):
    """The Abaqus deck of a plate_mesh .msh.

    In:  msh_path str; yaml_path str | None -- the SG yaml with the
         `materials:` block (None -> <stem minus _plate>.yaml);
         out str | None (None -> <msh stem>.inp); q float [MPa],
         positive pushes DOWN on the top face; a, b float | None --
         the plate span/width for the nonuniform loads (None ->
         measured off the mesh); bc 'clamped' | 'clamped-x';
         load 'uniform' | 'linear-x' | 'linear-y'; job_note str --
         extra ** comment line
    Out: dict {inp, n_nodes, n_elems, n_top_faces, bc, load}."""
    t0 = time.perf_counter()
    nodes, cells, mats = _read_msh(msh_path)
    E, k = cells.shape
    etype = {4: "C3D4", 8: "C3D8I"}[k]
    base = os.path.splitext(msh_path)[0]
    out = out or base + ".inp"
    if yaml_path is None:
        yaml_path = (base[:-6] if base.endswith("_plate")
                     else base) + ".yaml"
    # load_sg_input reads both solid yaml spellings and hands back the
    # CANONICAL materials mapping {id: {type, ..., angle?}} either way
    from opensg_solid.sg_mesh import load_sg_input
    mat_blocks = load_sg_input(yaml_path)["materials"]
    lo, hi = nodes.min(axis=0), nodes.max(axis=0)
    a = a or float(hi[0] - lo[0])
    b = b or float(hi[1] - lo[1])
    tol = 1e-6 * max(hi - lo)
    print("plate_inp : %d nodes, %d %s; plate %.4g x %.4g x %.4g"
          % (len(nodes), E, etype, a, b, hi[2] - lo[2]))

    def _qval(x, y):
        if load == "uniform":
            return q
        if load == "linear-x":
            return q * (x - lo[0]) / a
        if load == "linear-y":
            return q * (y - lo[1]) / b
        raise SystemExit("load must be uniform | linear-x | linear-y,"
                         " got %r" % load)

    # the loaded TOP faces: every face whose nodes ALL sit on z = zmax
    on_top = np.abs(nodes[:, 2] - hi[2]) < tol
    top = []
    for lab, loc in _FACES[k]:
        m = on_top[cells[:, list(loc)]].all(axis=1)
        for e in np.nonzero(m)[0]:
            c = nodes[cells[e, list(loc)]].mean(axis=0)
            top.append((e + 1, lab, _qval(c[0], c[1])))
    if not top:
        raise SystemExit("no element face sits on z = zmax -- wrong"
                         " mesh orientation?")
    print("plate_inp : %d top faces loaded (%s, q = %g MPa)"
          % (len(top), load, q))

    # clamped node sets
    side = np.abs(nodes[:, 0] - lo[0]) < tol
    side |= np.abs(nodes[:, 0] - hi[0]) < tol
    if bc == "clamped":
        side |= np.abs(nodes[:, 1] - lo[1]) < tol
        side |= np.abs(nodes[:, 1] - hi[1]) < tol
    elif bc != "clamped-x":
        raise SystemExit("bc must be clamped | clamped-x, got %r" % bc)
    fix = np.nonzero(side)[0] + 1
    print("plate_inp : %s -> %d ENCASTRE nodes" % (bc, len(fix)))

    with open(out, "w", buffering=1 << 22) as f:
        f.write("*Heading\n** plate deck from %s (opensg_solid.helper)"
                "\n** bc=%s load=%s q=%g  %s\n"
                % (os.path.basename(msh_path), bc, load, q, job_note))
        f.write("*Node\n")
        ids = np.arange(1, len(nodes) + 1)
        np.savetxt(f, np.column_stack([ids, nodes]),
                   fmt=["%d", "%.8f", "%.8f", "%.8f"], delimiter=", ")
        f.write("*Element, type=%s\n" % etype)
        np.savetxt(f, np.column_stack(
            [np.arange(1, E + 1), cells + 1]), fmt="%d", delimiter=", ")
        for mid in sorted(set(mats.tolist())):
            el = np.nonzero(mats == mid)[0] + 1
            f.write("*Elset, elset=MAT%d\n" % mid)
            for i in range(0, len(el), 16):
                f.write(", ".join(str(v) for v in el[i:i + 16]) + "\n")
            blk = mat_blocks[mid]
            ang = float(blk.get("angle", 0.0) or 0.0)
            if int(blk["type"]) == 1 and not _is_iso(blk):
                # ENGINEERING CONSTANTS is anisotropic to Abaqus, and on
                # solid elements it REFUSES the material without a local
                # orientation -- so EVERY orthotropic section gets one,
                # angle 0 included (OpenSG `angle: a` == Abaqus `3, -a`)
                f.write("*Orientation, name=ORI%d\n1., 0., 0., 0., 1.,"
                        " 0.\n3, %g\n" % (mid, -ang))
                f.write("*Solid Section, elset=MAT%d, material=M%d,"
                        " orientation=ORI%d\n,\n" % (mid, mid, mid))
            else:
                f.write("*Solid Section, elset=MAT%d, material=M%d\n,\n"
                        % (mid, mid))
        for mid in sorted(set(mats.tolist())):
            blk = mat_blocks[mid]
            f.write("*Material, name=M%d\n" % mid)
            if int(blk["type"]) == 0:
                f.write("*Elastic\n%g, %g\n" % (blk["E"], blk["nu"]))
            elif int(blk["type"]) == 1 and _is_iso(blk):
                # a type-1 block whose nine constants ARE one isotropic
                # material (the loaders broadcast E/nu to three) goes out
                # as plain *Elastic: same stiffness, and Abaqus does not
                # demand the orientation it requires of anisotropy
                e = [float(v) for v in blk["engineering"]]
                f.write("*Elastic\n%g, %g\n" % (e[0], e[6]))
            elif int(blk["type"]) == 1:
                e = [float(v) for v in blk["engineering"]]
                # yaml order E1 E2 E3 G12 G13 G23 v12 v13 v23 ->
                # Abaqus E1 E2 E3 v12 v13 v23 G12 G13 G23
                f.write("*Elastic, type=ENGINEERING CONSTANTS\n"
                        "%g, %g, %g, %g, %g, %g, %g, %g,\n%g\n"
                        % (e[0], e[1], e[2], e[6], e[7], e[8],
                           e[3], e[4], e[5]))
            else:
                raise SystemExit("material type %s not supported by the"
                                 " deck writer" % blk["type"])
        f.write("*Nset, nset=FIX\n")
        for i in range(0, len(fix), 16):
            f.write(", ".join(str(v) for v in fix[i:i + 16]) + "\n")
        f.write("*Boundary\nFIX, ENCASTRE\n")
        f.write("*Step, name=STATIC, nlgeom=NO\n*Static\n")
        f.write("*Dload\n")
        for e, lab, v in top:
            f.write("%d, %s, %.8g\n" % (e, lab, v))
        f.write("*Output, field\n*Node Output\nU\n")
        f.write("*Element Output, position=CENTROIDAL\nS\n")
        f.write("*End Step\n")
    print("plate_inp : wrote %s  (%.1f s)"
          % (out, time.perf_counter() - t0))
    return {"inp": out, "n_nodes": len(nodes), "n_elems": E,
            "n_top_faces": len(top), "bc": bc, "load": load}
