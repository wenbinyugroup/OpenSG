"""materials_db.py -- the OpenSG material LIBRARY reader: materials.yaml
-> material blocks, plus the NAME[:ANGLE] spec grammar the
`opensg msh_to_yaml` mat flags speak (one flag per gmsh physical tag K:
--mat<K> NAME[:ANGLE]).

The library schema IS the SG solid yamls' own material-block dialect, so
filling a yaml is a passthrough -- a top-level `materials:` list of

    - name: Al
      aliases: [alu, aluminum]        # optional extra names, same card
      density: 2700.0
      elastic:
        E: [7.0e+10, 7.0e+10, 7.0e+10]
        G: [26923076923.076923, ...]  # iso: E/(2(1+nu)) precomputed
        nu: [0.3, 0.3, 0.3]

-- plus optional `layups:` descriptors (informational for now: no solver
route consumes them yet):

    layups:
    - name: pm45
      plies:
      - {material: HC_ply, angle: 45}
      - {material: HC_ply, angle: -45}

A ply ANGLE is never stored in the library -- it is per-use, appended to
the CLI spec as NAME:ANGLE (deg, about the thickness axis y3, the
`angle:` slot of the yaml dialect).

WHERE THE LIBRARY COMES FROM (load_materials/load_library resolution
order):  1. the explicit path argument,  2. ./materials.yaml in the CWD,
3. materials.yaml next to the input mesh (the `near` argument),  4. the
PACKAGED default next to this module (the seeded library).  Name lookup
is case-insensitive, aliases included; a miss names every available
material.  The old XML schema is DROPPED: an explicit .xml path gets a
clear error, not a parse attempt.

Use:  from opensg_solid.io.materials_db import load_library, resolve_spec
      lib = load_library()                        # the packaged default
      blk, ang = resolve_spec("HC_ply:-45", lib)  # (block dict, -45.0)
"""
import os

import yaml

# the packaged seeded library, the last resort of find_library
PACKAGED_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "materials.yaml")


def find_library(path=None, near=None):
    """WHICH materials.yaml a conversion will read (no parse).

    In:  path str | None -- an explicit library path (must exist; a .xml
         path is refused: the library is YAML now);
         near str | None -- a file (the input .msh) whose directory is
         searched for a sibling materials.yaml
    Out: str path -- the first hit of: explicit > ./materials.yaml (CWD) >
         <near dir>/materials.yaml > PACKAGED_PATH; FileNotFoundError
         naming every location tried."""
    if path is not None:
        if str(path).lower().endswith(".xml"):
            raise ValueError(
                "%s: the material library is YAML now (materials.yaml, the"
                " SG yamls' own material-block dialect) -- the XML schema"
                " was dropped; convert the library file" % path)
        if not os.path.exists(path):
            raise FileNotFoundError("no material library at %s" % path)
        return path
    tried = [os.path.join(os.getcwd(), "materials.yaml")]
    if os.path.exists(tried[0]):
        return tried[0]
    if near:
        sib = os.path.join(os.path.dirname(os.path.abspath(near)),
                           "materials.yaml")
        if sib not in tried:
            tried.append(sib)
            if os.path.exists(sib):
                return sib
    tried.append(PACKAGED_PATH)
    if os.path.exists(PACKAGED_PATH):
        return PACKAGED_PATH
    raise FileNotFoundError(
        "no materials.yaml found -- tried %s.  (A pip install without the"
        " package-data yaml lacks the packaged default: pass an explicit"
        " library path, or keep a materials.yaml in the CWD or next to the"
        " mesh.)" % ", ".join(tried))


def load_library(path=None, near=None):
    """The material library: materials AND layups, keyed case-insensitively.

    In:  path str | None, near str | None -- see find_library
    Out: dict {path str, materials {lowercase name/alias: block dict, the
         yaml's own material-block dialect -- aliases map to the SAME
         dict}, names [str] canonical display names, layups {lowercase
         name: [(material name, angle deg), ...]}}."""
    where = find_library(path, near=near)
    with open(where) as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict) or not isinstance(raw.get("materials"),
                                                   list):
        raise ValueError("%s carries no top-level `materials:` list"
                         % where)
    materials, names = {}, []
    for blk in raw["materials"]:
        nm = blk.get("name")
        ela = blk.get("elastic")
        if not nm or not isinstance(ela, dict) or not all(
                k in ela for k in ("E", "G", "nu")):
            raise ValueError(
                "%s: material entry %r needs name + elastic {E, G, nu}"
                % (where, nm or blk))
        names.append(str(nm))
        for k in [str(nm)] + [str(a) for a in blk.get("aliases") or []]:
            k = k.strip().lower()
            if k and k in materials:
                raise ValueError("duplicate material name/alias %r in %s"
                                 % (k, where))
            if k:
                materials[k] = blk
    layups = {}
    for lay in raw.get("layups") or []:
        nm = str(lay.get("name") or "").strip()
        if nm:
            layups[nm.lower()] = [(p.get("material"),
                                   float(p.get("angle", 0.0)))
                                  for p in lay.get("plies") or []]
    return {"path": where, "materials": materials, "names": names,
            "layups": layups}


def load_materials(path=None, near=None):
    """The {name: block} face of load_library (the task-level entry point).

    In:  path str | None, near str | None -- see find_library
    Out: dict {lowercase name/alias: material block dict}."""
    return load_library(path, near=near)["materials"]


def parse_spec(spec):
    """Split a CLI material spec into (name, angle).

    'Al' -> ('Al', None);  'HC_ply:-45' -> ('HC_ply', -45.0).  Only a
    trailing :<number> is an angle -- a colon followed by anything else
    stays part of the name (and will then miss the library loudly).

    In:  spec str
    Out: (name str, angle float | None)."""
    s = str(spec).strip()
    if ":" in s:
        head, tail = s.rsplit(":", 1)
        try:
            return head.strip(), float(tail)
        except ValueError:
            pass
    return s, None


def resolve_spec(spec, lib):
    """A CLI spec NAME[:ANGLE] -> (its library block, the angle).

    In:  spec str; lib dict -- load_library's output
    Out: (block dict, angle float | None); ValueError naming every
         available material on a miss (case-insensitive lookup, aliases
         included)."""
    name, angle = parse_spec(spec)
    blk = lib["materials"].get(name.lower())
    if blk is None:
        raise ValueError(
            "unknown material %r -- the library at %s has: %s"
            % (name, lib["path"], ", ".join(lib["names"])))
    return blk, angle


def to_solid_material(block, name=None, angle=None):
    """A library block -> the material dict the solid yaml writer consumes
    (io.msh_to_yaml._material_block: {name, density, E, G, nu, angle}).
    Near-passthrough: the library already speaks the SG dialect, only the
    name (the CLI overrides it with the SET name, because the solid
    reader binds sets to materials by name) and the per-use angle differ.

    In:  block dict -- a load_library material block;
         name str | None -- the yaml material name (None: the library
         name);  angle float | None -- ply angle deg about y3
    Out: dict {name, density, E, G, nu, angle}."""
    ela = block["elastic"]
    return {"name": name or block["name"],
            "density": float(block.get("density", 0.0)),
            "E": ela["E"], "G": ela["G"], "nu": ela["nu"], "angle": angle}
