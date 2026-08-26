"""materials_xml.py -- the OpenSG material LIBRARY reader: a PreVABS-
flavored materials.xml -> canonical material blocks, plus the NAME[:ANGLE]
spec grammar the `opensg msh_to_yaml` mat flags speak (one flag per gmsh
physical tag K: --mat<K> NAME[:ANGLE]).

The library file carries <material> cards in two types --

    <material name="Al" type="isotropic" aliases="alu,aluminum">
        <density>2700</density>
        <elastic><e>7.0e10</e><nu>0.3</nu></elastic></material>
    <material name="HC_ply" type="orthotropic">
        <density>0.0</density>
        <elastic><e1>..</e1><e2>..</e2><e3>..</e3>
                 <g12>..</g12><g13>..</g13><g23>..</g23>
                 <nu12>..</nu12><nu13>..</nu13><nu23>..</nu23>
        </elastic></material>

-- and optional <layup name="pm45"><ply material="HC_ply" angle="45"/>...
</layup> descriptors (informational for now: no solver route consumes them
yet).  Cards map to the SAME canonical block dialect the SG loaders speak
(io.sg_input): type 0 iso {E, nu, density}, type 1 engineering
[E1 E2 E3 G12 G13 G23 nu12 nu13 nu23]; a ply ANGLE is never stored in the
library -- it is per-use, appended to the CLI spec as NAME:ANGLE (deg,
about the thickness axis y3, the `angle:` slot of the yaml dialect).

WHERE THE LIBRARY COMES FROM (load_materials/load_library resolution
order):  1. the explicit xml_path argument,  2. ./materials.xml in the
CWD,  3. materials.xml next to the input mesh (the `near` argument),
4. the PACKAGED default next to this module (the seeded library).  Name
lookup is case-insensitive, aliases included; a miss names every
available material.

Use:  from opensg_solid.io.materials_xml import load_library, resolve_spec
      lib = load_library()                        # the packaged default
      blk, ang = resolve_spec("HC_ply:-45", lib)  # (type-1 block, -45.0)
"""
import os
import xml.etree.ElementTree as ET

# the engineering-constant slots of a type-1 block, in yaml/VABS row order
_ORTHO_KEYS = ("e1", "e2", "e3", "g12", "g13", "g23", "nu12", "nu13", "nu23")


def _block_from_material_element(el):
    """One <material> xml element -> its canonical block dict.

    In:  el xml.etree Element -- <material name= type=isotropic|orthotropic>
         with <density> and <elastic> children (see the module docstring)
    Out: dict -- iso {name, type 0, E, nu, density} or orthotropic
         {name, type 1, engineering (9,) tuple, density}; ValueError on a
         missing/unknown field."""
    name = el.get("name")
    if not name:
        raise ValueError("a <material> without a name= attribute")
    kind = (el.get("type") or "").strip().lower()
    d = el.find("density")
    density = float(d.text) if d is not None else 0.0
    ela = el.find("elastic")
    if ela is None:
        raise ValueError("material %r has no <elastic> block" % name)
    if kind == "isotropic":
        e, nu = ela.find("e"), ela.find("nu")
        if e is None or nu is None:
            raise ValueError("isotropic material %r needs <e> and <nu>"
                             % name)
        return {"name": name, "type": 0, "E": float(e.text),
                "nu": float(nu.text), "density": density}
    if kind == "orthotropic":
        vals = []
        for k in _ORTHO_KEYS:
            c = ela.find(k)
            if c is None:
                raise ValueError("orthotropic material %r is missing <%s>"
                                 % (name, k))
            vals.append(float(c.text))
        return {"name": name, "type": 1, "engineering": tuple(vals),
                "density": density}
    raise ValueError("material %r has type=%r -- isotropic or orthotropic"
                     % (name, el.get("type")))


def find_library(xml_path=None, near=None):
    """WHICH materials.xml a conversion will read (no parse).

    In:  xml_path str | None -- an explicit library path (must exist);
         near str | None -- a file (the input .msh) whose directory is
         searched for a sibling materials.xml
    Out: str path -- the first hit of: explicit > ./materials.xml (CWD) >
         <near dir>/materials.xml > the packaged default next to this
         module; FileNotFoundError naming every location tried."""
    if xml_path is not None:
        if not os.path.exists(xml_path):
            raise FileNotFoundError("no material library at %s" % xml_path)
        return xml_path
    tried = []
    cwd = os.path.join(os.getcwd(), "materials.xml")
    tried.append(cwd)
    if os.path.exists(cwd):
        return cwd
    if near:
        sib = os.path.join(os.path.dirname(os.path.abspath(near)),
                           "materials.xml")
        if sib not in tried:
            tried.append(sib)
            if os.path.exists(sib):
                return sib
    packaged = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "materials.xml")
    tried.append(packaged)
    if os.path.exists(packaged):
        return packaged
    raise FileNotFoundError(
        "no materials.xml found -- tried %s.  (A pip install without the"
        " package-data xml lacks the packaged default: pass an explicit"
        " library path, or keep a materials.xml in the CWD or next to the"
        " mesh.)" % ", ".join(tried))


def load_library(xml_path=None, near=None):
    """The material library: materials AND layups, keyed case-insensitively.

    In:  xml_path str | None, near str | None -- see find_library
    Out: dict {path str, materials {lowercase name/alias: block dict --
         aliases map to the SAME dict}, names [str] canonical display
         names, layups {lowercase name: [(material name, angle deg),
         ...]}}."""
    path = find_library(xml_path, near=near)
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as e:
        raise ValueError("%s is not well-formed xml: %s" % (path, e))
    materials, names, layups = {}, [], {}
    for el in root.iter("material"):
        blk = _block_from_material_element(el)
        names.append(blk["name"])
        keys = [blk["name"]] + [a for a in
                                (el.get("aliases") or "").split(",") if a]
        for k in keys:
            k = k.strip().lower()
            if k and k in materials:
                raise ValueError("duplicate material name/alias %r in %s"
                                 % (k, path))
            if k:
                materials[k] = blk
    for el in root.iter("layup"):
        nm = (el.get("name") or "").strip()
        plies = [(p.get("material"), float(p.get("angle", 0.0)))
                 for p in el.iter("ply")]
        if nm:
            layups[nm.lower()] = plies
    return {"path": path, "materials": materials, "names": names,
            "layups": layups}


def load_materials(xml_path=None, near=None):
    """The {name: block} face of load_library (the task-level entry point).

    In:  xml_path str | None, near str | None -- see find_library
    Out: dict {lowercase name/alias: canonical block dict}."""
    return load_library(xml_path, near=near)["materials"]


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
    """A canonical library block -> the material dict the solid yaml writer
    consumes (io.msh_to_yaml._material_block: {name, density, E, G, nu,
    angle}).

    In:  block dict -- type 0 or type 1 (load_library values);
         name str | None -- the yaml material name (None: the library
         name; the CLI overrides it with the SET name, because the solid
         reader binds sets to materials by name);
         angle float | None -- ply angle deg about y3 (None/0 omitted)
    Out: dict {name, density, E (3,), G (3,), nu (3,), angle}."""
    if int(block.get("type", 1)) == 0:
        E, nu = float(block["E"]), float(block["nu"])
        G = E / (2.0 * (1.0 + nu))
        e123, g123, nu123 = (E, E, E), (G, G, G), (nu, nu, nu)
    else:
        eng = [float(v) for v in block["engineering"]]
        e123, g123, nu123 = tuple(eng[0:3]), tuple(eng[3:6]), tuple(eng[6:9])
    return {"name": name or block["name"], "density": float(
        block.get("density", 0.0)), "E": e123, "G": g123, "nu": nu123,
        "angle": angle}
