"""One-time converter: UDcomp_2D.msh (gmsh 2.2) -> UDcomp_2D.yaml (OpenSG dialect).

The mesh -> yaml work is opensg_solid.io.msh_to_yaml.convert (nodes,
elements, the constant e1/e2/e3 frame and the physical-tag element sets);
this driver only owns the PATHS and the MATERIALS -- the modelling choices
the mesh cannot supply.  Everything the analysis needs then lives in the
yaml, exactly as make_schwarz_yaml.py does on the shell side.

The emitted yaml carries the nodes as '- [x y z]' rows, the triangles as
'- [n1 n2 n3]' (1-based), constant elementOrientations (e1 = axial z,
e2 = +x, e3 = +y), the two materials with engineering constants (MPa), and
sets.element for the matrix and fiber phases.

Run (from this folder):  python make_udcomp_yaml.py
"""
from opensg_solid.io.msh_to_yaml import convert

MSH = "UDcomp_2D.msh"
YAML = "UDcomp_2D.yaml"
MATERIALS = [
    {"name": "matrix", "density": 1200.0,
     "E": [4760.0, 4760.0, 4760.0],
     "G": [4760.0/2/1.37]*3,
     "nu": [0.37, 0.37, 0.37]},
    {"name": "fiber", "density": 1800.0,
     "E": [276000.0, 19500.0, 19500.0],
     "G": [70000.0, 70000.0, 5735.0],
     "nu": [0.28, 0.28, 0.70]},
]
PHASES = [("matrix", 0), ("fiber", 1)]      # (set name, gmsh physical tag)

d = convert(MSH, out_path=YAML, materials=MATERIALS, phases=PHASES)
print("wrote %s  (%d nodes, %d tris, %s)"
      % (d["path"], d["n_nodes"], d["n_elements"],
         " / ".join("%s %d" % kv for kv in d["sets"].items())))
