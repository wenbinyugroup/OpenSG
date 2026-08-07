"""Inspect the r020 solid 2-D yaml schema (throwaway helper)."""
import yaml, os

P = os.path.expanduser(
    "~/OpenSG_io/examples/mesh_out/iea_prevabs_refined/r020_solid_boundary.yaml")
d = yaml.safe_load(open(P))
print("keys:", {k: (len(v) if isinstance(v, list) else type(v).__name__)
                for k, v in d.items()})
for k in d:
    v = d[k]
    if isinstance(v, list) and v:
        print("\n---", k, "first entries:")
        for row in v[:2]:
            print("   ", repr(row)[:180])
if "sections" in d:
    print("\nsections[0]:", repr(d["sections"][0])[:400])
if "sets" in d:
    s = d["sets"]
    print("\nsets keys:", list(s.keys()) if isinstance(s, dict) else type(s))
    if isinstance(s, dict) and "element" in s:
        print("n element sets:", len(s["element"]))
        print("first set:", repr(s["element"][0])[:200])
if "materials" in d:
    print("\nn materials:", len(d["materials"]))
    print("material names:", [m.get("name") for m in d["materials"]][:12])
print("\nreference:", d.get("reference"))
