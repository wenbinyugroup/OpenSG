"""Convert a SwiftComp 3-D SG .sc (linear tets) to the OpenSG 3-D solid yaml,
and report the element type check (tet volumes: nonzero -> SOLID mesh, not a
shell surface).  Aluminum E = 70 GPa, nu = 0.3 (the .sc embeds 69000/0.3 --
overridden per the study definition).

Run (from this folder):  python sc_to_3dyaml.py"""
import numpy as np

SC = ["../../../tests/08072026_square_shell_mesh/TPMS_SC_files/Sample_1.sc",
      "../../../tests/08072026_square_shell_mesh/TPMS_SC_files/Sample_2.sc"]
E, nu = 70.0e9, 0.30
G = E/(2*(1+nu))

for sc in SC:
    toks = [ln.split() for ln in open(sc) if ln.strip()]
    k = 2                                   # skip "0" and "0 0 0 0"
    dim, nn, ne = int(toks[k][0]), int(toks[k][1]), int(toks[k][2])
    nd = np.array([[float(v) for v in toks[k+1+i][1:4]] for i in range(nn)])
    el = np.array([[int(v) for v in toks[k+1+nn+i][2:6]] for i in range(ne)])
    p = nd[el - 1]
    vol = np.abs(np.einsum('ij,ij->i',
                           np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0]),
                           p[:, 3]-p[:, 0]))/6.0
    bb = nd.max(0) - nd.min(0)
    name = sc.split("/")[-1].replace(".sc", "")
    print("%s: dim=%d  %d nodes  %d tets  bbox %s  vol[min/med] %.2e/%.2e"
          " -> %s" % (name, dim, nn, ne, np.round(bb, 4), vol.min(),
                      np.median(vol), "SOLID" if vol.min() > 0 else "DEGEN"))
    o = ["nodes:"]
    o += ["- [%.12f %.12f %.12f]" % (x, y, z) for x, y, z in nd]
    o.append("elements:")
    o += ["- [%d %d %d %d]" % tuple(e) for e in el]
    o += ["materials:", "- name: alu", "  density: 2700.0", "  elastic:",
          "    E: [%.6e, %.6e, %.6e]" % (E, E, E),
          "    G: [%.6e, %.6e, %.6e]" % (G, G, G),
          "    nu: [%.6f, %.6f, %.6f]" % (nu, nu, nu),
          "sets:", "  element:", "  - name: alu_all", "    labels: all"]
    open("%s_3Dsolid.yaml" % name, "w").write("\n".join(o) + "\n")
    print("wrote %s_3Dsolid.yaml" % name)
